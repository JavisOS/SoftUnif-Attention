import math

import torch
import torch.nn as nn


class RelationConditionedEntityAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_relations: int,
        dropout: float = 0.1,
        use_relation_conditioning: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_relations = num_relations
        self.use_relation_conditioning = use_relation_conditioning

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        self.q_hop = nn.Linear(hidden_size, hidden_size)
        self.k_hop = nn.Linear(hidden_size, hidden_size)
        self.v_hop = nn.Linear(hidden_size, hidden_size)
        self.q_obj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.rel_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_relations),
        )

        self.rel_gamma = nn.Embedding(num_relations, hidden_size)
        self.rel_beta = nn.Embedding(num_relations, hidden_size)

        self.rel_bias = nn.Parameter(torch.zeros(num_relations))
        self.rel_bias_hop = nn.Parameter(torch.zeros(num_relations))

        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_size)

    def pair_rel_logits(self, e_i: torch.Tensor, e_j: torch.Tensor) -> torch.Tensor:
        feats = torch.cat([e_i, e_j, e_i * e_j], dim=-1)
        return self.rel_mlp(feats)

    def forward(
        self,
        entity_embs: torch.Tensor,
        valid_mask: torch.Tensor,
        obj_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, max_entities, hidden_size = entity_embs.shape
        out = torch.zeros_like(entity_embs)
        batch_scores_hop = torch.full(
            (bsz, max_entities, max_entities),
            -1e4,
            device=entity_embs.device,
            dtype=entity_embs.dtype,
        )
        batch_rel_logits = torch.full(
            (bsz, max_entities, max_entities, self.num_relations),
            -1e4,
            device=entity_embs.device,
            dtype=entity_embs.dtype,
        )

        for batch_idx in range(bsz):
            n_valid = int(valid_mask[batch_idx].sum().item())
            if n_valid <= 0:
                continue

            entities = entity_embs[batch_idx, :n_valid]
            e_i = entities.unsqueeze(1).expand(n_valid, n_valid, hidden_size)
            e_j = entities.unsqueeze(0).expand(n_valid, n_valid, hidden_size)
            feats = torch.cat([e_i, e_j, e_i * e_j], dim=-1)
            rel_logits = self.rel_mlp(feats)
            batch_rel_logits[batch_idx, :n_valid, :n_valid] = rel_logits
            rel_probs = torch.softmax(rel_logits, dim=-1)

            if self.use_relation_conditioning:
                gamma = torch.matmul(rel_probs, self.rel_gamma.weight)
                beta = torch.matmul(rel_probs, self.rel_beta.weight)
            else:
                gamma = torch.zeros(n_valid, n_valid, hidden_size, device=entity_embs.device, dtype=entity_embs.dtype)
                beta = torch.zeros_like(gamma)

            q = self.q_proj(entities)
            k = self.k_proj(entities)
            v = self.v_proj(entities)

            scores = torch.matmul(q, k.transpose(0, 1)) / math.sqrt(hidden_size)
            if self.use_relation_conditioning:
                scores = scores + torch.matmul(rel_probs, self.rel_bias)
            attn = self.dropout(torch.softmax(scores, dim=-1))

            v_expand = v.unsqueeze(0).expand(n_valid, n_valid, hidden_size)
            v_cond = v_expand * (1.0 + gamma) + beta
            msg_global = torch.sum(attn.unsqueeze(-1) * v_cond, dim=1)

            q_h = self.q_hop(entities)
            if obj_indices is not None:
                obj_idx = int(obj_indices[batch_idx].item())
                if 0 <= obj_idx < n_valid:
                    q_h = q_h + self.q_obj(entities[obj_idx]).unsqueeze(0)
            k_h = self.k_hop(entities)
            v_h = self.v_hop(entities)

            scores_hop = torch.matmul(q_h, k_h.transpose(0, 1)) / math.sqrt(hidden_size)
            if self.use_relation_conditioning:
                scores_hop = scores_hop + torch.matmul(rel_probs, self.rel_bias_hop)
            batch_scores_hop[batch_idx, :n_valid, :n_valid] = scores_hop

            attn_hop = self.dropout(torch.softmax(scores_hop, dim=-1))
            v_h_expand = v_h.unsqueeze(0).expand(n_valid, n_valid, hidden_size)
            v_h_cond = v_h_expand * (1.0 + gamma) + beta
            msg_hop = torch.sum(attn_hop.unsqueeze(-1) * v_h_cond, dim=1)

            out[batch_idx, :n_valid] = self.ln(entities + self.dropout(msg_global + msg_hop))

        return out, batch_scores_hop, batch_rel_logits
