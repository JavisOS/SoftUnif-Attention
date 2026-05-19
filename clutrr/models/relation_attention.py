import math

import torch
import torch.nn as nn


class RelationConditionedEntityAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_relations: int,
        dropout: float = 0.1,
        top_k: int | None = 8,
        use_relation_conditioning: bool = True,
        relation_score_mode: str = "mlp",
        relation_rank: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_relations = num_relations
        self.top_k = top_k
        self.use_relation_conditioning = use_relation_conditioning
        if relation_score_mode not in {"mlp", "bilinear"}:
            raise ValueError(f"Unsupported relation_score_mode: {relation_score_mode}")
        self.relation_score_mode = relation_score_mode
        self.relation_rank = relation_rank

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
        self.rel_left = nn.Linear(hidden_size, num_relations * relation_rank, bias=False)
        self.rel_right = nn.Linear(hidden_size, num_relations * relation_rank, bias=False)
        self.rel_score_bias = nn.Parameter(torch.zeros(num_relations))

        self.rel_gamma = nn.Embedding(num_relations, hidden_size)
        self.rel_beta = nn.Embedding(num_relations, hidden_size)

        self.rel_bias = nn.Parameter(torch.zeros(num_relations))
        self.rel_bias_hop = nn.Parameter(torch.zeros(num_relations))

        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_size)

    def pair_rel_logits(self, e_i: torch.Tensor, e_j: torch.Tensor) -> torch.Tensor:
        if self.relation_score_mode == "bilinear":
            left = self.rel_left(e_i).view(*e_i.shape[:-1], self.num_relations, self.relation_rank)
            right = self.rel_right(e_j).view(*e_j.shape[:-1], self.num_relations, self.relation_rank)
            logits = (left * right).sum(dim=-1) / math.sqrt(self.relation_rank)
            return logits + self.rel_score_bias

        feats = torch.cat([e_i, e_j, e_i * e_j], dim=-1)
        return self.rel_mlp(feats)

    def _select_sparse_edges(
        self,
        base_scores: torch.Tensor,
        valid_mask: torch.Tensor,
        forced_edge_index: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz, max_entities, _ = base_scores.shape
        dense_mode = self.top_k is None or self.top_k <= 0
        k = max_entities if dense_mode else min(self.top_k, max_entities)

        dst_valid = valid_mask.unsqueeze(1).expand(-1, max_entities, -1)
        candidate_scores = base_scores.masked_fill(~dst_valid, -1e4)
        if not dense_mode and max_entities > 1:
            eye = torch.eye(max_entities, device=base_scores.device, dtype=torch.bool).unsqueeze(0)
            candidate_scores = candidate_scores.masked_fill(eye, -1e4)

        _, edge_index = torch.topk(candidate_scores, k=k, dim=-1)
        if forced_edge_index is None or dense_mode:
            return edge_index

        forced = forced_edge_index.to(device=base_scores.device, dtype=torch.long)
        forced_clamped = forced.clamp(min=0, max=max_entities - 1)
        forced_dst_valid = torch.gather(valid_mask, 1, forced_clamped)
        forced_valid = (forced >= 0) & valid_mask & forced_dst_valid
        has_forced = (edge_index == forced_clamped.unsqueeze(-1)).any(dim=-1)
        replace = forced_valid & ~has_forced

        edge_index = edge_index.clone()
        edge_index[:, :, -1] = torch.where(replace, forced_clamped, edge_index[:, :, -1])
        return edge_index

    def forward(
        self,
        entity_embs: torch.Tensor,
        valid_mask: torch.Tensor,
        obj_indices: torch.Tensor | None = None,
        forced_edge_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        bsz, max_entities, hidden_size = entity_embs.shape
        valid_mask = valid_mask.bool()

        q_h = self.q_hop(entity_embs)
        if obj_indices is not None:
            obj_idx = obj_indices.clamp(min=0, max=max_entities - 1)
            b_idx = torch.arange(bsz, device=entity_embs.device)
            obj_emb = entity_embs[b_idx, obj_idx]
            obj_valid = torch.gather(valid_mask, 1, obj_idx.unsqueeze(1)).squeeze(1)
            q_h = q_h + self.q_obj(obj_emb).unsqueeze(1) * obj_valid[:, None, None].to(q_h.dtype)

        k_h = self.k_hop(entity_embs)
        base_scores_hop = torch.matmul(q_h, k_h.transpose(1, 2)) / math.sqrt(hidden_size)
        edge_index = self._select_sparse_edges(base_scores_hop, valid_mask, forced_edge_index)

        q = self.q_proj(entity_embs)
        k = self.k_proj(entity_embs)
        v = self.v_proj(entity_embs)
        v_h = self.v_hop(entity_embs)
        base_scores_global = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(hidden_size)

        src_embs = entity_embs.unsqueeze(2).expand(-1, -1, edge_index.size(-1), -1)
        gather_index = edge_index.unsqueeze(-1).expand(-1, -1, -1, hidden_size)
        expanded_entities = entity_embs.unsqueeze(1).expand(-1, max_entities, -1, -1)
        dst_embs = torch.gather(expanded_entities, 2, gather_index)

        rel_logits = self.pair_rel_logits(src_embs, dst_embs)
        rel_probs = torch.softmax(rel_logits, dim=-1)
        if self.use_relation_conditioning:
            gamma = torch.matmul(rel_probs, self.rel_gamma.weight)
            beta = torch.matmul(rel_probs, self.rel_beta.weight)
        else:
            gamma = torch.zeros_like(src_embs)
            beta = torch.zeros_like(src_embs)

        expanded_valid = valid_mask.unsqueeze(1).expand(-1, max_entities, -1)
        dst_edge_valid = torch.gather(expanded_valid, 2, edge_index)
        edge_valid = valid_mask.unsqueeze(-1) & dst_edge_valid

        sparse_global_scores = torch.gather(base_scores_global, 2, edge_index)
        sparse_hop_scores = torch.gather(base_scores_hop, 2, edge_index)
        if self.use_relation_conditioning:
            sparse_global_scores = sparse_global_scores + torch.matmul(rel_probs, self.rel_bias)
            sparse_hop_scores = sparse_hop_scores + torch.matmul(rel_probs, self.rel_bias_hop)
        sparse_global_scores = sparse_global_scores.masked_fill(~edge_valid, -1e4)
        sparse_hop_scores = sparse_hop_scores.masked_fill(~edge_valid, -1e4)

        attn = self.dropout(torch.softmax(sparse_global_scores, dim=-1))
        attn_hop = self.dropout(torch.softmax(sparse_hop_scores, dim=-1))

        expanded_v = v.unsqueeze(1).expand(-1, max_entities, -1, -1)
        expanded_v_h = v_h.unsqueeze(1).expand(-1, max_entities, -1, -1)
        dst_v = torch.gather(expanded_v, 2, gather_index)
        dst_v_h = torch.gather(expanded_v_h, 2, gather_index)

        v_cond = dst_v * (1.0 + gamma) + beta
        v_h_cond = dst_v_h * (1.0 + gamma) + beta
        msg_global = torch.sum(attn.unsqueeze(-1) * v_cond, dim=2)
        msg_hop = torch.sum(attn_hop.unsqueeze(-1) * v_h_cond, dim=2)

        updated = self.ln(entity_embs + self.dropout(msg_global + msg_hop))
        out = torch.where(valid_mask.unsqueeze(-1), updated, torch.zeros_like(updated))

        sparse_edges = {
            "edge_index": edge_index,
            "edge_valid": edge_valid,
            "hop_logits": sparse_hop_scores,
            "rel_logits": rel_logits,
        }
        return out, sparse_edges
