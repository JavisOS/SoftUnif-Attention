"""Controlled CLUTRR baselines using the same text and unit adapters as TRUA.

These models are protocol controls, not reproductions of the original MAC or
RCA systems. They hold the encoder, grounded entity units, prediction head,
renamed-view training, and model-selection protocol fixed while replacing the
unit reasoning core.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from clutrr.models.backbones import build_backbone_model, is_decoder_only_model
from clutrr.models.trua_core import TransitionRegularizedUnitAttentionCore


class ContentSelfAttentionCore(nn.Module):
    """Content-only self-attention over grounded units."""

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)

    def forward_with_scores(
        self,
        units: torch.Tensor,
        valid_mask: torch.Tensor,
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del query
        attended, weights = self.attention(
            units,
            units,
            units,
            key_padding_mask=~valid_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        updated = self.attn_norm(units + attended)
        updated = self.ffn_norm(updated + self.ffn(updated))
        updated = updated * valid_mask.unsqueeze(-1).to(updated.dtype)

        pair_mask = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
        logits = torch.log(weights.clamp_min(1e-8)).masked_fill(~pair_mask, -1e4)
        return updated, logits

    def forward(self, units: torch.Tensor, valid_mask: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        updated, _ = self.forward_with_scores(units, valid_mask, query)
        return updated


class MacStyleUnitCore(nn.Module):
    """A compact MAC-style recurrent controller over grounded units."""

    def __init__(self, hidden_size: int, steps: int = 4, dropout: float = 0.1):
        super().__init__()
        self.steps = steps
        self.control_cell = nn.GRUCell(hidden_size, hidden_size)
        self.memory_cell = nn.GRUCell(hidden_size, hidden_size)
        self.unit_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.control_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.read_score = nn.Linear(hidden_size, 1, bias=False)
        self.memory_to_units = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, units: torch.Tensor, valid_mask: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        control = query
        memory = torch.zeros_like(query)
        scale = math.sqrt(units.size(-1))
        unit_keys = self.unit_key(units)

        for _ in range(self.steps):
            control = self.control_cell(query, control)
            scores = self.read_score(
                torch.tanh(unit_keys + self.control_key(control).unsqueeze(1))
            ).squeeze(-1)
            scores = scores.masked_fill(~valid_mask, -1e4)
            weights = torch.softmax(scores / scale, dim=-1)
            read = torch.sum(weights.unsqueeze(-1) * units, dim=1)
            memory = self.memory_cell(read, memory)

        memory_expanded = memory.unsqueeze(1).expand_as(units)
        delta = self.memory_to_units(torch.cat([units, memory_expanded], dim=-1))
        updated = self.norm(units + delta)
        return updated * valid_mask.unsqueeze(-1).to(updated.dtype)


class RcaStyleUnitCore(nn.Module):
    """Relational cross-attention over all valid ordered unit pairs."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.relation = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.query_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.update = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, units: torch.Tensor, valid_mask: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        source = units.unsqueeze(2)
        destination = units.unsqueeze(1)
        pair_features = torch.cat(
            [
                source.expand(-1, -1, units.size(1), -1),
                destination.expand(-1, units.size(1), -1, -1),
                source * destination,
            ],
            dim=-1,
        )
        relations = self.relation(pair_features)
        query_key = self.query_key(query).unsqueeze(1).unsqueeze(2)
        scores = torch.sum(relations * query_key, dim=-1) / math.sqrt(units.size(-1))

        pair_mask = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
        scores = scores.masked_fill(~pair_mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * pair_mask.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        context = torch.sum(weights.unsqueeze(-1) * self.value(destination), dim=2)

        delta = self.update(torch.cat([units, context], dim=-1))
        updated = self.norm(units + delta)
        return updated * valid_mask.unsqueeze(-1).to(updated.dtype)


class ControlledClutrrBaseline(TransitionRegularizedUnitAttentionCore):
    """Shared wrapper for controlled encoder and unit-reasoning baselines."""

    SUPPORTED_CORES = {"encoder", "self_attention", "self_attention_matched", "mac", "rca"}

    def __init__(
        self,
        device: torch.device,
        tokenizer,
        *,
        core_type: str,
        model_type: str = "deberta",
        model_name_or_path: str | None = None,
        pooling: str | None = None,
        pair_feature_mode: str = "product",
        mac_steps: int = 4,
        lambda_transition: float = 0.0,
        lambda_edge: float = 0.0,
    ):
        super().__init__()
        if core_type not in self.SUPPORTED_CORES:
            raise ValueError(f"Unsupported controlled core: {core_type}")
        if pair_feature_mode not in {"product", "product_diff"}:
            raise ValueError(f"Unsupported pair features: {pair_feature_mode}")

        self.device = device
        self.tokenizer = tokenizer
        self.core_type = core_type
        self.model_type = model_type.lower()
        self.decoder_only = is_decoder_only_model(self.model_type)
        self.pooling = pooling or ("last_token" if self.decoder_only else "cls")
        self.pair_feature_mode = pair_feature_mode
        self.lambda_transition = lambda_transition
        self.lambda_edge = lambda_edge

        if core_type != "self_attention_matched" and (lambda_transition != 0.0 or lambda_edge != 0.0):
            raise ValueError("Matched path objectives are supported only by self_attention_matched")

        self.encoder = build_backbone_model(
            self.model_type,
            model_name_or_path=model_name_or_path,
        )
        self.hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, 21)

        if core_type != "encoder":
            self.query_ctx_proj = nn.Sequential(
                nn.Linear(self.hidden_size * 3, self.hidden_size),
                nn.Tanh(),
            )
            pair_feature_dim = self.hidden_size * (4 if pair_feature_mode == "product_diff" else 3)
            self.pair_classifier = nn.Sequential(
                nn.Linear(pair_feature_dim, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 21),
            )

        if core_type in {"self_attention", "self_attention_matched"}:
            self.unit_core = ContentSelfAttentionCore(self.hidden_size)
        elif core_type == "mac":
            self.unit_core = MacStyleUnitCore(self.hidden_size, steps=mac_steps)
        elif core_type == "rca":
            self.unit_core = RcaStyleUnitCore(self.hidden_size)
        else:
            self.unit_core = None

        if core_type == "self_attention_matched":
            self.edge_classifier = nn.Sequential(
                nn.Linear(pair_feature_dim, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 21),
            )

    def _matched_path_losses(self, attention_logits, units, path_node_ids, path_rel_ids):
        transition_sum = attention_logits.new_tensor(0.0)
        edge_sum = attention_logits.new_tensor(0.0)
        transition_count = 0
        edge_count = 0

        for batch_index in range(path_node_ids.size(0)):
            path = path_node_ids[batch_index]
            path = path[path >= 0]
            if path.numel() < 2:
                continue

            source = path[:-1]
            destination = path[1:]
            transition_sum = transition_sum + nn.functional.cross_entropy(
                attention_logits[batch_index, source],
                destination,
                reduction="sum",
            )
            transition_count += int(source.numel())

            if path_rel_ids is None:
                continue
            relations = path_rel_ids[batch_index]
            relations = relations[relations >= 0]
            edge_total = min(source.numel(), relations.numel())
            if edge_total == 0:
                continue
            source = source[:edge_total]
            destination = destination[:edge_total]
            pair = self._pair_features(
                units[batch_index, source],
                units[batch_index, destination],
            )
            edge_sum = edge_sum + nn.functional.cross_entropy(
                self.edge_classifier(pair),
                relations[:edge_total],
                reduction="sum",
            )
            edge_count += int(edge_total)

        return (
            transition_sum / max(transition_count, 1),
            edge_sum / max(edge_count, 1),
        )

    def compute_logits(
        self,
        input_ids,
        attention_mask,
        entity_spans,
        query_indices,
        path_node_ids=None,
    ):
        del path_node_ids
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence = output.last_hidden_state.to(self.classifier.weight.dtype)
        pooled = self._pool_sequence(sequence, attention_mask)
        logits_cls = self.classifier(pooled)

        if self.unit_core is None:
            return logits_cls, None, None

        units = self.get_entity_embeddings(sequence, entity_spans)
        valid_mask = entity_spans[:, :, 0] != -1
        query = self._build_query_context(units, query_indices)
        if self.core_type == "self_attention_matched":
            updated_units, attention_logits = self.unit_core.forward_with_scores(
                units,
                valid_mask,
                query,
            )
            unit_count = units.size(1)
            edge_index = torch.arange(unit_count, device=units.device)
            edge_index = edge_index.view(1, 1, unit_count).expand(units.size(0), unit_count, -1)
            edge_valid = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
            sparse_edges = {
                "edge_index": edge_index,
                "edge_valid": edge_valid,
                "hop_logits": attention_logits,
            }
        else:
            updated_units = self.unit_core(units, valid_mask, query)
            sparse_edges = None

        subject = query_indices[:, 0].clamp(min=0, max=updated_units.size(1) - 1)
        object_ = query_indices[:, 1].clamp(min=0, max=updated_units.size(1) - 1)
        batch_index = torch.arange(updated_units.size(0), device=updated_units.device)
        pair = self._pair_features(
            updated_units[batch_index, subject],
            updated_units[batch_index, object_],
        )
        logits = logits_cls + self.pair_classifier(pair)
        return logits, updated_units, sparse_edges

    def forward(self, batch_data):
        logits, updated_units, sparse_edges = self.compute_logits(
            input_ids=batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
            entity_spans=batch_data["entity_spans"],
            query_indices=batch_data["query_indices"],
        )
        loss = nn.functional.cross_entropy(logits, batch_data["labels"])

        if batch_data.get("aug_input_ids") is not None:
            augmented_logits, _, _ = self.compute_logits(
                input_ids=batch_data["aug_input_ids"],
                attention_mask=batch_data["aug_attention_mask"],
                entity_spans=batch_data["aug_entity_spans"],
                query_indices=batch_data["query_indices"],
            )
            augmented_loss = nn.functional.cross_entropy(augmented_logits, batch_data["labels"])
            loss = 0.5 * (loss + augmented_loss)

        transition_loss = loss.new_tensor(0.0)
        edge_loss = loss.new_tensor(0.0)
        if self.core_type == "self_attention_matched":
            transition_loss, edge_loss = self._matched_path_losses(
                sparse_edges["hop_logits"],
                updated_units,
                batch_data["path_node_ids"],
                batch_data.get("path_rel_ids"),
            )
            loss = loss + self.lambda_transition * transition_loss + self.lambda_edge * edge_loss

        return {
            "loss": loss,
            "logits": logits,
            "losses": {
                "main": float((loss - self.lambda_transition * transition_loss - self.lambda_edge * edge_loss).item()),
                "nexthop": float(transition_loss.item()),
                "edge": float(edge_loss.item()),
            },
        }
