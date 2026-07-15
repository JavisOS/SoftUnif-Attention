"""Shared TRUA core utilities.

The task-specific adapters decide what a reasoning unit is (an entity, fact,
rule, proposition, or sentence). The core provides common pooling,
query-conditioned unit features, and intermediate-supervision losses.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ReasoningAdapterSpec:
    name: str
    unit_type: str
    supervision_type: str


ENTITY_PATH_ADAPTER = ReasoningAdapterSpec(
    name="entity_path",
    unit_type="entity",
    supervision_type="ordered_path",
)

PROPOSITION_EVIDENCE_ADAPTER = ReasoningAdapterSpec(
    name="proposition_evidence",
    unit_type="fact_rule_or_proposition",
    supervision_type="evidence_set",
)

# Kept for exploratory scripts written before proposition supervision was
# correctly distinguished from an ordered proof sequence.
PROPOSITION_SEQUENCE_ADAPTER = PROPOSITION_EVIDENCE_ADAPTER


class TransitionRegularizedUnitAttentionCore(nn.Module):
    """Mixin-style core for reasoning models with optional intermediate labels.

    Subclasses provide encoder modules and task heads.  This class deliberately
    avoids dataset assumptions beyond span/unit tensors and masks.
    """

    def _pair_features(self, e_src, e_dst):
        features = [e_src, e_dst, e_src * e_dst]
        if self.pair_feature_mode == "product_diff":
            features.append(e_src - e_dst)
        return torch.cat(features, dim=-1)

    def _pool_sequence(self, sequence_output, attention_mask):
        if self.pooling == "cls":
            return sequence_output[:, 0, :]

        token_lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        b_idx = torch.arange(sequence_output.size(0), device=sequence_output.device)
        return sequence_output[b_idx, token_lengths, :]

    @staticmethod
    def _pool_spans(last_hidden_state, spans):
        original_shape = spans.shape[:-1]
        flat_spans = spans.reshape(spans.size(0), -1, 2)
        seq_len = last_hidden_state.size(1)
        valid_mask = flat_spans[:, :, 0] != -1

        starts = flat_spans[:, :, 0].clamp(min=0, max=seq_len - 1)
        ends = flat_spans[:, :, 1].clamp(min=1, max=seq_len)
        ends = torch.maximum(ends, starts + 1)

        positions = torch.arange(seq_len, device=last_hidden_state.device).view(1, 1, seq_len)
        span_mask = (positions >= starts.unsqueeze(-1)) & (positions < ends.unsqueeze(-1))
        span_mask = span_mask & valid_mask.unsqueeze(-1)

        weights = span_mask.to(last_hidden_state.dtype)
        token_counts = weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
        pooled = torch.bmm(weights, last_hidden_state) / token_counts
        pooled = pooled * valid_mask.unsqueeze(-1).to(pooled.dtype)
        return pooled.reshape(*original_shape, last_hidden_state.size(-1))

    def _build_query_context(self, base_unit_embs, query_indices):
        max_units = base_unit_embs.size(1)
        sub_idx = query_indices[:, 0].clamp(min=0, max=max_units - 1)
        obj_idx = query_indices[:, 1].clamp(min=0, max=max_units - 1)
        b_idx = torch.arange(base_unit_embs.size(0), device=base_unit_embs.device)
        e_sub = base_unit_embs[b_idx, sub_idx]
        e_obj = base_unit_embs[b_idx, obj_idx]
        return self.query_ctx_proj(torch.cat([e_sub, e_obj, e_sub * e_obj], dim=-1))

    def get_entity_embeddings(self, last_hidden_state, entity_spans):
        return self._pool_spans(last_hidden_state, entity_spans)


def masked_evidence_distribution_loss(scores, mask, evidence):
    """Cross entropy against a uniform distribution over marked evidence units."""

    has_evidence = (evidence * mask.float()).sum(dim=1) > 0
    if not has_evidence.any():
        return scores.new_tensor(0.0)
    masked_scores = scores[has_evidence].masked_fill(~mask[has_evidence], -1e4)
    target = evidence[has_evidence] * mask[has_evidence].float()
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1.0)
    return -(target * torch.log_softmax(masked_scores, dim=-1)).sum(dim=-1).mean()


def masked_trace_distribution_loss(scores, mask, trace):
    """Backward-compatible alias for older proposition experiment scripts."""

    return masked_evidence_distribution_loss(scores, mask, trace)
