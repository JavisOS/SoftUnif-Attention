"""Shared TRUA core utilities.

The task-specific adapters decide what a reasoning unit is (an entity, fact,
rule, proposition, or sentence).  The core only provides common pooling,
query-conditioned unit features, and trace-supervision losses.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ReasoningAdapterSpec:
    name: str
    unit_type: str
    trace_type: str


ENTITY_PATH_ADAPTER = ReasoningAdapterSpec(
    name="entity_path",
    unit_type="entity",
    trace_type="path_node_sequence",
)

PROPOSITION_SEQUENCE_ADAPTER = ReasoningAdapterSpec(
    name="proposition_sequence",
    unit_type="fact_rule_or_proposition",
    trace_type="proof_step_sequence",
)


class TransitionRegularizedUnitAttentionCore(nn.Module):
    """Mixin-style core for trace-supervised reasoning models.

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

    @staticmethod
    def _aggregate_mentions_mean(mention_embs, mention_mask):
        mention_mask_f = mention_mask.unsqueeze(-1).to(mention_embs.dtype)
        mention_counts = mention_mask_f.sum(dim=2).clamp(min=1.0)
        return (mention_embs * mention_mask_f).sum(dim=2) / mention_counts

    def _aggregate_mentions_query_aware(self, mention_embs, mention_mask, query_indices):
        base_unit_embs = self._aggregate_mentions_mean(mention_embs, mention_mask)
        query_ctx = self._build_query_context(base_unit_embs, query_indices)
        mention_keys = self.mention_key_proj(mention_embs)
        query_bias = self.mention_query_proj(query_ctx).unsqueeze(1).unsqueeze(2)
        mention_scores = self.mention_score(torch.tanh(mention_keys + query_bias)).squeeze(-1)
        mention_scores = mention_scores.masked_fill(~mention_mask, -1e4)

        attn = torch.softmax(mention_scores, dim=-1)
        attn = attn * mention_mask.to(attn.dtype)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return (attn.unsqueeze(-1) * mention_embs).sum(dim=2)

    def get_unit_embeddings_from_spans(
        self,
        last_hidden_state,
        unit_spans,
        query_indices,
        unit_mention_spans=None,
    ):
        if self.entity_pooling == "mean" or unit_mention_spans is None:
            return self._pool_spans(last_hidden_state, unit_spans)

        mention_mask = unit_mention_spans[:, :, :, 0] != -1
        mention_embs = self._pool_spans(last_hidden_state, unit_mention_spans)
        if self.entity_pooling == "multi_mention":
            return self._aggregate_mentions_mean(mention_embs, mention_mask)
        return self._aggregate_mentions_query_aware(mention_embs, mention_mask, query_indices)

    def get_entity_embeddings(self, last_hidden_state, entity_spans, query_indices, entity_mention_spans=None):
        return self.get_unit_embeddings_from_spans(
            last_hidden_state,
            entity_spans,
            query_indices,
            unit_mention_spans=entity_mention_spans,
        )


def masked_trace_distribution_loss(scores, mask, trace):
    """Cross entropy against one or more gold trace units per example."""

    has_trace = (trace * mask.float()).sum(dim=1) > 0
    if not has_trace.any():
        return scores.new_tensor(0.0)
    masked_scores = scores[has_trace].masked_fill(~mask[has_trace], -1e4)
    target = trace[has_trace] * mask[has_trace].float()
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1.0)
    return -(target * torch.log_softmax(masked_scores, dim=-1)).sum(dim=-1).mean()
