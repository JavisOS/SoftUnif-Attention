import torch
import torch.nn as nn

from clutrr.models.backbones import build_backbone_model, is_decoder_only_model
from clutrr.config.relation_schema import RELATION_ID_MAP_21_WITH_NOTHING as relation_id_map
from clutrr.models.relation_attention import RelationConditionedEntityAttention


class TruaReasonerModelPreCore(nn.Module):
    def __init__(
        self,
        device,
        tokenizer,
        model_type="roberta",
        *,
        model_name_or_path=None,
        use_qlora=False,
        load_in_4bit=False,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        pooling=None,
        prediction_head="cls_pair",
        consistency_mode="kl",
        use_relation_conditioning=True,
        edge_supervision_target="latent",
        pair_feature_mode="product",
        sparse_top_k=8,
        force_gold_edges=True,
        use_path_algebra=False,
        path_algebra_steps=0,
        residual_gate_init=-1.5,
        relation_score_mode="mlp",
        relation_rank=64,
    ):
        super().__init__()
        self.device = device
        self.force_gold_edges = force_gold_edges
        self.use_path_algebra = use_path_algebra
        self.path_algebra_steps = path_algebra_steps
        self.nothing_rel_id = relation_id_map["nothing"]

        self.model_type = model_type.lower()
        self.decoder_only = is_decoder_only_model(self.model_type)
        self.pooling = pooling or ("last_token" if self.decoder_only else "cls")
        if prediction_head not in {"cls_pair", "cls_only", "pair_only", "gated"}:
            raise ValueError(f"Unsupported prediction_head mode: {prediction_head}")
        if consistency_mode not in {"kl", "sym_kl", "js", "mse"}:
            raise ValueError(f"Unsupported consistency_mode: {consistency_mode}")
        if edge_supervision_target not in {"latent", "separate"}:
            raise ValueError(f"Unsupported edge_supervision_target: {edge_supervision_target}")
        if pair_feature_mode not in {"product", "product_diff"}:
            raise ValueError(f"Unsupported pair_feature_mode: {pair_feature_mode}")

        self.prediction_head = prediction_head
        self.consistency_mode = consistency_mode
        self.edge_supervision_target = edge_supervision_target
        self.pair_feature_mode = pair_feature_mode

        self.encoder = build_backbone_model(
            self.model_type,
            model_name_or_path=model_name_or_path,
            use_qlora=use_qlora,
            load_in_4bit=load_in_4bit,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        self.tokenizer = tokenizer
        self.hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Linear(self.hidden_size, 21)
        self.entity_attn = RelationConditionedEntityAttention(
            hidden_size=self.hidden_size,
            num_relations=21,
            dropout=0.1,
            top_k=sparse_top_k,
            use_relation_conditioning=use_relation_conditioning,
            relation_score_mode=relation_score_mode,
            relation_rank=relation_rank,
        )
        pair_feature_dim = self.hidden_size * (4 if self.pair_feature_mode == "product_diff" else 3)
        self.pair_classifier = nn.Sequential(
            nn.Linear(pair_feature_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 21),
        )
        self.fusion_logit = nn.Parameter(torch.tensor(0.0))
        self.rel_proj = nn.Sequential(
            nn.Linear(pair_feature_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 21),
        )
        self.rel_comp_logits = nn.Parameter(torch.zeros(21, 21, 21))
        comp_prior = torch.zeros(21, 21, 21)
        for rel_id in range(21):
            comp_prior[self.nothing_rel_id, rel_id, rel_id] = 4.0
            comp_prior[rel_id, self.nothing_rel_id, rel_id] = 4.0
        spouse_inverse_pairs = [
            (relation_id_map["husband"], relation_id_map["wife"]),
            (relation_id_map["wife"], relation_id_map["husband"]),
        ]
        for rel_a, rel_b in spouse_inverse_pairs:
            comp_prior[rel_a, rel_b, self.nothing_rel_id] = 2.0
        self.register_buffer("rel_comp_prior", comp_prior)
        self.register_buffer("inverse_pairs", torch.tensor(spouse_inverse_pairs, dtype=torch.long))
        self.residual_gate_logit = nn.Parameter(torch.tensor(float(residual_gate_init)))

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

    def _pool_spans(self, last_hidden_state, spans):
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

    def get_entity_embeddings(self, last_hidden_state, entity_spans):
        return self._pool_spans(last_hidden_state, entity_spans)

    def _build_forced_edge_index(self, path_node_ids, max_entities):
        if path_node_ids is None:
            return None

        forced = torch.full(
            (path_node_ids.size(0), max_entities),
            -1,
            dtype=torch.long,
            device=path_node_ids.device,
        )
        for batch_idx in range(path_node_ids.size(0)):
            path = path_node_ids[batch_idx]
            valid_path = path[path != -1]
            if valid_path.numel() < 2:
                continue
            src = valid_path[:-1]
            dst = valid_path[1:]
            in_bounds = (src >= 0) & (src < max_entities) & (dst >= 0) & (dst < max_entities)
            if in_bounds.any():
                forced[batch_idx, src[in_bounds]] = dst[in_bounds]
        return forced

    def compute_logits(
        self,
        input_ids,
        attention_mask,
        entity_spans,
        query_indices,
        path_node_ids=None,
    ):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = out.last_hidden_state
        head_dtype = self.classifier.weight.dtype
        sequence_for_heads = sequence_output.to(head_dtype)

        cls_output = self._pool_sequence(sequence_for_heads, attention_mask)
        logits_cls = self.classifier(cls_output)

        entity_embs = self.get_entity_embeddings(sequence_for_heads, entity_spans)
        valid_mask = entity_spans[:, :, 0] != -1
        forced_edges = None
        if self.training and self.force_gold_edges:
            forced_edges = self._build_forced_edge_index(path_node_ids, entity_spans.size(1))

        obj_indices = query_indices[:, 1]
        entity_embs_upd, sparse_edges = self.entity_attn(
            entity_embs,
            valid_mask,
            obj_indices=obj_indices,
            forced_edge_index=forced_edges,
        )

        sub_idx = query_indices[:, 0].clamp(min=0)
        obj_idx = query_indices[:, 1].clamp(min=0)
        b_idx = torch.arange(entity_embs_upd.size(0), device=entity_embs_upd.device)
        e_sub = entity_embs_upd[b_idx, sub_idx]
        e_obj = entity_embs_upd[b_idx, obj_idx]
        pair_feats = self._pair_features(e_sub, e_obj)
        logits_pair = self.pair_classifier(pair_feats)

        if self.prediction_head == "cls_only":
            text_logits = logits_cls
        elif self.prediction_head == "pair_only":
            text_logits = logits_pair
        elif self.prediction_head == "gated":
            cls_weight = torch.sigmoid(self.fusion_logit)
            text_logits = cls_weight * logits_cls + (1.0 - cls_weight) * logits_pair
        else:
            text_logits = logits_cls + logits_pair

        if self.use_path_algebra:
            path_logits = self.compute_path_algebra_logits(sparse_edges, query_indices)
            residual_alpha = torch.sigmoid(self.residual_gate_logit)
            logits = path_logits + residual_alpha * text_logits
            sparse_edges["path_logits"] = path_logits
            sparse_edges["text_logits"] = text_logits
        else:
            logits = text_logits
        return logits, entity_embs_upd, sparse_edges

    def compute_path_algebra_logits(self, sparse_edges, query_indices):
        edge_index = sparse_edges["edge_index"]
        edge_valid = sparse_edges["edge_valid"]
        hop_logits = sparse_edges["hop_logits"]
        rel_logits = sparse_edges["rel_logits"]

        bsz, max_entities, top_k = edge_index.shape
        num_relations = rel_logits.size(-1)
        neg_inf = torch.finfo(hop_logits.dtype).min / 4

        rel_log_probs = torch.log_softmax(rel_logits, dim=-1)
        edge_log_probs = torch.log_softmax(hop_logits.masked_fill(~edge_valid, -1e4), dim=-1)
        log_comp = torch.log_softmax(self.rel_comp_logits + self.rel_comp_prior.to(self.rel_comp_logits.dtype), dim=-1)

        state = hop_logits.new_full((bsz, max_entities, num_relations), neg_inf)
        sub_idx = query_indices[:, 0].clamp(min=0, max=max_entities - 1)
        b_idx = torch.arange(bsz, device=hop_logits.device)
        state[b_idx, sub_idx, self.nothing_rel_id] = 0.0

        max_steps = self.path_algebra_steps if self.path_algebra_steps and self.path_algebra_steps > 0 else max_entities - 1
        collected = []
        for _ in range(max_steps):
            messages_by_batch = [[[] for _ in range(max_entities)] for _ in range(bsz)]
            for edge_pos in range(top_k):
                dst = edge_index[:, :, edge_pos]
                valid = edge_valid[:, :, edge_pos]
                edge_rel = rel_log_probs[:, :, edge_pos, :] + edge_log_probs[:, :, edge_pos].unsqueeze(-1)
                composed = torch.logsumexp(
                    state.unsqueeze(3).unsqueeze(4)
                    + edge_rel.unsqueeze(2).unsqueeze(4)
                    + log_comp.unsqueeze(0).unsqueeze(0),
                    dim=(2, 3),
                )
                composed = composed.masked_fill(~valid.unsqueeze(-1), neg_inf)
                for batch_idx in range(bsz):
                    valid_src = valid[batch_idx]
                    if not bool(valid_src.any()):
                        continue
                    for src_idx in torch.nonzero(valid_src, as_tuple=False).flatten():
                        dst_idx = int(dst[batch_idx, src_idx].item())
                        messages_by_batch[batch_idx][dst_idx].append(composed[batch_idx, src_idx])

            batch_states = []
            empty_row = hop_logits.new_full((num_relations,), neg_inf)
            for batch_idx in range(bsz):
                entity_rows = []
                for dst_idx in range(max_entities):
                    messages = messages_by_batch[batch_idx][dst_idx]
                    if messages:
                        entity_rows.append(torch.logsumexp(torch.stack(messages, dim=0), dim=0))
                    else:
                        entity_rows.append(empty_row)
                batch_states.append(torch.stack(entity_rows, dim=0))
            state = torch.stack(batch_states, dim=0)
            collected.append(state)

        obj_idx = query_indices[:, 1].clamp(min=0, max=max_entities - 1)
        path_states = torch.stack(collected, dim=0)
        obj_scores = path_states[:, b_idx, obj_idx, :].transpose(0, 1)
        path_logits = torch.logsumexp(obj_scores, dim=1)
        return path_logits

    def _composition_distribution(self):
        logits = self.rel_comp_logits + self.rel_comp_prior.to(self.rel_comp_logits.dtype)
        return torch.softmax(logits, dim=-1)

    def compute_algebra_loss(self):
        comp = self._composition_distribution()
        eps = 1e-8
        rel_ids = torch.arange(comp.size(0), device=comp.device)

        left_identity = -torch.log(comp[self.nothing_rel_id, rel_ids, rel_ids].clamp(min=eps)).mean()
        right_identity = -torch.log(comp[rel_ids, self.nothing_rel_id, rel_ids].clamp(min=eps)).mean()
        identity_loss = 0.5 * (left_identity + right_identity)

        left_assoc = torch.einsum("abx,xcy->abcy", comp, comp)
        right_assoc = torch.einsum("bcx,axy->abcy", comp, comp)
        assoc_loss = torch.mean((left_assoc - right_assoc).square())

        inv_loss = comp.new_tensor(0.0)
        if self.inverse_pairs.numel() > 0:
            inv_a = self.inverse_pairs[:, 0]
            inv_b = self.inverse_pairs[:, 1]
            inv_loss = -torch.log(comp[inv_a, inv_b, self.nothing_rel_id].clamp(min=eps)).mean()

        return identity_loss + assoc_loss + 0.5 * inv_loss

    def _matched_path_rel_log_probs(self, sparse_edges, path_node_ids):
        edge_index = sparse_edges["edge_index"]
        rel_logits = sparse_edges["rel_logits"]
        rows = []
        for batch_idx in range(path_node_ids.size(0)):
            path = path_node_ids[batch_idx]
            valid_path = path[path != -1]
            if valid_path.numel() < 2:
                continue
            src = valid_path[:-1]
            dst = valid_path[1:]
            row_edges = edge_index[batch_idx, src, :]
            match = row_edges == dst.unsqueeze(-1)
            has_match = match.any(dim=-1)
            if not bool(has_match.any()):
                continue
            edge_rel_logits = rel_logits[batch_idx, src, :, :]
            combined = edge_rel_logits.masked_fill(~match.unsqueeze(-1), -1e4)
            combined = torch.logsumexp(combined, dim=1)
            rows.append(torch.log_softmax(combined[has_match], dim=-1))
        if not rows:
            return None
        return torch.cat(rows, dim=0)

    def compute_rename_equivariance_loss(
        self,
        logits_orig,
        logits_aug,
        entity_embs_orig,
        entity_embs_aug,
        sparse_edges_orig,
        sparse_edges_aug,
        entity_spans,
        path_node_ids,
    ):
        losses = []

        rel_log_orig = self._matched_path_rel_log_probs(sparse_edges_orig, path_node_ids)
        rel_log_aug = self._matched_path_rel_log_probs(sparse_edges_aug, path_node_ids)
        if rel_log_orig is not None and rel_log_aug is not None:
            n = min(rel_log_orig.size(0), rel_log_aug.size(0))
            if n > 0:
                p_orig = rel_log_orig[:n].exp().detach()
                p_aug = rel_log_aug[:n].exp().detach()
                rel_kl = 0.5 * nn.functional.kl_div(rel_log_aug[:n], p_orig, reduction="batchmean")
                rel_kl = rel_kl + 0.5 * nn.functional.kl_div(rel_log_orig[:n], p_aug, reduction="batchmean")
                losses.append(rel_kl)

        valid_entities = entity_spans[:, :, 0] != -1
        if bool(valid_entities.any()):
            ent_orig = nn.functional.normalize(entity_embs_orig[valid_entities], dim=-1)
            ent_aug = nn.functional.normalize(entity_embs_aug[valid_entities], dim=-1)
            losses.append(0.1 * nn.functional.mse_loss(ent_orig, ent_aug))

        if "path_logits" in sparse_edges_orig and "path_logits" in sparse_edges_aug:
            log_path_orig = nn.functional.log_softmax(sparse_edges_orig["path_logits"], dim=-1)
            log_path_aug = nn.functional.log_softmax(sparse_edges_aug["path_logits"], dim=-1)
            p_path_orig = log_path_orig.exp().detach()
            p_path_aug = log_path_aug.exp().detach()
            path_kl = 0.5 * nn.functional.kl_div(log_path_aug, p_path_orig, reduction="batchmean")
            path_kl = path_kl + 0.5 * nn.functional.kl_div(log_path_orig, p_path_aug, reduction="batchmean")
            losses.append(path_kl)

        if not losses:
            return logits_orig.new_tensor(0.0)
        return torch.stack(losses).sum()

    def _compute_consistency_loss(self, logits_orig, logits_aug):
        if self.consistency_mode == "mse":
            return nn.functional.mse_loss(logits_aug, logits_orig.detach())

        p_orig = nn.functional.softmax(logits_orig, dim=1)
        log_p_orig = nn.functional.log_softmax(logits_orig, dim=1)
        p_aug = nn.functional.softmax(logits_aug, dim=1)
        log_p_aug = nn.functional.log_softmax(logits_aug, dim=1)
        if self.consistency_mode == "sym_kl":
            cons_loss = 0.5 * nn.functional.kl_div(log_p_aug, p_orig.detach(), reduction="batchmean")
            return cons_loss + 0.5 * nn.functional.kl_div(log_p_orig, p_aug.detach(), reduction="batchmean")
        if self.consistency_mode == "js":
            m = 0.5 * (p_orig.detach() + p_aug.detach()).clamp(min=1e-8)
            cons_loss = 0.5 * nn.functional.kl_div(log_p_orig, m, reduction="batchmean")
            return cons_loss + 0.5 * nn.functional.kl_div(log_p_aug, m, reduction="batchmean")
        return nn.functional.kl_div(log_p_aug, p_orig.detach(), reduction="batchmean")

    def _compute_sparse_trace_losses(self, sparse_edges, entity_embs, path_node_ids, path_rel_ids):
        edge_index = sparse_edges["edge_index"]
        hop_logits = sparse_edges["hop_logits"]
        rel_logits = sparse_edges["rel_logits"]

        trace_loss_sum = hop_logits.new_tensor(0.0)
        edge_loss_sum = hop_logits.new_tensor(0.0)
        total_hops = 0
        total_edges = 0

        for batch_idx in range(path_node_ids.size(0)):
            path = path_node_ids[batch_idx]
            valid_path = path[path != -1]
            if valid_path.numel() < 2:
                continue

            src = valid_path[:-1]
            dst = valid_path[1:]
            row_logits = hop_logits[batch_idx, src, :]
            row_edges = edge_index[batch_idx, src, :]
            match = row_edges == dst.unsqueeze(-1)

            log_probs = torch.log_softmax(row_logits, dim=-1)
            masked_log_probs = log_probs.masked_fill(~match, -1e4)
            edge_log_prob = torch.logsumexp(masked_log_probs, dim=-1)
            has_match = match.any(dim=-1)
            edge_log_prob = torch.where(
                has_match,
                edge_log_prob,
                edge_log_prob.new_full(edge_log_prob.shape, -20.0),
            )
            trace_loss_sum = trace_loss_sum - edge_log_prob.sum()
            total_hops += edge_log_prob.numel()

            if path_rel_ids is None:
                continue

            rel_targets = path_rel_ids[batch_idx]
            valid_rel_targets = rel_targets[rel_targets != -1]
            num_edges = min(src.numel(), valid_rel_targets.numel())
            if num_edges <= 0:
                continue

            edge_match = match[:num_edges]
            edge_has_match = edge_match.any(dim=-1)
            if not edge_has_match.any():
                continue

            if self.edge_supervision_target == "separate":
                e_src = entity_embs[batch_idx, src[:num_edges]]
                e_dst = entity_embs[batch_idx, dst[:num_edges]]
                edge_logits = self.rel_proj(self._pair_features(e_src, e_dst))
                edge_loss_sum = edge_loss_sum + nn.functional.cross_entropy(
                    edge_logits[edge_has_match],
                    valid_rel_targets[:num_edges][edge_has_match],
                    reduction="sum",
                )
            else:
                edge_rel_logits = rel_logits[batch_idx, src[:num_edges], :, :]
                combined_rel_logits = edge_rel_logits.masked_fill(~edge_match.unsqueeze(-1), -1e4)
                combined_rel_logits = torch.logsumexp(combined_rel_logits, dim=1)
                edge_loss_sum = edge_loss_sum + nn.functional.cross_entropy(
                    combined_rel_logits[edge_has_match],
                    valid_rel_targets[:num_edges][edge_has_match],
                    reduction="sum",
                )
            total_edges += int(edge_has_match.sum().item())

        trace_loss = trace_loss_sum / max(total_hops, 1)
        edge_loss = edge_loss_sum / max(total_edges, 1)
        return trace_loss, edge_loss

    def forward(
        self,
        batch_data,
        lambda1=1.0,
        lambda_edge=0.0,
        lambda_cons=0.0,
        lambda_gate=0.0,
        lambda_alg=0.0,
        lambda_eq=0.0,
    ):
        input_ids = batch_data["input_ids"]
        attention_mask = batch_data["attention_mask"]
        labels = batch_data["labels"]
        entity_spans = batch_data["entity_spans"]
        path_node_ids = batch_data["path_node_ids"]
        path_rel_ids = batch_data.get("path_rel_ids")
        query_indices = batch_data["query_indices"]

        logits_orig, entity_embs_upd, sparse_edges = self.compute_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            entity_spans=entity_spans,
            query_indices=query_indices,
            path_node_ids=path_node_ids,
        )
        main_loss = nn.functional.cross_entropy(logits_orig, labels)

        cons_loss = torch.tensor(0.0, device=self.device)
        if batch_data.get("aug_input_ids") is not None:
            logits_aug, entity_embs_aug, sparse_edges_aug = self.compute_logits(
                input_ids=batch_data["aug_input_ids"],
                attention_mask=batch_data["aug_attention_mask"],
                entity_spans=batch_data["aug_entity_spans"],
                query_indices=query_indices,
                path_node_ids=path_node_ids,
            )
            loss_aug_ce = nn.functional.cross_entropy(logits_aug, labels)
            main_loss = 0.5 * main_loss + 0.5 * loss_aug_ce
            cons_loss = self._compute_consistency_loss(logits_orig, logits_aug)
        else:
            logits_aug = None
            entity_embs_aug = None
            sparse_edges_aug = None

        loss_trace_marg, loss_edge = self._compute_sparse_trace_losses(
            sparse_edges,
            entity_embs_upd,
            path_node_ids,
            path_rel_ids,
        )
        residual_alpha = torch.sigmoid(self.residual_gate_logit)
        gate_loss = residual_alpha.square() if self.use_path_algebra else residual_alpha.new_tensor(0.0)
        alg_loss = self.compute_algebra_loss() if self.use_path_algebra else residual_alpha.new_tensor(0.0)
        eq_loss = residual_alpha.new_tensor(0.0)
        if logits_aug is not None and lambda_eq != 0.0:
            eq_loss = self.compute_rename_equivariance_loss(
                logits_orig,
                logits_aug,
                entity_embs_upd,
                entity_embs_aug,
                sparse_edges,
                sparse_edges_aug,
                entity_spans,
                path_node_ids,
            )

        total_loss = (
            main_loss
            + lambda1 * loss_trace_marg
            + lambda_edge * loss_edge
            + lambda_cons * cons_loss
            + lambda_gate * gate_loss
            + lambda_alg * alg_loss
            + lambda_eq * eq_loss
        )
        return {
            "loss": total_loss,
            "losses": {
                "main": main_loss.item(),
                "nexthop": loss_trace_marg.item(),
                "trace_marg": loss_trace_marg.item(),
                "edge": loss_edge.item(),
                "cons": cons_loss.item(),
                "gate": gate_loss.item(),
                "alg": alg_loss.item(),
                "eq": eq_loss.item(),
                "residual_alpha": residual_alpha.item(),
            },
            "logits": logits_orig,
        }
