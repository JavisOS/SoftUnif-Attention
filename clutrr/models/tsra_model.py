import torch
import torch.nn as nn

from clutrr.models.backbones import build_backbone_model, is_decoder_only_model
from clutrr.models.relation_attention import RelationConditionedEntityAttention


class TsraReasonerModel(nn.Module):
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
        entity_pooling="mean",
        prediction_head="cls_pair",
        consistency_mode="kl",
        use_relation_conditioning=True,
        edge_supervision_target="separate",
        pair_feature_mode="product",
    ):
        super().__init__()
        self.device = device

        self.model_type = model_type.lower()
        self.decoder_only = is_decoder_only_model(self.model_type)
        self.pooling = pooling or ("last_token" if self.decoder_only else "cls")
        if entity_pooling not in {"mean", "multi_mention", "query_aware"}:
            raise ValueError(f"Unsupported entity_pooling mode: {entity_pooling}")
        if prediction_head not in {"cls_pair", "cls_only", "pair_only", "gated"}:
            raise ValueError(f"Unsupported prediction_head mode: {prediction_head}")
        if consistency_mode not in {"kl", "sym_kl", "js", "mse"}:
            raise ValueError(f"Unsupported consistency_mode: {consistency_mode}")
        if edge_supervision_target not in {"separate", "latent"}:
            raise ValueError(f"Unsupported edge_supervision_target: {edge_supervision_target}")
        if pair_feature_mode not in {"product", "product_diff"}:
            raise ValueError(f"Unsupported pair_feature_mode: {pair_feature_mode}")
        self.entity_pooling = entity_pooling
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
            use_relation_conditioning=use_relation_conditioning,
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
        self.query_ctx_proj = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.Tanh(),
        )
        self.mention_key_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.mention_query_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.mention_score = nn.Linear(self.hidden_size, 1)

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

    def _pool_single_span(self, hidden, span):
        start, end = span
        start = min(int(start), hidden.size(0) - 1)
        end = min(int(end), hidden.size(0))
        if start >= end:
            end = start + 1
        return hidden[start:end].mean(dim=0)

    def _get_single_span_embeddings(self, last_hidden_state, entity_spans):
        bsz, max_entities, _ = entity_spans.shape
        valid_mask = entity_spans[:, :, 0] != -1

        emb_list = []
        for i in range(bsz):
            sample_embs = []
            hidden = last_hidden_state[i]
            for j in range(max_entities):
                if valid_mask[i, j]:
                    sample_embs.append(self._pool_single_span(hidden, entity_spans[i, j]))
                else:
                    sample_embs.append(torch.zeros(self.hidden_size).to(self.device))
            emb_list.append(torch.stack(sample_embs))

        return torch.stack(emb_list)

    def _get_mention_embeddings(self, last_hidden_state, entity_mention_spans):
        bsz, max_entities, max_mentions, _ = entity_mention_spans.shape
        mention_mask = entity_mention_spans[:, :, :, 0] != -1
        mention_embs = torch.zeros(
            bsz,
            max_entities,
            max_mentions,
            self.hidden_size,
            device=last_hidden_state.device,
            dtype=last_hidden_state.dtype,
        )

        for i in range(bsz):
            hidden = last_hidden_state[i]
            for j in range(max_entities):
                for k in range(max_mentions):
                    if mention_mask[i, j, k]:
                        mention_embs[i, j, k] = self._pool_single_span(hidden, entity_mention_spans[i, j, k])

        return mention_embs, mention_mask

    def _build_query_context(self, base_entity_embs, query_indices):
        max_entities = base_entity_embs.size(1)
        sub_idx = query_indices[:, 0].clamp(min=0, max=max_entities - 1)
        obj_idx = query_indices[:, 1].clamp(min=0, max=max_entities - 1)
        b_idx = torch.arange(base_entity_embs.size(0), device=base_entity_embs.device)
        e_sub = base_entity_embs[b_idx, sub_idx]
        e_obj = base_entity_embs[b_idx, obj_idx]
        return self.query_ctx_proj(torch.cat([e_sub, e_obj, e_sub * e_obj], dim=-1))

    @staticmethod
    def _aggregate_mentions_mean(mention_embs, mention_mask):
        mention_mask_f = mention_mask.unsqueeze(-1).to(mention_embs.dtype)
        mention_counts = mention_mask_f.sum(dim=2).clamp(min=1.0)
        return (mention_embs * mention_mask_f).sum(dim=2) / mention_counts

    def _aggregate_mentions_query_aware(self, mention_embs, mention_mask, query_indices):
        base_entity_embs = self._aggregate_mentions_mean(mention_embs, mention_mask)
        query_ctx = self._build_query_context(base_entity_embs, query_indices)
        mention_keys = self.mention_key_proj(mention_embs)
        query_bias = self.mention_query_proj(query_ctx).unsqueeze(1).unsqueeze(2)
        mention_scores = self.mention_score(torch.tanh(mention_keys + query_bias)).squeeze(-1)
        mention_scores = mention_scores.masked_fill(~mention_mask, -1e4)

        attn = torch.softmax(mention_scores, dim=-1)
        attn = attn * mention_mask.to(attn.dtype)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return (attn.unsqueeze(-1) * mention_embs).sum(dim=2)

    def get_entity_embeddings(self, last_hidden_state, entity_spans, query_indices, entity_mention_spans=None):
        if self.entity_pooling == "mean" or entity_mention_spans is None:
            return self._get_single_span_embeddings(last_hidden_state, entity_spans)

        mention_embs, mention_mask = self._get_mention_embeddings(last_hidden_state, entity_mention_spans)
        if self.entity_pooling == "multi_mention":
            return self._aggregate_mentions_mean(mention_embs, mention_mask)
        return self._aggregate_mentions_query_aware(mention_embs, mention_mask, query_indices)

    def compute_logits(
        self,
        input_ids,
        attention_mask,
        entity_spans,
        query_indices,
        entity_mention_spans=None,
        return_rel_logits=False,
    ):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = out.last_hidden_state
        head_dtype = self.classifier.weight.dtype
        sequence_for_heads = sequence_output.to(head_dtype)

        cls_output = self._pool_sequence(sequence_for_heads, attention_mask)
        logits_cls = self.classifier(cls_output)

        entity_embs = self.get_entity_embeddings(
            sequence_for_heads,
            entity_spans,
            query_indices,
            entity_mention_spans=entity_mention_spans,
        )
        valid_mask = entity_spans[:, :, 0] != -1
        obj_indices = query_indices[:, 1]
        entity_embs_upd, attn_scores, rel_logits = self.entity_attn(entity_embs, valid_mask, obj_indices=obj_indices)

        sub_idx = query_indices[:, 0].clamp(min=0)
        obj_idx = query_indices[:, 1].clamp(min=0)
        b_idx = torch.arange(entity_embs_upd.size(0), device=entity_embs_upd.device)
        e_sub = entity_embs_upd[b_idx, sub_idx]
        e_obj = entity_embs_upd[b_idx, obj_idx]
        pair_feats = self._pair_features(e_sub, e_obj)
        logits_pair = self.pair_classifier(pair_feats)

        if self.prediction_head == "cls_only":
            logits = logits_cls
        elif self.prediction_head == "pair_only":
            logits = logits_pair
        elif self.prediction_head == "gated":
            cls_weight = torch.sigmoid(self.fusion_logit)
            logits = cls_weight * logits_cls + (1.0 - cls_weight) * logits_pair
        else:
            logits = logits_cls + logits_pair
        if return_rel_logits:
            return logits, entity_embs_upd, attn_scores, rel_logits
        return logits, entity_embs_upd, attn_scores

    def forward(self, batch_data, lambda1=1.0, lambda_edge=0.0, lambda_cons=0.0):
        input_ids = batch_data["input_ids"]
        attention_mask = batch_data["attention_mask"]
        labels = batch_data["labels"]
        entity_spans = batch_data["entity_spans"]
        entity_mention_spans = batch_data.get("entity_mention_spans")
        path_node_ids = batch_data["path_node_ids"]
        path_rel_ids = batch_data.get("path_rel_ids")
        query_indices = batch_data["query_indices"]

        logits_orig, entity_embs_upd, attn_scores, rel_logits = self.compute_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            entity_spans=entity_spans,
            query_indices=query_indices,
            entity_mention_spans=entity_mention_spans,
            return_rel_logits=True,
        )
        main_loss = nn.functional.cross_entropy(logits_orig, labels)

        cons_loss = torch.tensor(0.0).to(self.device)
        if batch_data.get("aug_input_ids") is not None:
            logits_aug, _, _ = self.compute_logits(
                input_ids=batch_data["aug_input_ids"],
                attention_mask=batch_data["aug_attention_mask"],
                entity_spans=batch_data["aug_entity_spans"],
                query_indices=query_indices,
                entity_mention_spans=batch_data.get("aug_entity_mention_spans"),
            )
            loss_aug_ce = nn.functional.cross_entropy(logits_aug, labels)
            main_loss = 0.5 * main_loss + 0.5 * loss_aug_ce
            if self.consistency_mode == "mse":
                cons_loss = nn.functional.mse_loss(logits_aug, logits_orig.detach())
            else:
                p_orig = nn.functional.softmax(logits_orig, dim=1)
                log_p_orig = nn.functional.log_softmax(logits_orig, dim=1)
                p_aug = nn.functional.softmax(logits_aug, dim=1)
                log_p_aug = nn.functional.log_softmax(logits_aug, dim=1)
                if self.consistency_mode == "sym_kl":
                    cons_loss = 0.5 * nn.functional.kl_div(log_p_aug, p_orig.detach(), reduction="batchmean")
                    cons_loss = cons_loss + 0.5 * nn.functional.kl_div(log_p_orig, p_aug.detach(), reduction="batchmean")
                elif self.consistency_mode == "js":
                    m = 0.5 * (p_orig.detach() + p_aug.detach()).clamp(min=1e-8)
                    cons_loss = 0.5 * nn.functional.kl_div(log_p_orig, m, reduction="batchmean")
                    cons_loss = cons_loss + 0.5 * nn.functional.kl_div(log_p_aug, m, reduction="batchmean")
                else:
                    cons_loss = nn.functional.kl_div(log_p_aug, p_orig.detach(), reduction="batchmean")

        entity_embs = entity_embs_upd
        loss_nexthop = torch.tensor(0.0).to(self.device)
        loss_edge = torch.tensor(0.0).to(self.device)
        total_hops = 0
        total_edges = 0
        batch_nexthop_loss = torch.tensor(0.0, device=self.device)
        batch_edge_loss = torch.tensor(0.0, device=self.device)

        for i in range(len(input_ids)):
            path = path_node_ids[i]
            valid_path = path[path != -1]
            if len(valid_path) < 2:
                continue

            input_indices = valid_path[:-1]
            target_indices = valid_path[1:]
            step_logits = attn_scores[i, input_indices, :]
            step_loss = nn.functional.cross_entropy(step_logits, target_indices)
            batch_nexthop_loss += step_loss
            total_hops += 1

            if path_rel_ids is not None:
                rel_targets = path_rel_ids[i]
                valid_rel_targets = rel_targets[rel_targets != -1]
                num_edges = min(input_indices.numel(), valid_rel_targets.numel())
                if num_edges > 0:
                    edge_src = input_indices[:num_edges]
                    edge_dst = target_indices[:num_edges]
                    edge_labels = valid_rel_targets[:num_edges]

                    if self.edge_supervision_target == "latent":
                        edge_logits = rel_logits[i, edge_src, edge_dst]
                    else:
                        e_src = entity_embs[i, edge_src]
                        e_dst = entity_embs[i, edge_dst]
                        edge_feats = self._pair_features(e_src, e_dst)
                        edge_logits = self.rel_proj(edge_feats)
                    batch_edge_loss += nn.functional.cross_entropy(
                        edge_logits,
                        edge_labels,
                        reduction="sum",
                    )
                    total_edges += num_edges

        if total_hops > 0:
            loss_nexthop = batch_nexthop_loss / total_hops
        if total_edges > 0:
            loss_edge = batch_edge_loss / total_edges

        total_loss = main_loss + lambda1 * loss_nexthop + lambda_edge * loss_edge + lambda_cons * cons_loss
        return {
            "loss": total_loss,
            "losses": {
                "main": main_loss.item(),
                "nexthop": loss_nexthop.item(),
                "edge": loss_edge.item(),
                "cons": cons_loss.item(),
            },
            "logits": logits_orig,
        }
