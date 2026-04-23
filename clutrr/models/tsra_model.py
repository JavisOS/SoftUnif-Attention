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
    ):
        super().__init__()
        self.device = device

        self.model_type = model_type.lower()
        self.decoder_only = is_decoder_only_model(self.model_type)
        self.pooling = pooling or ("last_token" if self.decoder_only else "cls")
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
        )
        self.pair_classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 21),
        )
        self.rel_proj = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 21),
        )
        self.rel_comp_emb = nn.Embedding(21, self.hidden_size)
        self.rel_comp_cell = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.rel_comp_norm = nn.LayerNorm(self.hidden_size)
        self.comp_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 21),
        )

    def _pool_sequence(self, sequence_output, attention_mask):
        if self.pooling == "cls":
            return sequence_output[:, 0, :]

        token_lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        b_idx = torch.arange(sequence_output.size(0), device=sequence_output.device)
        return sequence_output[b_idx, token_lengths, :]

    def get_entity_embeddings(self, last_hidden_state, entity_spans):
        bsz, max_entities, _ = entity_spans.shape
        valid_mask = entity_spans[:, :, 0] != -1

        emb_list = []
        for i in range(bsz):
            sample_embs = []
            hidden = last_hidden_state[i]
            for j in range(max_entities):
                if valid_mask[i, j]:
                    start, end = entity_spans[i, j]
                    start = min(start, hidden.size(0) - 1)
                    end = min(end, hidden.size(0))
                    if start >= end:
                        end = start + 1
                    pool = hidden[start:end].mean(dim=0)
                    sample_embs.append(pool)
                else:
                    sample_embs.append(hidden.new_zeros(self.hidden_size))
            emb_list.append(torch.stack(sample_embs))

        return torch.stack(emb_list)

    def compute_shared_edge_composition_logits(self, path_node_ids, entity_embs):
        if path_node_ids is None:
            return None, None, {"shared_edge_count": 0, "edge_entropy": 0.0}

        bsz = path_node_ids.size(0)
        valid_samples = torch.zeros(bsz, device=entity_embs.device, dtype=torch.bool)
        final_states = []

        entropy_sum = torch.zeros((), device=entity_embs.device, dtype=entity_embs.dtype)
        shared_edge_count = 0

        for i in range(bsz):
            path = path_node_ids[i]
            valid_path = path[path != -1]
            if valid_path.numel() < 2:
                final_states.append(entity_embs.new_zeros(self.hidden_size))
                continue

            valid_samples[i] = True
            src_idx = valid_path[:-1]
            dst_idx = valid_path[1:]

            e_src = entity_embs[i, src_idx]
            e_dst = entity_embs[i, dst_idx]
            edge_feats = torch.cat([e_src, e_dst, e_src * e_dst], dim=-1)
            edge_logits = self.rel_proj(edge_feats)
            edge_probs = torch.softmax(edge_logits, dim=-1)

            soft_rel_embs = torch.matmul(edge_probs, self.rel_comp_emb.weight)

            sample_state = entity_embs.new_zeros((1, self.hidden_size))
            for step_idx in range(soft_rel_embs.size(0)):
                sample_state = self.rel_comp_cell(
                    soft_rel_embs[step_idx].unsqueeze(0),
                    sample_state,
                )
            final_states.append(sample_state.squeeze(0))

            step_entropy = -(edge_probs * torch.log(edge_probs.clamp(min=1e-8))).sum(dim=-1)
            entropy_sum = entropy_sum + step_entropy.sum()
            shared_edge_count += int(edge_probs.size(0))

        state = torch.stack(final_states, dim=0)
        state = self.rel_comp_norm(state)
        comp_logits = self.comp_classifier(state).to(self.classifier.weight.dtype)

        edge_entropy = (entropy_sum / shared_edge_count).item() if shared_edge_count > 0 else 0.0
        diag = {
            "shared_edge_count": shared_edge_count,
            "edge_entropy": edge_entropy,
        }
        return comp_logits, valid_samples, diag

    def compute_logits(self, input_ids, attention_mask, entity_spans, query_indices):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = out.last_hidden_state
        head_dtype = self.classifier.weight.dtype
        sequence_for_heads = sequence_output.to(head_dtype)

        cls_output = self._pool_sequence(sequence_for_heads, attention_mask)
        logits_cls = self.classifier(cls_output)

        entity_embs = self.get_entity_embeddings(sequence_for_heads, entity_spans)
        valid_mask = entity_spans[:, :, 0] != -1
        obj_indices = query_indices[:, 1]
        entity_embs_upd, attn_scores = self.entity_attn(entity_embs, valid_mask, obj_indices=obj_indices)

        sub_idx = query_indices[:, 0].clamp(min=0)
        obj_idx = query_indices[:, 1].clamp(min=0)
        b_idx = torch.arange(entity_embs_upd.size(0), device=entity_embs_upd.device)
        e_sub = entity_embs_upd[b_idx, sub_idx]
        e_obj = entity_embs_upd[b_idx, obj_idx]
        pair_feats = torch.cat([e_sub, e_obj, e_sub * e_obj], dim=-1)
        logits_pair = self.pair_classifier(pair_feats)

        logits = logits_cls + logits_pair
        return logits, entity_embs_upd, attn_scores

    def forward(
        self,
        batch_data,
        lambda1=1.0,
        lambda_edge=0.0,
        lambda_cons=0.0,
        lambda_comp=0.1,
        use_relation_composition=True,
        use_composition_logits=False,
        composition_logit_weight=0.1,
    ):
        input_ids = batch_data["input_ids"]
        attention_mask = batch_data["attention_mask"]
        labels = batch_data["labels"]
        entity_spans = batch_data["entity_spans"]
        path_node_ids = batch_data["path_node_ids"]
        path_rel_ids = batch_data.get("path_rel_ids")
        query_indices = batch_data["query_indices"]

        logits_orig, entity_embs_upd, attn_scores = self.compute_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            entity_spans=entity_spans,
            query_indices=query_indices,
        )

        comp_logits = None
        comp_valid = None
        loss_comp = torch.tensor(0.0, device=logits_orig.device)
        comp_correct = 0
        comp_total = 0
        logits_for_main = logits_orig

        comp_diag = {"shared_edge_count": 0, "edge_entropy": 0.0}
        if use_relation_composition and path_node_ids is not None:
            comp_logits, comp_valid, comp_diag = self.compute_shared_edge_composition_logits(
                path_node_ids,
                entity_embs_upd,
            )

            if comp_logits is not None and comp_valid is not None and comp_valid.any():
                comp_targets = labels[comp_valid]
                comp_logits_valid = comp_logits[comp_valid]
                loss_comp = nn.functional.cross_entropy(comp_logits_valid, comp_targets)

                comp_preds = torch.argmax(comp_logits_valid, dim=1)
                comp_correct = int((comp_preds == comp_targets).sum().item())
                comp_total = int(comp_targets.numel())

                if use_composition_logits and composition_logit_weight != 0.0:
                    comp_gate = comp_valid.to(logits_orig.dtype).unsqueeze(-1)
                    logits_for_main = logits_orig + composition_logit_weight * comp_logits * comp_gate

        main_loss = nn.functional.cross_entropy(logits_for_main, labels)

        cons_loss = torch.tensor(0.0).to(self.device)
        if batch_data.get("aug_input_ids") is not None:
            logits_aug, entity_embs_aug, _ = self.compute_logits(
                input_ids=batch_data["aug_input_ids"],
                attention_mask=batch_data["aug_attention_mask"],
                entity_spans=batch_data["aug_entity_spans"],
                query_indices=query_indices,
            )

            logits_aug_for_main = logits_aug
            if (
                use_relation_composition
                and use_composition_logits
                and composition_logit_weight != 0.0
                and path_node_ids is not None
            ):
                comp_logits_aug, comp_valid_aug, _ = self.compute_shared_edge_composition_logits(
                    path_node_ids,
                    entity_embs_aug,
                )
                has_aug_comp = (
                    comp_logits_aug is not None
                    and comp_valid_aug is not None
                    and comp_valid_aug.any()
                )
                if has_aug_comp:
                    comp_aug_gate = comp_valid_aug.to(logits_aug.dtype).unsqueeze(-1)
                    logits_aug_for_main = (
                        logits_aug + composition_logit_weight * comp_logits_aug * comp_aug_gate
                    )

            loss_aug_ce = nn.functional.cross_entropy(logits_aug_for_main, labels)
            p_clean = nn.functional.softmax(logits_for_main, dim=1)
            p_aug = nn.functional.log_softmax(logits_aug_for_main, dim=1)
            kl_loss = nn.functional.kl_div(p_aug, p_clean, reduction="batchmean")
            main_loss = 0.5 * main_loss + 0.5 * loss_aug_ce
            cons_loss = kl_loss

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

                    e_src = entity_embs[i, edge_src]
                    e_dst = entity_embs[i, edge_dst]
                    edge_feats = torch.cat([e_src, e_dst, e_src * e_dst], dim=-1)
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

        total_loss = (
            main_loss
            + lambda1 * loss_nexthop
            + lambda_edge * loss_edge
            + lambda_cons * cons_loss
            + lambda_comp * loss_comp
        )
        return {
            "loss": total_loss,
            "losses": {
                "main": main_loss.item(),
                "nexthop": loss_nexthop.item(),
                "edge": loss_edge.item(),
                "cons": cons_loss.item(),
                "comp": loss_comp.item(),
            },
            "metrics": {
                "comp_correct": comp_correct,
                "comp_total": comp_total,
                "comp_acc": (comp_correct / comp_total) if comp_total > 0 else 0.0,
                "comp_shared_edge_count": int(comp_diag["shared_edge_count"]),
                "comp_edge_entropy": float(comp_diag["edge_entropy"]),
                "comp_from_shared_edges": 1 if comp_diag["shared_edge_count"] > 0 else 0,
            },
            "logits": logits_for_main,
        }
