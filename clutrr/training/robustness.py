import torch
import torch.nn as nn


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _move_batch_to_device(batch: dict, device: torch.device, skip_keys: set[str] | None = None) -> dict:
    skip_keys = skip_keys or set()
    moved = {}
    for key, value in batch.items():
        if key in skip_keys:
            moved[key] = value
        elif isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def evaluate_robustness(model, loader, device):
    model.eval()
    base_model = _unwrap_model(model)

    total = 0
    correct_base = 0
    correct_mod = 0
    consistent = 0
    consistent_and_correct = 0

    count_changed_story = 0
    count_changed_query = 0

    by_hop_total = {}
    by_hop_correct = {}
    composition_correct = 0
    composition_total = 0

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            batch = _move_batch_to_device(batch, device=device, skip_keys={"raw_batch"})

            y_target = batch["labels"]
            hops = batch["hops"]

            logits_base, entity_embs_base, _ = base_model.compute_logits(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                entity_spans=batch["entity_spans"],
                query_indices=batch["query_indices"],
            )
            preds_base = torch.argmax(logits_base, dim=1)

            logits_mod, _, _ = base_model.compute_logits(
                input_ids=batch["aug_input_ids"],
                attention_mask=batch["aug_attention_mask"],
                entity_spans=batch["aug_entity_spans"],
                query_indices=batch["query_indices"],
            )
            preds_mod = torch.argmax(logits_mod, dim=1)

            path_node_ids = batch.get("path_node_ids")
            if path_node_ids is not None and hasattr(base_model, "compute_shared_edge_composition_logits"):
                comp_logits, comp_valid, _ = base_model.compute_shared_edge_composition_logits(
                    path_node_ids,
                    entity_embs_base,
                )
                if comp_logits is not None and comp_valid is not None and comp_valid.any():
                    comp_preds = torch.argmax(comp_logits[comp_valid], dim=1)
                    comp_targets = y_target[comp_valid]
                    composition_correct += (comp_preds == comp_targets).sum().item()
                    composition_total += comp_targets.numel()

            for sample in batch["raw_batch"]:
                story_text = sample["story"]
                query_text = f"{sample['query'][0]} and {sample['query'][1]}"
                story_mod = sample.get("aug_story", story_text)
                query_mod = (
                    f"{sample.get('aug_query', sample['query'])[0]} "
                    f"and {sample.get('aug_query', sample['query'])[1]}"
                )
                if story_mod != story_text:
                    count_changed_story += 1
                if query_mod != query_text:
                    count_changed_query += 1

            total += len(y_target)
            correct_mask = preds_base == y_target
            correct_mod_mask = preds_mod == y_target
            consistent_mask = preds_base == preds_mod

            correct_base += correct_mask.sum().item()
            correct_mod += correct_mod_mask.sum().item()
            consistent += consistent_mask.sum().item()
            consistent_and_correct += (consistent_mask & correct_mask).sum().item()

            preds_np = preds_base.cpu().numpy()
            targets_np = y_target.cpu().numpy()
            hops_np = hops.cpu().numpy()
            for hop, pred, target in zip(hops_np, preds_np, targets_np):
                if hop not in by_hop_total:
                    by_hop_total[hop] = 0
                    by_hop_correct[hop] = 0
                by_hop_total[hop] += 1
                if pred == target:
                    by_hop_correct[hop] += 1

    acc_overall = correct_base / total if total > 0 else 0
    acc_renamed = correct_mod / total if total > 0 else 0
    prob_consistent = consistent / total if total > 0 else 0
    prob_robust_correct = consistent_and_correct / total if total > 0 else 0

    changed_story_rate = count_changed_story / total if total > 0 else 0
    changed_query_rate = count_changed_query / total if total > 0 else 0

    short_corr = sum([by_hop_correct.get(h, 0) for h in [2, 3]])
    short_tot = sum([by_hop_total.get(h, 0) for h in [2, 3]])
    short_acc = short_corr / short_tot if short_tot > 0 else 0

    long_corr = sum([by_hop_correct.get(h, 0) for h in range(6, 15)])
    long_tot = sum([by_hop_total.get(h, 0) for h in range(6, 15)])
    long_acc = long_corr / long_tot if long_tot > 0 else 0
    composition_acc = composition_correct / composition_total if composition_total > 0 else 0

    print(f"[Stats] Evaluated {total} samples.")
    print(f"  Changed Story Rate:   {changed_story_rate:.4f}")
    print(f"  Changed Query Rate:   {changed_query_rate:.4f}")

    return {
        "overall": acc_overall,
        "renamed": acc_renamed,
        "consistency": prob_consistent,
        "consistent_and_correct": prob_robust_correct,
        "short_hop": short_acc,
        "long_hop": long_acc,
        "composition_acc": composition_acc,
        "composition_total": composition_total,
    }
