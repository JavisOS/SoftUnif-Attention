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


def _count_reference_transition_hits(sparse_edges, path_node_ids):
    """Count top-1 destination hits over all annotated path transitions."""
    if sparse_edges is None or path_node_ids is None:
        return 0, 0

    edge_index = sparse_edges["edge_index"]
    hop_logits = sparse_edges["hop_logits"]
    edge_valid = sparse_edges.get("edge_valid", torch.ones_like(edge_index, dtype=torch.bool))
    selected_position = hop_logits.masked_fill(~edge_valid, -1e4).argmax(dim=-1, keepdim=True)
    selected_destination = torch.gather(edge_index, 2, selected_position).squeeze(-1)

    hits = 0
    total = 0
    max_units = selected_destination.size(1)
    for batch_index, padded_path in enumerate(path_node_ids):
        path = padded_path[padded_path >= 0]
        if path.numel() < 2:
            continue
        source = path[:-1]
        destination = path[1:]
        in_bounds = (
            (source >= 0)
            & (source < max_units)
            & (destination >= 0)
            & (destination < max_units)
        )
        if not bool(in_bounds.any()):
            continue
        source = source[in_bounds]
        destination = destination[in_bounds]
        predictions = selected_destination[batch_index, source]
        hits += int((predictions == destination).sum().item())
        total += int(destination.numel())
    return hits, total


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
    transition_hits = 0
    transition_total = 0

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            batch = _move_batch_to_device(batch, device=device, skip_keys={"raw_batch"})

            y_target = batch["labels"]
            hops = batch["hops"]

            logits_base, _, sparse_edges_base = base_model.compute_logits(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                entity_spans=batch["entity_spans"],
                query_indices=batch["query_indices"],
            )
            preds_base = torch.argmax(logits_base, dim=1)
            batch_transition_hits, batch_transition_total = _count_reference_transition_hits(
                sparse_edges_base,
                batch.get("path_node_ids"),
            )
            transition_hits += batch_transition_hits
            transition_total += batch_transition_total

            logits_mod, _, _ = base_model.compute_logits(
                input_ids=batch["aug_input_ids"],
                attention_mask=batch["aug_attention_mask"],
                entity_spans=batch["aug_entity_spans"],
                query_indices=batch["query_indices"],
            )
            preds_mod = torch.argmax(logits_mod, dim=1)

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

    per_hop = {}
    for hop in sorted(by_hop_total):
        hop_total = by_hop_total[hop]
        hop_correct = by_hop_correct.get(hop, 0)
        per_hop[int(hop)] = {
            "accuracy": hop_correct / hop_total if hop_total > 0 else 0,
            "correct": int(hop_correct),
            "total": int(hop_total),
        }

    print(f"[Stats] Evaluated {total} samples.")
    print(f"  Changed Story Rate:   {changed_story_rate:.4f}")
    print(f"  Changed Query Rate:   {changed_query_rate:.4f}")
    if transition_total > 0:
        print(f"  Transition@1:         {transition_hits / transition_total:.4f}")

    return {
        "overall": acc_overall,
        "renamed": acc_renamed,
        "consistency": prob_consistent,
        "consistent_and_correct": prob_robust_correct,
        "short_hop": short_acc,
        "long_hop": long_acc,
        "per_hop": per_hop,
        "transition_at_1": transition_hits / transition_total if transition_total > 0 else 0,
        "transition_hits": transition_hits,
        "transition_total": transition_total,
    }
