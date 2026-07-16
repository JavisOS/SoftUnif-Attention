"""Paired evaluation for semantics-preserving CLUTRR query reversal."""

from __future__ import annotations

import torch
import torch.nn as nn


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if key == "raw_batch":
            moved[key] = value
        elif isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _predict(base_model: nn.Module, batch: dict) -> torch.Tensor:
    logits, _, _ = base_model.compute_logits(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        entity_spans=batch["entity_spans"],
        query_indices=batch["query_indices"],
    )
    return logits.argmax(dim=1)


def evaluate_paired_query_direction(
    model: nn.Module,
    original_loader,
    reversed_loader,
    device: torch.device,
) -> dict:
    """Evaluate aligned original/reversed queries over identical stories."""

    model.eval()
    base_model = _unwrap_model(model)
    total = 0
    original_correct = 0
    reversed_correct = 0
    both_correct = 0
    changed_label_total = 0
    changed_prediction = 0
    changed_both_correct = 0
    same_label_total = 0

    with torch.no_grad():
        original_iter = iter(original_loader)
        reversed_iter = iter(reversed_loader)
        while True:
            try:
                original_batch = next(original_iter)
            except StopIteration:
                try:
                    next(reversed_iter)
                except StopIteration:
                    break
                raise ValueError("Reversed query loader has extra batches")
            try:
                reversed_batch = next(reversed_iter)
            except StopIteration as error:
                raise ValueError("Original query loader has extra batches") from error

            if original_batch is None or reversed_batch is None:
                raise ValueError("Direction evaluation received an empty batch")
            if len(original_batch["raw_batch"]) != len(reversed_batch["raw_batch"]):
                raise ValueError("Direction evaluation batches are not aligned")

            for original_item, reversed_item in zip(
                original_batch["raw_batch"],
                reversed_batch["raw_batch"],
            ):
                if original_item["story"] != reversed_item["story"]:
                    raise ValueError("Query reversal changed the CLUTRR story")
                if tuple(reversed(reversed_item["query"])) != tuple(original_item["query"]):
                    raise ValueError("Reversed query endpoints are not aligned")

            original_batch = _move_batch(original_batch, device)
            reversed_batch = _move_batch(reversed_batch, device)
            original_predictions = _predict(base_model, original_batch)
            reversed_predictions = _predict(base_model, reversed_batch)
            original_labels = original_batch["labels"]
            reversed_labels = reversed_batch["labels"]

            original_hits = original_predictions == original_labels
            reversed_hits = reversed_predictions == reversed_labels
            changed_labels = original_labels != reversed_labels
            same_labels = ~changed_labels

            batch_size = int(original_labels.numel())
            total += batch_size
            original_correct += int(original_hits.sum().item())
            reversed_correct += int(reversed_hits.sum().item())
            both_correct += int((original_hits & reversed_hits).sum().item())
            changed_label_total += int(changed_labels.sum().item())
            same_label_total += int(same_labels.sum().item())
            changed_prediction += int(
                ((original_predictions != reversed_predictions) & changed_labels).sum().item()
            )
            changed_both_correct += int(
                (original_hits & reversed_hits & changed_labels).sum().item()
            )

    return {
        "total": total,
        "original_accuracy": original_correct / total if total else 0.0,
        "reversed_accuracy": reversed_correct / total if total else 0.0,
        "both_correct_rate": both_correct / total if total else 0.0,
        "changed_label_total": changed_label_total,
        "same_label_total": same_label_total,
        "prediction_change_rate_changed_labels": (
            changed_prediction / changed_label_total if changed_label_total else 0.0
        ),
        "both_correct_rate_changed_labels": (
            changed_both_correct / changed_label_total if changed_label_total else 0.0
        ),
    }
