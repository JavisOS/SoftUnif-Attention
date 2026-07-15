"""Validation-selected CREST-inspired counterfactual CLUTRR control.

This is not an official reproduction of CREST. It evaluates a previously used
counterfactual-augmentation idea under the paper's final text-input protocol:
the same train/validation split, encoder family, epoch budget, checkpoint rule,
and one post-selection test evaluation as the other controlled baselines.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from clutrr.cli.baseline import BaselineModel
from clutrr.config.relation_schema import RELATION_ID_MAP_21_WITH_NOTHING
from clutrr.models.backbones import build_tokenizer
from clutrr.training.model_selection import (
    clone_model_state,
    repository_revision,
    restore_model_state,
    stratified_train_validation_split,
    write_metrics,
)
from clutrr.utils.seed import set_seed
from scripts.crest_clutrr_baseline import (
    CrestCLUTRRDataset,
    CrestCollator,
    evaluate_crest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model_type", default="deberta")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation_fraction", type=float, default=0.1)
    parser.add_argument("--validation_seed", type=int, default=2027)
    parser.add_argument("--train_data_percentage", type=int, default=100)
    parser.add_argument("--test_data_percentage", type=int, default=100)
    parser.add_argument("--cf_weight", type=float, default=1.0)
    parser.add_argument("--consistency_weight", type=float, default=0.5)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--metrics_out", required=True)
    return parser


def counterfactual_loss(model, batch, cf_weight: float, consistency_weight: float):
    base = model(batch["base_input_ids"], batch["base_attention_mask"])
    renamed = model(batch["renamed_input_ids"], batch["renamed_attention_mask"])
    reverse = model(batch["reverse_input_ids"], batch["reverse_attention_mask"])

    main_loss = F.cross_entropy(base["logits"], batch["labels"])
    rename_loss = F.cross_entropy(renamed["logits"], batch["labels"])
    if batch["reverse_mask"].any():
        reverse_loss = F.cross_entropy(
            reverse["logits"][batch["reverse_mask"]],
            batch["reverse_labels"][batch["reverse_mask"]],
        )
    else:
        reverse_loss = main_loss.new_zeros(())
    consistency_loss = F.kl_div(
        F.log_softmax(renamed["logits"], dim=-1),
        F.softmax(base["logits"].detach(), dim=-1),
        reduction="batchmean",
    )
    total = (
        main_loss
        + cf_weight * 0.5 * (rename_loss + reverse_loss)
        + consistency_weight * consistency_loss
    )
    return total, {
        "main": float(main_loss.detach()),
        "rename": float(rename_loss.detach()),
        "reverse": float(reverse_loss.detach()),
        "consistency": float(consistency_loss.detach()),
    }


def run_training() -> None:
    args = build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    code_revision = repository_revision()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer(args.model_type, model_name_or_path=args.model_name_or_path)
    collator = CrestCollator(tokenizer, device=device, model_type=args.model_type)

    full_train = CrestCLUTRRDataset(
        args.root,
        args.dataset,
        "train",
        data_percentage=args.train_data_percentage,
        seed=42,
    )
    train_items = [full_train[index] for index in range(len(full_train))]
    strata = [(item["hops"], item["label"]) for item in train_items]
    train_data, validation_data = stratified_train_validation_split(
        full_train,
        strata,
        validation_fraction=args.validation_fraction,
        seed=args.validation_seed,
    )
    test_data = CrestCLUTRRDataset(
        args.root,
        args.dataset,
        "test",
        data_percentage=args.test_data_percentage,
        seed=999,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    model = BaselineModel(
        args.model_type,
        num_labels=len(RELATION_ID_MAP_21_WITH_NOTHING),
        model_name_or_path=args.model_name_or_path,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started_at = time.perf_counter()

    best_validation = -1.0
    best_epoch = -1
    best_state = None
    validation_history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {key: 0.0 for key in ("loss", "main", "rename", "reverse", "consistency")}
        batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, components = counterfactual_loss(
                model,
                batch,
                args.cf_weight,
                args.consistency_weight,
            )
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.detach())
            for key, value in components.items():
                totals[key] += value
            batches += 1

        validation_metrics = evaluate_crest(model, validation_loader)
        validation_history.append(
            {
                "epoch": epoch,
                "train": {key: value / max(batches, 1) for key, value in totals.items()},
                "validation": validation_metrics,
            }
        )
        print(
            f"Epoch {epoch:02d}: loss={totals['loss'] / max(batches, 1):.4f}, "
            f"validation={validation_metrics['overall']:.4f}",
            flush=True,
        )
        if validation_metrics["overall"] > best_validation:
            best_validation = validation_metrics["overall"]
            best_epoch = epoch
            best_state = clone_model_state(model)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started_at
    peak_memory_gib = (
        torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
    )
    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    restore_model_state(model, best_state)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_started_at = time.perf_counter()
    test_metrics = evaluate_crest(model, test_loader)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_seconds = time.perf_counter() - test_started_at

    result = {
        "code_revision": code_revision,
        "protocol": "controlled_text_input_final_v1",
        "implementation_scope": "crest_inspired_control_not_official_reproduction",
        "dataset": args.dataset,
        "model_type": args.model_type,
        "model_name_or_path": args.model_name_or_path,
        "seed": args.seed,
        "train_size": len(train_data),
        "validation_size": len(validation_data),
        "test_size": len(test_data),
        "validation_fraction": args.validation_fraction,
        "validation_seed": args.validation_seed,
        "selected_epoch": best_epoch,
        "selected_validation_overall": best_validation,
        "selection_rule": "highest validation answer accuracy; earliest epoch breaks ties",
        "validation_history": validation_history,
        "test": test_metrics,
        "resource_usage": {
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "peak_allocated_memory_gib": peak_memory_gib,
            "training_seconds": training_seconds,
            "test_seconds": test_seconds,
            "training_examples_per_second": (
                len(train_data) * args.epochs / max(training_seconds, 1e-9)
            ),
        },
        "configuration": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "cf_weight": args.cf_weight,
            "consistency_weight": args.consistency_weight,
            "training_views": ["original", "gender_preserving_rename", "reversed_query"],
            "test_time_input": "original_story_query_only",
            "uses_reference_path_or_edges": False,
        },
    }
    write_metrics(args.metrics_out, result)
    print(
        f"Selected epoch {best_epoch}; test overall={test_metrics['overall']:.4f}, "
        f"short={test_metrics['short_hop']:.4f}, long={test_metrics['long_hop']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    run_training()
