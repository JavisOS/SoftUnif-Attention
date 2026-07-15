"""Train controlled baselines under a declared CLUTRR checkpoint protocol."""

from __future__ import annotations

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from clutrr.config.defaults import DEFAULT_CLUTRR_DATASET, DEFAULT_CLUTRR_ROOT
from clutrr.data.trua_collator import TruaBatchCollator
from clutrr.data.trua_dataset import TruaClutrrDataset
from clutrr.models.backbones import build_tokenizer
from clutrr.models.controlled_unit_baselines import ControlledClutrrBaseline
from clutrr.training.model_selection import (
    clone_model_state,
    repository_revision,
    restore_model_state,
    stratified_train_validation_split,
    write_metrics,
)
from clutrr.training.robustness import evaluate_robustness
from clutrr.utils.seed import set_seed


def _move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if key == "raw_batch":
            continue
        moved[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return moved


def _deduplicate_dataset(dataset):
    seen = set()
    unique = []
    for item in dataset.data:
        key = (item["story"], tuple(item["query"]), item["target_id"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    removed = len(dataset.data) - len(unique)
    dataset.data = unique
    return removed


def _validation_score(metrics, selection_metric):
    if selection_metric == "overall":
        return float(metrics["overall"])
    selected = [
        item
        for hop, item in metrics.get("per_hop", {}).items()
        if 4 <= int(hop) <= 10
    ]
    correct = sum(item["correct"] for item in selected)
    total = sum(item["total"] for item in selected)
    if not total:
        raise ValueError("Validation data contain no official 4--10 hop examples")
    return correct / total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--core_type",
        required=True,
        choices=sorted(ControlledClutrrBaseline.SUPPORTED_CORES),
    )
    parser.add_argument("--model_type", default="deberta")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--root", default=DEFAULT_CLUTRR_ROOT)
    parser.add_argument("--dataset", default=DEFAULT_CLUTRR_DATASET)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation_fraction", type=float, default=0.1)
    parser.add_argument("--validation_seed", type=int, default=2027)
    parser.add_argument("--external_validation_root", default=None)
    parser.add_argument("--external_validation_dataset", default=None)
    parser.add_argument(
        "--validation_selection_metric",
        choices=("overall", "unseen_4_10"),
        default="overall",
    )
    parser.add_argument(
        "--checkpoint_selection",
        choices=("validation", "final"),
        default="validation",
    )
    parser.add_argument("--train_data_percentage", type=int, default=100)
    parser.add_argument("--test_data_percentage", type=int, default=100)
    parser.add_argument("--pair_feature_mode", default="product", choices=["product", "product_diff"])
    parser.add_argument("--mac_steps", type=int, default=4)
    parser.add_argument("--lambda_transition", type=float, default=0.0)
    parser.add_argument("--lambda_edge", type=float, default=0.0)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--metrics_out", required=True)
    return parser


def run_training() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    has_external_validation = bool(
        args.external_validation_root or args.external_validation_dataset
    )
    if bool(args.external_validation_root) != bool(args.external_validation_dataset):
        raise ValueError(
            "external validation requires both root and dataset arguments"
        )
    if has_external_validation and args.validation_fraction != 0.0:
        raise ValueError(
            "external validation cannot be combined with a training-set holdout"
        )
    if (
        args.checkpoint_selection == "validation"
        and args.validation_fraction == 0.0
        and not has_external_validation
    ):
        raise ValueError("validation checkpoint selection requires validation data")
    if has_external_validation and args.checkpoint_selection != "validation":
        raise ValueError("external validation requires validation checkpoint selection")
    if args.core_type == "self_attention_matched":
        if args.lambda_transition == 0.0 and args.lambda_edge == 0.0:
            raise ValueError("self_attention_matched requires a nonzero matched objective")
    elif args.lambda_transition != 0.0 or args.lambda_edge != 0.0:
        raise ValueError("Matched objectives require --core_type self_attention_matched")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer(args.model_type, model_name_or_path=args.model_name_or_path)

    full_train = TruaClutrrDataset(
        args.root,
        args.dataset,
        "train",
        args.train_data_percentage,
        tokenizer=tokenizer,
        augment=True,
    )
    full_train.data = [item for item in full_train.data if item is not None]
    validation_source = None
    if has_external_validation:
        train_data = full_train
        validation_data = TruaClutrrDataset(
            args.external_validation_root,
            args.external_validation_dataset,
            "test",
            100,
            tokenizer=tokenizer,
            augment=True,
            augment_seed=args.validation_seed,
        )
        validation_data.data = [
            item for item in validation_data.data if item is not None
        ]
        duplicates_removed = _deduplicate_dataset(validation_data)
        validation_source = {
            "type": "independent_generated_test_split",
            "root": args.external_validation_root,
            "dataset": args.external_validation_dataset,
            "duplicates_removed": duplicates_removed,
        }
    elif args.validation_fraction > 0.0:
        strata = [(item["hops"], item["target_id"]) for item in full_train.data]
        train_data, validation_data = stratified_train_validation_split(
            full_train,
            strata,
            validation_fraction=args.validation_fraction,
            seed=args.validation_seed,
        )
        validation_source = {
            "type": "stratified_training_holdout",
            "fraction": args.validation_fraction,
            "seed": args.validation_seed,
        }
    else:
        train_data = full_train
        validation_data = None
    test_data = TruaClutrrDataset(
        args.root,
        args.dataset,
        "test",
        args.test_data_percentage,
        tokenizer=tokenizer,
        augment=True,
        augment_seed=999,
    )
    test_data.data = [item for item in test_data.data if item is not None]

    collator = TruaBatchCollator(tokenizer, device, model_type=args.model_type)
    data_order_seed = args.seed + 271828
    data_order_generator = torch.Generator()
    data_order_generator.manual_seed(data_order_seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        generator=data_order_generator,
    )
    validation_loader = None
    if validation_data is not None:
        validation_loader = DataLoader(
            validation_data,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers,
        )
    test_loader = DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
    )

    model = ControlledClutrrBaseline(
        device,
        tokenizer,
        core_type=args.core_type,
        model_type=args.model_type,
        model_name_or_path=args.model_name_or_path,
        pair_feature_mode=args.pair_feature_mode,
        mac_steps=args.mac_steps,
        lambda_transition=args.lambda_transition,
        lambda_edge=args.lambda_edge,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    training_start = time.perf_counter()

    best_validation = None
    best_epoch = -1
    best_state = None
    validation_history = []
    optimization_seed = args.seed + 314159
    torch.manual_seed(optimization_seed)
    torch.cuda.manual_seed_all(optimization_seed)

    print(
        f"Controlled baseline: core={args.core_type}, seed={args.seed}, "
        "train/validation/test="
        f"{len(train_data)}/{len(validation_data) if validation_data is not None else 0}/"
        f"{len(test_data)}; checkpoint={args.checkpoint_selection}"
    )
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for batch in train_loader:
            if batch is None:
                continue
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            output["loss"].backward()
            optimizer.step()
            epoch_loss += float(output["loss"].item())
            batches += 1

        if validation_loader is not None:
            validation_metrics = evaluate_robustness(model, validation_loader, device)
            selection_score = _validation_score(
                validation_metrics, args.validation_selection_metric
            )
            validation_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": epoch_loss / max(batches, 1),
                    "selection_score": selection_score,
                    **validation_metrics,
                }
            )
            print(
                f"Epoch {epoch + 1:02d}: loss={epoch_loss / max(batches, 1):.4f}, "
                f"validation={validation_metrics['overall']:.4f}, "
                f"selection={selection_score:.4f} ({args.validation_selection_metric})"
            )
            if (
                args.checkpoint_selection == "validation"
                and (best_validation is None or selection_score > best_validation)
            ):
                best_validation = selection_score
                best_epoch = epoch + 1
                best_state = clone_model_state(model)
        else:
            print(f"Epoch {epoch + 1:02d}: loss={epoch_loss / max(batches, 1):.4f}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_start
    peak_memory_gib = (
        torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
    )

    if args.checkpoint_selection == "validation":
        if best_state is None:
            raise RuntimeError("No validation checkpoint was selected")
        restore_model_state(model, best_state)
    else:
        best_epoch = args.epochs
        if validation_history:
            best_validation = _validation_score(
                validation_history[-1], args.validation_selection_metric
            )
    selected_validation_overall = next(
        (
            record["overall"]
            for record in validation_history
            if record["epoch"] == best_epoch
        ),
        None,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_start = time.perf_counter()
    test_metrics = evaluate_robustness(model, test_loader, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_seconds = time.perf_counter() - test_start

    processed_training_examples = len(train_data) * args.epochs
    resource_usage = {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "peak_allocated_memory_gib": peak_memory_gib,
        "training_seconds": training_seconds,
        "test_seconds": test_seconds,
        "training_examples_per_second": processed_training_examples / max(training_seconds, 1e-9),
    }
    result = {
        "code_revision": repository_revision(),
        "protocol": "controlled_text_input_declared_checkpoint_v2",
        "implementation_scope": "adapted_protocol_control_not_official_reproduction",
        "dataset": args.dataset,
        "core_type": args.core_type,
        "model_type": args.model_type,
        "model_name_or_path": args.model_name_or_path,
        "seed": args.seed,
        "train_size": len(train_data),
        "validation_size": len(validation_data) if validation_data is not None else 0,
        "test_size": len(test_data),
        "validation_fraction": args.validation_fraction,
        "validation_seed": args.validation_seed,
        "validation_source": validation_source,
        "validation_selection_metric": args.validation_selection_metric,
        "checkpoint_selection": args.checkpoint_selection,
        "selected_epoch": best_epoch,
        "selected_validation_score": best_validation,
        "selected_validation_overall": selected_validation_overall,
        "validation_history": validation_history,
        "test": test_metrics,
        "resource_usage": resource_usage,
        "configuration": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "pair_feature_mode": args.pair_feature_mode,
            "mac_steps": args.mac_steps if args.core_type == "mac" else None,
            "renamed_view_training": True,
            "path_or_edge_supervision": args.lambda_transition != 0.0 or args.lambda_edge != 0.0,
            "lambda_transition": args.lambda_transition,
            "lambda_edge": args.lambda_edge,
            "data_order_seed": data_order_seed,
            "optimization_seed": optimization_seed,
        },
    }
    write_metrics(args.metrics_out, result)
    print(
        f"Selected {args.checkpoint_selection} epoch {best_epoch}; "
        f"test overall={test_metrics['overall']:.4f}, "
        f"short={test_metrics['short_hop']:.4f}, long={test_metrics['long_hop']:.4f}; "
        f"peak={peak_memory_gib:.2f} GiB, throughput={resource_usage['training_examples_per_second']:.2f} ex/s"
    )


if __name__ == "__main__":
    run_training()
