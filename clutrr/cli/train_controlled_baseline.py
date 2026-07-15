"""Train validation-selected controlled baselines under the final CLUTRR protocol."""

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
    strata = [(item["hops"], item["target_id"]) for item in full_train.data]
    train_data, validation_data = stratified_train_validation_split(
        full_train,
        strata,
        validation_fraction=args.validation_fraction,
        seed=args.validation_seed,
    )
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
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
    )
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

    best_validation = -1.0
    best_epoch = -1
    best_state = None
    validation_history = []

    print(
        f"Controlled baseline: core={args.core_type}, seed={args.seed}, "
        f"train/validation/test={len(train_data)}/{len(validation_data)}/{len(test_data)}"
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

        validation_metrics = evaluate_robustness(model, validation_loader, device)
        validation_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_loss / max(batches, 1),
                **validation_metrics,
            }
        )
        print(
            f"Epoch {epoch + 1:02d}: loss={epoch_loss / max(batches, 1):.4f}, "
            f"validation={validation_metrics['overall']:.4f}"
        )
        if validation_metrics["overall"] > best_validation:
            best_validation = validation_metrics["overall"]
            best_epoch = epoch + 1
            best_state = clone_model_state(model)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_start
    peak_memory_gib = (
        torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
    )

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    restore_model_state(model, best_state)
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
        "protocol": "controlled_text_input_final_v1",
        "implementation_scope": "adapted_protocol_control_not_official_reproduction",
        "dataset": args.dataset,
        "core_type": args.core_type,
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
        },
    }
    write_metrics(args.metrics_out, result)
    print(
        f"Selected epoch {best_epoch}; test overall={test_metrics['overall']:.4f}, "
        f"short={test_metrics['short_hop']:.4f}, long={test_metrics['long_hop']:.4f}; "
        f"peak={peak_memory_gib:.2f} GiB, throughput={resource_usage['training_examples_per_second']:.2f} ex/s"
    )


if __name__ == "__main__":
    run_training()
