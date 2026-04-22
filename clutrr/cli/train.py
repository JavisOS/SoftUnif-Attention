import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from clutrr.data.tsra_collator import TsraBatchCollator
from clutrr.data.tsra_dataset import TsraClutrrDataset
from clutrr.models.tsra_model import TsraReasonerModel
from clutrr.utils.distributed import (
    is_distributed as _is_distributed,
    is_main_process as _is_main_process,
    setup_ddp,
)
from clutrr.training.robustness import evaluate_robustness
from clutrr.models.backbones import build_tokenizer
from clutrr.config.defaults import DEFAULT_CLUTRR_DATASET, DEFAULT_CLUTRR_ROOT
from clutrr.utils.seed import set_seed


BASE_TRAIN_DEFAULTS = {
    "model_type": "deberta",
    "model_name_or_path": None,
    "epochs": 10,
    "lr": 2e-5,
    "batch_size": 16,
    "eval_batch_size": 32,
    "num_workers": 0,
    "root": DEFAULT_CLUTRR_ROOT,
    "dataset": DEFAULT_CLUTRR_DATASET,
    "gpus": "0",
    "strategy": "single",
    "seed": 42,
    "lambda_nexthop": 1.0,
    "lambda_edge": 1.0,
    "lambda_consistency": 5.0,
    "use_qlora": False,
    "load_in_4bit": False,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "pooling": None,
}


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


def _make_train_pbar(iterable, desc: str):
    # Avoid corrupted multi-line bars in non-TTY logs and on terminal resize.
    is_tty = sys.stderr.isatty()
    return tqdm(
        iterable,
        desc=desc,
        disable=not is_tty,
        dynamic_ncols=False,
        ncols=100,
        leave=False,
    )


def _extract_config_path(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None)
    args, _ = parser.parse_known_args(argv)
    return args.config


def _load_yaml_config(config_path: str | None) -> dict:
    if not config_path:
        return {}

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping/dict: {config_path}")
    return data


def build_arg_parser(defaults=None):
    defaults = defaults or BASE_TRAIN_DEFAULTS
    parser = argparse.ArgumentParser(
        prog="python -m clutrr.cli.train",
        description="Train TSRA model on CLUTRR only.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config path. CLI args override YAML values.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=defaults["model_type"],
        choices=[
            "roberta",
            "roberta-large",
            "deberta",
            "deberta-v3",
            "deberta-v3-large",
            "bert",
            "modernbert",
            "qwen3-8b",
        ],
        help="Model backbone type",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=defaults["model_name_or_path"],
        help="Optional local/remote model path overriding the built-in model id.",
    )
    parser.add_argument("--epochs", type=int, default=defaults["epochs"], help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=defaults["lr"], help="Learning rate")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=defaults["batch_size"],
        help="Train batch size (per process for DDP)",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=defaults["eval_batch_size"],
        help="Eval batch size (per process for DDP; eval runs on rank0 only)",
    )
    parser.add_argument("--num_workers", type=int, default=defaults["num_workers"], help="DataLoader num_workers")
    parser.add_argument("--root", type=str, default=defaults["root"], help="Data root directory")
    parser.add_argument("--dataset", type=str, default=defaults["dataset"], help="Dataset folder name")
    parser.add_argument(
        "--gpus",
        type=str,
        default=defaults["gpus"],
        help="选择可见 GPU，例如 '0' 或 '0,1,2,3'。脚本会设置 CUDA_VISIBLE_DEVICES。",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=defaults["strategy"],
        choices=["single", "dp", "ddp"],
        help="single=单卡；dp=DataParallel；ddp=DistributedDataParallel(推荐，多进程同步)",
    )
    parser.add_argument("--seed", type=int, default=defaults["seed"], help="Random seed")
    parser.add_argument(
        "--lambda_nexthop",
        type=float,
        default=defaults["lambda_nexthop"],
        help="Weight for next-hop supervision loss",
    )
    parser.add_argument(
        "--lambda_edge",
        type=float,
        default=defaults["lambda_edge"],
        help="Weight for edge-relation supervision loss",
    )
    parser.add_argument(
        "--lambda_consistency",
        type=float,
        default=defaults["lambda_consistency"],
        help="Weight for consistency regularization loss",
    )
    parser.add_argument(
        "--use_qlora",
        action=argparse.BooleanOptionalAction,
        default=defaults["use_qlora"],
        help="Enable LoRA adapters on decoder-only backbones.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action=argparse.BooleanOptionalAction,
        default=defaults["load_in_4bit"],
        help="Load backbone in 4-bit quantization (recommended for QLoRA).",
    )
    parser.add_argument("--lora_r", type=int, default=defaults["lora_r"], help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=defaults["lora_alpha"], help="LoRA alpha.")
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=defaults["lora_dropout"],
        help="LoRA dropout.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default=defaults["pooling"],
        choices=["cls", "last_token", None],
        help="Sequence pooling strategy for classifier head. Default picks model-specific strategy.",
    )
    return parser


def parse_training_args():
    config_path = _extract_config_path()
    defaults = dict(BASE_TRAIN_DEFAULTS)

    yaml_config = _load_yaml_config(config_path)
    unknown_keys = sorted(set(yaml_config.keys()) - set(defaults.keys()))
    if unknown_keys:
        raise ValueError(f"Unknown config keys in {config_path}: {', '.join(unknown_keys)}")
    defaults.update(yaml_config)

    parser = build_arg_parser(defaults=defaults)
    return parser.parse_args()


def run_training():
    args = parse_training_args()

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)

    if args.strategy == "ddp":
        rank, world_size, local_rank = setup_ddp(
            strict=True,
            launch_hint="torchrun --standalone --nproc_per_node=4 -m clutrr.cli.train --strategy ddp --gpus 0,1,2,3 ...",
        )
        device = torch.device("cuda", local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_qlora and args.strategy != "single":
        raise ValueError("QLoRA currently supports strategy=single only in this training script.")

    set_seed(args.seed)
    if _is_main_process():
        print(f"Device: {device}")
        print(f"Selected Model: {args.model_type}")
        print(f"GPU Visible (CUDA_VISIBLE_DEVICES): {os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
        print(f"Strategy: {args.strategy} (rank={rank}, world_size={world_size}, local_rank={local_rank})")

    tokenizer = build_tokenizer(args.model_type, model_name_or_path=args.model_name_or_path)

    root = args.root
    dset = args.dataset

    print("Loading Data (Augmentation Enabled)...")
    train_ds = TsraClutrrDataset(root, dset, "train", 100, tokenizer=tokenizer, augment=True)
    train_ds.data = [d for d in train_ds.data if d is not None]

    test_ds = TsraClutrrDataset(
        root,
        dset,
        "test",
        100,
        tokenizer=tokenizer,
        augment=True,
        augment_seed=999,
    )
    test_ds.data = [d for d in test_ds.data if d is not None]

    collator = TsraBatchCollator(tokenizer, device, model_type=args.model_type)

    if args.strategy == "ddp":
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=train_sampler,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=args.num_workers,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers,
        )

    model = TsraReasonerModel(
        device,
        tokenizer,
        model_type=args.model_type,
        model_name_or_path=args.model_name_or_path,
        use_qlora=args.use_qlora,
        load_in_4bit=args.load_in_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        pooling=args.pooling,
    )
    if not getattr(model.encoder, "is_loaded_in_4bit", False):
        model = model.to(device)
    else:
        model.classifier = model.classifier.to(device)
        model.entity_attn = model.entity_attn.to(device)
        model.pair_classifier = model.pair_classifier.to(device)
        model.rel_proj = model.rel_proj.to(device)

    if args.strategy == "dp":
        if not torch.cuda.is_available():
            raise RuntimeError("dp 需要 CUDA 可用")
        n = torch.cuda.device_count()
        if n <= 1:
            if _is_main_process():
                print("[Warn] dp 但当前可见 GPU <= 1，将退化为单卡")
        else:
            model = nn.DataParallel(model, device_ids=list(range(n)))
    elif args.strategy == "ddp":
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print("Starting Training...")

    for epoch in range(args.epochs):
        if args.strategy == "ddp":
            train_sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0
        accum_aux = 0.0
        accum_edge = 0.0
        accum_cons = 0.0

        pbar = _make_train_pbar(train_loader, desc=f"Ep {epoch + 1}")
        for batch in pbar:
            if batch is None:
                continue

            batch_for_model = _move_batch_to_device(batch, device=device, skip_keys={"raw_batch"})
            if "raw_batch" in batch_for_model:
                batch_for_model.pop("raw_batch")

            optimizer.zero_grad()
            out = model(
                batch_for_model,
                lambda1=args.lambda_nexthop,
                lambda_edge=args.lambda_edge,
                lambda_cons=args.lambda_consistency,
            )

            loss = out["loss"]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            accum_aux += out["losses"]["nexthop"]
            accum_edge += out["losses"]["edge"]
            accum_cons += out["losses"]["cons"]
            pbar.set_postfix(
                {
                    "L_main": f"{out['losses']['main']:.3f}",
                    "L_aux": f"{out['losses']['nexthop']:.3f}",
                    "L_edge": f"{out['losses']['edge']:.3f}",
                    "L_cons": f"{out['losses']['cons']:.3f}",
                }
            )

        avg_loss = total_loss / len(train_loader)
        if _is_main_process():
            print(
                f"Epoch {epoch + 1} Done. Loss: {avg_loss:.4f} "
                f"(Aux: {accum_aux / len(train_loader):.4f}, "
                f"Edge: {accum_edge / len(train_loader):.4f}, "
                f"Cons: {accum_cons / len(train_loader):.4f})"
            )

        if _is_main_process():
            print(f"--> Evaluating Robustness Epoch {epoch + 1}...")
            metrics = evaluate_robustness(model, test_loader, device)

            print(f"  Overall Acc (Base):   {metrics['overall']:.4f}")
            print(f"  Renamed Acc (Mod):    {metrics['renamed']:.4f}")
            print(f"  Consistency:          {metrics['consistency']:.4f}")
            print(f"  Consistent & Correct: {metrics['consistent_and_correct']:.4f}")
            print(f"  Short Hop (2-3):      {metrics['short_hop']:.4f}")
            print(f"  Long Hop (>=6):       {metrics['long_hop']:.4f}")

    if args.strategy == "ddp" and _is_distributed():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    run_training()
