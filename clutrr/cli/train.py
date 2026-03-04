import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
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


def build_arg_parser():
    parser = argparse.ArgumentParser(
        prog="python -m clutrr.cli.train",
        description="Train TSRA model on CLUTRR only.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="deberta",
        choices=["roberta", "roberta-large", "deberta", "deberta-v3", "deberta-v3-large", "bert", "modernbert"],
        help="Model backbone type",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Train batch size (per process for DDP)")
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Eval batch size (per process for DDP; eval runs on rank0 only)",
    )
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument("--root", type=str, default=DEFAULT_CLUTRR_ROOT, help="Data root directory")
    parser.add_argument("--dataset", type=str, default=DEFAULT_CLUTRR_DATASET, help="Dataset folder name")
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="选择可见 GPU，例如 '0' 或 '0,1,2,3'。脚本会设置 CUDA_VISIBLE_DEVICES。",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="single",
        choices=["single", "dp", "ddp"],
        help="single=单卡；dp=DataParallel；ddp=DistributedDataParallel(推荐，多进程同步)",
    )
    return parser


def run_training():
    args = build_arg_parser().parse_args()

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    if args.strategy == "ddp":
        rank, world_size, local_rank = setup_ddp(
            strict=True,
            launch_hint="torchrun --standalone --nproc_per_node=4 -m clutrr.cli.train --strategy ddp --gpus 0,1,2,3 ...",
        )
        device = torch.device("cuda", local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(42)
    if _is_main_process():
        print(f"Device: {device}")
        print(f"Selected Model: {args.model_type}")
        print(f"GPU Visible (CUDA_VISIBLE_DEVICES): {os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
        print(f"Strategy: {args.strategy} (rank={rank}, world_size={world_size}, local_rank={local_rank})")

    tokenizer = build_tokenizer(args.model_type)

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

    collator = TsraBatchCollator(tokenizer, device)

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

    model = TsraReasonerModel(device, tokenizer, model_type=args.model_type).to(device)

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
    print("Starting Training (Scheme A + B)...")

    for epoch in range(args.epochs):
        if args.strategy == "ddp":
            train_sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0
        accum_aux = 0.0
        accum_cons = 0.0

        pbar = tqdm(train_loader, desc=f"Ep {epoch + 1}")
        for batch in pbar:
            if batch is None:
                continue

            batch_for_model = dict(batch)
            if "raw_batch" in batch_for_model:
                batch_for_model.pop("raw_batch")

            optimizer.zero_grad()
            out = model(batch_for_model, lambda1=1.0, lambda_cons=5.0)

            loss = out["loss"]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            accum_aux += out["losses"]["nexthop"]
            accum_cons += out["losses"]["cons"]
            pbar.set_postfix(
                {
                    "L_main": f"{out['losses']['main']:.3f}",
                    "L_aux": f"{out['losses']['nexthop']:.3f}",
                    "L_cons": f"{out['losses']['cons']:.3f}",
                }
            )

        avg_loss = total_loss / len(train_loader)
        if _is_main_process():
            print(
                f"Epoch {epoch + 1} Done. Loss: {avg_loss:.4f} "
                f"(Aux: {accum_aux / len(train_loader):.4f}, Cons: {accum_cons / len(train_loader):.4f})"
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
