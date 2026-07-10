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

from clutrr.data.trua_collator import TruaBatchCollator
from clutrr.data.trua_dataset import TruaClutrrDataset
from clutrr.models.trua_model import TruaReasonerModel
from clutrr.utils.distributed import (
    is_distributed as _is_distributed,
    is_main_process as _is_main_process,
    setup_ddp,
)
from clutrr.training.robustness import evaluate_robustness
from clutrr.training.model_selection import (
    clone_model_state,
    restore_model_state,
    stratified_train_validation_split,
    write_metrics,
)
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
    "sparse_top_k": 0,
    "force_gold_edges": False,
    "use_qlora": False,
    "load_in_4bit": False,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "pooling": None,
    "entity_pooling": "mean",
    "prediction_head": "cls_pair",
    "consistency_mode": "kl",
    "use_relation_conditioning": True,
    "edge_supervision_target": "separate",
    "pair_feature_mode": "product",
    "use_path_algebra": False,
    "path_algebra_steps": 0,
    "residual_gate_init": -1.5,
    "lambda_gate": 0.0,
    "lambda_alg": 0.0,
    "lambda_eq": 0.0,
    "relation_score_mode": "mlp",
    "relation_rank": 64,
    "use_goal_guidance": True,
    "use_aggregation_branch": True,
    "use_step_branch": True,
    "validation_fraction": 0.1,
    "validation_seed": 2027,
    "metrics_out": None,
    "train_data_percentage": 100,
    "test_data_percentage": 100,
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
        description="Train TRUA model on CLUTRR only.",
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
            "gpt2",
            "llama3.2-1b",
            "llama3.2-3b",
            "qwen2.5-7b",
            "qwen3-0.6b",
            "qwen3-0.6b-base",
            "qwen3-1.7b",
            "qwen3-1.7b-base",
            "qwen3-8b",
            "qwen3-8b-base",
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
        "--train_data_percentage",
        type=int,
        default=defaults["train_data_percentage"],
        help="Percentage of the training CSV used; keep at 100 for reported experiments.",
    )
    parser.add_argument(
        "--test_data_percentage",
        type=int,
        default=defaults["test_data_percentage"],
        help="Percentage of each test CSV used; keep at 100 for reported experiments.",
    )
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
        "--validation_fraction",
        type=float,
        default=defaults["validation_fraction"],
        help="Fraction of the training set reserved for checkpoint selection.",
    )
    parser.add_argument(
        "--validation_seed",
        type=int,
        default=defaults["validation_seed"],
        help="Fixed seed for the train/validation partition; independent of the model seed.",
    )
    parser.add_argument(
        "--metrics_out",
        type=str,
        default=defaults["metrics_out"],
        help="Optional JSON path for validation history and the selected test result.",
    )
    parser.add_argument(
        "--sparse_top_k",
        type=int,
        default=defaults["sparse_top_k"],
        help="Top-k sparse candidate outgoing edges per entity; <=0 keeps all entities.",
    )
    parser.add_argument(
        "--force_gold_edges",
        action=argparse.BooleanOptionalAction,
        default=defaults["force_gold_edges"],
        help="During training, force gold trace edges into the sparse candidate set.",
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
    parser.add_argument(
        "--entity_pooling",
        type=str,
        default=defaults["entity_pooling"],
        choices=["mean", "multi_mention", "query_aware"],
        help="Entity representation mode before relation attention.",
    )
    parser.add_argument(
        "--prediction_head",
        type=str,
        default=defaults["prediction_head"],
        choices=["cls_pair", "cls_only", "pair_only", "gated"],
        help="Final prediction head used for diagnostics.",
    )
    parser.add_argument(
        "--consistency_mode",
        type=str,
        default=defaults["consistency_mode"],
        choices=["kl", "sym_kl", "js", "mse"],
        help="Consistency loss variant for renamed examples.",
    )
    parser.add_argument(
        "--use_relation_conditioning",
        action=argparse.BooleanOptionalAction,
        default=defaults["use_relation_conditioning"],
        help="Enable relation-conditioned FiLM and relation attention biases.",
    )
    parser.add_argument(
        "--use_goal_guidance",
        action=argparse.BooleanOptionalAction,
        default=defaults["use_goal_guidance"],
        help="Inject the adapter-provided query goal into the step-selection query.",
    )
    parser.add_argument(
        "--use_aggregation_branch",
        action=argparse.BooleanOptionalAction,
        default=defaults["use_aggregation_branch"],
        help="Enable relation-conditioned inter-unit aggregation.",
    )
    parser.add_argument(
        "--use_step_branch",
        action=argparse.BooleanOptionalAction,
        default=defaults["use_step_branch"],
        help="Enable query-guided step-selection messages.",
    )
    parser.add_argument(
        "--edge_supervision_target",
        type=str,
        default=defaults["edge_supervision_target"],
        choices=["separate", "latent"],
        help="Use separate edge classifier or supervise sparse latent relation logits directly.",
    )
    parser.add_argument(
        "--pair_feature_mode",
        type=str,
        default=defaults["pair_feature_mode"],
        choices=["product", "product_diff"],
        help="Feature set for directional subject-object and edge classifiers.",
    )
    parser.add_argument(
        "--use_path_algebra",
        action=argparse.BooleanOptionalAction,
        default=defaults["use_path_algebra"],
        help="Use sparse latent relation transition algebra logits as the main answer path.",
    )
    parser.add_argument(
        "--path_algebra_steps",
        type=int,
        default=defaults["path_algebra_steps"],
        help="Number of sparse relation DP steps; <=0 uses max_entities - 1.",
    )
    parser.add_argument(
        "--residual_gate_init",
        type=float,
        default=defaults["residual_gate_init"],
        help="Initial logit for alpha in z_path + alpha * z_text.",
    )
    parser.add_argument(
        "--lambda_gate",
        type=float,
        default=defaults["lambda_gate"],
        help="Small prior penalty on the residual text gate alpha.",
    )
    parser.add_argument(
        "--lambda_alg",
        type=float,
        default=defaults["lambda_alg"],
        help="Weight for identity/associativity/inverse algebra regularization.",
    )
    parser.add_argument(
        "--lambda_eq",
        type=float,
        default=defaults["lambda_eq"],
        help="Weight for renamed entity/relation/path equivariance regularization.",
    )
    parser.add_argument(
        "--relation_score_mode",
        type=str,
        default=defaults["relation_score_mode"],
        choices=["mlp", "bilinear"],
        help="Relation scorer: MLP or low-rank factorized bilinear.",
    )
    parser.add_argument(
        "--relation_rank",
        type=int,
        default=defaults["relation_rank"],
        help="Rank for low-rank bilinear relation scoring.",
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
    if args.strategy == "ddp":
        raise ValueError("Validation-selected training currently supports strategy=single or dp; run independent seeds per GPU.")

    set_seed(args.seed)
    if _is_main_process():
        print(f"Device: {device}")
        print(f"Selected Model: {args.model_type}")
        print(f"GPU Visible (CUDA_VISIBLE_DEVICES): {os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
        print(f"Strategy: {args.strategy} (rank={rank}, world_size={world_size}, local_rank={local_rank})")
        print(
            "Diagnostics: "
            f"entity_pooling={args.entity_pooling}, prediction_head={args.prediction_head}, "
            f"relation_conditioning={args.use_relation_conditioning}, "
            f"goal_guidance={args.use_goal_guidance}, aggregation_branch={args.use_aggregation_branch}, "
            f"step_branch={args.use_step_branch}, "
            f"edge_target={args.edge_supervision_target}, pair_features={args.pair_feature_mode}, "
            f"consistency={args.consistency_mode}, sparse_top_k={args.sparse_top_k}, "
            f"force_gold_edges={args.force_gold_edges}, path_algebra={args.use_path_algebra}, "
            f"lambda_gate={args.lambda_gate}, lambda_alg={args.lambda_alg}, lambda_eq={args.lambda_eq}, "
            f"relation_score={args.relation_score_mode}/r{args.relation_rank}"
        )

    tokenizer = build_tokenizer(args.model_type, model_name_or_path=args.model_name_or_path)

    root = args.root
    dset = args.dataset

    print("Loading Data (Augmentation Enabled)...")
    full_train_ds = TruaClutrrDataset(
        root,
        dset,
        "train",
        args.train_data_percentage,
        tokenizer=tokenizer,
        augment=True,
    )
    full_train_ds.data = [d for d in full_train_ds.data if d is not None]
    train_strata = [(item["hops"], item["target_id"]) for item in full_train_ds.data]
    train_ds, validation_ds = stratified_train_validation_split(
        full_train_ds,
        train_strata,
        validation_fraction=args.validation_fraction,
        seed=args.validation_seed,
    )
    print(
        f"Train/validation split: {len(train_ds)}/{len(validation_ds)} "
        f"(fraction={args.validation_fraction}, seed={args.validation_seed})"
    )

    test_ds = TruaClutrrDataset(
        root,
        dset,
        "test",
        args.test_data_percentage,
        tokenizer=tokenizer,
        augment=True,
        augment_seed=999,
    )
    test_ds.data = [d for d in test_ds.data if d is not None]

    collator = TruaBatchCollator(tokenizer, device, model_type=args.model_type)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
    )
    validation_loader = DataLoader(
        validation_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
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

    model = TruaReasonerModel(
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
        entity_pooling=args.entity_pooling,
        prediction_head=args.prediction_head,
        consistency_mode=args.consistency_mode,
        use_relation_conditioning=args.use_relation_conditioning,
        use_goal_guidance=args.use_goal_guidance,
        use_aggregation_branch=args.use_aggregation_branch,
        use_step_branch=args.use_step_branch,
        edge_supervision_target=args.edge_supervision_target,
        pair_feature_mode=args.pair_feature_mode,
        sparse_top_k=args.sparse_top_k,
        force_gold_edges=args.force_gold_edges,
        use_path_algebra=args.use_path_algebra,
        path_algebra_steps=args.path_algebra_steps,
        residual_gate_init=args.residual_gate_init,
        relation_score_mode=args.relation_score_mode,
        relation_rank=args.relation_rank,
    )
    if not getattr(model.encoder, "is_loaded_in_4bit", False):
        model = model.to(device)
    else:
        model.classifier = model.classifier.to(device)
        model.entity_attn = model.entity_attn.to(device)
        model.pair_classifier = model.pair_classifier.to(device)
        if hasattr(model, "rel_proj"):
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
    best_validation = -1.0
    best_epoch = -1
    best_state = None
    validation_history = []

    for epoch in range(args.epochs):
        if args.strategy == "ddp":
            train_sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0
        accum_aux = 0.0
        accum_edge = 0.0
        accum_cons = 0.0
        accum_alg = 0.0
        accum_eq = 0.0

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
                lambda_gate=args.lambda_gate,
                lambda_alg=args.lambda_alg,
                lambda_eq=args.lambda_eq,
            )

            loss = out["loss"]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            accum_aux += out["losses"]["nexthop"]
            accum_edge += out["losses"]["edge"]
            accum_cons += out["losses"]["cons"]
            accum_alg += out["losses"].get("alg", 0.0)
            accum_eq += out["losses"].get("eq", 0.0)
            pbar.set_postfix(
                {
                    "L_main": f"{out['losses']['main']:.3f}",
                    "L_aux": f"{out['losses']['nexthop']:.3f}",
                    "L_edge": f"{out['losses']['edge']:.3f}",
                    "L_cons": f"{out['losses']['cons']:.3f}",
                    "L_alg": f"{out['losses'].get('alg', 0.0):.3f}",
                    "L_eq": f"{out['losses'].get('eq', 0.0):.3f}",
                    "alpha": f"{out['losses'].get('residual_alpha', 0.0):.3f}",
                }
            )

        avg_loss = total_loss / len(train_loader)
        if _is_main_process():
            print(
                f"Epoch {epoch + 1} Done. Loss: {avg_loss:.4f} "
                f"(Aux: {accum_aux / len(train_loader):.4f}, "
                f"Edge: {accum_edge / len(train_loader):.4f}, "
                f"Cons: {accum_cons / len(train_loader):.4f}, "
                f"Alg: {accum_alg / len(train_loader):.4f}, "
                f"Eq: {accum_eq / len(train_loader):.4f})"
            )

        if _is_main_process():
            validation_metrics = evaluate_robustness(model, validation_loader, device)
            validation_history.append({"epoch": epoch + 1, **validation_metrics})
            print(f"  Validation Overall:  {validation_metrics['overall']:.4f}")
            if validation_metrics["overall"] > best_validation:
                best_validation = validation_metrics["overall"]
                best_epoch = epoch + 1
                best_state = clone_model_state(model)

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    restore_model_state(model, best_state)
    test_metrics = evaluate_robustness(model, test_loader, device)
    print(f"Selected Validation Epoch: {best_epoch} (overall={best_validation:.4f})")
    print("--> Test Evaluation (checkpoint selected on validation only)")
    print(f"  Overall Acc (Base):   {test_metrics['overall']:.4f}")
    print(f"  Renamed Acc (Mod):    {test_metrics['renamed']:.4f}")
    print(f"  Consistency:          {test_metrics['consistency']:.4f}")
    print(f"  Consistent & Correct: {test_metrics['consistent_and_correct']:.4f}")
    print(f"  Short Hop (2-3):      {test_metrics['short_hop']:.4f}")
    print(f"  Long Hop (>=6):       {test_metrics['long_hop']:.4f}")
    per_hop = test_metrics.get("per_hop", {})
    if per_hop:
        parts = []
        for hop in sorted(per_hop):
            item = per_hop[hop]
            parts.append(f"{hop}={item['accuracy']:.4f} ({item['correct']}/{item['total']})")
        print(f"  Per-Hop Acc:          {', '.join(parts)}")
    write_metrics(
        args.metrics_out,
        {
            "dataset": dset,
            "model_type": args.model_type,
            "seed": args.seed,
            "train_size": len(train_ds),
            "validation_size": len(validation_ds),
            "test_size": len(test_ds),
            "validation_fraction": args.validation_fraction,
            "validation_seed": args.validation_seed,
            "selected_epoch": best_epoch,
            "selected_validation_overall": best_validation,
            "validation_history": validation_history,
            "test": test_metrics,
            "configuration": {
                "lambda_nexthop": args.lambda_nexthop,
                "lambda_edge": args.lambda_edge,
                "lambda_consistency": args.lambda_consistency,
                "use_goal_guidance": args.use_goal_guidance,
                "use_aggregation_branch": args.use_aggregation_branch,
                "use_step_branch": args.use_step_branch,
                "use_relation_conditioning": args.use_relation_conditioning,
            },
        },
    )


if __name__ == "__main__":
    run_training()
