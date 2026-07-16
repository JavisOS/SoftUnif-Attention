#!/usr/bin/env python3
"""Run the declared-checkpoint CLUTRR backbone/core comparison matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


DEFAULT_BACKBONES = (
    "bert:bert:/vepfs/trua_models/hf/bert-base-uncased",
    "roberta:roberta:/vepfs/trua_models/hf/roberta-base",
    "deberta-v3:deberta-v3:/vepfs/trua_models/hf/deberta-v3-base",
)
DEFAULT_CORES = ("encoder", "self_attention_matched", "trua")


def parse_backbone(specification: str) -> tuple[str, str, str]:
    try:
        name, model_type, model_path = specification.split(":", 2)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Backbones must use NAME:MODEL_TYPE:MODEL_PATH"
        ) from error
    if not name or not model_type or not model_path:
        raise argparse.ArgumentTypeError(
            "Backbones must use nonempty NAME:MODEL_TYPE:MODEL_PATH fields"
        )
    return name, model_type, model_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", default="data_089907f8")
    parser.add_argument(
        "--backbone",
        action="append",
        default=None,
        help="Repeat NAME:MODEL_TYPE:MODEL_PATH; defaults to BERT/RoBERTa/DeBERTa-v3.",
    )
    parser.add_argument(
        "--cores",
        nargs="+",
        choices=("encoder", "self_attention_matched", "trua"),
        default=list(DEFAULT_CORES),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 42])
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument(
        "--checkpoint-selection",
        choices=("validation", "final"),
        default="validation",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    parser.add_argument(
        "--external-validation-dataset",
        default="data_db9b8f04",
        help="Independent CLUTRR dataset whose deduplicated test split selects checkpoints.",
    )
    parser.add_argument(
        "--validation-selection-metric",
        choices=("overall", "unseen_4_10"),
        default="unseen_4_10",
    )
    parser.add_argument(
        "--trua-goal-representation",
        choices=("object", "endpoint_pair"),
        default="object",
    )
    parser.add_argument(
        "--reverse-query-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate every selected checkpoint on paired reversed queries.",
    )
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def command_for(
    *,
    repo_root: Path,
    core: str,
    model_type: str,
    model_path: str,
    data_root: str,
    dataset: str,
    seed: int,
    gpu: int,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    checkpoint_selection: str,
    validation_fraction: float,
    external_validation_dataset: str | None,
    validation_selection_metric: str,
    trua_goal_representation: str,
    reverse_query_eval: bool,
    metrics_path: Path,
) -> list[str]:
    common = [
        "--dataset",
        dataset,
        "--root",
        data_root,
        "--model_type",
        model_type,
        "--model_name_or_path",
        model_path,
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--eval_batch_size",
        str(eval_batch_size),
        "--seed",
        str(seed),
        "--validation_fraction",
        str(validation_fraction),
        "--validation_seed",
        "2027",
        "--checkpoint_selection",
        checkpoint_selection,
        "--validation_selection_metric",
        validation_selection_metric,
        "--gpus",
        str(gpu),
        "--metrics_out",
        str(metrics_path),
    ]
    if external_validation_dataset:
        common.extend(
            [
                "--external_validation_root",
                data_root,
                "--external_validation_dataset",
                external_validation_dataset,
            ]
        )
    if reverse_query_eval:
        common.append("--reverse_query_eval")
    if core == "trua":
        return [
            sys.executable,
            "-u",
            "-m",
            "clutrr.cli.train",
            "--config",
            str(repo_root / "configs" / "clutrr" / "train_trua.yaml"),
            *common,
            "--lambda_nexthop",
            "1.0",
            "--lambda_edge",
            "1.0",
            "--lambda_consistency",
            "0.0",
            "--goal_representation",
            trua_goal_representation,
        ]

    command = [
        sys.executable,
        "-u",
        "-m",
        "clutrr.cli.train_controlled_baseline",
        "--core_type",
        core,
        *common,
    ]
    if core == "self_attention_matched":
        command.extend(
            ["--lambda_transition", "1.0", "--lambda_edge", "1.0"]
        )
    return command


def valid_result(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(result.get("test"), dict) and "overall" in result["test"]


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.external_validation_dataset and args.validation_fraction != 0.0:
        raise ValueError(
            "Use either external validation or a training-set holdout, not both"
        )
    if args.checkpoint_selection == "final" and args.external_validation_dataset:
        raise ValueError(
            "Fixed-final runs must disable --external-validation-dataset"
        )
    if (
        args.checkpoint_selection == "validation"
        and args.validation_fraction == 0.0
        and not args.external_validation_dataset
    ):
        raise ValueError("Validation selection requires validation data")
    device_count = torch.cuda.device_count()
    invalid_gpus = [gpu for gpu in args.gpus if gpu < 0 or gpu >= device_count]
    if invalid_gpus:
        raise ValueError(
            f"Requested unavailable GPUs {invalid_gpus}; visible indices are "
            f"0--{max(device_count - 1, 0)}"
        )
    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.external_validation_dataset:
        validation_path = Path(args.data_root) / args.external_validation_dataset
        if not validation_path.exists():
            raise FileNotFoundError(
                f"Missing external validation dataset: {validation_path}"
            )
    backbones = [
        parse_backbone(specification)
        for specification in (args.backbone or DEFAULT_BACKBONES)
    ]

    jobs = []
    for backbone_name, model_type, model_path in backbones:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Missing backbone path: {model_path}")
        for core in args.cores:
            for seed in args.seeds:
                run_dir = output_root / backbone_name / core / f"seed_{seed}"
                metrics_path = run_dir / "metrics.json"
                jobs.append(
                    {
                        "backbone": backbone_name,
                        "model_type": model_type,
                        "model_path": model_path,
                        "core": core,
                        "seed": seed,
                        "run_dir": run_dir,
                        "metrics_path": metrics_path,
                    }
                )

    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "clutrr_backbone_core_matrix_declared_checkpoint",
        "dataset": args.dataset,
        "backbones": [
            {"name": name, "model_type": model_type, "model_path": model_path}
            for name, model_type, model_path in backbones
        ],
        "cores": args.cores,
        "seeds": args.seeds,
        "gpus": args.gpus,
        "protocol": {
            "epochs": args.epochs,
            "training_hops": [2, 3],
            "test_hops": list(range(2, 11)),
            "validation_fraction": args.validation_fraction,
            "validation_seed": 2027,
            "validation_source": {
                "type": "independent_generated_test_split",
                "dataset": args.external_validation_dataset,
                "deduplicate_exact_examples": True,
                "formal_split_overlap": 0,
            },
            "validation_selection_metric": args.validation_selection_metric,
            "checkpoint_selection": args.checkpoint_selection,
            "test_evaluations_per_run": 1,
            "randomness": {
                "model_initialization_seed": "reported seed",
                "data_order_seed": "reported seed + 271828",
                "optimization_seed": "reported seed + 314159",
            },
            "entity_unit_grounding": (
                "mean the aligned subwords within the first textual "
                "occurrence of each entity"
            ),
            "matched_unit_objectives": {
                "self_attention_matched": ["transition", "edge"],
                "trua": ["transition", "edge"],
            },
            "trua_goal_representation": args.trua_goal_representation,
            "reverse_query_evaluation": args.reverse_query_eval,
        },
        "jobs": [],
    }
    for job in jobs:
        command = command_for(
            repo_root=repo_root,
            core=job["core"],
            model_type=job["model_type"],
            model_path=job["model_path"],
            data_root=args.data_root,
            dataset=args.dataset,
            seed=job["seed"],
            gpu=args.gpus[0],
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            checkpoint_selection=args.checkpoint_selection,
            validation_fraction=args.validation_fraction,
            external_validation_dataset=args.external_validation_dataset,
            validation_selection_metric=args.validation_selection_metric,
            trua_goal_representation=args.trua_goal_representation,
            reverse_query_eval=args.reverse_query_eval,
            metrics_path=job["metrics_path"],
        )
        manifest["jobs"].append(
            {
                "backbone": job["backbone"],
                "core": job["core"],
                "seed": job["seed"],
                "metrics": str(job["metrics_path"]),
                "command_template": command,
                "status": "complete" if valid_result(job["metrics_path"]) else "pending",
            }
        )
    write_json(output_root / "manifest.json", manifest)

    pending = [job for job in jobs if not valid_result(job["metrics_path"])]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "jobs_total": len(jobs),
                    "jobs_complete": len(jobs) - len(pending),
                    "jobs_pending": len(pending),
                },
                indent=2,
            )
        )
        return

    active: dict[int, dict] = {}
    failures = []
    while pending or active:
        for gpu in args.gpus:
            if gpu in active or not pending:
                continue
            job = pending.pop(0)
            job["run_dir"].mkdir(parents=True, exist_ok=True)
            log_path = job["run_dir"] / "train.log"
            command = command_for(
                repo_root=repo_root,
                core=job["core"],
                model_type=job["model_type"],
                model_path=job["model_path"],
                data_root=args.data_root,
                dataset=args.dataset,
                seed=job["seed"],
                gpu=gpu,
                epochs=args.epochs,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                checkpoint_selection=args.checkpoint_selection,
                validation_fraction=args.validation_fraction,
                external_validation_dataset=args.external_validation_dataset,
                validation_selection_metric=args.validation_selection_metric,
                trua_goal_representation=args.trua_goal_representation,
                reverse_query_eval=args.reverse_query_eval,
                metrics_path=job["metrics_path"],
            )
            log_handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=repo_root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            active[gpu] = {
                **job,
                "process": process,
                "log_handle": log_handle,
                "log_path": log_path,
                "command": command,
            }
            print(
                f"launched gpu={gpu} backbone={job['backbone']} "
                f"core={job['core']} seed={job['seed']} pid={process.pid}",
                flush=True,
            )

        time.sleep(args.poll_seconds)
        for gpu, job in list(active.items()):
            return_code = job["process"].poll()
            if return_code is None:
                continue
            job["log_handle"].close()
            complete = return_code == 0 and valid_result(job["metrics_path"])
            print(
                f"finished gpu={gpu} backbone={job['backbone']} "
                f"core={job['core']} seed={job['seed']} rc={return_code} "
                f"complete={complete}",
                flush=True,
            )
            if not complete:
                failures.append(
                    {
                        "backbone": job["backbone"],
                        "core": job["core"],
                        "seed": job["seed"],
                        "return_code": return_code,
                        "log": str(job["log_path"]),
                    }
                )
            del active[gpu]

        status = {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "pending": len(pending),
            "active": [
                {
                    "gpu": gpu,
                    "backbone": job["backbone"],
                    "core": job["core"],
                    "seed": job["seed"],
                    "pid": job["process"].pid,
                }
                for gpu, job in active.items()
            ],
            "failures": failures,
        }
        write_json(output_root / "status.json", status)

    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["failures"] = failures
    for record in manifest["jobs"]:
        record["status"] = (
            "complete" if valid_result(Path(record["metrics"])) else "failed"
        )
    write_json(output_root / "manifest.json", manifest)
    if failures:
        raise RuntimeError(f"{len(failures)} CLUTRR matrix jobs failed")


if __name__ == "__main__":
    main()
