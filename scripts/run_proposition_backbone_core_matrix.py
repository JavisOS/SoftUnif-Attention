#!/usr/bin/env python3
"""Run matched proposition-core comparisons across tasks and backbones."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


DEFAULT_BACKBONES = (
    "bert:/vepfs/trua_models/hf/bert-base-uncased",
    "roberta:/vepfs/trua_models/hf/roberta-base",
    "deberta-v3:/vepfs/trua_models/hf/deberta-v3-base",
)
DEFAULT_CORES = ("encoder", "self_attention", "trua")

DATASETS = {
    "pfolio": {
        "dataset": "pfolio",
        "root": "/root/TRUA/data/pfolio",
        "extra": ("--max-query-len", "96"),
        "max_sentences": 32,
        "labels": ("false", "true", "unknown"),
    },
    "proofwriter": {
        "dataset": "proofwriter",
        "root": "/root/TRUA/data/proofwriter/raw/proofwriter-dataset-V2020.12.3",
        "extra": ("--train-depths", "0,1,2", "--test-depths", "3,5"),
        "max_sentences": 32,
        "labels": ("false", "true", "unknown"),
    },
    "ruletaker": {
        "dataset": "ruletaker_raw",
        "root": "/root/TRUA/data/rule-reasoning-dataset-V2020.2.5.0/original",
        "extra": (
            "--train-depths",
            "1,2",
            "--test-depths",
            "1,2,3,5",
            "--train-qdeps",
            "1,2",
            "--test-qdeps",
            "1,2,3,4,5",
        ),
        "max_sentences": 32,
        "labels": ("false", "true"),
    },
}


def parse_backbone(specification: str) -> tuple[str, str]:
    try:
        name, model_path = specification.split(":", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Backbones must use NAME:MODEL_PATH"
        ) from error
    if not name or not model_path:
        raise argparse.ArgumentTypeError(
            "Backbones must use nonempty NAME:MODEL_PATH fields"
        )
    return name, model_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--backbone",
        action="append",
        default=None,
        help="Repeat NAME:MODEL_PATH; defaults to BERT/RoBERTa/DeBERTa-v3.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=sorted(DATASETS),
    )
    parser.add_argument(
        "--cores",
        nargs="+",
        choices=DEFAULT_CORES,
        default=list(DEFAULT_CORES),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 42])
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--limit-train",
        type=int,
        default=0,
        help="Maximum training examples per task; 0 uses the complete split.",
    )
    parser.add_argument(
        "--limit-test",
        type=int,
        default=0,
        help="Maximum development/test examples per split; 0 uses complete splits.",
    )
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def command_for(
    *,
    repo_root: Path,
    dataset_name: str,
    core: str,
    model_path: str,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    limit_train: int,
    limit_test: int,
    metrics_path: Path,
) -> list[str]:
    specification = DATASETS[dataset_name]
    lambda_evidence = 0.0 if core == "encoder" else 1.0
    command = [
        sys.executable,
        "-u",
        str(repo_root / "scripts" / "transformer_trua_prop.py"),
        "--dataset",
        specification["dataset"],
        "--root",
        specification["root"],
        "--model-name",
        model_path,
        "--core-type",
        core,
        "--lambda-evidence",
        str(lambda_evidence),
        "--limit-train",
        str(limit_train),
        "--limit-test",
        str(limit_test),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(learning_rate),
        "--max-sents",
        str(specification["max_sentences"]),
        "--max-len",
        "512",
        "--validation-fraction",
        "0.1",
        "--validation-seed",
        "2027",
        "--seed",
        str(seed),
        "--out",
        str(metrics_path),
        *specification["extra"],
    ]
    if core == "self_attention":
        command.extend(["--no-use-goal-guidance", "--no-use-query-anchor"])
    elif core == "trua":
        command.extend(["--use-goal-guidance", "--use-query-anchor"])
    return command


def valid_result(path: Path, expected_labels: tuple[str, ...]) -> bool:
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    tests = result.get("results")
    labels = result.get("task_labels", {}).get("id_to_name", {})
    observed_labels = tuple(
        labels.get(str(index)) for index in range(len(expected_labels))
    )
    return (
        isinstance(tests, dict)
        and bool(tests)
        and observed_labels == expected_labels
    )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.limit_train < 0 or args.limit_test < 0:
        raise ValueError("Data limits must be nonnegative; use 0 for full splits")
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
    backbones = [
        parse_backbone(specification)
        for specification in (args.backbone or DEFAULT_BACKBONES)
    ]
    for _, model_path in backbones:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Missing backbone path: {model_path}")
    for dataset_name in args.datasets:
        if not Path(DATASETS[dataset_name]["root"]).exists():
            raise FileNotFoundError(
                f"Missing {dataset_name} root: {DATASETS[dataset_name]['root']}"
            )

    jobs = []
    for dataset_name in args.datasets:
        for backbone_name, model_path in backbones:
            for core in args.cores:
                for seed in args.seeds:
                    run_dir = (
                        output_root
                        / dataset_name
                        / backbone_name
                        / core
                        / f"seed_{seed}"
                    )
                    jobs.append(
                        {
                            "dataset": dataset_name,
                            "backbone": backbone_name,
                            "model_path": model_path,
                            "core": core,
                            "seed": seed,
                            "run_dir": run_dir,
                            "metrics_path": run_dir / "metrics.json",
                        }
                    )

    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "proposition_backbone_core_matrix_validation_selected",
        "datasets": args.datasets,
        "backbones": [
            {"name": name, "model_path": model_path}
            for name, model_path in backbones
        ],
        "cores": args.cores,
        "seeds": args.seeds,
        "gpus": args.gpus,
        "protocol": {
            "epochs": args.epochs,
            "data_limits": {
                "train": args.limit_train,
                "development_and_each_test": args.limit_test,
                "zero_means_complete_split": True,
            },
            "checkpoint_data": "dataset-provided development split",
            "checkpoint_selection": "validation accuracy; Evidence@1 tie-break",
            "test_evaluations_per_run": 1,
            "randomness": {
                "model_initialization_seed": "reported seed",
                "data_order_seed": "reported seed + 271828",
                "optimization_seed": "reported seed + 314159",
            },
            "evidence_regularization": {
                "encoder": 0.0,
                "self_attention": 1.0,
                "trua": 1.0,
            },
            "selection_query_routes": {
                "encoder": [],
                "self_attention": [],
                "trua": ["query_anchor", "explicit_goal_term"],
            },
            "task_labels": {
                dataset_name: list(DATASETS[dataset_name]["labels"])
                for dataset_name in args.datasets
            },
        },
        "jobs": [],
    }
    for job in jobs:
        manifest["jobs"].append(
            {
                "dataset": job["dataset"],
                "backbone": job["backbone"],
                "core": job["core"],
                "seed": job["seed"],
                "metrics": str(job["metrics_path"]),
                "status": (
                    "complete"
                    if valid_result(
                        job["metrics_path"],
                        DATASETS[job["dataset"]]["labels"],
                    )
                    else "pending"
                ),
            }
        )
    write_json(output_root / "manifest.json", manifest)

    pending = [
        job
        for job in jobs
        if not valid_result(
            job["metrics_path"], DATASETS[job["dataset"]]["labels"]
        )
    ]
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
                dataset_name=job["dataset"],
                core=job["core"],
                model_path=job["model_path"],
                seed=job["seed"],
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                limit_train=args.limit_train,
                limit_test=args.limit_test,
                metrics_path=job["metrics_path"],
            )
            log_handle = log_path.open("w", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment["HF_HUB_OFFLINE"] = "1"
            environment["TRANSFORMERS_OFFLINE"] = "1"
            environment["TOKENIZERS_PARALLELISM"] = "false"
            process = subprocess.Popen(
                command,
                cwd=repo_root,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            active[gpu] = {
                **job,
                "process": process,
                "log_handle": log_handle,
                "log_path": log_path,
            }
            print(
                f"launched gpu={gpu} dataset={job['dataset']} "
                f"backbone={job['backbone']} core={job['core']} "
                f"seed={job['seed']} pid={process.pid}",
                flush=True,
            )

        time.sleep(args.poll_seconds)
        for gpu, job in list(active.items()):
            return_code = job["process"].poll()
            if return_code is None:
                continue
            job["log_handle"].close()
            complete = return_code == 0 and valid_result(
                job["metrics_path"], DATASETS[job["dataset"]]["labels"]
            )
            print(
                f"finished gpu={gpu} dataset={job['dataset']} "
                f"backbone={job['backbone']} core={job['core']} "
                f"seed={job['seed']} rc={return_code} complete={complete}",
                flush=True,
            )
            if not complete:
                failures.append(
                    {
                        "dataset": job["dataset"],
                        "backbone": job["backbone"],
                        "core": job["core"],
                        "seed": job["seed"],
                        "return_code": return_code,
                        "log": str(job["log_path"]),
                    }
                )
            del active[gpu]

        write_json(
            output_root / "status.json",
            {
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "pending": len(pending),
                "active": [
                    {
                        "gpu": gpu,
                        "dataset": job["dataset"],
                        "backbone": job["backbone"],
                        "core": job["core"],
                        "seed": job["seed"],
                        "pid": job["process"].pid,
                    }
                    for gpu, job in active.items()
                ],
                "failures": failures,
            },
        )

    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["failures"] = failures
    for record in manifest["jobs"]:
        record["status"] = (
            "complete"
            if valid_result(
                Path(record["metrics"]), DATASETS[record["dataset"]]["labels"]
            )
            else "failed"
        )
    write_json(output_root / "manifest.json", manifest)
    if failures:
        raise RuntimeError(f"{len(failures)} proposition matrix jobs failed")


if __name__ == "__main__":
    main()
