#!/usr/bin/env python3
"""Audit the protocol factors behind historical and current CLUTRR scores."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


VARIANTS = {
    "full_final_cons5": {
        "validation_fraction": 0.0,
        "checkpoint_selection": "final",
        "lambda_consistency": 5.0,
    },
    "full_final_cons0": {
        "validation_fraction": 0.0,
        "checkpoint_selection": "final",
        "lambda_consistency": 0.0,
    },
    "heldout_final_cons0": {
        "validation_fraction": 0.1,
        "checkpoint_selection": "final",
        "lambda_consistency": 0.0,
    },
    "heldout_validation_cons0": {
        "validation_fraction": 0.1,
        "checkpoint_selection": "validation",
        "lambda_consistency": 0.0,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data-root", default="/root/TRUA/data")
    parser.add_argument("--dataset", default="data_089907f8")
    parser.add_argument(
        "--model-path",
        default="/vepfs/trua_models/hf/deberta-v3-base",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 42])
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def valid_result(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload.get("test"), dict) and "overall" in payload["test"]


def command_for(
    repo_root: Path,
    args: argparse.Namespace,
    variant: dict,
    seed: int,
    gpu: int,
    metrics_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "clutrr.cli.train",
        "--config",
        str(repo_root / "configs" / "clutrr" / "train_trua.yaml"),
        "--dataset",
        args.dataset,
        "--root",
        args.data_root,
        "--model_type",
        "deberta-v3",
        "--model_name_or_path",
        args.model_path,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        "16",
        "--eval_batch_size",
        "32",
        "--validation_fraction",
        str(variant["validation_fraction"]),
        "--validation_seed",
        "2027",
        "--checkpoint_selection",
        variant["checkpoint_selection"],
        "--lambda_nexthop",
        "1.0",
        "--lambda_edge",
        "1.0",
        "--lambda_consistency",
        str(variant["lambda_consistency"]),
        "--gpus",
        str(gpu),
        "--seed",
        str(seed),
        "--metrics_out",
        str(metrics_path),
    ]


def summarize(output_root: Path, seeds: list[int]) -> dict:
    summary = {}
    for variant_name in VARIANTS:
        runs = [
            json.loads(
                (output_root / variant_name / f"seed_{seed}" / "metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            for seed in seeds
        ]
        metrics = {}
        for metric_name in ("overall", "short_hop", "long_hop", "transition_at_1"):
            values = [float(run["test"][metric_name]) for run in runs]
            metrics[metric_name] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values),
                "values": values,
            }
        summary[variant_name] = {
            "protocol": VARIANTS[variant_name],
            "selected_epochs": [run["selected_epoch"] for run in runs],
            "metrics": metrics,
        }
    return summary


def main() -> None:
    args = build_parser().parse_args()
    device_count = torch.cuda.device_count()
    invalid = [gpu for gpu in args.gpus if gpu < 0 or gpu >= device_count]
    if invalid:
        raise ValueError(f"Unavailable GPU indices: {invalid}; visible count={device_count}")

    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for variant_name, variant in VARIANTS.items():
        for seed in args.seeds:
            run_dir = output_root / variant_name / f"seed_{seed}"
            jobs.append(
                {
                    "variant_name": variant_name,
                    "variant": variant,
                    "seed": seed,
                    "run_dir": run_dir,
                    "metrics_path": run_dir / "metrics.json",
                }
            )

    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "clutrr_protocol_regression_audit",
        "dataset": args.dataset,
        "model_path": args.model_path,
        "epochs": args.epochs,
        "seeds": args.seeds,
        "variants": VARIANTS,
        "jobs": [
            {
                "variant": job["variant_name"],
                "seed": job["seed"],
                "metrics": str(job["metrics_path"]),
                "status": "complete" if valid_result(job["metrics_path"]) else "pending",
            }
            for job in jobs
        ],
    }
    write_json(output_root / "manifest.json", manifest)
    pending = [job for job in jobs if not valid_result(job["metrics_path"])]
    if args.dry_run:
        print(json.dumps({"total": len(jobs), "pending": len(pending)}, indent=2))
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
                repo_root,
                args,
                job["variant"],
                job["seed"],
                gpu,
                job["metrics_path"],
            )
            log_handle = log_path.open("w", encoding="utf-8")
            environment = os.environ.copy()
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
                f"launched gpu={gpu} variant={job['variant_name']} "
                f"seed={job['seed']} pid={process.pid}",
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
                f"finished gpu={gpu} variant={job['variant_name']} "
                f"seed={job['seed']} rc={return_code} complete={complete}",
                flush=True,
            )
            if not complete:
                failures.append(
                    {
                        "variant": job["variant_name"],
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
                        "variant": job["variant_name"],
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
            "complete" if valid_result(Path(record["metrics"])) else "failed"
        )
    write_json(output_root / "manifest.json", manifest)
    if failures:
        raise RuntimeError(f"{len(failures)} protocol-audit jobs failed")
    write_json(output_root / "summary.json", summarize(output_root, args.seeds))


if __name__ == "__main__":
    main()
