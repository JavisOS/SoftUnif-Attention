#!/usr/bin/env python3
"""Compare first-occurrence and all-occurrence entity pooling on validation."""

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


SEEDS = (0, 1, 42)
POOLING_MODES = {
    "first_occurrence": "mean",
    "all_occurrences": "multi_mention",
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
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    return parser


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def valid_result(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        result.get("selected_validation_overall") is not None
        and isinstance(result.get("test"), dict)
    )


def command_for(
    repo_root: Path,
    *,
    data_root: str,
    dataset: str,
    model_path: str,
    pooling: str,
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
        dataset,
        "--root",
        data_root,
        "--model_type",
        "deberta-v3",
        "--model_name_or_path",
        model_path,
        "--epochs",
        "10",
        "--batch_size",
        "16",
        "--eval_batch_size",
        "32",
        "--seed",
        str(seed),
        "--validation_fraction",
        "0.1",
        "--validation_seed",
        "2027",
        "--entity_pooling",
        pooling,
        "--lambda_nexthop",
        "1.0",
        "--lambda_edge",
        "1.0",
        "--lambda_consistency",
        "0.0",
        "--gpus",
        str(gpu),
        "--metrics_out",
        str(metrics_path),
    ]


def summarize(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
        "values": values,
    }


def main() -> None:
    args = build_parser().parse_args()
    device_count = torch.cuda.device_count()
    invalid = [gpu for gpu in args.gpus if gpu < 0 or gpu >= device_count]
    if invalid:
        raise ValueError(
            f"Requested unavailable GPUs {invalid}; visible indices are 0--{device_count - 1}"
        )

    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for mode, pooling in POOLING_MODES.items():
        for seed in SEEDS:
            run_dir = output_root / mode / f"seed_{seed}"
            jobs.append(
                {
                    "mode": mode,
                    "pooling": pooling,
                    "seed": seed,
                    "run_dir": run_dir,
                    "metrics_path": run_dir / "metrics.json",
                }
            )

    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "clutrr_occurrence_pooling_validation_pilot",
        "model": args.model_path,
        "dataset": args.dataset,
        "seeds": list(SEEDS),
        "pooling_modes": POOLING_MODES,
        "fixed_protocol": {
            "epochs": 10,
            "validation_fraction": 0.1,
            "validation_seed": 2027,
            "checkpoint_selection": "validation answer accuracy",
            "transition_weight": 1.0,
            "edge_weight": 1.0,
            "consistency_weight": 0.0,
        },
        "preregistered_adapter_rule": (
            "adopt all-occurrence pooling only if mean selected validation "
            "accuracy is higher and at least two of three paired seeds are "
            "not lower than first-occurrence pooling"
        ),
    }
    write_json(output_root / "manifest.json", manifest)

    pending = [job for job in jobs if not valid_result(job["metrics_path"])]
    active: dict[int, dict] = {}
    failures = []
    while pending or active:
        for gpu in args.gpus:
            if gpu in active or not pending:
                continue
            job = pending.pop(0)
            job["run_dir"].mkdir(parents=True, exist_ok=True)
            log_path = job["run_dir"] / "train.log"
            log_handle = log_path.open("w", encoding="utf-8")
            command = command_for(
                repo_root,
                data_root=args.data_root,
                dataset=args.dataset,
                model_path=args.model_path,
                pooling=job["pooling"],
                seed=job["seed"],
                gpu=gpu,
                metrics_path=job["metrics_path"],
            )
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
                f"launched gpu={gpu} mode={job['mode']} seed={job['seed']} "
                f"pid={process.pid}",
                flush=True,
            )

        time.sleep(args.poll_seconds)
        for gpu, job in list(active.items()):
            return_code = job["process"].poll()
            if return_code is None:
                continue
            job["log_handle"].close()
            complete = valid_result(job["metrics_path"])
            print(
                f"finished gpu={gpu} mode={job['mode']} seed={job['seed']} "
                f"rc={return_code} complete={complete}",
                flush=True,
            )
            if return_code != 0 or not complete:
                failures.append(
                    {
                        "mode": job["mode"],
                        "seed": job["seed"],
                        "return_code": return_code,
                        "log": str(job["log_path"]),
                    }
                )
            del active[gpu]

    if failures:
        write_json(output_root / "failures.json", failures)
        raise RuntimeError(f"Pooling pilot failed: {failures}")

    validation = {}
    for mode in POOLING_MODES:
        values = []
        for seed in SEEDS:
            path = output_root / mode / f"seed_{seed}" / "metrics.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            values.append(float(result["selected_validation_overall"]))
        validation[mode] = summarize(values)

    paired = [
        all_value - first_value
        for all_value, first_value in zip(
            validation["all_occurrences"]["values"],
            validation["first_occurrence"]["values"],
        )
    ]
    adopt = (
        validation["all_occurrences"]["mean"]
        > validation["first_occurrence"]["mean"]
        and sum(difference >= 0.0 for difference in paired) >= 2
    )
    write_json(
        output_root / "selection_summary.json",
        {
            "selection_metric": "selected validation answer accuracy",
            "validation": validation,
            "paired_all_minus_first": paired,
            "criterion_passed": adopt,
            "selected_pooling": "all_occurrences" if adopt else "first_occurrence",
            "test_metrics_not_used_for_selection": True,
        },
    )
    print(f"selection complete: {'all_occurrences' if adopt else 'first_occurrence'}")


if __name__ == "__main__":
    main()
