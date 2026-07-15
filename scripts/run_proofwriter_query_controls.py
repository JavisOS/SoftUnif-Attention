#!/usr/bin/env python3
"""Run clean ProofWriter controls for query injection and attention type."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


CONTROLS = (
    (
        "explicit_goal_only",
        ("--core-type", "trua", "--use-goal-guidance", "--no-use-query-anchor"),
    ),
    (
        "query_anchor_only",
        ("--core-type", "trua", "--no-use-goal-guidance", "--use-query-anchor"),
    ),
    (
        "self_attention_matched",
        ("--core-type", "self_attention", "--no-use-goal-guidance", "--use-query-anchor"),
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 42])
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--limit-train", type=int, default=30000)
    parser.add_argument("--limit-test", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if len(args.gpus) < len(args.seeds):
        raise ValueError("Provide at least one GPU per concurrently launched seed")

    repo_root = Path(__file__).resolve().parents[1]
    runner = repo_root / "scripts" / "transformer_trua_prop.py"
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "proofwriter_query_mechanism_controls",
        "controls": [name for name, _ in CONTROLS],
        "seeds": args.seeds,
        "commands": [],
    }

    for control_name, control_flags in CONTROLS:
        processes = []
        for seed, gpu in zip(args.seeds, args.gpus):
            run_dir = output_root / control_name / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = run_dir / "metrics.json"
            log_path = run_dir / "train.log"
            command = [
                sys.executable,
                str(runner),
                "--dataset",
                "proofwriter",
                "--root",
                args.data_root,
                "--model-name",
                args.model_name,
                "--lambda-evidence",
                "1",
                "--train-depths",
                "0,1,2",
                "--test-depths",
                "3,5",
                "--limit-train",
                str(args.limit_train),
                "--limit-test",
                str(args.limit_test),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--max-sents",
                "32",
                "--max-len",
                "512",
                "--validation-seed",
                "2027",
                "--seed",
                str(seed),
                "--out",
                str(metrics_path),
                *control_flags,
            ]
            log_handle = log_path.open("w", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment["PYTHONPATH"] = str(repo_root)
            process = subprocess.Popen(
                command,
                cwd=repo_root,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((seed, process, log_handle, log_path))
            manifest["commands"].append(
                {
                    "control": control_name,
                    "seed": seed,
                    "gpu": gpu,
                    "command": command,
                }
            )

        failures = []
        for seed, process, log_handle, log_path in processes:
            return_code = process.wait()
            log_handle.close()
            if return_code != 0:
                failures.append(f"{control_name}/seed_{seed} (exit={return_code}, log={log_path})")
        if failures:
            raise RuntimeError("ProofWriter control failures: " + "; ".join(failures))

    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
