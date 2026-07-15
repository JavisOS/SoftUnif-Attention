"""Launch the validation-selected counterfactual CLUTRR control."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 42])
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--cf_weight", type=float, default=1.0)
    parser.add_argument("--consistency_weight", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if len(args.gpus) < len(args.seeds):
        raise ValueError("Provide at least one GPU per concurrently launched seed")

    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "crest_inspired_counterfactual_clutrr_control",
        "implementation_scope": "not_an_official_crest_reproduction",
        "code_revision": revision,
        "seeds": args.seeds,
        "commands": [],
    }

    processes = []
    for seed, gpu in zip(args.seeds, args.gpus):
        run_dir = output_root / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.json"
        log_path = run_dir / "train.log"
        command = [
            sys.executable,
            "-m",
            "clutrr.cli.train_counterfactual_control",
            "--root",
            args.root,
            "--dataset",
            args.dataset,
            "--model_type",
            "deberta",
            "--model_name_or_path",
            args.model_name_or_path,
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--batch_size",
            str(args.batch_size),
            "--eval_batch_size",
            str(args.eval_batch_size),
            "--seed",
            str(seed),
            "--validation_fraction",
            "0.1",
            "--validation_seed",
            "2027",
            "--cf_weight",
            str(args.cf_weight),
            "--consistency_weight",
            str(args.consistency_weight),
            "--gpus",
            str(gpu),
            "--metrics_out",
            str(metrics_path),
        ]
        log_handle = log_path.open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((seed, process, log_handle, log_path))
        manifest["commands"].append(
            {"seed": seed, "gpu": gpu, "command": command}
        )

    failures = []
    for seed, process, log_handle, log_path in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code != 0:
            failures.append(f"seed_{seed} (exit={return_code}, log={log_path})")
    if failures:
        raise RuntimeError("Counterfactual control failures: " + "; ".join(failures))

    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
