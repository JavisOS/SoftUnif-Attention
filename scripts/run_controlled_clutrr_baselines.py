"""Run four controlled CLUTRR baselines in parallel across four GPUs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


CORES = ("encoder", "self_attention", "mac", "rca")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 42])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "controlled_text_input_baselines",
        "cores": list(CORES),
        "seeds": args.seeds,
        "commands": [],
    }

    for seed in args.seeds:
        processes = []
        for gpu, core in enumerate(CORES):
            run_dir = output_root / core / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = run_dir / "metrics.json"
            log_path = run_dir / "train.log"
            command = [
                sys.executable,
                "-m",
                "clutrr.cli.train_controlled_baseline",
                "--core_type",
                core,
                "--model_type",
                "deberta",
                "--model_name_or_path",
                args.model_name_or_path,
                "--root",
                args.root,
                "--dataset",
                args.dataset,
                "--epochs",
                str(args.epochs),
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
                "--gpus",
                str(gpu),
                "--metrics_out",
                str(metrics_path),
            ]
            log_handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[1],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((core, process, log_handle, log_path))
            manifest["commands"].append({"core": core, "seed": seed, "gpu": gpu, "command": command})

        failures = []
        for core, process, log_handle, log_path in processes:
            return_code = process.wait()
            log_handle.close()
            if return_code != 0:
                failures.append(f"{core}/seed_{seed} (exit={return_code}, log={log_path})")
        if failures:
            raise RuntimeError("Controlled baseline failures: " + "; ".join(failures))

    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
