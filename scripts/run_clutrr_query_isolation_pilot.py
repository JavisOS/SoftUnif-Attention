#!/usr/bin/env python3
"""Run the paired CLUTRR joint/separate query-encoding diagnostic."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


VARIANTS = {
    "joint_guided": {"unit_encoding_mode": "joint", "use_goal_guidance": True},
    "joint_unguided": {"unit_encoding_mode": "joint", "use_goal_guidance": False},
    "separate_guided": {"unit_encoding_mode": "separate", "use_goal_guidance": True},
    "separate_unguided": {"unit_encoding_mode": "separate", "use_goal_guidance": False},
}
SEEDS = (0, 1, 42)


def repository_revision(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()


def valid_result(path: Path, revision: str) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("code_revision") == revision
        and payload.get("checkpoint_selection") == "validation"
        and payload.get("validation_selection_metric") == "unseen_4_10"
        and "overall" in payload.get("test", {})
    )


def command(
    repo: Path,
    data_root: Path,
    model_path: Path,
    variant: dict,
    seed: int,
    gpu: int,
    output: Path,
) -> list[str]:
    result = [
        sys.executable,
        "-u",
        "-m",
        "clutrr.cli.train",
        "--config",
        str(repo / "configs" / "clutrr" / "train_trua.yaml"),
        "--dataset",
        "data_089907f8",
        "--root",
        str(data_root),
        "--model_type",
        "deberta-v3",
        "--model_name_or_path",
        str(model_path),
        "--epochs",
        "10",
        "--batch_size",
        "16",
        "--eval_batch_size",
        "32",
        "--seed",
        str(seed),
        "--validation_fraction",
        "0.0",
        "--validation_seed",
        "2027",
        "--checkpoint_selection",
        "validation",
        "--validation_selection_metric",
        "unseen_4_10",
        "--external_validation_root",
        str(data_root),
        "--external_validation_dataset",
        "data_db9b8f04",
        "--lambda_nexthop",
        "1.0",
        "--lambda_edge",
        "1.0",
        "--lambda_consistency",
        "0.0",
        "--goal_representation",
        (
            "encoded_query"
            if variant["unit_encoding_mode"] == "separate"
            else "object"
        ),
        "--unit_encoding_mode",
        variant["unit_encoding_mode"],
        "--gpus",
        str(gpu),
        "--metrics_out",
        str(output),
    ]
    if not variant["use_goal_guidance"]:
        result.append("--no-use_goal_guidance")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/root/TRUA/data"))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/vepfs/trua_models/hf/deberta-v3-base"),
    )
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()

    revision = repository_revision(args.repo)
    jobs = []
    for name, variant in VARIANTS.items():
        for seed in SEEDS:
            run_dir = args.output_root / name / f"seed_{seed}"
            jobs.append(
                {
                    "name": name,
                    "variant": variant,
                    "seed": seed,
                    "run_dir": run_dir,
                    "metrics": run_dir / "metrics.json",
                }
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "clutrr_independent_query_goal_pilot",
        "code_revision": revision,
        "dataset": "data_089907f8",
        "development_dataset": "data_db9b8f04",
        "test_hops": list(range(2, 11)),
        "variants": VARIANTS,
        "seeds": list(SEEDS),
        "selection": "independent development accuracy on hops 4--10",
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    jobs = [job for job in jobs if not valid_result(job["metrics"], revision)]
    active = {}
    failures = []
    while jobs or active:
        for gpu in args.gpus:
            if gpu in active or not jobs:
                continue
            job = jobs.pop(0)
            job["run_dir"].mkdir(parents=True, exist_ok=True)
            log_path = job["run_dir"] / "train.log"
            log_handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command(
                    args.repo,
                    args.data_root,
                    args.model_path,
                    job["variant"],
                    job["seed"],
                    gpu,
                    job["metrics"],
                ),
                cwd=args.repo,
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
                f"launched gpu={gpu} variant={job['name']} "
                f"seed={job['seed']} pid={process.pid}",
                flush=True,
            )

        time.sleep(args.poll_seconds)
        for gpu, job in list(active.items()):
            return_code = job["process"].poll()
            if return_code is None:
                continue
            job["log_handle"].close()
            complete = return_code == 0 and valid_result(job["metrics"], revision)
            print(
                f"finished gpu={gpu} variant={job['name']} "
                f"seed={job['seed']} rc={return_code} complete={complete}",
                flush=True,
            )
            if not complete:
                failures.append(str(job["log_path"]))
            del active[gpu]

    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["failures"] = failures
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(f"Failed jobs: {failures}")


if __name__ == "__main__":
    main()
