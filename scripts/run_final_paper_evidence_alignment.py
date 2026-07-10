#!/usr/bin/env python3
"""Run the final validation-selected TRUA paper protocol."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TRUA_REPO_ROOT", "/root/TRUA")
import run_paper_evidence_alignment as base


REPO_ROOT = Path(os.environ["TRUA_REPO_ROOT"])
MODEL_ROOT = Path("/vepfs/trua_models/hf")


def build_tasks(result_dir: Path):
    seeds = (0, 1, 42)
    tasks = []
    clutrr_variants = (
        ("trua", (1, 1, 0), ()),
        ("with_consistency", (1, 1, 5), ()),
        ("answer_only", (0, 0, 0), ()),
        ("transition_only", (1, 0, 0), ()),
        ("edge_only", (0, 1, 0), ()),
        ("no_goal", (1, 1, 0), ("--no-use_goal_guidance",)),
        ("no_aggregation", (1, 1, 0), ("--no-use_aggregation_branch",)),
        ("no_step", (0, 1, 0), ("--no-use_step_branch",)),
        ("no_relation", (1, 1, 0), ("--no-use_relation_conditioning",)),
    )
    for variant, lambdas, flags in clutrr_variants:
        for seed in seeds:
            name = f"clutrr_deberta_{variant}_seed{seed}"
            tasks.append(
                base.clutrr_command(
                    name,
                    seed,
                    result_dir / f"{name}.json",
                    lambdas=lambdas,
                    flags=flags,
                )
            )
    for seed in seeds:
        name = f"clutrr_deberta_vanilla_seed{seed}"
        tasks.append(base.clutrr_command(name, seed, result_dir / f"{name}.json", vanilla=True))

    proof_root = REPO_ROOT / "data/proofwriter/raw/proofwriter-dataset-V2020.12.3"
    for variant, lambda_evidence, extra in (
        ("trua", 1.0, ()),
        ("no_evidence", 0.0, ()),
        ("no_goal", 1.0, ("--no-use-goal-guidance",)),
    ):
        for seed in seeds:
            name = f"proofwriter_bert_{variant}_seed{seed}"
            tasks.append(
                base.proposition_command(
                    name,
                    seed,
                    result_dir / f"{name}.json",
                    dataset="proofwriter",
                    root=proof_root,
                    model=MODEL_ROOT / "bert-base-uncased",
                    lambda_evidence=lambda_evidence,
                    extra=("--train-depths", "0,1,2", "--test-depths", "3,5", *extra),
                    max_sents=32,
                )
            )

    rule_root = REPO_ROOT / "data/rule-reasoning-dataset-V2020.2.5.0/original"
    for variant, lambda_evidence in (("trua", 1.0), ("no_evidence", 0.0)):
        for seed in seeds:
            name = f"ruletaker_raw_deberta_{variant}_seed{seed}"
            tasks.append(
                base.proposition_command(
                    name,
                    seed,
                    result_dir / f"{name}.json",
                    dataset="ruletaker_raw",
                    root=rule_root,
                    model=MODEL_ROOT / "deberta-base",
                    lambda_evidence=lambda_evidence,
                    extra=(
                        "--train-depths",
                        "1,2",
                        "--test-depths",
                        "1,2,3,5",
                        "--train-qdeps",
                        "1,2",
                        "--test-qdeps",
                        "1,2,3,4,5",
                    ),
                    max_sents=32,
                )
            )

    pronto_root = REPO_ROOT / "data/prontoqa_ood/processed/generated_ood_data"
    for variant, lambda_evidence in (("trua", 1.0), ("no_evidence", 0.0)):
        for seed in seeds:
            name = f"prontoqa_bert_{variant}_seed{seed}"
            tasks.append(
                base.proposition_command(
                    name,
                    seed,
                    result_dir / f"{name}.json",
                    dataset="prontoqa",
                    root=pronto_root,
                    model=MODEL_ROOT / "bert-base-uncased",
                    lambda_evidence=lambda_evidence,
                    limit_train=0,
                    limit_test=0,
                    max_sents=24,
                )
            )
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--task-pattern", default=".*")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    result_dir = run_root / "results"
    log_dir = run_root / "logs"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(args.task_pattern)
    tasks = [task for task in build_tasks(result_dir) if pattern.search(task.name)]
    if args.dry_run:
        print(json.dumps([asdict(task) for task in tasks], indent=2))
        return

    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    pending = [task for task in tasks if not base.result_is_complete(Path(task.result_path))]
    completed = [task.name for task in tasks if base.result_is_complete(Path(task.result_path))]
    failed = {}
    running = {}
    base.write_json(
        run_root / "manifest.json",
        {"created_at": datetime.now().isoformat(timespec="seconds"), "tasks": [asdict(task) for task in tasks]},
    )

    while pending or running:
        free_gpus = [gpu for gpu in gpus if gpu not in running]
        while pending and free_gpus:
            gpu = free_gpus.pop(0)
            task = pending.pop(0)
            log_path = log_dir / f"{task.name}.log"
            log_handle = log_path.open("w", encoding="utf-8")
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            command = [gpu if token == "__GPU__" else token for token in task.command]
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            running[gpu] = {"task": task, "process": process, "log": log_handle}
            print(f"launched {task.name} on GPU {gpu} pid={process.pid}", flush=True)

        time.sleep(args.poll_seconds)
        for gpu, item in list(running.items()):
            return_code = item["process"].poll()
            if return_code is None:
                continue
            item["log"].close()
            task = item["task"]
            if return_code == 0 and base.result_is_complete(Path(task.result_path)):
                completed.append(task.name)
                print(f"completed {task.name} on GPU {gpu}", flush=True)
            else:
                failed[task.name] = return_code
                print(f"failed {task.name} on GPU {gpu} rc={return_code}", flush=True)
            del running[gpu]

        base.write_json(
            run_root / "status.json",
            {
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "total": len(tasks),
                "completed": sorted(completed),
                "failed": failed,
                "pending": [task.name for task in pending],
                "running": {gpu: item["task"].name for gpu, item in running.items()},
            },
        )

    if failed:
        raise SystemExit(f"{len(failed)} tasks failed")
    print(f"all {len(tasks)} final tasks completed in {run_root}", flush=True)


if __name__ == "__main__":
    main()
