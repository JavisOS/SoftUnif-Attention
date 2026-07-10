#!/usr/bin/env python3
"""Run validation-selected TRUA experiments across the available GPUs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = Path("/vepfs/trua_outputs/paper_evidence_alignment")
MODEL_ROOT = Path("/vepfs/tsra_models/hf")


@dataclass
class Task:
    name: str
    command: list[str]
    result_path: str


def clutrr_command(name, seed, result, *, vanilla=False, lambdas=(1.0, 1.0, 5.0), flags=()):
    common = [
        sys.executable,
        "-u",
        "-m",
        "clutrr.cli.baseline" if vanilla else "clutrr.cli.train",
        "--dataset",
        "data_089907f8",
        "--root",
        str(REPO_ROOT / "data"),
        "--model_type",
        "deberta",
        "--model_name_or_path",
        str(MODEL_ROOT / "deberta-base"),
        "--epochs",
        "10",
        "--batch_size",
        "16",
        "--eval_batch_size",
        "32",
        "--gpus",
        "__GPU__",
        "--seed",
        str(seed),
        "--validation_fraction",
        "0.1",
        "--validation_seed",
        "2027",
        "--metrics_out",
        str(result),
    ]
    if not vanilla:
        common.extend(
            [
                "--lambda_nexthop",
                str(lambdas[0]),
                "--lambda_edge",
                str(lambdas[1]),
                "--lambda_consistency",
                str(lambdas[2]),
                *flags,
            ]
        )
    return Task(name=name, command=common, result_path=str(result))


def proposition_command(
    name,
    seed,
    result,
    *,
    dataset,
    root,
    model,
    lambda_trace,
    extra=(),
    limit_train=30000,
    limit_test=5000,
    max_sents=16,
):
    command = [
        sys.executable,
        "-u",
        "scripts/transformer_trua_prop.py",
        "--dataset",
        dataset,
        "--root",
        str(root),
        "--model-name",
        str(model),
        "--lambda-trace",
        str(lambda_trace),
        "--limit-train",
        str(limit_train),
        "--limit-test",
        str(limit_test),
        "--epochs",
        "10",
        "--batch-size",
        "16",
        "--lr",
        "2e-5",
        "--max-sents",
        str(max_sents),
        "--max-len",
        "512",
        "--validation-seed",
        "2027",
        "--seed",
        str(seed),
        "--out",
        str(result),
        *extra,
    ]
    return Task(name=name, command=command, result_path=str(result))


def build_tasks(result_dir):
    seeds = (0, 1, 42)
    tasks = []
    clutrr_variants = [
        ("full", (1, 1, 5), ()),
        ("label_only", (0, 0, 0), ()),
        ("no_goal", (1, 1, 5), ("--no-use_goal_guidance",)),
        ("no_aggregation", (1, 1, 5), ("--no-use_aggregation_branch",)),
        ("no_step", (0, 1, 5), ("--no-use_step_branch",)),
        ("no_relation", (1, 1, 5), ("--no-use_relation_conditioning",)),
        ("transition_only", (1, 0, 0), ()),
        ("edge_only", (0, 1, 0), ()),
        ("consistency_only", (0, 0, 5), ()),
        ("no_transition", (0, 1, 5), ()),
        ("no_edge", (1, 0, 5), ()),
        ("no_consistency", (1, 1, 0), ()),
    ]
    for variant, lambdas, flags in clutrr_variants:
        for seed in seeds:
            name = f"clutrr_deberta_{variant}_seed{seed}"
            tasks.append(
                clutrr_command(name, seed, result_dir / f"{name}.json", lambdas=lambdas, flags=flags)
            )
    for seed in seeds:
        name = f"clutrr_deberta_vanilla_seed{seed}"
        tasks.append(clutrr_command(name, seed, result_dir / f"{name}.json", vanilla=True))

    proof_root = REPO_ROOT / "data/proofwriter/raw/proofwriter-dataset-V2020.12.3"
    for variant, lambda_trace, extra in [
        ("full", 1.0, ()),
        ("no_trace", 0.0, ()),
        ("no_goal", 1.0, ("--no-use-goal-guidance",)),
    ]:
        for seed in seeds:
            name = f"proofwriter_bert_{variant}_seed{seed}"
            tasks.append(
                proposition_command(
                    name,
                    seed,
                    result_dir / f"{name}.json",
                    dataset="proofwriter",
                    root=proof_root,
                    model=MODEL_ROOT / "bert-base-uncased",
                    lambda_trace=lambda_trace,
                    extra=("--train-depths", "0,1,2", "--test-depths", "3,5", *extra),
                    max_sents=32,
                )
            )

    rule_root = REPO_ROOT / "data/rule-reasoning-dataset-V2020.2.5.0/original"
    for variant, lambda_trace in [("full", 1.0), ("no_trace", 0.0)]:
        for seed in seeds:
            name = f"ruletaker_raw_deberta_{variant}_seed{seed}"
            tasks.append(
                proposition_command(
                    name,
                    seed,
                    result_dir / f"{name}.json",
                    dataset="ruletaker_raw",
                    root=rule_root,
                    model=MODEL_ROOT / "deberta-base",
                    lambda_trace=lambda_trace,
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
    for variant, lambda_trace in [("full", 1.0), ("no_trace", 0.0)]:
        for seed in seeds:
            name = f"prontoqa_bert_{variant}_seed{seed}"
            tasks.append(
                proposition_command(
                    name,
                    seed,
                    result_dir / f"{name}.json",
                    dataset="prontoqa",
                    root=pronto_root,
                    model=MODEL_ROOT / "bert-base-uncased",
                    lambda_trace=lambda_trace,
                    limit_train=0,
                    limit_test=0,
                    max_sents=24,
                )
            )
    return tasks


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def result_is_complete(path):
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except (OSError, json.JSONDecodeError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--task-pattern", default=".*", help="Regular expression selecting task names.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = (args.run_root or DEFAULT_OUTPUT_ROOT / timestamp).resolve()
    result_dir = run_root / "results"
    log_dir = run_root / "logs"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    task_pattern = re.compile(args.task_pattern)
    tasks = [task for task in build_tasks(result_dir) if task_pattern.search(task.name)]
    if args.dry_run:
        print(json.dumps([asdict(task) for task in tasks], indent=2))
        return
    pending = [task for task in tasks if not result_is_complete(Path(task.result_path))]
    completed = [task.name for task in tasks if result_is_complete(Path(task.result_path))]
    failed = {}
    running = {}

    write_json(run_root / "manifest.json", {"created_at": timestamp, "tasks": [asdict(task) for task in tasks]})
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
            running[gpu] = {"task": task, "process": process, "log": log_handle, "started": time.time()}
            print(f"launched {task.name} on GPU {gpu} pid={process.pid}", flush=True)

        time.sleep(args.poll_seconds)
        for gpu, item in list(running.items()):
            return_code = item["process"].poll()
            if return_code is None:
                continue
            item["log"].close()
            task = item["task"]
            if return_code == 0 and result_is_complete(Path(task.result_path)):
                completed.append(task.name)
                print(f"completed {task.name} on GPU {gpu}", flush=True)
            else:
                failed[task.name] = return_code
                print(f"failed {task.name} on GPU {gpu} rc={return_code}", flush=True)
            del running[gpu]

        write_json(
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
    print(f"all {len(tasks)} tasks completed in {run_root}", flush=True)


if __name__ == "__main__":
    main()
