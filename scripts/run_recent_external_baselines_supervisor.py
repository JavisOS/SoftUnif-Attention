#!/usr/bin/env python3
"""Queue recent external baselines without disturbing active TRUA jobs.

The supervisor registers AAI, LoGiPT, Coconut, and CODI in order. It launches a
job only when enough GPUs are free and its preflight check passes. Missing model
checkpoints or unfinished adapters remain in a `.pending` state instead of being
reported as experiment failures.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_ROOT_PARENT = Path("/vepfs/tsra_outputs/recent_external_baselines")
# Treat keepalive allocations (~1 GB/GPU) as occupied. Truly idle A100s show
# only a few MB from the driver in this environment.
GPU_MEM_THRESHOLD_MB = 100


@dataclass(frozen=True)
class Task:
    name: str
    method: str
    dataset: str
    gpus: int
    workdir: str
    ready: str
    command: str
    notes: str


def now() -> str:
    return datetime.now().strftime("%F %T")


def shell_quote(items) -> str:
    return " ".join(shlex.quote(str(x)) for x in items)


def queue() -> list[Task]:
    qwen = "/vepfs/tsra_models/hf/Qwen3-32B"
    logipt_pw = "/vepfs/tsra_models/hf/logipt/LoGiPT-CodeLlama-13b-Instruct-hf-proofwriter"
    codi_ckpt = "/vepfs/tsra_models/hf/zen-E-CODI-gpt2"
    return [
        Task(
            name="01_aai_proofwriter_qwen3_32b",
            method="AAI",
            dataset="ProofWriter",
            gpus=1,
            workdir="/root/TRUA/external_baselines/AAI",
            ready=f"test -d {shlex.quote(qwen)}",
            command=(
                "MODEL_NAME=/vepfs/tsra_models/hf/Qwen3-32B "
                "DATA_PATH=./data/ProofWriter/test.logiccotkb_prompting.json "
                "bash -lc 'python ./src/infer_llm.py --model_name \"$MODEL_NAME\" "
                "--prompting_data_path \"$DATA_PATH\" --max_new_tokens 2000 "
                "--logical_masked_func no_masked_attn --batch_size 4 && "
                "python ./src/infer_llm.py --model_name \"$MODEL_NAME\" "
                "--prompting_data_path \"$DATA_PATH\" --max_new_tokens 2000 "
                "--logical_masked_func generate_focusing_rule_inc_attn_masked_positions "
                "--batch_size 4 --apply_dynamic_attn_pattern --strong_att_const 0.04 "
                "--filter_head_attn_pattern filter_high_diagonal_attention'"
            ),
            notes="Official AAI ProofWriter run: Symbolic-Aided CoT baseline plus AAI attention intervention. Qwen3-32B is expected at /vepfs/tsra_models/hf/Qwen3-32B, normally a symlink to /tos/lxh/models/qwen3_32.",
        ),
        Task(
            name="02_logipt_proofwriter_codellama13b",
            method="LoGiPT",
            dataset="ProofWriter",
            gpus=1,
            workdir="/root/TRUA",
            ready=(
                f"test -f {shlex.quote(logipt_pw)}/model-00001-of-00003.safetensors && "
                f"test -f {shlex.quote(logipt_pw)}/model-00002-of-00003.safetensors && "
                f"test -f {shlex.quote(logipt_pw)}/model-00003-of-00003.safetensors && "
                f"test -f {shlex.quote(logipt_pw)}/tokenizer.model && "
                "test -f scripts/run_logipt_eval.py"
            ),
            command=(
                "python3 scripts/run_logipt_eval.py "
                "--model /vepfs/tsra_models/hf/logipt/LoGiPT-CodeLlama-13b-Instruct-hf-proofwriter "
                "--dataset proofwriter --split depth3,depth5 "
                "--batch-size 4 --max-new-tokens 8 "
                "--out /vepfs/tsra_outputs/recent_external_baselines/latest/results/logipt_proofwriter.json"
            ),
            notes="LoGiPT is a strong internal-deduction reference. Pending until the official ProofWriter checkpoint and TRUA eval adapter are available.",
        ),
        Task(
            name="03_coconut_prontoqa_ood_gpt2",
            method="Coconut",
            dataset="PrOntoQA-OOD",
            gpus=4,
            workdir="/root/TRUA/external_baselines/Coconut_official",
            ready=(
                "test -f /vepfs/tsra_outputs/recent_external_data/prontoqa_ood/"
                "prontoqa_ood_coconut.yaml"
            ),
            command=(
                "WANDB_MODE=offline torchrun --nnodes 1 --nproc_per_node 4 run.py "
                "/vepfs/tsra_outputs/recent_external_data/prontoqa_ood/"
                "prontoqa_ood_coconut.yaml"
            ),
            notes="Official Coconut code on TRUA PrOntoQA-OOD converted to question/answer/steps format. This is the modern latent-reasoning baseline.",
        ),
    ]


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, **kwargs)


def free_gpus() -> list[str]:
    proc = run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        check=True,
    )
    gpus = []
    for line in proc.stdout.strip().splitlines():
        idx, used = [x.strip() for x in line.split(",", 1)]
        if int(used) < GPU_MEM_THRESHOLD_MB:
            gpus.append(idx)
    return gpus


def write_queue(run_root: Path, tasks: list[Task]) -> None:
    with (run_root / "queue.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["name", "method", "dataset", "gpus", "workdir", "ready", "command", "notes"])
        for t in tasks:
            writer.writerow([t.name, t.method, t.dataset, t.gpus, t.workdir, t.ready, t.command, t.notes])


def log(run_root: Path, msg: str) -> None:
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with (run_root / "supervisor.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def check_ready(task: Task) -> tuple[bool, str]:
    proc = run(["bash", "-lc", task.ready])
    if proc.returncode == 0:
        return True, "ready"
    reason = (proc.stderr or proc.stdout or f"preflight exited {proc.returncode}").strip()
    return False, reason


def launch(run_root: Path, task: Task, gpus: list[str]) -> None:
    gpu_list = ",".join(gpus[: task.gpus])
    status = run_root / "status"
    logs = run_root / "logs"
    results = run_root / "results"
    started = status / f"{task.name}.started"
    done = status / f"{task.name}.done"
    failed = status / f"{task.name}.failed"
    log_file = logs / f"{task.name}.log"

    started.write_text(
        "\n".join(
            [
                f"name={task.name}",
                f"method={task.method}",
                f"dataset={task.dataset}",
                f"gpus={gpu_list}",
                f"started_at={now()}",
                f"notes={task.notes}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    wrapper = f"""
set +e
cd {shlex.quote(task.workdir)}
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/vepfs/tsra_models/hf_home
export TRANSFORMERS_CACHE=/vepfs/tsra_models/hf_home
export CUDA_VISIBLE_DEVICES={shlex.quote(gpu_list)}
{task.command} > {shlex.quote(str(log_file))} 2>&1
rc=$?
if [ "$rc" -eq 0 ]; then
  {{ cat {shlex.quote(str(started))}; echo finished_at=$(date '+%F %T'); tail -n 120 {shlex.quote(str(log_file))}; }} > {shlex.quote(str(done))}
else
  {{ cat {shlex.quote(str(started))}; echo failed_at=$(date '+%F %T'); echo exit_code=$rc; tail -n 200 {shlex.quote(str(log_file))}; }} > {shlex.quote(str(failed))}
fi
exit "$rc"
"""
    subprocess.Popen(
        ["bash", "-lc", wrapper],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    log(run_root, f"launched {task.name} on GPU(s) {gpu_list}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="")
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()

    if args.run_root:
        run_root = Path(args.run_root)
    else:
        run_root = DEFAULT_ROOT_PARENT / f"recent_external_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    for sub in ["status", "logs", "results"]:
        (run_root / sub).mkdir(parents=True, exist_ok=True)
    latest = DEFAULT_ROOT_PARENT / "latest"
    latest.parent.mkdir(parents=True, exist_ok=True)
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(run_root)

    tasks = queue()
    write_queue(run_root, tasks)
    log(run_root, f"recent external baseline supervisor started; tasks={len(tasks)} run_root={run_root}")

    while True:
        remaining = [
            t
            for t in tasks
            if not (run_root / "status" / f"{t.name}.started").exists()
            and not (run_root / "status" / f"{t.name}.done").exists()
        ]
        if not remaining:
            log(run_root, "all recent external baseline queue entries dispatched")
            return

        gpus = free_gpus()
        launched = False
        for task in remaining:
            ok, reason = check_ready(task)
            pending = run_root / "status" / f"{task.name}.pending"
            if not ok:
                pending.write_text(
                    f"name={task.name}\nmethod={task.method}\ndataset={task.dataset}\n"
                    f"last_checked_at={now()}\nreason={reason}\nnotes={task.notes}\n",
                    encoding="utf-8",
                )
                log(run_root, f"pending {task.name}: {reason or task.notes}")
                continue
            if pending.exists():
                pending.unlink()
            if len(gpus) < task.gpus:
                log(run_root, f"ready but waiting for {task.gpus} free GPU(s) for {task.name}; free={','.join(gpus) or '-'}")
                continue
            launch(run_root, task, gpus)
            launched = True
            break

        if not launched:
            log(run_root, "no recent external baseline launched in this poll cycle")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
