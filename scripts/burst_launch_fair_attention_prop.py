#!/usr/bin/env python3
"""Launch additional fair-attention jobs into currently idle GPUs.

This is a companion to run_fair_attention_prop_supervisor.sh. It uses the same
queue/status/result layout and starts only tasks without an existing .started
file, so it can safely fill idle GPUs while the conservative supervisor keeps
running.
"""

from __future__ import annotations

import csv
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


RUN_ROOT = Path("/vepfs/tsra_outputs/fair_attention_prop/latest").resolve()
MODEL = "/vepfs/tsra_models/hf/deberta-base"
GPU_MEM_THRESHOLD_MB = 1200


def now() -> str:
    return datetime.now().strftime("%F %T")


def free_gpus():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    gpus = []
    for line in out.strip().splitlines():
        idx, used = [x.strip() for x in line.split(",", 1)]
        if int(used) < GPU_MEM_THRESHOLD_MB:
            gpus.append(idx)
    return gpus


def load_queue():
    with (RUN_ROOT / "queue.tsv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def shell_quote_list(args):
    return " ".join(shlex.quote(str(x)) for x in args)


def launch(row, gpu):
    status = RUN_ROOT / "status"
    logs = RUN_ROOT / "logs"
    results = RUN_ROOT / "results"
    name = row["name"]
    started = status / f"{name}.started"
    done = status / f"{name}.done"
    failed = status / f"{name}.failed"
    log_file = logs / f"{name}.log"
    out_json = results / f"{name}.json"

    if started.exists():
        return False
    started.write_text(
        "\n".join(
            [
                f"name={name}",
                f"method={row['method']}",
                f"dataset={row['dataset']}",
                f"gpu={gpu}",
                f"seed={row['seed']}",
                f"epochs={row['epochs']}",
                f"started_at={now()}",
                "launcher=burst_launch_fair_attention_prop.py",
                "",
            ]
        ),
        encoding="utf-8",
    )

    args = [
        "python3",
        "scripts/fair_attention_prop.py",
        "--method",
        row["method"],
        "--dataset",
        row["dataset"],
        "--root",
        row["root"],
        "--model-name",
        MODEL,
        "--limit-train",
        "0",
        "--limit-test",
        "0",
        "--epochs",
        row["epochs"],
        "--batch-size",
        row["batch_size"],
        "--lr",
        row["lr"],
        "--max-sents",
        row["max_sents"],
        "--max-len",
        row["max_len"],
        "--seed",
        row["seed"],
        "--out",
        str(out_json),
    ]
    if row["train_depths"] != "-":
        args += ["--train-depths", row["train_depths"]]
    if row["test_depths"] != "-":
        args += ["--test-depths", row["test_depths"]]
    if row["train_qdeps"] != "-":
        args += ["--train-qdeps", row["train_qdeps"]]
    if row["test_qdeps"] != "-":
        args += ["--test-qdeps", row["test_qdeps"]]

    command = f"""
set +e
cd /root/TRUA
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES={shlex.quote(str(gpu))} {shell_quote_list(args)} > {shlex.quote(str(log_file))} 2>&1
rc=$?
if [ "$rc" -eq 0 ]; then
  {{ cat {shlex.quote(str(started))}; echo finished_at=$(date '+%F %T'); echo out={shlex.quote(str(out_json))}; tail -n 80 {shlex.quote(str(log_file))}; }} > {shlex.quote(str(done))}
else
  {{ cat {shlex.quote(str(started))}; echo failed_at=$(date '+%F %T'); echo exit_code=$rc; tail -n 160 {shlex.quote(str(log_file))}; }} > {shlex.quote(str(failed))}
fi
exit "$rc"
"""
    subprocess.Popen(["bash", "-lc", command], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    print(f"[{now()}] launched {name} on GPU {gpu}")
    return True


def main():
    queue = load_queue()
    gpus = free_gpus()
    launched = 0
    for gpu in gpus:
        row = next((r for r in queue if not (RUN_ROOT / "status" / f"{r['name']}.started").exists()), None)
        if row is None:
            break
        if launch(row, gpu):
            launched += 1
    print(f"[{now()}] burst launched {launched} job(s); free_gpus={','.join(gpus) or '-'} run_root={RUN_ROOT}")


if __name__ == "__main__":
    main()
