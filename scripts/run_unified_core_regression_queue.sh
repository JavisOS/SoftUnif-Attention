#!/usr/bin/env bash
set -euo pipefail

cd /root/TRUA

RUN_ROOT="/vepfs/tsra_outputs/unified_core_regression/unified_core_regression_$(date +%Y%m%d_%H%M%S)"
LATEST="/vepfs/tsra_outputs/unified_core_regression/latest"
QUEUE="$RUN_ROOT/queue.tsv"
LOG_DIR="$RUN_ROOT/logs"
RESULT_DIR="$RUN_ROOT/results"
STATUS_DIR="$RUN_ROOT/status"
LOCK_DIR="$RUN_ROOT/gpu_locks"
GPUS="${TRUA_CORE_REGRESSION_GPUS:-0,1,2,3,4,5,6,7}"
GPU_MEM_THRESHOLD_MB="${TRUA_GPU_MEM_THRESHOLD_MB:-1000}"
WAIT_SECONDS="${TRUA_QUEUE_WAIT_SECONDS:-120}"

mkdir -p "$LOG_DIR" "$RESULT_DIR" "$STATUS_DIR" "$LOCK_DIR"
ln -sfn "$RUN_ROOT" "$LATEST"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

DEBERTA="/vepfs/tsra_models/hf/deberta-base"
ROBERTA="/vepfs/tsra_models/hf/roberta-base-clean-current"

cat > "$QUEUE" <<EOF
name	kind	seed
clutrr_089_roberta_tsra_seed0	clutrr_roberta	0
clutrr_089_roberta_tsra_seed1	clutrr_roberta	1
clutrr_089_roberta_tsra_seed42	clutrr_roberta	42
proofwriter_deberta_trua_core_seed0	proofwriter	0
proofwriter_deberta_trua_core_seed1	proofwriter	1
proofwriter_deberta_trua_core_seed42	proofwriter	42
ruletaker_raw_deberta_trua_core_seed0	ruletaker_raw	0
ruletaker_raw_deberta_trua_core_seed1	ruletaker_raw	1
ruletaker_raw_deberta_trua_core_seed42	ruletaker_raw	42
prontoqa_deberta_trua_core_seed0	prontoqa	0
prontoqa_deberta_trua_core_seed1	prontoqa	1
prontoqa_deberta_trua_core_seed42	prontoqa	42
EOF

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$RUN_ROOT/supervisor.log"
}

free_gpu() {
  local gpu used lock
  IFS=',' read -ra candidates <<< "$GPUS"
  for gpu in "${candidates[@]}"; do
    lock="$LOCK_DIR/gpu${gpu}.lock"
    [[ -f "$lock" ]] && continue
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
    if [[ "${used:-999999}" -lt "$GPU_MEM_THRESHOLD_MB" ]]; then
      echo "$gpu"
      return 0
    fi
  done
  return 1
}

started_count() {
  find "$STATUS_DIR" -name "*.started" -type f | wc -l
}

finished_count() {
  local done failed
  done="$(find "$STATUS_DIR" -name "*.done" -type f | wc -l)"
  failed="$(find "$STATUS_DIR" -name "*.failed" -type f | wc -l)"
  echo $((done + failed))
}

write_started() {
  local name="$1" kind="$2" seed="$3" gpu="$4" out_file="$5"
  {
    echo "task=$name"
    echo "kind=$kind"
    echo "seed=$seed"
    echo "epochs=10"
    echo "gpu=$gpu"
    echo "code_state=unified_core_regression"
    echo "started_at=$(date '+%F %T')"
    echo "out=$out_file"
  } > "$STATUS_DIR/${name}.started"
}

run_clutrr_roberta() {
  local gpu="$1" name="$2" seed="$3" log_file="$4"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m clutrr.cli.train \
    --config configs/clutrr/train_trua.yaml \
    --dataset data_089907f8 \
    --root data \
    --model_type roberta \
    --model_name_or_path "$ROBERTA" \
    --epochs 10 \
    --batch_size 16 \
    --eval_batch_size 32 \
    --gpus "$gpu" \
    --strategy single \
    --seed "$seed" \
    --lambda_nexthop 1.0 \
    --lambda_edge 1.0 \
    --lambda_consistency 5.0 \
    > "$log_file" 2>&1
}

run_prop() {
  local gpu="$1" kind="$2" seed="$3" out_file="$4" log_file="$5"
  local common=(--model-name "$DEBERTA" --seed "$seed" --epochs 10 --limit-train 0 --limit-test 0 --batch-size 16 --lr 2e-5 --lambda-trace 1.0 --out "$out_file")
  case "$kind" in
    proofwriter)
      CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/transformer_trua_prop.py \
        --dataset proofwriter \
        --root data/proofwriter/raw/proofwriter-dataset-V2020.12.3 \
        --train-depths 0,1,2 \
        --test-depths 3,5 \
        --max-sents 16 \
        --max-len 192 \
        "${common[@]}" \
        > "$log_file" 2>&1
      ;;
    ruletaker_raw)
      CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/transformer_trua_prop.py \
        --dataset ruletaker_raw \
        --root data/rule-reasoning-dataset-V2020.2.5.0/original \
        --train-depths 1,2 \
        --test-depths 1,2,3,5 \
        --train-qdeps 1,2 \
        --test-qdeps 1,2,3,4,5 \
        --max-sents 24 \
        --max-len 192 \
        "${common[@]}" \
        > "$log_file" 2>&1
      ;;
    prontoqa)
      CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/transformer_trua_prop.py \
        --dataset prontoqa \
        --root data/prontoqa_ood/processed/generated_ood_data \
        --max-sents 24 \
        --max-len 192 \
        "${common[@]}" \
        > "$log_file" 2>&1
      ;;
    *)
      echo "unknown prop kind: $kind" >&2
      return 2
      ;;
  esac
}

run_task() {
  local gpu="$1" name="$2" kind="$3" seed="$4"
  local log_file="$LOG_DIR/${name}.log"
  local out_file="$RESULT_DIR/${name}.json"
  local lock="$LOCK_DIR/gpu${gpu}.lock"
  touch "$lock"
  write_started "$name" "$kind" "$seed" "$gpu" "$out_file"
  (
    trap 'rm -f "$lock"' EXIT
    set +e
    if [[ "$kind" == "clutrr_roberta" ]]; then
      run_clutrr_roberta "$gpu" "$name" "$seed" "$log_file"
    else
      run_prop "$gpu" "$kind" "$seed" "$out_file" "$log_file"
    fi
    rc=$?
    if [[ "$rc" -eq 0 ]]; then
      { cat "$STATUS_DIR/${name}.started"; echo "finished_at=$(date '+%F %T')"; tail -n 160 "$log_file"; } > "$STATUS_DIR/${name}.done"
      log "finished $name on GPU $gpu"
    else
      { cat "$STATUS_DIR/${name}.started"; echo "failed_at=$(date '+%F %T')"; echo "exit_code=$rc"; tail -n 220 "$log_file"; } > "$STATUS_DIR/${name}.failed"
      log "failed $name on GPU $gpu rc=$rc"
    fi
    exit 0
  ) &
  log "launched $name kind=$kind seed=$seed on GPU $gpu pid=$!"
}

aggregate_prop_results() {
  python3 - "$RUN_ROOT" <<'PY'
import json
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

run_root = Path(sys.argv[1])
result_dir = run_root / "results"
out_dir = Path("/root/TRUA/docs/aggregated_results")
out_dir.mkdir(parents=True, exist_ok=True)

def fmt(vals):
    vals = [float(v) for v in vals]
    if not vals:
        return "-"
    mean = sum(vals) / len(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{mean:.4f} +/- {sd:.4f}"

groups = defaultdict(list)
name_re = re.compile(r"(.+)_seed(\d+)$")
for path in sorted(result_dir.glob("*.json")):
    match = name_re.match(path.stem)
    if not match:
        continue
    key, seed = match.groups()
    data = json.loads(path.read_text())
    data["_seed"] = int(seed)
    data["_path"] = str(path)
    groups[key].append(data)

rows = []
for key, items in sorted(groups.items()):
    items = sorted(items, key=lambda x: x["_seed"])
    split_names = sorted({split for item in items for split in item.get("results", {})})
    split_rows = []
    for split in split_names:
        split_items = [item["results"][split] for item in items if split in item.get("results", {})]
        by_depth_keys = sorted({d for item in split_items for d in item.get("by_depth", {})}, key=lambda x: (x == "-1", int(x) if x.lstrip("-").isdigit() else 999))
        split_rows.append({
            "split": split,
            "accuracy": fmt([item.get("accuracy", 0.0) for item in split_items]),
            "trace_top1": fmt([item.get("trace_top1", 0.0) for item in split_items]),
            "total": [item.get("total", 0) for item in split_items],
            "by_depth": {d: fmt([item.get("by_depth", {}).get(d, 0.0) for item in split_items if d in item.get("by_depth", {})]) for d in by_depth_keys},
        })
    rows.append({
        "name": key,
        "seeds": [item["_seed"] for item in items],
        "n": len(items),
        "train": [item.get("train") for item in items],
        "splits": split_rows,
        "files": [item["_path"] for item in items],
    })

payload = {
    "run_root": str(run_root),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "rows": rows,
}
(out_dir / "UNIFIED_CORE_PROP_REGRESSION.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = [
    "# Unified Core Proposition Regression",
    "",
    f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} from `{run_root}`.",
    "",
    "Values are `mean +/- sample-std` across completed seeds.",
    "",
    "| Task | Seeds | Split | Accuracy | Trace Top-1 | By Depth |",
    "| --- | ---: | --- | ---: | ---: | --- |",
]
for row in rows:
    for split in row["splits"]:
        by_depth = ", ".join(f"{k}: {v}" for k, v in split["by_depth"].items()) or "-"
        lines.append(
            f"| {row['name']} | {row['n']} | {split['split']} | {split['accuracy']} | {split['trace_top1']} | {by_depth} |"
        )
lines.append("")
(out_dir / "UNIFIED_CORE_PROP_REGRESSION.md").write_text("\n".join(lines), encoding="utf-8")
print("Wrote unified core proposition aggregate.")
PY
}

total_tasks=$(( $(wc -l < "$QUEUE") - 1 ))
log "Unified-core regression queue started. total_tasks=$total_tasks run_root=$RUN_ROOT gpus=$GPUS threshold=${GPU_MEM_THRESHOLD_MB}MB"

while [[ "$(started_count)" -lt "$total_tasks" ]]; do
  gpu="$(free_gpu || true)"
  if [[ -z "$gpu" ]]; then
    log "No free GPU among $GPUS below ${GPU_MEM_THRESHOLD_MB}MB; waiting ${WAIT_SECONDS}s."
    sleep "$WAIT_SECONDS"
    continue
  fi

  launched=0
  while IFS=$'\t' read -r name kind seed; do
    [[ "$name" == "name" ]] && continue
    [[ -f "$STATUS_DIR/${name}.started" ]] && continue
    run_task "$gpu" "$name" "$kind" "$seed"
    launched=1
    break
  done < "$QUEUE"

  [[ "$launched" -eq 0 ]] && log "No launchable task found; waiting ${WAIT_SECONDS}s."
  sleep 30
done

log "All queue entries dispatched. Waiting for completion."
while [[ "$(finished_count)" -lt "$total_tasks" ]]; do
  log "Progress: $(finished_count)/$total_tasks finished."
  sleep "$WAIT_SECONDS"
done

python3 scripts/aggregate_clutrr_perhop.py \
  --run-root "$RUN_ROOT" \
  --name CLUTRR_ROBERTA_UNIFIED_CORE_REGRESSION \
  >> "$RUN_ROOT/supervisor.log" 2>&1 || true
aggregate_prop_results >> "$RUN_ROOT/supervisor.log" 2>&1 || true
log "Unified-core regression queue finished."
