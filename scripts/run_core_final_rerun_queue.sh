#!/usr/bin/env bash
set -euo pipefail

cd /root/TRUA

RUN_ROOT="/vepfs/tsra_outputs/core_final_rerun/core_final_rerun_$(date +%Y%m%d_%H%M%S)"
LATEST="/vepfs/tsra_outputs/core_final_rerun/latest"
QUEUE="$RUN_ROOT/queue.tsv"
LOG_DIR="$RUN_ROOT/logs"
RESULT_DIR="$RUN_ROOT/results"
STATUS_DIR="$RUN_ROOT/status"
LOCK_DIR="$RUN_ROOT/gpu_locks"
GPUS="${TRUA_CORE_FINAL_GPUS:-0,1,2,3,4,5,7}"
GPU_MEM_THRESHOLD_MB="${TRUA_GPU_MEM_THRESHOLD_MB:-1000}"
WAIT_SECONDS="${TRUA_QUEUE_WAIT_SECONDS:-120}"

mkdir -p "$LOG_DIR" "$RESULT_DIR" "$STATUS_DIR" "$LOCK_DIR"
ln -sfn "$RUN_ROOT" "$LATEST"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

BERT="/vepfs/tsra_models/hf/bert-base-uncased"
ROBERTA="/vepfs/tsra_models/hf/roberta-base-clean-current"
DEBERTA="/vepfs/tsra_models/hf/deberta-base"
DEBERTA_V3="/vepfs/tsra_models/hf/deberta-v3-base"

cat > "$QUEUE" <<EOF
name	kind	backbone	seed
proofwriter_bert_trua_corefinal_seed0	proofwriter	bert	0
proofwriter_bert_trua_corefinal_seed1	proofwriter	bert	1
proofwriter_bert_trua_corefinal_seed42	proofwriter	bert	42
clutrr_089_deberta_tsra_seed0	clutrr_089	deberta	0
clutrr_089_deberta_tsra_seed1	clutrr_089	deberta	1
clutrr_089_deberta_tsra_seed42	clutrr_089	deberta	42
clutrr_089_deberta-v3_tsra_seed0	clutrr_089	deberta-v3	0
clutrr_089_deberta-v3_tsra_seed1	clutrr_089	deberta-v3	1
clutrr_089_deberta-v3_tsra_seed42	clutrr_089	deberta-v3	42
clutrr_db9_deberta-v3_tsra_seed0	clutrr_db9	deberta-v3	0
clutrr_db9_deberta-v3_tsra_seed1	clutrr_db9	deberta-v3	1
clutrr_db9_deberta-v3_tsra_seed42	clutrr_db9	deberta-v3	42
ruletaker_gfair_bert_trua_corefinal_seed0	ruletaker_gfair	bert	0
ruletaker_gfair_bert_trua_corefinal_seed1	ruletaker_gfair	bert	1
ruletaker_gfair_bert_trua_corefinal_seed42	ruletaker_gfair	bert	42
ruletaker_gfair_roberta_trua_corefinal_seed0	ruletaker_gfair	roberta	0
ruletaker_gfair_roberta_trua_corefinal_seed1	ruletaker_gfair	roberta	1
ruletaker_gfair_roberta_trua_corefinal_seed42	ruletaker_gfair	roberta	42
ruletaker_gfair_deberta_trua_corefinal_seed0	ruletaker_gfair	deberta	0
ruletaker_gfair_deberta_trua_corefinal_seed1	ruletaker_gfair	deberta	1
ruletaker_gfair_deberta_trua_corefinal_seed42	ruletaker_gfair	deberta	42
prontoqa_bert_trua_corefinal_seed0	prontoqa	bert	0
prontoqa_bert_trua_corefinal_seed1	prontoqa	bert	1
prontoqa_bert_trua_corefinal_seed42	prontoqa	bert	42
prontoqa_roberta_trua_corefinal_seed0	prontoqa	roberta	0
prontoqa_roberta_trua_corefinal_seed1	prontoqa	roberta	1
prontoqa_roberta_trua_corefinal_seed42	prontoqa	roberta	42
proofwriter_roberta_trua_corefinal_seed0	proofwriter	roberta	0
proofwriter_roberta_trua_corefinal_seed1	proofwriter	roberta	1
proofwriter_roberta_trua_corefinal_seed42	proofwriter	roberta	42
EOF

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$RUN_ROOT/supervisor.log"
}

model_path() {
  case "$1" in
    bert) echo "$BERT" ;;
    roberta) echo "$ROBERTA" ;;
    deberta) echo "$DEBERTA" ;;
    deberta-v3) echo "$DEBERTA_V3" ;;
    *) echo "unknown backbone: $1" >&2; return 2 ;;
  esac
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
  local name="$1" kind="$2" backbone="$3" seed="$4" gpu="$5" out_file="$6"
  {
    echo "task=$name"
    echo "kind=$kind"
    echo "backbone=$backbone"
    echo "seed=$seed"
    echo "epochs=10"
    echo "gpu=$gpu"
    echo "code_state=unified_core_final_rerun"
    echo "started_at=$(date '+%F %T')"
    echo "out=$out_file"
  } > "$STATUS_DIR/${name}.started"
}

run_clutrr() {
  local gpu="$1" name="$2" kind="$3" backbone="$4" seed="$5" log_file="$6"
  local dataset model
  model="$(model_path "$backbone")"
  if [[ "$kind" == "clutrr_089" ]]; then
    dataset="data_089907f8"
  elif [[ "$kind" == "clutrr_db9" ]]; then
    dataset="data_db9b8f04"
  else
    echo "unknown clutrr kind: $kind" >&2
    return 2
  fi
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m clutrr.cli.train \
    --config configs/clutrr/train_trua.yaml \
    --dataset "$dataset" \
    --root data \
    --model_type "$backbone" \
    --model_name_or_path "$model" \
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
  local gpu="$1" kind="$2" backbone="$3" seed="$4" out_file="$5" log_file="$6"
  local model
  model="$(model_path "$backbone")"
  local common=(--model-name "$model" --seed "$seed" --epochs 10 --limit-train 0 --limit-test 0 --batch-size 16 --lr 2e-5 --lambda-trace 1.0 --out "$out_file")
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
    ruletaker_gfair)
      CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/transformer_trua_prop.py \
        --dataset ruletaker_gfair \
        --root external_baselines/GFaiR/data/ruletaker_3ext_sat \
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
  local gpu="$1" name="$2" kind="$3" backbone="$4" seed="$5"
  local log_file="$LOG_DIR/${name}.log"
  local out_file="$RESULT_DIR/${name}.json"
  local lock="$LOCK_DIR/gpu${gpu}.lock"
  touch "$lock"
  write_started "$name" "$kind" "$backbone" "$seed" "$gpu" "$out_file"
  (
    trap 'rm -f "$lock"' EXIT
    set +e
    if [[ "$kind" == clutrr_* ]]; then
      run_clutrr "$gpu" "$name" "$kind" "$backbone" "$seed" "$log_file"
    else
      run_prop "$gpu" "$kind" "$backbone" "$seed" "$out_file" "$log_file"
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
  log "launched $name kind=$kind backbone=$backbone seed=$seed on GPU $gpu pid=$!"
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
        by_depth_keys = sorted(
            {d for item in split_items for d in item.get("by_depth", {})},
            key=lambda x: int(x) if str(x).lstrip("-").isdigit() else 999,
        )
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
(out_dir / "CORE_FINAL_PROP_RERUN.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = [
    "# Core-Final Proposition TRUA Rerun",
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
(out_dir / "CORE_FINAL_PROP_RERUN.md").write_text("\n".join(lines), encoding="utf-8")
print("Wrote core-final proposition aggregate.")
PY
}

total_tasks=$(( $(wc -l < "$QUEUE") - 1 ))
log "Core-final targeted rerun queue started. total_tasks=$total_tasks run_root=$RUN_ROOT gpus=$GPUS threshold=${GPU_MEM_THRESHOLD_MB}MB"

while [[ "$(started_count)" -lt "$total_tasks" ]]; do
  gpu="$(free_gpu || true)"
  if [[ -z "$gpu" ]]; then
    log "No free GPU among $GPUS below ${GPU_MEM_THRESHOLD_MB}MB; waiting ${WAIT_SECONDS}s."
    sleep "$WAIT_SECONDS"
    continue
  fi

  launched=0
  while IFS=$'\t' read -r name kind backbone seed; do
    [[ "$name" == "name" ]] && continue
    [[ -f "$STATUS_DIR/${name}.started" ]] && continue
    run_task "$gpu" "$name" "$kind" "$backbone" "$seed"
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
  --name CORE_FINAL_CLUTRR_RERUN \
  >> "$RUN_ROOT/supervisor.log" 2>&1 || true
aggregate_prop_results >> "$RUN_ROOT/supervisor.log" 2>&1 || true
log "Core-final targeted rerun queue finished."
