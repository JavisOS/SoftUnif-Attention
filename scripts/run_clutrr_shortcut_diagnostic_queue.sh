#!/usr/bin/env bash
set -euo pipefail

cd /root/TRUA

RUN_ROOT="/vepfs/tsra_outputs/clutrr_shortcut_diagnostic/clutrr_shortcut_diag_$(date +%Y%m%d_%H%M%S)"
LATEST="/vepfs/tsra_outputs/clutrr_shortcut_diagnostic/latest"
QUEUE="$RUN_ROOT/queue.tsv"
LOG_DIR="$RUN_ROOT/logs"
RESULT_DIR="$RUN_ROOT/results"
STATUS_DIR="$RUN_ROOT/status"
GPUS="${TRUA_SHORTCUT_DIAG_GPUS:-0,1,2}"
GPU_MEM_THRESHOLD_MB="${TRUA_GPU_MEM_THRESHOLD_MB:-1000}"

mkdir -p "$LOG_DIR" "$RESULT_DIR" "$STATUS_DIR"
ln -sfn "$RUN_ROOT" "$LATEST"

export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

DEBERTA_V3="/vepfs/tsra_models/hf/deberta-v3-base"

cat > "$QUEUE" <<EOF
name	variant	gpu	model_type	model_path	seed	epochs
shortcut_diag_vanilla_debertav3_seed0	vanilla	0	deberta-v3	$DEBERTA_V3	0	3
shortcut_diag_crest_debertav3_seed0	crest	1	deberta-v3	$DEBERTA_V3	0	3
shortcut_diag_trua_debertav3_seed0	tsra	2	deberta-v3	$DEBERTA_V3	0	3
EOF

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$RUN_ROOT/supervisor.log"
}

wait_gpu() {
  local gpu="$1"
  while true; do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
    if [[ "${used:-999999}" -lt "$GPU_MEM_THRESHOLD_MB" ]]; then
      return 0
    fi
    log "GPU $gpu has ${used}MB used; waiting."
    sleep 60
  done
}

run_task() {
  local name="$1" variant="$2" gpu="$3" model_type="$4" model_path="$5" seed="$6" epochs="$7"
  local out_file="$RESULT_DIR/${name}.json"
  local log_file="$LOG_DIR/${name}.log"
  local started="$STATUS_DIR/${name}.started"
  local done="$STATUS_DIR/${name}.done"
  local failed="$STATUS_DIR/${name}.failed"
  wait_gpu "$gpu"
  {
    echo "task=$name"
    echo "variant=$variant"
    echo "dataset=data_089907f8"
    echo "model_type=$model_type"
    echo "model_path=$model_path"
    echo "seed=$seed"
    echo "epochs=$epochs"
    echo "gpu=$gpu"
    echo "diagnostic=pilot_token_occlusion_shortcut_reasoning"
    echo "out=$out_file"
    echo "started_at=$(date '+%F %T')"
  } > "$started"
  (
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" python3 -u scripts/clutrr_shortcut_diagnostic.py \
      --variant "$variant" \
      --dataset data_089907f8 \
      --root data \
      --model_type "$model_type" \
      --model_name_or_path "$model_path" \
      --seed "$seed" \
      --epochs "$epochs" \
      --batch_size 16 \
      --eval_batch_size 64 \
      --extract_n 120 \
      --pattern_top_k 1 \
      --out "$out_file" \
      > "$log_file" 2>&1
    rc=$?
    if [[ "$rc" -eq 0 ]]; then
      { cat "$started"; echo "finished_at=$(date '+%F %T')"; tail -n 160 "$log_file"; } > "$done"
    else
      { cat "$started"; echo "failed_at=$(date '+%F %T')"; echo "exit_code=$rc"; tail -n 220 "$log_file"; } > "$failed"
    fi
    exit "$rc"
  ) &
  log "Launched $name on GPU $gpu pid=$!"
}

total_tasks=$(( $(wc -l < "$QUEUE") - 1 ))
log "CLUTRR shortcut diagnostic queue started. run_root=$RUN_ROOT total_tasks=$total_tasks"

while IFS=$'\t' read -r name variant gpu model_type model_path seed epochs; do
  [[ "$name" == "name" ]] && continue
  run_task "$name" "$variant" "$gpu" "$model_type" "$model_path" "$seed" "$epochs"
done < "$QUEUE"

while true; do
  done_count="$(find "$STATUS_DIR" -name "*.done" -type f | wc -l)"
  failed_count="$(find "$STATUS_DIR" -name "*.failed" -type f | wc -l)"
  finished=$((done_count + failed_count))
  log "Progress: $finished/$total_tasks finished; failed=$failed_count"
  [[ "$finished" -ge "$total_tasks" ]] && break
  sleep 120
done

python3 scripts/aggregate_clutrr_shortcut_diagnostic.py --run-root "$RUN_ROOT" >> "$RUN_ROOT/supervisor.log" 2>&1
log "Aggregation finished."
