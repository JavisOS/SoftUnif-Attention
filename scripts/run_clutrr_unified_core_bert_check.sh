#!/usr/bin/env bash
set -euo pipefail

cd /root/TRUA

RUN_ROOT="/vepfs/tsra_outputs/clutrr_unified_core_bert_check/clutrr_unified_core_bert_check_$(date +%Y%m%d_%H%M%S)"
QUEUE="$RUN_ROOT/queue.tsv"
STATUS_DIR="$RUN_ROOT/status"
LOG_DIR="$RUN_ROOT/logs"
GPUS="${TRUA_UNIFIED_CORE_GPUS:-4,5,6,7}"
GPU_MEM_THRESHOLD_MB="${TRUA_GPU_MEM_THRESHOLD_MB:-1000}"

mkdir -p "$STATUS_DIR" "$LOG_DIR"
ln -sfn "$RUN_ROOT" /vepfs/tsra_outputs/clutrr_unified_core_bert_check/latest

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

BERT="/vepfs/tsra_models/hf/bert-base-uncased"

cat > "$QUEUE" <<EOF
name	seed
clutrr_089_bert_trua_seed0	0
clutrr_089_bert_trua_seed1	1
clutrr_089_bert_trua_seed42	42
EOF

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$RUN_ROOT/supervisor.log"
}

free_gpu() {
  local gpu used
  IFS=',' read -ra candidates <<< "$GPUS"
  for gpu in "${candidates[@]}"; do
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

run_task() {
  local gpu="$1" name="$2" seed="$3"
  local started="$STATUS_DIR/${name}.started"
  local done="$STATUS_DIR/${name}.done"
  local failed="$STATUS_DIR/${name}.failed"
  local log_file="$LOG_DIR/${name}.log"
  {
    echo "task=$name"
    echo "dataset=data_089907f8"
    echo "model_type=bert"
    echo "model_path=$BERT"
    echo "variant=trua"
    echo "seed=$seed"
    echo "epochs=10"
    echo "gpu=$gpu"
    echo "code_state=unified_core_entity_path_adapter"
    echo "started_at=$(date '+%F %T')"
  } > "$started"
  (
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m clutrr.cli.train \
      --config configs/clutrr/train_trua.yaml \
      --dataset data_089907f8 \
      --root data \
      --model_type bert \
      --model_name_or_path "$BERT" \
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
    rc=$?
    if [[ "$rc" -eq 0 ]]; then
      { cat "$started"; echo "finished_at=$(date '+%F %T')"; tail -n 160 "$log_file"; } > "$done"
    else
      { cat "$started"; echo "failed_at=$(date '+%F %T')"; echo "exit_code=$rc"; tail -n 220 "$log_file"; } > "$failed"
    fi
    exit "$rc"
  ) &
  log "launched $name on GPU $gpu pid=$!"
}

total_tasks=$(( $(wc -l < "$QUEUE") - 1 ))
log "Unified-core BERT check started. total_tasks=$total_tasks run_root=$RUN_ROOT gpus=$GPUS"

while [[ "$(started_count)" -lt "$total_tasks" ]]; do
  gpu="$(free_gpu || true)"
  if [[ -z "$gpu" ]]; then
    log "No free GPU among $GPUS below ${GPU_MEM_THRESHOLD_MB}MB; waiting."
    sleep 60
    continue
  fi
  launched=0
  while IFS=$'\t' read -r name seed; do
    [[ "$name" == "name" ]] && continue
    [[ -f "$STATUS_DIR/${name}.started" ]] && continue
    run_task "$gpu" "$name" "$seed"
    launched=1
    break
  done < "$QUEUE"
  [[ "$launched" -eq 0 ]] && log "No launchable task found; waiting."
  sleep 45
done

log "All queue entries dispatched. Waiting for completion."
while [[ "$(finished_count)" -lt "$total_tasks" ]]; do
  log "Progress: $(finished_count)/$total_tasks finished."
  sleep 120
done

python3 scripts/aggregate_clutrr_perhop.py \
  --run-root "$RUN_ROOT" \
  --name CLUTRR_UNIFIED_CORE_BERT_CHECK \
  >> "$RUN_ROOT/supervisor.log" 2>&1 || true
log "Unified-core BERT check aggregation finished."
