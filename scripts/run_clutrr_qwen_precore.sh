#!/usr/bin/env bash
set -euo pipefail

cd /root/TRUA

RUN_ROOT="/vepfs/tsra_outputs/clutrr_qwen_precore/clutrr_qwen_precore_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$RUN_ROOT/logs"
STATUS_DIR="$RUN_ROOT/status"
mkdir -p "$LOG_DIR" "$STATUS_DIR"
ln -sfn "$RUN_ROOT" /vepfs/tsra_outputs/clutrr_qwen_precore/latest

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-/tos/hgf/models/Qwen3/Qwen3-0.6B-Base}"
MODEL_TYPE="${MODEL_TYPE:-qwen3-0.6b-base}"
GPU_LIST="${GPU_LIST:-2,3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
MEMORY_LIMIT_MB="${MEMORY_LIMIT_MB:-1000}"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
SEEDS=(0 1 42)

gpu_memory_used_mb() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print $1}'
}

pick_free_gpu() {
  local gpu used
  for gpu in "${GPUS[@]}"; do
    if [[ -e "$STATUS_DIR/gpu_${gpu}.lock" ]]; then
      continue
    fi
    used="$(gpu_memory_used_mb "$gpu" 2>/dev/null || echo 999999)"
    if [[ "$used" =~ ^[0-9]+$ ]] && [[ "$used" -lt "$MEMORY_LIMIT_MB" ]]; then
      echo "$gpu"
      return 0
    fi
  done
  return 1
}

echo "[$(date '+%F %T')] Qwen pre-core queue started. model_type=$MODEL_TYPE model=$MODEL gpus=$GPU_LIST batch=$BATCH_SIZE" \
  | tee -a "$RUN_ROOT/supervisor.log"

for seed in "${SEEDS[@]}"; do
  name="clutrr_089_${MODEL_TYPE//./_}_trua_seed${seed}"
  started="$STATUS_DIR/${name}.started"
  done="$STATUS_DIR/${name}.done"
  failed="$STATUS_DIR/${name}.failed"
  log_file="$LOG_DIR/${name}.log"

  while true; do
    if gpu="$(pick_free_gpu)"; then
      break
    fi
    echo "[$(date '+%F %T')] No unlocked free GPU among $GPU_LIST below ${MEMORY_LIMIT_MB}MB; waiting." \
      | tee -a "$RUN_ROOT/supervisor.log"
    sleep 60
  done
  gpu_lock="$STATUS_DIR/gpu_${gpu}.lock"
  echo "$name" > "$gpu_lock"

  {
    echo "task=$name"
    echo "dataset=data_089907f8"
    echo "model_type=$MODEL_TYPE"
    echo "model_path=$MODEL"
    echo "variant=trua"
    echo "seed=$seed"
    echo "epochs=10"
    echo "batch_size=$BATCH_SIZE"
    echo "eval_batch_size=$EVAL_BATCH_SIZE"
    echo "gpu=$gpu"
    echo "code_state=pre_unified_core_trua_model_precore"
    echo "started_at=$(date '+%F %T')"
  } > "$started"

  (
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m clutrr.cli.train_precore \
      --config configs/clutrr/train_trua.yaml \
      --dataset data_089907f8 \
      --root data \
      --model_type "$MODEL_TYPE" \
      --model_name_or_path "$MODEL" \
      --epochs 10 \
      --batch_size "$BATCH_SIZE" \
      --eval_batch_size "$EVAL_BATCH_SIZE" \
      --gpus "$gpu" \
      --strategy single \
      --seed "$seed" \
      --pooling last_token \
      --lambda_nexthop 1.0 \
      --lambda_edge 1.0 \
      --lambda_consistency 5.0 \
      > "$log_file" 2>&1
    rc=$?
    if [[ "$rc" -eq 0 ]]; then
      { cat "$started"; echo "finished_at=$(date '+%F %T')"; tail -n 180 "$log_file"; } > "$done"
    else
      { cat "$started"; echo "failed_at=$(date '+%F %T')"; echo "exit_code=$rc"; tail -n 260 "$log_file"; } > "$failed"
    fi
    rm -f "$gpu_lock"
    exit "$rc"
  ) &
  echo "[$(date '+%F %T')] launched $name on GPU $gpu pid=$!" | tee -a "$RUN_ROOT/supervisor.log"
  sleep 20
done

wait

python3 scripts/aggregate_clutrr_perhop.py \
  --run-root "$RUN_ROOT" \
  --name CLUTRR_QWEN_PRECORE \
  >> "$RUN_ROOT/supervisor.log" 2>&1 || true
echo "[$(date '+%F %T')] finished Qwen pre-core queue" | tee -a "$RUN_ROOT/supervisor.log"
