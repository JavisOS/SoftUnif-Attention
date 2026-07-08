#!/usr/bin/env bash
set -euo pipefail

cd /root/TRUA

MODEL="${MODEL:?Set MODEL to a local HuggingFace checkpoint path.}"
MODEL_TYPE="${MODEL_TYPE:?Set MODEL_TYPE, e.g. qwen3-1.7b-base.}"
RUN_NAME="${RUN_NAME:-clutrr_decoder_directional_precore}"
OUTPUT_BASE="${OUTPUT_BASE:-/vepfs/tsra_outputs/${RUN_NAME}}"
RUN_ROOT="${OUTPUT_BASE}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$RUN_ROOT/logs"
STATUS_DIR="$RUN_ROOT/status"
mkdir -p "$LOG_DIR" "$STATUS_DIR"
ln -sfn "$RUN_ROOT" "${OUTPUT_BASE}/latest"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

GPU_LIST="${GPU_LIST:-0,1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
MEMORY_LIMIT_MB="${MEMORY_LIMIT_MB:-1000}"
USE_QLORA="${USE_QLORA:-0}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-0}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
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

safe_model_type="${MODEL_TYPE//./_}"
safe_model_type="${safe_model_type//\//_}"

extra_args=()
if [[ "$USE_QLORA" == "1" ]]; then
  extra_args+=(--use_qlora --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT")
fi
if [[ "$LOAD_IN_4BIT" == "1" ]]; then
  extra_args+=(--load_in_4bit)
fi

echo "[$(date '+%F %T')] Decoder directional pre-core queue started. model_type=$MODEL_TYPE model=$MODEL gpus=$GPU_LIST batch=$BATCH_SIZE qlora=$USE_QLORA 4bit=$LOAD_IN_4BIT" \
  | tee -a "$RUN_ROOT/supervisor.log"

for seed in "${SEEDS[@]}"; do
  name="clutrr_089_${safe_model_type}_trua_seed${seed}"
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
    echo "use_qlora=$USE_QLORA"
    echo "load_in_4bit=$LOAD_IN_4BIT"
    echo "gpu=$gpu"
    echo "code_state=pre_unified_core_trua_model_precore_directed_decoder_prompt"
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
      "${extra_args[@]}" \
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
  --name "$(echo "$RUN_NAME" | tr '[:lower:]' '[:upper:]')" \
  >> "$RUN_ROOT/supervisor.log" 2>&1 || true
echo "[$(date '+%F %T')] finished decoder directional pre-core queue" | tee -a "$RUN_ROOT/supervisor.log"
