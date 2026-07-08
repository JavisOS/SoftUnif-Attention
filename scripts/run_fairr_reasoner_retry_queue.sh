#!/usr/bin/env bash
set -euo pipefail

cd /root/TRUA

MODEL_ROOT="/vepfs/tsra_models/hf"
RUN_ROOT="/vepfs/tsra_outputs/official_external"
LOG="$RUN_ROOT/logs/fairr_reasoner_official_t5large_retry_torch_adafactor.log"
mkdir -p "$RUN_ROOT/logs"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRUA_T5_LARGE_PATH="$MODEL_ROOT/t5-large"

choose_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '$2 + 0 < 1000 {gsub(/ /, "", $1); print $1; exit}'
}

echo "Queued FaiRR reasoner retry at $(date)" > "$RUN_ROOT/logs/fairr_reasoner_retry_queue.log"
GPU_ID=""
while [[ -z "$GPU_ID" ]]; do
  GPU_ID="$(choose_gpu || true)"
  if [[ -z "$GPU_ID" ]]; then
    echo "[$(date)] No free GPU yet for FaiRR reasoner retry; waiting 120s..." >> "$RUN_ROOT/logs/fairr_reasoner_retry_queue.log"
    sleep 120
  fi
done

echo "[$(date)] Starting FaiRR reasoner retry on GPU $GPU_ID" >> "$RUN_ROOT/logs/fairr_reasoner_retry_queue.log"
(
  cd external_baselines/FaiRR
  CUDA_VISIBLE_DEVICES="$GPU_ID" python main.py \
    --override fairr_reasoner \
    --dataset pw_leq_0to3_OWA_reasoner \
    --hf_name "$MODEL_ROOT/t5-large" \
    --gpus 1 \
    --max_epochs 5 \
    --train_batch_size 4 \
    --eval_batch_size 8 \
    --accumulate_grad_batches 4 \
    --save_checkpoint
) > "$LOG" 2>&1

echo "[$(date)] FaiRR reasoner retry finished." >> "$RUN_ROOT/logs/fairr_reasoner_retry_queue.log"
