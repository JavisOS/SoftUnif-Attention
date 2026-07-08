#!/usr/bin/env bash
set -euo pipefail

cd /root/TRUA

RUN_ROOT="/vepfs/tsra_outputs/clutrr_gpt2_precore/clutrr_gpt2_precore_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$RUN_ROOT/logs"
STATUS_DIR="$RUN_ROOT/status"
mkdir -p "$LOG_DIR" "$STATUS_DIR"
ln -sfn "$RUN_ROOT" /vepfs/tsra_outputs/clutrr_gpt2_precore/latest

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MODEL="/vepfs/tsra_models/hf/openai-community-gpt2"
GPUS=(4 5 6)
SEEDS=(0 1 42)

for idx in "${!SEEDS[@]}"; do
  seed="${SEEDS[$idx]}"
  gpu="${GPUS[$idx]}"
  name="clutrr_089_gpt2_trua_seed${seed}"
  started="$STATUS_DIR/${name}.started"
  done="$STATUS_DIR/${name}.done"
  failed="$STATUS_DIR/${name}.failed"
  log_file="$LOG_DIR/${name}.log"
  {
    echo "task=$name"
    echo "dataset=data_089907f8"
    echo "model_type=gpt2"
    echo "model_path=$MODEL"
    echo "variant=trua"
    echo "seed=$seed"
    echo "epochs=10"
    echo "gpu=$gpu"
    echo "code_state=pre_unified_core_trua_path"
    echo "started_at=$(date '+%F %T')"
  } > "$started"
  (
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m clutrr.cli.train \
      --config configs/clutrr/train_trua.yaml \
      --dataset data_089907f8 \
      --root data \
      --model_type gpt2 \
      --model_name_or_path "$MODEL" \
      --epochs 10 \
      --batch_size 16 \
      --eval_batch_size 32 \
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
      { cat "$started"; echo "finished_at=$(date '+%F %T')"; tail -n 160 "$log_file"; } > "$done"
    else
      { cat "$started"; echo "failed_at=$(date '+%F %T')"; echo "exit_code=$rc"; tail -n 220 "$log_file"; } > "$failed"
    fi
    exit "$rc"
  ) &
  echo "[$(date '+%F %T')] launched $name on GPU $gpu pid=$!" | tee -a "$RUN_ROOT/supervisor.log"
done

wait

python3 scripts/aggregate_clutrr_perhop.py \
  --run-root "$RUN_ROOT" \
  --name CLUTRR_GPT2_PRECORE \
  >> "$RUN_ROOT/supervisor.log" 2>&1 || true
echo "[$(date '+%F %T')] finished GPT2 pre-core queue" | tee -a "$RUN_ROOT/supervisor.log"
