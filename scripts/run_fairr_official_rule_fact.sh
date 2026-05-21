#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_ROOT="/vepfs/tsra_models/hf"
RUN_ROOT="/vepfs/tsra_outputs/official_external"
mkdir -p "$RUN_ROOT/logs"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TSRA_ROBERTA_LARGE_PATH="$MODEL_ROOT/roberta-large"

if [[ ! -f "$MODEL_ROOT/roberta-large/config.json" ]]; then
  echo "Missing roberta-large under $MODEL_ROOT/roberta-large" >&2
  exit 2
fi

echo "[1/4] FaiRR official preprocessing: rule selector"
(
  cd external_baselines/FaiRR
  python process_proofwriter.py --dataset pwq_leq_0to3 --fairr_model fairr_rule --arch roberta_large
) > "$RUN_ROOT/logs/fairr_preprocess_rule_official_roberta.log" 2>&1

echo "[2/4] FaiRR official preprocessing: fact selector"
(
  cd external_baselines/FaiRR
  python process_proofwriter.py --dataset pwq_leq_0to3 --fairr_model fairr_fact --arch roberta_large
) > "$RUN_ROOT/logs/fairr_preprocess_fact_official_roberta.log" 2>&1

: > "$RUN_ROOT/fairr_rule_fact_pids.txt"

echo "[3/4] Launch FaiRR rule selector"
(
  cd external_baselines/FaiRR
  CUDA_VISIBLE_DEVICES=4 python main.py --override fairr_ruleselector,pwq_leq_0to3_OWA_rule --hf_name "$MODEL_ROOT/roberta-large" --gpus 1 --max_epochs 5 --train_batch_size 16 --eval_batch_size 32 --save_checkpoint
) > "$RUN_ROOT/logs/fairr_ruleselector_official_roberta_train.log" 2>&1 &
echo "$! fairr_ruleselector" >> "$RUN_ROOT/fairr_rule_fact_pids.txt"

echo "[4/4] Launch FaiRR fact selector"
(
  cd external_baselines/FaiRR
  CUDA_VISIBLE_DEVICES=5 python main.py --override fairr_factselector,pwq_leq_0to3_OWA_fact --hf_name "$MODEL_ROOT/roberta-large" --gpus 1 --max_epochs 5 --train_batch_size 16 --eval_batch_size 32 --save_checkpoint
) > "$RUN_ROOT/logs/fairr_factselector_official_roberta_train.log" 2>&1 &
echo "$! fairr_factselector" >> "$RUN_ROOT/fairr_rule_fact_pids.txt"

echo "Launched FaiRR official rule/fact modules. PID file: $RUN_ROOT/fairr_rule_fact_pids.txt"
