#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_ROOT="/vepfs/tsra_models/hf"
RUN_ROOT="/vepfs/tsra_outputs/official_external"
mkdir -p "$RUN_ROOT"/{fairr,gfair,logs}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TSRA_ROBERTA_LARGE_PATH="$MODEL_ROOT/roberta-large"
export TSRA_T5_LARGE_PATH="$MODEL_ROOT/t5-large"

require_model() {
  local path="$1"
  if [[ ! -f "$path/config.json" ]]; then
    echo "Missing model at $path" >&2
    exit 2
  fi
}

require_model "$MODEL_ROOT/roberta-large"
require_model "$MODEL_ROOT/t5-large"
require_model "$MODEL_ROOT/xlnet-large-cased"

mkdir -p model
ln -sfn "$MODEL_ROOT/xlnet-large-cased" model/xlnet
ln -sfn "$MODEL_ROOT/t5-large" model/T5

echo "[1/6] FaiRR official preprocessing: rule selector"
(
  cd external_baselines/FaiRR
  python process_proofwriter.py --dataset pwq_leq_0to3 --fairr_model fairr_rule --arch roberta_large
) > "$RUN_ROOT/logs/fairr_preprocess_rule.log" 2>&1

echo "[2/6] FaiRR official preprocessing: fact selector"
(
  cd external_baselines/FaiRR
  python process_proofwriter.py --dataset pwq_leq_0to3 --fairr_model fairr_fact --arch roberta_large
) > "$RUN_ROOT/logs/fairr_preprocess_fact.log" 2>&1

echo "[3/6] FaiRR official preprocessing: reasoner"
(
  cd external_baselines/FaiRR
  python process_proofwriter.py --dataset pw_leq_0to3 --fairr_model fairr_reasoner --arch t5_large
) > "$RUN_ROOT/logs/fairr_preprocess_reasoner.log" 2>&1

echo "[4/6] GFaiR official preprocessing"
(
  cd external_baselines/GFaiR/data
  python create_my_example.py --model_type all --direc ruletaker_3ext_sat/
) > "$RUN_ROOT/logs/gfair_preprocess_ruletaker_3ext_sat.log" 2>&1

echo "[5/6] Launch FaiRR official module training"
(
  cd external_baselines/FaiRR
  CUDA_VISIBLE_DEVICES=4 python main.py --override fairr_ruleselector,pwq_leq_0to3_OWA_rule --hf_name "$MODEL_ROOT/roberta-large" --gpus 1 --max_epochs 5 --train_batch_size 16 --eval_batch_size 32 --save_checkpoint
) > "$RUN_ROOT/logs/fairr_ruleselector_train.log" 2>&1 &
echo "$! fairr_ruleselector" >> "$RUN_ROOT/pids.txt"

(
  cd external_baselines/FaiRR
  CUDA_VISIBLE_DEVICES=5 python main.py --override fairr_factselector,pwq_leq_0to3_OWA_fact --hf_name "$MODEL_ROOT/roberta-large" --gpus 1 --max_epochs 5 --train_batch_size 16 --eval_batch_size 32 --save_checkpoint
) > "$RUN_ROOT/logs/fairr_factselector_train.log" 2>&1 &
echo "$! fairr_factselector" >> "$RUN_ROOT/pids.txt"

(
  cd external_baselines/FaiRR
  CUDA_VISIBLE_DEVICES=6 python main.py --override fairr_reasoner --dataset pw_leq_0to3_OWA_reasoner --hf_name "$MODEL_ROOT/t5-large" --gpus 1 --max_epochs 5 --train_batch_size 4 --eval_batch_size 8 --accumulate_grad_batches 4 --save_checkpoint
) > "$RUN_ROOT/logs/fairr_reasoner_train.log" 2>&1 &
echo "$! fairr_reasoner" >> "$RUN_ROOT/pids.txt"

echo "[6/6] Launch GFaiR official module training"
(
  cd external_baselines/GFaiR
  CUDA_VISIBLE_DEVICES=7 python run_selector2.py --input_dir data/ruletaker_3ext_sat --output_dir "$RUN_ROOT/gfair/out_selector2_ruletaker" --model "$MODEL_ROOT/xlnet-large-cased" --do_train --num_train_epochs 10 --train_batch_size 32 --gradient_accumulation_steps 4 --alpha 0.05
) > "$RUN_ROOT/logs/gfair_selector2_train.log" 2>&1 &
echo "$! gfair_selector2" >> "$RUN_ROOT/pids.txt"

echo "Launched official external pipeline modules. PID file: $RUN_ROOT/pids.txt"
