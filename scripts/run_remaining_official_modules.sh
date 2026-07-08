#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL_ROOT="/vepfs/tsra_models/hf"
RUN_ROOT="/vepfs/tsra_outputs/official_external"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/gfair/out_selector2_ruletaker"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRUA_T5_LARGE_PATH="$MODEL_ROOT/t5-large"
export TRUA_ROBERTA_LARGE_PATH="$MODEL_ROOT/roberta-large"
: > "$RUN_ROOT/remaining_modules_pids.txt"

echo "[1/3] FaiRR official preprocessing: reasoner"
(
  cd external_baselines/FaiRR
  python process_proofwriter.py --dataset pw_leq_0to3 --fairr_model fairr_reasoner --arch t5_large
) > "$RUN_ROOT/logs/fairr_preprocess_reasoner_official_t5large.log" 2>&1

echo "[2/3] Launch FaiRR reasoner"
(
  cd external_baselines/FaiRR
  CUDA_VISIBLE_DEVICES=6 python main.py --override fairr_reasoner --dataset pw_leq_0to3_OWA_reasoner --hf_name "$MODEL_ROOT/t5-large" --gpus 1 --max_epochs 5 --train_batch_size 4 --eval_batch_size 8 --accumulate_grad_batches 4 --save_checkpoint
) > "$RUN_ROOT/logs/fairr_reasoner_official_t5large_train.log" 2>&1 &
echo "$! fairr_reasoner" >> "$RUN_ROOT/remaining_modules_pids.txt"

echo "[3/3] Launch GFaiR selector2/post-selector verifier"
(
  cd external_baselines/GFaiR
  CUDA_VISIBLE_DEVICES=7 python run_selector2.py --input_dir data/ruletaker_3ext_sat --output_dir "$RUN_ROOT/gfair/out_selector2_ruletaker" --model "$MODEL_ROOT/xlnet-large-cased" --do_train --num_train_epochs 30 --train_batch_size 32 --gradient_accumulation_steps 4 --alpha 0.05
) > "$RUN_ROOT/logs/gfair_selector2_official_xlnet_train.log" 2>&1 &
echo "$! gfair_selector2" >> "$RUN_ROOT/remaining_modules_pids.txt"

echo "Launched remaining official modules: $RUN_ROOT/remaining_modules_pids.txt"
