#!/usr/bin/env bash
set +e

cd /root/TRUA || exit 1

ROOT=/vepfs/tsra_outputs/recent_external_baselines/latest
DATA_DIR=/vepfs/tsra_outputs/recent_external_data/codi_prontoqa
OUT_ROOT="$ROOT/codi_prontoqa_scheme2"
STATUS="$ROOT/status/06_codi_prontoqa_scheme2"
TRAIN_LOG="$ROOT/logs/06_codi_prontoqa_scheme2_train.log"
EVAL_LOG="$ROOT/logs/06_codi_prontoqa_scheme2_eval.log"
RESULT="$ROOT/results/codi_prontoqa_scheme2_test.json"
MODEL_PATH=/vepfs/tsra_models/hf/openai-community-gpt2
EXPT=codi_prontoqa_gpt2
CKPT_DIR="$OUT_ROOT/$EXPT/$(basename "$MODEL_PATH")/ep_40/lr_0.003/seed_11"

mkdir -p "$ROOT/logs" "$ROOT/status" "$ROOT/results" "$OUT_ROOT" "$DATA_DIR"
rm -f "${STATUS}.done" "${STATUS}.failed"

{
  echo task=codi_prontoqa_scheme2
  echo started_at="$(date '+%F %T')"
  echo gpu=4
  echo model="$MODEL_PATH"
  echo train_data="$DATA_DIR/prontoqa_train.json"
  echo test_data="$DATA_DIR/prontoqa_test.json"
  echo output_root="$OUT_ROOT"
  echo checkpoint_dir="$CKPT_DIR"
} > "${STATUS}.started"

python3 scripts/prepare_codi_prontoqa_data.py \
  --input-dir /vepfs/tsra_outputs/recent_external_data/prontoqa_ood \
  --output-dir "$DATA_DIR" >> "$TRAIN_LOG" 2>&1
prep_rc=$?
if [ "$prep_rc" -ne 0 ]; then
  {
    cat "${STATUS}.started"
    echo failed_at="$(date '+%F %T')"
    echo stage=prepare_data
    echo exit_code="$prep_rc"
    tail -n 200 "$TRAIN_LOG"
  } > "${STATUS}.failed"
  exit "$prep_rc"
fi

export TOKENIZERS_PARALLELISM=false
export HF_HOME=/vepfs/tsra_models/hf_home
export TRANSFORMERS_CACHE=/vepfs/tsra_models/hf_home
export CUDA_VISIBLE_DEVICES=4

cd /root/TRUA/external_baselines/CODI || exit 1

python train.py \
  --output_dir "$OUT_ROOT" \
  --expt_name "$EXPT" \
  --logging_dir "$OUT_ROOT/logs" \
  --logging_steps 10 \
  --model_name_or_path "$MODEL_PATH" \
  --data_name prontoqa \
  --icot_train_path "$DATA_DIR/prontoqa_train.json" \
  --seed 11 \
  --model_max_length 512 \
  --per_device_train_batch_size 64 \
  --gradient_accumulation_steps 2 \
  --bf16 \
  --num_train_epochs 40 \
  --learning_rate 3e-3 \
  --max_grad_norm 2.0 \
  --use_lora True \
  --lora_r 128 \
  --lora_alpha 32 \
  --lora_init \
  --save_strategy "no" \
  --save_safetensors False \
  --save_total_limit 1 \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --do_train \
  --report_to none \
  --num_latent 6 \
  --logging_strategy "steps" \
  --use_prj True \
  --prj_dim 768 \
  --prj_dropout 0.0 \
  --distill_loss_div_std True \
  --exp_mode False \
  --exp_data_num 2000 \
  --remove_eos True \
  --print_ref_model_stats True \
  --max_token_num 1000 > "$TRAIN_LOG" 2>&1
train_rc=$?
if [ "$train_rc" -ne 0 ]; then
  {
    cat "${STATUS}.started"
    echo failed_at="$(date '+%F %T')"
    echo stage=train
    echo exit_code="$train_rc"
    tail -n 240 "$TRAIN_LOG"
  } > "${STATUS}.failed"
  exit "$train_rc"
fi

cd /root/TRUA || exit 1
python3 scripts/run_codi_prontoqa_eval.py \
  --data "$DATA_DIR/prontoqa_test.json" \
  --model-name-or-path "$MODEL_PATH" \
  --ckpt-dir "$CKPT_DIR" \
  --out "$RESULT" \
  --batch-size 32 \
  --max-new-tokens 96 \
  --keep-predictions > "$EVAL_LOG" 2>&1
eval_rc=$?
if [ "$eval_rc" -ne 0 ]; then
  {
    cat "${STATUS}.started"
    echo failed_at="$(date '+%F %T')"
    echo stage=eval
    echo exit_code="$eval_rc"
    tail -n 120 "$TRAIN_LOG"
    tail -n 240 "$EVAL_LOG"
  } > "${STATUS}.failed"
  exit "$eval_rc"
fi

{
  cat "${STATUS}.started"
  echo finished_at="$(date '+%F %T')"
  echo result="$RESULT"
  tail -n 80 "$TRAIN_LOG"
  tail -n 120 "$EVAL_LOG"
} > "${STATUS}.done"

exit 0
