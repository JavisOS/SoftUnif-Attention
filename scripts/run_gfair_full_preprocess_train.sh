#!/usr/bin/env bash
set -euo pipefail

cd /root/TSRA

RUN_ROOT="/vepfs/tsra_outputs/official_external/gfair_full_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$RUN_ROOT/logs"
STATUS_DIR="$RUN_ROOT/status"
mkdir -p "$LOG_DIR" "$STATUS_DIR"
ln -sfn "$RUN_ROOT" /vepfs/tsra_outputs/official_external/latest_gfair_full

MODEL_ROOT="/vepfs/tsra_models/hf"
GFAIR_ROOT="/root/TSRA/external_baselines/GFaiR"
DATA_DIR="$GFAIR_ROOT/data"
SELECTOR_CKPT="/vepfs/tsra_outputs/official_external/gfair/out_selector2_ruletaker/pytorch_model.bin26"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p model
ln -sfn "$MODEL_ROOT/xlnet-large-cased" model/xlnet
ln -sfn "$MODEL_ROOT/t5-large" model/T5

run_gpu() {
  local name="$1"
  local gpu="$2"
  shift 2
  local log="$LOG_DIR/${name}.log"
  local started="$STATUS_DIR/${name}.started"
  local done="$STATUS_DIR/${name}.done"
  local failed="$STATUS_DIR/${name}.failed"
  {
    echo "task=$name"
    echo "gpu=$gpu"
    echo "started_at=$(date '+%F %T')"
  } > "$started"
  (
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" "$@" > "$log" 2>&1
    rc=$?
    if [[ "$rc" -eq 0 ]]; then
      { cat "$started"; echo "finished_at=$(date '+%F %T')"; tail -n 160 "$log"; } > "$done"
    else
      { cat "$started"; echo "failed_at=$(date '+%F %T')"; echo "exit_code=$rc"; tail -n 220 "$log"; } > "$failed"
    fi
    exit "$rc"
  ) &
  echo "$! $name gpu=$gpu log=$log" | tee -a "$RUN_ROOT/pids.txt"
}

{
  echo "task=gfair_preprocess_full"
  echo "started_at=$(date '+%F %T')"
} > "$STATUS_DIR/gfair_preprocess_full.started"

(
  set +e
  cd "$DATA_DIR"
  python RT2prover9.py
  rc1=$?
  python prover92FOL.py
  rc2=$?
  python create_my_example.py --model_type convertor --direc ruletaker_3ext_sat/
  rc3=$?
  python create_my_example.py --model_type reasoner --direc ruletaker_3ext_sat/
  rc4=$?
  if [[ "$rc1" -eq 0 && "$rc2" -eq 0 && "$rc3" -eq 0 && "$rc4" -eq 0 ]]; then
    exit 0
  fi
  echo "return_codes: RT2prover9=$rc1 prover92FOL=$rc2 convertor=$rc3 reasoner=$rc4"
  exit 1
) > "$LOG_DIR/gfair_preprocess_full.log" 2>&1
rc=$?

if [[ "$rc" -eq 0 ]]; then
  { cat "$STATUS_DIR/gfair_preprocess_full.started"; echo "finished_at=$(date '+%F %T')"; tail -n 160 "$LOG_DIR/gfair_preprocess_full.log"; } > "$STATUS_DIR/gfair_preprocess_full.done"
else
  { cat "$STATUS_DIR/gfair_preprocess_full.started"; echo "failed_at=$(date '+%F %T')"; echo "exit_code=$rc"; tail -n 220 "$LOG_DIR/gfair_preprocess_full.log"; } > "$STATUS_DIR/gfair_preprocess_full.failed"
  exit "$rc"
fi

run_gpu gfair_convertor_train_official 2 \
  python "$GFAIR_ROOT/run_convertor.py" \
    --input_dir "$DATA_DIR/ruletaker_3ext_sat" \
    --output_dir "$RUN_ROOT/out_convertor_ruletaker" \
    --model "$MODEL_ROOT/t5-large" \
    --do_train \
    --num_train_epochs 10 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --gradient_accumulation_steps 16

run_gpu gfair_reasoner_train_official 4 \
  python "$GFAIR_ROOT/run_reasoner.py" \
    --input_dir "$DATA_DIR/ruletaker_3ext_sat" \
    --output_dir "$RUN_ROOT/out_reasoner_ruletaker" \
    --model "$MODEL_ROOT/t5-large" \
    --do_train \
    --learning_rate 1e-4 \
    --num_train_epochs 20 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --gradient_accumulation_steps 16 \
    --seed 42

wait

if find "$STATUS_DIR" -maxdepth 1 -name 'gfair_*_train_official.failed' | grep -q .; then
  echo "training_failed_at=$(date '+%F %T')" > "$STATUS_DIR/gfair_full_inference.skipped"
  exit 1
fi

CONVERTOR_CKPT="$(ls -1v "$RUN_ROOT"/out_convertor_ruletaker/pytorch_model.bin* | tail -1)"
REASONER_BEST_EPOCH="$(grep -E 'best_epoch = ' "$LOG_DIR/gfair_reasoner_train_official.log" | tail -1 | awk '{print $NF}')"
if [[ -n "${REASONER_BEST_EPOCH:-}" && -f "$RUN_ROOT/out_reasoner_ruletaker/pytorch_model.bin${REASONER_BEST_EPOCH}" ]]; then
  REASONER_CKPT="$RUN_ROOT/out_reasoner_ruletaker/pytorch_model.bin${REASONER_BEST_EPOCH}"
else
  REASONER_CKPT="$(ls -1v "$RUN_ROOT"/out_reasoner_ruletaker/pytorch_model.bin* | tail -1)"
fi

echo "convertor_ckpt=$CONVERTOR_CKPT" > "$RUN_ROOT/gfair_inference_ckpts.txt"
echo "reasoner_ckpt=$REASONER_CKPT" >> "$RUN_ROOT/gfair_inference_ckpts.txt"
echo "selector_ckpt=$SELECTOR_CKPT" >> "$RUN_ROOT/gfair_inference_ckpts.txt"

run_gpu gfair_full_inference_ruletaker_3ext 5 \
  python "$GFAIR_ROOT/inference_rt.py" \
    --input_dir "$DATA_DIR/ruletaker_3ext_sat" \
    --output_dir "$RUN_ROOT/full_inference_ruletaker_3ext" \
    --convertor_init_weights_dir "$CONVERTOR_CKPT" \
    --selector_init_weights_dir "$SELECTOR_CKPT" \
    --reasoner_init_weights_dir "$REASONER_CKPT" \
    --inference_batch_size 4 \
    --beam_size 2 \
    --candidates_num 2 \
    --maxdep 3

wait
echo "finished_at=$(date '+%F %T')" > "$RUN_ROOT/all_done"
