#!/usr/bin/env bash
set -euo pipefail

cd /root/TRUA

RUN_ROOT="/vepfs/tsra_outputs/official_external/more_external_eval_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$RUN_ROOT/logs"
STATUS_DIR="$RUN_ROOT/status"
mkdir -p "$LOG_DIR" "$STATUS_DIR"
ln -sfn "$RUN_ROOT" /vepfs/tsra_outputs/official_external/latest_more_external_eval

MODEL_ROOT="/vepfs/tsra_models/hf"
GFAIR_ROOT="/root/TRUA/external_baselines/GFaiR"
FAIRR_ROOT="/root/TRUA/external_baselines/FaiRR"

RULE_CKPT="/vepfs/tsra_outputs/official_external/fairr_saved/fairr_ruleselector_pwq_leq_0to3_OWA_rule_roberta_large_20_05_2026_625b5f32/checkpoints/epoch=1-step=22555.ckpt"
FACT_CKPT="/vepfs/tsra_outputs/official_external/fairr_saved/fairr_factselector_pwq_leq_0to3_OWA_fact_roberta_large_20_05_2026_755c65fa/checkpoints/epoch=4-step=21654.ckpt"
REASONER_CKPT="/vepfs/tsra_outputs/official_external/fairr_saved/fairr_reasoner_pw_leq_0to3_OWA_reasoner_t5_large_20_05_2026_2fa5534f/checkpoints/epoch=0-step=3906.ckpt"
GFAIR_SELECTOR_CKPT="/vepfs/tsra_outputs/official_external/gfair/out_selector2_ruletaker/pytorch_model.bin26"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TRUA_ROBERTA_LARGE_PATH="$MODEL_ROOT/roberta-large"
export TRUA_T5_LARGE_PATH="$MODEL_ROOT/t5-large"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

mkdir -p model
ln -sfn "$MODEL_ROOT/xlnet-large-cased" model/xlnet
ln -sfn "$MODEL_ROOT/t5-large" model/T5

run_task() {
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

run_task gfair_selector2_test_best26 1 \
  python "$GFAIR_ROOT/run_selector2.py" \
    --input_dir "$GFAIR_ROOT/data/ruletaker_3ext_sat" \
    --output_dir "$RUN_ROOT/gfair_selector2_test_best26" \
    --model "$MODEL_ROOT/xlnet-large-cased" \
    --init_weights_dir "$GFAIR_SELECTOR_CKPT" \
    --do_test \
    --eval_batch_size 32 \
    --alpha 0.05

run_task gfair_convertor_train_official 2 \
  python "$GFAIR_ROOT/run_convertor.py" \
    --input_dir "$GFAIR_ROOT/data/ruletaker_3ext_sat" \
    --output_dir "$RUN_ROOT/gfair/out_convertor_ruletaker" \
    --model "$MODEL_ROOT/t5-large" \
    --do_train \
    --num_train_epochs 10 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --gradient_accumulation_steps 16

run_task gfair_reasoner_train_official 4 \
  python "$GFAIR_ROOT/run_reasoner.py" \
    --input_dir "$GFAIR_ROOT/data/ruletaker_3ext_sat" \
    --output_dir "$RUN_ROOT/gfair/out_reasoner_ruletaker" \
    --model "$MODEL_ROOT/t5-large" \
    --do_train \
    --learning_rate 1e-4 \
    --num_train_epochs 20 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --gradient_accumulation_steps 16 \
    --seed 42

run_task fairr_end_to_end_pwu_leq_3 6 \
  bash -lc "cd '$FAIRR_ROOT' && python process_proofwriter.py --dataset pwu_leq_3 && python main.py --override fairr_inference,evaluate --dataset pwu_leq_3_OWA --ruleselector_ckpt '$RULE_CKPT' --factselector_ckpt '$FACT_CKPT' --reasoner_ckpt '$REASONER_CKPT' --gpus 1 --eval_batch_size 8 --num_workers 4 --eval_splits test"

echo "run_root=$RUN_ROOT"
