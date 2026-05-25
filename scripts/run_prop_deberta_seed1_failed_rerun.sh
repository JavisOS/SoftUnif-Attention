#!/usr/bin/env bash
set -euo pipefail

cd /root/TSRA

RUN_ROOT="/vepfs/tsra_outputs/formal_10ep/prop_deberta_seed1_failed_rerun_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$RUN_ROOT/logs"
RESULT_DIR="$RUN_ROOT/results"
STATUS_DIR="$RUN_ROOT/status"
MODEL="/vepfs/tsra_models/hf/deberta-base"

mkdir -p "$LOG_DIR" "$RESULT_DIR" "$STATUS_DIR"
ln -sfn "$RUN_ROOT" /vepfs/tsra_outputs/formal_10ep/latest_prop_deberta_seed1_failed_rerun

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

run_task() {
  local name="$1"
  local gpu="$2"
  shift 2
  local log="$LOG_DIR/${name}.log"
  local out="$RESULT_DIR/${name}.json"
  local started="$STATUS_DIR/${name}.started"
  local done="$STATUS_DIR/${name}.done"
  local failed="$STATUS_DIR/${name}.failed"

  {
    echo "task=$name"
    echo "gpu=$gpu"
    echo "seed=1"
    echo "epochs=10"
    echo "started_at=$(date '+%F %T')"
    echo "out=$out"
  } > "$started"

  echo "[$(date '+%F %T')] >>> $name on gpu=$gpu" | tee -a "$RUN_ROOT/queue.log"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/transformer_tsra_prop.py "$@" \
    --model-name "$MODEL" \
    --seed 1 \
    --epochs 10 \
    --out "$out" \
    > "$log" 2>&1
  rc=$?
  set -e

  if [[ "$rc" -eq 0 ]]; then
    { cat "$started"; echo "finished_at=$(date '+%F %T')"; tail -n 80 "$log"; } > "$done"
    echo "[$(date '+%F %T')] <<< $name done" | tee -a "$RUN_ROOT/queue.log"
  else
    { cat "$started"; echo "failed_at=$(date '+%F %T')"; echo "exit_code=$rc"; tail -n 140 "$log"; } > "$failed"
    echo "[$(date '+%F %T')] <<< $name failed rc=$rc" | tee -a "$RUN_ROOT/queue.log"
  fi
}

run_pronto_gpu5() {
  run_task prontoqa_deberta_baseline_seed1_10ep 5 \
    --dataset prontoqa \
    --root data/prontoqa_ood/processed/generated_ood_data \
    --limit-train 0 \
    --limit-test 0 \
    --batch-size 16 \
    --lr 2e-5 \
    --lambda-trace 0.0 \
    --max-sents 24 \
    --max-len 192

  run_task prontoqa_deberta_tsra_seed1_10ep 5 \
    --dataset prontoqa \
    --root data/prontoqa_ood/processed/generated_ood_data \
    --limit-train 0 \
    --limit-test 0 \
    --batch-size 16 \
    --lr 2e-5 \
    --lambda-trace 1.0 \
    --max-sents 24 \
    --max-len 192
}

run_task proofwriter_deberta_tsra_seed1_10ep 7 \
  --dataset proofwriter \
  --root data/proofwriter/raw/proofwriter-dataset-V2020.12.3 \
  --train-depths 0,1,2 \
  --test-depths 3,5 \
  --limit-train 0 \
  --limit-test 0 \
  --batch-size 16 \
  --lr 2e-5 \
  --lambda-trace 1.0 \
  --max-sents 16 \
  --max-len 192 &

run_task ruletaker_deberta_baseline_seed1_10ep 0 \
  --dataset ruletaker_gfair \
  --root external_baselines/GFaiR/data/ruletaker_3ext_sat \
  --limit-train 0 \
  --limit-test 0 \
  --batch-size 16 \
  --lr 2e-5 \
  --lambda-trace 0.0 \
  --max-sents 24 \
  --max-len 192 &

run_task ruletaker_deberta_tsra_seed1_10ep 3 \
  --dataset ruletaker_gfair \
  --root external_baselines/GFaiR/data/ruletaker_3ext_sat \
  --limit-train 0 \
  --limit-test 0 \
  --batch-size 16 \
  --lr 2e-5 \
  --lambda-trace 1.0 \
  --max-sents 24 \
  --max-len 192 &

run_pronto_gpu5 &

wait
echo "[$(date '+%F %T')] prop deberta seed1 failed-task rerun finished" | tee -a "$RUN_ROOT/queue.log"
