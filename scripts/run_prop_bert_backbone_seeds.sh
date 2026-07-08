#!/usr/bin/env bash
set -euo pipefail

cd /root/TRUA

RUN_ROOT="/vepfs/tsra_outputs/formal_10ep/prop_bert_backbone_seeds_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$RUN_ROOT/logs"
RESULT_DIR="$RUN_ROOT/results"
STATUS_DIR="$RUN_ROOT/status"
MODEL="/vepfs/tsra_models/hf/bert-base-uncased"

mkdir -p "$LOG_DIR" "$RESULT_DIR" "$STATUS_DIR"
ln -sfn "$RUN_ROOT" /vepfs/tsra_outputs/formal_10ep/latest_prop_bert_backbone_seeds

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

run_task() {
  local gpu="$1"
  local name="$2"
  shift 2
  local log="$LOG_DIR/${name}.log"
  local out="$RESULT_DIR/${name}.json"
  local started="$STATUS_DIR/${name}.started"
  local done="$STATUS_DIR/${name}.done"
  local failed="$STATUS_DIR/${name}.failed"

  {
    echo "task=$name"
    echo "gpu=$gpu"
    echo "model=$MODEL"
    echo "epochs=10"
    echo "started_at=$(date '+%F %T')"
    echo "out=$out"
  } > "$started"

  echo "[$(date '+%F %T')] >>> $name on gpu=$gpu" | tee -a "$RUN_ROOT/queue.log"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/transformer_trua_prop.py "$@" \
    --model-name "$MODEL" \
    --epochs 10 \
    --out "$out" \
    > "$log" 2>&1
  rc=$?
  set -e

  if [[ "$rc" -eq 0 ]]; then
    { cat "$started"; echo "finished_at=$(date '+%F %T')"; tail -n 80 "$log"; } > "$done"
    echo "[$(date '+%F %T')] <<< $name done" | tee -a "$RUN_ROOT/queue.log"
  else
    { cat "$started"; echo "failed_at=$(date '+%F %T')"; echo "exit_code=$rc"; tail -n 160 "$log"; } > "$failed"
    echo "[$(date '+%F %T')] <<< $name failed rc=$rc" | tee -a "$RUN_ROOT/queue.log"
  fi
}

run_proofwriter_pair() {
  local gpu="$1"
  local seed="$2"
  run_task "$gpu" "proofwriter_bert_baseline_seed${seed}_10ep" \
    --dataset proofwriter \
    --root data/proofwriter/raw/proofwriter-dataset-V2020.12.3 \
    --train-depths 0,1,2 \
    --test-depths 3,5 \
    --limit-train 0 \
    --limit-test 0 \
    --batch-size 16 \
    --lr 2e-5 \
    --lambda-trace 0.0 \
    --max-sents 16 \
    --max-len 192 \
    --seed "$seed"

  run_task "$gpu" "proofwriter_bert_tsra_seed${seed}_10ep" \
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
    --max-len 192 \
    --seed "$seed"
}

run_ruletaker_pair() {
  local gpu="$1"
  local seed="$2"
  run_task "$gpu" "ruletaker_bert_baseline_seed${seed}_10ep" \
    --dataset ruletaker_gfair \
    --root external_baselines/GFaiR/data/ruletaker_3ext_sat \
    --limit-train 0 \
    --limit-test 0 \
    --batch-size 16 \
    --lr 2e-5 \
    --lambda-trace 0.0 \
    --max-sents 24 \
    --max-len 192 \
    --seed "$seed"

  run_task "$gpu" "ruletaker_bert_tsra_seed${seed}_10ep" \
    --dataset ruletaker_gfair \
    --root external_baselines/GFaiR/data/ruletaker_3ext_sat \
    --limit-train 0 \
    --limit-test 0 \
    --batch-size 16 \
    --lr 2e-5 \
    --lambda-trace 1.0 \
    --max-sents 24 \
    --max-len 192 \
    --seed "$seed"
}

run_pronto_pair() {
  local gpu="$1"
  local seed="$2"
  run_task "$gpu" "prontoqa_bert_baseline_seed${seed}_10ep" \
    --dataset prontoqa \
    --root data/prontoqa_ood/processed/generated_ood_data \
    --limit-train 0 \
    --limit-test 0 \
    --batch-size 16 \
    --lr 2e-5 \
    --lambda-trace 0.0 \
    --max-sents 24 \
    --max-len 192 \
    --seed "$seed"

  run_task "$gpu" "prontoqa_bert_tsra_seed${seed}_10ep" \
    --dataset prontoqa \
    --root data/prontoqa_ood/processed/generated_ood_data \
    --limit-train 0 \
    --limit-test 0 \
    --batch-size 16 \
    --lr 2e-5 \
    --lambda-trace 1.0 \
    --max-sents 24 \
    --max-len 192 \
    --seed "$seed"
}

# GPU0 is reserved for NLProofS. GPU1/2 are currently used by RoBERTa ProofWriter TRUA.
run_proofwriter_pair 3 0 &
run_proofwriter_pair 4 1 &
run_ruletaker_pair 5 0 &
run_ruletaker_pair 6 1 &
run_pronto_pair 7 0 &
run_pronto_pair 7 1 &

wait
echo "[$(date '+%F %T')] bert backbone seed queue finished" | tee -a "$RUN_ROOT/queue.log"
