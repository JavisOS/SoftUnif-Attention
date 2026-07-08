#!/usr/bin/env bash
set -euo pipefail

REPO=/root/TRUA
BASE=/root/TRUA/external_baselines/NLProofS
DATA=/vepfs/tsra_outputs/official_external/nlproofs_data/proofwriter-dataset-V2020.12.3/preprocessed_OWA/depth-3ext
RUN_ROOT="/vepfs/tsra_outputs/official_external/nlproofs_ruletaker_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/status" "$RUN_ROOT/prover" "$RUN_ROOT/verifier"
ln -sfn "$RUN_ROOT" /vepfs/tsra_outputs/official_external/latest_nlproofs_ruletaker

export PYTHONPATH="$BASE:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Waiting for NLProofS preprocessed RuleTaker data under $DATA" | tee "$RUN_ROOT/status/driver.log"
for f in meta-train.jsonl meta-dev.jsonl meta-test.jsonl; do
  while [ ! -s "$DATA/$f" ]; do
    sleep 30
  done
  echo "ready $DATA/$f $(stat -c%s "$DATA/$f")" | tee -a "$RUN_ROOT/status/driver.log"
done

cd "$BASE/prover"
(
  echo "started $(date -Is)"
  CUDA_VISIBLE_DEVICES=1 python main.py fit \
    --config cli_ruletaker_stepwise_t5-large.yaml \
    --trainer.default_root_dir "$RUN_ROOT/prover" \
    --trainer.max_epochs 20 \
    --trainer.limit_val_batches 200 \
    --trainer.num_sanity_val_steps 0 \
    --model.model_name /vepfs/tsra_models/hf/t5-large \
    --data.path_train "$DATA/meta-train.jsonl" \
    --data.path_val "$DATA/meta-dev.jsonl" \
    --data.path_test "$DATA/meta-test.jsonl"
  echo "done $(date -Is)"
) > "$RUN_ROOT/logs/prover_train.log" 2>&1 &
echo "$! nlproofs_prover gpu=1 log=$RUN_ROOT/logs/prover_train.log" | tee "$RUN_ROOT/prover.pid"
touch "$RUN_ROOT/status/prover.started"

cd "$BASE/verifier"
(
  echo "started $(date -Is)"
  CUDA_VISIBLE_DEVICES=6 python main.py fit \
    --config cli_ruletaker.yaml \
    --trainer.default_root_dir "$RUN_ROOT/verifier" \
    --trainer.max_epochs 50 \
    --trainer.num_sanity_val_steps 0 \
    --model.model_name /vepfs/tsra_models/hf/roberta-large \
    --data.path_train "$DATA/meta-train.jsonl" \
    --data.path_val "$DATA/meta-dev.jsonl"
  echo "done $(date -Is)"
) > "$RUN_ROOT/logs/verifier_train.log" 2>&1 &
echo "$! nlproofs_verifier gpu=6 log=$RUN_ROOT/logs/verifier_train.log" | tee "$RUN_ROOT/verifier.pid"
touch "$RUN_ROOT/status/verifier.started"

echo "$RUN_ROOT"
