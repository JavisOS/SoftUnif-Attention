#!/usr/bin/env bash
set -euo pipefail

cd /root/TRUA

RUN_ROOT="/vepfs/tsra_outputs/official_external/fairr_test_only_compat_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$RUN_ROOT/logs"
STATUS_DIR="$RUN_ROOT/status"
mkdir -p "$LOG_DIR" "$STATUS_DIR"
ln -sfn "$RUN_ROOT" /vepfs/tsra_outputs/official_external/latest_fairr_test_only_compat

MODEL_ROOT="/vepfs/tsra_models/hf"
RULE_CKPT="/vepfs/tsra_outputs/official_external/fairr_saved/fairr_ruleselector_pwq_leq_0to3_OWA_rule_roberta_large_20_05_2026_625b5f32/checkpoints/epoch=1-step=22555.ckpt"
FACT_CKPT="/vepfs/tsra_outputs/official_external/fairr_saved/fairr_factselector_pwq_leq_0to3_OWA_fact_roberta_large_20_05_2026_755c65fa/checkpoints/epoch=4-step=21654.ckpt"
REASONER_CKPT="/vepfs/tsra_outputs/official_external/fairr_saved/fairr_reasoner_pw_leq_0to3_OWA_reasoner_t5_large_20_05_2026_2fa5534f/checkpoints/epoch=0-step=3906.ckpt"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# PyTorch 2.6 changed torch.load's default to weights_only=True; Lightning
# checkpoints in FaiRR include callback state, so evaluation needs legacy load.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

run_eval() {
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
    cd /root/TRUA/external_baselines/FaiRR
    CUDA_VISIBLE_DEVICES="$gpu" python main.py "$@" > "$log" 2>&1
    rc=$?
    if [[ "$rc" -eq 0 ]]; then
      { cat "$started"; echo "finished_at=$(date '+%F %T')"; tail -n 120 "$log"; } > "$done"
    else
      { cat "$started"; echo "failed_at=$(date '+%F %T')"; echo "exit_code=$rc"; tail -n 160 "$log"; } > "$failed"
    fi
    exit "$rc"
  ) &
  echo "$! $name gpu=$gpu log=$log" | tee -a "$RUN_ROOT/pids.txt"
}

run_eval fairr_ruleselector_test 1 \
  --override fairr_ruleselector,pwq_leq_0to3_OWA_rule \
  --hf_name "$MODEL_ROOT/roberta-large" \
  --gpus 1 \
  --evaluate_ckpt \
  --ckpt_path "$RULE_CKPT" \
  --eval_splits test \
  --eval_batch_size 32

run_eval fairr_factselector_test 2 \
  --override fairr_factselector,pwq_leq_0to3_OWA_fact \
  --hf_name "$MODEL_ROOT/roberta-large" \
  --gpus 1 \
  --evaluate_ckpt \
  --ckpt_path "$FACT_CKPT" \
  --eval_splits test \
  --eval_batch_size 32

run_eval fairr_reasoner_test 4 \
  --override fairr_reasoner \
  --dataset pw_leq_0to3_OWA_reasoner \
  --hf_name "$MODEL_ROOT/t5-large" \
  --gpus 1 \
  --evaluate_ckpt \
  --ckpt_path "$REASONER_CKPT" \
  --eval_splits test \
  --eval_batch_size 8 \
  --train_batch_size 4 \
  --accumulate_grad_batches 4

wait
echo "finished_at=$(date '+%F %T')" > "$RUN_ROOT/all_done"
