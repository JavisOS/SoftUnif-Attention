#!/usr/bin/env bash
set +e

cd /root/TRUA || exit 1

ROOT=/vepfs/tsra_outputs/recent_external_baselines/latest
LOG="$ROOT/logs/05_logipt_logiclm_proofwriter_cot.log"
OUT="$ROOT/results/logipt_logiclm_proofwriter_cot.json"
STATUS="$ROOT/status/05_logipt_logiclm_proofwriter_cot"

mkdir -p "$ROOT/logs" "$ROOT/status" "$ROOT/results"

{
  echo task=logipt_logiclm_proofwriter_cot
  echo started_at="$(date '+%F %T')"
  echo gpu=6
  echo model=/vepfs/tsra_models/hf/logipt/LoGiPT-CodeLlama-13b-Instruct-hf-proofwriter
  echo data=/root/TRUA/external_baselines/Logic-LM/data/ProofWriter/test.json
  echo prompt=/root/TRUA/external_baselines/Logic-LM/baselines/icl_examples/ProofWriter_CoT.txt
} > "${STATUS}.started"
rm -f "${STATUS}.done" "${STATUS}.failed"

export TOKENIZERS_PARALLELISM=false
export HF_HOME=/vepfs/tsra_models/hf_home
export TRANSFORMERS_CACHE=/vepfs/tsra_models/hf_home
export CUDA_VISIBLE_DEVICES=6

python3 scripts/run_logipt_logiclm_eval.py \
  --model /vepfs/tsra_models/hf/logipt/LoGiPT-CodeLlama-13b-Instruct-hf-proofwriter \
  --logiclm-root /root/TRUA/external_baselines/Logic-LM \
  --dataset-name ProofWriter \
  --split test \
  --mode CoT \
  --batch-size 4 \
  --max-input-tokens 4096 \
  --max-new-tokens 256 \
  --out "$OUT" \
  --keep-generations > "$LOG" 2>&1
rc=$?

if [ "$rc" -eq 0 ]; then
  {
    cat "${STATUS}.started"
    echo finished_at="$(date '+%F %T')"
    tail -n 120 "$LOG"
  } > "${STATUS}.done"
else
  {
    cat "${STATUS}.started"
    echo failed_at="$(date '+%F %T')"
    echo exit_code="$rc"
    tail -n 200 "$LOG"
  } > "${STATUS}.failed"
fi

exit "$rc"
