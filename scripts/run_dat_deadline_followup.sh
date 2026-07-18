#!/usr/bin/env bash
set -euo pipefail

readonly REPO=/vepfs/trua_worktrees/TRUA_dat_formal
readonly OUT=/vepfs/trua_outputs/dat_formal_debertav3_20260718
readonly PYTHON=/root/miniconda3/bin/python
readonly MODEL=/vepfs/trua_models/hf/deberta-v3-base
readonly CURRENT_PROOFWRITER_SUPERVISOR=114792

export TRUA_DUAL_ATTENTION_ROOT=/root/TRUA/external_baselines/dual-attention
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

while kill -0 "${CURRENT_PROOFWRITER_SUPERVISOR}" 2>/dev/null; do
    sleep 60
done

mkdir -p "${OUT}/proposition" "${OUT}/clutrr"

"${PYTHON}" -u "${REPO}/scripts/run_proposition_backbone_core_matrix.py" \
    --output-root "${OUT}/proposition" \
    --backbone "deberta-v3:${MODEL}" \
    --datasets ruletaker pfolio \
    --cores dual_attention \
    --seeds 0 1 42 \
    --gpus 0 1 2 \
    --epochs 10 \
    --batch-size 16 \
    --lr 2e-5 \
    --limit-train 0 \
    --limit-test 0 \
    --poll-seconds 20 \
    >"${OUT}/proposition_supervisor.log" 2>&1 &
readonly proposition_pid=$!

"${PYTHON}" -u "${REPO}/scripts/run_clutrr_backbone_core_matrix.py" \
    --output-root "${OUT}/clutrr" \
    --data-root /root/TRUA/data \
    --dataset data_089907f8 \
    --backbone "deberta-v3:deberta_v3:${MODEL}" \
    --cores dual_attention \
    --seeds 0 1 42 \
    --gpus 3 4 5 \
    --epochs 10 \
    --batch-size 16 \
    --eval-batch-size 32 \
    --checkpoint-selection validation \
    --validation-fraction 0.0 \
    --external-validation-dataset data_db9b8f04 \
    --validation-selection-metric unseen_4_10 \
    --poll-seconds 15 \
    >"${OUT}/clutrr_supervisor.log" 2>&1 &
readonly clutrr_pid=$!

wait "${proposition_pid}"
wait "${clutrr_pid}"

while [[ $(find "${OUT}/proofwriter" -name metrics.json -type f | wc -l | tr -d ' ') != 3 ]]; do
    sleep 60
done

"${PYTHON}" "${REPO}/scripts/aggregate_dat_deadline_matrix.py" "${OUT}"

