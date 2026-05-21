#!/usr/bin/env bash
set -euo pipefail

cd /root/TSRA

OUT="/vepfs/tsra_models/hf/roberta-base-clean-$(date +%Y%m%d_%H%M%S)"
CACHE="/vepfs/tsra_models/hf_cache_clean_roberta"
LOG="/vepfs/tsra_outputs/supervisor/roberta_redownload_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUT" "$CACHE" "$(dirname "$LOG")"
echo "Downloading clean roberta-base to $OUT" | tee "$LOG"

HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}" python3 - <<PY >>"$LOG" 2>&1
from huggingface_hub import snapshot_download
from pathlib import Path

out = Path("$OUT")
cache = Path("$CACHE")
snapshot_download(
    repo_id="roberta-base",
    local_dir=str(out),
    cache_dir=str(cache),
    local_dir_use_symlinks=False,
    force_download=True,
    resume_download=False,
    allow_patterns=[
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "pytorch_model.bin",
        "model.safetensors",
    ],
)
print("SNAPSHOT_DONE", out)
PY

python3 - <<PY >>"$LOG" 2>&1
from transformers import AutoModel, AutoTokenizer

p = "$OUT"
AutoTokenizer.from_pretrained(p, use_fast=True)
m = AutoModel.from_pretrained(p)
print("LOAD_OK", type(m).__name__, p)
PY

ln -sfn "$OUT" /vepfs/tsra_models/hf/roberta-base-clean-current
echo "DONE $OUT" | tee -a "$LOG"
