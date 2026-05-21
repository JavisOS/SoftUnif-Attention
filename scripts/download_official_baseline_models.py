#!/usr/bin/env python3
"""Download official external-baseline checkpoints to shared storage."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = [
    "roberta-large",
    "t5-large",
    "xlnet-large-cased",
]

ALLOW_PATTERNS = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "vocab.txt",
    "spiece.model",
    "special_tokens_map.json",
    "pytorch_model.bin",
    "model.safetensors",
    "generation_config.json",
]


def main():
    root = Path("/vepfs/tsra_models/hf")
    root.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        target = root / model
        print(f"Downloading {model} -> {target}", flush=True)
        snapshot_download(
            repo_id=model,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            resume_download=True,
            endpoint="https://hf-mirror.com",
            allow_patterns=ALLOW_PATTERNS,
        )
        print(f"Finished {model}", flush=True)


if __name__ == "__main__":
    main()
