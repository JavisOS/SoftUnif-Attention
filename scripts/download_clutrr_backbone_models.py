from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ROOT = Path("/vepfs/tsra_models/hf")
MODELS = {
    "roberta-base": "roberta-base",
    "deberta-base": "microsoft/deberta-base",
}


def main() -> None:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    for local_name, repo_id in MODELS.items():
        local_dir = MODEL_ROOT / local_name
        print(f"Downloading {repo_id} -> {local_dir}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            endpoint="https://hf-mirror.com",
            resume_download=True,
            local_dir_use_symlinks=False,
            allow_patterns=[
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.txt",
                "vocab.json",
                "merges.txt",
                "spm.model",
                "pytorch_model.bin",
                "model.safetensors",
            ],
        )
        print(f"Finished {local_name}", flush=True)


if __name__ == "__main__":
    main()
