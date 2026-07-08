#!/usr/bin/env python3
"""Prepare TRUA data views for recent external reasoning baselines.

This intentionally writes only small JSON views and config files. Large model
checkpoints stay outside the repo under /vepfs.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import yaml


def _walk_prontoqa_items(obj: Any):
    if isinstance(obj, dict):
        if {"question", "query", "chain_of_thought"}.issubset(obj):
            yield obj
            return
        for value in obj.values():
            yield from _walk_prontoqa_items(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_prontoqa_items(value)


def _iter_prontoqa_examples(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    for item in _walk_prontoqa_items(raw):
        question = str(item.get("question", "")).strip()
        query = str(item.get("query", "")).strip()
        steps = [str(x).strip() for x in item.get("chain_of_thought", []) if str(x).strip()]
        if not question or not query or not steps:
            continue
        answer = steps[-1]
        yield {
            "question": f"{question}\n{query}",
            "answer": answer,
            "steps": steps[:-1] if len(steps) > 1 else steps,
            "trua_query": query,
            "trua_context": question,
            "trua_chain_of_thought": steps,
            "source_file": path.name,
        }


def _load_many(root: Path, names: list[str]):
    examples: list[dict[str, Any]] = []
    for name in names:
        path = root / name
        if not path.exists():
            raise FileNotFoundError(path)
        examples.extend(_iter_prontoqa_examples(path))
    return examples


def prepare_prontoqa(root: Path, out_dir: Path, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    train_names = [
        "1hop_ProofsOnly_random_noadj.json",
        "2hop_ProofsOnly_random_noadj.json",
    ]
    test_names = [
        "3hop_ProofsOnly_random_noadj.json",
        "4hop_ProofsOnly_random_noadj.json",
        "4hop_OOD_Composed_random_noadj.json",
    ]

    train = _load_many(root, train_names)
    rng = random.Random(seed)
    rng.shuffle(train)
    valid_size = max(20, min(len(train) // 5, 80))
    valid = train[:valid_size]
    train = train[valid_size:]
    test = _load_many(root, test_names)

    (out_dir / "prontoqa_ood_coconut_train.json").write_text(
        json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "prontoqa_ood_coconut_valid.json").write_text(
        json.dumps(valid, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "prontoqa_ood_coconut_test.json").write_text(
        json.dumps(test, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "seed": seed,
        "train_files": train_names,
        "test_files": test_names,
        "train": len(train),
        "valid": len(valid),
        "test": len(test),
        "format": "Coconut-compatible list of {question, answer, steps}; answer is the final proof conclusion.",
    }
    (out_dir / "README_RECENT_EXTERNAL_DATA.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_coconut_configs(out_dir: Path, save_root: Path) -> None:
    base = {
        "project": "tsra_recent_external",
        "save_path": str(save_root),
        "name": "prontoqa-ood-coconut",
        "only_eval": False,
        "coconut": True,
        "cot": False,
        "no_thoughts": False,
        "no_cot": False,
        "c_thought": 1,
        "epochs_per_stage": 5,
        "max_latent_stage": 6,
        "pad_latent_to_max": True,
        "save_only_improve": False,
        "uniform_prob": 0.0,
        "model_id": "openai-community/gpt2",
        "load_model_path": "None",
        "seed": 0,
        "resume": 0,
        "bf16": False,
        "train_path": str(out_dir / "prontoqa_ood_coconut_train.json"),
        "val_path": str(out_dir / "prontoqa_ood_coconut_valid.json"),
        "reset_optimizer": True,
        "batch_size_training": 32,
        "debug": False,
        "gradient_accumulation_steps": 1,
        "num_epochs": 50,
        "lr": 1e-4,
        "weight_decay": 0.01,
    }
    eval_cfg = dict(base)
    eval_cfg.update(
        {
            "only_eval": True,
            "name": "prontoqa-ood-coconut-eval",
            "load_model_path": "PENDING_COCONUT_CHECKPOINT",
            "resume": 40,
            "train_path": str(out_dir / "prontoqa_ood_coconut_train.json"),
            "val_path": str(out_dir / "prontoqa_ood_coconut_test.json"),
        }
    )
    (out_dir / "prontoqa_ood_coconut.yaml").write_text(
        yaml.safe_dump(base, sort_keys=False), encoding="utf-8"
    )
    (out_dir / "prontoqa_ood_coconut_eval.yaml").write_text(
        yaml.safe_dump(eval_cfg, sort_keys=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prontoqa-root",
        default="data/prontoqa_ood/processed/generated_ood_data",
    )
    parser.add_argument(
        "--out-dir",
        default="/vepfs/tsra_outputs/recent_external_data/prontoqa_ood",
    )
    parser.add_argument(
        "--save-root",
        default="/vepfs/tsra_outputs/recent_external_baselines/checkpoints",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    prepare_prontoqa(Path(args.prontoqa_root), out_dir, args.seed)
    write_coconut_configs(out_dir, Path(args.save_root))
    print(f"prepared recent external PrOntoQA-OOD data under {out_dir}")


if __name__ == "__main__":
    main()
