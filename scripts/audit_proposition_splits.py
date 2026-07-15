#!/usr/bin/env python3
"""Audit the exact full proposition splits and tokenization limits."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from generic_trua_prop import load_proofwriter, load_ruletaker_raw


DEFAULT_BACKBONES = (
    "bert:/vepfs/trua_models/hf/bert-base-uncased",
    "roberta:/vepfs/trua_models/hf/roberta-base",
    "deberta-v3:/vepfs/trua_models/hf/deberta-v3-base",
)


def parse_backbone(specification):
    name, path = specification.split(":", 1)
    return name, path


def load_splits(proofwriter_root, ruletaker_root):
    return {
        "proofwriter": {
            "train": load_proofwriter(proofwriter_root, [0, 1, 2], "train"),
            "validation": load_proofwriter(proofwriter_root, [0, 1, 2], "dev"),
            "test:depth-3": load_proofwriter(proofwriter_root, [3], "test"),
            "test:depth-5": load_proofwriter(proofwriter_root, [5], "test"),
        },
        "ruletaker": {
            "train": load_ruletaker_raw(
                ruletaker_root, [1, 2], "train", qdeps=[1, 2]
            ),
            "validation": load_ruletaker_raw(
                ruletaker_root, [1, 2], "dev", qdeps=[1, 2]
            ),
            "test": load_ruletaker_raw(
                ruletaker_root, [1, 2, 3, 5], "test", qdeps=[1, 2, 3, 4, 5]
            ),
        },
    }


def split_statistics(samples):
    ids = [str(sample.get("id", "")) for sample in samples]
    evidence_examples = sum(bool(sum(sample.get("trace_labels") or [])) for sample in samples)
    return {
        "examples": len(samples),
        "unique_ids": len(set(ids)),
        "duplicate_ids": len(ids) - len(set(ids)),
        "positive_fraction": sum(int(sample.get("label", 0)) for sample in samples)
        / max(len(samples), 1),
        "evidence_fraction": evidence_examples / max(len(samples), 1),
        "max_sentences": max((len(sample.get("sentences") or []) for sample in samples), default=0),
    }


def batched_max_length(tokenizer, texts, batch_size=1024):
    maximum = 0
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_length=True,
        )
        maximum = max(maximum, max(encoded["length"], default=0))
    return maximum


def tokenization_statistics(tokenizer, samples):
    texts = sorted(
        {
            sample["context"] + " [SEP] " + sample["query"]
            for sample in samples
        }
    )
    queries = sorted({sample["query"] for sample in samples})
    sentences = sorted(
        {
            sentence
            for sample in samples
            for sentence in (sample.get("sentences") or [])
        }
    )
    return {
        "max_context_query_tokens": batched_max_length(tokenizer, texts),
        "max_query_tokens": batched_max_length(tokenizer, queries),
        "max_sentence_tokens": batched_max_length(tokenizer, sentences),
        "unique_context_queries": len(texts),
        "unique_queries": len(queries),
        "unique_sentences": len(sentences),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--proofwriter-root",
        default="/root/TRUA/data/proofwriter/raw/proofwriter-dataset-V2020.12.3",
    )
    parser.add_argument(
        "--ruletaker-root",
        default="/root/TRUA/data/rule-reasoning-dataset-V2020.2.5.0/original",
    )
    parser.add_argument("--backbone", action="append", default=None)
    args = parser.parse_args()

    splits = load_splits(Path(args.proofwriter_root), Path(args.ruletaker_root))
    report = {"datasets": {}, "limits": {"sentences": 32, "context_query": 512, "query": 64, "sentence": 96}}
    failures = []
    for dataset, dataset_splits in splits.items():
        report["datasets"][dataset] = {
            "splits": {
                name: split_statistics(samples)
                for name, samples in dataset_splits.items()
            },
            "id_overlaps": {},
            "tokenization": {},
        }
        id_sets = {
            name: {str(sample.get("id", "")) for sample in samples}
            for name, samples in dataset_splits.items()
        }
        for left, right in combinations(dataset_splits, 2):
            overlap = len(id_sets[left] & id_sets[right])
            report["datasets"][dataset]["id_overlaps"][f"{left}|{right}"] = overlap
            if overlap:
                failures.append(f"{dataset}: {left}/{right} overlap={overlap}")
        for split_name, stats in report["datasets"][dataset]["splits"].items():
            if stats["duplicate_ids"]:
                failures.append(
                    f"{dataset}/{split_name}: duplicate_ids={stats['duplicate_ids']}"
                )
            if stats["max_sentences"] > 32:
                failures.append(
                    f"{dataset}/{split_name}: max_sentences={stats['max_sentences']}"
                )

        all_samples = [
            sample for samples in dataset_splits.values() for sample in samples
        ]
        for specification in args.backbone or DEFAULT_BACKBONES:
            backbone, model_path = parse_backbone(specification)
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, local_files_only=True
            )
            stats = tokenization_statistics(tokenizer, all_samples)
            report["datasets"][dataset]["tokenization"][backbone] = stats
            if stats["max_context_query_tokens"] > 512:
                failures.append(
                    f"{dataset}/{backbone}: context_query={stats['max_context_query_tokens']}"
                )
            if stats["max_query_tokens"] > 64:
                failures.append(
                    f"{dataset}/{backbone}: query={stats['max_query_tokens']}"
                )
            if stats["max_sentence_tokens"] > 96:
                failures.append(
                    f"{dataset}/{backbone}: sentence={stats['max_sentence_tokens']}"
                )

    report["failures"] = failures
    report["pass"] = not failures
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
