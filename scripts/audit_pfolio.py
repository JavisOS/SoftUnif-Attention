#!/usr/bin/env python3
"""Audit P-FOLIO parsing, official split mapping, and evidence coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from generic_trua_prop import PFOLIO_LABEL_NAMES, load_pfolio_corpus


def split_fingerprint(samples):
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(sample["id"].encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def tokenizer_audit(specification, samples):
    from transformers import AutoTokenizer

    try:
        name, model_path = specification.split("=", 1)
    except ValueError as error:
        raise ValueError("Tokenizer specifications must use NAME=MODEL_PATH") from error
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    pair_lengths = []
    query_lengths = []
    sentence_lengths = []
    for sample in samples:
        pair_lengths.append(
            len(
                tokenizer(
                    sample["context"],
                    sample["query"],
                    add_special_tokens=True,
                    truncation=False,
                )["input_ids"]
            )
        )
        query_lengths.append(
            len(tokenizer(sample["query"], add_special_tokens=True, truncation=False)["input_ids"])
        )
        sentence_lengths.extend(
            len(tokenizer(sentence, add_special_tokens=True, truncation=False)["input_ids"])
            for sentence in sample["sentences"]
        )
    return name, {
        "context_query": {
            "maximum": max(pair_lengths),
            "p95": percentile(pair_lengths, 0.95),
            "over_512": sum(length > 512 for length in pair_lengths),
            "over_512_fraction": sum(length > 512 for length in pair_lengths) / len(pair_lengths),
        },
        "query": {
            "maximum": max(query_lengths),
            "p95": percentile(query_lengths, 0.95),
            "over_64": sum(length > 64 for length in query_lengths),
        },
        "premise_unit": {
            "maximum": max(sentence_lengths),
            "p95": percentile(sentence_lengths, 0.95),
            "over_96": sum(length > 96 for length in sentence_lengths),
            "over_96_fraction": sum(length > 96 for length in sentence_lengths)
            / len(sentence_lengths),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out")
    parser.add_argument(
        "--tokenizer",
        action="append",
        default=[],
        help="Repeat NAME=LOCAL_MODEL_PATH to audit the formal token limits.",
    )
    args = parser.parse_args()

    samples, alignment_audit = load_pfolio_corpus(Path(args.root), return_audit=True)
    report = {
        "dataset": "P-FOLIO",
        "examples": len(samples),
        "splits": {},
        "max_premises": max(len(sample["sentences"]) for sample in samples),
        "max_proof_steps": max(sample["depth"] for sample in samples),
        "alignment": alignment_audit,
    }
    split_ids = {}
    for split in ("train", "validation", "test"):
        subset = [sample for sample in samples if sample["split"] == split]
        split_ids[split] = {sample["id"] for sample in subset}
        labels = Counter(sample["label"] for sample in subset)
        evidence_examples = sum(any(sample["trace_labels"]) for sample in subset)
        evidence_status = Counter(sample["evidence_status"] for sample in subset)
        report["splits"][split] = {
            "examples": len(subset),
            "sha256": split_fingerprint(subset),
            "labels": {
                PFOLIO_LABEL_NAMES[label]: count
                for label, count in sorted(labels.items())
            },
            "evidence_examples": evidence_examples,
            "evidence_fraction": evidence_examples / max(len(subset), 1),
            "evidence_status": dict(sorted(evidence_status.items())),
            "proof_depth": dict(sorted(Counter(sample["depth"] for sample in subset).items())),
        }
    report["split_overlap"] = {
        "train_validation": len(split_ids["train"] & split_ids["validation"]),
        "train_test": len(split_ids["train"] & split_ids["test"]),
        "validation_test": len(split_ids["validation"] & split_ids["test"]),
    }
    if any(report["split_overlap"].values()):
        raise ValueError(f"P-FOLIO split overlap detected: {report['split_overlap']}")
    report["tokenization"] = dict(
        tokenizer_audit(specification, samples) for specification in args.tokenizer
    )

    rendered = json.dumps(report, indent=2, ensure_ascii=True) + "\n"
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
