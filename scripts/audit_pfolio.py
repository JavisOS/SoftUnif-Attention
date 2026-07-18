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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out")
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

    rendered = json.dumps(report, indent=2, ensure_ascii=True) + "\n"
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
