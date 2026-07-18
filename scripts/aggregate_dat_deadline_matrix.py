#!/usr/bin/env python3
"""Audit and aggregate the deadline-scoped DeBERTa-v3 DAT comparison."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


SEEDS = (0, 1, 42)


def mean_std(values):
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
        "values": values,
    }


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def proposition_paths(root, dataset):
    base = root / ("proofwriter" if dataset == "proofwriter" else "proposition")
    return [
        base
        / dataset
        / "deberta-v3"
        / "dual_attention"
        / f"seed_{seed}"
        / "metrics.json"
        for seed in SEEDS
    ]


def aggregate_proposition(root, dataset):
    records = [read(path) for path in proposition_paths(root, dataset)]
    for seed, record in zip(SEEDS, records):
        assert record["dataset"] == ("ruletaker_raw" if dataset == "ruletaker" else dataset)
        assert record["seed"] == seed
        assert record["epochs"] == 10
        assert record["architecture"]["core_type"] == "dual_attention"
        assert record["lambda_evidence"] == 0.0
        assert record["model_name"].endswith("deberta-v3-base")
    split_names = list(records[0]["results"])
    return {
        split: {
            metric: mean_std([record["results"][split][metric] for record in records])
            for metric in ("accuracy", "evidence_at_1")
        }
        for split in split_names
    }


def aggregate_clutrr(root):
    records = [
        read(
            root
            / "clutrr"
            / "deberta-v3"
            / "dual_attention"
            / f"seed_{seed}"
            / "metrics.json"
        )
        for seed in SEEDS
    ]
    for seed, record in zip(SEEDS, records):
        assert record["seed"] == seed
        assert record["core_type"] == "dual_attention"
        assert record["checkpoint_selection"] == "validation"
        assert record["configuration"]["path_or_edge_supervision"] is False
    return {
        metric: mean_std([record["test"][metric] for record in records])
        for metric in ("overall", "short_hop", "long_hop", "transition_at_1")
    }


def markdown(summary):
    def formatted(metric):
        return f"{metric['mean']:.4f} +/- {metric['sample_std']:.4f}"

    rows = [
        "# Formal DeBERTa-v3 DAT Results",
        "",
        "All rows use the task's audited text adapter, ten epochs, validation-only checkpoint selection, and seeds 0/1/42. DAT is trained with final labels only.",
        "",
        "| Dataset | Answer accuracy | Intermediate selection |",
        "| --- | ---: | ---: |",
        f"| CLUTRR | {formatted(summary['clutrr']['overall'])} | {formatted(summary['clutrr']['transition_at_1'])} |",
        f"| ProofWriter D3 | {formatted(summary['proofwriter']['depth-3']['accuracy'])} | {formatted(summary['proofwriter']['depth-3']['evidence_at_1'])} |",
        f"| ProofWriter D5 | {formatted(summary['proofwriter']['depth-5']['accuracy'])} | {formatted(summary['proofwriter']['depth-5']['evidence_at_1'])} |",
        f"| RuleTaker | {formatted(summary['ruletaker']['test']['accuracy'])} | {formatted(summary['ruletaker']['test']['evidence_at_1'])} |",
        f"| P-FOLIO | {formatted(summary['pfolio']['test']['accuracy'])} | {formatted(summary['pfolio']['test']['evidence_at_1'])} |",
        "",
    ]
    return "\n".join(rows)


def main():
    root = Path(sys.argv[1]).resolve()
    summary = {
        "scope": "DeBERTa-v3 DAT adapted with the official DualAttention module",
        "seeds": list(SEEDS),
        "clutrr": aggregate_clutrr(root),
        "proofwriter": aggregate_proposition(root, "proofwriter"),
        "ruletaker": aggregate_proposition(root, "ruletaker"),
        "pfolio": aggregate_proposition(root, "pfolio"),
    }
    (root / "aggregate.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "aggregate.md").write_text(markdown(summary), encoding="utf-8")
    print(markdown(summary))


if __name__ == "__main__":
    main()

