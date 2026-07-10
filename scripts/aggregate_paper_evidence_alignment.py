#!/usr/bin/env python3
"""Aggregate validation-selected paper experiments from result JSON files."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


NAME_RE = re.compile(r"(.+)_seed(\d+)$")


def mean_std(values):
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std, "values": values}


def fmt(summary):
    return f"{summary['mean']:.4f} +/- {summary['std']:.4f}"


def load_groups(result_dir):
    groups = defaultdict(list)
    for path in sorted(result_dir.glob("*.json")):
        match = NAME_RE.match(path.stem)
        if not match:
            continue
        group, seed = match.groups()
        payload = json.loads(path.read_text(encoding="utf-8"))
        groups[group].append((int(seed), payload))
    return groups


def aggregate_group(group, items):
    items = sorted(items)
    row = {"group": group, "seeds": [seed for seed, _ in items]}
    if group.startswith("clutrr_"):
        tests = [payload["test"] for _, payload in items]
        row["kind"] = "clutrr"
        row["selected_epoch"] = mean_std([payload["selected_epoch"] for _, payload in items])
        row["selected_validation"] = mean_std(
            [payload["selected_validation_overall"] for _, payload in items]
        )
        for metric in ("overall", "short_hop", "long_hop"):
            row[metric] = mean_std([test[metric] for test in tests])
        for metric in ("renamed", "consistency", "consistent_and_correct"):
            values = [test[metric] for test in tests if metric in test]
            if values:
                row[metric] = mean_std(values)
        transition_values = [test["transition_at_1"] for test in tests if "transition_at_1" in test]
        if len(transition_values) == len(tests):
            row["transition_at_1"] = mean_std(transition_values)
        hops = sorted({hop for test in tests for hop in test.get("per_hop", {})}, key=int)
        row["per_hop"] = {
            hop: mean_std([test["per_hop"][hop]["accuracy"] for test in tests if hop in test.get("per_hop", {})])
            for hop in hops
        }
    else:
        row["kind"] = "proposition"
        row["selected_epoch"] = mean_std([payload["selected_epoch"] for _, payload in items])
        row["selected_validation"] = mean_std(
            [payload["selected_validation_accuracy"] for _, payload in items]
        )
        selected_evidence = [
            payload["selected_validation_evidence_at_1"]
            for _, payload in items
            if "selected_validation_evidence_at_1" in payload
        ]
        if selected_evidence:
            row["selected_validation_evidence_at_1"] = mean_std(selected_evidence)
        splits = sorted({split for _, payload in items for split in payload["results"]})
        row["splits"] = {}
        for split in splits:
            results = [payload["results"][split] for _, payload in items if split in payload["results"]]
            depths = sorted({depth for result in results for depth in result.get("by_depth", {})}, key=int)
            row["splits"][split] = {
                "accuracy": mean_std([result["accuracy"] for result in results]),
                "evidence_at_1": mean_std(
                    [
                        result["evidence_at_1"]
                        if "evidence_at_1" in result
                        else result["trace_top1"]
                        for result in results
                    ]
                ),
                "by_depth": {
                    depth: mean_std(
                        [result["by_depth"][depth] for result in results if depth in result.get("by_depth", {})]
                    )
                    for depth in depths
                },
            }
    return row


def markdown(rows):
    lines = [
        "# Validation-Selected TRUA Evidence Alignment Results",
        "",
        "All checkpoints are selected on a fixed training-set validation partition. The test set is evaluated once.",
        "",
        "## CLUTRR",
        "",
        "| Group | Seeds | Selected epoch | Overall | Short | Long >=6 | Transition@1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["kind"] != "clutrr":
            continue
        lines.append(
            f"| {row['group']} | {len(row['seeds'])} | {fmt(row['selected_epoch'])} | "
            f"{fmt(row['overall'])} | {fmt(row['short_hop'])} | {fmt(row['long_hop'])} | "
            f"{fmt(row['transition_at_1']) if 'transition_at_1' in row else '--'} |"
        )
    lines.extend(
        [
            "",
            "## Proposition Tasks",
            "",
            "| Group | Split | Seeds | Selected epoch | Accuracy | Evidence@1 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        if row["kind"] != "proposition":
            continue
        for split, metrics in row["splits"].items():
            lines.append(
                f"| {row['group']} | {split} | {len(row['seeds'])} | {fmt(row['selected_epoch'])} | "
                f"{fmt(metrics['accuracy'])} | {fmt(metrics['evidence_at_1'])} |"
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    groups = load_groups(args.run_root / "results")
    rows = [aggregate_group(group, items) for group, items in sorted(groups.items())]
    (args.run_root / "aggregated.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.run_root / "aggregated.md").write_text(markdown(rows), encoding="utf-8")
    print(f"aggregated {len(rows)} groups from {sum(len(items) for items in groups.values())} runs")


if __name__ == "__main__":
    main()
