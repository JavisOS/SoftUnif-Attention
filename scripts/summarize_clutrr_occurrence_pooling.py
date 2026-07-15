#!/usr/bin/env python3
"""Summarize the full CLUTRR 2--10 hop occurrence-pooling comparison."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SEEDS = (0, 1, 42)
MODES = ("first_occurrence", "all_occurrences")
HOPS = tuple(range(2, 11))


def summarize(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
        "values": values,
    }


def load_run(root: Path, mode: str, seed: int) -> dict:
    path = root / mode / f"seed_{seed}" / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed run: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_hop = payload["test"]["per_hop"]
    missing_hops = [hop for hop in HOPS if str(hop) not in per_hop]
    if missing_hops:
        raise ValueError(f"{path} is missing test hops {missing_hops}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs = {
        mode: {seed: load_run(args.root, mode, seed) for seed in SEEDS}
        for mode in MODES
    }
    summary = {
        "comparison_role": "adapter development on the released CLUTRR test suite",
        "selection_metric_requested": "mean full-test overall accuracy over hops 2--10",
        "seeds": list(SEEDS),
        "hops": list(HOPS),
        "modes": {},
    }

    for mode in MODES:
        overall = [float(runs[mode][seed]["test"]["overall"]) for seed in SEEDS]
        transition = [
            float(runs[mode][seed]["test"]["transition_at_1"])
            for seed in SEEDS
        ]
        hop_values = {
            str(hop): [
                float(runs[mode][seed]["test"]["per_hop"][str(hop)]["accuracy"])
                for seed in SEEDS
            ]
            for hop in HOPS
        }
        hop_macro = [
            statistics.mean(
                runs[mode][seed]["test"]["per_hop"][str(hop)]["accuracy"]
                for hop in HOPS
            )
            for seed in SEEDS
        ]
        summary["modes"][mode] = {
            "test_overall": summarize(overall),
            "test_hop_macro": summarize(hop_macro),
            "test_transition_at_1": summarize(transition),
            "test_per_hop": {
                hop: summarize(values) for hop, values in hop_values.items()
            },
        }

    first = summary["modes"]["first_occurrence"]
    all_occurrences = summary["modes"]["all_occurrences"]
    paired_overall = [
        all_value - first_value
        for all_value, first_value in zip(
            all_occurrences["test_overall"]["values"],
            first["test_overall"]["values"],
        )
    ]
    paired_macro = [
        all_value - first_value
        for all_value, first_value in zip(
            all_occurrences["test_hop_macro"]["values"],
            first["test_hop_macro"]["values"],
        )
    ]
    per_hop_delta = {
        str(hop): (
            all_occurrences["test_per_hop"][str(hop)]["mean"]
            - first["test_per_hop"][str(hop)]["mean"]
        )
        for hop in HOPS
    }
    selected = max(
        MODES,
        key=lambda mode: summary["modes"][mode]["test_overall"]["mean"],
    )
    summary["comparison"] = {
        "selected_by_requested_full_test_overall": selected,
        "paired_all_minus_first_overall": summarize(paired_overall),
        "paired_all_minus_first_hop_macro": summarize(paired_macro),
        "all_minus_first_per_hop_mean": per_hop_delta,
        "all_occurrences_hop_wins": sum(delta > 0.0 for delta in per_hop_delta.values()),
        "all_occurrences_hop_ties": sum(delta == 0.0 for delta in per_hop_delta.values()),
    }

    rendered = json.dumps(summary, indent=2, ensure_ascii=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
