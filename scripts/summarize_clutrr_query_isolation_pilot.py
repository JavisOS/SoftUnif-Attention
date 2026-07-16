#!/usr/bin/env python3
"""Summarize the paired CLUTRR query-isolation architecture pilot."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


VARIANTS = (
    "joint_guided",
    "joint_unguided",
    "separate_guided",
    "separate_unguided",
)
SEEDS = (0, 1, 42)
METRICS = ("overall", "long_hop", "transition_at_1")
CONTRASTS = {
    "explicit_goal_with_joint_encoding": ("joint_guided", "joint_unguided"),
    "explicit_goal_with_separate_encoding": (
        "separate_guided",
        "separate_unguided",
    ),
    "separate_vs_joint_when_guided": ("separate_guided", "joint_guided"),
    "separate_vs_joint_when_unguided": (
        "separate_unguided",
        "joint_unguided",
    ),
}


def mean_sd(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def load_runs(root: Path) -> tuple[dict, str]:
    runs = {}
    revisions = set()
    for variant in VARIANTS:
        runs[variant] = {}
        for seed in SEEDS:
            path = root / variant / f"seed_{seed}" / "metrics.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            revisions.add(payload["code_revision"])
            configuration = payload["configuration"]
            expected_mode = "separate" if variant.startswith("separate") else "joint"
            expected_guidance = variant.endswith("guided") and not variant.endswith("unguided")
            if configuration.get("unit_encoding_mode") != expected_mode:
                raise ValueError(f"Unexpected unit encoding in {path}")
            if configuration.get("use_goal_guidance") is not expected_guidance:
                raise ValueError(f"Unexpected goal-guidance setting in {path}")
            runs[variant][seed] = payload
    if len(revisions) != 1:
        raise ValueError(f"Mixed code revisions: {sorted(revisions)}")
    return runs, revisions.pop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    runs, revision = load_runs(args.root)
    aggregates = {}
    for variant in VARIANTS:
        aggregates[variant] = {
            metric: mean_sd(
                [runs[variant][seed]["test"][metric] for seed in SEEDS]
            )
            for metric in METRICS
        }

    contrasts = {}
    for name, (left, right) in CONTRASTS.items():
        contrasts[name] = {
            metric: mean_sd(
                [
                    runs[left][seed]["test"][metric]
                    - runs[right][seed]["test"][metric]
                    for seed in SEEDS
                ]
            )
            for metric in METRICS
        }

    summary = {
        "code_revision": revision,
        "seeds": list(SEEDS),
        "aggregates": aggregates,
        "paired_contrasts": contrasts,
    }
    (args.root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# CLUTRR Independent Query-Goal Pilot",
        "",
        f"Code revision: `{revision}`",
        "",
        "| Variant | Overall | Long-hop | Transition@1 |",
        "|---|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        cells = []
        for metric in METRICS:
            item = aggregates[variant][metric]
            cells.append(f"{item['mean']:.4f} +/- {item['sd']:.4f}")
        lines.append(f"| {variant} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Paired contrasts use the same seed and report left minus right.",
            "",
            "| Contrast | Overall delta | Long-hop delta | Transition@1 delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in CONTRASTS:
        cells = []
        for metric in METRICS:
            item = contrasts[name][metric]
            cells.append(f"{item['mean']:+.4f} +/- {item['sd']:.4f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    (args.root / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
