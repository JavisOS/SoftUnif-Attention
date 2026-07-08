#!/usr/bin/env python3
"""Aggregate CREST-style CLUTRR JSON outputs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


METRICS = ["overall", "short_hop", "long_hop", "rename_consistency", "reverse_cf_accuracy"]


def mean_std(values):
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def fmt(values):
    mean, std = mean_std(values)
    return f"{mean:.4f} +/- {std:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    result_dir = run_root / "results"
    rows = []
    for path in sorted(result_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data["run_meta"]
        rows.append(
            {
                "path": str(path),
                "seed": meta["seed"],
                "dataset": meta["dataset"],
                "model_type": meta["model_type"],
                "best": data["best"],
                "final": data["final"],
            }
        )

    grouped = {}
    for row in rows:
        key = (row["dataset"], row["model_type"])
        grouped.setdefault(key, []).append(row)

    md_lines = [
        "# CREST-style CLUTRR Counterfactual Baseline",
        "",
        f"Run root: `{run_root}`",
        "",
        "Method label: `crest_style_query_reverse_rename`.",
        "Official CREST code was not found in the public search performed for DOI `10.1016/j.ipm.2025.104418`; this is a fair-input adaptation using final-label supervision plus counterfactual query reversal/entity-renaming.",
        "",
        "Values are `mean +/- sample-std` over completed seeds.",
        "",
    ]

    aggregate = {"run_root": str(run_root), "groups": {}}
    for key, items in sorted(grouped.items()):
        dataset, model_type = key
        group_id = f"{dataset}/{model_type}"
        aggregate["groups"][group_id] = {"seeds": [item["seed"] for item in items], "best": {}, "final": {}}
        md_lines += [
            f"## {dataset} / {model_type}",
            "",
            "| Selection | Seeds | Overall | Short 2-3 | Long >=6 | Rename consistency | Reverse-CF accuracy |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for selection in ["best", "final"]:
            values = {metric: [item[selection]["eval"][metric] for item in items if item.get(selection)] for metric in METRICS}
            aggregate["groups"][group_id][selection] = {
                metric: {"mean": mean_std(vals)[0], "std": mean_std(vals)[1], "values": vals}
                for metric, vals in values.items()
            }
            md_lines.append(
                f"| {selection.capitalize()} | {len(items)} | {fmt(values['overall'])} | {fmt(values['short_hop'])} | {fmt(values['long_hop'])} | {fmt(values['rename_consistency'])} | {fmt(values['reverse_cf_accuracy'])} |"
            )
        md_lines += ["", "### Per-hop Best", ""]
        hop_ids = sorted(
            {
                int(h)
                for item in items
                for h in item["best"]["eval"].get("per_hop", {})
                if h.isdigit()
            }
        )
        md_lines.append("| Hop | Accuracy |")
        md_lines.append("| ---: | ---: |")
        per_hop = {}
        for hop in hop_ids:
            vals = [item["best"]["eval"]["per_hop"][str(hop)]["accuracy"] for item in items if str(hop) in item["best"]["eval"].get("per_hop", {})]
            per_hop[str(hop)] = {"mean": mean_std(vals)[0], "std": mean_std(vals)[1], "values": vals}
            md_lines.append(f"| {hop} | {fmt(vals)} |")
        aggregate["groups"][group_id]["best"]["per_hop"] = per_hop
        md_lines.append("")

    out_json = Path("/root/TRUA/docs/aggregated_results/CREST_CLUTRR_COUNTERFACTUAL.json")
    out_md = Path("/root/TRUA/docs/aggregated_results/CREST_CLUTRR_COUNTERFACTUAL.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
