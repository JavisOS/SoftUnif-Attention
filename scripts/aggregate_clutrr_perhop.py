#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


DEFAULT_RUN = Path("/vepfs/tsra_outputs/clutrr_perhop_representative/latest")
DEFAULT_OUT = Path("/root/TRUA/docs/aggregated_results")

EPOCH_RE = re.compile(r"Epoch (\d+)(?: Done|:)")
METRIC_RE = re.compile(
    r"\s*(Overall Acc \(Base\)|Short Hop \(2-3\)|Long Hop \(>=6\)):\s+([0-9.]+)"
)
PER_HOP_RE = re.compile(r"\s*Per-Hop Acc:\s+(.*)")
PER_HOP_ITEM_RE = re.compile(r"(\d+)=([0-9.]+)\s+\((\d+)/(\d+)\)")
NAME_RE = re.compile(r"clutrr_(089|db9)_(.+)_(vanilla|tsra)_seed(\d+)$")

DISPLAY = {
    "deberta": "DeBERTa",
    "deberta-v3": "DeBERTa-v3",
    "roberta": "RoBERTa",
}


def parse_per_hop(text: str) -> dict[int, dict]:
    out = {}
    for hop, acc, correct, total in PER_HOP_ITEM_RE.findall(text):
        out[int(hop)] = {
            "accuracy": float(acc),
            "correct": int(correct),
            "total": int(total),
        }
    return out


def parse_log(path: Path) -> list[dict]:
    epochs: list[dict] = []
    current: dict | None = None
    for line in path.read_text(errors="ignore").splitlines():
        m = EPOCH_RE.search(line)
        if m:
            current = {"epoch": int(m.group(1)), "per_hop": {}}
            epochs.append(current)
            continue
        if current is None:
            continue
        m = METRIC_RE.match(line)
        if m:
            key = {
                "Overall Acc (Base)": "overall",
                "Short Hop (2-3)": "short",
                "Long Hop (>=6)": "long",
            }[m.group(1)]
            current[key] = float(m.group(2))
            continue
        m = PER_HOP_RE.match(line)
        if m:
            current["per_hop"] = parse_per_hop(m.group(1))
    return [e for e in epochs if all(k in e for k in ("overall", "short", "long"))]


def parse_runs(run_root: Path) -> list[dict]:
    rows = []
    for log in sorted((run_root / "logs").glob("clutrr_*.log")):
        m = NAME_RE.match(log.stem)
        if not m:
            continue
        split_key, backbone, variant, seed = m.groups()
        epochs = parse_log(log)
        if not epochs:
            continue
        split = {"089": "data_089907f8", "db9": "data_db9b8f04"}[split_key]
        rows.append(
            {
                "name": log.stem,
                "split": split,
                "backbone": backbone.replace("_", "-"),
                "variant": variant,
                "seed": int(seed),
                "best": max(epochs, key=lambda e: e["overall"]),
                "final": max(epochs, key=lambda e: e["epoch"]),
                "epochs": epochs,
                "log": str(log),
            }
        )
    return rows


def fmt(values: list[float]) -> str:
    if not values:
        return "-"
    mean = sum(values) / len(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.4f} +/- {sd:.4f}"


def aggregate_runs(runs: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for run in runs:
        groups[(run["split"], run["backbone"], run["variant"])].append(run)

    rows = []
    for (split, backbone, variant), items in sorted(groups.items()):
        items = sorted(items, key=lambda r: r["seed"])
        row = {
            "split": split,
            "backbone": backbone,
            "variant": variant,
            "seeds": [r["seed"] for r in items],
            "n": len(items),
        }
        for phase in ("best", "final"):
            row[phase] = {
                "overall": fmt([r[phase]["overall"] for r in items]),
                "short": fmt([r[phase]["short"] for r in items]),
                "long": fmt([r[phase]["long"] for r in items]),
                "per_hop": {},
            }
            all_hops = sorted({h for r in items for h in r[phase].get("per_hop", {})})
            for hop in all_hops:
                row[phase]["per_hop"][str(hop)] = fmt(
                    [
                        r[phase]["per_hop"][hop]["accuracy"]
                        for r in items
                        if hop in r[phase].get("per_hop", {})
                    ]
                )
        rows.append(row)
    return rows


def table(rows: list[dict], split: str, phase: str) -> str:
    selected = [r for r in rows if r["split"] == split]
    if not selected:
        return "_No completed rows._"

    hops = sorted({int(h) for r in selected for h in r[phase]["per_hop"]})
    header = [
        "Backbone",
        "Variant",
        "Seeds",
        "Overall",
        "Short 2-3",
        "Long >=6",
        *[f"Hop {h}" for h in hops],
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---", "---", "---:", *["---:"] * (len(header) - 3)]) + " |",
    ]
    order = {"roberta": 0, "deberta": 1, "deberta-v3": 2}
    for r in sorted(selected, key=lambda x: (order.get(x["backbone"], 99), x["variant"])):
        metric = r[phase]
        values = [
            DISPLAY.get(r["backbone"], r["backbone"]),
            r["variant"].replace("_", "-"),
            str(r["n"]),
            metric["overall"],
            metric["short"],
            metric["long"],
        ]
        values.extend(metric["per_hop"].get(str(h), "-") for h in hops)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_markdown(run_root: Path, rows: list[dict]) -> str:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# CLUTRR Per-Hop Representative Rerun",
        "",
        f"Generated on {generated} from `{run_root}`.",
        "",
        "Values are `mean +/- sample-std` over seeds `0/1/42` when all three seeds are complete.",
        "`Best` selects the epoch with the highest overall accuracy for each seed; `Final` uses the last logged epoch.",
        "",
        "## data_089907f8 Best Epoch",
        "",
        table(rows, "data_089907f8", "best"),
        "",
        "## data_089907f8 Final Epoch",
        "",
        table(rows, "data_089907f8", "final"),
        "",
        "## data_db9b8f04 Best Epoch",
        "",
        table(rows, "data_db9b8f04", "best"),
        "",
        "## data_db9b8f04 Final Epoch",
        "",
        table(rows, "data_db9b8f04", "final"),
        "",
        "## Configuration Notes",
        "",
        "- `data_089907f8`: primary CLUTRR split, 2/3-hop training and 2-10-hop testing.",
        "- `data_db9b8f04`: follow-up CLUTRR split, 2/3/4-hop training and long-hop testing.",
        "- `vanilla`: plain story+query Transformer classifier trained with final-label cross entropy only.",
        "- `tsra`: TRUA architecture with `lambda_nexthop=1.0`, `lambda_edge=1.0`, `lambda_consistency=5.0`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--name", default="CLUTRR_PERHOP_REPRESENTATIVE")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = parse_runs(run_root)
    rows = aggregate_runs(runs)

    payload = {
        "run_root": str(run_root),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runs": runs,
        "rows": rows,
    }
    json_path = args.out_dir / f"{args.name}.json"
    md_path = args.out_dir / f"{args.name}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(run_root, rows), encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
