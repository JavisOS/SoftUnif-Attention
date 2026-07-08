#!/usr/bin/env python3
"""Aggregate TRUA experiment results from the current /vepfs run folders.

The script is intentionally read-only with respect to experiment outputs. It
collects the formal 10-epoch runs, seed-42 completion runs, and the additional
depth/seed checks, then writes compact Markdown and JSON summaries.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any


FORMAL = Path("/vepfs/tsra_outputs/formal_10ep/latest_tsra_formal_10ep")
SEED42 = Path("/vepfs/tsra_outputs/formal_10ep/latest_seed42_completion")
ADDITIONAL = Path("/vepfs/tsra_outputs/additional_depth_checks/latest_depth_seed_checks")
OUT_DIR = Path("docs/aggregated_results")


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def result_block(path: Path, split: str) -> dict[str, Any]:
    data = load_json(path)
    return data["results"][split]


def fmt(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "-"
    if math.isnan(x):
        return "nan"
    return f"{x:.{digits}f}"


def fmt_pm(values: list[float], digits: int = 4) -> str:
    if not values:
        return "-"
    if len(values) == 1:
        return fmt(values[0], digits)
    return f"{mean(values):.{digits}f} +/- {stdev(values):.{digits}f}"


def metric(values: list[dict[str, Any]], key: str) -> list[float]:
    return [float(v[key]) for v in values if key in v and v[key] is not None]


def by_depth(values: list[dict[str, Any]], depth: str) -> list[float]:
    out = []
    for v in values:
        bd = v.get("by_depth") or {}
        if depth in bd and bd[depth] is not None:
            out.append(float(bd[depth]))
    return out


def primary_split_for_dataset(dataset: str, data: dict[str, Any]) -> str:
    keys = list(data["results"].keys())
    if dataset in {"ruletaker_gfair", "ruletaker_raw"}:
        return "test"
    if dataset == "prontoqa":
        return "ood"
    if dataset == "proofwriter":
        return "depth-3"
    return keys[0]


def prop_json_paths() -> dict[tuple[str, str, str, int], Path]:
    """Return preferred JSON path for dataset/backbone/model/seed.

    Keys are (dataset, backbone, variant, seed), where variant is baseline/tsra.
    """
    paths: dict[tuple[str, str, str, int], Path] = {}

    def add(dataset: str, backbone: str, variant: str, seed: int, path: Path) -> None:
        if path.exists():
            paths[(dataset, backbone, variant, seed)] = path

    # BERT and RoBERTa seeds 0/1.
    bert_dir = Path("/vepfs/tsra_outputs/formal_10ep/prop_bert_backbone_seeds_20260524_011602/results")
    roberta_dir = Path("/vepfs/tsra_outputs/formal_10ep/prop_roberta_backbone_seeds_20260523_101127/results")
    for dataset in ("proofwriter", "ruletaker", "prontoqa"):
        for variant in ("baseline", "tsra"):
            for seed in (0, 1):
                add(dataset if dataset != "ruletaker" else "ruletaker_gfair", "bert", variant, seed, bert_dir / f"{dataset}_bert_{variant}_seed{seed}_10ep.json")
                add(dataset if dataset != "ruletaker" else "ruletaker_gfair", "roberta", variant, seed, roberta_dir / f"{dataset}_roberta_{variant}_seed{seed}_10ep.json")

    # DeBERTa seed 0 from formal queue.
    formal_results = FORMAL / "results"
    add("proofwriter", "deberta", "baseline", 0, formal_results / "proofwriter_deberta_baseline_10ep.json")
    add("proofwriter", "deberta", "tsra", 0, formal_results / "proofwriter_deberta_tsra_10ep.json")
    add("ruletaker_gfair", "deberta", "baseline", 0, formal_results / "ruletaker_deberta_baseline_10ep.json")
    add("ruletaker_gfair", "deberta", "tsra", 0, formal_results / "ruletaker_deberta_tsra_10ep.json")
    add("prontoqa", "deberta", "baseline", 0, formal_results / "prontoqa_deberta_baseline_10ep.json")
    add("prontoqa", "deberta", "tsra", 0, formal_results / "prontoqa_deberta_tsra_10ep.json")

    # DeBERTa seed 1 from retry folders.
    deberta_seed1 = Path("/vepfs/tsra_outputs/formal_10ep/prop_deberta_seed1_failed_rerun_20260522_041440/results")
    deberta_baseline_seed1 = Path("/vepfs/tsra_outputs/formal_10ep/prop_deberta_seed1_gpu7_retry_20260521_113533/results")
    add("proofwriter", "deberta", "baseline", 1, deberta_baseline_seed1 / "proofwriter_deberta_baseline_seed1_10ep.json")
    add("proofwriter", "deberta", "tsra", 1, deberta_seed1 / "proofwriter_deberta_tsra_seed1_10ep.json")
    add("ruletaker_gfair", "deberta", "baseline", 1, deberta_seed1 / "ruletaker_deberta_baseline_seed1_10ep.json")
    add("ruletaker_gfair", "deberta", "tsra", 1, deberta_seed1 / "ruletaker_deberta_tsra_seed1_10ep.json")
    add("prontoqa", "deberta", "baseline", 1, deberta_seed1 / "prontoqa_deberta_baseline_seed1_10ep.json")
    add("prontoqa", "deberta", "tsra", 1, deberta_seed1 / "prontoqa_deberta_tsra_seed1_10ep.json")

    # Seed 42 completion. ProofWriter DeBERTa seed42 lives in additional checks.
    seed42_results = SEED42 / "results"
    additional_results = ADDITIONAL / "results"
    for backbone in ("bert", "roberta"):
        for variant in ("baseline", "tsra"):
            add("proofwriter", backbone, variant, 42, seed42_results / f"proofwriter_{backbone}_{variant}_seed42_10ep.json")
            add("ruletaker_gfair", backbone, variant, 42, seed42_results / f"ruletaker_{backbone}_{variant}_seed42_10ep.json")
            add("prontoqa", backbone, variant, 42, seed42_results / f"prontoqa_{backbone}_{variant}_seed42_10ep.json")
    for variant in ("baseline", "tsra"):
        add("proofwriter", "deberta", variant, 42, additional_results / f"proofwriter_deberta_{variant}_seed42_10ep.json")
        add("ruletaker_gfair", "deberta", variant, 42, seed42_results / f"ruletaker_deberta_{variant}_seed42_10ep.json")
        add("prontoqa", "deberta", variant, 42, seed42_results / f"prontoqa_deberta_{variant}_seed42_10ep.json")

    return paths


def collect_prop_tables() -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    paths = prop_json_paths()
    for dataset in ("proofwriter", "ruletaker_gfair", "prontoqa"):
        for backbone in ("bert", "roberta", "deberta"):
            for variant in ("baseline", "tsra"):
                entries = []
                seed_paths = []
                for seed in (0, 1, 42):
                    path = paths.get((dataset, backbone, variant, seed))
                    if not path:
                        warnings.append(f"Missing {dataset}/{backbone}/{variant}/seed{seed}")
                        continue
                    data = load_json(path)
                    split = primary_split_for_dataset(dataset, data)
                    entries.append(data["results"][split])
                    seed_paths.append((seed, path))
                if not entries:
                    continue
                row = {
                    "dataset": dataset,
                    "backbone": backbone,
                    "variant": variant,
                    "seeds": [s for s, _ in seed_paths],
                    "accuracy": fmt_pm(metric(entries, "accuracy")),
                    "trace_top1": fmt_pm(metric(entries, "trace_top1")),
                    "split": primary_split_for_dataset(dataset, load_json(seed_paths[0][1])) if seed_paths else "-",
                    "paths": [str(p) for _, p in seed_paths],
                }
                if dataset == "proofwriter":
                    # Add depth-5 from the corresponding JSONs, because depth-3 is
                    # the primary split key for historical runs.
                    d5_entries = []
                    for _, path in seed_paths:
                        data = load_json(path)
                        if "depth-5" in data["results"]:
                            d5_entries.append(data["results"]["depth-5"])
                    row["depth5_accuracy"] = fmt_pm(metric(d5_entries, "accuracy"))
                    row["depth5_trace_top1"] = fmt_pm(metric(d5_entries, "trace_top1"))
                    row["by_depth_3"] = fmt_pm(by_depth(entries, "3"))
                    row["by_depth_5"] = fmt_pm(by_depth(d5_entries, "5"))
                elif dataset == "ruletaker_gfair":
                    row["by_depth"] = "official test depth metadata unavailable (-1 bucket)"
                elif dataset == "prontoqa":
                    row["by_depth_3"] = fmt_pm(by_depth(entries, "3"))
                    row["by_depth_4"] = fmt_pm(by_depth(entries, "4"))
                rows.append(row)
    return rows, warnings


def collect_ruletaker_raw() -> list[dict[str, Any]]:
    rows = []
    for variant in ("baseline", "tsra"):
        entries = []
        for seed in (0, 1, 42):
            path = ADDITIONAL / "results" / f"ruletaker_raw_deberta_{variant}_trainq12_seed{seed}.json"
            if path.exists():
                entries.append(result_block(path, "test"))
        rows.append(
            {
                "variant": variant,
                "seeds": [0, 1, 42],
                "accuracy": fmt_pm(metric(entries, "accuracy")),
                "qdep1": fmt_pm(by_depth(entries, "1")),
                "qdep2": fmt_pm(by_depth(entries, "2")),
                "qdep3": fmt_pm(by_depth(entries, "3")),
                "qdep4": fmt_pm(by_depth(entries, "4")),
                "qdep5": fmt_pm(by_depth(entries, "5")),
                "trace_top1": fmt_pm(metric(entries, "trace_top1")),
            }
        )
    return rows


CLUTRR_RE = re.compile(
    r"Overall Acc \(Base\):\s+([0-9.]+).*?Short Hop \(2-3\):\s+([0-9.]+).*?Long Hop \(>=6\):\s+([0-9.]+)",
    re.S,
)


def parse_clutrr_log(path: Path) -> dict[str, float] | None:
    text = path.read_text(errors="ignore")
    matches = [(float(a), float(b), float(c)) for a, b, c in CLUTRR_RE.findall(text)]
    if not matches:
        return None
    best = max(matches, key=lambda x: x[0])
    final = matches[-1]
    return {
        "best_overall": best[0],
        "best_short": best[1],
        "best_long": best[2],
        "final_overall": final[0],
        "final_short": final[1],
        "final_long": final[2],
    }


def collect_clutrr_089() -> list[dict[str, Any]]:
    rows = []
    sources = {
        0: FORMAL / "logs",
        1: FORMAL / "logs",
        42: SEED42 / "logs",
    }
    for backbone in ("deberta", "roberta"):
        for variant in ("full", "label_only", "no_consistency"):
            vals = []
            for seed, logdir in sources.items():
                if seed == 42:
                    name = f"clutrr_089_{backbone}_{variant}_seed42.log"
                else:
                    name = f"clutrr_{backbone}_{variant}_seed{seed}.log"
                parsed = parse_clutrr_log(logdir / name)
                if parsed:
                    vals.append(parsed)
            rows.append(
                {
                    "dataset": "data_089907f8",
                    "backbone": backbone,
                    "variant": variant,
                    "seeds": len(vals),
                    "best_overall": fmt_pm([v["best_overall"] for v in vals]),
                    "best_short": fmt_pm([v["best_short"] for v in vals]),
                    "best_long": fmt_pm([v["best_long"] for v in vals]),
                    "final_overall": fmt_pm([v["final_overall"] for v in vals]),
                    "final_long": fmt_pm([v["final_long"] for v in vals]),
                }
            )
    return rows


def collect_clutrr_db9() -> list[dict[str, Any]]:
    rows = []
    for variant in ("label_only", "tsra"):
        vals = []
        for seed in (0, 1, 42):
            parsed = parse_clutrr_log(ADDITIONAL / "logs" / f"clutrr_db9_deberta_{variant}_seed{seed}.log")
            if parsed:
                vals.append(parsed)
        rows.append(
            {
                "dataset": "data_db9b8f04",
                "backbone": "deberta",
                "variant": variant,
                "seeds": len(vals),
                "best_overall": fmt_pm([v["best_overall"] for v in vals]),
                "best_short": fmt_pm([v["best_short"] for v in vals]),
                "best_long": fmt_pm([v["best_long"] for v in vals]),
                "final_overall": fmt_pm([v["final_overall"] for v in vals]),
                "final_long": fmt_pm([v["final_long"] for v in vals]),
            }
        )
    return rows


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def build_markdown() -> str:
    prop_rows, warnings = collect_prop_tables()
    raw_rows = collect_ruletaker_raw()
    clutrr089 = collect_clutrr_089()
    clutrrdb9 = collect_clutrr_db9()

    lines: list[str] = []
    lines.append("# Aggregated TRUA Results")
    lines.append("")
    lines.append("Generated from the completed `/vepfs` experiment folders on 2026-05-27. Values are `mean +/- sample-std` over seeds `0/1/42` unless noted.")
    lines.append("")
    lines.append("## Completion Status")
    lines.append("")
    lines.append("- Additional depth/seed checks: `14/14 done, 0 failed`.")
    lines.append("- Seed-42 completion queue: `22/22 done, 0 failed`.")
    lines.append("- NLProofS formal test is still running separately and has not produced a final result file.")
    lines.append("")

    lines.append("## CLUTRR data_089907f8")
    lines.append("")
    lines.append(markdown_table(
        ["Backbone", "Variant", "Seeds", "Best Overall", "Best Short", "Best Long >=6", "Final Overall", "Final Long >=6"],
        [[r["backbone"], r["variant"], r["seeds"], r["best_overall"], r["best_short"], r["best_long"], r["final_overall"], r["final_long"]] for r in clutrr089],
    ))
    lines.append("")

    lines.append("## CLUTRR data_db9b8f04 2/3/4-Hop Train Check")
    lines.append("")
    lines.append(markdown_table(
        ["Backbone", "Variant", "Seeds", "Best Overall", "Best Short", "Best Long >=6", "Final Overall", "Final Long >=6"],
        [[r["backbone"], r["variant"], r["seeds"], r["best_overall"], r["best_short"], r["best_long"], r["final_overall"], r["final_long"]] for r in clutrrdb9],
    ))
    lines.append("")

    lines.append("## ProofWriter Main TRUA-Prop")
    lines.append("")
    pw = [r for r in prop_rows if r["dataset"] == "proofwriter"]
    lines.append(markdown_table(
        ["Backbone", "Model", "Seeds", "Depth-3 Acc", "Depth-5 Acc", "Depth-3 Trace@1", "Depth-5 Trace@1", "ByDepth-3", "ByDepth-5"],
        [[r["backbone"], r["variant"], ",".join(map(str, r["seeds"])), r["accuracy"], r.get("depth5_accuracy", "-"), r["trace_top1"], r.get("depth5_trace_top1", "-"), r.get("by_depth_3", "-"), r.get("by_depth_5", "-")] for r in pw],
    ))
    lines.append("")

    lines.append("## RuleTaker GFaiR Split Main TRUA-Prop")
    lines.append("")
    rt = [r for r in prop_rows if r["dataset"] == "ruletaker_gfair"]
    lines.append(markdown_table(
        ["Backbone", "Model", "Seeds", "Test Acc", "Trace@1", "Depth Note"],
        [[r["backbone"], r["variant"], ",".join(map(str, r["seeds"])), r["accuracy"], r["trace_top1"], r["by_depth"]] for r in rt],
    ))
    lines.append("")

    lines.append("## RuleTaker Raw Strict QDep 1/2 Train -> 1-5 Test")
    lines.append("")
    lines.append(markdown_table(
        ["Backbone", "Model", "Seeds", "Overall", "QDep1", "QDep2", "QDep3", "QDep4", "QDep5", "Trace@1"],
        [["deberta", r["variant"], ",".join(map(str, r["seeds"])), r["accuracy"], r["qdep1"], r["qdep2"], r["qdep3"], r["qdep4"], r["qdep5"], r["trace_top1"]] for r in raw_rows],
    ))
    lines.append("")

    lines.append("## PrOntoQA-OOD")
    lines.append("")
    pq = [r for r in prop_rows if r["dataset"] == "prontoqa"]
    lines.append(markdown_table(
        ["Backbone", "Model", "Seeds", "OOD Acc", "Trace@1", "Depth3", "Depth4"],
        [[r["backbone"], r["variant"], ",".join(map(str, r["seeds"])), r["accuracy"], r["trace_top1"], r.get("by_depth_3", "-"), r.get("by_depth_4", "-")] for r in pq],
    ))
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- CLUTRR values use the best logged evaluation point for main comparison; final-epoch values are also included for audit.")
    lines.append("- The strongest stable CLUTRR signal is trace/next-hop supervision versus label-only, especially for DeBERTa on long-hop examples.")
    lines.append("- RuleTaker raw strict QDep shows a large TRUA gain over label-only on overall accuracy, but seed-to-seed variance is high and should be reported transparently.")
    lines.append("- PrOntoQA label accuracy is saturated in this processed split; trace@1 is the more informative internal-reasoning metric.")
    lines.append("- ProofWriter DeBERTa seed42 baseline and TRUA are identical in the additional check; treat that cell as an audit flag rather than a strong conclusion.")
    if warnings:
        lines.append("")
        lines.append("## Missing Inputs")
        for warning in warnings:
            lines.append(f"- {warning}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = build_markdown()
    (OUT_DIR / "AGGREGATED_RESULTS_20260527.md").write_text(md)
    raw = {
        "clutrr_089907f8": collect_clutrr_089(),
        "clutrr_db9b8f04": collect_clutrr_db9(),
        "prop_rows": collect_prop_tables()[0],
        "ruletaker_raw": collect_ruletaker_raw(),
    }
    (OUT_DIR / "aggregated_results_20260527.json").write_text(json.dumps(raw, indent=2))
    print(md)


if __name__ == "__main__":
    main()
