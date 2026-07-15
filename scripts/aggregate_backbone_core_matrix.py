#!/usr/bin/env python3
"""Aggregate the audited backbone/core matrix into paper-ready artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SEEDS = (0, 1, 42)
BACKBONES = ("bert", "roberta", "deberta-v3")
CORES = ("encoder", "self_attention", "trua")
CORE_ALIASES = {"self_attention_matched": "self_attention"}
METHOD_LABELS = {
    "encoder": "Encoder",
    "self_attention": "Content self-attention",
    "trua": "TRUA",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clutrr-root", required=True)
    parser.add_argument(
        "--proposition-root",
        action="append",
        required=True,
        help="Repeat for separately scheduled proposition waves.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--output-tex", required=True)
    return parser


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read result: {path}") from error


def summary(values: list[float]) -> dict:
    if len(values) != len(SEEDS):
        raise ValueError(f"Expected {len(SEEDS)} values, found {len(values)}")
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
        "values": values,
    }


def optional_summary(values: list[float | None]) -> dict | None:
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"Partially missing metric: {values}")
    return summary([float(value) for value in values if value is not None])


def collect_clutrr(root: Path) -> dict:
    runs = {}
    for path in root.glob("*/*/seed_*/metrics.json"):
        relative = path.relative_to(root)
        backbone, raw_core, seed_dir, _ = relative.parts
        core = CORE_ALIASES.get(raw_core, raw_core)
        seed = int(seed_dir.removeprefix("seed_"))
        key = (backbone, core, seed)
        if key in runs:
            raise ValueError(f"Duplicate CLUTRR result: {key}")
        runs[key] = load_json(path)

    aggregated = {}
    for backbone in BACKBONES:
        aggregated[backbone] = {}
        for core in CORES:
            selected = []
            for seed in SEEDS:
                key = (backbone, core, seed)
                if key not in runs:
                    raise ValueError(f"Missing CLUTRR result: {key}")
                selected.append(runs[key])
            test_sizes = {run["test_size"] for run in selected}
            if len(test_sizes) != 1:
                raise ValueError(
                    f"CLUTRR test-size mismatch for {backbone}/{core}: {test_sizes}"
                )
            per_hop = {}
            for hop in map(str, range(2, 11)):
                per_hop[hop] = summary(
                    [float(run["test"]["per_hop"][hop]["accuracy"]) for run in selected]
                )
            aggregated[backbone][core] = {
                "seeds": list(SEEDS),
                "code_revisions": sorted({run["code_revision"] for run in selected}),
                "test_size": test_sizes.pop(),
                "overall": summary([float(run["test"]["overall"]) for run in selected]),
                "transition_at_1": None
                if core == "encoder"
                else optional_summary(
                    [
                        run["test"].get("transition_at_1")
                        if run["test"].get("transition_total", 0) > 0
                        else None
                        for run in selected
                    ]
                ),
                "per_hop": per_hop,
            }
    return aggregated


def collect_proposition(roots: list[Path]) -> dict:
    runs = {}
    for root in roots:
        for path in root.glob("*/*/*/seed_*/metrics.json"):
            relative = path.relative_to(root)
            dataset, backbone, core, seed_dir, _ = relative.parts
            seed = int(seed_dir.removeprefix("seed_"))
            key = (dataset, backbone, core, seed)
            result = load_json(path)
            if key in runs and runs[key] != result:
                raise ValueError(f"Conflicting proposition result: {key}")
            runs[key] = result

    aggregated = {}
    split_names = {
        "proofwriter": ("depth-3", "depth-5"),
        "ruletaker": ("test",),
    }
    primary_splits = {"proofwriter": "depth-5", "ruletaker": "test"}
    for dataset, dataset_splits in split_names.items():
        aggregated[dataset] = {}
        for backbone in BACKBONES:
            aggregated[dataset][backbone] = {}
            for core in CORES:
                selected = []
                for seed in SEEDS:
                    key = (dataset, backbone, core, seed)
                    if key not in runs:
                        raise ValueError(f"Missing proposition result: {key}")
                    selected.append(runs[key])
                split_summaries = {}
                for split_name in dataset_splits:
                    splits = [run["results"][split_name] for run in selected]
                    test_sizes = {split["total"] for split in splits}
                    if len(test_sizes) != 1:
                        raise ValueError(
                            "Test-size mismatch for "
                            f"{dataset}/{backbone}/{core}/{split_name}: {test_sizes}"
                        )
                    depth_keys = sorted(
                        set.intersection(
                            *(set(split["by_depth"]) for split in splits)
                        ),
                        key=int,
                    )
                    by_depth = {}
                    for depth in depth_keys:
                        totals = {
                            split["by_depth_counts"][depth]["total"]
                            for split in splits
                        }
                        if len(totals) != 1:
                            raise ValueError(
                                "Depth-count mismatch for "
                                f"{dataset}/{backbone}/{core}/{split_name}/D{depth}: "
                                f"{totals}"
                            )
                        by_depth[depth] = {
                            "accuracy": summary(
                                [float(split["by_depth"][depth]) for split in splits]
                            ),
                            "total": totals.pop(),
                        }
                    split_summaries[split_name] = {
                        "test_size": test_sizes.pop(),
                        "accuracy": summary(
                            [float(split["accuracy"]) for split in splits]
                        ),
                        "evidence_at_1": optional_summary(
                            [split.get("evidence_at_1") for split in splits]
                        ),
                        "by_depth": by_depth,
                    }

                primary = split_summaries[primary_splits[dataset]]
                aggregated[dataset][backbone][core] = {
                    "seeds": list(SEEDS),
                    "code_revisions": sorted(
                        {run["code_revision"] for run in selected}
                    ),
                    "split": primary_splits[dataset],
                    "test_size": primary["test_size"],
                    "accuracy": primary["accuracy"],
                    "evidence_at_1": primary["evidence_at_1"],
                    "by_depth": primary["by_depth"],
                    "splits": split_summaries,
                }
    return aggregated


def metric_text(metric: dict | None, digits: int = 3) -> str:
    if metric is None:
        return "--"
    return f"{metric['mean']:.{digits}f} +/- {metric['std']:.{digits}f}"


def latex_metric(metric: dict | None, digits: int = 3) -> str:
    if metric is None:
        return "--"
    mean = f"{metric['mean']:.{digits}f}".lstrip("0")
    std = f"{metric['std']:.{digits}f}".lstrip("0")
    return f"${mean}{{\\pm}}{std}$"


def table_rows(clutrr: dict, proposition: dict) -> list[dict]:
    rows = []
    for backbone in BACKBONES:
        for core in CORES:
            rows.append(
                {
                    "backbone": backbone,
                    "core": core,
                    "method": METHOD_LABELS[core],
                    "clutrr_accuracy": clutrr[backbone][core]["overall"],
                    "clutrr_transition_at_1": clutrr[backbone][core]["transition_at_1"],
                    "proofwriter_accuracy": proposition["proofwriter"][backbone][core]["accuracy"],
                    "proofwriter_evidence_at_1": proposition["proofwriter"][backbone][core]["evidence_at_1"],
                    "ruletaker_accuracy": proposition["ruletaker"][backbone][core]["accuracy"],
                    "ruletaker_evidence_at_1": proposition["ruletaker"][backbone][core]["evidence_at_1"],
                }
            )
    return rows


def render_markdown(rows: list[dict]) -> str:
    lines = [
        "# Backbone/core matrix",
        "",
        "Values are mean +/- sample standard deviation over seeds 0/1/42.",
        "ProofWriter uses the held-out depth-5 configuration; RuleTaker uses the full QDep1--5 test view.",
        "",
        "| Backbone | Method | CLUTRR Acc. | T@1 | ProofWriter Acc. | E@1 | RuleTaker Acc. | E@1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {backbone} | {method} | {ca} | {ct} | {pa} | {pe} | {ra} | {re} |".format(
                backbone=row["backbone"],
                method=row["method"],
                ca=metric_text(row["clutrr_accuracy"]),
                ct=metric_text(row["clutrr_transition_at_1"]),
                pa=metric_text(row["proofwriter_accuracy"]),
                pe=metric_text(row["proofwriter_evidence_at_1"]),
                ra=metric_text(row["ruletaker_accuracy"]),
                re=metric_text(row["ruletaker_evidence_at_1"]),
            )
        )
    lines.extend(
        [
            "",
            "Encoder rows use answer labels only and do not expose a unit-selection score. Content self-attention and TRUA use the same task-available transition/evidence regularization.",
            "",
        ]
    )
    return "\n".join(lines)


def render_tex(rows: list[dict]) -> str:
    lines = []
    for row in rows:
        lines.append(
            "{backbone} & {method} & {ca} & {ct} & {pa} & {pe} & {ra} & {re} \\\\".format(
                backbone=row["backbone"],
                method=row["method"],
                ca=latex_metric(row["clutrr_accuracy"]),
                ct=latex_metric(row["clutrr_transition_at_1"]),
                pa=latex_metric(row["proofwriter_accuracy"]),
                pe=latex_metric(row["proofwriter_evidence_at_1"]),
                ra=latex_metric(row["ruletaker_accuracy"]),
                re=latex_metric(row["ruletaker_evidence_at_1"]),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    clutrr = collect_clutrr(Path(args.clutrr_root))
    proposition = collect_proposition(
        [Path(root) for root in args.proposition_root]
    )
    rows = table_rows(clutrr, proposition)
    payload = {
        "protocol": {
            "seeds": list(SEEDS),
            "backbones": list(BACKBONES),
            "cores": list(CORES),
            "main_table_proofwriter_split": "depth-5",
            "main_table_ruletaker_split": "test",
        },
        "clutrr": clutrr,
        "proposition": proposition,
        "table_rows": rows,
    }
    Path(args.output_json).write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    Path(args.output_markdown).write_text(
        render_markdown(rows), encoding="utf-8"
    )
    Path(args.output_tex).write_text(render_tex(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
