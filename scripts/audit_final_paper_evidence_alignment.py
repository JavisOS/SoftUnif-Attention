#!/usr/bin/env python3
"""Audit the final validation-selected TRUA paper experiment artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path


SEEDS = (0, 1, 42)
REVISION = "64f8ae0a7b0757bd6c05d1ac1c1402c10c622eba"
NAME_RE = re.compile(r"(.+)_seed(0|1|42)$")
CLUTRR_TOTALS = {
    "2": 38,
    "3": 105,
    "4": 190,
    "5": 174,
    "6": 107,
    "7": 144,
    "8": 150,
    "9": 119,
    "10": 119,
}

CLUTRR_VARIANTS = {
    "trua": (1.0, 1.0, 0.0, True, True, True, True),
    "with_consistency": (1.0, 1.0, 5.0, True, True, True, True),
    "answer_only": (0.0, 0.0, 0.0, True, True, True, True),
    "transition_only": (1.0, 0.0, 0.0, True, True, True, True),
    "edge_only": (0.0, 1.0, 0.0, True, True, True, True),
    "no_goal": (1.0, 1.0, 0.0, True, False, True, True),
    "no_aggregation": (1.0, 1.0, 0.0, True, True, False, True),
    "no_step": (0.0, 1.0, 0.0, True, True, True, False),
    "no_relation": (1.0, 1.0, 0.0, False, True, True, True),
}

PROPOSITION_GROUPS = {
    "proofwriter_bert_trua": {
        "dataset": "proofwriter",
        "model_suffix": "/bert-base-uncased",
        "lambda_evidence": 1.0,
        "goal": True,
        "train": 30000,
        "validation": 5000,
        "validation_evidence_total": 3465,
        "train_depths": "0,1,2",
        "test_depths": "3,5",
        "train_qdeps": "",
        "test_qdeps": "",
        "max_sentences": 32,
        "splits": {"depth-3": (5000, 3627), "depth-5": (5000, 3877)},
    },
    "proofwriter_bert_no_evidence": {
        "dataset": "proofwriter",
        "model_suffix": "/bert-base-uncased",
        "lambda_evidence": 0.0,
        "goal": True,
        "train": 30000,
        "validation": 5000,
        "validation_evidence_total": 3465,
        "train_depths": "0,1,2",
        "test_depths": "3,5",
        "train_qdeps": "",
        "test_qdeps": "",
        "max_sentences": 32,
        "splits": {"depth-3": (5000, 3627), "depth-5": (5000, 3877)},
    },
    "proofwriter_bert_no_goal": {
        "dataset": "proofwriter",
        "model_suffix": "/bert-base-uncased",
        "lambda_evidence": 1.0,
        "goal": False,
        "train": 30000,
        "validation": 5000,
        "validation_evidence_total": 3465,
        "train_depths": "0,1,2",
        "test_depths": "3,5",
        "train_qdeps": "",
        "test_qdeps": "",
        "max_sentences": 32,
        "splits": {"depth-3": (5000, 3627), "depth-5": (5000, 3877)},
    },
    "ruletaker_raw_deberta_trua": {
        "dataset": "ruletaker_raw",
        "model_suffix": "/deberta-base",
        "lambda_evidence": 1.0,
        "goal": True,
        "train": 30000,
        "validation": 5000,
        "validation_evidence_total": 5000,
        "train_depths": "1,2",
        "test_depths": "1,2,3,5",
        "train_qdeps": "1,2",
        "test_qdeps": "1,2,3,4,5",
        "max_sentences": 32,
        "splits": {"test": (5000, 5000)},
    },
    "ruletaker_raw_deberta_no_evidence": {
        "dataset": "ruletaker_raw",
        "model_suffix": "/deberta-base",
        "lambda_evidence": 0.0,
        "goal": True,
        "train": 30000,
        "validation": 5000,
        "validation_evidence_total": 5000,
        "train_depths": "1,2",
        "test_depths": "1,2,3,5",
        "train_qdeps": "1,2",
        "test_qdeps": "1,2,3,4,5",
        "max_sentences": 32,
        "splits": {"test": (5000, 5000)},
    },
    "prontoqa_bert_trua": {
        "dataset": "prontoqa",
        "model_suffix": "/bert-base-uncased",
        "lambda_evidence": 1.0,
        "goal": True,
        "train": 180,
        "validation": 20,
        "validation_evidence_total": 20,
        "train_depths": "0,1,2",
        "test_depths": "3,5",
        "train_qdeps": "",
        "test_qdeps": "",
        "max_sentences": 24,
        "splits": {"ood": (300, 300)},
    },
    "prontoqa_bert_no_evidence": {
        "dataset": "prontoqa",
        "model_suffix": "/bert-base-uncased",
        "lambda_evidence": 0.0,
        "goal": True,
        "train": 180,
        "validation": 20,
        "validation_evidence_total": 20,
        "train_depths": "0,1,2",
        "test_depths": "3,5",
        "train_qdeps": "",
        "test_qdeps": "",
        "max_sentences": 24,
        "splits": {"ood": (300, 300)},
    },
}


def expected_names() -> set[str]:
    names = {
        f"clutrr_deberta_{variant}_seed{seed}"
        for variant in (*CLUTRR_VARIANTS, "vanilla")
        for seed in SEEDS
    }
    names.update(
        f"{group}_seed{seed}" for group in PROPOSITION_GROUPS for seed in SEEDS
    )
    return names


def load_json(path: Path, errors: list[str]):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read {path}: {exc}")
        return None


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def close(actual, expected, tolerance=1e-12) -> bool:
    try:
        return math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def metric(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and 0.0 <= value <= 1.0


def audit_validation(payload: dict, name: str, kind: str, errors: list[str]) -> None:
    history = payload.get("validation_history", [])
    check(len(history) == 10, f"{name}: validation history has {len(history)} epochs", errors)
    check(
        [row.get("epoch") for row in history] == list(range(1, 11)),
        f"{name}: validation epochs are not exactly 1--10",
        errors,
    )
    epoch = payload.get("selected_epoch")
    check(isinstance(epoch, int) and 1 <= epoch <= 10, f"{name}: invalid selected epoch {epoch}", errors)
    if len(history) != 10 or not isinstance(epoch, int) or not 1 <= epoch <= 10:
        return
    if kind == "clutrr":
        selected = history[epoch - 1].get("overall")
        check(
            close(payload.get("selected_validation_overall", -1), selected),
            f"{name}: stored selected validation score does not match selected epoch",
            errors,
        )
        best = max(range(10), key=lambda i: history[i]["overall"])
    else:
        selected = history[epoch - 1]
        check(
            close(payload.get("selected_validation_accuracy", -1), selected.get("accuracy", -2)),
            f"{name}: stored selected validation accuracy does not match selected epoch",
            errors,
        )
        check(
            close(
                payload.get("selected_validation_evidence_at_1", -1),
                selected.get("evidence_at_1", -2),
            ),
            f"{name}: stored selected validation Evidence@1 does not match selected epoch",
            errors,
        )
        best = max(
            range(10),
            key=lambda i: (history[i]["accuracy"], history[i]["evidence_at_1"]),
        )
    check(epoch == best + 1, f"{name}: selected epoch is not validation-optimal", errors)


def audit_clutrr(payload: dict, name: str, group: str, errors: list[str]) -> None:
    check(payload.get("dataset") == "data_089907f8", f"{name}: wrong CLUTRR dataset", errors)
    check(payload.get("model_type") == "deberta", f"{name}: wrong CLUTRR model", errors)
    for field, expected in (("train_size", 9083), ("validation_size", 1011), ("test_size", 1146)):
        check(payload.get(field) == expected, f"{name}: {field} != {expected}", errors)
    check(close(payload.get("validation_fraction", -1), 0.1), f"{name}: wrong validation fraction", errors)
    check(payload.get("validation_seed") == 2027, f"{name}: wrong validation seed", errors)
    audit_validation(payload, name, "clutrr", errors)

    test = payload.get("test", {})
    check(metric(test.get("overall")), f"{name}: invalid overall accuracy", errors)
    check(metric(test.get("short_hop")), f"{name}: invalid short accuracy", errors)
    check(metric(test.get("long_hop")), f"{name}: invalid long accuracy", errors)
    per_hop = test.get("per_hop", {})
    check(set(per_hop) == set(CLUTRR_TOTALS), f"{name}: wrong official length keys", errors)
    for hop, total in CLUTRR_TOTALS.items():
        row = per_hop.get(hop, {})
        check(row.get("total") == total, f"{name}: length {hop} total != {total}", errors)
        check(metric(row.get("accuracy")), f"{name}: invalid length-{hop} accuracy", errors)

    variant = group.removeprefix("clutrr_deberta_")
    if variant == "vanilla":
        check("configuration" not in payload, f"{name}: vanilla unexpectedly has unit configuration", errors)
        check("transition_at_1" not in test, f"{name}: vanilla unexpectedly has Transition@1", errors)
        return
    config = payload.get("configuration", {})
    expected = CLUTRR_VARIANTS[variant]
    actual = (
        float(config.get("lambda_nexthop", -1)),
        float(config.get("lambda_edge", -1)),
        float(config.get("lambda_consistency", -1)),
        config.get("use_relation_conditioning"),
        config.get("use_goal_guidance"),
        config.get("use_aggregation_branch"),
        config.get("use_step_branch"),
    )
    check(actual == expected, f"{name}: configuration {actual} != {expected}", errors)
    check(metric(test.get("transition_at_1")), f"{name}: invalid Transition@1", errors)
    check(test.get("transition_total") == 5645, f"{name}: transition denominator != 5645", errors)


def audit_proposition(payload: dict, name: str, group: str, errors: list[str]) -> None:
    spec = PROPOSITION_GROUPS[group]
    check(payload.get("dataset") == spec["dataset"], f"{name}: wrong dataset", errors)
    check(str(payload.get("model_name", "")).endswith(spec["model_suffix"]), f"{name}: wrong model checkpoint", errors)
    check(close(payload.get("lambda_evidence", -1), spec["lambda_evidence"]), f"{name}: wrong evidence weight", errors)
    check(payload.get("train") == spec["train"], f"{name}: wrong train count", errors)
    check(payload.get("validation") == spec["validation"], f"{name}: wrong validation count", errors)
    for field in ("train_depths", "test_depths", "train_qdeps", "test_qdeps"):
        check(payload.get(field) == spec[field], f"{name}: wrong {field}", errors)
    check(payload.get("epochs") == 10, f"{name}: epochs != 10", errors)
    check(payload.get("max_sentences") == spec["max_sentences"], f"{name}: wrong sentence limit", errors)
    check(payload.get("max_context_tokens") == 512, f"{name}: wrong context limit", errors)
    check(payload.get("input_truncation_allowed") is False, f"{name}: input truncation was allowed", errors)
    check(
        payload.get("selection_rule") == "validation accuracy; Evidence@1 breaks exact ties",
        f"{name}: wrong selection rule",
        errors,
    )
    architecture = payload.get("architecture", {})
    check(architecture.get("relation_channels") == 8, f"{name}: wrong relation channel count", errors)
    check(architecture.get("use_relation_conditioning") is True, f"{name}: relation conditioning off", errors)
    check(architecture.get("use_aggregation_branch") is True, f"{name}: aggregation off", errors)
    check(architecture.get("use_step_branch") is True, f"{name}: selection branch off", errors)
    check(architecture.get("use_goal_guidance") is spec["goal"], f"{name}: wrong goal-guidance flag", errors)
    audit_validation(payload, name, "proposition", errors)
    for row in payload.get("validation_history", []):
        check(
            row.get("evidence_total") == spec["validation_evidence_total"],
            f"{name}: validation evidence denominator changed at epoch {row.get('epoch')}",
            errors,
        )

    results = payload.get("results", {})
    check(set(results) == set(spec["splits"]), f"{name}: wrong test splits", errors)
    for split, (total, evidence_total) in spec["splits"].items():
        row = results.get(split, {})
        check(row.get("total") == total, f"{name}/{split}: total != {total}", errors)
        check(row.get("evidence_total") == evidence_total, f"{name}/{split}: evidence denominator != {evidence_total}", errors)
        check(metric(row.get("accuracy")), f"{name}/{split}: invalid accuracy", errors)
        check(metric(row.get("evidence_at_1")), f"{name}/{split}: invalid Evidence@1", errors)
        for depth, value in row.get("by_depth", {}).items():
            check(metric(value), f"{name}/{split}: invalid by-depth metric at {depth}", errors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--expected-revision", default=REVISION)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    result_dir = run_root / "results"
    errors: list[str] = []
    expected = expected_names()
    files = {path.stem: path for path in result_dir.glob("*.json")}
    check(set(files) == expected, f"result task set mismatch: missing={sorted(expected-set(files))}, extra={sorted(set(files)-expected)}", errors)

    manifest = load_json(run_root / "manifest.json", errors) or {}
    manifest_tasks = manifest.get("tasks", [])
    manifest_names = {task.get("name") for task in manifest_tasks}
    check(manifest_names == expected, "manifest task names do not match the final protocol", errors)
    manifest_text = json.dumps(manifest, sort_keys=True).lower()
    check("tsra" not in manifest_text, "manifest still contains TSRA naming", errors)
    check("lambda_trace" not in manifest_text, "manifest contains the deprecated lambda_trace flag", errors)

    status = load_json(run_root / "status.json", errors) or {}
    check(status.get("total") == 51, "status total is not 51", errors)
    check(set(status.get("completed", [])) == expected, "status does not mark all tasks complete", errors)
    check(not status.get("failed"), "status contains failed tasks", errors)
    check(not status.get("pending"), "status contains pending tasks", errors)
    check(not status.get("running"), "status contains running tasks", errors)

    revisions = set()
    for name in sorted(expected & set(files)):
        payload = load_json(files[name], errors)
        if payload is None:
            continue
        match = NAME_RE.fullmatch(name)
        if match is None:
            errors.append(f"{name}: invalid task name")
            continue
        group, seed_text = match.groups()
        seed = int(seed_text)
        check(payload.get("seed") == seed, f"{name}: payload seed does not match filename", errors)
        revisions.add(payload.get("code_revision"))
        check(payload.get("code_revision") == args.expected_revision, f"{name}: wrong code revision", errors)
        if group.startswith("clutrr_deberta_"):
            audit_clutrr(payload, name, group, errors)
        elif group in PROPOSITION_GROUPS:
            audit_proposition(payload, name, group, errors)
        else:
            errors.append(f"{name}: unknown group {group}")

    summary = {
        "audited_at": datetime.now().isoformat(timespec="seconds"),
        "run_root": str(run_root),
        "expected_revision": args.expected_revision,
        "observed_revisions": sorted(str(value) for value in revisions),
        "expected_tasks": 51,
        "observed_tasks": len(files),
        "expected_groups": 17,
        "observed_groups": len({NAME_RE.fullmatch(name).group(1) for name in files if NAME_RE.fullmatch(name)}),
        "seeds": list(SEEDS),
        "errors": errors,
        "passed": not errors,
    }
    (run_root / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Final TRUA Artifact Audit",
        "",
        f"- Status: {'PASS' if not errors else 'FAIL'}",
        f"- Tasks: {len(files)}/51",
        f"- Groups: {summary['observed_groups']}/17",
        f"- Seeds: {', '.join(map(str, SEEDS))}",
        f"- Expected revision: `{args.expected_revision}`",
        "- Checks: task manifest, revision, validation selection, configuration, split sizes, official CLUTRR length totals, and evidence denominators",
    ]
    if errors:
        lines.extend(["", "## Errors", "", *[f"- {error}" for error in errors]])
    (run_root / "audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
