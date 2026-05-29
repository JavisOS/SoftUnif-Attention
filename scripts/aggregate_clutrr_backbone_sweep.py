#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


REPO = Path("/root/TSRA")
RUN = Path("/vepfs/tsra_outputs/clutrr_backbone_sweep/latest").resolve()
OUT_DIR = REPO / "docs" / "aggregated_results"
OUT_MD = OUT_DIR / "CLUTRR_BACKBONE_SWEEP_20260529.md"
OUT_JSON = OUT_DIR / "clutrr_backbone_sweep_20260529.json"
REPORT = REPO / "EXPERIMENT_REPORT.md"
OLD_JSON = OUT_DIR / "aggregated_results_20260527.json"

EPOCH_RE = re.compile(r"Epoch (\d+) Done")
METRIC_RE = re.compile(r"\s*(Overall Acc \(Base\)|Short Hop \(2-3\)|Long Hop \(>=6\)):\s+([0-9.]+)")
NAME_RE = re.compile(r"clutrr_(089|db9)_(.+)_(label_only|full)_seed(\d+)$")

DISPLAY = {
    "bert": "BERT",
    "roberta": "RoBERTa",
    "deberta": "DeBERTa",
    "deberta-v3": "DeBERTa-v3",
    "modernbert": "ModernBERT",
}


def parse_log(path: Path) -> tuple[dict, dict]:
    epochs: list[dict] = []
    current: dict | None = None
    for line in path.read_text(errors="ignore").splitlines():
        m = EPOCH_RE.search(line)
        if m:
            current = {"epoch": int(m.group(1))}
            epochs.append(current)
            continue
        m = METRIC_RE.match(line)
        if m and current is not None:
            key = {
                "Overall Acc (Base)": "overall",
                "Short Hop (2-3)": "short",
                "Long Hop (>=6)": "long",
            }[m.group(1)]
            current[key] = float(m.group(2))
    complete = [e for e in epochs if all(k in e for k in ("overall", "short", "long"))]
    if not complete:
        raise RuntimeError(f"No complete metric block found in {path}")
    best = max(complete, key=lambda e: e["overall"])
    final = max(complete, key=lambda e: e["epoch"])
    return best, final


def fmt(values) -> str:
    vals = list(values)
    mean = sum(vals) / len(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{mean:.4f} +/- {sd:.4f}"


def collect_sweep_rows() -> list[dict]:
    rows = []
    for log in sorted((RUN / "logs").glob("clutrr_*.log")):
        m = NAME_RE.match(log.stem)
        if not m:
            continue
        split_key, backbone, variant, seed = m.groups()
        split = {"089": "data_089907f8", "db9": "data_db9b8f04"}[split_key]
        backbone = backbone.replace("_", "-")
        best, final = parse_log(log)
        rows.append(
            {
                "split": split,
                "backbone": backbone,
                "variant": "tsra" if variant == "full" else "label_only",
                "seed": int(seed),
                "best": best,
                "final": final,
                "log": str(log),
            }
        )
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["backbone"], row["variant"])].append(row)

    out = []
    for (split, backbone, variant), items in sorted(groups.items()):
        items = sorted(items, key=lambda r: r["seed"])
        out.append(
            {
                "split": split,
                "backbone": backbone,
                "variant": variant,
                "seeds": [r["seed"] for r in items],
                "n": len(items),
                "best_overall": fmt(r["best"]["overall"] for r in items),
                "best_short": fmt(r["best"]["short"] for r in items),
                "best_long": fmt(r["best"]["long"] for r in items),
                "final_overall": fmt(r["final"]["overall"] for r in items),
                "final_short": fmt(r["final"]["short"] for r in items),
                "final_long": fmt(r["final"]["long"] for r in items),
            }
        )
    return out


def load_existing_clutrr_rows() -> list[dict]:
    if not OLD_JSON.exists():
        return []
    data = json.loads(OLD_JSON.read_text())
    rows: list[dict] = []
    for key in ("clutrr_089907f8", "clutrr_db9b8f04"):
        for row in data.get(key, []):
            variant = row["variant"]
            if variant == "full":
                variant = "tsra"
            rows.append(
                {
                    "split": row["dataset"],
                    "backbone": row["backbone"],
                    "variant": variant,
                    "seeds": [0, 1, 42],
                    "n": row.get("seeds", 3),
                    "best_overall": row["best_overall"],
                    "best_short": row["best_short"],
                    "best_long": row["best_long"],
                    "final_overall": row["final_overall"],
                    "final_long": row["final_long"],
                    "source": "aggregated_results_20260527",
                }
            )
    return rows


def merge_existing_and_sweep(sweep_rows: list[dict]) -> list[dict]:
    merged: dict[tuple[str, str, str], dict] = {}
    for row in load_existing_clutrr_rows():
        merged[(row["split"], row["backbone"], row["variant"])] = row
    for row in sweep_rows:
        row = dict(row)
        row["source"] = "clutrr_backbone_sweep_20260529"
        merged[(row["split"], row["backbone"], row["variant"])] = row
    return [merged[k] for k in sorted(merged)]


def table(rows: list[dict], split: str, *, include_existing: bool = True) -> str:
    selected = [r for r in rows if r["split"] == split]
    by_key = {(r["backbone"], r["variant"]): r for r in selected}
    lines = [
        "| Backbone | Variant | Seeds | Best Overall | Best Short | Best Long >=6 | Final Overall | Final Long >=6 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    order = ["bert", "roberta", "deberta", "deberta-v3", "modernbert"]
    variants = ["label_only", "tsra", "no_consistency"]
    for backbone in order:
        for variant in variants:
            r = by_key.get((backbone, variant))
            if r is None:
                continue
            lines.append(
                f"| {DISPLAY.get(backbone, backbone)} | {variant.replace('_', '-')} | {r['n']} "
                f"| {r['best_overall']} | {r['best_short']} | {r['best_long']} "
                f"| {r['final_overall']} | {r['final_long']} |"
            )
    return "\n".join(lines)


def metric_value(s: str) -> float:
    return float(s.split("+/-")[0].strip())


def build_markdown(rows: list[dict]) -> str:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = {
        "started": len(list((RUN / "status").glob("*.started"))),
        "done": len(list((RUN / "status").glob("*.done"))),
        "failed": len(list((RUN / "status").glob("*.failed"))),
    }
    lines = [
        "# CLUTRR Backbone Sweep",
        "",
        f"Generated on {generated} from `{RUN}`.",
        "",
        f"Completion: `{status['done']}/{status['started']} done`, `{status['failed']} failed`.",
        "",
        "Values are `mean +/- sample-std` over seeds `0/1/42`. `Best` uses the best logged evaluation point by overall accuracy across 10 epochs; `Final` is epoch 10.",
        "",
        "## data_089907f8",
        "",
        table(rows, "data_089907f8"),
        "",
        "## data_db9b8f04",
        "",
        table(rows, "data_db9b8f04"),
        "",
        "## Notes",
        "",
        "- This file merges the newly completed CLUTRR backbone sweep with the previously audited DeBERTa/RoBERTa CLUTRR rows from `AGGREGATED_RESULTS_20260527.md`.",
        "- The merged tables replace the older CLUTRR same-backbone table from the draft, whose configuration was not reliable.",
        "- `data_089907f8` is the primary TSRA CLUTRR split with 2/3-hop training.",
        "- `data_db9b8f04` is the follow-up 2/3/4-hop training split.",
    ]
    return "\n".join(lines) + "\n"


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    a = text.index(start)
    b = text.index(end, a)
    return text[:a] + replacement + text[b:]


def update_report(rows: list[dict]) -> None:
    text = REPORT.read_text()
    text = text.replace("#### ProofWriter\n#### ProofWriter\n", "#### ProofWriter\n")
    text = text.replace("### ProofWriter\n### ProofWriter\n", "### ProofWriter\n")
    updated = datetime.now().strftime("%Y-%m-%d %H:%M Asia/Shanghai")
    snapshot = f"""## 0. Latest Status Snapshot

Updated on **{updated}** after the CLUTRR same-backbone rerun finished. This section is the current authoritative summary. Values are `mean +/- sample-std` over seeds `0/1/42` unless stated otherwise.

### Completion Status

- **CLUTRR backbone sweep is complete:** `/vepfs/tsra_outputs/clutrr_backbone_sweep/latest` is `42/42 done, 0 failed`.
- **TSRA/backbone runs are complete:** additional depth/seed checks are `14/14 done, 0 failed`; seed-42 completion is `22/22 done, 0 failed`.
- **Final aggregated artifacts:** `docs/aggregated_results/AGGREGATED_RESULTS_20260527.md`, `docs/aggregated_results/aggregated_results_20260527.json`, `docs/aggregated_results/CLUTRR_BACKBONE_SWEEP_20260529.md`, and `docs/aggregated_results/clutrr_backbone_sweep_20260529.json`.
- **Aggregation scripts:** `scripts/aggregate_experiment_results.py` and `scripts/aggregate_clutrr_backbone_sweep.py`.
- **NLProofS formal RuleTaker test is complete:** final result file is at `/vepfs/tsra_outputs/official_external/latest_nlproofs_ruletaker_test/prover_test/lightning_logs/version_0/results_test.json`. Reported test metrics: answer overall `0.6796`, proof overall `0.9187`.

### Completed TSRA Main Results

#### CLUTRR `data_089907f8`

This is the primary CLUTRR split used throughout TSRA, with 2/3-hop training. This table replaces the older draft same-backbone CLUTRR table whose configuration was not reliable.

{table(rows, "data_089907f8")}

Interpretation: the merged rerun gives a mixed but useful backbone story. BERT, DeBERTa, RoBERTa, and ModernBERT show TSRA gains over label-only on long-hop examples. DeBERTa-v3 improves overall accuracy but is roughly tied/slightly lower on long-hop in the primary split. RoBERTa's no-consistency ablation remains stronger than full TSRA, so the consistency term should be reported cautiously.

#### CLUTRR `data_db9b8f04` 2/3/4-Hop Train Check

This follow-up trains on 2/3/4-hop examples and tests long-hop generalization.

{table(rows, "data_db9b8f04")}

Interpretation: the 2/3/4-hop split gives the strongest CLUTRR same-backbone evidence. TSRA improves long-hop accuracy for BERT, RoBERTa, DeBERTa, DeBERTa-v3, and ModernBERT.

"""
    text = replace_between(text, "## 0. Latest Status Snapshot\n", "#### ProofWriter\n", snapshot + "#### ProofWriter\n")
    text = replace_between(
        text,
        "### CLUTRR Main Comparison\n",
        "### ProofWriter\n",
        f"""### CLUTRR Main Comparison

Same-backbone rerun, primary `data_089907f8` split:

{table(rows, "data_089907f8")}

Same-backbone rerun, `data_db9b8f04` 2/3/4-hop train split:

{table(rows, "data_db9b8f04")}

External/reference baselines:

| Model | Input setting | Overall | Short-hop | Long-hop >=6 | Paper-use status |
|---|---|---:|---:|---:|---|
| Edge Transformer | structured graph edges | 0.8100 | 0.9762 | 0.6847 | structured/reference baseline |
| RAT | structured relation-aware baseline | 0.5755 | 0.9762 | 0.3483 | structured/reference baseline |
| Dual Attention adapted | raw text, DeBERTa unfrozen | 0.2548 | 0.9580 | 0.1424 | external adapted diagnostic |
| Abstractor/RCA adapted | raw text, DeBERTa unfrozen | 0.1571 | 0.4336 | 0.1095 | external adapted diagnostic |
| MAC-style attention adapted | raw text, local MAC-style model | 0.2173 | 0.6573 | 0.1283 | attention baseline diagnostic |

Key CLUTRR takeaway:

- The older draft same-backbone table should be retired; the current table is the audited 10-epoch, 3-seed rerun.
- Edge Transformer remains strongest, but it uses structured graph-edge input rather than the same raw-text setting.
- On the 2/3/4-hop split, TSRA gives the clearest same-backbone long-hop gains across BERT, DeBERTa, DeBERTa-v3, and ModernBERT.

### ProofWriter
""",
    )
    old = "CLUTRR 是目前最干净的主结果。我们一直使用的 `data_089907f8` split 上，DeBERTa label-only 的 overall/long-hop 分别是 `0.4706 +/- 0.0298` 和 `0.3414 +/- 0.0293`，DeBERTa full TSRA 提升到 `0.6262 +/- 0.0214` 和 `0.4198 +/- 0.0335`。新增的 `data_db9b8f04` 2/3/4-hop train 检查也支持同样结论：label-only long-hop 是 `0.4925 +/- 0.0314`，TSRA 是 `0.6508 +/- 0.0181`。这说明 trace-supervised step selection 对 long-hop generalization 有稳定贡献。"
    new = "CLUTRR 的旧论文初稿 same-backbone 表已经废弃；当前可用的是 2026-05-29 完成的 10 epoch、3 seed rerun，并合并了此前已经审计过的 DeBERTa/RoBERTa 行。`data_089907f8` 主 split 上，BERT、DeBERTa、RoBERTa、ModernBERT 都显示 TSRA 相比 label-only 的 long-hop 提升；DeBERTa-v3 overall 有提升但 long-hop 基本持平/略低，RoBERTa no-consistency ablation 仍强于 full TSRA，应作为例外如实报告。`data_db9b8f04` 2/3/4-hop train split 上证据更强：BERT、RoBERTa、DeBERTa、DeBERTa-v3、ModernBERT 的 TSRA long-hop 均高于 label-only，其中 DeBERTa 从 `0.4925 +/- 0.0314` 到 `0.6508 +/- 0.0181`，DeBERTa-v3 从 `0.7010 +/- 0.0262` 到 `0.7462 +/- 0.0140`，ModernBERT 从 `0.5737 +/- 0.0406` 到 `0.6273 +/- 0.0442`。这说明 trace-supervised step selection 对 long-hop generalization 有稳定贡献，但 backbone 和 regularization 会影响幅度。"
    if old in text:
        text = text.replace(old, new)
    text = re.sub(
        r"- \*\*Follow-up split:\*\* `data/data_db9b8f04`, with `1\.2,1\.3,1\.4_train\.csv` for 2/3/4-hop training\.[^\n]*",
        "- **Follow-up split:** `data/data_db9b8f04`, with `1.2,1.3,1.4_train.csv` for 2/3/4-hop training. The merged CLUTRR backbone sweep is complete under `/vepfs/tsra_outputs/clutrr_backbone_sweep/latest`.",
        text,
    )
    text = re.sub(
        r"- \*\*Current status:\*\* ready; TSRA and external CLUTRR baselines have been run on the correct `data_089907f8` split\.",
        "- **Current status:** ready; same-backbone CLUTRR reruns are complete for the audited `data_089907f8` and `data_db9b8f04` splits, and external CLUTRR baselines are available.",
        text,
    )
    while "#### ProofWriter\n#### ProofWriter\n" in text:
        text = text.replace("#### ProofWriter\n#### ProofWriter\n", "#### ProofWriter\n")
    while "### ProofWriter\n### ProofWriter\n" in text:
        text = text.replace("### ProofWriter\n### ProofWriter\n", "### ProofWriter\n")
    REPORT.write_text(text)


def main() -> None:
    rows = collect_sweep_rows()
    sweep_agg = aggregate(rows)
    agg = merge_existing_and_sweep(sweep_agg)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"run_root": str(RUN), "rows": agg}, indent=2) + "\n")
    OUT_MD.write_text(build_markdown(agg))
    update_report(agg)
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")
    print(f"Updated {REPORT}")
    for split in ("data_089907f8", "data_db9b8f04"):
        print(f"\n{split}")
        print(table(agg, split))


if __name__ == "__main__":
    main()
