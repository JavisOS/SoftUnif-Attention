#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(x):
    return f"{x:.4f}" if isinstance(x, float) else str(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root)
    rows = []
    for path in sorted((run_root / "results").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        diag = data["diagnostic"]
        rows.append(
            {
                "variant": data["variant"],
                "seed": data["seed"],
                "epochs": data["epochs"],
                "iid_acc": diag["iid_acc"],
                "ood_acc": diag["ood_acc"],
                "extract_n": diag["extract_n"],
                "supported_patterns": diag["supported_patterns"],
                "shortcut_count": diag["shortcut_count"],
                "shortcut_rate": diag["shortcut_rate"],
                "avg_shortcut_delta": diag["avg_shortcut_delta"],
                "patterns": diag["patterns"][:12],
                "path": str(path),
            }
        )

    md = [
        "# CLUTRR Shortcut-Reasoning Diagnostic Pilot",
        "",
        f"Run root: `{run_root}`",
        "",
        "Method: pilot token-occlusion approximation of Haraguchi et al. (2023) shortcut-reasoning diagnostics.",
        "This is not an official IG/input-reduction reproduction. IID is CLUTRR short-hop test examples (2/3-hop); OOD is long-hop test examples (>=6-hop).",
        "",
        "| Variant | Seed | Epochs | IID short acc | OOD long acc | Supported patterns | Shortcut count | Shortcut rate | Avg shortcut delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['variant']} | {row['seed']} | {row['epochs']} | {fmt(row['iid_acc'])} | {fmt(row['ood_acc'])} | {row['supported_patterns']} | {row['shortcut_count']} | {fmt(row['shortcut_rate'])} | {fmt(row['avg_shortcut_delta'])} |"
        )
    for row in rows:
        md += ["", f"## Top Patterns: {row['variant']}", "", "| Pattern | Label | IID support | OOD support | IID acc | OOD pred rate | OOD acc | Delta | Shortcut |", "|---|---|---:|---:|---:|---:|---:|---:|---|"]
        for pat in row["patterns"]:
            md.append(
                f"| `{pat['pattern']}` | {pat['label']} | {pat['iid_support']} | {pat['ood_support']} | {fmt(pat['iid_acc'])} | {fmt(pat['ood_pred_rate'])} | {fmt(pat['ood_acc'])} | {fmt(pat['delta_vs_ood'])} | {pat['shortcut_flag']} |"
            )

    aggregate = {"run_root": str(run_root), "rows": rows}
    out_json = Path("/root/TRUA/docs/aggregated_results/CLUTRR_SHORTCUT_DIAGNOSTIC_PILOT.json")
    out_md = Path("/root/TRUA/docs/aggregated_results/CLUTRR_SHORTCUT_DIAGNOSTIC_PILOT.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
