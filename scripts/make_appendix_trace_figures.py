"""Regenerate appendix trace/evidence figures from aggregated TRUA results.

Run on the TRUA development machine with matplotlib available:

    python3 scripts/make_appendix_trace_figures.py --out docs/figures

Values are from:
- docs/aggregated_results/CORE_FINAL_PROP_RERUN.md
- docs/aggregated_results/UNIFIED_CORE_PROP_REGRESSION.md
- docs/aggregated_results/CORE_FINAL_CLUTRR_RERUN.md
- docs/aggregated_results/aggregated_results_20260527.json for no-trace
  baselines and proof-depth baseline curves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D


TEAL = "#007C89"
TEAL_DARK = "#00636D"
TEAL_LIGHT = "#D5EFF0"
RED = "#B23A48"
RED_LIGHT = "#F4D6DA"
BLUE_LIGHT = "#DDEBFA"
GREEN_LIGHT = "#D7F2E5"
YELLOW_LIGHT = "#FFF2B8"
GRAY = "#66717D"
LIGHT_GRAY = "#EEF2F5"
EDGE = "#7B8794"
TEXT = "#1F2933"


def add_box(ax, xy, text, width=1.35, height=0.55, facecolor=LIGHT_GRAY, fontsize=7.0):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.04",
        linewidth=0.8,
        edgecolor=EDGE,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)
    return box


def add_arrow(ax, start, end, color=TEAL, width=1.8, alpha=1.0, rad=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=width,
        color=color,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def make_proof_graph(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.12, 5.855))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(
        5,
        5.45,
        "ProofWriter Proof-Step Reasoning Attention",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=TEXT,
    )
    ax.text(
        5,
        5.02,
        "Representative depth-5 proof item from ProofWriter OWA. Edge width shows TRUA step-selection mass on the gold proof chain.",
        ha="center",
        va="center",
        fontsize=8,
        color=GRAY,
    )

    panel = FancyBboxPatch(
        (1.1, 4.72),
        3.8,
        0.36,
        boxstyle="round,pad=0.035,rounding_size=0.03",
        facecolor="white",
        edgecolor="#CFD8E3",
        linewidth=0.7,
    )
    ax.add_patch(panel)
    ax.text(1.22, 4.9, "BERT depth-5 trace@1", fontsize=7.5, fontweight="bold", va="center")
    ax.add_patch(plt.Rectangle((2.28, 4.82), 0.46, 0.12, color=RED, alpha=0.95))
    ax.text(2.78, 4.9, "baseline 0.146", fontsize=7.0, va="center")
    ax.add_patch(plt.Rectangle((3.67, 4.82), 0.98, 0.12, color=TEAL, alpha=0.95))
    ax.text(4.70, 4.9, "TRUA 0.786", fontsize=7.0, va="center")

    nodes = {
        "triple9\nSquirrel is round": (1.08, 3.78),
        "rule6\nround -> eats cow": (3.05, 4.14),
        "int3\nSquirrel eats cow": (4.75, 3.70),
        "rule4\neats cow -> sees squirrel": (6.05, 4.28),
        "int2\nSquirrel sees squirrel": (7.42, 3.78),
        "rule1\nsees squirrel + eats cow\n-> cow is round": (6.98, 2.92),
        "query\nCow is round": (8.85, 3.25),
        "triple4\nLion sees cow": (1.28, 2.10),
        "rule2\ngreen -> eats tiger": (3.45, 2.08),
        "triple10\nTiger not green": (5.15, 2.03),
    }
    colors = {
        "triple9\nSquirrel is round": YELLOW_LIGHT,
        "rule6\nround -> eats cow": BLUE_LIGHT,
        "int3\nSquirrel eats cow": GREEN_LIGHT,
        "rule4\neats cow -> sees squirrel": BLUE_LIGHT,
        "int2\nSquirrel sees squirrel": GREEN_LIGHT,
        "rule1\nsees squirrel + eats cow\n-> cow is round": BLUE_LIGHT,
        "query\nCow is round": TEAL_LIGHT,
    }

    boxes = {}
    for label, pos in nodes.items():
        boxes[label] = add_box(ax, pos, label, width=1.28 if "rule1" not in label else 1.55, height=0.54, facecolor=colors.get(label, "#F7F9FB"), fontsize=7.0)

    add_arrow(ax, (2.36, 4.03), (3.05, 4.42), TEAL, 2.8)
    add_arrow(ax, (4.33, 4.28), (4.75, 3.98), TEAL, 2.8)
    add_arrow(ax, (5.43, 4.24), (6.10, 4.55), TEAL, 2.2)
    add_arrow(ax, (7.33, 4.38), (7.47, 4.05), TEAL, 2.2)
    add_arrow(ax, (8.03, 3.77), (8.03, 3.47), TEAL, 2.2)
    add_arrow(ax, (5.40, 3.73), (7.00, 3.23), TEAL, 2.2, rad=-0.12)
    add_arrow(ax, (8.51, 3.18), (8.85, 3.43), TEAL, 2.8)

    add_arrow(ax, (2.56, 2.31), (3.45, 2.31), "#AEB7C2", 0.9)
    add_arrow(ax, (4.73, 2.30), (5.15, 2.26), "#AEB7C2", 0.9)
    add_arrow(ax, (6.43, 2.32), (7.05, 3.03), "#AEB7C2", 0.9)

    ax.legend(
        handles=[
            Line2D([0], [0], color=TEAL, lw=2.8, label="gold proof-step path"),
            Line2D([0], [0], color="#AEB7C2", lw=1.0, label="distractor / low mass"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.94, 0.12),
    )

    fig.tight_layout(pad=0.4)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(out_dir / f"fig3_proof_step_attention_graph.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def bar_with_errors(ax, labels, base, base_err, tsra, tsra_err, title, ylabel, base_label="baseline"):
    x = range(len(labels))
    width = 0.34
    ax.bar([i - width / 2 for i in x], base, width, yerr=base_err, color=RED, alpha=0.82, capsize=3, label=base_label)
    ax.bar([i + width / 2 for i in x], tsra, width, yerr=tsra_err, color=TEAL, alpha=0.90, capsize=3, label="TRUA")
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.6, alpha=0.7)
    ax.tick_params(axis="both", labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def make_trace_summary(out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.995, 8.894))
    fig.suptitle(
        "Trace Supervision Improves Reasoning Alignment and Deep Generalization",
        fontsize=13,
        fontweight="bold",
        y=0.965,
    )

    labels3 = ["BERT", "RoBERTa", "DeBERTa"]
    bar_with_errors(
        axes[0, 0],
        labels3,
        [0.146, 0.169, 0.169],
        [0.006, 0.018, 0.023],
        [0.786, 0.789, 0.604],
        [0.003, 0.009, 0.315],
        "ProofWriter depth-5: trace@1",
        "gold trace top-1",
    )
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")

    bar_with_errors(
        axes[0, 1],
        labels3,
        [0.213, 0.173, 0.251],
        [0.045, 0.033, 0.037],
        [0.467, 0.486, 0.518],
        [0.040, 0.088, 0.134],
        "PrOntoQA-OOD: trace@1",
        "gold trace top-1",
    )

    ax = axes[1, 0]
    metric_labels = ["BERT\nanswer", "BERT\ntrace@1", "RoBERTa\nanswer", "RoBERTa\ntrace@1"]
    base_vals = [0.8818, 0.1459, 0.8064, 0.1694]
    base_errs = [0.0086, 0.0061, 0.0783, 0.0178]
    tsra_vals = [0.8842, 0.7863, 0.8511, 0.7890]
    tsra_errs = [0.0037, 0.0029, 0.0064, 0.0086]
    x = range(len(metric_labels))
    width = 0.34
    ax.bar([i - width / 2 for i in x], base_vals, width, yerr=base_errs, color=RED, alpha=0.82, capsize=3, label="baseline")
    ax.bar([i + width / 2 for i in x], tsra_vals, width, yerr=tsra_errs, color=TEAL, alpha=0.90, capsize=3, label="TRUA")
    ax.set_xticks(list(x), metric_labels)
    ax.set_title("ProofWriter depth-5: answer vs trace alignment", fontsize=10, fontweight="bold")
    ax.set_ylabel("score", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.6, alpha=0.7)
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    labels5 = ["BERT", "RoBERTa", "DeBERTa", "DeBERTa-v3", "ModernBERT"]
    bar_with_errors(
        axes[1, 1],
        labels5,
        [0.1946, 0.2223, 0.2415, 0.2426, 0.1873],
        [0.0254, 0.0054, 0.0122, 0.0750, 0.0235],
        [0.3066, 0.3930, 0.4198, 0.5169, 0.3859],
        [0.0615, 0.0334, 0.0335, 0.0622, 0.0147],
        "CLUTRR 2/3-hop train -> 6-10-hop test",
        "long-hop accuracy",
        base_label="vanilla encoder",
    )
    axes[1, 1].tick_params(axis="x", rotation=18)

    fig.text(
        0.5,
        0.035,
        "Plotted metrics are from current 3-seed aggregated experiment artifacts. CLUTRR uses the true vanilla encoder classifier audit, not the TRUA label-only ablation.",
        ha="center",
        fontsize=8,
        color=GRAY,
    )
    fig.tight_layout(rect=(0.04, 0.07, 0.98, 0.93), h_pad=2.2, w_pad=1.8)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(out_dir / f"fig4_trace_faithfulness_depth_curves.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    make_proof_graph(args.out)
    make_trace_summary(args.out)


if __name__ == "__main__":
    main()
