"""Create a clean, orthogonal-arrow TRUA overview figure.

Design principles:
- the upper lane is inference computation only;
- the lower lane contains training-only supervision;
- arrows are horizontal, vertical, or right-angle polylines;
- the diagram is conceptual and avoids formula clutter.

Run on the TRUA development machine:

    python3 scripts/make_figure1_overview.py --out docs/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D


TEXT = "#17212B"
MUTED = "#5F6C7B"
LINE = "#2F3A4A"
BLUE = "#2D6C9F"
BLUE_FILL = "#EAF3FB"
TEAL = "#007C89"
TEAL_FILL = "#E6F5F4"
GREEN = "#3D8B5B"
GREEN_FILL = "#EDF8F1"
ORANGE = "#C8741F"
ORANGE_FILL = "#FFF3E3"
PURPLE = "#7B4D91"
PURPLE_FILL = "#F4ECF7"
GRAY = "#CBD5E1"
GRAY_FILL = "#F8FAFC"


def box(ax, x, y, w, h, text, face, edge, fontsize=7.0, weight="normal", color=TEXT, dashed=False):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.055",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.0,
        linestyle=(0, (3, 2)) if dashed else "solid",
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=color,
        linespacing=1.15,
    )
    return patch


def panel(ax, x, y, w, h, title, face, edge):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.045,rounding_size=0.08",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.05,
    )
    ax.add_patch(patch)
    ax.text(x + 0.12, y + h - 0.20, title, ha="left", va="top", fontsize=7.7, fontweight="bold", color=TEXT)
    return patch


def orth_arrow(ax, pts, color=LINE, lw=1.15, dashed=False, ms=8.5):
    """Draw a horizontal/vertical polyline with an arrow head on the last segment."""
    style = (0, (3, 2)) if dashed else "solid"
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=lw, linestyle=style)
    start, end = pts[-2], pts[-1]
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        linestyle=style,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arr)
    return arr


def token_stack(ax, x, y, color=BLUE, count=4):
    for i in range(count):
        xx = x + i * 0.18
        patch = FancyBboxPatch(
            (xx, y),
            0.10,
            0.44,
            boxstyle="round,pad=0.008,rounding_size=0.025",
            facecolor="#DCEBFA",
            edgecolor=color,
            linewidth=0.65,
        )
        ax.add_patch(patch)
        ax.plot([xx, xx + 0.10], [y + 0.22, y + 0.22], color=color, linewidth=0.45)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.25, 3.65))
    ax.set_xlim(0, 13.8)
    ax.set_ylim(0, 6.55)
    ax.axis("off")

    # Title and legend.
    ax.text(0.10, 6.22, "Transition-Regularized Unit Attention (TRUA)", fontsize=10.0, fontweight="bold", color=TEXT)
    ax.text(0.10, 5.97, "A shared trace-supervised core operates on text-grounded reasoning units.", fontsize=6.4, color=MUTED)
    ax.legend(
        handles=[
            Line2D([0], [0], color=LINE, lw=1.4, label="inference computation"),
            Line2D([0], [0], color=ORANGE, lw=1.4, linestyle=(0, (3, 2)), label="training-only supervision"),
        ],
        loc="upper right",
        frameon=False,
        fontsize=6.3,
        handlelength=1.8,
        bbox_to_anchor=(0.995, 1.01),
    )

    # Inference lane.
    ax.text(0.12, 5.50, "Inference path", fontsize=6.9, fontweight="bold", color=LINE)
    box(ax, 0.25, 4.25, 1.55, 0.78, "story + query\n$x=(s,q)$", "white", BLUE, fontsize=6.4, weight="bold")
    box(ax, 2.05, 4.25, 1.75, 0.78, "pretrained\nencoder $f_\\theta$", BLUE_FILL, BLUE, fontsize=6.25, weight="bold")
    box(ax, 4.05, 4.25, 1.95, 0.78, "task adapter\ntext-grounded units $U$", "white", BLUE, fontsize=6.1, weight="bold")
    box(ax, 6.35, 4.02, 2.70, 1.24, "shared TRUA core\nrelation-conditioned aggregation\nquery-guided step logits $S^s$", TEAL_FILL, TEAL, fontsize=6.05, weight="bold")
    box(ax, 9.45, 4.25, 1.85, 0.78, "updated units\n$U'$", "white", TEAL, fontsize=6.3, weight="bold")
    box(ax, 11.55, 4.25, 1.75, 0.78, "prediction head\n$z,\\ \\hat y$", PURPLE_FILL, PURPLE, fontsize=6.25, weight="bold")

    y_mid = 4.64
    orth_arrow(ax, [(1.80, y_mid), (2.05, y_mid)])
    orth_arrow(ax, [(3.80, y_mid), (4.05, y_mid)])
    orth_arrow(ax, [(6.00, y_mid), (6.35, y_mid)])
    orth_arrow(ax, [(9.05, y_mid), (9.45, y_mid)])
    orth_arrow(ax, [(11.30, y_mid), (11.55, y_mid)])

    # Adapter details below the adapter box, without extra diagonal arrows.
    token_stack(ax, 4.23, 3.55, BLUE, count=5)
    ax.text(4.77, 3.40, "entity spans or candidate sentences", fontsize=5.5, color=MUTED, ha="center")
    orth_arrow(ax, [(5.02, 4.25), (5.02, 3.99), (4.77, 3.99), (4.77, 3.92)], color=BLUE, lw=0.9, ms=6.5)

    # Internal TRUA details. These are contained, not part of external routing.
    box(ax, 6.62, 3.10, 1.12, 0.52, "global\nmessages $m^g$", GREEN_FILL, GREEN, fontsize=5.3, weight="bold")
    box(ax, 7.96, 3.10, 1.12, 0.52, "step\nmessages $m^s$", ORANGE_FILL, ORANGE, fontsize=5.3, weight="bold")
    orth_arrow(ax, [(7.18, 4.02), (7.18, 3.62)], color=TEAL, lw=0.9, ms=6.5)
    orth_arrow(ax, [(8.52, 4.02), (8.52, 3.62)], color=TEAL, lw=0.9, ms=6.5)
    box(ax, 7.25, 2.42, 1.25, 0.42, "fuse $m^g+m^s$", "white", TEAL, fontsize=5.25, weight="bold")
    orth_arrow(ax, [(7.18, 3.10), (7.18, 2.63), (7.25, 2.63)], color=TEAL, lw=0.85, ms=6.0)
    orth_arrow(ax, [(8.52, 3.10), (8.52, 2.63), (8.50, 2.63)], color=TEAL, lw=0.85, ms=6.0)

    # CLS bypass, routed orthogonally above the main lane.
    ax.text(6.95, 5.42, "$h_{cls}$ bypass to classifier", fontsize=5.55, color=MUTED, ha="center")
    orth_arrow(ax, [(2.93, 5.03), (2.93, 5.32), (11.98, 5.32), (11.98, 5.03)], lw=0.95, ms=6.8)

    # Training lane.
    panel(ax, 0.25, 0.58, 13.05, 1.55, "Training signals only; not provided at inference", "#FFF8EF", "#E8C894")
    box(ax, 0.62, 1.08, 1.62, 0.48, "final label $y$", "white", ORANGE, fontsize=5.8)
    box(ax, 2.55, 1.08, 1.72, 0.48, "$L_{main}$ on $z$", "white", ORANGE, fontsize=5.8)
    box(ax, 4.85, 1.08, 1.78, 0.48, "gold trace $\\tau$", "white", ORANGE, fontsize=5.8)
    box(ax, 6.95, 1.08, 1.84, 0.48, "$L_{step}$ on $S^s$", "white", ORANGE, fontsize=5.8)
    box(ax, 9.25, 1.08, 1.85, 0.48, "optional\n$L_{edge}, L_{cons}$", "white", ORANGE, fontsize=5.35, dashed=True)
    box(ax, 11.72, 0.95, 1.85, 0.74, "total loss\n$L$", "#FFE7C4", ORANGE, fontsize=6.05, weight="bold")

    orth_arrow(ax, [(2.24, 1.32), (2.55, 1.32)], color=ORANGE, dashed=True, lw=1.0, ms=6.6)
    orth_arrow(ax, [(6.63, 1.32), (6.95, 1.32)], color=ORANGE, dashed=True, lw=1.0, ms=6.6)
    orth_arrow(ax, [(4.27, 1.32), (4.85, 1.32)], color=ORANGE, dashed=True, lw=1.0, ms=6.6)
    orth_arrow(ax, [(4.27, 1.32), (4.56, 1.32), (4.56, 1.80), (7.85, 1.80), (7.85, 1.56)], color=ORANGE, dashed=True, lw=0.95, ms=6.0)
    orth_arrow(ax, [(8.79, 1.32), (11.72, 1.32)], color=ORANGE, dashed=True, lw=1.0, ms=6.6)
    orth_arrow(ax, [(11.10, 1.32), (11.72, 1.32)], color=ORANGE, dashed=True, lw=1.0, ms=6.6)
    orth_arrow(ax, [(4.27, 1.32), (4.56, 1.32), (4.56, 2.28), (8.52, 2.28), (8.52, 3.10)], color=ORANGE, dashed=True, lw=0.9, ms=6.0)
    orth_arrow(ax, [(12.48, 4.25), (12.48, 2.13), (12.48, 1.69)], color=ORANGE, dashed=True, lw=0.9, ms=6.0)

    # Bottom guardrail.
    box(
        ax,
        0.25,
        0.12,
        13.05,
        0.30,
        "Inference receives story/query text and adapter-derived grounding only: no gold graph edges, trace/path/proof, or relation labels.",
        GRAY_FILL,
        GRAY,
        fontsize=5.65,
        weight="bold",
        color="#334155",
    )

    fig.tight_layout(pad=0.25)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(args.out / f"figure1_trua_overview.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
