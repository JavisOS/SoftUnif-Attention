"""Draw corrected Figure 4 without requiring matplotlib.

The original matplotlib script remains the canonical generator when the
development machine is available. This local fallback uses Pillow plus
ReportLab so the paper figure can be regenerated on a minimal Mac runtime.
"""

from __future__ import annotations

import math
import base64
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


OUT_DIR = Path("tsra_aaai2026_final/figures")
PAPER_FIG_DIR = Path("tsra_paper_figures")
W, H = 2400, 1940

TEAL = "#007C89"
RED = "#B23A48"
GRID = "#E2E8F0"
TEXT = "#1F2933"
GRAY = "#66717D"
WHITE = "#FFFFFF"

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], s: str, fnt, fill=TEXT) -> None:
    bbox = draw.textbbox((0, 0), s, font=fnt)
    draw.text((xy[0] - (bbox[2] - bbox[0]) / 2, xy[1] - (bbox[3] - bbox[1]) / 2), s, font=fnt, fill=fill)


def draw_rotated_label(img: Image.Image, xy: tuple[float, float], s: str, angle: float, fnt, fill=TEXT) -> None:
    pad = 12
    bbox = ImageDraw.Draw(img).textbbox((0, 0), s, font=fnt)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tmp = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (255, 255, 255, 0))
    td = ImageDraw.Draw(tmp)
    td.text((pad, pad), s, font=fnt, fill=fill)
    rot = tmp.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    img.alpha_composite(rot, (int(xy[0] - rot.width / 2), int(xy[1] - rot.height / 2)))


def draw_panel(
    img: Image.Image,
    box: tuple[int, int, int, int],
    title: str,
    ylabel: str,
    labels: list[str],
    base: list[float],
    base_err: list[float],
    tsra: list[float],
    tsra_err: list[float],
    base_label: str,
    legend: bool = False,
    rotate_x: bool = False,
) -> None:
    draw = ImageDraw.Draw(img)
    x0, y0, w, h = box
    title_font = font(34, bold=True)
    axis_font = font(26)
    tick_font = font(25)
    legend_font = font(25)
    text_center(draw, (x0 + w / 2, y0 + 24), title, title_font)

    left = x0 + 120
    right = x0 + w - 45
    top = y0 + 95
    bottom = y0 + h - 110
    axis_w = right - left
    axis_h = bottom - top

    draw.line((left, top, left, bottom), fill=TEXT, width=4)
    draw.line((left, bottom, right, bottom), fill=TEXT, width=4)

    for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = bottom - t * axis_h
        draw.line((left, y, right, y), fill=GRID, width=3)
        label = f"{t:.1f}"
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((left - 22 - (bbox[2] - bbox[0]), y - 12), label, font=tick_font, fill=TEXT)

    draw_rotated_label(img, (x0 + 30, top + axis_h / 2), ylabel, 90, axis_font)

    n = len(labels)
    group_w = axis_w / n
    bar_w = min(54, group_w * 0.28)

    def yval(v: float) -> float:
        return bottom - max(0.0, min(1.0, v)) * axis_h

    for i, label in enumerate(labels):
        cx = left + group_w * (i + 0.5)
        for offset, value, err, color in [(-bar_w * 0.65, base[i], base_err[i], RED), (bar_w * 0.65, tsra[i], tsra_err[i], TEAL)]:
            bx0 = cx + offset - bar_w / 2
            bx1 = cx + offset + bar_w / 2
            by = yval(value)
            draw.rounded_rectangle((bx0, by, bx1, bottom), radius=3, fill=color)
            ey0, ey1 = yval(value + err), yval(value - err)
            ex = cx + offset
            draw.line((ex, ey0, ex, ey1), fill=TEXT, width=4)
            draw.line((ex - 12, ey0, ex + 12, ey0), fill=TEXT, width=4)
            draw.line((ex - 12, ey1, ex + 12, ey1), fill=TEXT, width=4)
        lx = cx
        ly = bottom + 45
        if rotate_x:
            draw_rotated_label(img, (lx, ly), label, 18, tick_font)
        else:
            for j, part in enumerate(label.split("\n")):
                text_center(draw, (lx, ly + j * 28), part, tick_font)

    if legend:
        lx, ly = right - 290, top + 10
        draw.rounded_rectangle((lx, ly, lx + 22, ly + 22), radius=2, fill=RED)
        draw.text((lx + 34, ly - 2), base_label, font=legend_font, fill=TEXT)
        draw.rounded_rectangle((lx, ly + 38, lx + 22, ly + 60), radius=2, fill=TEAL)
        draw.text((lx + 34, ly + 36), "TRUA", font=legend_font, fill=TEXT)


def build_png(path: Path) -> None:
    img = Image.new("RGBA", (W, H), WHITE)
    draw = ImageDraw.Draw(img)

    text_center(
        draw,
        (W / 2, 70),
        "Trace Supervision Improves Reasoning Alignment and Long-Hop Generalization",
        font(42, bold=True),
    )

    margin_x = 90
    gap_x = 70
    gap_y = 115
    panel_w = (W - 2 * margin_x - gap_x) // 2
    panel_h = 760
    y_top = 140
    y_bottom = y_top + panel_h + gap_y

    labels3 = ["BERT", "RoBERTa", "DeBERTa"]
    draw_panel(
        img,
        (margin_x, y_top, panel_w, panel_h),
        "ProofWriter depth-5: trace@1",
        "gold trace top-1",
        labels3,
        [0.146, 0.169, 0.169],
        [0.006, 0.018, 0.023],
        [0.786, 0.789, 0.604],
        [0.003, 0.009, 0.315],
        "baseline",
        legend=True,
    )
    draw_panel(
        img,
        (margin_x + panel_w + gap_x, y_top, panel_w, panel_h),
        "PrOntoQA-OOD: trace@1",
        "gold trace top-1",
        labels3,
        [0.213, 0.173, 0.251],
        [0.045, 0.033, 0.037],
        [0.467, 0.486, 0.518],
        [0.040, 0.088, 0.134],
        "baseline",
    )

    draw_panel(
        img,
        (margin_x, y_bottom, panel_w, panel_h),
        "ProofWriter depth-5: answer vs trace alignment",
        "score",
        ["BERT\nanswer", "BERT\ntrace@1", "RoBERTa\nanswer", "RoBERTa\ntrace@1"],
        [0.8818, 0.1459, 0.8064, 0.1694],
        [0.0086, 0.0061, 0.0783, 0.0178],
        [0.8842, 0.7863, 0.8511, 0.7890],
        [0.0037, 0.0029, 0.0064, 0.0086],
        "baseline",
        legend=True,
    )
    draw_panel(
        img,
        (margin_x + panel_w + gap_x, y_bottom, panel_w, panel_h),
        "CLUTRR 2/3-hop train -> 6-10-hop test",
        "long-hop accuracy",
        ["BERT", "RoBERTa", "DeBERTa", "DeBERTa-v3", "ModernBERT"],
        [0.1946, 0.2223, 0.2415, 0.2426, 0.1873],
        [0.0254, 0.0054, 0.0122, 0.0750, 0.0235],
        [0.3066, 0.3930, 0.4198, 0.5169, 0.3859],
        [0.0615, 0.0334, 0.0335, 0.0622, 0.0147],
        "vanilla encoder",
        legend=True,
        rotate_x=True,
    )

    note = "Metrics are 3-seed means +/- sample std. CLUTRR compares the true vanilla encoder classifier audit with TRUA, not the TRUA label-only ablation."
    text_center(draw, (W / 2, H - 55), note, font(25), fill=GRAY)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, quality=95)


def png_to_pdf(png: Path, pdf: Path) -> None:
    pdf.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(pdf), pagesize=(W, H))
    c.drawImage(ImageReader(str(png)), 0, 0, width=W, height=H)
    c.showPage()
    c.save()


def png_to_svg(png: Path, svg: Path) -> None:
    encoded = base64.b64encode(png.read_bytes()).decode("ascii")
    svg.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
                f'  <image href="data:image/png;base64,{encoded}" width="{W}" height="{H}"/>',
                "</svg>",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    png = OUT_DIR / "fig4_trace_faithfulness_depth_curves.png"
    pdf = OUT_DIR / "fig4_trace_faithfulness_depth_curves.pdf"
    svg = OUT_DIR / "fig4_trace_faithfulness_depth_curves.svg"
    build_png(png)
    png_to_pdf(png, pdf)
    png_to_svg(png, svg)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    paper_png = PAPER_FIG_DIR / "fig4_trace_faithfulness_depth_curves.png"
    build_png(paper_png)
    png_to_pdf(paper_png, PAPER_FIG_DIR / "fig4_trace_faithfulness_depth_curves.pdf")
    png_to_svg(paper_png, PAPER_FIG_DIR / "fig4_trace_faithfulness_depth_curves.svg")


if __name__ == "__main__":
    main()
