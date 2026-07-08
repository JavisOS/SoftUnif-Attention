"""Generate external-method comparison figures for TRUA.

This script intentionally uses Pillow + ReportLab instead of matplotlib so the
figures can be regenerated on the local Codex runtime. Values are copied from
EXPERIMENT_REPORT.md and current aggregated artifacts.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


OUT_DIR = Path("tsra_aaai2026_final/figures")
PAPER_DIR = Path("tsra_paper_figures")
W, H = 2400, 1550
GRID = "#E2E8F0"
TEXT = "#1F2933"
GRAY = "#66717D"
TRUA = "#007C89"
BASE = "#B23A48"
EXT = "#4666A6"
REF = "#7C5C9E"
LIGHT = "#F7FAFC"
WHITE = "#FFFFFF"

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


@dataclass
class Bar:
    label: str
    value: float
    err: float = 0.0
    color: str = EXT
    note: str = ""


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fnt, fill=TEXT) -> None:
    bbox = draw.textbbox((0, 0), text, font=fnt)
    draw.text((xy[0] - (bbox[2] - bbox[0]) / 2, xy[1] - (bbox[3] - bbox[1]) / 2), text, font=fnt, fill=fill)


def rotated(img: Image.Image, xy: tuple[float, float], text: str, angle: float, fnt, fill=TEXT) -> None:
    bbox = ImageDraw.Draw(img).textbbox((0, 0), text, font=fnt)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tmp = Image.new("RGBA", (tw + 24, th + 24), (255, 255, 255, 0))
    ImageDraw.Draw(tmp).text((12, 12), text, font=fnt, fill=fill)
    rot = tmp.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    img.alpha_composite(rot, (int(xy[0] - rot.width / 2), int(xy[1] - rot.height / 2)))


def wrap_label(label: str, max_chars: int = 13) -> str:
    if "\n" in label:
        return label
    words = label.split()
    lines: list[str] = []
    current = ""
    for word in words:
        nxt = word if not current else f"{current} {word}"
        if len(nxt) <= max_chars:
            current = nxt
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines[:3])


def draw_bar_panel(
    img: Image.Image,
    box: tuple[int, int, int, int],
    title: str,
    ylabel: str,
    bars: list[Bar],
    ymax: float = 1.0,
    rotate_x: bool = False,
    footnote: str | None = None,
) -> None:
    draw = ImageDraw.Draw(img)
    x0, y0, w, h = box
    draw.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=16, fill=WHITE, outline="#D9E2EC", width=2)
    center(draw, (x0 + w / 2, y0 + 35), title, font(31, True))
    left = x0 + 105
    right = x0 + w - 42
    top = y0 + 92
    bottom = y0 + h - 150
    aw = right - left
    ah = bottom - top
    draw.line((left, top, left, bottom), fill=TEXT, width=3)
    draw.line((left, bottom, right, bottom), fill=TEXT, width=3)
    for k in range(6):
        val = ymax * k / 5
        y = bottom - val / ymax * ah
        draw.line((left, y, right, y), fill=GRID, width=2)
        s = f"{val:.1f}" if ymax <= 1.05 else f"{val:.0f}"
        bbox = draw.textbbox((0, 0), s, font=font(22))
        draw.text((left - bbox[2] + bbox[0] - 18, y - 12), s, font=font(22), fill=TEXT)
    rotated(img, (x0 + 32, top + ah / 2), ylabel, 90, font(23))
    bw = min(62, aw / len(bars) * 0.58)
    group = aw / len(bars)
    for i, bar in enumerate(bars):
        cx = left + group * (i + 0.5)
        yv = bottom - min(bar.value, ymax) / ymax * ah
        draw.rounded_rectangle((cx - bw / 2, yv, cx + bw / 2, bottom), radius=4, fill=bar.color)
        if bar.err:
            ey0 = bottom - min(bar.value + bar.err, ymax) / ymax * ah
            ey1 = bottom - max(bar.value - bar.err, 0) / ymax * ah
            draw.line((cx, ey0, cx, ey1), fill=TEXT, width=3)
            draw.line((cx - 12, ey0, cx + 12, ey0), fill=TEXT, width=3)
            draw.line((cx - 12, ey1, cx + 12, ey1), fill=TEXT, width=3)
        value_text = f"{bar.value:.3f}" if bar.value < 1 else f"{bar.value:.2f}"
        center(draw, (cx, yv - 18), value_text, font(19), fill=TEXT)
        label = wrap_label(bar.label)
        if rotate_x:
            rotated(img, (cx, bottom + 52), label.replace("\n", " "), 18, font(20))
        else:
            for j, part in enumerate(label.split("\n")):
                center(draw, (cx, bottom + 34 + j * 24), part, font(19), fill=TEXT)
        if bar.note:
            center(draw, (cx, bottom + 100), bar.note, font(16), fill=GRAY)
    if footnote:
        center(draw, (x0 + w / 2, y0 + h - 16), footnote, font(16), fill=GRAY)


def legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    items = [(BASE, "baseline / adapted"), (TRUA, "TRUA"), (EXT, "external method"), (REF, "official/reference")]
    for i, (color, label) in enumerate(items):
        xx = x + i * 300
        draw.rounded_rectangle((xx, y, xx + 24, y + 24), radius=3, fill=color)
        draw.text((xx + 34, y - 1), label, font=font(21), fill=TEXT)


def png_to_pdf(png: Path, pdf: Path) -> None:
    c = canvas.Canvas(str(pdf), pagesize=(W, H))
    c.drawImage(ImageReader(str(png)), 0, 0, width=W, height=H)
    c.showPage()
    c.save()


def png_to_svg(png: Path, svg: Path) -> None:
    encoded = base64.b64encode(png.read_bytes()).decode("ascii")
    svg.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
        f'  <image href="data:image/png;base64,{encoded}" width="{W}" height="{H}"/>\n'
        "</svg>\n",
        encoding="utf-8",
    )


def save_all(name: str, img: Image.Image) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for directory in (OUT_DIR, PAPER_DIR):
        png = directory / f"{name}.png"
        pdf = directory / f"{name}.pdf"
        svg = directory / f"{name}.svg"
        img.convert("RGB").save(png, quality=95)
        png_to_pdf(png, pdf)
        png_to_svg(png, svg)


def make_overview() -> None:
    img = Image.new("RGBA", (W, H), LIGHT)
    draw = ImageDraw.Draw(img)
    center(draw, (W / 2, 58), "External Baseline Comparisons Across TRUA Benchmarks", font(42, True))
    legend(draw, 390, 104)
    pw = (W - 150) // 2
    ph = 570
    draw_bar_panel(
        img,
        (50, 165, pw, ph),
        "CLUTRR long-hop accuracy",
        "accuracy",
        [
            Bar("Abstractor/RCA", 0.1095, color=BASE, note="adapted"),
            Bar("DAT", 0.1424, color=BASE, note="adapted"),
            Bar("RAT", 0.3483, color=EXT, note="graph"),
            Bar("TRUA DeBERTa-v3", 0.5169, 0.0622, TRUA, "ours"),
            Bar("EdgeTransformer", 0.6847, color=REF, note="graph"),
        ],
        footnote="Primary split data_089907f8; long-hop = 6-10.",
    )
    draw_bar_panel(
        img,
        (100 + pw, 165, pw, ph),
        "ProofWriter depth-5 accuracy",
        "accuracy",
        [
            Bar("DAT", 0.6390, 0.3111, BASE, "same-input"),
            Bar("LoGiPT", 0.6564, color=EXT, note="LM"),
            Bar("Abstractor/RCA", 0.8221, 0.0930, BASE, "same-input"),
            Bar("AAI Qwen32B", 0.8350, color=EXT, note="LLM"),
            Bar("TRUA BERT", 0.8842, 0.0037, TRUA, "ours"),
            Bar("FaiRR", 0.9840, color=REF, note="official"),
        ],
        footnote="FaiRR uses its official proof pipeline; other rows are not identical task interfaces.",
    )
    draw_bar_panel(
        img,
        (50, 800, pw, ph),
        "RuleTaker result comparison",
        "score",
        [
            Bar("DAT raw-QDep", 0.5583, 0.0605, BASE, "same-input"),
            Bar("Abstractor/RCA raw-QDep", 0.5603, 0.0639, BASE, "same-input"),
            Bar("NLProofS", 0.6796, color=EXT, note="answer"),
            Bar("GFaiR full", 0.9086, color=REF, note="proof"),
            Bar("IBR", 0.9372, color=REF, note="full"),
            Bar("TRUA DeBERTa", 0.9667, 0.0011, TRUA, "GFaiR split"),
        ],
        footnote="Mixed RuleTaker splits/metrics; use as method-reference overview.",
    )
    draw_bar_panel(
        img,
        (100 + pw, 800, pw, ph),
        "PrOntoQA-OOD trace / latent references",
        "score",
        [
            Bar("Abstractor/RCA trace", 0.2044, 0.0158, BASE, "same-input"),
            Bar("DAT trace", 0.2400, 0.0850, BASE, "same-input"),
            Bar("TRUA DeBERTa trace", 0.5378, 0.1482, TRUA, "ours"),
            Bar("CODI GPT2 acc", 0.8144, color=EXT, note="latent"),
            Bar("CODI Llama1B acc", 0.8730, color=EXT, note="latent"),
            Bar("Coconut acc", 1.0000, color=REF, note="val"),
        ],
        footnote="Trace@1 and final-statement accuracy are shown separately as scores.",
    )
    center(draw, (W / 2, H - 35), "Values are from 3-seed TRUA artifacts or completed official/reference runs recorded in EXPERIMENT_REPORT.md.", font(22), fill=GRAY)
    save_all("fig_external_baseline_overview", img)


def make_single(name: str, title: str, ylabel: str, bars: list[Bar], footnote: str) -> None:
    img = Image.new("RGBA", (W, 950), LIGHT)
    global H
    old_h = H
    H = 950
    draw = ImageDraw.Draw(img)
    center(draw, (W / 2, 60), title, font(44, True))
    legend(draw, 390, 104)
    draw_bar_panel(img, (80, 165, W - 160, 720), "", ylabel, bars, footnote=footnote, rotate_x=False)
    save_all(name, img)
    H = old_h


def main() -> None:
    make_overview()
    make_single(
        "fig_clutrr_external_comparison",
        "CLUTRR External Comparison: Long-Hop Accuracy",
        "long-hop accuracy",
        [
            Bar("Abstractor/RCA adapted", 0.1095, color=BASE, note="raw text"),
            Bar("DAT adapted", 0.1424, color=BASE, note="raw text"),
            Bar("RAT", 0.3483, color=EXT, note="graph"),
            Bar("Vanilla DeBERTa-v3", 0.2426, 0.0750, BASE, "plain"),
            Bar("TRUA DeBERTa-v3", 0.5169, 0.0622, TRUA, "ours"),
            Bar("EdgeTransformer", 0.6847, color=REF, note="graph"),
        ],
        "Primary split data_089907f8; long-hop = 6-10. Graph rows use structured inputs.",
    )
    make_single(
        "fig_proofwriter_external_comparison",
        "ProofWriter External Comparison",
        "depth-5 accuracy",
        [
            Bar("DAT", 0.6390, 0.3111, BASE, "same-input"),
            Bar("LoGiPT raw", 0.6564, color=EXT, note="LM"),
            Bar("Abstractor/RCA", 0.8221, 0.0930, BASE, "same-input"),
            Bar("AAI no-intervention", 0.8350, color=EXT, note="Qwen32B"),
            Bar("TRUA BERT", 0.8842, 0.0037, TRUA, "ours"),
            Bar("FaiRR", 0.9840, color=REF, note="official"),
        ],
        "FaiRR uses its official end-to-end proof pipeline; same-input adapters use raw text + query.",
    )
    make_single(
        "fig_ruletaker_external_comparison",
        "RuleTaker External Comparison",
        "score",
        [
            Bar("DAT raw-QDep", 0.5583, 0.0605, BASE, "same-input"),
            Bar("Abstractor/RCA raw-QDep", 0.5603, 0.0639, BASE, "same-input"),
            Bar("NLProofS answer", 0.6796, color=EXT, note="official"),
            Bar("GFaiR proof", 0.9086, color=REF, note="official"),
            Bar("IBR full", 0.9372, color=REF, note="official"),
            Bar("TRUA DeBERTa", 0.9667, 0.0011, TRUA, "GFaiR split"),
        ],
        "Rows are completed RuleTaker-family evaluations but use different released splits/metrics.",
    )
    make_single(
        "fig_prontoqa_external_comparison",
        "PrOntoQA-OOD External Comparison",
        "score",
        [
            Bar("Abstractor/RCA trace@1", 0.2044, 0.0158, BASE, "same-input"),
            Bar("DAT trace@1", 0.2400, 0.0850, BASE, "same-input"),
            Bar("TRUA DeBERTa trace@1", 0.5378, 0.1482, TRUA, "ours"),
            Bar("CODI GPT2 acc", 0.8144, color=EXT, note="latent"),
            Bar("CODI Llama1B acc", 0.8730, color=EXT, note="latent"),
            Bar("Coconut acc", 1.0000, color=REF, note="val"),
        ],
        "The discriminative label task is saturated; trace@1 is shown for same-input methods.",
    )


if __name__ == "__main__":
    main()
