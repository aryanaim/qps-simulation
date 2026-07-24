"""Line chart rendering — SVG (zero-dependency) and PNG (via matplotlib).

The output format is determined by the file extension on ``path``:

* ``.svg`` — dependency-free hand-written SVG.
* ``.png`` — rendered via matplotlib with the Agg backend.

Each series dict supports ``{"name": str, "points": [(x, y), ...],
"color": str, "ci_low": [(x, y), ...], "ci_high": [(x, y), ...]}``.
"""

from __future__ import annotations

import html
import math
import re
from pathlib import Path
from typing import Any, Iterable

from .io import ensure_dir


# ── Publication-ready color palette (colorblind-safe, grayscale-distinguishable) ─────
# Based on ColorBrewer and viridis/ibm palettes - works in color and B&W
PALETTE = [
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#009E73",  # Bluish green
    "#CC79A7",  # Reddish purple
    "#56B4E9",  # Sky blue
    "#E69F00",  # Orange
    "#F0E442",  # Yellow
    "#000000",  # Black
]

# Marker styles for grayscale differentiation
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

# Line styles for additional differentiation
LINE_STYLES = ["-", "--", "-.", ":"]

_RICH_TAG = re.compile(r"</?(?:sub|sup)>")

# ── helpers shared by both renderers ──────────────────────────────


def _strip_html(text: str) -> str:
    """Remove <sub>/<sup> tags for plain-text renderers."""
    return _RICH_TAG.sub("", text)


def _rich_text_svg(text: str) -> str:
    """Escape text while preserving simple SVG subscript/superscript markup."""
    tags = ("<sub>", "</sub>", "<sup>", "</sup>")
    result: list[str] = []
    active: str | None = None
    index = 0
    while index < len(text):
        if text.startswith("<sub>", index):
            active = "sub"
            index += len("<sub>")
            continue
        if text.startswith("<sup>", index):
            active = "super"
            index += len("<sup>")
            continue
        if text.startswith("</sub>", index) or text.startswith("</sup>", index):
            active = None
            index += len("</sub>") if text.startswith("</sub>", index) else len("</sup>")
            continue

        next_tag = min((pos for tag in tags if (pos := text.find(tag, index)) != -1), default=len(text))
        chunk = html.escape(text[index:next_tag])
        if chunk:
            if active:
                result.append(f'<tspan baseline-shift="{active}" font-size="75%">{chunk}</tspan>')
            else:
                result.append(chunk)
        index = next_tag
    return "".join(result)


def _fmt(value: float) -> str:
    """Format value with appropriate precision for axis ticks."""
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if abs(value) >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if abs(value) >= 0.01:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.2e}"


def _nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    """Generate nice tick positions."""
    if not math.isfinite(lo) or not math.isfinite(hi):
        return [0.0]
    if hi == lo:
        return [lo]
    step = (hi - lo) / max(1, count - 1)
    return [lo + step * i for i in range(count)]


# ── PNG renderer (publication-quality matplotlib) ────────────────


def _html_to_math(text: str) -> str:
    """Convert HTML-style ``<sub>``/``<sup>`` tags to matplotlib math text.

    Handles stacked sub+sup (e.g. ``C<sub>n</sub><sup>2</sup>`` →
    ``$C_n^2$``), individual tags (e.g. ``N<sub>tok</sub>`` → ``$N_{tok}$``),
    and mixed text (e.g. ``"r<sub>c</sub>=10 tokens/hour"`` →
    ``"$r_c$=10 tokens/hour"``).

    Falls back to plain text when no tags are present.
    """
    if not _RICH_TAG.search(text):
        return text

    s = text
    # Merge adjacent sub+sup on same single-letter base
    s = re.sub(r"([a-zA-Z])<sub>([^<]*)</sub>\s*<sup>([^<]*)</sup>", r"$\1_{\2}^{\3}$", s)
    s = re.sub(r"([a-zA-Z])<sup>([^<]*)</sup>\s*<sub>([^<]*)</sub>", r"$\1^{\2}_{\3}$", s)
    # Single sub/sup with a letter base
    s = re.sub(r"([a-zA-Z])<sub>([^<]*)</sub>", r"$\1_{\2}$", s)
    s = re.sub(r"([a-zA-Z])<sup>([^<]*)</sup>", r"$\1^{\2}$", s)
    # Remaining standalone tags (no preceding letter)
    s = re.sub(r"<sub>([^<]*)</sub>", r"$_{\1}$", s)
    s = re.sub(r"<sup>([^<]*)</sup>", r"$^{\1}$", s)
    return s


def _setup_publication_style(column: str = "double") -> None:
    """Configure matplotlib rcParams for publication-ready figures.

    Args:
        column: "single" for single-column width (3.5"), "double" for double-column (7"),
                or "full" for full page width (7.5")

    Uses STIX fonts (mathematical companion to Times New Roman), appropriate
    font sizes, and minimal grid styling.
    """
    import matplotlib

    # Base font sizes for different column widths
    if column == "single":
        base_font = 8
        label_font = 9
        title_font = 10
        tick_font = 7
        legend_font = 7
        line_width = 1.5
        marker_size = 3
    elif column == "double":
        base_font = 10
        label_font = 11
        title_font = 12
        tick_font = 9
        legend_font = 9
        line_width = 2.0
        marker_size = 4
    else:  # full
        base_font = 11
        label_font = 12
        title_font = 14
        tick_font = 10
        legend_font = 10
        line_width = 2.5
        marker_size = 5

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX", "Times New Roman", "DejaVu Serif"],
            "font.size": base_font,
            "mathtext.fontset": "stix",
            "axes.labelsize": label_font,
            "axes.titlesize": title_font,
            "xtick.labelsize": tick_font,
            "ytick.labelsize": tick_font,
            "legend.fontsize": legend_font,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.pad_inches": 0.1,
            "xtick.major.size": 4,
            "xtick.major.width": 0.8,
            "ytick.major.size": 4,
            "ytick.major.width": 0.8,
            "xtick.minor.size": 2,
            "xtick.minor.width": 0.5,
            "ytick.minor.size": 2,
            "ytick.minor.width": 0.5,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.3,
            "lines.linewidth": line_width,
            "lines.markersize": marker_size,
            "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        }
    )


def _matplotlib_render(
    output: Path,
    title: str,
    x_label: str,
    y_label: str,
    materialized: list[dict[str, Any]],
    width: int,
    height: int,
    x_scale: str,
    y_lo: float,
    y_hi: float,
    thresholds: list[dict] | None,
    bands: list[dict] | None,
    subtitle: str | None,
    column: str = "double",
    x_min: float | None = None,
    x_max: float | None = None,
    markers: bool = True,
    show_minor_grid: bool = True,
) -> None:
    """Render a publication-quality PNG line chart via matplotlib Agg backend."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter, LogFormatter

    matplotlib.use("Agg")  # no X server required
    _setup_publication_style(column)

    # Convert all text labels to proper math text
    fmt_title = _html_to_math(title)
    fmt_x = _html_to_math(x_label)
    fmt_y = _html_to_math(y_label)
    fmt_sub = _html_to_math(subtitle) if subtitle else None

    # Figure size: standard journal column widths at 300 DPI
    if column == "single":
        fig_w = 3.5  # inches (standard single-column width)
    elif column == "double":
        fig_w = 7.0  # inches (standard double-column width)
    else:
        fig_w = 7.5  # inches (full page width)
    fig_h = fig_w * (height / width)  # maintain aspect ratio
    dpi = 300
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")

    # ── bands (horizontal spans) ──
    if bands:
        for band in bands:
            y0, y1 = float(band["from"]), float(band["to"])
            color = band.get("color", "#fee2e2")
            ax.axhspan(min(y0, y1), max(y0, y1), color=color, alpha=0.35, lw=0, zorder=0)

    # ── data series and CI bands ──
    for idx, item in enumerate(materialized):
        color = item.get("color", PALETTE[idx % len(PALETTE)])
        points = item.get("points", [])
        if not points:
            continue
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]

        ci_low = item.get("ci_low", [])
        ci_high = item.get("ci_high", [])
        if ci_low and ci_high and len(ci_low) == len(points) and len(ci_high) == len(points):
            low_ys = [float(p[1]) for p in ci_low]
            high_ys = [float(p[1]) for p in ci_high]
            ax.fill_between(xs, low_ys, high_ys, color=color, alpha=0.15, lw=0, zorder=1)

        # Select marker and line style for this series
        marker = MARKERS[idx % len(MARKERS)] if markers else None
        ls = LINE_STYLES[idx % len(LINE_STYLES)]

        ax.plot(
            xs,
            ys,
            color=color,
            label=_html_to_math(str(item.get("name", ""))),
            ls=ls,
            marker=marker,
            markevery=max(1, len(xs) // 10),  # Show ~10 markers max
            zorder=3,
        )

    # ── thresholds (dashed lines) ──
    if thresholds:
        for threshold in thresholds:
            val = float(threshold["value"])
            color = threshold.get("color", "#666666")
            label = _html_to_math(str(threshold.get("label", "")))
            ax.axhline(val, color=color, ls=":", lw=1.2, label=label if label else None, zorder=2)

    # ── axis labels ──
    ax.set_xlabel(fmt_x)
    ax.set_ylabel(fmt_y)

    # ── ranges and scale ──
    ax.set_ylim(y_lo, y_hi)
    if x_min is not None and x_max is not None:
        ax.set_xlim(x_min, x_max)
    if x_scale == "log":
        ax.set_xscale("log", base=10)
        # Use nice log formatting
        ax.xaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False))
    else:
        # Use scalar formatter with fixed precision for linear scale
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    # ── grid (light, behind data) ──
    ax.grid(True, which="major", ls=":", lw=0.5, color="#cccccc", alpha=0.5, zorder=0)
    if show_minor_grid:
        ax.minorticks_on()
        ax.grid(True, which="minor", ls=":", lw=0.3, color="#dddddd", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # ── legend — inside the plot for few series, below for many ──
    n_series = len(materialized)
    handles, labels = ax.get_legend_handles_labels()
    # Filter out threshold labels if they duplicate series labels
    unique_labels = []
    unique_handles = []
    seen = set()
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)

    if n_series <= 4:
        legend = ax.legend(
            unique_handles,
            unique_labels,
            loc="best",
            frameon=True,
            fancybox=False,
            edgecolor="#cccccc",
            facecolor="white",
            framealpha=0.9,
            fontsize=matplotlib.rcParams["legend.fontsize"],
        )
    else:
        legend = ax.legend(
            unique_handles,
            unique_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=min(n_series, 4),
            frameon=False,
            fontsize=matplotlib.rcParams["legend.fontsize"],
        )

    # ── title ──
    if title:
        ax.set_title(fmt_title, fontsize=matplotlib.rcParams["axes.titlesize"], pad=16, fontweight="bold")

    # ── subtitle as text annotation ──
    if fmt_sub:
        fig.text(0.5, 0.01, fmt_sub, ha="center", va="bottom", fontsize=matplotlib.rcParams["legend.fontsize"], style="italic")

    fig.tight_layout(rect=[0, 0.06 if n_series > 4 else 0.02, 1, 0.95])
    fig.savefig(output, format="png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


# ── SVG renderer (zero-dependency) ────────────────────────────────


def _svg_render(
    output: Path,
    title: str,
    x_label: str,
    y_label: str,
    materialized: list[dict[str, Any]],
    width: int,
    height: int,
    x_scale: str,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    thresholds: list[dict] | None,
    bands: list[dict] | None,
    subtitle: str | None,
) -> None:
    """Render an SVG line chart — no non-stdlib dependencies."""
    left, right, top, bottom = 86, 30, 64, 86
    plot_w = width - left - right
    plot_h = height - top - bottom

    def tx_input(x: float) -> float:
        return math.log10(max(x, 1e-300)) if x_scale == "log" else x

    def sx(x: float) -> float:
        return left + (tx_input(x) - x_lo) / (x_hi - x_lo) * plot_w

    def sy(y: float) -> float:
        return top + (y_hi - y) / (y_hi - y_lo) * plot_h

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    if title:
        svg.append(
            f'<text x="{left}" y="34" font-family="STIX, Times New Roman, serif" font-size="22" font-weight="700" fill="#111827">{_rich_text_svg(title)}</text>'
        )
    if bands:
        for band in bands:
            y0, y1 = float(band["from"]), float(band["to"])
            color = html.escape(str(band.get("color", "#fee2e2")))
            top_y = sy(max(y0, y1))
            bottom_y = sy(min(y0, y1))
            svg.append(
                f'<rect x="{left}" y="{top_y:.2f}" width="{plot_w}" height="{max(0.0, bottom_y - top_y):.2f}" fill="{color}" opacity="0.35"/>'
            )

    # Y-axis ticks and grid
    for tick in _nice_ticks(y_lo, y_hi, 6):
        y = sy(tick)
        svg.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.2f}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="0.5"/>')
        svg.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="STIX, Times New Roman, serif" font-size="12" fill="#4b5563">{_rich_text_svg(_fmt(tick))}</text>'
        )

    # X-axis ticks
    if x_scale == "log":
        ticks = []
        lo_exp = math.floor(x_lo)
        hi_exp = math.ceil(x_hi)
        for exponent in range(lo_exp, hi_exp + 1):
            ticks.append(10.0**exponent)
    else:
        ticks = _nice_ticks(x_lo, x_hi, 6)
    for tick in ticks:
        x = sx(tick)
        svg.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{top}" y2="{height - bottom}" stroke="#f3f4f6" stroke-width="0.5"/>')
        label = f"1e{int(round(math.log10(tick)))}" if x_scale == "log" else _fmt(tick)
        svg.append(
            f'<text x="{x:.2f}" y="{height - bottom + 24}" text-anchor="middle" font-family="STIX, Times New Roman, serif" font-size="12" fill="#4b5563">{_rich_text_svg(label)}</text>'
        )

    # Axes
    svg.append(f'<line x1="{left}" x2="{width - right}" y1="{height - bottom}" y2="{height - bottom}" stroke="#111827" stroke-width="1"/>')
    svg.append(f'<line x1="{left}" x2="{left}" y1="{top}" y2="{height - bottom}" stroke="#111827" stroke-width="1"/>')

    if thresholds:
        for thresh in thresholds:
            y = sy(float(thresh["value"]))
            color = html.escape(str(thresh.get("color", "#991b1b")))
            label = str(thresh.get("label", ""))
            svg.append(
                f'<line x1="{left}" x2="{width - right}" y1="{y:.2f}" y2="{y:.2f}" stroke="{color}" stroke-width="1.5" stroke-dasharray="6 5"/>'
            )
            if label:
                svg.append(
                    f'<text x="{width - right - 4}" y="{y - 7:.2f}" text-anchor="end" font-family="STIX, Times New Roman, serif" font-size="12" fill="{color}">{_rich_text_svg(label)}</text>'
                )

    for idx, item in enumerate(materialized):
        color = html.escape(str(item.get("color", PALETTE[idx % len(PALETTE)])))
        points = item["points"]
        if not points:
            continue
        ci_low = [(float(x), float(y)) for x, y in item.get("ci_low", [])]
        ci_high = [(float(x), float(y)) for x, y in item.get("ci_high", [])]
        if ci_low and ci_high and len(ci_low) == len(points) and len(ci_high) == len(points):
            forward = " ".join(
                f"{'M' if i == 0 else 'L'}{sx(x):.2f},{sy(y):.2f}"
                for i, (x, y) in enumerate(ci_high)
            )
            backward = " ".join(
                f"L{sx(x):.2f},{sy(y):.2f}" for x, y in reversed(ci_low)
            )
            svg.append(f'<path d="{forward} {backward} Z" fill="{color}" opacity="0.18"/>')
        path_parts: list[str] = []
        for point_idx, (x, y) in enumerate(points):
            cmd = "M" if point_idx == 0 else "L"
            path_parts.append(f"{cmd}{sx(x):.2f},{sy(y):.2f}")
        svg.append(
            f'<path d="{" ".join(path_parts)}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
        )

        # Add markers at data points
        if len(points) > 1:
            marker_step = max(1, len(points) // 10)
            marker = MARKERS[idx % len(MARKERS)]
            for point_idx, (x, y) in enumerate(points):
                if point_idx % marker_step == 0:
                    mx, my = sx(x), sy(y)
                    r = 4
                    if marker == "o":
                        svg.append(f'<circle cx="{mx:.2f}" cy="{my:.2f}" r="{r}" fill="{color}" stroke="#ffffff" stroke-width="1"/>')
                    elif marker == "s":
                        svg.append(f'<rect x="{mx-r:.2f}" y="{my-r:.2f}" width="{2*r}" height="{2*r}" fill="{color}" stroke="#ffffff" stroke-width="1"/>')
                    elif marker == "^":
                        svg.append(f'<polygon points="{mx:.2f},{my-r:.2f} {mx-r:.2f},{my+r:.2f} {mx+r:.2f},{my+r:.2f}" fill="{color}" stroke="#ffffff" stroke-width="1"/>')
                    elif marker == "D":
                        svg.append(f'<polygon points="{mx:.2f},{my-r:.2f} {mx+r:.2f},{my:.2f} {mx:.2f},{my+r:.2f} {mx-r:.2f},{my:.2f}" fill="{color}" stroke="#ffffff" stroke-width="1"/>')

    # Legend
    legend_x = left
    legend_y = height - 30
    for idx, item in enumerate(materialized):
        color = html.escape(str(item.get("color", PALETTE[idx % len(PALETTE)])))
        name = str(item.get("name", f"Series {idx + 1}"))
        lx = legend_x + idx * 190
        svg.append(f'<line x1="{lx}" x2="{lx + 24}" y1="{legend_y}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        svg.append(
            f'<text x="{lx + 32}" y="{legend_y + 4}" font-family="STIX, Times New Roman, serif" font-size="13" fill="#111827">{_rich_text_svg(name)}</text>'
        )

    svg.append(
        f'<text x="{left + plot_w / 2:.2f}" y="{height - 44}" text-anchor="middle" font-family="STIX, Times New Roman, serif" font-size="14" fill="#111827">{_rich_text_svg(x_label)}</text>'
    )
    svg.append(
        f'<text x="22" y="{top + plot_h / 2:.2f}" text-anchor="middle" transform="rotate(-90 22 {top + plot_h / 2:.2f})" font-family="STIX, Times New Roman, serif" font-size="14" fill="#111827">{_rich_text_svg(y_label)}</text>'
    )
    if subtitle:
        svg.append(
            f'<text x="{left + plot_w / 2:.2f}" y="{height - 10}" text-anchor="middle" font-family="STIX, Times New Roman, serif" font-size="11" fill="#6b7280" font-style="italic">{_rich_text_svg(subtitle)}</text>'
        )
    svg.append("</svg>")
    output.write_text("\n".join(svg), encoding="utf-8")


# ── public entry point ────────────────────────────────────────────


def write_line_chart(
    path: str | Path,
    title: str,
    x_label: str,
    y_label: str,
    series: Iterable[dict],
    width: int = 960,
    height: int = 540,
    x_scale: str = "linear",
    y_min: float | None = None,
    y_max: float | None = None,
    thresholds: list[dict] | None = None,
    bands: list[dict] | None = None,
    subtitle: str | None = None,
    column: str = "double",  # "single", "double", "full"
    x_min: float | None = None,
    x_max: float | None = None,
    markers: bool = True,
    show_minor_grid: bool = True,
) -> Path:
    """Write a line chart. Format is determined by the file extension.

    ``.svg`` — zero-dependency hand-written SVG (works everywhere).
    ``.png`` — rendered via matplotlib (requires ``matplotlib``).

    Each series is {"name": str, "points": [(x, y), ...], "color": optional str,
    "ci_low": optional [(x, y), ...], "ci_high": optional [(x, y), ...]}.
    Thresholds are {"value": y, "label": str, "color": str}.
    Bands are {"from": y0, "to": y1, "color": str}.

    Publication options:
        column: "single" (3.5"), "double" (7"), or "full" (7.5") width
        markers: Add markers to lines for grayscale differentiation
        show_minor_grid: Show minor grid lines
    """
    output = Path(path)
    ensure_dir(output.parent)
    materialized = [dict(item) for item in series]
    all_points: list[tuple[float, float]] = []
    all_ci: list[float] = []
    for item in materialized:
        pts = [(float(x), float(y)) for x, y in item.get("points", [])]
        item["points"] = pts
        all_points.extend(pts)
        for key in ("ci_low", "ci_high"):
            ci = item.get(key, [])
            if ci:
                all_ci.extend(float(p[1]) for p in ci)

    if not all_points:
        all_points = [(0.0, 0.0), (1.0, 1.0)]

    x_values = [math.log10(max(x, 1e-300)) if x_scale == "log" else x for x, _ in all_points]
    y_values = [y for _, y in all_points]
    if all_ci:
        y_values = y_values + all_ci
    x_lo, x_hi = min(x_values), max(x_values)
    y_lo = min(y_values) if y_min is None else y_min
    y_hi = max(y_values) if y_max is None else y_max
    if thresholds:
        y_hi = max(y_hi, max(float(t["value"]) for t in thresholds))
        y_lo = min(y_lo, min(float(t["value"]) for t in thresholds))
    if bands:
        y_hi = max(y_hi, max(float(b["to"]) for b in bands))
        y_lo = min(y_lo, min(float(b["from"]) for b in bands))
    if x_hi == x_lo:
        x_hi = x_lo + 1.0
    if y_hi == y_lo:
        y_hi = y_lo + 1.0

    y_pad = (y_hi - y_lo) * 0.08
    y_lo -= y_pad
    y_hi += y_pad

    # Apply x-axis limits if provided
    if x_min is not None:
        x_lo = x_min
    if x_max is not None:
        x_hi = x_max

    fmt = output.suffix.lower()
    if fmt == ".png":
        _matplotlib_render(
            output, title, x_label, y_label, materialized, width, height, x_scale, y_lo, y_hi,
            thresholds, bands, subtitle, column, x_min, x_max, markers, show_minor_grid
        )
    else:
        _svg_render(
            output, title, x_label, y_label, materialized, width, height, x_scale,
            x_lo, x_hi, y_lo, y_hi, thresholds, bands, subtitle
        )
    return output


# ── Multi-panel figure support ────────────────────────────────────


def write_multi_panel(
    output: Path,
    panels: list[dict],
    column: str = "double",
    width: int = 960,
    height: int = 540,
    shared_x: bool = False,
    shared_y: bool = False,
) -> Path:
    """Create a multi-panel figure (subplots) for publication.

    Args:
        output: Output path (.png or .svg)
        panels: List of panel configs, each with:
            - "title": panel title
            - "series": list of series dicts
            - "x_label": x-axis label
            - "y_label": y-axis label
            - "x_scale": "linear" or "log"
            - "y_min", "y_max": y-axis limits
            - "thresholds": list of threshold dicts
            - "bands": list of band dicts
            - "subtitle": panel subtitle
        column: "single", "double", or "full"
        width, height: Base dimensions (used for aspect ratio)
        shared_x: Share x-axis across panels
        shared_y: Share y-axis across panels

    Returns:
        Output path
    """
    import matplotlib
    matplotlib.use("Agg")
    _setup_publication_style(column)
    import matplotlib.pyplot as plt

    n_panels = len(panels)
    if n_panels == 0:
        raise ValueError("At least one panel required")

    # Determine grid layout
    if n_panels == 1:
        nrows, ncols = 1, 1
    elif n_panels == 2:
        nrows, ncols = 1, 2
    elif n_panels <= 4:
        nrows, ncols = 2, 2
    elif n_panels <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 3, 3

    # Figure size
    if column == "single":
        fig_w = 3.5
    elif column == "double":
        fig_w = 7.0
    else:
        fig_w = 7.5
    fig_h = fig_w * (height / width) * nrows / max(1, ncols)
    dpi = 300

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=dpi,
                              sharex=shared_x, sharey=shared_y, squeeze=False)
    fig.patch.set_facecolor("white")

    axes_flat = axes.flatten()

    for idx, panel in enumerate(panels):
        ax = axes_flat[idx]

        title = panel.get("title", "")
        x_label = panel.get("x_label", "")
        y_label = panel.get("y_label", "")
        series = panel.get("series", [])
        x_scale = panel.get("x_scale", "linear")
        y_min = panel.get("y_min")
        y_max = panel.get("y_max")
        thresholds = panel.get("thresholds")
        bands = panel.get("bands")
        subtitle = panel.get("subtitle")

        fmt_title = _html_to_math(title)
        fmt_x = _html_to_math(x_label)
        fmt_y = _html_to_math(y_label)
        fmt_sub = _html_to_math(subtitle) if subtitle else None

        # Bands
        if bands:
            for band in bands:
                y0, y1 = float(band["from"]), float(band["to"])
                color = band.get("color", "#fee2e2")
                ax.axhspan(min(y0, y1), max(y0, y1), color=color, alpha=0.35, lw=0, zorder=0)

        # Series
        for sidx, item in enumerate(series):
            color = item.get("color", PALETTE[sidx % len(PALETTE)])
            points = item.get("points", [])
            if not points:
                continue
            xs = [float(p[0]) for p in points]
            ys = [float(p[1]) for p in points]

            ci_low = item.get("ci_low", [])
            ci_high = item.get("ci_high", [])
            if ci_low and ci_high and len(ci_low) == len(points) and len(ci_high) == len(points):
                low_ys = [float(p[1]) for p in ci_low]
                high_ys = [float(p[1]) for p in ci_high]
                ax.fill_between(xs, low_ys, high_ys, color=color, alpha=0.15, lw=0, zorder=1)

            marker = MARKERS[sidx % len(MARKERS)]
            ls = LINE_STYLES[sidx % len(LINE_STYLES)]

            ax.plot(xs, ys, color=color, label=_html_to_math(str(item.get("name", ""))),
                    ls=ls, marker=marker, markevery=max(1, len(xs) // 10), zorder=3)

        # Thresholds
        if thresholds:
            for threshold in thresholds:
                val = float(threshold["value"])
                color = threshold.get("color", "#666666")
                label = _html_to_math(str(threshold.get("label", "")))
                ax.axhline(val, color=color, ls=":", lw=1.2, label=label if label else None, zorder=2)

        ax.set_xlabel(fmt_x)
        ax.set_ylabel(fmt_y)

        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)
        if x_scale == "log":
            ax.set_xscale("log", base=10)
            ax.xaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False))
        else:
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        ax.grid(True, which="major", ls=":", lw=0.5, color="#cccccc", alpha=0.5, zorder=0)
        ax.minorticks_on()
        ax.grid(True, which="minor", ls=":", lw=0.3, color="#dddddd", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

        if title:
            ax.set_title(fmt_title, fontsize=matplotlib.rcParams["axes.titlesize"], pad=10, fontweight="bold")

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        seen = set()
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                unique_handles.append(h)
                unique_labels.append(l)
        if unique_handles:
            n_ser = len(series)
            if n_ser <= 3:
                ax.legend(unique_handles, unique_labels, loc="best", frameon=True,
                         fancybox=False, edgecolor="#cccccc", facecolor="white", framealpha=0.9,
                         fontsize=matplotlib.rcParams["legend.fontsize"])
            else:
                ax.legend(unique_handles, unique_labels, loc="upper center",
                         bbox_to_anchor=(0.5, -0.2), ncol=min(n_ser, 4), frameon=False,
                         fontsize=matplotlib.rcParams["legend.fontsize"])

    # Hide unused axes
    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(output, format=output.suffix[1:], dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return output


# Import for multi-panel
from matplotlib.ticker import ScalarFormatter, LogFormatter
