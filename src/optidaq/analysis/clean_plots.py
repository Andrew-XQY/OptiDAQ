"""Fixed-metric PDF report and publication-sized figures. No data access."""

import os
from pathlib import Path
import textwrap
import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import matplotlib.pyplot as plt
import numpy as np

COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
LONG_GAP_FACTOR = 3.0
STYLE = {
    "font.family": "Arial",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.65,
    "lines.linewidth": 1.15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "pdf.use14corefonts": False,
    "axes.formatter.use_mathtext": False,
}


def require_arial():
    """Use installed Arial or a user-supplied licensed font, never a substitute."""
    explicit = os.environ.get("CLEAR2026_ARIAL_FONT")
    if explicit:
        path = Path(explicit).expanduser()
        font_manager.fontManager.addfont(str(path))
        if font_manager.FontProperties(fname=path).get_name() != "Arial":
            raise RuntimeError("CLEAR2026_ARIAL_FONT must point to an Arial font file")
    try:
        return font_manager.findfont("Arial", fallback_to_default=False)
    except ValueError as exc:
        raise RuntimeError(
            "Arial is required for report.pdf. Install Arial or set "
            "CLEAR2026_ARIAL_FONT to your licensed Arial.ttf file."
        ) from exc


def number(value, decimals=2):
    return "-" if value is None or not np.isfinite(value) else ("%%.%df" % decimals) % value


def percentage(value, *, fraction=False, signed=False):
    if value is None or not np.isfinite(value):
        return "-"
    if fraction:
        return "%.3f%%" % (100 * value)
    return ("%+.2f%%" if signed else "%.2f%%") % value


def review_flags(audit, frames):
    """Screening rules, separate from the integrity PASS/FAIL decision."""
    flags = []
    for cam, check in audit["cameras"].items():
        gaps = check["frame_interval_s"]
        median, maximum = gaps.get("median"), gaps.get("max")
        if median and maximum is not None and maximum > LONG_GAP_FACTOR * median:
            flags.append(
                "%s: longest received-image gap %s s exceeds 3 times the median (%s s)."
                % (cam, number(maximum, 3), number(median, 3))
            )
        if check["identical_duplicate_images"]:
            flags.append(
                "%s: %d images exactly repeat earlier pixel values."
                % (cam, check["identical_duplicate_images"])
            )
        flat = sum(f["camera"] == cam and f["max"] == f["min"] for f in frames)
        if flat:
            flags.append("%s: %d images have the same value at every pixel." % (cam, flat))
    return flags


def _text(fig, x, y, text, size=9, color="#222222", weight="normal"):
    fig.text(
        x, y, text, fontsize=size, color=color, weight=weight, ha="left", va="top", linespacing=1.45
    )


def _table(fig, y, headings, rows, widths):
    height = 0.029 * (len(rows) + 1)
    ax = fig.add_axes([0.075, y - height, 0.85, height])
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=headings,
        colWidths=widths,
        bbox=[0, 0, 1, 1],
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        cell.PAD = 0.035
        cell.set_facecolor("white")
        cell.set_edgecolor("#BBBBBB")
        cell.set_linewidth(0.45)
        cell.visible_edges = "B" if row == 0 or row == len(rows) else ""
        if row == 0:
            cell.get_text().set_weight("bold")
    return y - height - 0.025


def summary_figure(a, frames, *, cache_used=False):
    fig = plt.figure(figsize=(8.27, 11.69))
    cameras = list(a["cameras"])
    checks, stats, p = a["cameras"], a["statistics"], a["pairing"]
    _text(fig, 0.075, 0.951, "Dataset check", 18, weight="bold")
    _text(fig, 0.075, 0.917, a["run"], 10)
    _text(
        fig,
        0.075,
        0.876,
        "Files and recorded pairing: " + ("PASS" if a["validation_passed"] else "FAIL"),
        12,
        color="#176440" if a["validation_passed"] else "#A32724",
        weight="bold",
    )
    _text(
        fig,
        0.075,
        0.847,
        "%d/%d patterns completed    |    %s seconds    |    %s saved pairs/second"
        % (
            a["completed_patterns"],
            a["requested_patterns"],
            number(a["duration_s"]),
            number(a.get("complete_pairs_per_s"), 3),
        ),
    )
    rows = [
        [
            cam + " images / information / database records",
            "%d / %d / %d; expected %d each"
            % (c["png_count"], c["sidecar_count"], c["db_count"], c["expected_frames"]),
        ]
        for cam, c in checks.items()
    ]
    rows += [
        [
            "Readable images",
            "; ".join("%s: %d" % (cam, checks[cam]["decoded_count"]) for cam in cameras),
        ],
        [
            "Complete camera pairs",
            "%d; incomplete %d; ambiguous %d"
            % (p["complete_groups"], len(p["incomplete_ids"]), len(p["ambiguous_ids"])),
        ],
        [
            "Both images labelled with the same pattern",
            "%d / %d pairs" % (p["same_host_pattern_pairs"], p["complete_groups"]),
        ],
        ["File / image-information errors", str(sum(len(c["errors"]) for c in checks.values()))],
        ["Database check", a["database_integrity"]],
    ]
    y = _table(fig, 0.812, ["Completeness", "Result"], rows, [0.54, 0.46])
    rows = [
        ["Time between received images: median (s)"]
        + [number(checks[c]["frame_interval_s"].get("median"), 3) for c in cameras],
        ["Time between received images: longest (s)"]
        + [number(checks[c]["frame_interval_s"].get("max"), 3) for c in cameras],
        ["Skipped camera image numbers"]
        + [
            str(checks[c]["unobserved_source_counters"]) if checks[c]["decoded_count"] else "-"
            for c in cameras
        ],
        ["Average image brightness: lowest to highest"]
        + [
            number(stats[c]["mean_dn"].get("min")) + " to " + number(stats[c]["mean_dn"].get("max"))
            for c in cameras
        ],
        ["Brightest pixel (camera counts)"]
        + [number(stats[c]["frame_peak_dn"].get("max"), 0) for c in cameras],
        ["Average brightness: first-to-last change"]
        + [percentage(stats[c].get("mean_first_to_last_percent"), signed=True) for c in cameras],
        ["Pixels with a value of 4095 or higher"]
        + [
            percentage(stats[c].get("stored_pixel_fraction_ge_4095"), fraction=True)
            for c in cameras
        ],
    ]
    y = _table(
        fig,
        y,
        ["Measurement"] + cameras,
        rows,
        [0.55] + [0.45 / max(len(cameras), 1)] * len(cameras),
    )
    _text(
        fig,
        0.075,
        y,
        "Difference between paired image-ready times: median %s; largest %s microseconds.\nConfigured pairing limit: %s microseconds."
        % (
            number(p["span_us"].get("median")),
            number(p["span_us"].get("max")),
            number(p["tolerance_us"]),
        ),
    )
    y -= 0.057
    flags = list(a["failures"]) + review_flags(a, frames)
    _text(
        fig, 0.075, y, "Items to review" if flags else "Review rules: no flags", 11, weight="bold"
    )
    y -= 0.026
    displayed = 0
    for flag in flags:
        wrapped = textwrap.fill("- " + flag, width=112, subsequent_indent="  ")
        line_height = 0.020 * len(wrapped.splitlines())
        if y - line_height < 0.145:
            break
        _text(fig, 0.075, y, wrapped, 8.7)
        y -= line_height
        displayed += 1
    if len(flags) > displayed:
        _text(
            fig,
            0.075,
            y,
            "%d further items are recorded in results.npz." % (len(flags) - displayed),
            8.7,
        )
        y -= 0.020
    _text(
        fig,
        0.075,
        max(0.10, y - 0.010),
        "Review rules: gaps >3 times the median, exact repeats, or flat images.\nThese do not establish lost files or faults. Brightness changes may follow DMD patterns.",
        8,
    )
    _text(
        fig,
        0.075,
        0.070,
        "cam1: MMF output. cam2: Chromox screen. Values are saved, cropped/averaged camera counts.\nRecorded pairing does not verify simultaneous exposure or the quality of MMF reconstruction.",
        8,
    )
    _text(
        fig,
        0.075,
        0.030,
        "%d source-file hashes checked; measurements %s."
        % (
            len(a["source_manifest"]),
            "reused from cache" if cache_used else "computed from images",
        ),
        7.5,
    )
    return fig


def _panel(ax, letter, title):
    ax.set_title("(%s) %s" % (letter, title), loc="left", pad=7)
    ax.grid(axis="y", color="#E3E3E3", linewidth=0.4)
    ax.set_axisbelow(True)


def overview_figure(audit, frames, pairs, products):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.8))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.16, top=0.94, wspace=0.40, hspace=0.72)
    upper = max(1, max((float(p["maximum"].max()) for p in products.values()), default=1))
    edges = np.linspace(0, upper, min(256, max(2, int(upper) + 1)) + 1)
    for i, cam in enumerate(audit["cameras"]):
        rows = sorted(
            (f for f in frames if f["camera"] == cam and f.get("time_s") is not None),
            key=lambda f: f["time_s"],
        )
        color, style = COLORS[i % len(COLORS)], ("-", "--", "-.", ":")[i % 4]
        for ax, metric in ((axes[0, 0], "mean"), (axes[0, 1], "max")):
            ax.plot(
                [f["time_s"] for f in rows],
                [f[metric] for f in rows],
                color=color,
                linestyle=style,
                label=cam,
            )
        hist = products[cam]["histogram"]
        if hist.sum():
            counts, _ = np.histogram(np.arange(len(hist)), bins=edges, weights=hist)
            percent = counts.astype(float) * 100 / hist.sum()
            percent[percent == 0] = np.nan
            axes[1, 0].stairs(percent, edges, color=color, linestyle=style, label=cam)
    _panel(axes[0, 0], "a", "Average brightness per image")
    _panel(axes[0, 1], "b", "Brightest pixel per image")
    for ax in axes[0]:
        ax.set_xlabel("Seconds since the pattern\nsequence started")
    axes[0, 0].set_ylabel("Average pixel value\n(camera counts)")
    axes[0, 1].set_ylabel("Highest pixel value\n(camera counts)")
    axes[0, 0].legend(loc="best")
    _panel(axes[1, 0], "c", "How common each brightness is")
    axes[1, 0].set(
        xlabel="Pixel value (camera counts)",
        ylabel="Pixels in each value range (%)",
        yscale="log",
        xlim=(0, upper),
    )
    axes[1, 0].yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    axes[1, 0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: "%g" % value))
    axes[1, 0].yaxis.set_minor_formatter(NullFormatter())
    axes[1, 0].minorticks_off()
    axes[1, 0].legend(loc="best")
    _panel(axes[1, 1], "d", "Difference between recorded times")
    axes[1, 1].plot(
        [p["sample_id"] for p in pairs],
        [p["span_us"] for p in pairs],
        color="#333333",
        linewidth=0.7,
        marker="o",
        markersize=2,
    )
    axes[1, 1].set(
        xlabel="Image-pair number", ylabel="Time difference\n(microseconds)", ylim=(0, None)
    )
    fig.text(
        0.11,
        0.055,
        "Time uses image receipt at the computer. All pixels, including background, enter (c); its vertical scale is log.\nPanel (d) uses the image-ready times used for pairing. It does not establish simultaneous exposure.",
        fontsize=7.3,
        ha="left",
        va="top",
        linespacing=1.4,
    )
    return fig


def maximum_figure(audit, products):
    cameras = list(audit["cameras"])
    rows = len(cameras)
    fig = plt.figure(figsize=(7.0, 2.70 * rows + 1.30))
    grid = fig.add_gridspec(
        rows + 1,
        2,
        height_ratios=[1] * rows + [0.055],
        left=0.11,
        right=0.97,
        bottom=0.17,
        top=0.92,
        wspace=0.28,
        hspace=0.44,
    )
    maximum = max((float(p["maximum"].max()) for p in products.values()), default=1)
    norms = [Normalize(0, max(maximum, 1)), LogNorm(1, max(maximum, 1.000001))]
    cmap = plt.colormaps["inferno"].copy()
    cmap.set_bad("black")
    images = []
    for row, cam in enumerate(cameras):
        array, count = products[cam]["maximum"], audit["cameras"][cam]["decoded_count"]
        for col in range(2):
            ax = fig.add_subplot(grid[row, col])
            artist = ax.imshow(
                np.ma.masked_less_equal(array, 0) if col else array,
                cmap=cmap,
                norm=norms[col],
                interpolation="none",
                origin="upper",
            )
            if not count:
                ax.text(
                    0.5,
                    0.5,
                    "No readable images",
                    transform=ax.transAxes,
                    ha="center",
                    color="white",
                )
            ax.set_xlabel("Horizontal pixel position")
            if col == 0:
                ax.set_ylabel("Vertical pixel position")
            ax.set_title(
                "(%s) %s - %d images" % (chr(97 + row * 2 + col), cam, count), loc="left", pad=5
            )
            if row == 0:
                images.append(artist)
    for col, title in enumerate(("Linear color scale", "Log color scale")):
        bar = fig.colorbar(
            images[col], cax=fig.add_subplot(grid[rows, col]), orientation="horizontal"
        )
        bar.set_label("Pixel value (camera counts)", fontsize=8, labelpad=5)
        if col:
            ticks = [t for t in (1, 10, 100, 1000, 10000) if t <= max(maximum, 1)]
            bar.set_ticks(ticks)
            bar.set_ticklabels([str(t) for t in ticks])
        bar.minorticks_off()
        fig.text((0.28, 0.73)[col], 0.962, title, fontsize=10, ha="center", va="top")
    fig.text(
        0.11,
        0.055,
        "Each pixel is the brightest value at that position across all saved images. The cameras share color limits.\ncam1: MMF output. cam2: Chromox screen. Log colors reveal dim detail; no background removal or smoothing.",
        fontsize=7.3,
        ha="left",
        va="top",
        linespacing=1.4,
    )
    return fig


def write_report(out, audit, frames, pairs, products, *, cache_used=False):
    """One three-page PDF, embedded Arial and vector plot elements."""
    require_arial()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.pdf"
    with (
        plt.rc_context(STYLE),
        PdfPages(
            path,
            metadata={
                "Title": audit["run"] + " dataset check",
                "Author": "OptiDAQ",
                "Subject": "Fixed dataset checks and measured image statistics",
                "CreationDate": None,
                "ModDate": None,
            },
        ) as pdf,
    ):
        for make in (
            lambda: summary_figure(audit, frames, cache_used=cache_used),
            lambda: overview_figure(audit, frames, pairs, products),
            lambda: maximum_figure(audit, products),
        ):
            fig = make()
            try:
                pdf.savefig(fig, dpi=300)
            finally:
                plt.close(fig)
    return path
