"""Sample figure of a fit step: measured against synthetic images, per fibre and fit.

One figure per saved frame, drawn from samples.npz and frames.csv of one pipeline
step (their layout is in forward_fit/run.py). Nothing is recomputed from datasets.

One group of columns per fibre. In a group, the first column is the measurement
and every further column is one fit. Rows:

    (a) beam camera: the measured beam; for a fit, the sum of the laser-spot
        images with the fit's weights. Marks: plus = fibre axis of the group,
        small circle = centre of the measured beam, dashed circle = the points
        at that centre's distance from the axis.
    (b) fibre image, divided by the 99.9th percentile of the measured image.
        The synthetic image is first scaled by one factor: least squares
        against the measured image inside the fibre mask.
    (c) synthetic minus measured fibre image, in the units of row (b).

Numbers in the panels are read from frames.csv: beam_rel_l2 in row (a),
shape_rel_l2 in row (b). The number of fibres and fits follows the artefact.

Figure block; every key but "figure" and "source" is optional:

    figure      "fit_samples"
    source      name of the pipeline step whose artefacts are drawn
    fibres      fibre labels to draw, in this order; null: all in the artefact
    fits        fits to draw, in this order; null: all in the artefact
    samples     sample ids to draw; null: every saved frame
    labels      display text {key: text}; keys: "measured", "intensity",
                "difference", a fit name, a fibre label or a metric name
    width_in    figure width in inches; null: 3.375 for one fibre, 7 for more
    diff_limit  colour range of row (c), as a share of the measured peak
    marks       draw the marks of row (a)
    formats     file types to write, e.g. ["png", "pdf"]
    dpi         resolution of raster files
    style       overrides of the shared style (figures/style.py)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np

from optidaq.analysis.config import fill
from optidaq.analysis.figures.style import DOUBLE_COLUMN, SINGLE_COLUMN, rc, resolve

DEFAULTS = {
    "figure": "fit_samples",
    "source": None,
    "fibres": None,
    "fits": None,
    "samples": None,
    "labels": {},
    "width_in": None,
    "diff_limit": 0.08,
    "marks": True,
    "formats": ["png"],
    "dpi": 400,
    "style": {},
}
LABELS = {
    "measured": "Measured",
    "intensity": r"$I/I_{\max}$",
    "difference": r"$\Delta I/I_{\max}$ (%)",
    "beam_rel_l2": r"$\varepsilon_{\rm ref}$",
    "shape_rel_l2": r"$\varepsilon_{\rm MMF}$",
}
# Geometry in inches: left margin, right margin, gap between fit columns, gap after
# the measured column, gap between fibre groups, gap between rows, header height.
LEFT, RIGHT, GAP, SEP, GROUP, ROW, HEAD = 0.20, 0.01, 0.03, 0.07, 0.16, 0.05, 0.20


def draw(params: Mapping, *, root, out_dir, style: Mapping = None, log=print) -> list:
    """Draw the figures of one config block; returns the written paths."""

    p = fill(params, DEFAULTS, "figure fit_samples", required=("source",))
    look = resolve(style, p["style"])
    source, out_dir = Path(root) / p["source"], Path(out_dir)
    with open(source / "frames.csv", newline="") as f:
        numbers = {(r["fibre"], r["fit"], r["run_id"], r["sample_id"]): r for r in csv.DictReader(f)}
    written = []
    with np.load(source / "samples.npz") as data, matplotlib.rc_context(rc(look)):
        fibres = _subset(p["fibres"], data["fibres"], "fibre")
        fits = _subset(p["fits"], data["fits"], "fit")
        saved = [str(v) for v in data["sample_id"]]
        wanted = _subset(p["samples"], saved, "saved sample")
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, sample_id in enumerate(saved):
            if sample_id not in wanted:
                continue
            fig = _figure(data, i, fibres, fits, numbers, p, look)
            stem = ("%s_%s" % (data["run_id"][i], sample_id)).replace("/", "_").replace(" ", "_")
            for suffix in p["formats"]:
                written.append(out_dir / ("%s.%s" % (stem, suffix)))
                fig.savefig(written[-1], dpi=p["dpi"])
            plt.close(fig)
    log("%d files written: %s" % (len(written), out_dir))
    return written


def _subset(wanted, available, what: str) -> list:
    available = [str(v) for v in available]
    if wanted is None:
        return available
    wanted = [str(v) for v in wanted]
    missing = [v for v in wanted if v not in available]
    if missing or not wanted:
        raise ValueError("%s %s not in the artefact; it has %s" % (what, missing, available))
    return wanted


def _layout(width: float, groups: int, columns: int):
    """Left edge of every column, panel size, bottom edge of the three rows and
    figure height, all in inches."""

    panel = (width - LEFT - RIGHT - groups * (SEP + (columns - 2) * GAP) - (groups - 1) * GROUP) / (groups * columns)
    x = [LEFT]
    for c in range(1, groups * columns):
        x.append(x[-1] + panel + (GROUP if c % columns == 0 else SEP if c % columns == 1 else GAP))
    top = HEAD + 0.23
    height = top + 3 * panel + 2 * ROW + 0.03
    y = [height - top - panel - r * (panel + ROW) for r in range(3)]
    return x, panel, y, height


def _figure(data, i: int, fibres: list, fits: list, numbers: Mapping, p: Mapping, look: Mapping):
    labels = {**LABELS, **{name: name.replace("_", " ").capitalize() for name in fits}, **p["labels"]}
    columns = 1 + len(fits)
    width = p["width_in"] or (SINGLE_COLUMN if len(fibres) == 1 else DOUBLE_COLUMN)
    x, panel, y, height = _layout(width, len(fibres), columns)
    scale = look["font_size"] / 8.0
    small, head = scale * min(6.5, 6.5 * panel / 0.78), scale * min(6.8, 6.8 * panel / 0.70)
    limit = float(p["diff_limit"])
    fig = plt.figure(figsize=(width, height))

    def axes(left, bottom):
        ax = fig.add_axes([left / width, bottom / height, panel / width, panel / height])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        return ax

    def number(ax, row, metric, form):
        if row.get(metric) not in (None, ""):
            ax.text(0.05, 0.94, "%s = %s" % (labels.get(metric, metric), form % float(row[metric])),
                    transform=ax.transAxes, ha="left", va="top", fontsize=small, color="white")

    beam, centre = data["beam"][i], data["beam_centre"][i]
    key = (str(data["run_id"][i]), str(data["sample_id"][i]))
    signed = plt.get_cmap(look["cmap_diff"]).copy()
    signed.set_bad("white")

    for g, fibre in enumerate(fibres):
        xs = x[g * columns:(g + 1) * columns]
        mask, axis = data["mask/" + fibre].astype(bool), data["axis/" + fibre]
        real = np.maximum(data["measured/" + fibre][i].astype(np.float64), 0.0) * mask
        peak = max(float(np.percentile(real[mask], 99.9)), 1e-30)
        rows, cols = np.nonzero(mask)
        crop = (slice(max(rows.min() - 3, 0), rows.max() + 4), slice(max(cols.min() - 3, 0), cols.max() + 4))

        def beam_panel(ax, image, vmax):
            ax.imshow(image, cmap=look["cmap"], vmin=0, vmax=max(vmax, 1e-30))
            if p["marks"] and np.isfinite(axis).all():
                ax.add_patch(Circle(axis, float(np.hypot(*(centre - axis))), fill=False, ec="white", lw=0.6,
                                    ls=(0, (2.5, 1.5))))
                ax.plot(*axis, "+", color="white", ms=4.5, mew=0.8)
                ax.plot(*centre, "o", mfc="none", mec="white", ms=3.2, mew=0.7)
            ax.set_xlim(-0.5, image.shape[1] - 0.5)
            ax.set_ylim(image.shape[0] - 0.5, -0.5)

        beam_panel(axes(xs[0], y[0]), beam, float(np.percentile(beam, 99.9)))
        axes(xs[0], y[1]).imshow((real / peak)[crop], cmap=look["cmap"], vmin=0, vmax=1)
        for c, fit in enumerate(fits, 1):
            row = numbers.get((fibre, fit) + key, {})
            light = data["light/" + fit][i]
            ax = axes(xs[c], y[0])
            beam_panel(ax, light, float(light.max()))
            number(ax, row, "beam_rel_l2", "%.2f")
            synthetic = data["synthetic/%s/%s" % (fibre, fit)][i].astype(np.float64)
            gain = float((real * synthetic)[mask].sum() / max((synthetic * synthetic)[mask].sum(), 1e-300))
            synthetic = gain * synthetic * mask
            ax = axes(xs[c], y[1])
            ax.imshow((synthetic / peak)[crop], cmap=look["cmap"], vmin=0, vmax=1)
            number(ax, row, "shape_rel_l2", "%.3f")
            axes(xs[c], y[2]).imshow(np.where(mask, (synthetic - real) / peak, np.nan)[crop], cmap=signed,
                                     vmin=-limit, vmax=limit)

        for c, name in enumerate(["measured", *fits]):
            fig.text((xs[c] + panel / 2) / width, (y[0] + panel + 0.035) / height, labels[name],
                     ha="center", va="bottom", fontsize=head, linespacing=1.0)
        left, right, line = xs[0] / width, (xs[-1] + panel) / width, (y[0] + panel + HEAD) / height
        fig.add_artist(plt.Line2D([left, right], [line, line], color="black", lw=0.6))
        fig.text((left + right) / 2, line + 0.035 / height, labels.get(fibre, fibre), ha="center", va="bottom",
                 fontsize=scale * 7.5)

    for r, letter in enumerate("abc"):
        fig.text((x[0] - 0.035) / width, (y[r] + panel) / height, "(%s)" % letter, ha="right", va="top",
                 fontsize=look["font_size"])
    bars = (
        (look["cmap"], Normalize(0, 1), labels["intensity"], [0, 0.5, 1]),
        (look["cmap_diff"], Normalize(-100 * limit, 100 * limit), labels["difference"], [-100 * limit, 0, 100 * limit]),
    )
    drop = max(0.0, (panel - 0.80) / 2)       # the two colour bars sit in the empty slot of row (c)
    for j, (cmap, norm, title, ticks) in enumerate(bars):
        cax = fig.add_axes([(x[0] + 0.05) / width, (y[2] + panel - drop - 0.17 - 0.30 * j) / height,
                            (panel - 0.10) / width, 0.045 / height])
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=ticks)
        bar.outline.set_linewidth(0.5)
        cax.tick_params(labelsize=scale * 6.0, length=2, width=0.5, pad=1.2, direction="out", top=False)
        cax.minorticks_off()
        cax.set_title(title, fontsize=scale * 6.5, pad=2)
    return fig
