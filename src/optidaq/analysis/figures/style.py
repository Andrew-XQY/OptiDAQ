"""Look of the figures, set from the config.

Style block (top-level "style" for all figures, "style" in a figure block to
override it for that figure); every key is optional:

    font       font family (default Arial)
    font_size  base size in pt (default 8); the other sizes follow it
    cmap       colour map of intensity images (default viridis)
    cmap_diff  colour map of signed images (default RdBu_r)
    rc         any matplotlib rcParams, applied last

The defaults follow the APS (Physical Review) author guidelines: figures at
final size, thin black axes, ticks inward on all four sides, no grid, no legend
frames, text kept as text in the PDF.
"""

from __future__ import annotations

from typing import Mapping

from optidaq.analysis.config import fill

SINGLE_COLUMN, DOUBLE_COLUMN = 3.375, 7.0     # APS figure widths, inches
DEFAULTS = {"font": "Arial", "font_size": 8, "cmap": "viridis", "cmap_diff": "RdBu_r", "rc": {}}


def resolve(*styles: Mapping) -> dict:
    """The style blocks merged on top of the defaults; a later block overrides
    an earlier one. Unknown keys are errors."""

    out = dict(DEFAULTS)
    for style in styles:
        style = style or {}
        fill(style, DEFAULTS, "style")
        out = {**out, **style, "rc": {**out["rc"], **style.get("rc", {})}}
    return out


def rc(style: Mapping) -> dict:
    """matplotlib rcParams of a resolved style, for matplotlib.rc_context."""

    font, size = style["font"], style["font_size"]
    return {
        "font.family": "sans-serif", "font.sans-serif": [font, "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": size, "axes.labelsize": size, "axes.titlesize": size,
        "xtick.labelsize": size - 1, "ytick.labelsize": size - 1, "legend.fontsize": size - 1,
        "mathtext.fontset": "custom", "mathtext.rm": font, "mathtext.it": font + ":italic",
        "mathtext.bf": font + ":bold",
        "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3.5,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0, "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "legend.frameon": False, "legend.handlelength": 1.6, "axes.grid": False,
        "image.cmap": style["cmap"], "image.interpolation": "nearest",
        "figure.dpi": 150, "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.titlepad": 3.0, "axes.labelpad": 2.0,
        **style["rc"],
    }
