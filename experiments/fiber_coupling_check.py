"""Plot input-beam centroids against output-camera peak and total intensity.

Run from the repository root:

    python experiments/fiber_coupling_check.py INPUT_DIR OUTPUT_DIR [OUTPUT_DIR2]

Files are sorted by name and paired by rank, so the folders must already be
matched. Each output camera gets one pair of scatter plots. NumPy and
matplotlib are required. Matplotlib scales 8-bit PNGs to [0, 1]; these metrics
therefore use that scale rather than raw camera counts.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ---------------------------------------------------------------------------
# Control parameters
# ---------------------------------------------------------------------------
CONFIG = {
    "fiber_input_dir": None,     # one PNG folder -> centroid (x, y) positions
    "fiber_output_dirs": [],     # list of 1-2 PNG folders -> max & total pixel value
    "cmap": "viridis",           # matplotlib colormap for the scatter colors
    "scale": "linear",           # "linear" or "log" -> scatter color/colorbar scaling
    "point_size": 8,            # scatter marker size (matplotlib 's')
    "save_dir": "/opt/data/CLEAR2026/output/",            # None -> just show; else save figure(s) here
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_pngs(folder: Path) -> list[Path]:
    """Sorted (rank-by-name) top-level .png files, case-insensitive."""
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    )


def to_intensity(img: np.ndarray) -> np.ndarray:
    """Reduce a loaded image to a 2D float intensity array."""
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 3:            # (H, W, 3/4) -> drop alpha, average RGB
        img = img[..., :3].mean(axis=2)
    return img


def load_intensity(path: Path) -> np.ndarray:
    """Read a PNG as a 2D float intensity array (matplotlib, no Pillow)."""
    return to_intensity(plt.imread(path))


def weighted_centroid(img: np.ndarray) -> tuple[float, float]:
    """Intensity-weighted centroid (x=col, y=row). Image center if all-zero."""
    total = img.sum()
    if total <= 0:
        h, w = img.shape
        return (w - 1) / 2.0, (h - 1) / 2.0
    ys, xs = np.indices(img.shape)
    cx = float((xs * img).sum() / total)
    cy = float((ys * img).sum() / total)
    return cx, cy


def progress(label: str, done: int, total: int, width: int = 20) -> None:
    """Print one line per ~10% step (and at completion).

    Uses plain newlines, not carriage-return redraws, so it never floods
    terminals/logs that don't honor in-place overwrite (the acc-py run-in-env
    stdout among them). A 2000+ image folder yields ~11 lines, not thousands.
    """
    if total <= 0:
        return
    crossed_decile = done * 10 // total != (done - 1) * 10 // total
    if crossed_decile or done == total:
        frac = done / total
        bar = "#" * int(width * frac) + "-" * (width - int(width * frac))
        print(f"{label}: [{bar}] {done}/{total} ({frac:4.0%})", flush=True)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def input_centroids(
    files: list[Path],
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Centroid (x, y) per input image; also return the input image (H, W)."""
    xs: list[float] = []
    ys: list[float] = []
    shape: tuple[int, int] | None = None
    n = len(files)
    for i, path in enumerate(files, start=1):
        img = load_intensity(path)
        if shape is None:
            shape = img.shape
        cx, cy = weighted_centroid(img)
        xs.append(cx)
        ys.append(cy)
        progress("Input centroids", i, n)
    return np.array(xs), np.array(ys), shape


def output_metrics(files: list[Path], label: str) -> tuple[np.ndarray, np.ndarray]:
    """Max and total (sum) pixel value per output image."""
    maxima: list[float] = []
    totals: list[float] = []
    n = len(files)
    for i, path in enumerate(files, start=1):
        img = load_intensity(path)
        maxima.append(float(img.max()))
        totals.append(float(img.sum()))
        progress(label, i, n)
    return np.array(maxima), np.array(totals)


def color_norm(values: np.ndarray):
    """LogNorm when CONFIG['scale']=='log' (positive values), else linear."""
    if CONFIG["scale"] == "log":
        positive = values[values > 0]
        if positive.size:
            return LogNorm(vmin=positive.min(), vmax=values.max())
        print("All non-positive values; falling back to linear scale.")
    return None


def make_figure(cx, cy, maxima, totals, shape, out_name, save_path):
    """Two scatter subplots (max, total) over the input pixel plane."""
    h, w = shape
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    panels = [("Max pixel value", maxima), ("Total pixel value", totals)]
    for ax, (title, values) in zip(axes, panels):
        sc = ax.scatter(cx, cy, c=values, cmap=CONFIG["cmap"],
                        s=CONFIG["point_size"], norm=color_norm(values))
        fig.colorbar(sc, ax=ax, label=f"{title} ({CONFIG['scale']})")
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.invert_yaxis()                 # image convention: origin top-left
        ax.set_aspect("equal")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.set_title(title)
    fig.suptitle(f"Fiber coupling - output '{out_name}'  ({len(cx)} samples)")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Fiber coupling check.")
    parser.add_argument("input_dir", type=Path, nargs="?", default=None,
                        help="Input PNG folder (overrides CONFIG).")
    parser.add_argument("output_dirs", type=Path, nargs="*",
                        help="1-2 output PNG folders (override CONFIG).")
    args = parser.parse_args()

    input_dir = args.input_dir or (
        Path(CONFIG["fiber_input_dir"]) if CONFIG["fiber_input_dir"] else None)
    output_dirs = (args.output_dirs
                   or [Path(d) for d in CONFIG["fiber_output_dirs"]])

    if input_dir is None:
        sys.exit("No input dir: set CONFIG['fiber_input_dir'] or pass it on the CLI.")
    if not input_dir.is_dir():
        sys.exit(f"Not a directory: {input_dir}")
    if not output_dirs:
        sys.exit("No output dirs: set CONFIG['fiber_output_dirs'] or pass on the CLI.")
    if len(output_dirs) > 2:
        sys.exit("At most two output folders are supported.")
    for d in output_dirs:
        if not d.is_dir():
            sys.exit(f"Not a directory: {d}")

    in_files = find_pngs(input_dir)
    if not in_files:
        sys.exit(f"No .png files in {input_dir}")
    out_files = [find_pngs(d) for d in output_dirs]
    for d, files in zip(output_dirs, out_files):
        if not files:
            sys.exit(f"No .png files in {d}")

    # Match by rank: clip every folder to the shared sample count.
    counts = [len(in_files)] + [len(f) for f in out_files]
    n = min(counts)
    if len(set(counts)) > 1:
        print(f"Warning: PNG counts differ {counts}; using first {n} of each.")
    in_files = in_files[:n]
    out_files = [f[:n] for f in out_files]

    cx, cy, shape = input_centroids(in_files)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(CONFIG["save_dir"]) if CONFIG["save_dir"] else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for d, files in zip(output_dirs, out_files):
        maxima, totals = output_metrics(files, f"Output {d.name}")
        save_path = (save_dir / f"fiber_coupling_{d.name}_{stamp}.png"
                     if save_dir else None)
        make_figure(cx, cy, maxima, totals, shape, d.name, save_path)

    plt.show()


if __name__ == "__main__":
    main()
