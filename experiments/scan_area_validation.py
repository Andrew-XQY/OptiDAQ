"""Display the maximum intensity projection of a folder of PNG images.

    python experiments/scan_area_validation.py /path/to/png_dir

Requires NumPy and matplotlib. Images must have the same dimensions.
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
    "dataset_dir": "/path/to/png_folder",  # path to the PNG folder (CLI arg overrides this if given)
    "cmap": "viridis",    # any matplotlib colormap name (e.g. "gray", "inferno")
    "scale": "linear",    # "linear" or "log" -> display/colorbar scaling
    "save_dir": "",       # empty -> save into dataset_dir's parent folder; else this dir
}


def find_pngs(dataset_dir: Path) -> list[Path]:
    """Return sorted top-level .png files (case-insensitive)."""
    return sorted(
        p for p in dataset_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    )


def progress(done: int, total: int, width: int = 40) -> None:
    """Minimal in-place text progress bar (stdlib only)."""
    frac = done / total
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\r[{bar}] {done}/{total} ({frac:5.1%})")
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")


def to_intensity(img: np.ndarray) -> np.ndarray:
    """Reduce a loaded image to a 2D float intensity array."""
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 3:                     # (H, W, 3/4) -> drop alpha, average RGB
        img = img[..., :3].mean(axis=2)
    return img


def maximum_intensity_projection(files: list[Path]) -> np.ndarray:
    """Pixel-wise maximum across all images (running max, low memory)."""
    mip: np.ndarray | None = None
    total = len(files)
    for i, path in enumerate(files, start=1):
        frame = to_intensity(plt.imread(path))
        if mip is None:
            mip = frame.copy()
        elif frame.shape != mip.shape:
            raise ValueError(
                f"Dimension mismatch: {path.name} is {frame.shape}, "
                f"expected {mip.shape}"
            )
        else:
            np.maximum(mip, frame, out=mip)
        progress(i, total)
    return mip


def show_projection(mip: np.ndarray, n_frames: int, dataset_dir: Path) -> None:
    """Plot the MIP, saving a timestamped copy first."""
    fig, ax = plt.subplots(figsize=(8, 6))

    norm = None
    if CONFIG["scale"] == "log":
        positive = mip[mip > 0]
        if positive.size:
            norm = LogNorm(vmin=positive.min(), vmax=mip.max())
        else:
            print("All-zero projection; using linear scale.")

    im = ax.imshow(mip, cmap=CONFIG["cmap"], norm=norm,
                   origin="upper", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="intensity")
    ax.set_title(f"MIP over {n_frames} PNG frames ({CONFIG['scale']} scale)")
    fig.tight_layout()

    out_dir = Path(CONFIG["save_dir"]) if CONFIG["save_dir"] else dataset_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mip_{datetime.now():%Y%m%d_%H%M%S}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved projection to {out_path}")

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Maximum intensity projection over a folder of PNGs."
    )
    parser.add_argument("dataset_dir", type=Path, nargs="?", default=None,
                        help="Directory of PNGs (overrides CONFIG['dataset_dir']).")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir or (
        Path(CONFIG["dataset_dir"]) if CONFIG["dataset_dir"] else None
    )
    if dataset_dir is None:
        sys.exit("No dataset dir: set CONFIG['dataset_dir'] or pass one on the CLI.")
    if not dataset_dir.is_dir():
        sys.exit(f"Not a directory: {dataset_dir}")

    files = find_pngs(dataset_dir)
    if not files:
        sys.exit(f"No .png files found in {dataset_dir}")

    print(f"Projecting {len(files)} PNG frames from {dataset_dir} ...")
    mip = maximum_intensity_projection(files)
    show_projection(mip, len(files), dataset_dir)


if __name__ == "__main__":
    main()
