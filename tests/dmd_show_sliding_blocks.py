#!/usr/bin/env python3
"""Loop through a centered 8-row by 8-column grid on the ViALUX DMD.

On the DMD PC, run from the OptiDAQ repository root:

    python tests/dmd_show_sliding_blocks.py

The grid uses the full footprint and 45-degree rotation of diamond_cross
in experiments/dmd_control.py. It lights one tile at 255 on black in a
serpentine loop through all 64 positions.
Use --interval to set the hold time in seconds. Ctrl-C stops and releases the DMD.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from optidaq.data.transforms import fit_center, rotate  # noqa: E402
from optidaq.extensions.vialux_alp import AlpDmd  # noqa: E402


GRID_ROWS = 8
GRID_COLS = 8


def diamond_block_map(height: int, width: int) -> np.ndarray:
    """Label tiles in display order, with zero outside the full diamond."""
    # Same centered square and rotation as dmd_control.py's diamond_cross.
    side = int(min(height, width) / np.sqrt(2))
    if side < max(GRID_ROWS, GRID_COLS):
        raise ValueError(f"The grid does not fit the {width}x{height} DMD.")

    row_edges = np.linspace(0, side, GRID_ROWS + 1, dtype=int)
    col_edges = np.linspace(0, side, GRID_COLS + 1, dtype=int)
    square = np.zeros((side, side), dtype=np.uint8)
    label = 1
    for row in range(GRID_ROWS):
        columns = range(GRID_COLS)
        if row % 2:
            columns = range(GRID_COLS - 1, -1, -1)
        for col in columns:
            square[row_edges[row]:row_edges[row + 1], col_edges[col]:col_edges[col + 1]] = label
            label += 1

    return rotate(fit_center(square, height=height, width=width), degrees=45)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--interval", type=float, default=0.5, help="seconds to hold each block (default: %(default)s)")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be positive and finite")

    device = AlpDmd(lib_dir="/usr/lib64/")
    try:
        device.open()
        blocks = diamond_block_map(device.height, device.width)
        print(
            f"DMD {device.width}x{device.height} | {GRID_ROWS}x{GRID_COLS} grid across full diamond | "
            "intensity 255 | "
            f"{args.interval:g} s per block | Ctrl-C to stop",
            flush=True,
        )
        while True:
            for label in range(1, GRID_ROWS * GRID_COLS + 1):
                device.show((blocks == label).astype(np.uint8) * 255)
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        device.close()


if __name__ == "__main__":
    main()
