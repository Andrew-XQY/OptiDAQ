#!/usr/bin/env python3
"""Loop through a centered 8-row by 8-column grid on the ViALUX DMD.

On the DMD PC, run from the OptiDAQ repository root:

    python tests/dmd_show_sliding_blocks.py

Each frame lights one 32x32 square of native mirrors at 255 on black.
The square steps by 32 mirrors in a serpentine path through the 64 positions.
Use --interval to set the hold time in seconds. Ctrl-C stops and releases the DMD.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import math
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from optidaq.extensions.vialux_alp import AlpDmd  # noqa: E402


GRID_ROWS = 8
GRID_COLS = 8
BLOCK_SIZE = 32
GRID_HEIGHT = GRID_ROWS * BLOCK_SIZE
GRID_WIDTH = GRID_COLS * BLOCK_SIZE


def block_frames(height: int, width: int) -> Iterator[np.ndarray]:
    """Yield a full native frame for each square, starting at the upper left."""
    if height < GRID_HEIGHT or width < GRID_WIDTH:
        raise ValueError(
            f"The {GRID_WIDTH}x{GRID_HEIGHT} grid does not fit the {width}x{height} DMD."
        )
    top = (height - GRID_HEIGHT) // 2
    left = (width - GRID_WIDTH) // 2

    for row in range(GRID_ROWS):
        columns = range(GRID_COLS)
        if row % 2:
            columns = range(GRID_COLS - 1, -1, -1)
        y = top + row * BLOCK_SIZE
        for col in columns:
            x = left + col * BLOCK_SIZE
            frame = np.zeros((height, width), dtype=np.uint8)
            frame[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE] = 255
            yield frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--interval", type=float, default=0.5, help="seconds to hold each block (default: %(default)s)")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be positive and finite")

    device = AlpDmd(lib_dir="/usr/lib64/")
    try:
        device.open()
        print(
            f"DMD {device.width}x{device.height} | {GRID_ROWS}x{GRID_COLS} grid | "
            f"{BLOCK_SIZE}x{BLOCK_SIZE} mirrors at 255 | "
            f"{args.interval:g} s per block | Ctrl-C to stop",
            flush=True,
        )
        while True:
            for frame in block_frames(device.height, device.width):
                device.show(frame)
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        device.close()


if __name__ == "__main__":
    main()
