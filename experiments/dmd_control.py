"""Manual DMD check without cameras or file writing.

Run from the repository root on the PC with the DMD:

    python experiments/dmd_control.py --config configs/experiment_lab_v1.json --test white
    python experiments/dmd_control.py --config configs/experiment_lab_v1.json --interval 1

``--test`` shows a full-frame white or checkerboard frame, or ``diamond``: the
largest full-intensity square rotated 45 deg, which looks like an upright square
when the DMD is mounted at 45 deg; ``diamond_cross`` adds a black aiming cross
through its centre, edge to edge. It runs until Enter and needs only the
dmd.device settings. Without ``--test`` it steps through the configured dmd
stream mix, one pattern per interval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from optidaq.actuators.dmd_pattern_scan import DmdPatternScanController  # noqa: E402
from optidaq.data.transforms import fit_center, rotate  # noqa: E402
from optidaq.extensions.vialux_alp import AlpDmd  # noqa: E402


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "experiment_lab_v1.json"


def test_frame(
    kind: str, height: int, width: int, block: int = 64, cross: int = 40
) -> np.ndarray:
    if kind == "white":
        return np.full((height, width), 255, dtype=np.uint8)
    if kind in ("diamond", "diamond_cross"):
        side = int(min(height, width) / np.sqrt(2))  # its 45 deg diagonal fits the short side
        square = np.full((side, side), 255, dtype=np.uint8)
        if kind == "diamond_cross":  # drawn before rotating, so it is upright when projected
            start = (side - cross) // 2
            square[start:start + cross, :] = 0
            square[:, start:start + cross] = 0
        return rotate(fit_center(square, height=height, width=width), degrees=45)
    rows = (np.arange(height) // block)[:, None]
    cols = (np.arange(width) // block)[None, :]
    return (((rows + cols) % 2) * 255).astype(np.uint8)


def show_test(cfg: dict, kind: str) -> None:
    device = AlpDmd(**dict(cfg["dmd"].get("device", {}) or {}))
    device.open()
    try:
        print("dmd=%dx%d showing %s; press Enter to stop" % (device.height, device.width, kind))
        device.show(test_frame(kind, device.height, device.width))
        input()
    finally:
        device.close()


def step_streams(cfg: dict, interval_s: float, count: Optional[int]) -> None:
    controller = DmdPatternScanController(None, cfg)
    controller.preflight()
    try:
        for point in controller.plan[:count]:
            info, _ = controller.apply_point(point)
            print("pattern %03d %s" % (point.index, info))
            time.sleep(interval_s)
    finally:
        controller.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--test", choices=("white", "checker", "diamond", "diamond_cross"))
    parser.add_argument("--interval", type=float, default=1.0, help="seconds per pattern")
    parser.add_argument("--count", type=int, help="patterns to show (default dmd.count)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    if args.test:
        show_test(cfg, args.test)
    else:
        step_streams(cfg, args.interval, args.count)


if __name__ == "__main__":
    main()
