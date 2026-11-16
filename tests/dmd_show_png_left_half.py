#!/usr/bin/env python3
"""Show the left half of each PNG in a folder on the ViALUX DMD.

Standalone: needs numpy, Pillow and ALP4lib, no repository imports. Run on the
PC with the DMD after editing CONFIG:

    python tests/dmd_show_png_left_half.py

Each image is cut at the middle of its width. The left half is brightened to
the configured peak, with optional gamma compression, then centered on the DMD
(optionally enlarged by an integer factor), then rotated and flipped to match
the input camera's orientation. Black pixels remain black.
Peak scaling is per image; gamma=1 preserves within-image intensity ratios,
while gamma<1 lifts dim features and changes those ratios.
Ctrl-C stops early; the DMD is always released.
"""
from pathlib import Path
import time

import numpy as np
from PIL import Image

CONFIG = {
    "png_dir": "/path/to/png_dir",
    "start": 0,              # first image in numeric file order (0.png, 1.png, ...)
    "count": 10,             # images to show; None shows all from start
    "interval_s": 1.0,       # how long each image stays on the DMD
    "upscale": 2,            # each image pixel becomes upscale x upscale mirrors
    "rotation_deg": 45.0,    # counterclockwise, around the native DMD frame center
    "flip_vertical": True,  # AFTER rotation: measured DMD-to-input-camera reflection
    "peak_value": 250,       # brightest source pixel maps here (1..255), per image
    "gamma": 0.8,            # <1 brightens dim features; 1 keeps intensity ratios
    "alp_version": "4.3",
    "alp_lib_dir": "/usr/lib64/",
    "bit_depth": 8,          # 8: gray levels; 1: mirror on when value >= 128
    "picture_time_us": 40000,
}


def left_half(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        # Preserve scientific grayscale PNG values until after normalization:
        # converting a 16-bit image to L first would clip values above 255.
        if img.mode in {"1", "L", "I", "F"} or img.mode.startswith("I;16"):
            gray = np.asarray(img, dtype=np.float64)
        else:
            gray = np.asarray(img.convert("L"), dtype=np.float64)
    return gray[:, : gray.shape[1] // 2]


def brighten_pattern(pattern: np.ndarray, peak_value: int, gamma: float) -> np.ndarray:
    if not 1 <= peak_value <= 255 or int(peak_value) != peak_value:
        raise ValueError("peak_value must be an integer from 1 to 255")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be positive and finite")
    values = np.asarray(pattern, dtype=np.float64)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("pattern must be a nonempty, finite grayscale image")
    values = np.maximum(values, 0.0)
    maximum = float(values.max())
    if maximum == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.rint(peak_value * (values / maximum) ** gamma).astype(np.uint8)


def to_dmd_frame(
    pattern: np.ndarray, height: int, width: int, upscale: int,
    *, rotation_deg: float = 0.0, flip_vertical: bool = False,
) -> np.ndarray:
    """Center on the native frame, rotate CCW, then optionally flip vertically.

    Orientation defaults retain the original centering-only API. CONFIG applies
    the measured lab correction explicitly; values use nearest-neighbour sampling.
    """
    if not np.isfinite(rotation_deg):
        raise ValueError("rotation_deg must be finite")
    if type(flip_vertical) is not bool:
        raise ValueError("flip_vertical must be a bool")
    pattern = pattern.repeat(upscale, axis=0).repeat(upscale, axis=1)
    rows, cols = min(pattern.shape[0], height), min(pattern.shape[1], width)
    src_top, src_left = (pattern.shape[0] - rows) // 2, (pattern.shape[1] - cols) // 2
    dst_top, dst_left = (height - rows) // 2, (width - cols) // 2
    frame = np.zeros((height, width), dtype=np.uint8)
    frame[dst_top:dst_top + rows, dst_left:dst_left + cols] = pattern[
        src_top:src_top + rows, src_left:src_left + cols
    ]
    if rotation_deg:
        theta = np.deg2rad(float(rotation_deg))
        rows, cols = np.indices(frame.shape, dtype=np.float64)
        dy, dx = rows - (height - 1) / 2.0, cols - (width - 1) / 2.0
        src_x = np.rint(np.cos(theta) * dx - np.sin(theta) * dy + (width - 1) / 2.0)
        src_y = np.rint(np.sin(theta) * dx + np.cos(theta) * dy + (height - 1) / 2.0)
        inside = (src_x >= 0) & (src_x < width) & (src_y >= 0) & (src_y < height)
        rotated = np.zeros_like(frame)
        rotated[inside] = frame[src_y[inside].astype(np.intp), src_x[inside].astype(np.intp)]
        frame = rotated
    if flip_vertical:
        frame = frame[::-1]
    return np.ascontiguousarray(frame)


def main() -> None:
    from ALP4 import ALP4

    files = sorted(Path(CONFIG["png_dir"]).glob("*.png"), key=lambda p: (len(p.stem), p.stem))
    stop = None if CONFIG["count"] is None else CONFIG["start"] + CONFIG["count"]
    files = files[CONFIG["start"]:stop]
    if not files:
        raise SystemExit("no PNG files selected in %s" % CONFIG["png_dir"])

    dmd = ALP4(version=CONFIG["alp_version"], libDir=CONFIG["alp_lib_dir"])
    dmd.Initialize()
    try:
        dmd.SeqAlloc(nbImg=1, bitDepth=CONFIG["bit_depth"])
        dmd.SetTiming(pictureTime=CONFIG["picture_time_us"])
        print("DMD %dx%d, showing %d image(s)" % (dmd.nSizeY, dmd.nSizeX, len(files)))
        print("Per-image peak=%d, gamma=%.3g" % (CONFIG["peak_value"], CONFIG["gamma"]))
        for path in files:
            source = left_half(path)
            pattern = brighten_pattern(source, CONFIG["peak_value"], CONFIG["gamma"])
            frame = to_dmd_frame(pattern, dmd.nSizeY, dmd.nSizeX, CONFIG["upscale"],
                                 rotation_deg=CONFIG["rotation_deg"],
                                 flip_vertical=CONFIG["flip_vertical"])
            dmd.Halt()
            dmd.SeqPut(imgData=frame.ravel())
            dmd.Run()
            print("%s  left half %s  source max %.6g -> DMD max %d" % (
                path.name, pattern.shape, source.max(), pattern.max()
            ))
            time.sleep(CONFIG["interval_s"])
    except KeyboardInterrupt:
        print("stopped")
    finally:
        for step in (dmd.Halt, dmd.FreeSeq, dmd.Free):
            try:
                step()
            except Exception as exc:
                print("WARNING: %s failed: %s" % (step.__name__, exc))


if __name__ == "__main__":
    main()
