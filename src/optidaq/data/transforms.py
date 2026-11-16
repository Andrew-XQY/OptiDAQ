"""Chainable numpy image transforms for OptiDAQ (training / derived datasets).

Unified contract — every transform is::

    fn(image: np.ndarray, context=None, **params) -> np.ndarray

so a list of them composes into a factory pipeline (output of one is the input of
the next). The exact same signature is the image-hook contract, so any transform
here can also be dropped into ``hooks.image`` in the export config. The DMD
pattern streams use the same registry (``to_uint8``, ``fit_center``, ``upscale``,
``rotate``).

IMPORTANT: these are *lossy* with respect to the raw sensor data (crop drops
pixels, bin averages them, normalize casts to float in [0, 1]). They are meant for
the runtime dataloader or a *derived* training set — NOT for the lossless
source-of-truth PNG export. In particular ``normalize`` returns float32, which the
PNG sink rejects by design.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence, Union

import numpy as np


def center_crop(image: np.ndarray, context: Any = None, *, size: int) -> np.ndarray:
    """Crop the centered ``size`` x ``size`` square (e.g. 1216 -> 1024)."""

    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError("center_crop expects a 2D image, got %s" % (arr.shape,))
    size = int(size)
    height, width = arr.shape
    if size > height or size > width:
        raise ValueError(
            "center_crop size %d exceeds image %s" % (size, arr.shape)
        )
    top = (height - size) // 2
    left = (width - size) // 2
    return arr[top:top + size, left:left + size]


def crop_box(
    image: np.ndarray,
    context: Any = None,
    *,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> np.ndarray:
    """Crop an explicit rectangle by corners: top-left (x1, y1), bottom-right (x2, y2).

    Coordinates are in the input image's pixel frame (x = column, y = row) and may
    be fractional; they are rounded and clamped to the image bounds. The slice is
    half-open ``[x1, x2)`` x ``[y1, y2)``, so width = round(x2) - round(x1).
    """

    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError("crop_box expects a 2D image, got %s" % (arr.shape,))
    height, width = arr.shape
    xa, xb = sorted((int(round(x1)), int(round(x2))))
    ya, yb = sorted((int(round(y1)), int(round(y2))))
    xa, xb = max(0, xa), min(width, xb)
    ya, yb = max(0, ya), min(height, yb)
    if xb <= xa or yb <= ya:
        raise ValueError(
            "crop_box is empty/outside image %s: x[%s,%s] y[%s,%s]"
            % (arr.shape, x1, x2, y1, y2)
        )
    return arr[ya:yb, xa:xb]


def avg_bin(
    image: np.ndarray,
    context: Any = None,
    *,
    factor: int,
    dtype: Any = None,
) -> np.ndarray:
    """Average-bin by an integer ``factor`` (e.g. 1024 /4 -> 256).

    Averaging is the anti-alias + decimation step; it preserves mean intensity and
    raises SNR. Image dims must be divisible by ``factor`` (crop first if not).

    ``dtype`` controls the output: ``None`` -> float32 (default, for the runtime
    pipeline). Pass an integer dtype (e.g. ``"uint16"``) to round to nearest and
    cast, so the result can be written as a derived PNG. Integer output drops the
    sub-count fractional part — a derived set, not the lossless master.
    """

    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError("avg_bin expects a 2D image, got %s" % (arr.shape,))
    factor = int(factor)
    if factor < 1:
        raise ValueError("avg_bin factor must be >= 1")
    if factor == 1:
        binned = arr.astype(np.float64, copy=False)
    else:
        height, width = arr.shape
        if height % factor or width % factor:
            raise ValueError(
                "avg_bin: image %s not divisible by factor %d (center_crop first)"
                % (arr.shape, factor)
            )
        blocks = arr.reshape(height // factor, factor, width // factor, factor)
        binned = blocks.astype(np.float64).mean(axis=(1, 3))

    if dtype is None:
        return binned.astype(np.float32)
    out_dtype = np.dtype(dtype)
    if np.issubdtype(out_dtype, np.integer):
        info = np.iinfo(out_dtype)
        binned = np.clip(np.rint(binned), info.min, info.max)
    return binned.astype(out_dtype)


def normalize(
    image: np.ndarray,
    context: Any = None,
    *,
    scale: float = 4095.0,
    clip: bool = True,
) -> np.ndarray:
    """Scale to [0, 1] by a FIXED sensor scale (default mono12 max = 4095).

    Fixed-scale (not per-image) keeps source and target on the same absolute
    intensity scale — required for supervised image-to-image reconstruction.
    """

    arr = np.asarray(image, dtype=np.float32)
    out = arr / np.float32(scale)
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out


def to_uint8(
    image: np.ndarray,
    context: Any = None,
    *,
    max_value: float = 255.0,
) -> np.ndarray:
    """Map ``[0, max_value]`` linearly to uint8 ``[0, 255]``, rounding and clipping."""

    arr = np.asarray(image)
    if arr.dtype == np.uint8 and float(max_value) == 255.0:
        return arr
    out = arr.astype(np.float64) * (255.0 / float(max_value))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def fit_center(
    image: np.ndarray,
    context: Any = None,
    *,
    height: int,
    width: int,
    fill: float = 0,
) -> np.ndarray:
    """Center-pad and/or center-crop to exactly ``height`` x ``width`` (no scaling)."""

    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError("fit_center expects a 2D image, got %s" % (arr.shape,))
    out = np.full((int(height), int(width)), fill, dtype=arr.dtype)
    rows = min(arr.shape[0], out.shape[0])
    cols = min(arr.shape[1], out.shape[1])
    src_top = (arr.shape[0] - rows) // 2
    src_left = (arr.shape[1] - cols) // 2
    dst_top = (out.shape[0] - rows) // 2
    dst_left = (out.shape[1] - cols) // 2
    out[dst_top:dst_top + rows, dst_left:dst_left + cols] = arr[
        src_top:src_top + rows, src_left:src_left + cols
    ]
    return out


def upscale(image: np.ndarray, context: Any = None, *, factor: int) -> np.ndarray:
    """Repeat each pixel into a ``factor`` x ``factor`` block (exact inverse of binning)."""

    factor = int(factor)
    if factor < 1:
        raise ValueError("upscale factor must be >= 1")
    arr = np.asarray(image)
    return arr.repeat(factor, axis=0).repeat(factor, axis=1)


def rotate(
    image: np.ndarray,
    context: Any = None,
    *,
    degrees: float,
    fill: float = 0,
) -> np.ndarray:
    """Rotate counter-clockwise (as displayed) about the center, keeping the size.

    Nearest-neighbour, so pixel values are copied, never blended. Used to undo a
    DMD mounted at 45 deg so patterns appear upright in the lab frame.
    """

    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError("rotate expects a 2D image, got %s" % (arr.shape,))
    height, width = arr.shape
    theta = np.deg2rad(float(degrees))
    rows, cols = np.indices((height, width), dtype=np.float64)
    dy, dx = rows - (height - 1) / 2.0, cols - (width - 1) / 2.0
    src_x = np.rint(np.cos(theta) * dx - np.sin(theta) * dy + (width - 1) / 2.0)
    src_y = np.rint(np.sin(theta) * dx + np.cos(theta) * dy + (height - 1) / 2.0)
    inside = (src_x >= 0) & (src_x < width) & (src_y >= 0) & (src_y < height)
    out = np.full(arr.shape, fill, dtype=arr.dtype)
    out[inside] = arr[src_y[inside].astype(np.intp), src_x[inside].astype(np.intp)]
    return out


def flip_vertical(image: np.ndarray, context: Any = None) -> np.ndarray:
    """Reverse image rows, preserving values; used after DMD lab rotation."""
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError("flip_vertical expects a 2D image, got %s" % (arr.shape,))
    return np.ascontiguousarray(arr[::-1])


_TRANSFORMS: dict[str, Callable[..., np.ndarray]] = {
    "center_crop": center_crop,
    "crop_box": crop_box,
    "avg_bin": avg_bin,
    "normalize": normalize,
    "to_uint8": to_uint8,
    "fit_center": fit_center,
    "upscale": upscale,
    "rotate": rotate,
    "flip_vertical": flip_vertical,
}


def build_pipeline(
    specs: Optional[Sequence[Union[str, Mapping[str, Any]]]],
) -> Callable[[np.ndarray, Any], np.ndarray]:
    """Compose a config list into one callable ``(image, context=None) -> image``.

    Each spec is a name string or ``{"name": ..., "params": {...}}``. The returned
    callable applies them in order — the factory pipeline.
    """

    steps: list[tuple[Callable[..., np.ndarray], dict[str, Any]]] = []
    for spec in specs or []:
        if isinstance(spec, str):
            name, params = spec, {}
        elif isinstance(spec, Mapping):
            name = str(spec.get("name", ""))
            params = dict(spec.get("params", {}) or {})
        else:
            raise ValueError("transform spec must be a string or object")
        if name not in _TRANSFORMS:
            raise ValueError("unknown transform %r" % name)
        steps.append((_TRANSFORMS[name], params))

    def run(image: np.ndarray, context: Any = None) -> np.ndarray:
        for fn, params in steps:
            image = fn(image, context, **params)
        return image

    return run
