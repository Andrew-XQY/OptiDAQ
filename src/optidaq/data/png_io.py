"""Shared lossless PNG writer for acquisition and offline export."""

from __future__ import annotations

from pathlib import Path
import struct
import zlib

import numpy as np


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def write_png_gray(path: Path, image: np.ndarray, *, compress_level: int = 6) -> None:
    """Write a 2D uint8/uint16 grayscale PNG without changing pixel values."""

    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError("PNG export expects a 2D grayscale image, got %s" % (arr.shape,))
    write_png(path, arr, compress_level=compress_level)


def write_png(path: Path, image: np.ndarray, *, compress_level: int = 6) -> None:
    """Write uint8/uint16 grayscale or uint8 RGB without changing values."""

    arr = np.asarray(image)
    if arr.ndim == 2:
        color_type = 0
    elif arr.ndim == 3 and arr.shape[2] == 3 and arr.dtype == np.dtype("uint8"):
        color_type = 2
    else:
        raise ValueError("PNG expects grayscale or uint8 RGB, got %s" % (arr.shape,))
    if not arr.size:
        raise ValueError("PNG image must not be empty")

    if arr.dtype == np.dtype("uint8"):
        bit_depth = 8
        png_arr = arr
    elif arr.dtype == np.dtype("uint16"):
        bit_depth = 16
        png_arr = arr.astype(">u2", copy=False)
    else:
        raise ValueError("PNG export supports only uint8 or uint16, got %s" % arr.dtype)

    level = int(compress_level)
    if level < 0 or level > 9:
        raise ValueError("PNG compress_level must be in 0..9")

    height, width = png_arr.shape[:2]
    rows = [
        b"\x00" + np.ascontiguousarray(row).tobytes(order="C")
        for row in png_arr
    ]
    raw = b"".join(rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(PNG_SIGNATURE)
        handle.write(
            _png_chunk(
                b"IHDR",
                struct.pack(
                    ">IIBBBBB",
                    int(width),
                    int(height),
                    bit_depth,
                    color_type,
                    0,  # deflate
                    0,  # adaptive filtering; rows here use filter type 0
                    0,  # no interlace
                ),
            )
        )
        handle.write(_png_chunk(b"IDAT", zlib.compress(raw, level)))
        handle.write(_png_chunk(b"IEND", b""))


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return (
        struct.pack(">I", len(data))
        + body
        + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    )
