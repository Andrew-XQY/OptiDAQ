"""ViALUX ALP DMD driver for OptiDAQ.

The driver displays exactly the frame it receives: a 2D uint8 array with the
DMD's native height and width. It never pads, scales or thresholds; image
preparation belongs to the pattern stream transforms. Pixel values follow the
ALP MSB-aligned convention, so with ``bit_depth=1`` a mirror is on when its
value is >= 128, and with ``bit_depth=8`` values 0..255 are gray levels.

``ALP4`` (pip package ``ALP4lib``) is imported on ``open`` only, so this module
imports on machines without the ViALUX library.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


class AlpDmd:
    """One ALP device holding a single 1-frame sequence that is re-filled per image."""

    def __init__(
        self,
        *,
        version: str = "4.3",
        lib_dir: Optional[str] = None,
        serial: Optional[int] = None,
        bit_depth: int = 8,
        picture_time_us: int = 40000,
    ) -> None:
        self.version = str(version)
        self.lib_dir = lib_dir
        self.serial = None if serial is None else int(serial)
        self.bit_depth = int(bit_depth)
        self.picture_time_us = int(picture_time_us)
        self.height: Optional[int] = None
        self.width: Optional[int] = None
        self._alp: Any = None

    def open(self) -> None:
        if self._alp is not None:
            return
        from ALP4 import ALP4

        alp = ALP4(version=self.version, libDir=self.lib_dir)
        alp.Initialize(self.serial)
        try:
            alp.SeqAlloc(nbImg=1, bitDepth=self.bit_depth)
            alp.SetTiming(pictureTime=self.picture_time_us)
        except BaseException:
            alp.Free()
            raise
        self._alp = alp
        self.height = int(alp.nSizeY)
        self.width = int(alp.nSizeX)

    def check_frame(self, frame: np.ndarray) -> None:
        if self._alp is None:
            raise RuntimeError("DMD is not open")
        arr = np.asarray(frame)
        if arr.dtype != np.uint8 or arr.shape != (self.height, self.width):
            raise ValueError(
                "DMD frame must be uint8 with shape (%d, %d), got %s %s; "
                "fix it with display_transforms"
                % (self.height, self.width, arr.dtype, arr.shape)
            )

    def show(self, frame: np.ndarray) -> None:
        """Display one frame continuously until the next ``show`` or ``halt``."""

        self.check_frame(frame)
        self._alp.Halt()
        self._alp.SeqPut(imgData=np.ascontiguousarray(frame).ravel())
        self._alp.Run()

    def halt(self) -> None:
        if self._alp is not None:
            self._alp.Halt()

    def close(self) -> None:
        """Stop display and release the device; each step runs even if one fails."""

        alp, self._alp = self._alp, None
        if alp is None:
            return
        errors = []
        for step in (alp.Halt, alp.FreeSeq, alp.Free):
            try:
                step()
            except Exception as exc:
                errors.append("%s: %s" % (step.__name__, exc))
        if errors:
            raise RuntimeError("DMD close failed: %s" % "; ".join(errors))

    def metadata(self) -> dict:
        return {
            "driver": "ALP4lib",
            "version": self.version,
            "serial": self.serial,
            "height": self.height,
            "width": self.width,
            "bit_depth": self.bit_depth,
            "picture_time_us": self.picture_time_us,
        }
