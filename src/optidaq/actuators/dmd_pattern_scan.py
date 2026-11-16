"""DMD pattern-scan actuator for OptiDAQ.

Each plan point shows the next pattern from the configured stream mix and holds
it while the experiment script collects frames. Pattern sources and transforms
live in ``optidaq.data.pattern_stream``; ``display_transforms`` map a pattern
to the DMD frame; the device driver in ``optidaq.extensions.vialux_alp``
displays that frame unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
import itertools
import time
from typing import Any, Callable, Optional

import numpy as np

from optidaq.data.pattern_stream import SOURCES, PatternItem, build_pattern_stream
from optidaq.data.transforms import build_pipeline
from optidaq.extensions.vialux_alp import AlpDmd


PLACEHOLDER_PREFIX = "<FILL"


@dataclass(frozen=True)
class DmdPoint:
    index: int

    def to_dict(self) -> dict:
        return {"index": self.index}


def make_dmd_plan(dmd_cfg: Mapping[str, Any]) -> list[DmdPoint]:
    count = int(dmd_cfg["count"])
    if count <= 0:
        raise RuntimeError("dmd.count must be positive")
    return [DmdPoint(index=index) for index in range(count)]


def _placeholders(value: Any, path: str) -> list[str]:
    if isinstance(value, str) and value.startswith(PLACEHOLDER_PREFIX):
        return [path]
    if isinstance(value, Mapping):
        items = value.items()
    elif isinstance(value, list):
        items = enumerate(value)
    else:
        return []
    found = []
    for key, child in items:
        found += _placeholders(child, "%s.%s" % (path, key))
    return found


def validate_dmd_config(dmd_cfg: Mapping[str, Any]) -> None:
    """Check the dmd block without touching pattern files or hardware."""

    unfilled = _placeholders(dmd_cfg, "dmd")
    if unfilled:
        raise RuntimeError("fill the dmd config placeholders: %s" % ", ".join(unfilled))
    make_dmd_plan(dmd_cfg)
    streams = dmd_cfg.get("streams") or []
    if not streams:
        raise RuntimeError("dmd.streams must list at least one stream")
    for spec in streams:
        if spec.get("name") not in SOURCES:
            raise RuntimeError(
                "unknown dmd stream source %r (known: %s)"
                % (spec.get("name"), ", ".join(SOURCES))
            )
        build_pipeline(spec.get("transforms"))
    build_pipeline(dmd_cfg.get("display_transforms"))


class DmdPatternScanController:
    mode = "dmd"

    def __init__(self, client: Any, cfg: Mapping[str, Any]) -> None:
        self.client = client
        self.cfg = cfg
        self.dmd_cfg = cfg["dmd"]
        self.scan_cfg = cfg["scan"]
        validate_dmd_config(self.dmd_cfg)
        self.plan = make_dmd_plan(self.dmd_cfg)
        self.frames_per_point = int(self.scan_cfg["frames_per_point"])
        self.settle_s = float(self.dmd_cfg.get("settle_s", 0.0))
        self.stream_names = [str(spec["name"]) for spec in self.dmd_cfg["streams"]]
        self.display: Optional[Callable] = None
        self.device = AlpDmd(**dict(self.dmd_cfg.get("device", {}) or {}))
        self.stream: Optional[Iterator[PatternItem]] = None

    def preflight(self) -> None:
        stream = build_pattern_stream(
            self.dmd_cfg["streams"], seed=self.dmd_cfg.get("seed")
        )
        self.device.open()
        try:
            # Native dimensions belong to the opened device, not a guessed
            # config value. Explicit dimensions remain supported.
            display_specs = deepcopy(self.dmd_cfg.get("display_transforms") or [])
            for spec in display_specs:
                if isinstance(spec, Mapping) and spec.get("name") == "fit_center":
                    params = spec.setdefault("params", {})
                    for key, size in (("height", self.device.height), ("width", self.device.width)):
                        if params.get(key) is None:
                            params[key] = size
            self.display = build_pipeline(display_specs)
            first = next(stream)
            self.device.check_frame(self.display(first[2]))
        except BaseException:
            self.device.close()
            raise
        self.stream = itertools.chain([first], stream)
        print(
            "actuator=dmd points=%d frames_per_point=%d dmd=%dx%d streams=%s "
            "settle_s=%.3f"
            % (
                len(self.plan),
                self.frames_per_point,
                self.device.height,
                self.device.width,
                ",".join(self.stream_names),
                self.settle_s,
            )
        )

    def prepare(self) -> None:
        return None

    def apply_point(self, point: DmdPoint) -> tuple[dict, np.ndarray]:
        """Show the next pattern; return its log entry and the pattern before display."""

        if self.stream is None or self.display is None:
            raise RuntimeError("DMD preflight must run before apply")
        stream_index, source_id, pattern = next(self.stream)
        self.device.show(self.display(pattern))
        if self.settle_s > 0:
            time.sleep(self.settle_s)
        info = {
            "dmd_stream": stream_index,
            "dmd_source": self.stream_names[stream_index],
            "dmd_source_id": source_id,
        }
        return info, pattern

    def apply(self, point: DmdPoint) -> None:
        self.apply_point(point)

    def dwell(self) -> None:
        return None

    def abort(self) -> None:
        self.device.halt()

    def cleanup(self) -> None:
        self.device.close()

    def actuator_metadata(self) -> dict:
        return {
            "mode": self.mode,
            "points": len(self.plan),
            "device": self.device.metadata(),
        }

    def motor_metadata(self) -> dict:
        return {}

    def magnet_metadata(self) -> dict:
        return {}
