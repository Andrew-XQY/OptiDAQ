"""No-op actuator for camera-only or externally controlled OptiDAQ runs.

This module owns no hardware. It only provides a plan shape so the DAQ
orchestrator can collect frames while no actuator command is issued here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NoopPoint:
    index: int

    def to_dict(self) -> dict:
        return {"index": self.index}


def _actuator_cfg(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    actuator = cfg.get("actuator", {})
    if isinstance(actuator, Mapping):
        return actuator
    return {"mode": str(actuator)}


def make_noop_plan(cfg: Mapping[str, Any]) -> list[NoopPoint]:
    actuator = _actuator_cfg(cfg)
    raw_points = actuator.get(
        "points",
        actuator.get("n_points", actuator.get("points_total", 1)),
    )
    points_total = int(raw_points)
    if points_total <= 0:
        raise RuntimeError("actuator.points must be positive for noop mode")
    return [NoopPoint(index=index) for index in range(points_total)]


class NoopActuatorController:
    mode = "noop"

    def __init__(self, client: Any, cfg: Mapping[str, Any]) -> None:
        self.client = client
        self.cfg = cfg
        self.actuator_cfg = _actuator_cfg(cfg)
        self.scan_cfg = cfg["scan"]
        self.plan = make_noop_plan(cfg)
        self.frames_per_point = int(self.scan_cfg["frames_per_point"])

    def preflight(self) -> None:
        print(
            "actuator=noop points=%d frames_per_point=%d no hardware command"
            % (len(self.plan), self.frames_per_point)
        )

    def prepare(self) -> None:
        return None

    def apply(self, point: NoopPoint) -> None:
        print("noop point %03d" % point.index)

    def dwell(self) -> None:
        return None

    def abort(self) -> None:
        return None

    def cleanup(self) -> None:
        return None

    def actuator_metadata(self) -> dict:
        return {
            "mode": self.mode,
            "points": len(self.plan),
        }

    def motor_metadata(self) -> dict:
        return {}

    def magnet_metadata(self) -> dict:
        return {}
