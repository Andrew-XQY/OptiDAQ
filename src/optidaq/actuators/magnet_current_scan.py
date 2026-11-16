"""Magnet current-scan actuator for OptiDAQ.

This module owns magnet-side behavior only: current-point planning, preflight,
current setting, settle checks, restore, and metadata. DAQ synchronization and
frame waiting are orchestrated by the experiment script.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from typing import Any, Optional

from optidaq.extensions.cern_pyda import ClearMagnetSet, jsonable


@dataclass(frozen=True)
class MagnetPoint:
    index: int
    currents: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "currents": dict(self.currents),
        }


def magnet_reference(magnets_cfg: MappingABC[str, Any]):
    """Parse the optional magnets.reference {alias: current} preset, or None.

    Values are clamped to the hard limit at the set chokepoint, so this only
    validates aliases here. Returns a dict alias->float or None if not configured.
    """
    ref = magnets_cfg.get("reference")
    if ref is None:
        return None
    if not isinstance(ref, MappingABC) or not ref:
        raise RuntimeError("magnets.reference must be a non-empty object")
    devices = magnets_cfg.get("devices", {})
    out: dict[str, float] = {}
    for alias, value in ref.items():
        alias = str(alias)
        if alias.startswith("_"):
            continue  # skip _comment and other annotations
        if alias not in devices:
            raise RuntimeError("magnets.reference unknown alias %r" % alias)
        out[alias] = float(value)
    if not out:
        raise RuntimeError("magnets.reference has no magnet entries")
    return out


def make_magnet_current_plan(magnets_cfg: MappingABC[str, Any]) -> list[MagnetPoint]:
    devices = magnets_cfg.get("devices", {})
    if not isinstance(devices, MappingABC) or not devices:
        raise RuntimeError("magnets.devices must be a non-empty object")
    aliases = [str(alias) for alias in devices]

    scan_cfg = magnets_cfg.get("scan", {})
    raw_points = None
    if isinstance(scan_cfg, MappingABC):
        raw_points = scan_cfg.get("points")
    if raw_points is None:
        raw_points = magnets_cfg.get("points")
    if not isinstance(raw_points, list) or not raw_points:
        raise RuntimeError("magnets.scan.points must be a non-empty list")

    plan = []
    for index, raw in enumerate(raw_points):
        if not isinstance(raw, MappingABC):
            raise RuntimeError("magnets.scan.points[%d] must be an object" % index)
        currents_raw = raw.get("currents", raw)
        if not isinstance(currents_raw, MappingABC):
            raise RuntimeError(
                "magnets.scan.points[%d].currents must be an object" % index
            )
        unknown = sorted(
            key
            for key in currents_raw
            if str(key) not in aliases and not str(key).startswith("_")
        )
        if unknown:
            raise RuntimeError(
                "magnets.scan.points[%d] has unknown magnet alias(es): %s"
                % (index, ", ".join(unknown))
            )
        missing = [alias for alias in aliases if alias not in currents_raw]
        if missing:
            raise RuntimeError(
                "magnets.scan.points[%d] missing current(s) for: %s"
                % (index, ", ".join(missing))
            )
        plan.append(
            MagnetPoint(
                index=index,
                currents={alias: float(currents_raw[alias]) for alias in aliases},
            )
        )
    return plan


class MagnetCurrentScanController:
    mode = "magnet_current_scan"

    def __init__(self, client: Any, cfg: MappingABC[str, Any]) -> None:
        self.client = client
        self.cfg = cfg
        self.magnets_cfg = cfg["magnets"]
        self.scan_cfg = cfg["scan"]
        self.plan = make_magnet_current_plan(self.magnets_cfg)
        self.frames_per_point = int(self.scan_cfg["frames_per_point"])
        self.magnet_set: Optional[ClearMagnetSet] = None
        self.magnet_readback: dict[str, dict[str, Any]] = {}
        self.reference = magnet_reference(self.magnets_cfg)

    def preflight(self) -> None:
        self.magnet_set = ClearMagnetSet(self.client, self.magnets_cfg)
        self.magnet_readback = self.magnet_set.preflight()
        print(
            "actuator=magnet_current_scan points=%d frames_per_point=%d "
            "selector=%s magnets=%s"
            % (
                len(self.plan),
                self.frames_per_point,
                self.magnet_set.selector,
                ",".join(self.magnet_set.aliases),
            )
        )
        for alias in self.magnet_set.aliases:
            readback = self.magnet_readback[alias]
            print(
                "magnet %s device=%s currentAverage=%.6g current_status=%s"
                % (
                    alias,
                    readback["device"],
                    readback["currentAverage"],
                    readback["current_status"],
                )
            )
        for point in self.plan[:5]:
            print("magnet point %03d currents=%s" % (point.index, point.currents))
        if len(self.plan) > 5:
            print("magnet point preview truncated at 5/%d" % len(self.plan))

    def prepare(self) -> None:
        # Before the scan: drive magnets to the reference point (like the motor
        # move-to-reference). apply_currents waits for settle.
        if self.reference and self.magnet_set is not None:
            print("prepare: move magnets to reference %s" % self.reference)
            self.magnet_set.apply_currents(self.reference)

    def apply(self, point: MagnetPoint) -> dict[str, float]:
        return self.apply_point(point)

    def apply_point(self, point: MagnetPoint) -> dict[str, float]:
        if self.magnet_set is None:
            raise RuntimeError("magnet controller is not ready")
        print("magnet point %03d currents=%s" % (point.index, point.currents))
        return self.magnet_set.apply_currents(point.currents)

    def dwell(self) -> None:
        return None

    def abort(self) -> None:
        if self.magnet_set is not None:
            self.magnet_set.stop()

    def cleanup(self) -> None:
        # After the scan: return to the reference point if one is configured,
        # else restore the currents measured at preflight.
        if self.magnet_set is None:
            return
        if self.reference:
            print("end: move magnets to reference %s" % self.reference)
            self.magnet_set.apply_currents(self.reference)
        elif bool(self.magnets_cfg.get("restore_at_end", True)):
            print("end: restore magnet currents measured at preflight")
            self.magnet_set.restore_start_currents()

    def actuator_metadata(self) -> dict:
        return {"mode": self.mode}

    def motor_metadata(self) -> dict:
        return {}

    def magnet_metadata(self) -> dict:
        if self.magnet_set is None:
            return {}
        return {
            "devices": self.magnet_set.devices,
            "selector": self.magnet_set.selector,
            "preflight_readback": jsonable(self.magnet_readback),
        }
