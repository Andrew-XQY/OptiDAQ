"""Bounded random-walk magnet actuator.

Each step is reflected into the intersection of the device range, configured
hard limits and optional soft range. The controller inherits current setting,
settling, cleanup and metadata handling from MagnetCurrentScanController.
"""

from __future__ import annotations

import random
from collections.abc import Mapping as MappingABC
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from optidaq.actuators.magnet_current_scan import (
    MagnetCurrentScanController,
    MagnetPoint,
    magnet_reference,
)
from optidaq.extensions.cern_pyda import ClearMagnetSet


def _resolve_max_step(max_step: Any, alias: str) -> float:
    if isinstance(max_step, MappingABC):
        if alias not in max_step:
            raise RuntimeError("magnets.random_walk.max_step missing entry for %r" % alias)
        return float(max_step[alias])
    if max_step is None:
        raise RuntimeError("magnets.random_walk.max_step is required")
    return float(max_step)  # a scalar applies to every active magnet


def validate_magnet_random_walk_config(magnets_cfg: Mapping[str, Any]):
    """Static (no-hardware) config checks. Returns (active, limits, n_points, seed)."""
    devices = magnets_cfg.get("devices", {})
    if not isinstance(devices, MappingABC) or not devices:
        raise RuntimeError("magnets.devices must be a non-empty object")
    aliases = [str(a) for a in devices]

    limits_cfg = magnets_cfg.get("limits", {})
    if not isinstance(limits_cfg, MappingABC) or not limits_cfg:
        raise RuntimeError("magnets.limits is required for the magnet walk")
    limits: Dict[str, Tuple[float, float]] = {}
    for alias in aliases:
        if alias not in limits_cfg:
            raise RuntimeError("magnets.limits missing entry for device %r" % alias)
        lo, hi = float(limits_cfg[alias][0]), float(limits_cfg[alias][1])
        if not lo < hi:
            raise RuntimeError("magnets.limits.%s must have min < max" % alias)
        limits[alias] = (lo, hi)

    walk = magnets_cfg.get("random_walk", {})
    if not isinstance(walk, MappingABC):
        raise RuntimeError("magnets.random_walk must be an object")
    n_points = int(walk.get("n_points", 0))
    if n_points <= 0:
        raise RuntimeError("magnets.random_walk.n_points must be positive")

    active = walk.get("active", aliases)
    if not isinstance(active, Sequence) or isinstance(active, (str, bytes)) or not active:
        raise RuntimeError("magnets.random_walk.active must be a non-empty list")
    active = [str(a) for a in active]
    unknown = [a for a in active if a not in devices]
    if unknown:
        raise RuntimeError("magnets.random_walk.active unknown alias(es): %s" % ", ".join(unknown))

    max_step = walk.get("max_step")
    ranges = walk.get("range", {}) or {}
    if not isinstance(ranges, MappingABC):
        raise RuntimeError("magnets.random_walk.range must be an object")
    for alias in active:
        if _resolve_max_step(max_step, alias) <= 0:
            raise RuntimeError("magnets.random_walk.max_step for %s must be positive" % alias)
        if alias in ranges:
            rlo, rhi = float(ranges[alias][0]), float(ranges[alias][1])
            llo, lhi = limits[alias]
            if not rlo < rhi:
                raise RuntimeError("magnets.random_walk.range.%s must have lo < hi" % alias)
            if rlo < llo or rhi > lhi:
                raise RuntimeError(
                    "magnets.random_walk.range.%s %s exceeds hard limit %s"
                    % (alias, (rlo, rhi), limits[alias])
                )

    # Optional reference must sit inside the hard limit (fail fast).
    reference = magnet_reference(magnets_cfg)
    if reference is not None:
        for alias, value in reference.items():
            lo, hi = limits[alias]
            if not lo <= value <= hi:
                raise RuntimeError(
                    "magnets.reference.%s=%.3f outside hard limit [%.3f, %.3f]"
                    % (alias, value, lo, hi)
                )
    return active, limits, n_points, int(walk.get("seed", 0))


def make_magnet_random_walk_plan(
    active: Sequence[str],
    starts: Mapping[str, float],
    ranges: Mapping[str, Tuple[float, float]],
    max_steps: Mapping[str, float],
    n_points: int,
    seed: int,
) -> List[MagnetPoint]:
    """Bounded random walk: each step jumps by <= max_step from the current value,
    reflected at the range walls so it neither exceeds the range nor sticks at an edge."""
    rng = random.Random(seed)
    cur = {a: float(starts[a]) for a in active}
    plan: List[MagnetPoint] = []
    for index in range(n_points):
        currents: Dict[str, float] = {}
        for alias in active:
            lo, hi = ranges[alias]
            nxt = cur[alias] + rng.uniform(-max_steps[alias], max_steps[alias])
            if nxt > hi:
                nxt = hi - (nxt - hi)   # reflect off the upper wall
            if nxt < lo:
                nxt = lo + (lo - nxt)   # reflect off the lower wall
            nxt = min(max(nxt, lo), hi)  # final safety clamp
            cur[alias] = nxt
            currents[alias] = round(nxt, 4)
        plan.append(MagnetPoint(index=index, currents=currents))
    return plan


class MagnetRandomWalkController(MagnetCurrentScanController):
    """Magnet actuator that walks each selected magnet by a small bounded random step."""

    mode = "magnet_random_walk"

    def __init__(self, client: Any, cfg: Mapping[str, Any]) -> None:
        self.client = client
        self.cfg = cfg
        self.magnets_cfg = cfg["magnets"]
        self.scan_cfg = cfg["scan"]
        self.walk_cfg = self.magnets_cfg.get("random_walk", {})
        # Fail fast on bad config before touching hardware.
        self._active, self._limits, self._n_points, self._seed = (
            validate_magnet_random_walk_config(self.magnets_cfg)
        )
        self.frames_per_point = int(self.scan_cfg["frames_per_point"])
        self.magnet_set = None
        self.magnet_readback: Dict[str, Dict[str, Any]] = {}
        self.reference = magnet_reference(self.magnets_cfg)
        self.plan: List[MagnetPoint] = []      # built in preflight()
        self._eff_ranges: Dict[str, Tuple[float, float]] = {}

    def preflight(self) -> None:
        self.magnet_set = ClearMagnetSet(self.client, self.magnets_cfg)
        self.magnet_readback = self.magnet_set.preflight()  # checks OK/ON/REMOTE, reads starts

        max_step_cfg = self.walk_cfg.get("max_step")
        range_cfg = self.walk_cfg.get("range", {}) or {}
        starts: Dict[str, float] = {}
        max_steps: Dict[str, float] = {}
        for alias in self._active:
            sp = self.magnet_set.setting_ppm(alias)
            flo, fhi = float(sp["current_min"]), float(sp["current_max"])
            llo, lhi = self._limits[alias]
            rlo, rhi = (
                (float(range_cfg[alias][0]), float(range_cfg[alias][1]))
                if alias in range_cfg else (llo, lhi)
            )
            lo, hi = max(flo, llo, rlo), min(fhi, lhi, rhi)
            if lo >= hi:
                raise RuntimeError(
                    "magnet %s empty effective range: FESA[%.3f,%.3f] ∩ limit[%.3f,%.3f] ∩ range[%.3f,%.3f]"
                    % (alias, flo, fhi, llo, lhi, rlo, rhi)
                )
            self._eff_ranges[alias] = (lo, hi)
            # The walk starts from the reference (prepare() drives there first) if set,
            # otherwise from the live measured current; clamped into the effective range.
            base = (
                self.reference[alias]
                if self.reference and alias in self.reference
                else float(self.magnet_set.start_currents[alias])
            )
            starts[alias] = min(max(base, lo), hi)
            max_steps[alias] = _resolve_max_step(max_step_cfg, alias)

        self.plan = make_magnet_random_walk_plan(
            self._active, starts, self._eff_ranges, max_steps, self._n_points, self._seed
        )
        print(
            "actuator=magnet_random_walk points=%d frames_per_point=%d seed=%d active=%s"
            % (len(self.plan), self.frames_per_point, self._seed, ",".join(self._active))
        )
        for alias in self._active:
            lo, hi = self._eff_ranges[alias]
            print(
                "  walk %s start=%.4f A range=[%.4f, %.4f] max_step=%.3f"
                % (alias, starts[alias], lo, hi, max_steps[alias])
            )

    def actuator_metadata(self) -> dict:
        return {
            "mode": self.mode,
            "n_points": self._n_points,
            "seed": self._seed,
            "active": list(self._active),
            "effective_ranges": {a: list(r) for a, r in self._eff_ranges.items()},
        }
