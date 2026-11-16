"""Raster (grid) magnet actuator for OptiDAQ.

A deterministic alternative to the bounded random walk: hold a chosen set of
magnets at FIXED currents (typically the three quads) and sweep one or more other
magnets (typically the dh/dv correctors) through their range step by step, visiting
every grid point of the Cartesian product.

Like ``MagnetRandomWalkController`` this subclasses ``MagnetCurrentScanController``,
so apply/apply_point, dwell, abort, cleanup, prepare (move to reference),
metadata, and the experiment dispatch are all inherited. Only the plan differs:
instead of a random jump per step, the plan enumerates a fixed grid.

Config (under ``magnets``)::

    "raster": {
      "fixed": { "qf1": 50.0, "qd": 30.0, "qf2": 0.0 },
      "axes": [
        { "magnet": "dv", "start": -8.0, "stop": 8.0, "step": 4.0 },
        { "magnet": "dh", "start": -1.0, "stop":  9.0, "step": 2.0 }
      ],
      "serpentine": false
    }

* ``fixed``  : {alias: current_A} held constant at every point.
* ``axes``   : ordered list; the FIRST axis is the OUTER (slowest) loop and the
               LAST axis is the INNER (fastest) loop. Each axis gives ``start`` and
               ``stop`` plus exactly one of ``step`` (A between points) or ``num``
               (number of points, evenly spaced inclusive of both ends). Sweeps may
               be ascending or descending (start may be > stop).
* ``serpentine`` (default ``false``): if true, inner sweeps reverse direction every
               other outer step (boustrophedon) to minimise magnet jumps. Default is
               false so every inner sweep is traversed in the SAME direction, which
               keeps magnet hysteresis consistent across the grid.

Every requested range is intersected with the FESA ``SettingPPM`` range and the
``magnets.limits`` hard cap; ``ClearMagnetSet.apply_currents`` clamps again at the
set chokepoint so a current can never exceed its hard limit. Each point's currents
include both the fixed magnets and the swept axes, so ``apply_currents`` drives the
full state and blocks until every magnet has settled before the next point.
"""

from __future__ import annotations

import math
from collections.abc import Mapping as MappingABC
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from optidaq.actuators.magnet_current_scan import (
    MagnetCurrentScanController,
    MagnetPoint,
    magnet_reference,
)
from optidaq.extensions.cern_pyda import ClearMagnetSet

_STEP_EPS = 1e-9


def axis_values(
    start: float,
    stop: float,
    step: Any = None,
    num: Any = None,
) -> List[float]:
    """Inclusive 1-D sweep from ``start`` toward ``stop``.

    Exactly one of ``step`` or ``num`` must be given. ``num`` spaces ``num`` points
    evenly across [start, stop] (both ends included). ``step`` walks by a fixed
    magnitude from ``start`` toward ``stop`` without overshooting; the final point
    equals ``stop`` only when the span is an integer multiple of ``step``.
    Descending sweeps (start > stop) are supported.
    """
    start = float(start)
    stop = float(stop)
    if (step is None) == (num is None):
        raise RuntimeError("raster axis needs exactly one of 'step' or 'num'")

    if num is not None:
        n = int(num)
        if n <= 0:
            raise RuntimeError("raster axis 'num' must be positive")
        if n == 1:
            return [round(start, 4)]
        span = stop - start
        return [round(start + span * k / (n - 1), 4) for k in range(n)]

    step = float(step)
    if step <= 0:
        raise RuntimeError("raster axis 'step' must be positive")
    span = stop - start
    direction = 1.0 if span >= 0 else -1.0
    n = int(math.floor((abs(span) + step * _STEP_EPS) / step)) + 1
    return [round(start + direction * step * k, 4) for k in range(n)]


def _normalise_axes(
    raster_cfg: Mapping[str, Any],
    devices: Mapping[str, Any],
    fixed_aliases: Sequence[str],
    raster_key: str = "raster",
) -> List[Dict[str, Any]]:
    label = "magnets.%s" % raster_key
    raw_axes = raster_cfg.get("axes")
    if not isinstance(raw_axes, Sequence) or isinstance(raw_axes, (str, bytes)) or not raw_axes:
        raise RuntimeError("%s.axes must be a non-empty list" % label)

    seen: set[str] = set()
    axes: List[Dict[str, Any]] = []
    for index, raw in enumerate(raw_axes):
        if not isinstance(raw, MappingABC):
            raise RuntimeError("%s.axes[%d] must be an object" % (label, index))
        alias = str(raw.get("magnet", raw.get("alias", "")))
        if not alias:
            raise RuntimeError("%s.axes[%d] missing 'magnet'" % (label, index))
        if alias not in devices:
            raise RuntimeError("%s.axes[%d] unknown magnet %r" % (label, index, alias))
        if alias in seen:
            raise RuntimeError("%s.axes has duplicate magnet %r" % (label, alias))
        if alias in fixed_aliases:
            raise RuntimeError("%s magnet %r is both fixed and swept" % (label, alias))
        seen.add(alias)
        if "start" not in raw or "stop" not in raw:
            raise RuntimeError("%s.axes[%d] needs 'start' and 'stop'" % (label, index))
        values = axis_values(
            raw["start"], raw["stop"], raw.get("step"), raw.get("num")
        )
        axes.append(
            {
                "magnet": alias,
                "start": float(raw["start"]),
                "stop": float(raw["stop"]),
                "step": (None if raw.get("step") is None else float(raw["step"])),
                "num": (None if raw.get("num") is None else int(raw["num"])),
                "values": values,
            }
        )
    return axes


def validate_magnet_raster_config(
    magnets_cfg: Mapping[str, Any],
    raster_key: str = "raster",
):
    """Static (no-hardware) config checks.

    Returns (fixed, axes, limits, aliases, total_points).
    """
    devices = magnets_cfg.get("devices", {})
    if not isinstance(devices, MappingABC) or not devices:
        raise RuntimeError("magnets.devices must be a non-empty object")
    aliases = [str(a) for a in devices]

    label = "magnets.%s" % raster_key
    raster_cfg = magnets_cfg.get(raster_key, {})
    if not isinstance(raster_cfg, MappingABC) or not raster_cfg:
        raise RuntimeError("%s must be a non-empty object" % label)

    # --- fixed magnets -------------------------------------------------------
    fixed_raw = raster_cfg.get("fixed", {})
    if not isinstance(fixed_raw, MappingABC):
        raise RuntimeError("%s.fixed must be an object" % label)
    fixed: Dict[str, float] = {}
    for alias, value in fixed_raw.items():
        alias = str(alias)
        if alias.startswith("_"):
            continue  # skip _comment annotations
        if alias not in devices:
            raise RuntimeError("%s.fixed unknown magnet %r" % (label, alias))
        fixed[alias] = float(value)

    # --- swept axes ----------------------------------------------------------
    axes = _normalise_axes(raster_cfg, devices, list(fixed), raster_key)

    # --- hard limits (required for every magnet we will command) -------------
    limits_cfg = magnets_cfg.get("limits", {})
    if not isinstance(limits_cfg, MappingABC) or not limits_cfg:
        raise RuntimeError("magnets.limits is required for the raster scan")
    commanded = list(fixed) + [axis["magnet"] for axis in axes]
    limits: Dict[str, Tuple[float, float]] = {}
    for alias in commanded:
        if alias not in limits_cfg:
            raise RuntimeError("magnets.limits missing entry for magnet %r" % alias)
        lo, hi = float(limits_cfg[alias][0]), float(limits_cfg[alias][1])
        if not lo <= hi:
            raise RuntimeError("magnets.limits.%s must have min <= max" % alias)
        limits[alias] = (lo, hi)

    # Fixed values must sit inside the hard limit (fail fast; no silent clamp).
    for alias, value in fixed.items():
        lo, hi = limits[alias]
        if not lo <= value <= hi:
            raise RuntimeError(
                "%s.fixed.%s=%.4f outside hard limit [%.3f, %.3f]"
                % (label, alias, value, lo, hi)
            )
    # Swept endpoints must sit inside the hard limit.
    for axis in axes:
        alias = axis["magnet"]
        lo, hi = limits[alias]
        amin, amax = min(axis["start"], axis["stop"]), max(axis["start"], axis["stop"])
        if amin < lo or amax > hi:
            raise RuntimeError(
                "%s.axes.%s span [%.3f, %.3f] exceeds hard limit [%.3f, %.3f]"
                % (label, alias, amin, amax, lo, hi)
            )

    # Optional reference must sit inside the hard limit too (matches random walk).
    reference = magnet_reference(magnets_cfg)
    if reference is not None:
        for alias, value in reference.items():
            if alias in limits:
                lo, hi = limits[alias]
            elif alias in limits_cfg:
                lo, hi = float(limits_cfg[alias][0]), float(limits_cfg[alias][1])
            else:
                continue
            if not lo <= value <= hi:
                raise RuntimeError(
                    "magnets.reference.%s=%.3f outside hard limit [%.3f, %.3f]"
                    % (alias, value, lo, hi)
                )

    total = 1
    for axis in axes:
        total *= len(axis["values"])
    return fixed, axes, limits, aliases, total


def validate_magnet_raster_tuned_config(magnets_cfg: Mapping[str, Any]):
    fixed, axes, limits, aliases, total = validate_magnet_raster_config(
        magnets_cfg,
        raster_key="raster_tuned",
    )
    if fixed:
        raise RuntimeError("magnets.raster_tuned.fixed must be empty")
    return fixed, axes, limits, aliases, total


def _grid_index_tuples(sizes: Sequence[int], serpentine: bool) -> List[Tuple[int, ...]]:
    """Odometer over axis sizes (first = outer). With serpentine, each axis after
    the first reverses whenever the sum of the outer indices is odd (boustrophedon)."""
    combos: List[Tuple[int, ...]] = [()]
    for axis_pos, n in enumerate(sizes):
        nxt: List[Tuple[int, ...]] = []
        for prefix in combos:
            order = range(n)
            if serpentine and axis_pos > 0 and (sum(prefix) % 2 == 1):
                order = range(n - 1, -1, -1)
            for i in order:
                nxt.append(prefix + (i,))
        combos = nxt
    return combos


def make_magnet_raster_plan(
    fixed: Mapping[str, float],
    axes: Sequence[Mapping[str, Any]],
    aliases: Sequence[str],
    serpentine: bool,
) -> List[MagnetPoint]:
    """Enumerate the full grid as a list of MagnetPoint.

    Each point sets the fixed magnets plus one value per swept axis. Currents are
    ordered by the device alias order for stable, readable output.
    """
    sizes = [len(axis["values"]) for axis in axes]
    fixed = {a: round(float(v), 4) for a, v in fixed.items()}
    plan: List[MagnetPoint] = []
    for index, combo in enumerate(_grid_index_tuples(sizes, serpentine)):
        currents: Dict[str, float] = dict(fixed)
        for axis, value_index in zip(axes, combo):
            currents[axis["magnet"]] = axis["values"][value_index]
        ordered = {a: currents[a] for a in aliases if a in currents}
        plan.append(MagnetPoint(index=index, currents=ordered))
    return plan


class MagnetRasterScanController(MagnetCurrentScanController):
    """Hold the fixed magnets constant and sweep the axis magnets across a grid."""

    mode = "magnet_raster_scan"
    raster_key = "raster"

    def __init__(self, client: Any, cfg: Mapping[str, Any]) -> None:
        self.client = client
        self.cfg = cfg
        self.magnets_cfg = cfg["magnets"]
        self.scan_cfg = cfg["scan"]
        self.raster_cfg = self.magnets_cfg.get(self.raster_key, {})
        # Fail fast on bad config before touching hardware.
        (
            self._fixed,
            self._axes,
            self._limits,
            self._aliases,
            self._total_points,
        ) = validate_magnet_raster_config(self.magnets_cfg, self.raster_key)
        self._serpentine = bool(self.raster_cfg.get("serpentine", False))
        self.frames_per_point = int(self.scan_cfg["frames_per_point"])
        self.magnet_set = None
        self.magnet_readback: Dict[str, Dict[str, Any]] = {}
        self.reference = magnet_reference(self.magnets_cfg)
        self.plan: List[MagnetPoint] = []  # built in preflight()
        self._eff_ranges: Dict[str, Tuple[float, float]] = {}

    def preflight(self) -> None:
        self.magnet_set = ClearMagnetSet(self.client, self._magnet_set_config())
        self.magnet_readback = self.magnet_set.preflight()  # checks OK/ON/REMOTE, reads starts

        # Confirm fixed values and swept endpoints fall inside the live FESA range,
        # intersected with the hard limit. Fail fast rather than silently clamp.
        for alias, value in self._fixed.items():
            flo, fhi = self._fesa_range(alias)
            llo, lhi = self._limits[alias]
            lo, hi = max(flo, llo), min(fhi, lhi)
            if not lo <= value <= hi:
                raise RuntimeError(
                    "fixed magnet %s=%.4f outside effective range [%.4f, %.4f] "
                    "(FESA[%.3f,%.3f] ∩ limit[%.3f,%.3f])"
                    % (alias, value, lo, hi, flo, fhi, llo, lhi)
                )

        for axis in self._axes:
            alias = axis["magnet"]
            flo, fhi = self._fesa_range(alias)
            llo, lhi = self._limits[alias]
            lo, hi = max(flo, llo), min(fhi, lhi)
            amin = min(axis["values"])
            amax = max(axis["values"])
            if lo >= hi:
                raise RuntimeError(
                    "magnet %s empty effective range: FESA[%.3f,%.3f] ∩ limit[%.3f,%.3f]"
                    % (alias, flo, fhi, llo, lhi)
                )
            if amin < lo or amax > hi:
                raise RuntimeError(
                    "raster axis %s span [%.4f, %.4f] outside effective range "
                    "[%.4f, %.4f] (FESA[%.3f,%.3f] ∩ limit[%.3f,%.3f])"
                    % (alias, amin, amax, lo, hi, flo, fhi, llo, lhi)
                )
            self._eff_ranges[alias] = (lo, hi)

        self.plan = make_magnet_raster_plan(
            self._fixed, self._axes, self._aliases, self._serpentine
        )

        held = [
            a
            for a in self._aliases
            if a not in self._fixed and a not in self._eff_ranges
        ]
        print(
            "actuator=%s points=%d frames_per_point=%d serpentine=%s"
            % (self.mode, len(self.plan), self.frames_per_point, self._serpentine)
        )
        if self._fixed:
            print(
                "  fixed: %s"
                % ", ".join("%s=%.4f A" % (a, self._fixed[a]) for a in self._fixed)
            )
        for axis in self._axes:
            alias = axis["magnet"]
            lo, hi = self._eff_ranges[alias]
            print(
                "  sweep %s n=%d values=[%.4f .. %.4f] step/num=%s eff_range=[%.4f, %.4f]"
                % (
                    alias,
                    len(axis["values"]),
                    axis["values"][0],
                    axis["values"][-1],
                    ("step=%s" % axis["step"]) if axis["step"] is not None else ("num=%s" % axis["num"]),
                    lo,
                    hi,
                )
            )
        if held:
            print("  held at start/reference (not commanded): %s" % ", ".join(held))

    def _fesa_range(self, alias: str) -> Tuple[float, float]:
        sp = self.magnet_set.setting_ppm(alias)
        return float(sp["current_min"]), float(sp["current_max"])

    def _magnet_set_config(self) -> Mapping[str, Any]:
        return self.magnets_cfg

    def _commanded_aliases(self) -> List[str]:
        aliases = list(self._fixed)
        for axis in self._axes:
            alias = axis["magnet"]
            if alias not in aliases:
                aliases.append(alias)
        return aliases

    def actuator_metadata(self) -> dict:
        return {
            "mode": self.mode,
            "raster_key": self.raster_key,
            "n_points": self._total_points,
            "serpentine": self._serpentine,
            "fixed": dict(self._fixed),
            "axes": [
                {
                    "magnet": axis["magnet"],
                    "start": axis["start"],
                    "stop": axis["stop"],
                    "step": axis["step"],
                    "num": axis["num"],
                    "n_points": len(axis["values"]),
                }
                for axis in self._axes
            ],
            "effective_ranges": {a: list(r) for a, r in self._eff_ranges.items()},
        }


class MagnetRasterScanTunedController(MagnetRasterScanController):
    """Corrector-only raster that leaves tuned quadrupoles untouched."""

    mode = "magnet_raster_scan_tuned"
    raster_key = "raster_tuned"

    def __init__(self, client: Any, cfg: Mapping[str, Any]) -> None:
        super().__init__(client, cfg)
        if self._fixed:
            raise RuntimeError("magnets.raster_tuned.fixed must be empty")
        self.reference = None

    def _magnet_set_config(self) -> Mapping[str, Any]:
        aliases = set(self._commanded_aliases())
        devices = self.magnets_cfg.get("devices", {})
        cfg = dict(self.magnets_cfg)
        cfg["devices"] = {
            alias: devices[alias]
            for alias in self._aliases
            if alias in aliases
        }
        limits = self.magnets_cfg.get("limits", {})
        if isinstance(limits, MappingABC):
            cfg["limits"] = {
                alias: limits[alias]
                for alias in self._aliases
                if alias in aliases and alias in limits
            }
        return cfg

    def prepare(self) -> None:
        return None

    def apply_point(self, point: MagnetPoint) -> dict[str, float]:
        if self.magnet_set is None:
            raise RuntimeError("magnet controller is not ready")
        print("magnet point %03d currents=%s" % (point.index, point.currents))
        return self.magnet_set.apply_currents_when_idle(point.currents)

    def cleanup(self) -> None:
        if self.magnet_set is None:
            return
        restore = self.raster_cfg.get(
            "restore_at_end",
            self.magnets_cfg.get("restore_at_end", True),
        )
        if not bool(restore):
            return
        currents = {
            alias: self.magnet_set.start_currents[alias]
            for alias in self._commanded_aliases()
            if alias in self.magnet_set.start_currents
        }
        if currents:
            print("end: restore tuned raster currents measured at preflight %s" % currents)
            self.magnet_set.apply_currents_when_idle(currents)
