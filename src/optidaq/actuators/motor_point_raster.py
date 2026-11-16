"""Lab-only, stationary-point motor raster with read-back coordinates.

The origin is the live counter value at preflight. This controller never homes,
resets counters, changes motion settings or modifies interlocks. Raw-counter
operation is explicit; the saved validity flag is not relabelled as homed.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from fractions import Fraction
import math
import time
from typing import Any, Callable, Optional


AXES = ("x", "y")


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or (positive and value <= 0):
        raise ValueError("%s must be %s" % (name, "a positive integer" if positive else "an integer"))
    return value


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a finite number" % name)
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError("%s must be finite and %s" % (name, "positive" if positive else "nonnegative"))
    return result


def _axis_offsets(scan: Mapping[str, Any], name: str) -> list[int]:
    count = scan["n" + name]
    if "span_counts" in scan:
        span = scan["span_counts"][name]
        # Exact rational rounding keeps opposite points symmetric and includes
        # both endpoints when the requested interval is not an integer count.
        return [round(Fraction(span * (2 * index - count + 1), 2 * (count - 1)))
                for index in range(count)]
    return [scan["start_offset_counts"][name] + index * scan["stride_counts"][name]
            for index in range(count)]


def validate_motor_raster_config(cfg: Mapping[str, Any]) -> None:
    motors, scan = cfg.get("motors"), cfg.get("scan")
    if not isinstance(motors, Mapping) or not isinstance(scan, Mapping):
        raise ValueError("motor raster requires motors and scan objects")
    if motors.get("position_mode", "valid_position") not in ("raw_counter", "valid_position"):
        raise ValueError("motors.position_mode must be raw_counter or valid_position")
    if type(motors.get("return_to_start", True)) is not bool:
        raise ValueError("motors.return_to_start must be boolean")
    for key in ("nx", "ny"):
        _integer(scan.get(key), "scan.%s" % key, positive=True)
    if type(scan.get("frames_per_point", 1)) is not int or scan.get("frames_per_point", 1) != 1:
        raise ValueError("motor raster saves one paired sample per point; frames_per_point must be 1")
    if type(scan.get("serpentine", True)) is not bool:
        raise ValueError("scan.serpentine must be boolean")
    reference = motors.get("reference_counters")
    if "reference_counters" in motors:
        if not isinstance(reference, Mapping):
            raise ValueError("motors.reference_counters requires x and y")
        for name in AXES:
            value = _integer(reference.get(name), "motors.reference_counters.%s" % name)
            if not -(2**31) <= value < 2**31:
                raise ValueError("motors.reference_counters.%s must fit int32" % name)
    stride, start = scan.get("stride_counts"), scan.get("start_offset_counts")
    span = scan.get("span_counts")
    if "span_counts" in scan:
        if "stride_counts" in scan or "start_offset_counts" in scan:
            raise ValueError("scan.span_counts cannot be combined with stride_counts or start_offset_counts")
        if not isinstance(span, Mapping):
            raise ValueError("scan.span_counts requires x and y")
    elif not isinstance(stride, Mapping) or not isinstance(start, Mapping):
        raise ValueError("scan.stride_counts and scan.start_offset_counts require x and y")
    for name in AXES:
        axis = motors.get(name)
        if not isinstance(axis, Mapping) or not isinstance(axis.get("device"), str) or not axis["device"].strip():
            raise ValueError("motors.%s requires a device" % name)
        if axis.get("enabled", True) is not True:
            raise ValueError("both motors must be enabled for a raster")
        if span is not None:
            width = _integer(span.get(name), "scan.span_counts.%s" % name, positive=True)
            if width % 2 or scan["n" + name] < 2:
                raise ValueError("span_counts requires an even span and at least two points per axis")
            offsets = _axis_offsets(scan, name)
            step = min(right - left for left, right in zip(offsets, offsets[1:]))
        else:
            step = _integer(stride.get(name), "scan.stride_counts.%s" % name, positive=True)
            _integer(start.get(name), "scan.start_offset_counts.%s" % name)
        tolerance = _number(axis.get("tol_counts"), "motors.%s.tol_counts" % name)
        if step <= 2 * tolerance:
            raise ValueError("%s stride must exceed twice its counter tolerance" % name)
        for key in ("timeout_s", "poll_s"):
            _number(axis.get(key), "motors.%s.%s" % (name, key), positive=True)
        _number(axis.get("settle_s"), "motors.%s.settle_s" % name)
    if motors["x"]["device"] == motors["y"]["device"]:
        raise ValueError("x and y must use different motor devices")


@dataclass(frozen=True)
class MotorRasterPoint:
    index: int
    row: int
    column: int
    x_offset_counts: int
    y_offset_counts: int

    def to_dict(self) -> dict:
        return asdict(self)


def make_motor_raster_plan(cfg: Mapping[str, Any]) -> list[MotorRasterPoint]:
    validate_motor_raster_config(cfg)
    scan = cfg["scan"]
    x_offsets, y_offsets = (_axis_offsets(scan, name) for name in AXES)
    points = []
    for row in range(scan["ny"]):
        columns = range(scan["nx"])
        if scan.get("serpentine", True) and row % 2:
            columns = reversed(columns)
        for column in columns:
            points.append(MotorRasterPoint(
                len(points), row, column,
                x_offsets[column], y_offsets[row],
            ))
    return points


def _plain(value: Any) -> Any:
    """Preserve nested readbacks, including NumPy/PyDA scalar values."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if hasattr(value, "value"):
        value = value.value
    if getattr(value, "shape", None) == () and hasattr(value, "_value"):
        value = value._value
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if hasattr(value, "item"):
        value = value.item()
    return value


class MotorPointRasterController:
    mode = "motor_raster"
    frames_per_point = 1

    def __init__(
        self, client: Any, cfg: Mapping[str, Any],
        should_stop: Callable[[], bool] = lambda: False,
        *, axis_factory: Optional[Callable[..., Any]] = None, clock: Any = time,
    ) -> None:
        validate_motor_raster_config(cfg)
        self.client, self.cfg, self.should_stop = client, cfg, should_stop
        self._axis_factory = axis_factory
        self._clock = clock
        self.plan = make_motor_raster_plan(cfg)
        self.position_mode = str(cfg["motors"].get("position_mode", "valid_position"))
        self.axes: dict[str, Any] = {}
        self.initial_counters: dict[str, int] = {}
        self.reference_counters: dict[str, int] = {}
        self.origin = ("configured_reference_counters" if "reference_counters" in cfg["motors"]
                       else "live_preflight_counters")
        self.initial_readbacks: dict[str, dict] = {}
        self.final_readbacks: dict[str, dict] = {}
        self._raw_acquisition: dict[str, Callable] = {}
        self._last_readbacks: dict[str, dict] = {}
        self._commanded_axes: set[str] = set()
        self._abort_attempts: list[dict] = []
        self._current_point: Optional[MotorRasterPoint] = None
        self._ready = False
        self._cleaned = False
        self.restored = False

    def _check_stop(self) -> None:
        if self.should_stop():
            raise KeyboardInterrupt("motor raster stop requested")

    @staticmethod
    def _counter(axis: Any, values: Mapping) -> int:
        value = float(axis._counter(values))
        if not math.isfinite(value) or not value.is_integer() or not -(2**31) <= value < 2**31:
            raise RuntimeError("%s motor counter is not a valid int32 value" % axis.name)
        return int(value)

    def _read_checked(self, name: str) -> dict:
        self._check_stop()
        axis = self.axes[name]
        acq = _plain(self._raw_acquisition[name]())
        driver = _plain(axis.driver_acquisition())
        status = acq.get("status_channel")
        if type(status) is not int or status < 0 or status & 1:
            raise RuntimeError("%s motor channel status is invalid or reports a hardware problem" % name)
        if driver.get("ps_power_state") is not True or driver.get("ps_error") != 0:
            raise RuntimeError("%s motor driver is not healthy" % name)
        if type(acq.get("motor_busy")) is not bool or type(acq.get("position_valid")) is not bool:
            raise RuntimeError("%s motor busy/position validity readback is missing" % name)
        if self.position_mode == "valid_position" and not acq["position_valid"]:
            raise RuntimeError("%s motor position is not valid" % name)
        counter = self._counter(axis, acq)
        position = float(axis._position(acq))
        expected = float(axis.position_for_counter(counter))
        tolerance = abs(axis.units_scale) * axis.tol_counts
        if not math.isfinite(position) or not math.isfinite(expected) or abs(position - expected) > tolerance:
            raise RuntimeError("%s motor position disagrees with raw counter calibration" % name)
        self._last_readbacks[name] = {
            "device": axis.device, "position_counter": counter, "position": position,
            "position_valid": acq["position_valid"], "units": axis.units,
            "units_scale": axis.units_scale, "units_offset": axis.units_offset,
            "acquisition": acq, "driver": driver,
        }
        return acq

    def _read_idle(self, targets: Optional[Mapping[str, int]] = None) -> dict:
        for name, axis in self.axes.items():
            acq = axis.acquisition()
            if acq["motor_busy"]:
                raise RuntimeError("%s motor is still moving" % name)
            actual = self._counter(axis, acq)
            if targets is not None and abs(actual - targets[name]) > axis.tol_counts:
                raise RuntimeError("%s motor readback %d differs from target %d" % (name, actual, targets[name]))
        return deepcopy(self._last_readbacks)

    def preflight(self) -> None:
        if self.axes:
            raise RuntimeError("motor raster preflight may only run once")
        factory = self._axis_factory
        if factory is None:
            from optidaq.extensions.cern_pyda import MotorAxis
            factory = MotorAxis
        for name in AXES:
            self._check_stop()
            axis = factory(name, self.client, self.cfg["motors"][name],
                           require_valid_position=self.position_mode == "valid_position")
            self.axes[name] = axis
            calibration = (axis.units_scale, axis.units_offset, axis.position_min, axis.position_max)
            if not all(math.isfinite(float(value)) for value in calibration) or axis.units_scale == 0 or axis.position_min > axis.position_max:
                raise RuntimeError("%s motor has invalid calibration/limits" % name)
            self._raw_acquisition[name] = axis.acquisition
            axis.acquisition = lambda name=name: self._read_checked(name)
            axis.preflight()
            initial = self._counter(axis, axis.acquisition())
            self.initial_counters[name] = initial
            reference = self.cfg["motors"].get("reference_counters", {}).get(name, initial)
            self.reference_counters[name] = reference
            targets = [reference + getattr(point, name + "_offset_counts") for point in self.plan]
            if any(not -(2**31) <= target < 2**31 for target in targets):
                raise RuntimeError("%s raster exceeds the int32 motor counter range" % name)
            axis.range_check_absolute_counters(targets + [initial])
        self.initial_readbacks = self._read_idle(self.initial_counters)
        self._ready = True

    def prepare(self) -> None:
        if not self._ready:
            raise RuntimeError("motor raster preflight must complete before prepare")
        self._read_idle(self.initial_counters)

    def _targets(self, point: MotorRasterPoint) -> dict[str, int]:
        return {name: self.reference_counters[name] + getattr(point, name + "_offset_counts") for name in AXES}

    def _move(self, targets: Mapping[str, int]) -> dict:
        try:
            self._read_idle()
            for name, axis in self.axes.items():
                self._check_stop()
                actual = self._counter(axis, axis.acquisition())
                if abs(actual - targets[name]) > axis.tol_counts:
                    self._commanded_axes.add(name)
                axis.move_abs_raw(targets[name])
            return self._read_idle(targets)
        except BaseException:
            self.abort()
            raise

    def apply_point(self, point: MotorRasterPoint) -> dict:
        if not self._ready or self._cleaned:
            raise RuntimeError("motor raster is not ready")
        if not isinstance(point, MotorRasterPoint) or not 0 <= point.index < len(self.plan) or point != self.plan[point.index]:
            raise ValueError("point does not belong to this raster plan")
        started = time.time_ns()
        targets = self._targets(point)
        readbacks = self._move(targets)
        self._current_point = point
        return {
            "mode": self.mode, **point.to_dict(), "pattern_id": "position_%06d" % point.index,
            "initial_counters": dict(self.initial_counters), "target_counters": targets,
            "reference_counters": dict(self.reference_counters), "origin": self.origin,
            "readbacks": readbacks, "motor_position_mode": self.position_mode,
            "homing_performed": False, "change_begin_ns": started,
            "ready_host_time_ns": time.time_ns(),
            "readiness_method": "idle_counter_within_tolerance_calibration_consistent_and_settled",
        }

    def verify_held(self, point: MotorRasterPoint) -> dict:
        if point != self._current_point:
            raise RuntimeError("cannot verify a motor point that is not currently held")
        return self._read_idle(self._targets(point))

    def abort(self) -> bool:
        """Stop our moving axes and confirm idle without the cancellation hook.

        The shared adapter logs failed Stop writes, so a call returning is not
        proof of stopping. Bound the confirmation polling, retain its outcome,
        and keep trying the other axes even when one cannot be confirmed.
        """
        confirmed = True
        for name in AXES:
            if name not in self._commanded_axes:
                continue
            axis = self.axes[name]
            started = self._clock.monotonic()
            timeout = min(3.0, axis.timeout_s)
            result = {"axis": name, "device": axis.device, "stop_requested": False,
                      "idle_confirmed": False, "timeout_s": timeout}

            def read_busy():
                busy = _plain(self._raw_acquisition[name]()).get("motor_busy")
                if type(busy) is not bool:
                    raise RuntimeError("motor_busy is missing or invalid")
                result["last_motor_busy"] = busy
                return busy

            try:
                result["idle_confirmed"] = not read_busy()
            except BaseException as exc:
                result["initial_readback_error"] = "%s: %s" % (type(exc).__name__, exc)
            if not result["idle_confirmed"]:
                result["stop_requested"] = True
                try:
                    axis.stop()
                except BaseException as exc:
                    result["stop_error"] = "%s: %s" % (type(exc).__name__, exc)
                deadline = started + timeout
                while True:
                    try:
                        if not read_busy():
                            result["idle_confirmed"] = True
                            break
                    except BaseException as exc:
                        result["readback_error"] = "%s: %s" % (type(exc).__name__, exc)
                    remaining = deadline - self._clock.monotonic()
                    if remaining <= 0:
                        result["confirmation_error"] = "STOP not confirmed before timeout"
                        break
                    self._clock.sleep(min(axis.poll_s, remaining))
            result["elapsed_s"] = self._clock.monotonic() - started
            self._abort_attempts.append(result)
            confirmed = confirmed and result["idle_confirmed"]
        return confirmed

    def cleanup(self, success: bool = True) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        if not success or self.should_stop() or not self._ready:
            if not self.abort():
                raise RuntimeError("Motor STOP not confirmed; inspect actuator abort_attempts")
            return
        if self.cfg["motors"].get("return_to_start", True):
            self.final_readbacks = self._move(self.initial_counters)
            self.restored = True
        else:
            self.final_readbacks = self._read_idle()

    def actuator_metadata(self) -> dict:
        return {"mode": self.mode, "points": len(self.plan), "position_mode": self.position_mode,
                "origin": self.origin, "initial_counters": dict(self.initial_counters),
                "reference_counters": dict(self.reference_counters),
                "return_to_start": self.cfg["motors"].get("return_to_start", True),
                "homing_performed": False, "restored_within_tolerance": self.restored,
                "final_readbacks": deepcopy(self.final_readbacks),
                "abort_attempts": deepcopy(self._abort_attempts)}

    def motor_metadata(self) -> dict:
        return deepcopy(self.initial_readbacks)
