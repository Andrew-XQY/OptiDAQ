"""Motor row-sweep actuator for OptiDAQ.

This module owns motor-side behavior only: scan-line planning, motor preflight,
motion settings, moves, range checks, and cleanup. DAQ synchronization and frame
waiting are orchestrated by the experiment script.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from optidaq.extensions.cern_pyda import (
    MotorAxis,
    get_value,
    plain,
    set_value,
)


SPEED_MIN_STEPS_PER_S = 10.0
SPEED_MAX_STEPS_PER_S = 50000.0
ACCELERATION_MIN_STEPS_PER_S2 = 0.0
ACCELERATION_MAX_STEPS_PER_S2 = 50000.0
MOTOR_MOTION_FIELDS = ("speed_low", "speed_high", "acceleration", "trail")
MOTOR_MOTION_FLOAT_FIELDS = ("speed_low", "speed_high", "acceleration")
MOTOR_MOTION_INT_FIELDS = ("trail",)


@dataclass(frozen=True)
class ScanLine:
    index: int
    row: int
    y_offset: float
    x_start: float
    x_end: float

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "row": self.row,
            "y_offset": self.y_offset,
            "x_start": self.x_start,
            "x_end": self.x_end,
        }


def require_scan_units(axis: MotorAxis, scan_units: str) -> None:
    axis_units = axis.units.strip().lower()
    expected = scan_units.strip().lower()
    if axis_units != expected:
        raise RuntimeError(
            "scan config uses %r but %s motor units are %r"
            % (scan_units, axis.name, axis.units)
        )


def axis_offsets(count: int, span: float) -> list[float]:
    if count <= 0:
        raise ValueError("grid count must be positive")
    if count == 1:
        return [0.0]
    step = float(span) / float(count - 1)
    start = -float(span) / 2.0
    return [start + step * index for index in range(count)]


def make_row_sweep_plan(
    scan_cfg: Mapping[str, Any],
) -> tuple[list[ScanLine], float, float, list[float]]:
    pattern = str(scan_cfg.get("pattern", "row_sweep"))
    if pattern not in ("row_sweep", "boustrophedon_row_sweep", "continuous_raster"):
        raise RuntimeError(
            "scan.pattern must be row_sweep, boustrophedon_row_sweep, "
            "or continuous_raster"
        )

    ny = int(scan_cfg["ny"])
    width = float(scan_cfg["width"])
    height = float(scan_cfg["height"])
    ys = axis_offsets(ny, height)
    left = -width / 2.0
    right = width / 2.0

    lines = []
    for row, y_offset in enumerate(ys):
        if row % 2 == 0:
            x_start, x_end = left, right
        else:
            x_start, x_end = right, left
        lines.append(
            ScanLine(
                index=row,
                row=row,
                y_offset=y_offset,
                x_start=x_start,
                x_end=x_end,
            )
        )
    return lines, left, right, ys


def scan_axis_units(scan_cfg: Mapping[str, Any]) -> dict[str, str]:
    units = scan_cfg.get("coordinate_units", "steps")
    if isinstance(units, Mapping):
        return {
            "x": str(units.get("x", "steps")),
            "y": str(units.get("y", "steps")),
        }
    return {"x": str(units), "y": str(units)}


def run_motor_processes(
    *,
    phase: str,
    processes: list,
    axes: list[MotorAxis],
    reference_counters: Mapping[str, int],
) -> None:
    for process in processes:
        if not bool(process.get("enabled", True)):
            continue
        process_type = str(process["type"])
        if process_type == "move_to_reference":
            print("%s: move motors to reference counters" % phase)
            for axis in reversed(axes):
                target = reference_counters[axis.name]
                print(
                    "%s: %s motor -> reference counter %d"
                    % (phase, axis.name, target)
                )
                axis.move_abs_raw(target)
        else:
            raise RuntimeError("%s process %r is not supported" % (phase, process_type))


def read_axis_motion(client: Any, device: str) -> dict[str, float | int]:
    expert = get_value(client, "%s/ExpertSetting" % device)
    return {
        "speed_low": float(plain(expert["speed_low"])),
        "speed_high": float(plain(expert["speed_high"])),
        "acceleration": float(plain(expert["acceleration"])),
        "trail": int(plain(expert["trail"])),
    }


def validate_axis_motion(name: str, motion: Mapping[str, float | int]) -> None:
    speed_low = float(motion["speed_low"])
    speed_high = float(motion["speed_high"])
    acceleration = float(motion["acceleration"])
    trail = int(motion["trail"])

    for label, value in (("speed_low", speed_low), ("speed_high", speed_high)):
        if not (SPEED_MIN_STEPS_PER_S <= value <= SPEED_MAX_STEPS_PER_S):
            raise RuntimeError(
                "%s %s %.1f outside %.0f..%.0f steps/s"
                % (name, label, value, SPEED_MIN_STEPS_PER_S, SPEED_MAX_STEPS_PER_S)
            )
    if speed_low > speed_high:
        raise RuntimeError(
            "%s speed_low %.1f > speed_high %.1f"
            % (name, speed_low, speed_high)
        )
    if not (
        ACCELERATION_MIN_STEPS_PER_S2
        <= acceleration
        <= ACCELERATION_MAX_STEPS_PER_S2
    ):
        raise RuntimeError(
            "%s acceleration %.1f outside %.0f..%.0f steps/s^2"
            % (
                name,
                acceleration,
                ACCELERATION_MIN_STEPS_PER_S2,
                ACCELERATION_MAX_STEPS_PER_S2,
            )
        )
    if trail < 0:
        raise RuntimeError("%s trail %d must be non-negative" % (name, trail))


def motion_payload(motion: Mapping[str, float | int]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in MOTOR_MOTION_FLOAT_FIELDS:
        payload[key] = np.float64(motion[key])
    for key in MOTOR_MOTION_INT_FIELDS:
        payload[key] = np.int32(motion[key])
    return payload


def format_motion(motion: Mapping[str, float | int]) -> str:
    return (
        "low/high %.0f/%.0f steps/s accel %.0f steps/s^2 trail %d steps"
        % (
            float(motion["speed_low"]),
            float(motion["speed_high"]),
            float(motion["acceleration"]),
            int(motion["trail"]),
        )
    )


def write_axis_motion(
    client: Any,
    device: str,
    name: str,
    motion: Mapping[str, float | int],
) -> dict[str, float | int]:
    validate_axis_motion(name, motion)
    before = get_value(client, "%s/ExpertSetting" % device)
    set_value(
        client,
        "%s/ExpertSetting" % device,
        motion_payload(motion),
    )
    after = get_value(client, "%s/ExpertSetting" % device)

    guarded = ("position_min", "position_max", "units_scale", "units_offset")
    changed = [
        key
        for key in guarded
        if key in before and float(plain(before[key])) != float(plain(after[key]))
    ]
    if changed:
        try:
            set_value(client, "%s/ExpertSetting" % device, dict(before))
        except BaseException:
            pass
        raise RuntimeError(
            "%s motion write also changed %s; rollback attempted"
            % (name, changed)
        )
    return {
        "speed_low": float(plain(after["speed_low"])),
        "speed_high": float(plain(after["speed_high"])),
        "acceleration": float(plain(after["acceleration"])),
        "trail": int(plain(after["trail"])),
    }


def apply_motor_motion(
    client: Any,
    axes: list[MotorAxis],
    motors_cfg: Mapping[str, Any],
    backup: Optional[dict[str, dict[str, float | int]]] = None,
) -> dict[str, dict[str, float | int]]:
    if backup is None:
        backup = {}
    for axis in axes:
        cfg = motors_cfg[axis.name]
        requested = {key: cfg.get(key) for key in MOTOR_MOTION_FIELDS}
        if all(value is None for value in requested.values()):
            continue

        current = read_axis_motion(client, axis.device)
        new_motion = dict(current)
        for key, value in requested.items():
            if value is None:
                continue
            if key in MOTOR_MOTION_INT_FIELDS:
                new_motion[key] = int(value)
            else:
                new_motion[key] = float(value)
        validate_axis_motion(axis.name, new_motion)

        backup.setdefault(axis.name, current)
        got = write_axis_motion(
            client,
            axis.device,
            axis.name,
            new_motion,
        )
        print(
            "%s motion %s -> %s"
            % (axis.name, format_motion(current), format_motion(got))
        )
    return backup


def restore_motor_motion(
    client: Any,
    axes: list[MotorAxis],
    backup: Mapping[str, Mapping[str, float | int]],
) -> None:
    axes_by_name = {axis.name: axis for axis in axes}
    for name, motion in backup.items():
        axis = axes_by_name[name]
        try:
            write_axis_motion(client, axis.device, name, motion)
            print("restored %s motion to %s" % (name, format_motion(motion)))
        except BaseException as exc:
            print("WARNING: failed to restore %s motion: %s" % (name, exc))


def estimate_trapezoid_profile(
    distance_steps: float,
    motion: Mapping[str, float | int],
) -> dict[str, float | bool]:
    distance = abs(float(distance_steps))
    speed_low = float(motion["speed_low"])
    speed_high = float(motion["speed_high"])
    acceleration = float(motion["acceleration"])
    trail = max(0.0, float(motion["trail"]))

    if distance <= 0.0:
        return {
            "distance_steps": 0.0,
            "ramp_steps_each": 0.0,
            "cruise_steps": 0.0,
            "constant_fraction": 1.0,
            "duration_s": 0.0,
            "reaches_speed_high": True,
        }

    if speed_high <= speed_low or acceleration <= 0.0:
        speed = max(speed_high, speed_low)
        duration = distance / speed if speed > 0.0 else 0.0
        if speed_high <= speed_low:
            constant_steps = distance
        else:
            constant_steps = max(0.0, distance - trail)
        return {
            "distance_steps": distance,
            "ramp_steps_each": 0.0,
            "cruise_steps": constant_steps,
            "constant_fraction": constant_steps / distance,
            "duration_s": duration,
            "reaches_speed_high": speed_high <= speed_low,
        }

    ramp_steps = (speed_high * speed_high - speed_low * speed_low) / (
        2.0 * acceleration
    )
    ramp_time = (speed_high - speed_low) / acceleration
    cruise_steps = distance - 2.0 * ramp_steps - trail
    reaches_high = cruise_steps > 0.0
    if not reaches_high:
        available_steps = max(0.0, distance - trail)
        if available_steps <= 0.0 or speed_low <= 0.0:
            duration = distance / speed_low if speed_low > 0.0 else 0.0
            actual_ramp_steps = 0.0
        else:
            peak_speed = (speed_low * speed_low + acceleration * available_steps) ** 0.5
            duration = 2.0 * (peak_speed - speed_low) / acceleration
            duration += trail / speed_low
            actual_ramp_steps = available_steps / 2.0
        return {
            "distance_steps": distance,
            "ramp_steps_each": actual_ramp_steps,
            "cruise_steps": 0.0,
            "constant_fraction": 0.0,
            "duration_s": duration,
            "reaches_speed_high": False,
        }

    trail_time = trail / speed_low if speed_low > 0.0 else 0.0
    duration = 2.0 * ramp_time + cruise_steps / speed_high + trail_time
    return {
        "distance_steps": distance,
        "ramp_steps_each": ramp_steps,
        "cruise_steps": cruise_steps,
        "constant_fraction": cruise_steps / distance,
        "duration_s": duration,
        "reaches_speed_high": True,
    }


class MotorRowSweepController:
    mode = "motor"

    def __init__(self, client: Any, cfg: Mapping[str, Any]) -> None:
        self.client = client
        self.cfg = cfg
        self.scan_cfg = cfg["scan"]
        self.motors_cfg = cfg["motors"]
        self.prepare_cfg = cfg["prepare"]
        self.end_processes = list(cfg["end_processes"])
        self.reference_cfg = cfg["reference"]
        self.scan_units = scan_axis_units(self.scan_cfg)
        self.plan, self.x_left, self.x_right, self.y_offsets = make_row_sweep_plan(
            self.scan_cfg
        )
        self.frames_per_point = int(self.scan_cfg["frames_per_point"])
        self.x_samples_per_row = int(
            self.scan_cfg.get("nx", self.frames_per_point)
        )
        self.reference_counters = {
            "x": int(self.reference_cfg["x_counter"]),
            "y": int(self.reference_cfg["y_counter"]),
        }
        self.reference_tolerance = int(self.reference_cfg.get("tolerance_counts", 5))
        self.x_axis: Optional[MotorAxis] = None
        self.y_axis: Optional[MotorAxis] = None
        self.axes: list[MotorAxis] = []
        self.motion_backup: dict[str, dict[str, float | int]] = {}

    def preflight(self, *, require_valid_position: bool = True) -> None:
        x_cfg = self.motors_cfg["x"]
        y_cfg = self.motors_cfg["y"]
        self.x_axis = MotorAxis(
            "x", self.client, x_cfg, require_valid_position=require_valid_position
        )
        self.y_axis = (
            MotorAxis(
                "y", self.client, y_cfg, require_valid_position=require_valid_position
            ) if bool(y_cfg["enabled"]) else None
        )
        self.axes = [axis for axis in (self.x_axis, self.y_axis) if axis is not None]

        for axis in self.axes:
            axis.preflight()
            print(
                "motor %s device=%s start_position=%.6g units=%s units_scale=%s "
                "limits=%.6g..%.6g"
                % (
                    axis.name,
                    axis.device,
                    axis.start_position,
                    axis.units,
                    axis.units_scale,
                    axis.position_min,
                    axis.position_max,
                )
            )
            require_scan_units(axis, self.scan_units[axis.name])

        if self.x_axis is None:
            raise RuntimeError("x motor is not configured")
        x_offsets = [self.x_left, self.x_right]
        self.x_axis.range_check_reference_offsets(
            self.reference_counters["x"],
            x_offsets,
        )
        if self.y_axis is not None:
            self.y_axis.range_check_reference_offsets(
                self.reference_counters["y"],
                self.y_offsets,
            )
        else:
            if any(offset != 0 for offset in self.y_offsets):
                raise RuntimeError("scan requested y offsets while y motor is disabled")
            print("y motor disabled in config; y offsets must stay zero")

        x_target_counters = [
            self.x_axis.counter_for_offset(self.reference_counters["x"], offset)
            for offset in x_offsets
        ]
        y_target_counters = []
        if self.y_axis is not None:
            y_target_counters = [
                self.y_axis.counter_for_offset(self.reference_counters["y"], offset)
                for offset in self.y_offsets
            ]

        x_min, x_max = min(x_offsets), max(x_offsets)
        y_min, y_max = min(self.y_offsets), max(self.y_offsets)
        y_stride = (
            self.y_offsets[1] - self.y_offsets[0]
            if len(self.y_offsets) > 1
            else 0.0
        )
        print(
            "actuator=motor scan pattern=%s rows=%d x_samples_per_row=%d "
            "frames_per_row=%d x_units=%s x=%.6g..%.6g y_units=%s "
            "y=%.6g..%.6g y_stride=%.6g"
            % (
                self.scan_cfg.get("pattern", "row_sweep"),
                len(self.plan),
                self.x_samples_per_row,
                self.frames_per_point,
                self.scan_units["x"],
                x_min,
                x_max,
                self.scan_units["y"],
                y_min,
                y_max,
                y_stride,
            )
        )
        print(
            "scan start top-left x_offset=%.6g y_offset=%.6g"
            % (self.x_left, self.y_offsets[0])
        )
        print(
            "reference counters x=%d y=%d; target counter span x=%d..%d"
            % (
                self.reference_counters["x"],
                self.reference_counters["y"],
                min(x_target_counters),
                max(x_target_counters),
            )
        )
        if y_target_counters:
            print(
                "target counter span y=%d..%d"
                % (min(y_target_counters), max(y_target_counters))
            )

    def prepare(self) -> None:
        self._move_to_reference_if_requested()
        self._require_reference_position()
        self.motion_backup = {}
        apply_motor_motion(
            self.client,
            self.axes,
            self.motors_cfg,
            backup=self.motion_backup,
        )
        self._print_line_motion_estimate()
        self._move_to_scan_start()

    def apply(self, line: ScanLine) -> None:
        self.begin_line(line)
        self.sweep_line(line)

    def begin_line(self, line: ScanLine) -> None:
        direction = "L->R" if line.x_end > line.x_start else "R->L"
        print(
            "row %03d y_offset=%.6g sweep=%s to x_offset=%.6g"
            % (line.row, line.y_offset, direction, line.x_end)
        )

        if self.y_axis is not None:
            self.y_axis.move_to_offset(self.reference_counters["y"], line.y_offset)
        elif line.y_offset:
            raise RuntimeError("scan requested y move while y motor is disabled")

    def sweep_line(self, line: ScanLine) -> None:
        if self.x_axis is None:
            raise RuntimeError("motor controller is not ready")
        self.x_axis.move_to_offset(self.reference_counters["x"], line.x_end)

    def dwell(self) -> None:
        return None

    def abort(self) -> None:
        for axis in self.axes:
            axis.stop()

    def cleanup(self) -> None:
        if self.motion_backup:
            restore_motor_motion(self.client, self.axes, self.motion_backup)
        run_motor_processes(
            phase="end",
            processes=self.end_processes,
            axes=self.axes,
            reference_counters=self.reference_counters,
        )

    def actuator_metadata(self) -> dict:
        return {"mode": self.mode}

    def motor_metadata(self) -> dict:
        return {
            axis.name: {
                "device": axis.device,
                "units": axis.units,
                "units_scale": axis.units_scale,
                "units_offset": axis.units_offset,
                "start_position": axis.start_position,
                "start_counter": axis.start_counter,
                "reference_counter": self.reference_counters[axis.name],
            }
            for axis in self.axes
        }

    def magnet_metadata(self) -> dict:
        return {}

    def _move_to_scan_start(self) -> None:
        if self.x_axis is None:
            raise RuntimeError("x motor is not configured")
        print("prepare: move motors to scan start top-left")
        first = self.plan[0]
        if self.y_axis is not None:
            self.y_axis.move_to_offset(self.reference_counters["y"], first.y_offset)
        elif first.y_offset:
            raise RuntimeError("scan requested y move while y motor is disabled")
        self.x_axis.move_to_offset(self.reference_counters["x"], first.x_start)

    def _move_to_reference_if_requested(self) -> None:
        if not bool(self.prepare_cfg["move_to_reference"]):
            return
        if self.x_axis is None:
            raise RuntimeError("x motor is not configured")
        print("prepare: move motors to reference counters before DAQ start")
        self.x_axis.move_abs_raw(self.reference_counters["x"])
        if self.y_axis is not None:
            self.y_axis.move_abs_raw(self.reference_counters["y"])

    def _require_reference_position(self) -> None:
        if not bool(self.prepare_cfg["require_reference_position"]):
            return
        if self.x_axis is None:
            raise RuntimeError("x motor is not configured")
        self.x_axis.require_counter_near(
            self.reference_counters["x"],
            self.reference_tolerance,
        )
        if self.y_axis is not None:
            self.y_axis.require_counter_near(
                self.reference_counters["y"],
                self.reference_tolerance,
            )

    def _print_line_motion_estimate(self) -> None:
        if self.x_axis is None:
            raise RuntimeError("x motor is not configured")

        left_counter = self.x_axis.counter_for_offset(
            self.reference_counters["x"],
            self.x_left,
        )
        right_counter = self.x_axis.counter_for_offset(
            self.reference_counters["x"],
            self.x_right,
        )
        line_steps = abs(right_counter - left_counter)
        motion = read_axis_motion(self.client, self.x_axis.device)
        profile = estimate_trapezoid_profile(line_steps, motion)
        trigger_hz = float(self.scan_cfg["trigger_hz"])
        frame_window_s = self.frames_per_point / trigger_hz
        duration_s = float(profile["duration_s"])
        old_timeout_s = float(self.x_axis.timeout_s)
        new_timeout_s = max(old_timeout_s, 2.0 * duration_s)
        self.x_axis.timeout_s = new_timeout_s
        print(
            "x line motion estimate: distance=%d steps %s duration=%.3fs "
            "cruise=%.1f%% ramp_each=%.1f steps frame_window=%.3fs"
            % (
                line_steps,
                format_motion(motion),
                duration_s,
                100.0 * float(profile["constant_fraction"]),
                float(profile["ramp_steps_each"]),
                frame_window_s,
            )
        )
        if new_timeout_s > old_timeout_s:
            print(
                "x move timeout %.3fs -> %.3fs (2x line estimate)"
                % (old_timeout_s, new_timeout_s)
            )
        if not bool(profile["reaches_speed_high"]):
            print(
                "WARNING: x line is too short for this acceleration profile; "
                "speed_high will not be reached"
            )
        if duration_s < frame_window_s:
            print(
                "WARNING: x line motion is shorter than frames_per_point/"
                "trigger_hz; reduce speed_high or frames_per_point to avoid "
                "stationary frames at row end"
            )
