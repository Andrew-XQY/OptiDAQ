"""Manual control of the motors selected in the experiment config.

Run from the repository root in the CERN acc-py environment:

    python experiments/motor_control.py --config configs/experiment_v1.json

Uses the row-sweep motor controller without camera acquisition or file writing.
Manual raw-counter control does not require a homed position.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shlex
import signal
import sys
from typing import Any, Mapping, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from optidaq.actuators.motor_row_sweep import (  # noqa: E402
    MotorRowSweepController,
    ScanLine,
    format_motion,
    read_axis_motion,
    restore_motor_motion,
    write_axis_motion,
)
from optidaq.extensions.cern_pyda import make_client  # noqa: E402


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "experiment_v1.json"


def interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    for section in ("rbac", "motors", "reference", "prepare", "end_processes", "scan"):
        if section not in cfg:
            raise RuntimeError("config missing section %r" % section)
    if not cfg["motors"]["x"]["enabled"]:
        raise RuntimeError("x motor must be enabled")
    return cfg


def axes_by_name(controller: MotorRowSweepController) -> dict[str, Any]:
    return {axis.name: axis for axis in controller.axes}


def axis_counter(acq: Mapping[str, Any]) -> Optional[float]:
    for key in ("position_counter", "positionCounter"):
        if key in acq:
            return float(acq[key])
    return None


def print_status(controller: MotorRowSweepController) -> None:
    print("\nMotor status:")
    for axis in controller.axes:
        acq = axis.acquisition()
        counter = axis_counter(acq)
        position = acq.get("position")
        motion = read_axis_motion(controller.client, axis.device)
        print(
            "  %s device=%s position=%s counter=%s busy=%s valid=%s "
            "limits[-]=%s limits[+]=%s units=%s scale=%s"
            % (
                axis.name,
                axis.device,
                "%.6g" % float(position) if position is not None else "--",
                "%.1f" % counter if counter is not None else "--",
                bool(acq.get("motor_busy")),
                bool(acq.get("position_valid")),
                bool(acq.get("end_switch_minus_active")),
                bool(acq.get("end_switch_plus_active")),
                axis.units,
                axis.units_scale,
            )
        )
        print("    motion %s" % format_motion(motion))


def print_plan(controller: MotorRowSweepController, limit: int = 6) -> None:
    print(
        "\nPlan rows=%d x=%.6g..%.6g y=%.6g..%.6g"
        % (
            len(controller.plan),
            controller.x_left,
            controller.x_right,
            min(controller.y_offsets),
            max(controller.y_offsets),
        )
    )
    for line in controller.plan[:max(0, int(limit))]:
        direction = "L->R" if line.x_end > line.x_start else "R->L"
        print(
            "  row %03d y_offset=%.6g x %.6g -> %.6g %s"
            % (line.row, line.y_offset, line.x_start, line.x_end, direction)
        )
    if len(controller.plan) > limit:
        print("  ... %d more row(s)" % (len(controller.plan) - limit))


def print_help() -> None:
    print(
        """
Runtime motor controls:
  units note                manual abs/off/rel use raw counters/steps
  scan note                 prepare/row/scan/run still use configured scan plan units
  status | show             print motor position, counters, and motion settings
  plan [n]                  print configured scan row preview
  motion                    print current motion settings
  ref                       move configured axes to reference counters
  prepare                   apply configured motion and move to scan start
  row N                     run one configured row action from current state
  scan                      run all configured rows from current state
  run                       prepare, run all rows, then configured cleanup
  cleanup                   restore motion and run configured end_processes
  x abs COUNTER             move x to absolute raw counter
  y abs COUNTER             move y to absolute raw counter
  x off RAW_STEPS           move x to reference counter + raw steps
  y off RAW_STEPS           move y to reference counter + raw steps
  x rel RAW_STEPS           move x by raw relative steps
  y rel RAW_STEPS           move y by raw relative steps
  x speed LOW HIGH          set x low/high speed in raw steps/s
  y speed LOW HIGH          set y low/high speed in raw steps/s
  stop                      send Stop to all configured axes
  help                      print this menu
  quit                      close this frontend
""".strip()
    )


def print_motion(controller: MotorRowSweepController) -> None:
    for axis in controller.axes:
        motion = read_axis_motion(controller.client, axis.device)
        print("%s motion %s" % (axis.name, format_motion(motion)))


def move_reference(controller: MotorRowSweepController) -> None:
    refs = controller.reference_counters
    for axis in controller.axes:
        target = refs[axis.name]
        print("%s -> reference counter %d" % (axis.name, target))
        axis.move_abs_raw(target)


def move_axis(
    controller: MotorRowSweepController,
    axis_name: str,
    mode: str,
    value_text: str,
) -> None:
    axes = axes_by_name(controller)
    if axis_name not in axes:
        raise RuntimeError("%s motor is not configured" % axis_name)
    axis = axes[axis_name]
    value = float(value_text)
    if mode == "abs":
        print("%s -> absolute counter %.1f" % (axis_name, value))
        axis.move_abs_raw(value)
    elif mode == "off":
        reference = controller.reference_counters[axis_name]
        target = int(round(reference + value))
        print(
            "%s -> reference counter %d + raw steps %.6g = target counter %d"
            % (axis_name, reference, value, target)
        )
        axis.move_abs_raw(target)
    elif mode == "rel":
        print("%s -> relative raw steps %.6g" % (axis_name, value))
        axis.move_rel(value)
    else:
        raise RuntimeError("unknown move mode %r; use abs, off, or rel" % mode)


def set_axis_speed(
    controller: MotorRowSweepController,
    axis_name: str,
    low_text: str,
    high_text: str,
) -> None:
    axes = axes_by_name(controller)
    if axis_name not in axes:
        raise RuntimeError("%s motor is not configured" % axis_name)
    axis = axes[axis_name]
    before = read_axis_motion(controller.client, axis.device)
    requested = dict(before)
    requested["speed_low"] = float(low_text)
    requested["speed_high"] = float(high_text)
    got = write_axis_motion(controller.client, axis.device, axis.name, requested)
    print("%s motion %s -> %s" % (axis_name, format_motion(before), format_motion(got)))


def line_for_index(controller: MotorRowSweepController, text: str) -> ScanLine:
    index = int(text)
    if index < 0 or index >= len(controller.plan):
        raise RuntimeError("row index %d outside 0..%d" % (index, len(controller.plan) - 1))
    return controller.plan[index]


def run_rows(controller: MotorRowSweepController) -> None:
    for line in controller.plan:
        controller.apply(line)


def cleanup_controller(controller: MotorRowSweepController) -> None:
    controller.cleanup()
    controller.motion_backup = {}


def prepare_controller(controller: MotorRowSweepController) -> None:
    if controller.motion_backup:
        raise RuntimeError("prepare already applied; run cleanup before preparing again")
    controller.prepare()


def restore_motion_only(controller: MotorRowSweepController) -> None:
    if not controller.motion_backup:
        return
    restore_motor_motion(controller.client, controller.axes, controller.motion_backup)
    controller.motion_backup = {}


def handle_command(controller: MotorRowSweepController, raw: str) -> bool:
    parts = shlex.split(raw)
    if not parts:
        return True
    command = parts[0].lower()

    if command in ("quit", "exit", "q"):
        return False
    if command in ("help", "?"):
        print_help()
    elif command in ("status", "show"):
        print_status(controller)
    elif command == "plan":
        limit = int(parts[1]) if len(parts) > 1 else 6
        print_plan(controller, limit)
    elif command == "motion":
        print_motion(controller)
    elif command == "ref":
        move_reference(controller)
    elif command == "prepare":
        prepare_controller(controller)
    elif command == "row":
        if len(parts) != 2:
            raise RuntimeError("usage: row N")
        controller.apply(line_for_index(controller, parts[1]))
    elif command == "scan":
        run_rows(controller)
    elif command == "run":
        try:
            prepare_controller(controller)
            run_rows(controller)
        finally:
            cleanup_controller(controller)
    elif command == "cleanup":
        cleanup_controller(controller)
    elif command == "stop":
        controller.abort()
    elif command in ("x", "y"):
        if len(parts) >= 2 and parts[1].lower() == "speed":
            if len(parts) != 4:
                raise RuntimeError("usage: %s speed LOW HIGH" % command)
            set_axis_speed(controller, command, parts[2], parts[3])
        else:
            if len(parts) != 3:
                raise RuntimeError("usage: %s abs|off|rel VALUE" % command)
            move_axis(controller, command, parts[1].lower(), parts[2])
    else:
        raise RuntimeError("unknown command %r; type help" % command)
    return True


def command_loop(controller: MotorRowSweepController) -> None:
    print_help()
    print_status(controller)
    while True:
        raw = input("\nmotor> ").strip()
        try:
            if not handle_command(controller, raw):
                return
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print("ERROR: %s" % exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--no-rbac", action="store_true")
    parser.add_argument(
        "--no-reference-on-start",
        action="store_true",
        help=(
            "Disable prepare.move_to_reference and require_reference_position "
            "for this frontend session."
        ),
    )
    parser.add_argument(
        "--cleanup-on-exit",
        action="store_true",
        help="Run configured cleanup on exit if prepare applied motor motion settings.",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)

    cfg = load_config(args.config)
    if args.no_reference_on_start:
        cfg = copy.deepcopy(cfg)
        cfg["prepare"]["move_to_reference"] = False
        cfg["prepare"]["require_reference_position"] = False

    require_rbac = bool(cfg["rbac"].get("enabled", False)) and not args.no_rbac
    client = make_client(require_rbac=require_rbac)
    controller = MotorRowSweepController(client, cfg)

    try:
        controller.preflight(require_valid_position=False)
        print_plan(controller)
        command_loop(controller)
    except KeyboardInterrupt:
        print("\nInterrupted; stopping configured motors.")
    finally:
        try:
            controller.abort()
        finally:
            if args.cleanup_on_exit and controller.motion_backup:
                cleanup_controller(controller)
            else:
                restore_motion_only(controller)


if __name__ == "__main__":
    main()
