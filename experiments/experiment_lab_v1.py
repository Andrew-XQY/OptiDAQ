"""Acquire lab camera PNGs with a point-motor raster, DMD or camera-only plan.

    python experiments/experiment_lab_v1.py --config configs/experiment_lab_v1.json

Motor raster mode uses settled-position readbacks and explicit capture-ID pairs.
DMD/camera-only modes retain independent capture and timestamp finalization.
The active raw beam-time runner remains separate.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping as MappingABC
import json
from pathlib import Path
import re
import shutil
import signal
import sys
import time
from typing import Any, Callable, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from optidaq.after_collection import configured_actions, run_after_collection
from optidaq.actuators.dmd_pattern_scan import DmdPatternScanController, DmdPoint, validate_dmd_config
from optidaq.actuators.noop import NoopActuatorController, NoopPoint, make_noop_plan
from optidaq.daq.actuator_events import EVENT_LOG_FILENAME, append_actuator_event, write_actuator_frame_index
from optidaq.daq.contracts import Problem
from optidaq.daq.engine import run_engine
from optidaq.daq.pipeline import DAQPipeline
from optidaq.daq.preflight import make_write_access_check, run_preflight
from optidaq.daq.run_store import build_raw_spec, create_run_dir, write_resolved_config
from optidaq.daq.save_backends import make_save_backend, output_format, png_mode
from optidaq.daq.sync_tools import (
    pulse_id_candidate_fields, tap_capture_mode, tap_poll_interval_s,
    validate_poll_rate, validate_tap_stamp_sources,
)
from optidaq.daq.watchdog import CounterFreezeWatchdog
from optidaq.data.png_io import write_png_gray
from optidaq.data.scientific_png_export import finalize_png_dataset
from optidaq.extensions.cern_pyda import (
    PollingCameraTap, SubscribingCameraTap, camera_map_from_config,
    collect_camera_source_provenance, configure_cameras, ensure_cameras_alive,
    jsonable, make_client,
)

DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "experiment_lab_v1.json"
PNG_NAME_RE = re.compile(r"^(\d{20})_(\d{8})\.png$")
STOP_REQUESTED = False


class PlanProgress:
    def __init__(self, total: int, width: int = 24) -> None:
        self.total = max(0, int(total))
        self.width = max(1, int(width))
        self.done = 0
        self.started_at = time.monotonic()

    def advance(self) -> None:
        if self.total <= 0:
            return
        self.done = min(self.done + 1, self.total)
        ratio = self.done / self.total
        filled = int(ratio * self.width)
        if self.done >= self.total:
            bar = "=" * self.width
        else:
            filled = min(filled, self.width - 1)
            bar = "%s>%s" % ("=" * filled, "." * (self.width - filled - 1))
        eta = self._eta()
        print(
            "progress [%s] %d/%d %d%% eta=%s"
            % (bar, self.done, self.total, int(round(100.0 * ratio)), eta),
            flush=True,
        )

    def _eta(self) -> str:
        if self.done >= self.total:
            return "0s"
        if self.done <= 1:
            return "--"
        elapsed = time.monotonic() - self.started_at
        remaining = elapsed / float(self.done) * float(self.total - self.done)
        return self._format_duration(remaining)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds_i = max(0, int(round(seconds)))
        if seconds_i < 60:
            return "%ds" % seconds_i
        minutes, seconds_i = divmod(seconds_i, 60)
        if minutes < 60:
            return "%dm%02ds" % (minutes, seconds_i)
        hours, minutes = divmod(minutes, 60)
        return "%dh%02dm" % (hours, minutes)


def request_stop(signum: int, frame: object) -> None:
    global STOP_REQUESTED
    if not STOP_REQUESTED:
        print("\nStop requested; finishing cleanup...", flush=True)
    STOP_REQUESTED = True


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    validate_config_minimal(cfg)
    return cfg


def actuator_mode(cfg: Mapping[str, Any]) -> str:
    actuator = cfg.get("actuator", {})
    if isinstance(actuator, str):
        mode = actuator
    elif isinstance(actuator, MappingABC):
        mode = actuator.get("mode", "dmd")
    else:
        raise RuntimeError("actuator must be an object or string")
    mode = str(mode).lower()
    if mode in ("dmd", "dmd_pattern_scan"):
        return "dmd"
    if mode in ("motor_raster", "motor_point_raster"):
        return "motor_raster"
    if mode in ("none", "noop", "no_op", "dummy", "external", "external_actuator"):
        return "noop"
    raise RuntimeError("experiment_lab_v1 supports motor_raster, dmd and noop")


def validate_config_minimal(cfg: Mapping[str, Any]) -> None:
    if output_format(cfg) != "png":
        raise RuntimeError("experiment_lab_v1 requires output.format=png; use experiment_v1 for raw acquisition")
    png_mode(cfg)
    mode = actuator_mode(cfg)
    for section in ("run", "rbac", "cameras", "tap", "scan", "writer", "camera_setting", "sync"):
        if section not in cfg:
            raise RuntimeError("config missing section %r" % section)
    if not cfg["cameras"]:
        raise RuntimeError("config cameras must not be empty")
    if float(cfg["scan"]["trigger_hz"]) <= 0:
        raise RuntimeError("scan.trigger_hz must be positive")
    if int(cfg["scan"]["frames_per_point"]) <= 0:
        raise RuntimeError("scan.frames_per_point must be positive")
    if float(cfg["scan"]["point_timeout_s"]) <= 0:
        raise RuntimeError("scan.point_timeout_s must be positive")
    if mode == "motor_raster":
        from optidaq.daq.lab_motor_run import validate_motor_lab_config
        validate_motor_lab_config(cfg)
        return
    if mode == "dmd":
        if "dmd" not in cfg:
            raise RuntimeError("config missing section 'dmd'")
        validate_dmd_config(cfg["dmd"])
    else:
        make_noop_plan(cfg)
    validate_poll_rate(cfg["scan"], cfg["tap"], cfg["sync"])


def file_list(run_dir: Path) -> list:
    entries = []
    for path in sorted(run_dir.iterdir()):
        if path.is_file():
            size = path.stat().st_size
            name = path.name
        elif path.is_dir():
            size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
            name = "%s/" % path.name
        else:
            continue
        entries.append((name, size / (1024.0 * 1024.0)))
    return entries


def print_created_files(run_dir: Path) -> None:
    print("\ncreated files:")
    for name, size_mb in file_list(run_dir):
        print("  %s  %.3f MB" % (name, size_mb))


class FrameWaiter:
    def __init__(
        self,
        *,
        pipeline: DAQPipeline,
        aliases: list[str],
        frames_per_point: int,
        timeout_s: float,
        point_log: list,
        raise_if_failed: Callable[[], None],
        should_stop: Callable[[], bool],
    ) -> None:
        self.pipeline = pipeline
        self.aliases = aliases
        self.frames_per_point = frames_per_point
        self.timeout_s = timeout_s
        self.point_log = point_log
        self.raise_if_failed = raise_if_failed
        self.should_stop = should_stop

    def markers(self) -> dict[str, int]:
        stats = self.pipeline.stats()
        return {
            device: stats.per_device_saved.get(device, 0)
            for device in self.aliases
        }

    def wait(
        self,
        markers: Mapping[str, int],
        item: Any,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        label = self._label(item)
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            if self.should_stop():
                raise KeyboardInterrupt("stop requested")
            self.raise_if_failed()
            stats = self.pipeline.stats()
            saved = stats.per_device_saved
            if all(
                saved.get(device, 0) >= markers[device] + self.frames_per_point
                for device in self.aliases
            ):
                point_index = len(self.point_log) + 1
                entry = {
                    "point_index": point_index,
                    "per_device_saved_start": dict(markers),
                    "per_device_saved_end": dict(saved),
                    "latest_source_image_number": dict(
                        stats.latest_source_image_number
                    ),
                    "latest_acq_stamp_ns": dict(stats.latest_acq_stamp_ns),
                }
                if hasattr(item, "to_dict"):
                    entry.update(item.to_dict())
                if extra:
                    entry.update(dict(extra))
                self.point_log.append(entry)
                print("%s saved=%s" % (label, dict(saved)))
                return
            time.sleep(0.05)

        raise RuntimeError(
            "%s timeout: no %d saved frame(s) on all devices within %.1f s "
            "(markers=%s)"
            % (label, self.frames_per_point, self.timeout_s, dict(markers))
        )

    def _label(self, item: Any) -> str:
        return "point %d" % int(getattr(item, "index", len(self.point_log)))


def make_controller(client: Any, cfg: Mapping[str, Any]) -> Any:
    if actuator_mode(cfg) == "dmd":
        return DmdPatternScanController(client, cfg)
    return NoopActuatorController(client, cfg)


def postprocess_config(cfg: Mapping[str, Any]) -> dict:
    postprocess = cfg.get("postprocess", {})
    if not isinstance(postprocess, Mapping):
        raise RuntimeError("postprocess must be an object")
    return {
        "build_aligned_index": bool(postprocess.get("build_aligned_index", True)),
        "validate_after_scan": bool(postprocess.get("validate_after_scan", True)),
        "actions": configured_actions(cfg),
    }


def parse_png_name(path: Path) -> tuple[int, int]:
    match = PNG_NAME_RE.fullmatch(path.name)
    if match is None:
        raise RuntimeError("invalid png filename %s" % path.name)
    return int(match.group(1)), int(match.group(2))


def read_png_records_by_device(run_dir: Path, aliases: list[str]) -> dict[str, list[dict]]:
    records_by_device = {}
    for alias in aliases:
        records = []
        for path in sorted((run_dir / alias).glob("*.png")):
            acq_stamp_ns, frame_index = parse_png_name(path)
            records.append(
                {
                    "device": alias,
                    "frame_index": frame_index,
                    "acq_stamp_ns": acq_stamp_ns,
                    "path": path,
                }
            )
        records_by_device[alias] = records
    return records_by_device


def validate_png_device(
    run_dir: Path,
    alias: str,
    min_records: int,
) -> dict:
    errors = []
    warnings = []
    device_dir = run_dir / alias
    if not device_dir.is_dir():
        errors.append("png output directory is missing: %s" % device_dir)
        records = []
    else:
        records = read_png_records_by_device(run_dir, [alias])[alias]

    if len(records) < min_records:
        errors.append(
            "png record count below expected minimum: expected at least %d, got %d"
            % (min_records, len(records))
        )

    previous_stamp = None
    for record_number, record in enumerate(records):
        frame_index = record["frame_index"]
        acq_stamp_ns = record["acq_stamp_ns"]
        if frame_index != record_number:
            errors.append(
                "png frame_index jump at record %d: expected %d, got %d"
                % (record_number, record_number, frame_index)
            )
        if previous_stamp is not None and acq_stamp_ns < previous_stamp:
            errors.append(
                "png timestamp decreases at record %d: %d after %d"
                % (record_number, acq_stamp_ns, previous_stamp)
            )
        previous_stamp = acq_stamp_ns

    return {
        "device": alias,
        "n_records": len(records),
        "n_files": len(records),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    cfg = load_config(args.config)
    postprocess = postprocess_config(cfg)  # Reject incompatible actions before hardware access.
    if actuator_mode(cfg) == "motor_raster":
        from optidaq.daq.lab_motor_run import run_motor_raster
        run_motor_raster(cfg, args.config, should_stop=lambda: STOP_REQUESTED)
        return
    camera_map = camera_map_from_config(cfg)
    aliases = list(camera_map)
    sync_cfg = cfg["sync"]
    trigger_period_s = 1.0 / float(cfg["scan"]["trigger_hz"])
    problems: list[Problem] = []
    client = make_client(require_rbac=bool(cfg["rbac"]["enabled"]))
    tap_mode = tap_capture_mode(cfg["tap"])
    use_subscribe = tap_mode == "subscribe"
    tap_kwargs = dict(
        get_selector=cfg["tap"].get("get_selector"),
        poll_s=tap_poll_interval_s(cfg["tap"], required=not use_subscribe),
        stamp_fields=cfg["tap"]["stamp_fields"],
        min_ns_epoch=int(cfg["tap"]["min_ns_epoch"]),
        pulse_id_fields=pulse_id_candidate_fields(sync_cfg),
        settle_poll_s=float(cfg["tap"].get("settle_poll_s", 0.005)),
        settle_timeout_s=float(cfg["tap"].get("settle_timeout_s", 0.25)),
        max_trailing_zero_rows=cfg["tap"].get("max_trailing_zero_rows", 0),
        sync_cfg=sync_cfg,
    )
    if use_subscribe:
        tap_kwargs["selector"] = cfg["tap"].get("subscribe_selector")
    tap_class = SubscribingCameraTap if use_subscribe else PollingCameraTap
    taps = [tap_class(alias, fesa, client, **tap_kwargs) for alias, fesa in camera_map.items()]
    print("lab camera tap mode: %s; independent capture, no exposure handshake" % tap_mode)

    camera_readback = {}
    if bool(cfg["writer"]["configure_cameras"]):
        ensure_cameras_alive(client, camera_map, cfg["writer"])
        camera_readback = configure_cameras(
            client, camera_map, cfg["camera_setting"],
            get_selector=cfg["tap"].get("get_selector"), use_subscription=use_subscribe,
            subscribe_selector=cfg["tap"].get("subscribe_selector"),
        )
    provenance = collect_camera_source_provenance(client, camera_map)
    for tap in taps:
        tap.probe()
        print("camera %s fesa=%s shape=%s dtype=%s stamp_source=%s" % (
            tap.device, tap.fesa_device, tap.image_shape, tap.image_dtype, tap.stamp_source,
        ))
    validate_tap_stamp_sources(taps, sync_cfg)

    controller = make_controller(client, cfg)
    run_dir = None
    pipeline = None
    watchdog = None
    pipeline_started = watchdog_started = False
    primary_error: Optional[BaseException] = None
    point_log: list = []
    started = time.monotonic()
    try:
        # DMD preflight opens the device, so all subsequent setup shares cleanup.
        controller.preflight()
        points_total = len(controller.plan)
        frames_per_point = int(controller.frames_per_point)
        if not points_total:
            raise RuntimeError("scan plan is empty")
        root = Path(cfg["run"]["data_root"]) / str(cfg["run"]["runs_subdir"])
        failed = [result for result in run_preflight([make_write_access_check(root)]) if not result.ok]
        if failed:
            raise RuntimeError("preflight failed: " + "; ".join("%s: %s" % (r.name, r.message) for r in failed))
        estimated_bytes = sum(int(tap.payload_nbytes or 0) for tap in taps) * points_total * frames_per_point
        required_bytes = int(estimated_bytes * float(cfg["run"]["free_disk_margin"]))
        if shutil.disk_usage(root).free < required_bytes:
            raise RuntimeError("not enough free disk space under %s" % root)
        run_dir = create_run_dir(root)
        specs = {
            tap.device: build_raw_spec(
                file_prefix=tap.device, payload_nbytes=int(tap.payload_nbytes or 0),
                segment_max_bytes=int(cfg["writer"]["segment_max_bytes"]),
            ) for tap in taps
        }
        write_resolved_config(run_dir, {
            "config_path": str(args.config), "config": jsonable(cfg),
            "cameras": {
                tap.device: {
                    "fesa": tap.fesa_device, "shape": list(tap.image_shape or ()),
                    "dtype": tap.image_dtype, "payload_nbytes": tap.payload_nbytes,
                    "stamp_source": tap.stamp_source,
                    "cameraPixelFormat": cfg["camera_setting"].get(tap.device, {}).get("cameraPixelFormat"),
                } for tap in taps
            },
            "camera_setting_readback": jsonable(camera_readback),
            "camera_source_provenance": jsonable(provenance),
            "actuator": jsonable(controller.actuator_metadata()),
            "scan_plan": [point.to_dict() for point in controller.plan],
            "output_format": "png", "png_mode": png_mode(cfg),
            "postprocess": jsonable(postprocess), "sync": jsonable(sync_cfg),
            "trigger_period_ns": int(trigger_period_s * 1_000_000_000),
            "points_total": points_total, "script": Path(__file__).name,
        })
        print("run_dir=%s" % run_dir)
        backends = {
            tap.device: make_save_backend(
                tap.device, run_dir, specs[tap.device], image_shape=tap.image_shape,
                image_dtype=tap.image_dtype, config=cfg,
            ) for tap in taps
        }
        pipeline = DAQPipeline(
            taps=taps, backends=backends, queue_maxsize=int(cfg["writer"]["queue_maxsize"]),
            on_problem=problems.append,
        )
        watchdog = CounterFreezeWatchdog(
            pipeline.stats, aliases, problems.append,
            freeze_timeout_s=max(10.0, 5.0 * trigger_period_s), poll_interval_s=1.0,
        )

        def raise_if_failed_all() -> None:
            pipeline.raise_if_failed()
            for tap in taps:
                if tap.error is not None:
                    raise RuntimeError("camera tap %s failed: %s" % (tap.device, tap.error)) from tap.error

        waiter = FrameWaiter(
            pipeline=pipeline, aliases=aliases, frames_per_point=frames_per_point,
            timeout_s=float(cfg["scan"]["point_timeout_s"]), point_log=point_log,
            raise_if_failed=raise_if_failed_all, should_stop=lambda: STOP_REQUESTED,
        )
        progress = PlanProgress(points_total)
        event_path = run_dir / EVENT_LOG_FILENAME
        save_patterns = isinstance(controller, DmdPatternScanController) and cfg["dmd"].get("save_patterns", True)

        def apply_with_runtime(item: Any) -> None:
            if isinstance(controller, DmdPatternScanController):
                if not isinstance(item, DmdPoint):
                    raise RuntimeError("DMD plan item must be a DmdPoint")
                append_actuator_event(event_path, "change_begin", item.index)
                info, pattern = controller.apply_point(item)
                settled_ns = time.time_ns()
                if save_patterns:
                    relative = "dmd_patterns/%06d.png" % item.index
                    write_png_gray(run_dir / relative, pattern)
                    info["dmd_pattern_file"] = relative
                append_actuator_event(event_path, "settle_complete", item.index,
                                      metadata=jsonable(info), host_time_ns=settled_ns)
                waiter.wait(waiter.markers(), item, extra=info)
            else:
                if not isinstance(item, NoopPoint):
                    raise RuntimeError("camera-only plan item must be a NoopPoint")
                controller.apply(item)
                waiter.wait(waiter.markers(), item)

        def abort_with_event() -> None:
            try:
                if isinstance(controller, DmdPatternScanController):
                    append_actuator_event(event_path, "end")
            finally:
                controller.abort()

        def dwell_with_progress() -> None:
            controller.dwell()
            progress.advance()

        controller.prepare()
        pipeline.start()
        pipeline_started = True
        watchdog.start()
        watchdog_started = True
        run_engine(controller.plan, apply_with_runtime, dwell_with_progress,
                   abort=abort_with_event, should_stop=lambda: STOP_REQUESTED)
    except BaseException as exc:
        primary_error = exc
    finally:
        if watchdog_started:
            watchdog.stop()
        if pipeline_started:
            try:
                pipeline.stop(float(cfg["writer"]["drain_timeout_s"]))
            except BaseException as exc:
                problems.append(Problem(source="pipeline.stop", message=str(exc), severity="error", exception=exc))
                if primary_error is None:
                    primary_error = exc
        for tap in taps:
            if tap.error is not None:
                problems.append(Problem(source="camera_tap.%s" % tap.device,
                                        message=str(tap.error), severity="error", exception=tap.error))
                if primary_error is None:
                    primary_error = RuntimeError("camera tap %s failed" % tap.device)
        try:
            controller.cleanup()
        except BaseException as exc:
            problems.append(Problem(source="actuator.cleanup", message=str(exc), severity="error", exception=exc))
            if primary_error is None:
                primary_error = exc

    if primary_error is not None:
        if run_dir is not None:
            print_created_files(run_dir)
        raise primary_error
    print("lab scan elapsed %.1f s; points_done=%d/%d" % (
        time.monotonic() - started, len(point_log), points_total,
    ))
    if isinstance(controller, DmdPatternScanController) and postprocess["build_aligned_index"]:
        try:
            index_path = write_actuator_frame_index(run_dir, aliases)
            print("DMD frame associations: %s (approximate)" % index_path.name)
        except (ValueError, RuntimeError, OSError) as exc:
            print("WARNING: DMD frame association index failed: %s" % exc)
            problems.append(Problem(source="actuator_frame_index", message=str(exc), exception=exc))
    result = finalize_png_dataset(run_dir, cameras=aliases)
    print("PNG metadata database: %s" % result.db_path)
    validation_errors = False
    if postprocess["validate_after_scan"]:
        for alias in aliases:
            summary = validate_png_device(run_dir, alias, points_total * frames_per_point)
            validation_errors = validation_errors or bool(summary["errors"])
            print("%s png_files=%d errors=%d" % (alias, summary["n_files"], len(summary["errors"])))
            for error in summary["errors"]:
                print("  ERROR: %s" % error)
    collection_complete = not STOP_REQUESTED and len(point_log) == points_total
    if not collection_complete:
        print("collection stopped before all planned points finished")
    elif problems or validation_errors:
        print("finished with problems")
    else:
        print("collection complete; dataset files ready")
    print_created_files(run_dir)
    run_after_collection(
        run_dir, postprocess["actions"],
        completed=collection_complete and not STOP_REQUESTED and not problems and not validation_errors,
    )


if __name__ == "__main__":
    main()
