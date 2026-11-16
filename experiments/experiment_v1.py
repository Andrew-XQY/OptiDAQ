"""Acquire raw camera streams while running a motor, magnet, or noop scan.

Run from the repository root in the CERN acc-py environment:

    python experiments/experiment_v1.py --config configs/experiment_v1.json
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping as MappingABC
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from optidaq.actuators.motor_row_sweep import (  # noqa: E402
    MotorRowSweepController,
    ScanLine,
)
from optidaq.actuators.magnet_current_scan import (  # noqa: E402
    MagnetCurrentScanController,
    MagnetPoint,
    make_magnet_current_plan,
)
from optidaq.actuators.magnet_random_walk import (  # noqa: E402
    MagnetRandomWalkController,
    validate_magnet_random_walk_config,
)
from optidaq.actuators.magnet_raster_scan import (  # noqa: E402
    MagnetRasterScanController,
    MagnetRasterScanTunedController,
    validate_magnet_raster_config,
    validate_magnet_raster_tuned_config,
)
from optidaq.actuators.noop import (  # noqa: E402
    NoopActuatorController,
    NoopPoint,
    make_noop_plan,
)
from optidaq.daq.contracts import Problem  # noqa: E402
from optidaq.daq.engine import run_engine  # noqa: E402
from optidaq.daq.magnet_logger import MagnetCurrentLogger  # noqa: E402
from optidaq.daq.offline_validate import (  # noqa: E402
    build_configured_aligned_index,
    read_records_by_device,
    read_sync_sidecars,
    rewrite_sync_sidecars_with_selection,
    validate_device_stream,
    write_aligned_index_jsonl,
)
from optidaq.daq.pipeline import DAQPipeline  # noqa: E402
from optidaq.daq.preflight import make_write_access_check, run_preflight  # noqa: E402
from optidaq.daq.run_store import (  # noqa: E402
    build_raw_spec,
    create_run_dir,
    write_resolved_config,
)
from optidaq.daq.save_backends import RawStreamSaveBackend  # noqa: E402
from optidaq.daq.sync_tools import (  # noqa: E402
    pulse_id_candidate_fields,
    tap_capture_mode,
    tap_poll_interval_s,
    validate_poll_rate,
    validate_tap_stamp_sources,
)
from optidaq.daq.watchdog import CounterFreezeWatchdog  # noqa: E402
from optidaq.extensions.cern_pyda import (  # noqa: E402
    PollingCameraTap,
    SubscribingCameraTap,
    camera_map_from_config,
    collect_camera_source_provenance,
    configure_cameras,
    ensure_cameras_alive,
    jsonable,
    make_client,
)


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "experiment_v1.json"
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
        mode = actuator.get("mode", "motor")
    else:
        raise RuntimeError("actuator must be an object or string")

    mode = str(mode).lower()
    if mode in ("motor", "motor_row_sweep", "motors"):
        return "motor"
    if mode in ("magnet", "magnets", "magnet_random_walk", "magnet_walk", "magnet_brownian"):
        return "magnet_random_walk"
    if mode in ("magnet_current_scan", "magnet_scan", "magnet_list"):
        return "magnet_current_scan"
    if mode in ("magnet_raster_scan_tuned", "magnet_raster_tuned", "raster_tuned"):
        return "magnet_raster_scan_tuned"
    if mode in (
        "magnet_raster",
        "magnet_raster_scan",
        "magnet_grid",
        "raster",
        "magnet_corrector_raster",
    ):
        return "magnet_raster_scan"
    if mode in ("none", "noop", "no_op", "dummy", "external", "external_actuator"):
        return "noop"
    if mode in ("dmd", "dmd_pattern_scan"):
        raise RuntimeError(
            "DMD acquisition requires experiments/experiment_lab_v1.py"
        )
    raise RuntimeError("unknown actuator.mode %r" % mode)


def validate_config_minimal(cfg: Mapping[str, Any]) -> None:
    mode = actuator_mode(cfg)
    output = cfg.get("output", {})
    if not isinstance(output, MappingABC):
        raise RuntimeError("output must be an object")
    if str(output.get("format", "raw")).lower() != "raw":
        raise RuntimeError(
            "experiment_v1 requires output.format='raw'; "
            "use experiments/experiment_lab_v1.py for PNG acquisition"
        )
    postprocess_config(cfg)
    for section in [
        "run",
        "rbac",
        "cameras",
        "tap",
        "scan",
        "writer",
        "camera_setting",
        "sync",
    ]:
        if section not in cfg:
            raise RuntimeError("config missing section %r" % section)
    if not cfg["cameras"]:
        raise RuntimeError("config cameras must not be empty")
    if float(cfg["scan"]["trigger_hz"]) <= 0:
        raise RuntimeError("scan.trigger_hz must be positive")
    if int(cfg["scan"]["frames_per_point"]) <= 0:
        raise RuntimeError("scan.frames_per_point must be positive")
    if mode == "motor":
        for section in ["motors", "reference", "prepare", "end_processes"]:
            if section not in cfg:
                raise RuntimeError("config missing section %r" % section)
        if not cfg["motors"]["x"]["enabled"]:
            raise RuntimeError("x motor must be enabled for experiment_v1")
        if int(cfg["scan"].get("nx", 1)) <= 0 or int(cfg["scan"]["ny"]) <= 0:
            raise RuntimeError("scan nx/ny must be positive")
        if float(cfg["scan"]["width"]) < 0 or float(cfg["scan"]["height"]) < 0:
            raise RuntimeError("scan width/height must be non-negative")
    elif mode == "magnet_current_scan":
        if "magnets" not in cfg:
            raise RuntimeError("config missing section 'magnets'")
        make_magnet_current_plan(cfg["magnets"])
    elif mode == "magnet_random_walk":
        if "magnets" not in cfg:
            raise RuntimeError("config missing section 'magnets'")
        validate_magnet_random_walk_config(cfg["magnets"])
    elif mode == "magnet_raster_scan":
        if "magnets" not in cfg:
            raise RuntimeError("config missing section 'magnets'")
        validate_magnet_raster_config(cfg["magnets"])
    elif mode == "magnet_raster_scan_tuned":
        if "magnets" not in cfg:
            raise RuntimeError("config missing section 'magnets'")
        validate_magnet_raster_tuned_config(cfg["magnets"])
    elif mode == "noop":
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
        if isinstance(item, ScanLine):
            return "row %d" % item.row
        return "point %d" % int(getattr(item, "index", len(self.point_log)))


def make_controller(client: Any, cfg: Mapping[str, Any]) -> Any:
    mode = actuator_mode(cfg)
    if mode == "motor":
        return MotorRowSweepController(client, cfg)
    if mode == "magnet_current_scan":
        return MagnetCurrentScanController(client, cfg)
    if mode == "magnet_random_walk":
        return MagnetRandomWalkController(client, cfg)
    if mode == "magnet_raster_scan":
        return MagnetRasterScanController(client, cfg)
    if mode == "magnet_raster_scan_tuned":
        return MagnetRasterScanTunedController(client, cfg)
    if mode == "noop":
        return NoopActuatorController(client, cfg)
    raise RuntimeError("unknown actuator mode %r" % mode)


def postprocess_config(cfg: Mapping[str, Any]) -> dict:
    postprocess = cfg.get("postprocess", {})
    if not isinstance(postprocess, Mapping):
        raise RuntimeError("postprocess must be an object")
    if postprocess.get("actions", []) != []:
        raise RuntimeError(
            "postprocess.actions requires experiments/experiment_lab_v1.py; "
            "omit actions or set it to [] for raw acquisition"
        )
    return {
        "build_aligned_index": bool(postprocess.get("build_aligned_index", True)),
        "validate_after_scan": bool(postprocess.get("validate_after_scan", True)),
    }


def magnet_current_log_config(cfg: Mapping[str, Any]) -> dict:
    """Read optional magnet-current logging settings; active only for magnet controllers."""
    magnets = cfg.get("magnets", {})
    block = magnets.get("current_log", {}) if isinstance(magnets, Mapping) else {}
    if not isinstance(block, Mapping):
        raise RuntimeError("magnets.current_log must be an object")
    period_s = float(block.get("period_s", 0.1))
    if period_s <= 0:
        raise RuntimeError("magnets.current_log.period_s must be positive")
    # Optional explicit subset of magnet aliases to log. Omit or [] => all
    # configured magnets.devices. Validated against the real device set later.
    selected = block.get("magnets")
    if selected is not None:
        if not isinstance(selected, (list, tuple)) or not all(
            isinstance(alias, str) for alias in selected
        ):
            raise RuntimeError(
                "magnets.current_log.magnets must be a list of alias strings"
            )
        selected = [
            alias for alias in selected if alias and not alias.startswith("_")
        ]
        if not selected:
            selected = None
    return {
        "enabled": bool(block.get("enabled", False)),
        "period_s": period_s,
        "magnets": selected,
    }


def maybe_launch_quad_view(
    cfg: Mapping[str, Any],
    config_path: Path,
    run_dir: Path,
    magnet_active: bool,
) -> Optional["subprocess.Popen"]:
    """Launch the file-based viewer when enabled for a magnet scan with a display."""
    view = cfg.get("quad_scan_view", {})
    if not (isinstance(view, Mapping) and view.get("enabled") and magnet_active):
        return None
    cmd = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "quad_scan_view.py"),
        "--config", str(config_path),
        "--run", str(run_dir),
    ]
    if not os.environ.get("DISPLAY"):
        print("quad_scan_view enabled but no DISPLAY; run it where you have one:")
        print("  %s" % " ".join(cmd))
        return None
    try:
        print("quad_scan_view: launching live viewer (decoupled process)")
        return subprocess.Popen(cmd)
    except Exception as exc:  # never let the viewer affect the run
        print("WARNING: quad scan view launch failed: %s" % exc)
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    cfg = load_config(args.config)
    camera_map = camera_map_from_config(cfg)
    aliases = list(camera_map)
    postprocess = postprocess_config(cfg)
    build_aligned_index = bool(postprocess["build_aligned_index"])
    validate_after_scan = bool(postprocess["validate_after_scan"])
    sync_cfg = cfg["sync"]
    pulse_id_fields = pulse_id_candidate_fields(sync_cfg)
    trigger_period_s = 1.0 / float(cfg["scan"]["trigger_hz"])
    trigger_period_ns = int(trigger_period_s * 1_000_000_000)
    problems: list[Problem] = []

    client = make_client(require_rbac=bool(cfg["rbac"]["enabled"]))

    # tap.mode selects how frames are captured:
    #   "poll"      -> PollingCameraTap: re-read LastImage every last_image_poll_s
    #   "subscribe" -> SubscribingCameraTap: FESA pushes each new image (event-driven)
    tap_mode = tap_capture_mode(cfg["tap"])
    use_subscribe = tap_mode == "subscribe"
    common_tap_kwargs = dict(
        poll_s=tap_poll_interval_s(cfg["tap"], required=not use_subscribe),
        stamp_fields=cfg["tap"]["stamp_fields"],
        min_ns_epoch=int(cfg["tap"]["min_ns_epoch"]),
        pulse_id_fields=pulse_id_fields,
        settle_poll_s=float(cfg["tap"].get("settle_poll_s", 0.005)),
        settle_timeout_s=float(cfg["tap"].get("settle_timeout_s", 0.25)),
        max_trailing_zero_rows=cfg["tap"].get("max_trailing_zero_rows", 0),
        sync_cfg=sync_cfg,
        get_selector=cfg["tap"].get("get_selector"),
    )

    def _build_tap(alias: str, fesa: str):
        if use_subscribe:
            return SubscribingCameraTap(
                alias,
                fesa,
                client,
                selector=cfg["tap"].get("subscribe_selector"),
                **common_tap_kwargs,
            )
        return PollingCameraTap(alias, fesa, client, **common_tap_kwargs)

    print("camera tap mode: %s" % ("subscribe" if use_subscribe else "poll"))
    taps = [_build_tap(alias, fesa) for alias, fesa in camera_map.items()]

    camera_setting_readback = {}
    if bool(cfg["writer"]["configure_cameras"]):
        ensure_cameras_alive(client, camera_map, cfg["writer"])
        camera_setting_readback = configure_cameras(
            client,
            camera_map,
            cfg["camera_setting"],
            get_selector=cfg["tap"].get("get_selector"),
            use_subscription=use_subscribe,
            subscribe_selector=cfg["tap"].get("subscribe_selector"),
        )
    camera_source_provenance = collect_camera_source_provenance(
        client,
        camera_map,
    )

    for tap in taps:
        tap.probe()
        print(
            "camera %s fesa=%s shape=%s dtype=%s payload_nbytes=%s "
            "stamp_source=%s probe_pulse_id=%s"
            % (
                tap.device,
                tap.fesa_device,
                tap.image_shape,
                tap.image_dtype,
                tap.payload_nbytes,
                tap.stamp_source,
                tap.probe_pulse_id,
            )
        )
    validate_tap_stamp_sources(taps, sync_cfg)

    controller = make_controller(client, cfg)
    controller.preflight()
    points_total = len(controller.plan)
    frames_per_point = int(controller.frames_per_point)
    if not controller.plan:
        raise RuntimeError("scan plan is empty")

    # Decoupled magnet-current logging: only in magnet modes (controller owns a
    # magnet set) and only when explicitly enabled in config. Off => no-op.
    magnet_set = getattr(controller, "magnet_set", None)
    magnet_log_cfg = magnet_current_log_config(cfg)
    magnet_log_active = bool(magnet_log_cfg["enabled"]) and magnet_set is not None
    magnet_log_aliases = None
    if magnet_log_active:
        # Explicit subset if given, else every configured magnet. Validate now so
        # a misspelled alias fails at startup, not silently mid-run.
        magnet_log_aliases = magnet_log_cfg["magnets"] or list(magnet_set.devices)
        unknown = sorted(set(magnet_log_aliases) - set(magnet_set.devices))
        if unknown:
            raise RuntimeError(
                "magnets.current_log.magnets has unknown alias(es): %s "
                "(known: %s)"
                % (", ".join(unknown), ", ".join(magnet_set.devices))
            )

    root = Path(cfg["run"]["data_root"]) / str(cfg["run"]["runs_subdir"])
    preflight_results = run_preflight([make_write_access_check(root)])
    failed = [result for result in preflight_results if not result.ok]
    if failed:
        raise RuntimeError(
            "preflight failed: %s"
            % "; ".join("%s: %s" % (r.name, r.message) for r in failed)
        )

    estimated_bytes = sum(int(tap.payload_nbytes or 0) for tap in taps)
    estimated_bytes *= points_total * frames_per_point
    free_bytes = shutil.disk_usage(root).free
    required_bytes = int(estimated_bytes * float(cfg["run"]["free_disk_margin"]))
    print(
        "disk estimate=%.3f MB required_with_margin=%.3f MB free=%.3f MB"
        % (
            estimated_bytes / (1024.0 * 1024.0),
            required_bytes / (1024.0 * 1024.0),
            free_bytes / (1024.0 * 1024.0),
        )
    )
    if free_bytes < required_bytes:
        raise RuntimeError("not enough free disk space under %s" % root)

    run_dir = create_run_dir(root)
    specs = {
        tap.device: build_raw_spec(
            file_prefix=tap.device,
            payload_nbytes=int(tap.payload_nbytes or 0),
            segment_max_bytes=int(cfg["writer"]["segment_max_bytes"]),
        )
        for tap in taps
    }

    write_resolved_config(
        run_dir,
        {
            "config_path": str(args.config),
            "config": jsonable(cfg),
            "cameras": {
                tap.device: {
                    "fesa": tap.fesa_device,
                    "shape": list(tap.image_shape or ()),
                    "dtype": tap.image_dtype,
                    "payload_nbytes": tap.payload_nbytes,
                    "stamp_source": tap.stamp_source,
                    # Record the requested pixel format so downstream parsing knows the real
                    # bit depth (mono8 = 8-bit) instead of guessing from the int16 wire dtype.
                    "cameraPixelFormat": cfg.get("camera_setting", {})
                    .get(tap.device, {})
                    .get("cameraPixelFormat"),
                }
                for tap in taps
            },
            "camera_setting_readback": jsonable(camera_setting_readback),
            "camera_source_provenance": jsonable(camera_source_provenance),
            "actuator": jsonable(controller.actuator_metadata()),
            "motors": jsonable(controller.motor_metadata()),
            "magnets": jsonable(controller.magnet_metadata()),
            "scan_plan": [point.to_dict() for point in controller.plan],
            "output_format": "raw",
            "postprocess": jsonable(postprocess),
            "magnet_current_log": {
                "active": magnet_log_active,
                "period_s": magnet_log_cfg["period_s"],
                "magnets": magnet_log_aliases,
                "file": "magnets.samples.jsonl" if magnet_log_active else None,
            },
            "sync": jsonable(sync_cfg),
            "trigger_period_ns": trigger_period_ns,
            "points_total": points_total,
            "script": Path(__file__).name,
        },
    )
    print("run_dir=%s" % run_dir)

    magnet_logger = None
    if magnet_log_active:
        magnet_logger = MagnetCurrentLogger(
            run_dir / "magnets.samples.jsonl",
            lambda: magnet_set.read_sample(magnet_log_aliases),
            period_s=float(magnet_log_cfg["period_s"]),
            on_problem=lambda message: print("WARNING: %s" % message, flush=True),
        )

    viewer_proc = maybe_launch_quad_view(
        cfg, args.config, run_dir, magnet_set is not None
    )

    fsync_every = cfg["writer"].get("fsync_every_records")
    fsync_every = int(fsync_every) if fsync_every else None

    backends = {
        tap.device: RawStreamSaveBackend(
            tap.device, run_dir, specs[tap.device],
            fsync_every_records=fsync_every,
        )
        for tap in taps
    }
    pipeline = DAQPipeline(
        taps=taps,
        backends=backends,
        queue_maxsize=int(cfg["writer"]["queue_maxsize"]),
        on_problem=problems.append,
    )
    watchdog = CounterFreezeWatchdog(
        pipeline.stats,
        aliases,
        problems.append,
        freeze_timeout_s=max(10.0, 5.0 * trigger_period_s),
        poll_interval_s=1.0,
    )

    point_log: list = []

    def raise_if_failed_all() -> None:
        pipeline.raise_if_failed()
        for tap in taps:
            if tap.error is not None:
                raise RuntimeError(
                    "camera tap %s failed: %s" % (tap.device, tap.error)
                ) from tap.error

    frame_waiter = FrameWaiter(
        pipeline=pipeline,
        aliases=aliases,
        frames_per_point=frames_per_point,
        timeout_s=float(cfg["scan"]["point_timeout_s"]),
        point_log=point_log,
        raise_if_failed=raise_if_failed_all,
        should_stop=lambda: STOP_REQUESTED,
    )

    progress = PlanProgress(points_total)

    def apply_with_runtime(item: Any) -> None:
        if isinstance(controller, MotorRowSweepController):
            if not isinstance(item, ScanLine):
                raise RuntimeError("motor row-sweep plan item must be a ScanLine")
            controller.begin_line(item)
            markers = frame_waiter.markers()
            controller.sweep_line(item)
            frame_waiter.wait(markers, item)
            return
        if isinstance(controller, MagnetCurrentScanController):
            if not isinstance(item, MagnetPoint):
                raise RuntimeError("magnet current-scan plan item must be a MagnetPoint")
            readback = controller.apply_point(item)
            markers = frame_waiter.markers()
            frame_waiter.wait(markers, item, extra={"magnet_readback": readback})
            return
        if isinstance(controller, NoopActuatorController):
            if not isinstance(item, NoopPoint):
                raise RuntimeError("noop plan item must be a NoopPoint")
            controller.apply(item)
            markers = frame_waiter.markers()
            frame_waiter.wait(markers, item)
            return
        controller.apply(item)

    def dwell_with_progress() -> None:
        controller.dwell()
        progress.advance()

    pipeline_started = False
    watchdog_started = False
    primary_error: Optional[BaseException] = None
    t0 = time.time()

    try:
        controller.prepare()
        pipeline.start()
        pipeline_started = True
        watchdog.start()
        watchdog_started = True
        if magnet_logger is not None:
            magnet_logger.start()
            print(
                "magnet current log: period_s=%.3f magnets=%s -> %s"
                % (
                    magnet_logger.period_s,
                    ",".join(magnet_log_aliases),
                    magnet_logger.path.name,
                ),
                flush=True,
            )
        run_engine(
            controller.plan,
            apply_with_runtime,
            dwell_with_progress,
            abort=controller.abort,
            should_stop=lambda: STOP_REQUESTED,
        )
    except BaseException as exc:
        primary_error = exc
    finally:
        if watchdog_started:
            watchdog.stop()

        if magnet_logger is not None:
            magnet_logger.stop()

        if viewer_proc is not None and viewer_proc.poll() is None:
            viewer_proc.terminate()

        for tap in taps:
            if tap.error is not None:
                print("camera tap %s error: %s" % (tap.device, tap.error))
                problems.append(
                    Problem(
                        source="camera_tap.%s" % tap.device,
                        message=str(tap.error),
                        severity="error",
                        exception=tap.error,
                    )
                )
                if primary_error is None:
                    primary_error = RuntimeError("camera tap %s failed" % tap.device)

        # Stop/finalize DAQ before return_to_start so return-motion frames are
        # not mixed into the scan dataset.
        if pipeline_started:
            try:
                pipeline.stop(float(cfg["writer"]["drain_timeout_s"]))
            except BaseException as exc:
                print("WARNING: pipeline stop failed: %s" % exc)
                problems.append(
                    Problem(
                        source="pipeline.stop",
                        message=str(exc),
                        severity="error",
                        exception=exc,
                    )
                )
                if primary_error is None:
                    primary_error = exc

        try:
            controller.cleanup()
        except BaseException as exc:
            print("WARNING: actuator cleanup failed: %s" % exc)
            problems.append(
                Problem(
                    source="actuator.cleanup",
                    message=str(exc),
                    severity="error",
                    exception=exc,
                )
            )
            if primary_error is None:
                primary_error = exc

        controller.abort()

    elapsed = time.time() - t0
    print("scan elapsed %.1f s; points_done=%d/%d" % (
        elapsed,
        len(point_log),
        points_total,
    ))

    if magnet_logger is not None:
        summary = "magnet current log: samples=%d file=%s" % (
            magnet_logger.sample_count,
            magnet_logger.path.name,
        )
        if magnet_logger.error is not None:
            summary += " last_error=%s" % magnet_logger.error
        print(summary)

    if primary_error is not None:
        print("finished with problems: %s" % primary_error)
        print_created_files(run_dir)
        raise primary_error

    summaries = {}
    validation_errors = False
    aligned_entries = []
    invalid_entries = []
    sync_selection_report = None
    if validate_after_scan:
        for alias, spec in specs.items():
            summary = validate_device_stream(run_dir, spec, strict=False)
            summaries[alias] = summary
            validation_errors = validation_errors or bool(summary["errors"])

    if build_aligned_index:
        per_device_records = read_records_by_device(run_dir, specs)
        sync_records_by_device = read_sync_sidecars(run_dir, aliases)
        configured_index = build_configured_aligned_index(
            per_device_records,
            aliases,
            sync_cfg,
            sync_records_by_device,
        )
        aligned_entries = configured_index["entries"]
        sync_selection_report = configured_index["sync_report"]
        if sync_selection_report.get("selected_source"):
            rewrite_sync_sidecars_with_selection(
                run_dir,
                aliases,
                str(sync_selection_report["selected_source"]),
            )
        write_aligned_index_jsonl(run_dir, aligned_entries)
        invalid_entries = [
            entry for entry in aligned_entries if not bool(entry["valid"])
        ]
    complete_clusters = sum(bool(entry["valid"]) for entry in aligned_entries)

    if validate_after_scan:
        print("\nvalidation summaries:")
        for alias in aliases:
            summary = summaries[alias]
            print(
                "%s records=%d segments=%d errors=%d warnings=%d"
                % (
                    alias,
                    summary["n_records"],
                    summary["n_segments"],
                    len(summary["errors"]),
                    len(summary["warnings"]),
                )
            )
            for error in summary["errors"]:
                print("  ERROR: %s" % error)
            for warning in summary["warnings"]:
                print("  WARNING: %s" % warning)
    else:
        print("\nvalidation skipped by postprocess.validate_after_scan=false")

    print("\ntimestamp sources:")
    for tap in taps:
        print("  %s stamp_source=%s" % (tap.device, tap.stamp_source))

    if build_aligned_index:
        print("\nsync selection:")
        if sync_selection_report is None:
            print("  no sync selection report")
        else:
            selected_source = sync_selection_report.get("selected_source")
            selected_report = sync_selection_report.get("selected_report") or {}
            print(
                "  selected=%s passed=%s fallback_used=%s"
                % (
                    selected_source,
                    sync_selection_report.get("passed"),
                    sync_selection_report.get("fallback_used"),
                )
            )
            for reason in sync_selection_report.get("fallback_reasons", []):
                print("  fallback: %s" % reason)
            if selected_report:
                stats = selected_report.get("span_stats", {})
                print(
                    "  source=%s groups_valid=%s invalid=%s span_ns min=%s max=%s "
                    "mean=%s drift_ns=%s"
                    % (
                        selected_report.get("source"),
                        selected_report.get("valid_groups"),
                        selected_report.get("invalid_groups"),
                        stats.get("min"),
                        stats.get("max"),
                        stats.get("mean"),
                        selected_report.get("drift_ns"),
                    )
                )
                for reason in selected_report.get("reasons", []):
                    print("  ERROR: %s" % reason)
        print(
            "aligned_valid=%d invalid=%d"
            % (complete_clusters, len(invalid_entries))
        )
        for entry in invalid_entries[:5]:
            print(
                "  INVALID offline_shot_id=%d key=%s errors=%s"
                % (
                    entry["offline_shot_id"],
                    entry["sync_key"],
                    "; ".join(entry["errors"]),
                )
            )
    else:
        print("\naligned_index build skipped by postprocess.build_aligned_index=false")
    print_created_files(run_dir)
    expected_aligned = points_total * int(cfg["scan"]["frames_per_point"])
    aligned_errors = (
        build_aligned_index
        and (complete_clusters < expected_aligned or bool(invalid_entries))
    )
    sync_errors = (
        build_aligned_index
        and (
            sync_selection_report is None
            or not bool(sync_selection_report.get("passed", False))
        )
    )
    collection_complete = not STOP_REQUESTED and len(point_log) == points_total
    if not collection_complete:
        print("collection stopped before all planned points finished")
    elif problems or validation_errors or aligned_errors or sync_errors:
        print("finished with problems")
    else:
        print("collection complete; dataset files ready")
    if sync_errors:
        reasons = []
        if sync_selection_report is not None:
            reasons.extend(sync_selection_report.get("fallback_reasons", []))
            selected_report = sync_selection_report.get("selected_report") or {}
            reasons.extend(selected_report.get("reasons", []))
        raise RuntimeError(
            "sync validation failed: %s"
            % ("; ".join(str(reason) for reason in reasons) or "no configured source passed")
        )
    if aligned_errors:
        raise RuntimeError("aligned index validation failed")


if __name__ == "__main__":
    main()
