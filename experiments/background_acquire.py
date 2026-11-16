"""Acquire background frames with the experiment camera and storage settings.

Run from the repository root in the CERN acc-py environment:

    python experiments/background_acquire.py --config configs/experiment_v1.json --n 100

The background config sets the frame count per camera, timeout and output
subdirectory. This path does not run an actuator or build an aligned index.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping as MappingABC
import json
from pathlib import Path
import shutil
import signal
import sys
import time
from typing import Any, Mapping, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from optidaq.daq.pipeline import DAQPipeline  # noqa: E402
from optidaq.daq.preflight import make_write_access_check, run_preflight  # noqa: E402
from optidaq.daq.run_store import (  # noqa: E402
    build_raw_spec,
    create_run_dir,
    write_resolved_config,
)
from optidaq.daq.save_backends import (  # noqa: E402
    make_save_backend,
    output_format,
    png_mode,
)
from optidaq.data.scientific_png_export import finalize_png_dataset  # noqa: E402
from optidaq.daq.sync_tools import (  # noqa: E402
    pulse_id_candidate_fields,
    tap_capture_mode,
    tap_poll_interval_s,
    validate_poll_rate,
    validate_tap_stamp_sources,
)
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


def request_stop(signum: int, frame: object) -> None:
    global STOP_REQUESTED
    if not STOP_REQUESTED:
        print("\nStop requested; finishing cleanup...", flush=True)
    STOP_REQUESTED = True


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    for section in ("run", "rbac", "cameras", "camera_setting", "tap", "writer", "sync"):
        if section not in cfg:
            raise RuntimeError("config missing section %r" % section)
    if not cfg["cameras"]:
        raise RuntimeError("config cameras must not be empty")
    if int(background_config(cfg)["n_images"]) <= 0:
        raise RuntimeError("background.n_images must be positive")
    # Same poll-rate guard the main DAQ enforces (no-op in subscribe mode).
    if "scan" in cfg:
        validate_poll_rate(cfg["scan"], cfg["tap"], cfg["sync"])
    return cfg


def background_config(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    bg = cfg.get("background")
    if not isinstance(bg, MappingABC):
        raise RuntimeError("config missing section 'background' (object)")
    if "n_images" not in bg:
        raise RuntimeError("background.n_images is required")
    return bg


def wait_for_frames(
    pipeline: DAQPipeline,
    taps: list,
    aliases: list,
    n_images: int,
    timeout_s: float,
) -> dict:
    """Block until every camera has saved ``n_images`` frames, else raise."""

    deadline = time.monotonic() + timeout_s
    last_report = 0.0
    while True:
        if STOP_REQUESTED:
            raise KeyboardInterrupt("stop requested")
        pipeline.raise_if_failed()
        for tap in taps:
            if tap.error is not None:
                raise RuntimeError(
                    "camera tap %s failed: %s" % (tap.device, tap.error)
                ) from tap.error

        saved = pipeline.stats().per_device_saved
        counts = {alias: saved.get(alias, 0) for alias in aliases}
        if all(counts[alias] >= n_images for alias in aliases):
            return counts

        now = time.monotonic()
        if now >= deadline:
            raise RuntimeError(
                "timeout after %.1fs: saved=%s target=%d per camera"
                % (timeout_s, counts, n_images)
            )
        if now - last_report >= 2.0:
            print("progress saved=%s target=%d" % (counts, n_images), flush=True)
            last_report = now
        time.sleep(0.05)


def print_created_files(run_dir: Path) -> None:
    print("\ncreated files:")
    for path in sorted(run_dir.iterdir()):
        if path.is_file():
            size = path.stat().st_size
            name = path.name
        elif path.is_dir():
            size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
            name = "%s/" % path.name
        else:
            continue
        print("  %s  %.3f MB" % (name, size / (1024.0 * 1024.0)))


def main() -> None:
    parser = argparse.ArgumentParser(description="OptiDAQ camera-only background grab")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--n", type=int, default=None, help="override background.n_images")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    cfg = load_config(args.config)
    bg = background_config(cfg)
    n_images = int(args.n) if args.n is not None else int(bg["n_images"])
    if n_images <= 0:
        raise RuntimeError("n_images must be positive")
    timeout_s = float(bg.get("timeout_s", 120.0))
    runs_subdir = str(bg.get("runs_subdir") or cfg["run"]["runs_subdir"])

    camera_map = camera_map_from_config(cfg)
    aliases = list(camera_map)
    selected_output = output_format(cfg)
    selected_png_mode = png_mode(cfg) if selected_output == "png" else None
    sync_cfg = cfg["sync"]
    problems: list = []

    client = make_client(require_rbac=bool(cfg["rbac"]["enabled"]))

    # Build the same camera taps as the main DAQ (poll or subscribe per config).
    tap_mode = tap_capture_mode(cfg["tap"])
    use_subscribe = tap_mode == "subscribe"
    common_tap_kwargs = dict(
        poll_s=tap_poll_interval_s(cfg["tap"], required=not use_subscribe),
        stamp_fields=cfg["tap"]["stamp_fields"],
        min_ns_epoch=int(cfg["tap"]["min_ns_epoch"]),
        pulse_id_fields=pulse_id_candidate_fields(sync_cfg),
        settle_poll_s=float(cfg["tap"].get("settle_poll_s", 0.005)),
        settle_timeout_s=float(cfg["tap"].get("settle_timeout_s", 0.25)),
        max_trailing_zero_rows=cfg["tap"].get("max_trailing_zero_rows", 0),
        sync_cfg=sync_cfg,
        get_selector=cfg["tap"].get("get_selector"),
    )

    def _build_tap(alias: str, fesa: str) -> Any:
        if use_subscribe:
            return SubscribingCameraTap(
                alias,
                fesa,
                client,
                selector=cfg["tap"].get("subscribe_selector"),
                **common_tap_kwargs,
            )
        return PollingCameraTap(alias, fesa, client, **common_tap_kwargs)

    print("background grab: n_images=%d per camera tap_mode=%s output=%s"
          % (n_images, "subscribe" if use_subscribe else "poll", selected_output))
    taps = [_build_tap(alias, fesa) for alias, fesa in camera_map.items()]

    camera_setting_readback = {}
    if bool(cfg["writer"]["configure_cameras"]):
        ensure_cameras_alive(client, camera_map, cfg["writer"])
        camera_setting_readback = configure_cameras(
            client, camera_map, cfg["camera_setting"],
            get_selector=cfg["tap"].get("get_selector"),
            use_subscription=use_subscribe,
            subscribe_selector=cfg["tap"].get("subscribe_selector"),
        )
    camera_source_provenance = collect_camera_source_provenance(client, camera_map)

    for tap in taps:
        tap.probe()
        print(
            "camera %s fesa=%s shape=%s dtype=%s payload_nbytes=%s stamp_source=%s"
            % (
                tap.device,
                tap.fesa_device,
                tap.image_shape,
                tap.image_dtype,
                tap.payload_nbytes,
                tap.stamp_source,
            )
        )
    validate_tap_stamp_sources(taps, sync_cfg)

    root = Path(cfg["run"]["data_root"]) / runs_subdir
    preflight_results = run_preflight([make_write_access_check(root)])
    failed = [result for result in preflight_results if not result.ok]
    if failed:
        raise RuntimeError(
            "preflight failed: %s"
            % "; ".join("%s: %s" % (r.name, r.message) for r in failed)
        )

    estimated_bytes = sum(int(tap.payload_nbytes or 0) for tap in taps) * n_images
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
            "mode": "background_acquire",
            "config_path": str(args.config),
            "config": jsonable(cfg),
            "cameras": {
                tap.device: {
                    "fesa": tap.fesa_device,
                    "shape": list(tap.image_shape or ()),
                    "dtype": tap.image_dtype,
                    "payload_nbytes": tap.payload_nbytes,
                    "stamp_source": tap.stamp_source,
                    "cameraPixelFormat": cfg.get("camera_setting", {})
                    .get(tap.device, {})
                    .get("cameraPixelFormat"),
                }
                for tap in taps
            },
            "camera_setting_readback": jsonable(camera_setting_readback),
            "camera_source_provenance": jsonable(camera_source_provenance),
            "output_format": selected_output,
            "png_mode": selected_png_mode,
            "background": {
                "n_images": n_images,
                "timeout_s": timeout_s,
                "runs_subdir": runs_subdir,
            },
            "script": Path(__file__).name,
        },
    )
    print("run_dir=%s" % run_dir)

    fsync_every = cfg["writer"].get("fsync_every_records")
    fsync_every = int(fsync_every) if fsync_every else None

    backends = {
        tap.device: make_save_backend(
            tap.device, run_dir, specs[tap.device],
            image_shape=tap.image_shape, image_dtype=tap.image_dtype,
            config=cfg, fsync_every_records=fsync_every,
        )
        for tap in taps
    }
    pipeline = DAQPipeline(
        taps=taps,
        backends=backends,
        queue_maxsize=int(cfg["writer"]["queue_maxsize"]),
        on_problem=problems.append,
    )

    primary_error: Optional[BaseException] = None
    counts: dict = {}
    pipeline_started = False
    t0 = time.time()
    try:
        pipeline.start()
        pipeline_started = True
        counts = wait_for_frames(pipeline, taps, aliases, n_images, timeout_s)
    except BaseException as exc:
        primary_error = exc
    finally:
        for tap in taps:
            if tap.error is not None and primary_error is None:
                primary_error = RuntimeError("camera tap %s failed" % tap.device)
        if pipeline_started:
            try:
                pipeline.stop(float(cfg["writer"]["drain_timeout_s"]))
            except BaseException as exc:
                if primary_error is None:
                    primary_error = exc

    elapsed = time.time() - t0
    saved = pipeline.stats().per_device_saved
    final_counts = {alias: saved.get(alias, 0) for alias in aliases}
    print("background elapsed %.1f s; saved=%s" % (elapsed, final_counts))
    print_created_files(run_dir)

    if primary_error is not None:
        print("finished with problems: %s" % primary_error)
        raise primary_error
    if selected_output == "png":
        result = finalize_png_dataset(run_dir, cameras=aliases)
        print("PNG metadata database: %s" % result.db_path)
    if problems:
        print("finished with problems")
    else:
        print("final OK")


if __name__ == "__main__":
    main()
