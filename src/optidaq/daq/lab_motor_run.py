"""Lab-only point-motor acquisition, using the existing PNG storage contract."""
from __future__ import annotations

from copy import deepcopy
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import time

from optidaq.actuators.motor_point_raster import MotorPointRasterController, validate_motor_raster_config
from optidaq.daq.contracts import Frame
from optidaq.daq.lab_capture import (
    FreshPairGate, QUALITY, metadata, proxy_ns, validate_capture_config, validate_exposure_guard,
)
from optidaq.daq.run_store import build_raw_spec, create_run_dir, write_resolved_config
from optidaq.daq.save_backends import make_save_backend
from optidaq.data.lab_png_finalize import finalize_lab_png_dataset
from optidaq.extensions.lab_camera import (
    build_lab_taps, make_lab_client, read_camera_settings, verify_external_settings,
)


class Journal:
    def __init__(self, path):
        self.handle = Path(path).open("x", encoding="utf-8")

    def write(self, record, *, durable=False):
        self.handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self.handle.flush()
        if durable:
            os.fsync(self.handle.fileno())

    def close(self):
        try:
            self.handle.flush()
            os.fsync(self.handle.fileno())
        finally:
            self.handle.close()


def validate_motor_lab_config(cfg):
    validate_motor_raster_config(cfg)
    validate_capture_config(cfg)
    if cfg.get("output", {}).get("format") != "png" or cfg["output"]["png"].get("mode") != "raw16":
        raise ValueError("lab motor raster requires PNG raw16 storage")
    if cfg["output"]["png"].get("pairing") != "capture_id":
        raise ValueError("lab motor raster requires output.png.pairing=capture_id")
    if not cfg.get("rbac", {}).get("enabled"):
        raise ValueError("motor motion requires RBAC authentication")
    devices = [cfg["camera_setting"][cam]["device"] for cam in cfg["cameras"]]
    if len(set(devices)) != len(devices):
        raise ValueError("camera aliases must identify different physical cameras")


def collect_points(controller, gate, backends, run_dir, cfg, check, *, clock=time, on_commit=lambda: None):
    """Commit one pair per point. Kept independent of FESA for lifecycle tests."""
    resources = ExitStack()
    completed = 0
    try:
        events = Journal(run_dir / "capture.events.jsonl")
        resources.callback(events.close)
        commits = Journal(run_dir / "capture_commits.jsonl")
        resources.callback(commits.close)
        for point in controller.plan:
            check()
            events.write({"type": "change_begin", "step_id": point.index, "host_time_ns": clock.time_ns()})
            info = controller.apply_point(point)
            check()
            request = gate.begin(point.index, info)
            capture = request.metadata()
            events.write({"type": "ready", **capture})
            pair, selected_mono = gate.wait_pair(cfg["scan"]["point_timeout_s"], check)
            capture["selection_monotonic_ns"] = selected_mono
            capture["selection_host_time_ns"] = clock.time_ns()
            # A second motor readback confirms the held position before any PNG
            # becomes a committed sample. It is identical in both sidecars.
            capture["actuator_held_readbacks"] = controller.verify_held(point)
            check()
            saved = []
            for cam in gate.aliases:
                source = pair[cam]
                frame = Frame(source.device, source.source_image_number, source.acq_stamp_ns,
                              source.payload_bytes, source.pulse_id)
                sync = deepcopy(metadata(source))
                sync.update(capture=deepcopy(capture), chosen_sync_source="capture_id",
                            chosen_sync_value=capture["capture_id"])
                object.__setattr__(frame, "sync_metadata", sync)
                backend = backends[cam]
                index = backend.frame_count
                backend.save(frame)
                backend.flush()
                ack = {"camera": cam, "frame_index": index, "imageNumber": frame.source_image_number,
                       "png_path": "%s/%020d_%08d.png" % (cam, frame.acq_stamp_ns, index),
                       "host_time_ns": clock.time_ns()}
                events.write({"type": "save_ack", "capture_id": capture["capture_id"], **ack})
                saved.append(ack)
            check()
            held = controller.verify_held(point)
            commit = {"type": "capture_commit", **capture, "saved": saved,
                      "post_save_readbacks": held, "host_time_ns": clock.time_ns(),
                      "proxy_span_ns": abs(proxy_ns(pair[gate.aliases[0]]) - proxy_ns(pair[gate.aliases[1]]))}
            commits.write(commit, durable=True)
            completed += 1
            on_commit()
            target = info["target_counters"]
            print("point %d/%d saved: x=%s y=%s capture=%s" % (
                completed, len(controller.plan), target["x"], target["y"], capture["capture_id"]), flush=True)
        return completed
    finally:
        gate.end()
        resources.close()


def run_motor_raster(cfg, config_path, *, should_stop=lambda: False):
    """Configure once, continuously listen, and select only after motor readiness."""
    validate_motor_lab_config(cfg)
    from optidaq.extensions.cern_pyda import (
        camera_map_from_config, collect_camera_source_provenance, configure_cameras,
        ensure_cameras_alive, jsonable,
    )

    cfg = deepcopy(cfg)
    client = make_lab_client(require_rbac=True)
    camera_map = camera_map_from_config(cfg)
    aliases = list(camera_map)
    gate = FreshPairGate(aliases, cfg["capture"])
    taps, started_taps, backends, opened = [], [], {}, []
    run_dir, before, after, error = None, {}, {}, None
    completed = 0
    acquisition_complete = False

    def record_commit():
        nonlocal completed
        completed += 1

    def check():
        if should_stop():
            raise KeyboardInterrupt("collection stopped")
        gate.raise_if_failed()
        for tap in taps:
            if tap.error is not None:
                raise RuntimeError("camera %s failed: %s" % (tap.device, tap.error)) from tap.error

    def cancelled():
        return should_stop() or gate.error is not None or any(tap.error is not None for tap in taps)

    controller = MotorPointRasterController(client, cfg, should_stop=cancelled)
    try:
        # Range and motor health checks precede camera configuration or motion.
        controller.preflight()
        check()
        root = Path(cfg["run"]["data_root"]) / cfg["run"]["runs_subdir"]
        root.mkdir(parents=True, exist_ok=True)
        if cfg["writer"].get("configure_cameras", True):
            ensure_cameras_alive(client, camera_map, cfg["writer"])
            configure_cameras(client, camera_map, cfg["camera_setting"],
                              get_selector=cfg["tap"].get("get_selector"), use_subscription=True,
                              subscribe_selector=cfg["tap"]["subscribe_selector"])
        before = read_camera_settings(client, camera_map)
        verify_external_settings(before)
        validate_exposure_guard(cfg["capture"], before)
        provenance = collect_camera_source_provenance(client, camera_map)
        taps = build_lab_taps(client, cfg)
        for tap in taps:
            tap.probe()
        estimated = sum(tap.payload_nbytes for tap in taps) * len(controller.plan)
        if shutil.disk_usage(root).free < estimated * cfg["run"].get("free_disk_margin", 2):
            raise RuntimeError("insufficient space for planned lab PNG dataset")
        run_dir = create_run_dir(root)
        print("run_dir=%s; %d motor positions, one pair per position" % (run_dir, len(controller.plan)), flush=True)
        write_resolved_config(run_dir, {
            "config_path": str(config_path), "config": cfg, "script": "experiment_lab_v1.py",
            "cameras": {tap.device: {"fesa": tap.fesa_device, "shape": list(tap.image_shape),
                "dtype": tap.image_dtype, "payload_nbytes": tap.payload_nbytes,
                "cameraPixelFormat": "mono12"} for tap in taps},
            "camera_setting_readback": before, "camera_source_provenance": jsonable(provenance),
            "actuator": controller.actuator_metadata(), "motors": controller.motor_metadata(),
            "scan_plan": [point.to_dict() for point in controller.plan],
            "points_total": len(controller.plan), "output_format": "png", "png_mode": "raw16",
            "capture_contract": QUALITY, "pairing": "capture_id",
        })
        for tap in taps:
            spec = build_raw_spec(file_prefix=tap.device, payload_nbytes=tap.payload_nbytes,
                                  segment_max_bytes=cfg["writer"].get("segment_max_bytes", 536870912))
            backend = make_save_backend(tap.device, run_dir, spec, image_shape=tap.image_shape,
                                        image_dtype=tap.image_dtype, config=cfg)
            backend.open()
            opened.append(backend)
            backends[tap.device] = backend
        for tap in taps:
            started_taps.append(tap)
            tap.start(gate.receive)
        gate.wait_initial(cfg["scan"]["point_timeout_s"], check)
        controller.prepare()
        collect_points(controller, gate, backends, run_dir, cfg, check, on_commit=record_commit)
        check()
        acquisition_complete = completed == len(controller.plan)
        if not acquisition_complete:
            raise RuntimeError("motor raster ended without committing every planned point")
    except BaseException as exc:
        error = exc
    finally:
        gate.end()
        for tap in started_taps:
            try:
                tap.stop()
                tap.join(3.0)
                if tap._thread is not None and tap._thread.is_alive():
                    raise RuntimeError("camera %s subscription did not close" % tap.device)
                if tap.error is not None:
                    raise RuntimeError("camera %s: %s" % (tap.device, tap.error))
            except BaseException as exc:
                error = error or exc
        try:
            gate.raise_if_failed()
        except BaseException as exc:
            error = error or exc
        for backend in reversed(opened):
            try:
                backend.close()
            except BaseException as exc:
                error = error or exc
        if before:
            try:
                after = read_camera_settings(client, camera_map)
                verify_external_settings(before, after)
            except BaseException as exc:
                error = error or exc
        try:
            controller.cleanup(success=acquisition_complete and error is None and not should_stop())
        except BaseException as exc:
            error = error or exc

    if run_dir is None:
        if error is not None:
            raise error
        raise RuntimeError("no lab run was created")
    # On failure, retain all image facts but do not publish an apparently complete
    # database or advance the scan. No automatic retry can relabel an orphan PNG.
    if error is None and not should_stop():
        try:
            result = finalize_lab_png_dataset(run_dir, cameras=aliases)
            print("PNG metadata database: %s" % result.db_path)
        except BaseException as exc:
            error = exc
    elif error is None:
        error = KeyboardInterrupt("collection stopped before finalization")
    summary = {"ok": error is None, "error": None if error is None else "%s: %s" % (type(error).__name__, error),
               "points_requested": len(controller.plan), "points_committed": completed,
               "capture": gate.stats(), "camera_settings_after": after,
               "actuator": controller.actuator_metadata(), "pairing": "capture_id", "quality": QUALITY}
    with (run_dir / "run_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    if error is not None:
        raise error
    print("collection complete: %d paired motor samples" % completed)
    return run_dir
