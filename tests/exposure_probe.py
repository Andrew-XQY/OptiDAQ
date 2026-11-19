"""Independent, interactive exposure calibration for a held laser/DMD scene.

Run on the lab PC with an already configured, externally triggered camera pair:
    python tests/exposure_probe.py --source laser
    python tests/exposure_probe.py --source dmd --dmd-period-us 40000

The operator blocks/unblocks the source and holds representative scenes. Only
camera exposure is temporarily changed. No motor or DMD commands are issued.
Original exposures are restored; recommendations are never applied automatically.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import signal
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optidaq.extensions.lab_camera import verify_external_settings
from optidaq.analysis.exposure_probe import summarize_exposure, recommend_exposures

ALIASES = ("cam1", "cam2")


def validate_plan(plan):
    if plan["source"] not in ("laser", "dmd"):
        raise ValueError("source must be laser or dmd")
    if type(plan["repeats"]) is not int or plan["repeats"] < 3:
        raise ValueError("at least three repeated frames are required")
    if not 0 < plan["target_low"] < plan["target_high"] < 1:
        raise ValueError("target fractions must satisfy 0 < low < high < 1")
    if not plan["scenes"] or len(set(plan["scenes"])) != len(plan["scenes"]):
        raise ValueError("supply distinct representative scene names")
    if any(not isinstance(s, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", s) for s in plan["scenes"]):
        raise ValueError("scene names must contain only letters, digits, underscores or hyphens")
    ladders = plan["exposures_us"]
    if set(ladders) != set(ALIASES):
        raise ValueError("exposure ladders must specify cam1 and cam2")
    lengths = {len(ladders[c]) for c in ALIASES}
    if len(lengths) != 1 or next(iter(lengths)) < 3:
        raise ValueError("use equal-length ladders with at least three exposures per camera")
    for cam, ladder in ladders.items():
        if any(type(v) is not int or not 0 < v <= 2_000_000 for v in ladder):
            raise ValueError("%s exposures must be integer microseconds in 1..2000000" % cam)
        if any(b <= a for a, b in zip(ladder, ladder[1:])):
            raise ValueError("exposure ladders must increase strictly")
    period = plan.get("dmd_period_us")
    if plan["source"] == "dmd":
        if type(period) is not int or period <= 0:
            raise ValueError("DMD mode requires the actual --dmd-period-us")
        if any(v % period for ladder in ladders.values() for v in ladder):
            raise ValueError("every DMD exposure must be a whole display period")
    elif period is not None:
        raise ValueError("DMD period is only used in DMD mode")


def write_record(out, record):
    temporary = out / "probe.json.tmp"
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(out / "probe.json")


def roi_from_config(cfg, camera):
    transforms = cfg.get("output", {}).get("png", {}).get("transforms", {}).get(camera, [])
    if transforms and transforms[0].get("name") == "crop_box":
        p = transforms[0]["params"]
        return (p["x1"], p["y1"], p["x2"], p["y2"])
    return None


def report_text(record):
    lines = ["OptiDAQ exposure probe: " + record["status"],
             "Source: %s; scenes: %s" % (record["plan"]["source"], ", ".join(record["plan"]["scenes"])),
             "Original camera settings verified: " + str(record.get("original_settings_verified"))]
    if record.get("error"):
        lines.append(record["error"])
    lines.extend(record.get("capture_cleanup_errors", []))
    lines.extend(record["restore_errors"])
    recommendation = record.get("recommendation")
    if recommendation:
        lines.append("Selection: " + recommendation["status"])
        for cam, result in recommendation["cameras"].items():
            chosen = result["recommended_exposure_us"]
            provisional = result["best_unsaturated_provisional_exposure_us"]
            lines.append("%s (%s): recommended %s; provisional %s" % (
                cam, record["camera_roles"][cam],
                ("%g ms" % (chosen / 1000)) if chosen else "none",
                ("%g ms" % (provisional / 1000)) if provisional else "none"))
            if result["provisional_flags"]:
                lines.append("  Review: " + ", ".join(result["provisional_flags"]))
        norm = recommendation["normalization"]
        lines.append("Fibre/reference exposure ratio: " + str(norm["fibre_to_reference_exposure_ratio"]))
        lines.append("DN/s multipliers: " + json.dumps(norm["recommended_factors_to_counts_per_s"]))
    lines.extend(["", "Normalize EACH camera: (raw - background at the SAME exposure) * 1000000 / exposure_us.",
                  "Raw data are preserved. Gain stays 0. No common exposure ratio is imposed.",
                  record["scope"], "Recommendations apply only to tested scenes; raw brightness alone is not signal quality."])
    if record.get("measurements"):
        lines.extend(["", "camera scene exposure_ms native_max saturated_pixels corrected_p99.9 contrast/noise flags"])
        for cam, points in record["measurements"].items():
            for point in points:
                ratio = point["contrast_over_repeat_background_noise"]
                lines.append("%s %s %g %.0f %d %.1f %s %s" % (
                    cam, point["scene"], point["exposure_us"] / 1000,
                    point["native_max_counts"], point["native_saturated_pixels_total"],
                    point["corrected_signal_p99_9_counts"], "%.1f" % ratio if ratio is not None else "unavailable",
                    ",".join(point["flags"]) or "ok"))
    return "\n".join(lines) + "\n"


def check_initial_settings(snapshot):
    if set(snapshot) != set(ALIASES):
        raise ValueError("probe requires cam1 (fibre) and cam2 (reference)")
    verify_external_settings(snapshot)
    for cam, values in snapshot.items():
        if float(values["AcquiredParameters"]["acqCameraGain"]) != 0:
            raise RuntimeError("%s gain must already be zero; the probe changes exposure only" % cam)


def run_probe(cfg, plan, out, *, hardware, prompt=input):
    """Capture independent ladders, then restore settings even after partial SETs."""
    validate_plan(plan)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    record = {
        "schema_version": 1, "status": "starting", "plan": deepcopy(plan),
        "camera_roles": {"cam1": "fibre", "cam2": "reference"},
        "config": deepcopy(cfg), "captures": [], "restore_errors": [],
        "scope": "Held scenes only; motor motion, optical superposition and reconstruction accuracy are not validated.",
        "dmd_period_source": "operator supplied, not device-read-back" if plan["source"] == "dmd" else None,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "analysis_sha256": hashlib.sha256((ROOT / "src/optidaq/analysis/exposure_probe.py").read_bytes()).hexdigest(),
    }
    before = None
    attempted_write = False
    failure = None
    write_record(out, record)
    try:
        before = hardware.snapshot()
        check_initial_settings(before)
        record["settings_before"] = before
        # Reject the complete plan before the first write, using live limits.
        for cam in ALIASES:
            settings = before[cam]["CameraSetting"]
            low = settings.get("cameraExposureTimeUs_min")
            high = settings.get("cameraExposureTimeUs_max")
            if low is None or high is None:
                raise RuntimeError("%s exposure limits are unavailable" % cam)
            if any(not low <= v <= high for v in plan["exposures_us"][cam]):
                raise ValueError("%s exposure ladder exceeds live camera limits" % cam)
        record["status"] = "capturing"
        write_record(out, record)
        phases = [("background_before", None)] + [("signal", s) for s in plan["scenes"]] + [("background_after", None)]
        for phase_index, (phase, scene) in enumerate(phases):
            if phase == "signal":
                message = "Unblock ONLY the %s source; hold scene '%s' fixed for the whole ladder. Enter to capture, q to stop: " % (plan["source"], scene)
            else:
                message = "Block the %s signal, keep ambient lighting/optics unchanged (%s). DMD black alone does not block a laser. Enter to capture, q to stop: " % (plan["source"], phase)
            if prompt(message).strip().lower() in ("q", "quit", "stop"):
                raise KeyboardInterrupt("operator stopped the probe")
            for step in range(len(plan["exposures_us"]["cam1"])):
                exposures = {cam: plan["exposures_us"][cam][step] for cam in ALIASES}
                record["active_capture"] = {"phase": phase, "scene": scene, "exposures_us": exposures}
                write_record(out, record)
                for cam in ALIASES:
                    # A setter can reach the device and then raise; restore anyway.
                    attempted_write = True
                    hardware.set_exposure(cam, exposures[cam])
                applied = hardware.wait_settings(exposures)
                arrays, metadata = hardware.capture(exposures, plan["repeats"])
                filename = "frames_%02d_%02d.npz" % (phase_index, step)
                np.savez_compressed(out / filename, **arrays)
                record["captures"].append({
                    "phase": phase, "scene": scene, "step": step, "exposures_us": exposures,
                    "file": filename, "sha256": hashlib.sha256((out / filename).read_bytes()).hexdigest(),
                    "settings": applied, "metadata": metadata,
                })
                write_record(out, record)
                print("%s %s: cam1=%g ms cam2=%g ms" % (phase, scene or "", exposures["cam1"] / 1000, exposures["cam2"] / 1000), flush=True)
        points = {cam: [] for cam in ALIASES}
        captures = record["captures"]
        for step in range(len(plan["exposures_us"]["cam1"])):
            row = [c for c in captures if c["step"] == step]
            pre = next(c for c in row if c["phase"] == "background_before")
            post = next(c for c in row if c["phase"] == "background_after")
            with np.load(out / pre["file"], allow_pickle=False) as bg0, np.load(out / post["file"], allow_pickle=False) as bg1:
                for c in row:
                    if c["phase"] != "signal":
                        continue
                    with np.load(out / c["file"], allow_pickle=False) as signal_arrays:
                        for cam in ALIASES:
                            result = summarize_exposure(
                                cam, c["exposures_us"][cam], signal_arrays[cam], bg0[cam], bg1[cam],
                                target_low=plan["target_low"], target_high=plan["target_high"], roi=roi_from_config(cfg, cam),
                            )
                            result["scene"] = c["scene"]
                            points[cam].append(result)
        record["measurements"] = points
        record["recommendation"] = recommend_exposures(points)
        record["status"] = "complete"
    except BaseException as exc:
        failure = exc
        record["status"] = "failed"
        record["error"] = "%s: %s" % (type(exc).__name__, exc)
        record["capture_cleanup_errors"] = getattr(exc, "probe_cleanup_errors", [])
    finally:
        try:
            hardware.close()
        except BaseException as exc:
            record["restore_errors"].append("stream cleanup: %s" % exc)
        if before is not None and attempted_write:
            original = {c: int(before[c]["CameraSetting"]["cameraExposureTimeUs"]) for c in ALIASES}
            for cam in ALIASES:
                try:
                    hardware.set_exposure(cam, original[cam])
                except BaseException as exc:
                    record["restore_errors"].append("%s restore write: %s" % (cam, exc))
            try:
                restored = hardware.wait_settings(original)
                verify_external_settings(before, restored)
                record["settings_after"] = restored
                record["original_settings_verified"] = True
            except BaseException as exc:
                record["restore_errors"].append("restore verification: %s" % exc)
                record["original_settings_verified"] = False
        else:
            record["original_settings_verified"] = None
        if record["restore_errors"]:
            record["status"] = "failed"
        active = record.pop("active_capture", None)
        if failure is not None and active is not None:
            record["error_context"] = active
        write_record(out, record)
        (out / "report.txt").write_text(report_text(record))
    if failure is not None:
        raise failure
    if record["restore_errors"]:
        raise RuntimeError("Camera restoration/cleanup failed; see %s" % (out / "probe.json"))
    return record


class LabHardware:
    """Exposure-only PyDA adapter; each setting has new subscriptions/counters."""
    def __init__(self, cfg):
        from optidaq.extensions.lab_camera import make_lab_client
        from optidaq.extensions.cern_pyda import camera_map_from_config
        self.cfg = deepcopy(cfg)
        self.client = make_lab_client(True)
        self.camera_map = camera_map_from_config(cfg)
        if set(self.camera_map) != set(ALIASES):
            raise ValueError("probe requires exactly cam1 and cam2")
        self.taps = []
        self.initial = None

    def snapshot(self):
        from optidaq.extensions.lab_camera import read_camera_settings
        result = read_camera_settings(self.client, self.camera_map)
        if self.initial is None:
            self.initial = deepcopy(result)
        return result

    def set_exposure(self, camera, exposure_us):
        from optidaq.extensions.cern_pyda import set_value
        if self.taps:
            raise RuntimeError("close subscriptions before changing exposure")
        set_value(self.client, self.camera_map[camera] + "/CameraSetting", {"cameraExposureTimeUs": np.int64(exposure_us)})

    def wait_settings(self, exposures):
        deadline = time.monotonic() + 10
        while True:
            state = self.snapshot()
            if all(state[c]["CameraSetting"]["cameraExposureTimeUs"] == value and state[c]["AcquiredParameters"]["acqCameraExposureTimeUs"] == value for c, value in exposures.items()):
                check_initial_settings(state)
                expected = deepcopy(self.initial)
                for cam, value in exposures.items():
                    expected[cam]["CameraSetting"]["cameraExposureTimeUs"] = value
                    expected[cam]["AcquiredParameters"]["acqCameraExposureTimeUs"] = value
                verify_external_settings(expected, state)
                return state
            if time.monotonic() >= deadline:
                raise TimeoutError("exposure readback did not apply")
            time.sleep(0.1)

    def close(self):
        errors = []
        for tap in self.taps:
            try:
                tap.stop()
            except BaseException as exc:
                errors.append(str(exc))
        remaining = []
        for tap in self.taps:
            try:
                tap.join(3)
                if tap._thread is not None and tap._thread.is_alive():
                    remaining.append(tap)
                    errors.append("%s subscription remains alive" % tap.device)
            except BaseException as exc:
                remaining.append(tap)
                errors.append(str(exc))
        self.taps = remaining
        if errors:
            raise RuntimeError("; ".join(errors))

    def capture(self, exposures, repeats):
        from optidaq.extensions.lab_camera import build_lab_taps
        from optidaq.daq.lab_capture import FreshPairGate, metadata, validate_exposure_guard
        capture_cfg = dict(guard_s=max(2.0, max(exposures.values()) / 1e6), pair_tolerance_ms=1.0, max_frame_age_s=2.0, queue_size=32)
        settings = self.wait_settings(exposures)
        validate_exposure_guard(capture_cfg, settings)
        gate = FreshPairGate(ALIASES, capture_cfg)
        arrays, records = {c: [] for c in ALIASES}, []
        capture_failure = None
        def check():
            gate.raise_if_failed()
            for tap in self.taps:
                if tap.error is not None:
                    raise tap.error
        try:
            for tap in build_lab_taps(self.client, self.cfg):
                self.taps.append(tap)
                tap.probe()
                tap.start(gate.receive)
            gate.wait_initial(30, check)
            for index in range(repeats):
                request = gate.begin(index, {"mode": "held_exposure_probe"})
                pair, selected = gate.wait_pair(30, check)
                row = {"capture": request.metadata(), "selected_monotonic_ns": selected, "frames": {}}
                for tap in self.taps:
                    frame = pair[tap.device]
                    arrays[tap.device].append(np.frombuffer(frame.payload_bytes, dtype=tap.image_dtype).reshape(tap.image_shape).copy())
                    row["frames"][tap.device] = metadata(frame)
                records.append(row)
            check()
            self.wait_settings(exposures)
            return {c: np.stack(v) for c, v in arrays.items()}, {"capture_config": capture_cfg, "pairs": records}
        except BaseException as exc:
            capture_failure = exc
            raise
        finally:
            gate.end()
            try:
                self.close()
            except BaseException as exc:
                if capture_failure is None:
                    raise
                capture_failure.probe_cleanup_errors = ["capture cleanup: %s" % exc]


def build_plan(args):
    period = args.dmd_period_us
    if args.source == "dmd" and (period is None or period <= 0):
        raise ValueError("DMD mode requires --dmd-period-us from the active display settings")
    defaults = {"cam1": [25000, 50000, 100000, 150000, 200000], "cam2": [250, 500, 1000, 1500, 2000]}
    if args.source == "dmd":
        defaults = {"cam1": [period * n for n in (5, 10, 20, 30, 40)], "cam2": [period * n for n in (2, 5, 10, 15, 24)]}
    for cam in ALIASES:
        text = getattr(args, cam + "_us")
        if text:
            defaults[cam] = [int(v.strip()) for v in text.split(",")]
    plan = dict(source=args.source, exposures_us=defaults, scenes=args.scenes.split(","), repeats=args.repeats,
                dmd_period_us=period, target_low=0.70, target_high=0.85)
    validate_plan(plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=("laser", "dmd"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, help="new output directory; default /tmp/clear26_exposure_TIMESTAMP")
    parser.add_argument("--dmd-period-us", type=int, help="actual active grayscale picture period; required for DMD")
    parser.add_argument("--cam1-us", help="increasing comma-separated fibre exposures, microseconds")
    parser.add_argument("--cam2-us", help="increasing comma-separated reference exposures, microseconds")
    parser.add_argument("--scenes", default="bright", help="comma-separated held scene labels; e.g. bright,edge,weak")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    try:
        plan = build_plan(args)
    except ValueError as exc:
        parser.error(str(exc))
    path = args.config or ROOT / "configs" / ("experiment_lab_motor_raster.json" if args.source == "laser" else "experiment_lab_v1.json")
    cfg = json.loads(path.read_text())
    out = args.out or Path("/tmp") / ("clear26_exposure_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    if out.exists():
        parser.error("output directory already exists")
    def stop(*_):
        raise KeyboardInterrupt("probe interrupted; restoring exposures")
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print("Hold the source steady during each ladder. Backgrounds require physically blocking that source.")
    hardware = LabHardware(cfg)
    run_probe(cfg, plan, out, hardware=hardware)
    print("Report: %s" % (out / "report.txt"))


if __name__ == "__main__":
    main()
