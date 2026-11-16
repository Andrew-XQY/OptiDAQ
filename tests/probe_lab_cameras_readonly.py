#!/usr/bin/env python3
"""Read-only camera diagnosis. Run on the CERN VM; reports go to the Desktop.

No SET, reset, power control, motor commands, RBAC login or config edits.
Uses the installed PyDA/RDA3 SDK directly, independently of the app adapter.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
import threading
import time
import traceback
from typing import Any
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
np = None


def error_info(exc: BaseException) -> dict:
    return {"type": type(exc).__name__, "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))}


def plain(value: Any) -> Any:
    if hasattr(value, "value"):
        value = value.value
    if getattr(value, "shape", None) == () and hasattr(value, "_value"):
        value = value._value
    if hasattr(value, "item") and getattr(value, "shape", None) in ((), None):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


def safe(value: Any) -> Any:
    value = plain(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): ("<redacted>" if any(s in str(k).lower() for s in
                ("password", "token", "secret", "credential")) else safe(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if np is not None and hasattr(value, "shape"):
        arr = np.asarray(value)
        if arr.size <= 256:
            return safe(arr.tolist())
        result = {"shape": list(arr.shape), "dtype": str(arr.dtype), "size": int(arr.size),
                  "sha256": hashlib.sha256(arr.tobytes()).hexdigest()}
        if arr.size and np.issubdtype(arr.dtype, np.number):
            result.update(min=float(arr.min()), max=float(arr.max()), mean=float(arr.mean()))
        return result
    return str(value)


def bounded_call(function, timeout_s: float) -> dict:
    outcome = {}
    done = threading.Event()
    started = time.monotonic()
    def run():
        try:
            outcome["value"] = function()
        except BaseException as exc:
            outcome["error"] = error_info(exc)
        finally:
            done.set()
    threading.Thread(target=run, daemon=True, name="ProbeRead").start()
    if not done.wait(timeout_s):
        return {"error": {"type": "Timeout", "message": "Read timed out after %.2fs" % timeout_s},
                "elapsed_s": timeout_s}
    return {**outcome, "elapsed_s": round(time.monotonic() - started, 3)}


def header_info(response) -> dict:
    header = getattr(response, "header", None)
    return {key: safe(getattr(header, key)) for key in
            ("selector", "acquisition_timestamp", "cycle_timestamp", "first_update", "is_first_update")
            if hasattr(header, key)}


def response_info(response) -> dict:
    if getattr(response, "exception", None) is not None:
        return {"error": error_info(response.exception), "header": header_info(response)}
    return {"header": header_info(response), "data": safe(dict(response.data))}


def read_property(client, endpoint: str, timeout_s: float, selector=None) -> dict:
    result = bounded_call(
        lambda: client.get(endpoint, context=selector) if selector else client.get(endpoint), timeout_s,
    )
    if "value" in result:
        result["response"] = response_info(result.pop("value"))
    result["requested_selector"] = selector
    return result


def close_subscription(sub) -> dict:
    for name in ("stop", "cancel", "close", "unsubscribe"):
        closer = getattr(sub, name, None)
        if callable(closer):
            result = bounded_call(closer, 2.0)
            return {"method": name, "ok": "error" not in result,
                    **({"error": result["error"]} if "error" in result else {})}
    return {"ok": False, "error": "No subscription close method found"}


def observe_streams(client, selectors: dict, seconds: float, timeout_s: float = 4.0) -> dict:
    """Start subscriptions explicitly, observe both cameras together, then close."""
    states = {}
    started = time.monotonic()
    for device, selector in selectors.items():
        state = {"requested_selector": selector, "images": 0, "numbers": set(),
                 "errors": Counter(), "error_examples": {}, "first_images": [],
                 "last_images": deque(maxlen=3), "header_selectors": set(),
                 "live_updates_only": True, "_stop": threading.Event(), "_lock": threading.Lock()}
        states[device] = state
        def receive(sub, state=state):
            try:
                for response in sub:
                    if state["_stop"].is_set():
                        break
                    if getattr(response, "exception", None) is not None:
                        message = str(response.exception)
                        code = message.split(":", 1)[0]
                        with state["_lock"]:
                            if not state["_stop"].is_set():
                                state["errors"][code] += 1
                                state["error_examples"].setdefault(code, error_info(response.exception))
                        continue
                    data = dict(response.data)
                    fields = ("imageNumber", "frameNumber", "imageTimeStamp", "imageTimeStampRaw",
                              "acqStamp", "cycleNameGlob", "cycleStampGlob", "cycleNbGlob")
                    item = {"received_elapsed_s": round(time.monotonic() - started, 3),
                            "header": header_info(response),
                            "fields": {k: safe(data[k]) for k in fields if k in data}}
                    if "image2D" in data:
                        arr = np.asarray(plain(data["image2D"]))
                        item["image"] = {"shape": list(arr.shape), "dtype": str(arr.dtype),
                                         "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                                         "min": float(arr.min()) if arr.size else None,
                                         "max": float(arr.max()) if arr.size else None}
                    with state["_lock"]:
                        if state["_stop"].is_set():
                            break
                        state["images"] += 1
                        number = item["fields"].get("imageNumber")
                        if isinstance(number, (str, int, float)):
                            state["numbers"].add(number)
                        state["header_selectors"].add(str(item["header"].get("selector", "")))
                        if len(state["first_images"]) < 3:
                            state["first_images"].append(item)
                        state["last_images"].append(item)
            except BaseException as exc:
                with state["_lock"]:
                    if not state["_stop"].is_set():
                        state["iterator_error"] = error_info(exc)
        try:
            creation = bounded_call(lambda device=device, selector=selector:
                client.subscribe(device + "/LastImage", context=selector, receive_first_updates=False)
                if selector else client.subscribe(device + "/LastImage", receive_first_updates=False), timeout_s)
            if "error" in creation:
                state["start_error"] = creation["error"]
                continue
            sub = creation["value"]
            state["subscription"] = sub
            starter = getattr(sub, "start", None)
            if callable(starter):
                startup = bounded_call(starter, timeout_s)
                if "error" in startup:
                    state["start_error"] = startup["error"]
                    continue
            worker = threading.Thread(target=receive, args=(sub,), daemon=True, name="ProbeStream:" + device)
            state["_thread"] = worker
            worker.start()
        except BaseException as exc:
            state["start_error"] = error_info(exc)
    try:
        time.sleep(max(0, seconds - (time.monotonic() - started)))
    finally:
        for state in states.values():
            state["_stop"].set()
            sub = state.pop("subscription", None)
            if sub is not None:
                state["cleanup"] = close_subscription(sub)
            worker = state.get("_thread")
            if worker is not None:
                worker.join(0.25)
                state["consumer_stopped"] = not worker.is_alive()
    for state in states.values():
        with state["_lock"]:
            state["distinct_image_numbers"] = len(state["numbers"])
            state["counter_advanced"] = state["distinct_image_numbers"] >= 2
    return {device: {key: (sorted(value) if isinstance(value, set) else
                          list(value) if isinstance(value, deque) else
                          dict(value) if isinstance(value, Counter) else value)
                     for key, value in state.items() if not key.startswith("_") and key != "numbers"}
            for device, state in states.items()}


def main(argv=None) -> int:
    global np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/experiment_v1.json")
    parser.add_argument("--output-dir", type=Path, default=Path.home() / "Desktop")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--get-timeout", type=float, default=4.0)
    args = parser.parse_args(argv)
    if not (0 < args.seconds <= 60 and 0 < args.get_timeout <= 15):
        parser.error("--seconds must be 0..60 and --get-timeout 0..15, both greater than zero")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo("Europe/Paris")).strftime("%Y%m%d_%H%M%S")
    base = args.output_dir / ("CLEAR26_camera_probe_" + stamp)
    report = {"started_paris": stamp, "read_only": True, "hardware_sets": 0,
              "python": sys.version, "executable": sys.executable, "host": platform.node(),
              "config_path": str(args.config), "events": [], "devices": {}, "versions": {}}

    def save():
        payload = json.dumps(safe(report), indent=2, allow_nan=False)
        base.with_suffix(".json").write_text(payload + "\n")
        text = ["CLEAR26 read-only camera probe", "No device settings were changed.",
                "JSON report: " + str(base.with_suffix(".json")), ""]
        text.extend(report["events"])
        text.extend(["", "Full diagnostic data:", payload])
        base.with_suffix(".txt").write_text("\n".join(text) + "\n")

    def note(message):
        report["events"].append(message)
        print(message, flush=True)
        save()

    note("Starting read-only probe; reports: " + str(base) + ".json / .txt")
    try:
        for package in ("numpy", "pyda", "pyda-rda3", "pyrbac", "pyccda", "navpy"):
            try:
                report["versions"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                report["versions"][package] = "not installed / unknown distribution name"
        import numpy as numpy_module
        np = numpy_module
        import pyda
        from pyda_rda3 import RdaProvider
        report["sdk_signatures"] = {"get": str(inspect.signature(pyda.SimpleClient.get)),
                                    "subscribe": str(inspect.signature(pyda.SimpleClient.subscribe))}
        client = pyda.SimpleClient(provider=RdaProvider())
        cfg = json.loads(args.config.read_text())
        camera_map = {a: cfg["camera_setting"][a]["device"] for a in cfg["cameras"]}
        report["config"] = {"cameras": cfg["cameras"], "tap": cfg.get("tap"),
                            "writer": cfg.get("writer"), "sync": cfg.get("sync"),
                            "camera_setting": {a: cfg["camera_setting"][a] for a in camera_map}}
        report["source_hashes"] = {}
        for rel in ("experiments/camera_preview.py", "src/optidaq/extensions/cern_pyda.py"):
            path = REPO / rel
            if path.exists():
                report["source_hashes"][rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        report["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
        ccda = None
        try:
            import pyccda
            ccda = pyccda.SyncAPI()
        except Exception as exc:
            report["ccda_unavailable"] = error_info(exc)
        names = list(camera_map.values()) + ["CamTest1", "CamTest2", "Test.BI.Cam1.DigiCam", "Test.BI.Cam2.DigiCam"]
        report["catalog"] = {}
        if ccda is not None:
            for name in dict.fromkeys(names):
                note("Read catalog: " + name)
                def metadata(name=name):
                    d = ccda.Device.find(name_or_alias=name)
                    info = d.device_class_info
                    return {"name": str(d.name), "timing_domain": str(d.timing_domain),
                            "cycle_bound": bool(d.is_cycle_bound), "ppm": bool(d.is_ppm),
                            "server": str(d.server_name), "class": str(info.name), "version": str(info.version)}
                report["catalog"][name] = bounded_call(metadata, args.get_timeout)
        selectors = {}
        properties = ("AcquisitionStatus", "HardwareStatus", "CameraSetting", "Setting",
                      "AcquiredParameters", "ExpertSetting", "TriggerConfiguration")
        for alias, device in camera_map.items():
            entry = {"alias": alias, "properties_before": {}}
            report["devices"][device] = entry
            for prop in properties:
                note("Read " + device + "/" + prop)
                entry["properties_before"][prop] = read_property(client, device + "/" + prop, args.get_timeout)
            note("Test plain image GET: " + device)
            entry["plain_image_get"] = read_property(client, device + "/LastImage", args.get_timeout)
            meta = report["catalog"].get(device, {}).get("value", {})
            domain = meta.get("timing_domain")
            selectors[device] = (domain + ".USER.ALL" if meta.get("cycle_bound") and domain not in (None, "NONE")
                                 else cfg.get("tap", {}).get("subscribe_selector"))
            if selectors[device] is None and device.startswith("BTVDC_865_BTVDIGITAL_"):
                selectors[device] = "SPS.USER.ALL"
                entry["selector_note"] = "SPS fallback from previously verified lab-device metadata; catalog unavailable or inconclusive"
        note("Observe image subscriptions together for %.1f seconds; no settings writes" % args.seconds)
        report["streams"] = observe_streams(client, selectors, args.seconds, args.get_timeout)
        for device, stream in report["streams"].items():
            note("%s: %d updates, %d distinct image numbers, counter_advanced=%s" % (
                device, stream["images"], stream["distinct_image_numbers"], stream["counter_advanced"]))
            entry = report["devices"][device]
            candidates = []
            for key in ("AcquisitionStatus",):
                response = entry["properties_before"][key].get("response", {})
                user = response.get("data", {}).get("cycleNameGlob")
                if isinstance(user, str) and ".USER." in user and not user.endswith(".ALL"):
                    candidates.append(user)
            candidates = list(dict.fromkeys(stream["header_selectors"] + candidates))
            entry["selected_image_gets"] = {}
            for selector in [s for s in candidates if ".USER." in s and not s.endswith(".ALL")][:2]:
                note("Test selected image GET: " + device + " " + selector)
                entry["selected_image_gets"][selector] = read_property(
                    client, device + "/LastImage", args.get_timeout, selector)
            entry["status_after"] = read_property(client, device + "/AcquisitionStatus", args.get_timeout)
        report["completed"] = True
        note("Finished. Send the .json report back; the .txt contains the same data in readable form.")
        return 0
    except KeyboardInterrupt:
        report["interrupted"] = True
        note("Interrupted; partial report was saved.")
        return 130
    except BaseException as exc:
        report["probe_error"] = error_info(exc)
        note("Probe error; details and completed reads were saved: " + str(exc))
        return 1
    finally:
        save()


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Exit this diagnostic process even if an SDK GET worker remains blocked.
    os._exit(exit_code)
