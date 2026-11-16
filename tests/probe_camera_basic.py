#!/usr/bin/env python3
"""Basic BTVDC camera check, following skills/CERN_FESA_cameras.md. Read-only.

Run on the CERN VM in acc-py. Steps, as in the notes' recommended workflow:
  1. catalog entry (CCDA): class version, server, timing domain, selector flags
  2. identify the camera (HardwareStatus)
  3. health over time (AcquisitionStatus, sampled for all devices together)
  4. applied settings (AcquiredParameters, CameraSetting, Setting)
  5. image reads (LastImageURI, LastImage): plain, then with the device's
     current cycle user
Checks the two new lab devices, the June 2026 865.1.c17 devices, and any
--device names given (for example the device the CERN GUI shows).

GET only: no SET, no subscriptions, no RBAC login, no repository imports.
Reports go to ~/Desktop/CLEAR26_camera_basic_<time>.json and .txt.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

NEW_DEVICES = ("BTVDC_865_BTVDIGITAL_1", "BTVDC_865_BTVDIGITAL_2")
JUNE_DEVICES = ("865.1.c17.BI.Cam1.DigiCam", "865.1.c17.BI.Cam2.DigiCam",
                "865.1.c17.BI.Cam3.DigiCam")

# Enum values from skills/CERN_FESA_cameras.md (BTVDC 6.8.0 schema).
TRIGGERS = {0: "TRIG_NONE", 1: "TRIG_SOFTWARE", 2: "TRIG_LINE1", 3: "TRIG_LINE3"}
ON_OFF = {0: "OFF", 1: "ON"}
CONNECTED = {0: "NOT_CONNECTED", 1: "CONNECTED"}
ENUMS = {
    "acqCameraStatus": {-1: "MISSING", 0: "OK", 1: "ERROR", 2: "UNREACHABLE", 3: "REBOOTING",
                        4: "DISCONNECTED", 5: "SYNCHRONIZING_PARAMETERS"},
    "acqStatus": {0: "OK", 1: "NO_IMG_RECEIVED", 2: "ACQ_RATE_LESS_THAN_10Hz",
                  3: "NOTIFICATION_ERROR"},
    "acqTrigger": TRIGGERS,
    "acqAcqTrigger": TRIGGERS,
    "powerSwitchType": {0: "UNKNOWN", 1: "RUCKUS", 2: "GUDE", 3: "BTVCTRL", 4: "BTVI",
                        5: "BReleino"},
    "cameraSwitchStatus": ON_OFF,
    "ethernetSwitchStatus": ON_OFF,
    "cameraSwitchConnectionStatus": CONNECTED,
    "ethernetSwitchConnectionStatus": CONNECTED,
}

HARDWARE_FIELDS = ("cameraName", "modelName", "vendorName", "serialNumber", "ipAddress",
                   "macAddress", "powerSwitchName", "powerSwitchType", "powerSwitchPortNb",
                   "cameraSwitchStatus", "cameraSwitchConnectionStatus",
                   "ethernetSwitchStatus", "ethernetSwitchConnectionStatus",
                   "acqCameraStatus", "maxFrameWidth", "maxFrameHeight")
STATUS_FIELDS = ("cameraName", "acqCameraStatus", "acqStatus", "acqMsg", "acqLastImageNumber",
                 "acqLastImageAcqStamp", "acqImageFrameRateActual", "acqNbOfErrors",
                 "acqNbOfCommErrors", "acqNbOfReboots", "acqNbOfRebootAttempts",
                 "lastCommErrorTimeStampAcqStampStr", "lastCameraRebootAcqStampStr",
                 "cycleNameGlob")
APPLIED_FIELDS = ("acqAcqTrigger", "acqCameraExposureTimeUs", "acqCameraGain",
                  "acqCameraFrameRate", "acqCameraPixelFormat", "acqCameraFrameEnable",
                  "acqImageROIEnable", "acqImageROI")
CAMERA_SETTING_FIELDS = ("acqTrigger", "cameraExposureTimeUs", "cameraGain", "cameraFrameRate",
                         "cameraPixelFormat")
SETTING_FIELDS = ("imageROIEnable", "imageROI", "imageWindow")
IMAGE_FIELDS = ("imageNumber", "frameNumber", "imageTimeStamp", "cycleNameGlob", "width",
                "height")
URI_FIELDS = ("imageNumber", "remoteURL", "localURI", "width", "height", "cycleNameGlob")


def plain(value):
    # Unwrap PyDA scalar wrappers (see skills/pyda_fesa_working_notes.md).
    if hasattr(value, "value"):
        value = value.value
    shape = getattr(value, "shape", None)
    if shape == () and hasattr(value, "_value"):
        value = value._value
    if hasattr(value, "item") and shape in ((), None):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


def jsonable(value):
    value = plain(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist") and getattr(value, "size", 0) <= 16:
        return jsonable(value.tolist())
    return str(value)


def pick(data, fields):
    out = {}
    for key in fields:
        if key in data:
            out[key] = jsonable(data[key])
            names = ENUMS.get(key)
            if names is not None and isinstance(out[key], int):
                out[key + "_name"] = names.get(out[key], "unknown")
    return out


def error_text(exc):
    return "%s: %s" % (type(exc).__name__, exc)


def bounded(function, timeout_s):
    """Run one SDK call in a thread so a hung reply cannot stall the probe."""
    box = {}

    def run():
        try:
            box["value"] = function()
        except BaseException as exc:
            box["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise TimeoutError("no reply within %.0f s" % timeout_s)
    if "error" in box:
        raise box["error"]
    return box["value"]


def get(client, endpoint, selector=None, timeout_s=4.0):
    def call():
        if selector:
            return client.get(endpoint, context=selector)
        return client.get(endpoint)

    response = bounded(call, timeout_s)
    if response.exception is not None:
        raise response.exception
    return response.data, str(getattr(response.header, "selector", ""))


def read(client, endpoint, fields, selector=None):
    entry = {"endpoint": endpoint, "selector": selector,
             "time": datetime.now(ZoneInfo("Europe/Paris")).strftime("%H:%M:%S.%f")[:-3]}
    try:
        data, header_selector = get(client, endpoint, selector)
        entry["header_selector"] = header_selector
        entry["data"] = pick(data, fields)
    except Exception as exc:
        entry["error"] = error_text(exc)
    return entry


def read_image(client, np, device, selector):
    entry = read(client, device + "/LastImageURI", URI_FIELDS, selector)
    entry = {"uri": entry, "selector": selector}
    try:
        data, header_selector = get(client, device + "/LastImage", selector, timeout_s=8.0)
        entry["header_selector"] = header_selector
        entry["data"] = pick(data, IMAGE_FIELDS)
        if "image2D" in data:
            image = np.asarray(plain(data["image2D"]))
            entry["image"] = {"shape": list(image.shape), "dtype": str(image.dtype),
                              "min": float(image.min()) if image.size else None,
                              "max": float(image.max()) if image.size else None}
    except Exception as exc:
        entry["error"] = error_text(exc)
    return entry


def property_flags(device):
    """Best effort: per-property flags from CCDA. The pyccda layout is not verified here."""
    try:
        device_class = bounded(
            lambda: device.device_class() if callable(device.device_class) else device.device_class,
            6.0,
        )
        props = None
        for attr in ("properties", "fesa_class_properties", "device_class_properties"):
            props = getattr(device_class, attr, None)
            if props is not None:
                break
        if props is None:
            return {"error": "no property list found",
                    "class_attributes": sorted(a for a in dir(device_class) if not a.startswith("_"))}
        if callable(props):
            props = bounded(props, 6.0)
        flags = {}
        for prop in props:
            name = str(getattr(prop, "name", prop))
            flags[name] = {key: jsonable(getattr(prop, key)) for key in
                           ("is_cycle_bound", "is_multiplexed", "is_subscribable")
                           if hasattr(prop, key)}
        return flags
    except Exception as exc:
        return {"error": error_text(exc)}


def catalog(ccda, name):
    try:
        device = bounded(lambda: ccda.Device.find(name_or_alias=name), 6.0)
        info = device.device_class_info
        entry = {"name": str(device.name), "class": str(info.name), "version": str(info.version),
                 "server": str(device.server_name), "timing_domain": str(device.timing_domain),
                 "cycle_bound": bool(device.is_cycle_bound), "ppm": bool(device.is_ppm)}
    except Exception as exc:
        return {"error": error_text(exc)}
    entry["property_flags"] = property_flags(device)
    return entry


def verdict(entry, span_s):
    samples = [s["data"] for s in entry.get("status_samples", []) if "data" in s]
    if not samples:
        return "NO STATUS REPLY: the device server did not answer from this VM"
    last = samples[-1]
    status = last.get("acqCameraStatus")
    if status != 0:
        return ("CAMERA NOT USABLE BY ITS SERVER: acqCameraStatus=%s (%s). This is the "
                "camera/power/network side; image reads cannot work until it is OK."
                % (status, last.get("acqCameraStatus_name")))
    numbers = [s.get("acqLastImageNumber") for s in samples]
    if len(set(numbers)) < 2:
        return ("CAMERA CONNECTED, NO NEW IMAGES in %.0f s (image number stays %s): "
                "check the trigger source and acqTrigger." % (span_s, numbers[-1]))
    if not any("image" in r for r in entry.get("image_reads", [])):
        return ("IMAGES ARRIVING BUT LastImage GET FAILED: client-side read/selector "
                "problem, see image_reads.")
    return "OK: camera connected, images arriving, LastImage readable."


def describe(device, entry, span_s):
    lines = ["== %s ==" % device]
    cat = entry.get("catalog", {})
    if "error" in cat:
        lines.append("catalog: %s" % cat["error"].splitlines()[0])
    elif cat:
        lines.append("catalog: %s %s on %s, timing %s, cycle_bound=%s, ppm=%s" % (
            cat["class"], cat["version"], cat["server"], cat["timing_domain"],
            cat["cycle_bound"], cat["ppm"]))
    if entry.get("skipped"):
        lines.append("live reads skipped: %s" % entry["skipped"])
        return lines
    hw = entry["hardware"].get("data")
    if hw:
        lines.append("camera: name=%r model=%r ip=%r mac=%r serial=%r max=%sx%s" % (
            hw.get("cameraName"), hw.get("modelName"), hw.get("ipAddress"),
            hw.get("macAddress"), hw.get("serialNumber"), hw.get("maxFrameWidth"),
            hw.get("maxFrameHeight")))
        lines.append("power switch: %s %s port %s | camera port %s/%s | ethernet port %s/%s" % (
            hw.get("powerSwitchName"), hw.get("powerSwitchType_name"), hw.get("powerSwitchPortNb"),
            hw.get("cameraSwitchStatus_name"), hw.get("cameraSwitchConnectionStatus_name"),
            hw.get("ethernetSwitchStatus_name"), hw.get("ethernetSwitchConnectionStatus_name")))
    else:
        lines.append("HardwareStatus: %s" % entry["hardware"].get("error"))
    samples = [s["data"] for s in entry["status_samples"] if "data" in s]
    if samples:
        first, last = samples[0], samples[-1]
        lines.append("status: acqCameraStatus=%s (%s), acqStatus=%s (%s), image number %s -> %s "
                     "over %.0f s, rate %s Hz" % (
                         last.get("acqCameraStatus"), last.get("acqCameraStatus_name"),
                         last.get("acqStatus"), last.get("acqStatus_name"),
                         first.get("acqLastImageNumber"), last.get("acqLastImageNumber"),
                         span_s, last.get("acqImageFrameRateActual")))
        lines.append("message: %s" % last.get("acqMsg"))
        lines.append("cycle user now: %r" % last.get("cycleNameGlob"))
    applied = entry["applied"].get("data")
    if applied:
        lines.append("applied: trigger=%s exposure=%s us gain=%s rate=%s roi=%s %s" % (
            applied.get("acqAcqTrigger_name"), applied.get("acqCameraExposureTimeUs"),
            applied.get("acqCameraGain"), applied.get("acqCameraFrameRate"),
            applied.get("acqImageROIEnable"), applied.get("acqImageROI")))
    for item in entry.get("image_reads", []):
        label = item["selector"] or "no selector"
        if "image" in item:
            result = "image %s %s, imageNumber=%s" % (
                item["image"]["shape"], item["image"]["dtype"],
                item["data"].get("imageNumber"))
        else:
            result = (item.get("error") or "no image2D field").splitlines()[0]
        lines.append("LastImage (%s): %s" % (label, result))
    lines.append("VERDICT: %s" % entry.get("verdict", "not reached"))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", action="append", default=[],
                        help="extra device to check, e.g. the one the CERN GUI shows (repeatable)")
    parser.add_argument("--samples", type=int, default=6,
                        help="AcquisitionStatus reads per device (1-30, default 6)")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="seconds between status reads (0.2-10, default 2)")
    args = parser.parse_args()
    if not (1 <= args.samples <= 30 and 0.2 <= args.interval <= 10):
        parser.error("--samples must be 1-30 and --interval 0.2-10")

    stamp = datetime.now(ZoneInfo("Europe/Paris"))
    base = Path.home() / "Desktop" / ("CLEAR26_camera_basic_" + stamp.strftime("%Y%m%d_%H%M%S"))
    devices = list(dict.fromkeys(list(NEW_DEVICES) + list(JUNE_DEVICES) + args.device))
    report = {"started": stamp.isoformat(), "read_only": True, "hardware_sets": 0,
              "devices_checked": devices, "devices": {}, "completed": False}
    summary = []

    def save():
        base.parent.mkdir(parents=True, exist_ok=True)
        base.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        text = ["CLEAR26 basic camera probe (read-only, no settings changed)",
                "started %s" % report["started"], ""] + summary
        base.with_suffix(".txt").write_text("\n".join(text) + "\n")

    def time_limit(signum, frame):
        report["error"] = "probe exceeded its 180 s limit; partial report saved"
        save()
        print(report["error"], flush=True)
        os._exit(2)

    signal.signal(signal.SIGALRM, time_limit)
    signal.alarm(180)
    save()
    span_s = 0.0
    try:
        import numpy as np
        import pyda
        from pyda_rda3 import RdaProvider
        client = pyda.SimpleClient(provider=RdaProvider())
        try:
            import pyccda
            ccda = pyccda.SyncAPI()
        except Exception as exc:
            ccda = None
            report["ccda_unavailable"] = error_text(exc)

        live = []
        for device in devices:
            print("checking", device, flush=True)
            entry = {}
            report["devices"][device] = entry
            if ccda is not None:
                entry["catalog"] = catalog(ccda, device)
                if "EntityNotFound" in entry["catalog"].get("error", ""):
                    entry["skipped"] = "device is not in the CCDA catalog"
                    continue
            entry["hardware"] = read(client, device + "/HardwareStatus", HARDWARE_FIELDS)
            entry["applied"] = read(client, device + "/AcquiredParameters", APPLIED_FIELDS)
            entry["camera_setting"] = read(client, device + "/CameraSetting", CAMERA_SETTING_FIELDS)
            entry["setting"] = read(client, device + "/Setting", SETTING_FIELDS)
            entry["status_samples"] = []
            live.append(device)
            save()

        print("sampling AcquisitionStatus: %d reads, %.1f s apart" % (args.samples, args.interval),
              flush=True)
        started = time.monotonic()
        for index in range(args.samples):
            for device in live:
                report["devices"][device]["status_samples"].append(
                    read(client, device + "/AcquisitionStatus", STATUS_FIELDS))
            save()
            if index + 1 < args.samples:
                time.sleep(args.interval)
        span_s = time.monotonic() - started

        for device in live:
            entry = report["devices"][device]
            print("reading images", device, flush=True)
            samples = [s["data"] for s in entry["status_samples"] if "data" in s]
            user = samples[-1].get("cycleNameGlob") if samples else None
            selectors = [None] + ([user] if user else [])
            entry["image_reads"] = [read_image(client, np, device, s) for s in selectors]
            entry["verdict"] = verdict(entry, span_s)
            save()
        report["completed"] = True
    except Exception as exc:
        report["error"] = error_text(exc)
    for device, entry in report["devices"].items():
        try:
            summary.extend(describe(device, entry, span_s) + [""])
        except Exception as exc:
            summary.extend(["== %s ==" % device, "summary failed: %s" % error_text(exc), ""])
    signal.alarm(0)
    save()
    print("\n".join(summary), flush=True)
    if "error" in report:
        print("ERROR:", report["error"], flush=True)
    print("Saved: %s.json and .txt" % base, flush=True)
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    # Exit even if an SDK worker thread is still blocked on a reply.
    os._exit(main())
