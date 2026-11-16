#!/usr/bin/env python3
"""One-camera, read-only test. Run in acc-py; reports go to ~/Desktop.

Only AcquisitionStatus/HardwareStatus GET and LastImage SUB are used.
No experiment imports, configuration writes, SET, reset or RBAC login.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo


def scalar(value):
    value = getattr(value, "value", value)
    if hasattr(value, "item"):
        value = value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("BTVDC_865_BTVDIGITAL_1", "BTVDC_865_BTVDIGITAL_2"),
                        default="BTVDC_865_BTVDIGITAL_1")
    parser.add_argument("--seconds", type=int, choices=range(1, 31), default=6)
    parser.add_argument("--stdout-only", action="store_true", help="Save via SSH capture instead of VM Desktop")
    args = parser.parse_args()
    stamp = datetime.now(ZoneInfo("Europe/Paris"))
    report = {"started": stamp.isoformat(), "device": args.device, "hardware_sets": 0,
              "completed": False, "steps": {}}
    path = Path.home() / "Desktop" / ("CLEAR26_camera_step1_" + stamp.strftime("%Y%m%d_%H%M%S") + ".json")

    def save():
        if not args.stdout_only:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + "\n")

    def finish(signum, frame):
        report["error"] = "Probe exceeded its 40-second limit"
        save()
        print(json.dumps(report, indent=2), flush=True)
        os._exit(2)

    signal.signal(signal.SIGALRM, finish)
    signal.alarm(40)
    save()
    try:
        import pyda
        from pyda_rda3 import RdaProvider
        client = pyda.SimpleClient(provider=RdaProvider())

        def read(prop, fields):
            result = {}
            def get():
                try:
                    response = client.get(args.device + "/" + prop)
                    if response.exception is not None:
                        raise response.exception
                    result["data"] = {key: scalar(response.data[key]) for key in fields if key in response.data}
                    result["selector"] = str(getattr(response.header, "selector", ""))
                except Exception as exc:
                    result["error"] = str(exc)
            worker = threading.Thread(target=get, daemon=True)
            worker.start()
            worker.join(4)
            return {"error": "GET timed out after 4 seconds"} if worker.is_alive() else result

        status_fields = ("cameraName", "acqCameraStatus", "acqStatus", "acqMsg",
                         "acqImageFrameRateActual", "acqLastImageNumber", "cycleNameGlob")
        report["steps"]["status_before"] = read("AcquisitionStatus", status_fields)
        report["steps"]["identity"] = read("HardwareStatus", (
            "cameraName", "modelName", "serialNumber", "ipAddress", "macAddress"))
        save()
        stream = {"selector": "SPS.USER.ALL", "cached_first_updates": False,
                  "seconds": args.seconds, "live_images": 0, "samples": [], "errors": []}
        report["steps"]["live_stream"] = stream
        iterator = None
        try:
            subscription = client.subscribe(args.device + "/LastImage", context="SPS.USER.ALL",
                                            receive_first_updates=False)
            iterator = iter(subscription)  # PyDA starts monitoring here.
            deadline = time.monotonic() + args.seconds
            while time.monotonic() < deadline:
                try:
                    response = iterator.next(timeout=min(0.5, max(0.001, deadline - time.monotonic())))
                except Exception as exc:
                    if type(exc).__name__ == "SubscriptionTimeoutError":
                        continue
                    raise
                if response.exception is not None:
                    message = str(response.exception)
                    if message not in stream["errors"]:
                        stream["errors"].append(message)
                    continue
                stream["live_images"] += 1
                if len(stream["samples"]) < 5:
                    sample = {key: scalar(response.data[key]) for key in
                              ("imageNumber", "frameNumber", "imageTimeStamp", "cycleNameGlob")
                              if key in response.data}
                    sample["selector"] = str(getattr(response.header, "selector", ""))
                    pixels = getattr(response.data.get("image2D"), "value", response.data.get("image2D"))
                    sample["shape"] = list(getattr(pixels, "shape", ()))
                    stream["samples"].append(sample)
        except Exception as exc:
            stream["errors"].append(str(exc))
        finally:
            if iterator is not None:
                del iterator  # Release the SDK iterator; its finalizer stops monitoring.
                gc.collect()
        save()
        report["steps"]["status_after"] = read("AcquisitionStatus", status_fields)
        report["completed"] = True
    except Exception as exc:
        report["error"] = str(exc)
    report["finished"] = datetime.now(ZoneInfo("Europe/Paris")).isoformat()
    save()
    print(json.dumps(report, indent=2), flush=True)
    if not args.stdout_only:
        print("Saved:", path, flush=True)
    signal.alarm(0)
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    # Bound process lifetime even if an SDK background GET thread remains blocked.
    os._exit(main())
