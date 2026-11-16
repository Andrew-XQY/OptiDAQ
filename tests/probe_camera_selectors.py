#!/usr/bin/env python3
"""Read-only PyDA/RDA3 selector comparison; run in CERN acc-py, reports on Desktop.

No application imports, settings writes, resets, RBAC login, or image pixel export.
GETs are snapshots; only subscriptions with cached first updates disabled count live.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time
import traceback
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
DEVICES = ("BTVDC_865_BTVDIGITAL_1", "BTVDC_865_BTVDIGITAL_2")
OLD_DEVICES = ("Test.BI.Cam1.DigiCam", "Test.BI.Cam2.DigiCam")
PROPERTIES = ("LastImage", "AcquisitionStatus", "HardwareStatus", "CameraSetting", "Setting", "AcquiredParameters")
STATIC_FIELDS = ("cameraName", "cameraAcqTrigger", "enablePublishLastImg", "enableSynchronousSWTrigger", "maxFrameSize", "grabImageTimeoutMs")
IMAGE_FIELDS = ("imageNumber", "frameNumber", "imageTimeStamp", "imageTimeStampRaw", "acqStamp", "cycleNameGlob", "cycleNbGlob", "cycleStampGlob", "width", "height", "pixelFormat")


def error_info(exc):
    return {"type": type(exc).__name__, "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            if isinstance(exc, BaseException) else "unavailable: not an exception object"}


def safe(value, depth=0):
    """Keep metadata/scalars, never array pixels or arbitrary object state."""
    if depth > 24:
        return "<depth limit>"
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(k): safe(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v, depth + 1) for v in value[:128]]
    if getattr(value, "shape", None) == () and hasattr(value, "_value"):
        return safe(value._value, depth + 1)  # PyDA Numeric's scalar payload only.
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        if value.shape == () and hasattr(value, "item"):
            return safe(value.item(), depth + 1)
        return {"shape": list(value.shape), "dtype": str(value.dtype), "pixels": "not saved"}
    if hasattr(value, "value"):
        return safe(value.value, depth + 1)
    return str(value)


def bounded_call(function, timeout_s):
    outcome, done = {}, threading.Event()
    started = time.monotonic()
    def run():
        try:
            outcome["value"] = function()
        except BaseException as exc:
            outcome["error"] = error_info(exc)
        finally:
            done.set()
    threading.Thread(target=run, daemon=True, name="SelectorProbeRead").start()
    if not done.wait(timeout_s):
        return {"error": {"type": "Timeout", "message": "Operation exceeded %.2fs; daemon worker may still be pending" % timeout_s}, "elapsed_s": timeout_s}
    return {**outcome, "elapsed_s": round(time.monotonic() - started, 3)}


def public_attributes(obj, names):
    result = {}
    for name in names:
        try:
            value = getattr(obj, name)
            if not callable(value):
                result[name] = safe(value)
        except AttributeError:
            continue
        except Exception as exc:
            result[name] = {"error": error_info(exc)}
    return result


def schema_field_info(field):
    details = public_attributes(field, ("name", "field_name", "description", "direction", "item_role", "type", "data_type", "value_type", "enum_values", "enumeration", "enum_definition"))
    details["public_attributes"] = [n for n in dir(field) if not n.startswith("_")]
    for attr in ("type", "data_type", "value_type", "enumeration", "enum_definition"):
        candidate = getattr(field, attr, None)
        if candidate is not None and not callable(candidate) and not isinstance(candidate, (str, int, bool, float)):
            definition = public_attributes(candidate, ("name", "description", "values", "items", "enum_values", "enum_items"))
            definition["public_attributes"] = [n for n in dir(candidate) if not n.startswith("_")]
            details[attr + "_definition"] = definition
    if not any(k in details for k in ("type", "data_type", "value_type")):
        details["type_unavailable"] = "Installed public field API does not expose the checked type attributes"
    if not any("enum" in k or k.endswith("_definition") for k in details):
        details["enum_definition_unavailable"] = "No enum definition exposed by the checked public attributes; no numeric meanings inferred"
    return details


def collection(obj, name):
    """Guarded public traversal: installed pyccda collection layout may differ."""
    value = getattr(obj, name)
    value = value() if callable(value) else value
    return list(value.values()) if isinstance(value, dict) else list(value)


def schema_collection(obj, names):
    """Try only named public metadata collections; retain every failed path."""
    failures = {}
    for name in names:
        try:
            return collection(obj, name), name, failures
        except Exception as exc:
            failures[name] = error_info(exc)
    return None, None, failures


def catalog_entry(api, name):
    device = api.Device.find(name_or_alias=name)
    result = public_attributes(device, ("name", "server_name", "timing_domain", "is_cycle_bound", "is_ppm", "description"))
    result["device_class_info"] = public_attributes(device.device_class_info, ("name", "version"))
    try:
        schema = device.device_class()
        result["class_documentation"] = public_attributes(schema, ("name", "version", "description", "documentation", "documentation_url", "framework_version", "major_version", "minor_version", "tiny_version"))
        result["class_public_attributes"] = [n for n in dir(schema) if not n.startswith("_")]
        props, path, failures = schema_collection(schema, ("device_class_properties", "properties"))
        result["property_collection_path"] = "device_class()." + path if path else None
        if failures:
            result["property_collection_failures"] = failures
        if props is None:
            raise AttributeError("No readable device_class_properties or properties collection; see property_collection_failures")
        result["properties"] = {}
        for prop in props:
            prop_name = str(getattr(prop, "name", ""))
            if prop_name not in PROPERTIES:
                continue
            item = public_attributes(prop, ("name", "description", "documentation", "is_readable", "is_writable", "is_subscribable", "is_cycle_bound", "is_ppm", "is_multiplexed", "selector_required"))
            item["property_public_attributes"] = [n for n in dir(prop) if not n.startswith("_")]
            try:
                item["fields"] = []
                paths = (["device_class_property_fields"] if "device_class_property_fields" in item["property_public_attributes"] else []) + ["fields"]
                fields, field_path, field_failures = schema_collection(prop, paths)
                item["field_collection_path"] = field_path
                if field_failures:
                    item["field_collection_failures"] = field_failures
                if fields is None:
                    raise AttributeError("No readable field collection; see field_collection_failures and property_public_attributes")
                for field in fields:
                    item["fields"].append(schema_field_info(field))
            except Exception as exc:
                item["fields_unavailable"] = error_info(exc)
                item["public_attributes"] = item["property_public_attributes"]
            result["properties"][prop_name] = item
        result["properties_not_found"] = [p for p in PROPERTIES if p not in result["properties"]]
    except Exception as exc:
        result["schema_unavailable"] = error_info(exc)
    try:
        binding = api.FesaDevice.from_device(device)
        result["static_fields"] = {}
        for field in collection(binding, "field_values"):
            field_name = str(getattr(field, "field_name", ""))
            if field_name in STATIC_FIELDS:
                result["static_fields"][field_name] = safe(field.field_value)
        result["static_fields_not_found"] = [f for f in STATIC_FIELDS if f not in result["static_fields"]]
    except Exception as exc:
        result["static_fields_unavailable"] = error_info(exc)
    return result


def concrete_selector(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9_]+\.USER\.[A-Za-z0-9_]+", value)) and not value.upper().endswith((".ALL", "_ALL"))


def selector_plan(catalog, tap):
    domain = catalog.get("timing_domain")
    source = "catalog timing_domain"
    if not isinstance(domain, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", domain) or domain.upper() == "NONE":
        value = tap.get("subscribe_selector") or tap.get("get_selector")
        domain = value.split(".USER.")[0] if isinstance(value, str) and ".USER." in value else "SPS"
        source = "configured selector domain" if isinstance(value, str) and ".USER." in value else "SPS fallback: catalog/config domain unavailable"
    selected = next((tap[k] for k in ("subscribe_selector", "get_selector") if concrete_selector(tap.get(k))), None)
    return {"configured": selected, "all": domain + ".USER.ALL", "all_selector_source": source, "requested": {k: tap.get(k) for k in ("get_selector", "subscribe_selector")}}


def header_info(response):
    header = getattr(response, "header", None)
    info = public_attributes(header, ("selector", "acquisition_timestamp", "cycle_timestamp", "first_update", "is_first_update", "notification_type"))
    if hasattr(response, "notification_type"):
        info["response_notification_type"] = safe(response.notification_type)
    info["notification_names"] = [str(getattr(notice, "name", notice))
                                  for obj in (header, response) if hasattr(obj, "notification_type")
                                  for notice in (getattr(obj, "notification_type"),)]
    return info


def response_info(response, image=False):
    header = header_info(response)
    exc = getattr(response, "exception", None)
    if exc is not None:
        return {"header": header, "error": error_info(exc)}
    data = dict(response.data)
    result = {"header": header, "data": {str(k): safe(v) for k, v in data.items() if not image or k in IMAGE_FIELDS}}
    if image and "image2D" in data:
        pixels = getattr(data["image2D"], "value", data["image2D"])
        result["image2D"] = {"shape": list(getattr(pixels, "shape", ())), "dtype": str(getattr(pixels, "dtype", "unavailable")), "pixels": "not saved"}
    return result


def read_property(client, endpoint, timeout_s, selector=None):
    if selector is not None and not concrete_selector(selector):
        return {"requested_selector": selector, "skipped": "GET requires a concrete selector; no ALL/wildcard GET was attempted"}
    def read():
        response = client.get(endpoint, context=selector) if selector else client.get(endpoint)
        return response_info(response, image=endpoint.endswith("/LastImage"))
    return {"requested_selector": selector, "snapshot_only": True, **bounded_call(read, timeout_s)}


def observe_streams(client, plans, seconds, timeout_s):
    """Compare all streams during one common window, using the SDK iterator API."""
    states, workers, ready, gate, stop = {}, [], [], threading.Event(), threading.Event()
    def record_error(state, error):
        state["error_count"] += 1
        if len(state["errors"]) < 5 and error not in state["errors"]:
            state["errors"].append(error)
    def timeout(exc):
        return type(exc).__name__ == "SubscriptionTimeoutError"
    def discard(state, response):
        state["pre_window_discarded"] += 1
        if getattr(response, "exception", None) is not None:
            record_error(state, response_info(response, image=True))
    def consume(device, role, selector, state, prepared):
        subscription = iterator = response = None
        try:
            kwargs = {"receive_first_updates": False}
            if selector is not None:
                kwargs["context"] = selector
            subscription = client.subscribe(device + "/LastImage", **kwargs)
            iterator = iter(subscription)  # Starts monitoring; do not invent start/stop APIs.
            state["setup_ok"] = True
            prepared.set()
            while not gate.is_set() and not stop.is_set():
                try:
                    discard(state, iterator.next(timeout=0.1))
                except Exception as exc:
                    if not timeout(exc):
                        raise
            drain_deadline = time.monotonic() + timeout_s
            while not stop.is_set():
                try:
                    discard(state, iterator.next(timeout=0))
                except Exception as exc:
                    if not timeout(exc):
                        raise
                    state["window_ready"] = True
                    state["measurement_started_elapsed_s"] = round(time.monotonic() - start, 6)
                    break
                if time.monotonic() >= drain_deadline:
                    record_error(state, {"type": "DrainTimeout", "message": "Pre-window queue never emptied; comparison is inconclusive"})
                    return
            if not state["window_ready"]:
                return
            while not stop.is_set():
                try:
                    response = iterator.next(timeout=0.25)
                except Exception as exc:
                    if timeout(exc):
                        continue
                    raise
                if stop.is_set():
                    break
                sample = response_info(response, image=True)
                header = sample["header"]
                if "error" in sample:
                    record_error(state, sample)
                    continue
                notices = " ".join(str(header.get(k, "")) for k in ("notification_type", "response_notification_type", "notification_names")).upper().replace(" ", "_")
                if header.get("first_update") is True or header.get("is_first_update") is True or "FIRST_UPDATE" in notices:
                    state["cached_updates_ignored"] += 1
                    continue
                state["live_updates"] += 1
                header_selector = str(header.get("selector", ""))
                if header_selector not in state["header_selectors"] and len(state["header_selectors"]) < 64:
                    state["header_selectors"].append(header_selector)
                elapsed = time.monotonic() - start
                image = sample.get("image2D", {})
                shape = image.get("shape", [])
                valid = len(shape) == 2 and all(isinstance(n, int) and n > 0 for n in shape) and image.get("dtype") not in (None, "unavailable")
                if valid:
                    state["valid_image_updates"] += 1
                    if header_selector not in state["image_header_selectors"] and len(state["image_header_selectors"]) < 64:
                        state["image_header_selectors"].append(header_selector)
                configured = states[device]["configured"]
                overlap = valid and role == "all" and configured["window_ready"] and elapsed >= configured["measurement_started_elapsed_s"]
                if overlap:
                    state["overlap_valid_image_updates"] += 1
                    if concrete_selector(header_selector) and header_selector not in state["overlap_image_header_selectors"] and len(state["overlap_image_header_selectors"]) < 64:
                        state["overlap_image_header_selectors"].append(header_selector)
                for key in ("imageNumber", "frameNumber", "imageTimeStamp"):
                    value = sample["data"].get(key)
                    if isinstance(value, (str, int, float)):
                        state["_counters"][key].add(value)
                        if valid:
                            state["_image_counters"][key].add(value)
                        if overlap:
                            state["_overlap_counters"][key].add(value)
                if len(state["samples"]) < 3:
                    sample["received_monotonic_s"] = time.monotonic()
                    state["samples"].append(sample)
        except BaseException as exc:
            record_error(state, error_info(exc))
        finally:
            prepared.set()
            iterator = subscription = response = None  # Iterator finalizer stops the SDK monitor.
            gc.collect()
            state["iterator_released"] = True
    for device, plan in plans.items():
        states[device] = {}
        for role, selector in (("no_selector", None), ("configured", plan["configured"]), ("all", plan["all"])):
            state = {"requested_selector": selector, "setup_ok": False, "iterator_released": False, "window_ready": False, "measurement_started_elapsed_s": None, "pre_window_discarded": 0, "receive_first_updates": False, "live_updates": 0, "valid_image_updates": 0, "overlap_valid_image_updates": 0, "cached_updates_ignored": 0, "header_selectors": [], "image_header_selectors": [], "overlap_image_header_selectors": [], "samples": [], "errors": [], "error_count": 0}
            for category in ("_counters", "_image_counters", "_overlap_counters"):
                state[category] = {k: set() for k in ("imageNumber", "frameNumber", "imageTimeStamp")}
            states[device][role] = state
            if role == "configured" and selector is None:
                state["skipped"] = "No concrete configured selector"
                continue
            event = threading.Event()
            worker = threading.Thread(target=consume, args=(device, role, selector, state, event), daemon=True)
            workers.append((worker, state)); ready.append(event); worker.start()
    setup_deadline = time.monotonic() + timeout_s
    for event in ready:
        event.wait(max(0, setup_deadline - time.monotonic()))
    for worker, state in workers:
        if not state["setup_ok"] and not state["errors"]:
            record_error(state, {"type": "SetupTimeout", "message": "subscribe/iter setup exceeded common setup deadline"})
    start = time.monotonic()
    gate.set()
    try:
        stop.wait(seconds)
    finally:
        stop.set()
        for worker, state in workers:
            worker.join(0.5)
            state["consumer_stopped"] = not worker.is_alive()
    for streams in states.values():
        for state in streams.values():
            state["window_seconds"] = seconds
            state["header_selectors"] = sorted(set(state["header_selectors"]))
            state["distinct_counters"] = {k: len(v) for k, v in state.pop("_counters").items()}
            state["distinct_image_counters"] = {k: len(v) for k, v in state.pop("_image_counters").items()}
            state["distinct_overlap_image_counters"] = {k: len(v) for k, v in state.pop("_overlap_counters").items()}
            for sample in state["samples"]:
                sample["received_elapsed_s"] = round(sample.pop("received_monotonic_s") - start, 3)
    return states


def summarize(plan, streams):
    selected = plan["configured"]
    configured, all_users = streams["configured"], streams["all"]
    if configured["valid_image_updates"]:
        return "Configured concrete selector delivered fresh nonempty 2D image payloads in this run; selector works for this observation."
    other_users = [s for s in all_users["overlap_image_header_selectors"] if concrete_selector(s) and s != selected]
    clean = all(s["setup_ok"] and s["window_ready"] and s.get("consumer_stopped") and not s["error_count"] for s in (configured, all_users))
    advancing = any(n > 1 for n in all_users["distinct_overlap_image_counters"].values())
    if selected and clean and not configured["live_updates"] and all_users["overlap_valid_image_updates"] > 1 and advancing and other_users:
        return "Configured selector mismatch supported: concurrent ALL subscription received valid images with advancing counters for %s while %s received no notifications. Check timing-user intent before any change." % (", ".join(other_users), selected)
    return "Inconclusive: silence/errors alone cannot distinguish selector, trigger, publication, permissions, transport, or hardware problems."


def text_report(report):
    lines = ["CLEAR26 read-only camera selector probe", "No device settings changed; no image pixels saved.", "GET results are snapshots, never counted as live.", "", "Configured / requested / observed selector comparison:"]
    for device, entry in report["devices"].items():
        lines.extend(["", device, "  requested config: " + json.dumps(entry["plan"]["requested"]), "  ALL: %s (%s)" % (entry["plan"]["all"], entry["plan"]["all_selector_source"])])
        for role, stream in report.get("streams", {}).get(device, {}).items():
            lines.append("  %s: requested=%r fresh=%d valid_images=%d observed=%s distinct=%s" % (role, stream["requested_selector"], stream["live_updates"], stream["valid_image_updates"], stream["image_header_selectors"], stream["distinct_image_counters"]))
        lines.append("  " + entry.get("summary", "pending"))
    lines.extend(["", "Progress:", *report["events"], "", "Full diagnostic JSON:", json.dumps(safe(report), indent=2, allow_nan=False)])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", action="append", help="Repeat for multiple devices; defaults to the two active BTVDC devices")
    parser.add_argument("--config", type=Path, default=REPO / "configs/experiment_v1.json")
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--get-timeout", type=float, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path.home() / "Desktop")
    args = parser.parse_args(argv)
    if not 1 <= args.seconds <= 30 or not 0 < args.get_timeout <= 10:
        parser.error("--seconds must be 1..30; --get-timeout must be greater than 0 and at most 10")
    stamp = datetime.now(ZoneInfo("Europe/Paris"))
    base = args.output_dir.expanduser() / ("CLEAR26_camera_selectors_" + stamp.strftime("%Y%m%d_%H%M%S"))
    base.parent.mkdir(parents=True, exist_ok=True)
    report = {"started_paris": stamp.isoformat(), "read_only": True, "hardware_sets": 0, "completed": False, "events": [], "devices": {}, "catalog": {}, "python": sys.version, "executable": sys.executable, "sdk_paths": {}, "config_path": str(args.config), "global_limit_s": 240}
    def save():
        for suffix, payload in ((".json", json.dumps(safe(report), indent=2, allow_nan=False) + "\n"), (".txt", text_report(report))):
            path = base.with_suffix(suffix)
            temporary = path.with_suffix(suffix + ".tmp")
            temporary.write_text(payload); temporary.replace(path)
    def note(message):
        report["events"].append(message); print(message, flush=True); save()
    def timed_out(signum, frame):
        report["global_timeout"] = "240-second process limit reached; partial results retained"
        save(); os._exit(2)
    signal.signal(signal.SIGALRM, timed_out); signal.alarm(240)
    note("Reports: " + str(base) + ".json / .txt")
    try:
        report["versions"] = {}
        for package in ("pyda", "pyda-rda3", "pyccda", "navpy", "numpy"):
            try:
                report["versions"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                report["versions"][package] = "unavailable / distribution name unknown"
        cfg = {}
        try:
            raw = args.config.read_bytes(); cfg = json.loads(raw)
            report["config_sha256"] = hashlib.sha256(raw).hexdigest()
        except Exception as exc:
            report["config_unavailable"] = error_info(exc)
        tap = cfg.get("tap", {})
        report["configured_camera_devices"] = {k: v.get("device") for k, v in cfg.get("camera_setting", {}).items() if isinstance(v, dict)}
        report["source_hashes"] = {}
        for relative in ("tests/probe_camera_selectors.py", "src/optidaq/extensions/cern_pyda.py"):
            path = REPO / relative
            if path.is_file():
                report["source_hashes"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            import pyccda
            report["sdk_paths"]["pyccda"] = getattr(pyccda, "__file__", None)
            created = bounded_call(pyccda.SyncAPI, args.get_timeout)
            api = created.get("value")
            if api is None:
                report["catalog_unavailable"] = created
            else:
                for name in dict.fromkeys(list(args.device or DEVICES) + list(OLD_DEVICES)):
                    report["catalog"][name] = bounded_call(lambda name=name: catalog_entry(api, name), args.get_timeout)
                    note("Catalog/schema read: " + name + (" (metadata only)" if name in OLD_DEVICES else ""))
        except Exception as exc:
            report["catalog_unavailable"] = error_info(exc)
        import pyda
        import pyda_rda3
        from pyda_rda3 import RdaProvider
        report["sdk_paths"].update({"pyda": getattr(pyda, "__file__", None), "pyda_rda3": getattr(pyda_rda3, "__file__", None)})
        report["sdk_contract"] = {}
        for name in ("get", "subscribe"):
            method = getattr(pyda.SimpleClient, name)
            try:
                signature = str(inspect.signature(method))
            except Exception as exc:
                signature = error_info(exc)
            report["sdk_contract"][name] = {"signature": signature, "documentation": inspect.getdoc(method)}
        created = bounded_call(lambda: pyda.SimpleClient(provider=RdaProvider()), args.get_timeout)
        if "error" in created:
            report["client_unavailable"] = created; return 1
        client = created["value"]
        plans = {}
        for device in dict.fromkeys(args.device or DEVICES):
            plan = selector_plan(report["catalog"].get(device, {}).get("value", {}), tap)
            entry = {"plan": plan, "before": {}, "image_gets": {}}
            report["devices"][device] = entry; plans[device] = plan
            for prop in PROPERTIES[1:]:
                entry["before"][prop] = read_property(client, device + "/" + prop, args.get_timeout)
                note("Snapshot before: " + device + "/" + prop)
            candidates = [None, tap.get("get_selector"), plan["configured"]]
            user = entry["before"]["AcquisitionStatus"].get("value", {}).get("data", {}).get("cycleNameGlob")
            if concrete_selector(user):
                candidates.append(user)
            for selector in dict.fromkeys(candidates):
                entry["image_gets"][str(selector)] = read_property(client, device + "/LastImage", args.get_timeout, selector)
                note("Image snapshot: %s selector=%r" % (device, selector))
        note("Concurrent fresh-update comparison for %.1fs; cached first updates disabled" % args.seconds)
        report["streams"] = observe_streams(client, plans, args.seconds, args.get_timeout)
        save()
        for device, entry in report["devices"].items():
            streams = report["streams"][device]
            entry["summary"] = summarize(entry["plan"], streams)
            entry["observed_user_image_gets"] = {}
            observed = [s for s in streams["all"]["header_selectors"] if concrete_selector(s)][:3]
            for selector in observed:
                entry["observed_user_image_gets"][selector] = read_property(client, device + "/LastImage", args.get_timeout, selector)
                note("Observed-user snapshot: %s %s" % (device, selector))
            entry["after"] = {}
            for prop in PROPERTIES[1:]:
                entry["after"][prop] = read_property(client, device + "/" + prop, args.get_timeout)
                note("Snapshot after: " + device + "/" + prop)
            note(device + ": " + entry["summary"])
        report["completed"] = True
        note("Finished. Return the JSON report for comparison; TXT is readable alongside it.")
        return 0
    except BaseException as exc:
        report["probe_error"] = error_info(exc)
        note("Probe stopped; completed reads and full error retained: " + str(exc))
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        report["finished_paris"] = datetime.now(ZoneInfo("Europe/Paris")).isoformat()
        save(); signal.alarm(0)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)  # A blocked SDK daemon worker cannot prolong the diagnostic process.
