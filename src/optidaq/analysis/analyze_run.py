#!/usr/bin/env python3
"""Check a saved PNG/DMD run and write results.npz plus a fixed-metric PDF.

The dataset is read-only. Repeated runs verify source hashes and reuse the
cached measurements; --refresh recomputes them. No hardware calls.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import zipfile

import numpy as np
from PIL import Image

ROLES = {"cam1": "MMF output", "cam2": "Chromox ground truth"}
CACHE_SCHEMA_VERSION = 1


def read_lines(path, errors=None):
    errors = [] if errors is None else errors
    if not path.exists():
        errors.append("Missing metadata file: " + path.name)
        return []
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("record is not an object")
            rows.append(row)
        except ValueError as exc:
            errors.append("%s line %d: %s" % (path.name, number, exc))
    return rows


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_manifest(run):
    return {str(p.relative_to(run)): {"bytes": p.stat().st_size, "sha256": file_hash(p)}
            for p in sorted(run.rglob("*")) if p.is_file()
            and not any(v.startswith("data_analysis") for v in p.relative_to(run).parts)}


def describe(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    a = a[np.isfinite(a)]
    return {"n": int(len(a)), "min": float(a.min()), "median": float(np.median(a)),
            "max": float(a.max()), "mean": float(a.mean())} if len(a) else {"n": 0}


def stable_intervals(events):
    spans, errors, active, seen = [], [], None, set()
    previous_ns = None
    for e in events:
        kind, sid, ns = e.get("event"), e.get("step_id"), e.get("host_time_ns")
        if (not isinstance(kind, str) or type(ns) is not int
                or kind in ("change_begin", "settle_complete") and type(sid) is not int):
            errors.append("Invalid pattern event record")
            continue
        if previous_ns is not None and ns < previous_ns:
            errors.append("Event clock decreases")
        previous_ns = ns
        if kind in ("change_begin", "end"):
            if active:
                if "settle_ns" in active:
                    active["end_ns"] = ns
                    spans.append(active)
                else:
                    errors.append("Missing settle event")
            active = None
            if kind == "change_begin":
                if sid in seen:
                    errors.append("Repeated step ID")
                seen.add(sid)
                active = {"step_id": sid, "start_ns": ns}
        elif kind == "settle_complete":
            if active is None or active["step_id"] != sid or "settle_ns" in active:
                errors.append("Unmatched/repeated settle event")
            else:
                active["settle_ns"] = ns
    if active:
        errors.append("Unclosed final pattern")
    return spans, errors


def match_step(ns, spans):
    if ns is None:
        return None
    return next((s["step_id"] for s in spans if s["settle_ns"] <= ns < s["end_ns"]), None)


def grouped(rows, fields):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row.get(k) for k in fields)].append(row)
    return result


def histogram_percentiles(histogram, percentiles):
    """Exact NumPy-style linear percentiles of integer pixel counts."""
    total = int(histogram.sum())
    if total <= 0:
        raise ValueError("Cannot calculate percentiles of an empty histogram")
    ranks = np.asarray(percentiles, dtype=float) / 100 * (total - 1)
    low = np.floor(ranks).astype(np.int64)
    high = np.ceil(ranks).astype(np.int64)
    cumulative = np.cumsum(histogram, dtype=np.int64)
    lo_values = np.searchsorted(cumulative, low, side="right")
    hi_values = np.searchsorted(cumulative, high, side="right")
    return (lo_values + (hi_values - lo_values) * (ranks - low)).tolist()


def code_hashes():
    return {name: file_hash(Path(__file__).parent / name)
            for name in ("analyze_run.py", "clean_plots.py")}


def compute_results(run, manifest=None):
    """Decode each image once, retaining only measurements, maxima and counts."""
    run = Path(run)
    manifest = source_manifest(run) if manifest is None else manifest
    cfg = json.loads((run / "resolved_config.json").read_text())["config"]
    cameras, requested, quota = cfg["cameras"], cfg["dmd"]["count"], cfg["scan"]["frames_per_point"]
    failures = []
    events = read_lines(run / "actuator.events.jsonl", failures)
    spans, event_errors = stable_intervals(events)
    failures.extend(event_errors)
    event_times = [e["host_time_ns"] for e in events if type(e.get("host_time_ns")) is int]
    t0 = event_times[0] if event_times else 0
    duration = (event_times[-1] - t0) / 1e9 if len(event_times) > 1 else 0
    db_rows, db_status = [], "missing"
    if (run / "dataset.db").is_file():
        try:
            with sqlite3.connect((run / "dataset.db").as_uri() + "?mode=ro", uri=True) as con:
                db_status = con.execute("PRAGMA integrity_check").fetchone()[0]
                con.row_factory = sqlite3.Row
                db_rows = [dict(row) for row in con.execute("SELECT * FROM images")]
                required = {"camera", "frame_index", "png_path", "sample_id", "sync_time_ns",
                            "stat_mean", "stat_max", "stat_sum"}
                if any(not required.issubset(row) for row in db_rows):
                    db_status = "Image records lack required database fields"
                    db_rows = []
        except sqlite3.Error as exc:
            db_status = str(exc)
    if db_status != "ok":
        failures.append("Database integrity: " + db_status)
    db_map = grouped(db_rows, ("camera", "frame_index"))
    if any(row.get("camera") not in cameras for row in db_rows):
        failures.append("Database contains records for an unconfigured camera")
    labels = read_lines(run / "actuator_frame_index.jsonl", failures)
    if any(row.get("device") not in cameras for row in labels):
        failures.append("DMD index contains records for an unconfigured camera")
    label_map = grouped(labels, ("device", "frame_index"))
    frames, products, checks, statistics = [], {}, {}, {}

    for cam in cameras:
        errors = []
        sidecar = read_lines(run / (cam + ".sync.jsonl"), errors)
        side_map = grouped(sidecar, ("frame_index",))
        paths = sorted((run / cam).glob("*.png"))
        stored = json.loads((run / (cam + ".png.json")).read_text())["stored"]
        bits = {"uint8": 8, "uint16": 16}[stored["dtype"]]
        indices, hashes, current = [], [], []
        maximum = np.zeros(stored["shape"], dtype=np.uint16)
        histogram = np.zeros(65536, dtype=np.int64)
        for path in paths:
            try:
                idx = int(path.stem.rsplit("_", 1)[1])
                indices.append(idx)
                sr, dr, ar = side_map.get((idx,), []), db_map.get((cam, idx), []), label_map.get((cam, idx), [])
                if any(len(rows) != 1 for rows in (sr, dr, ar)):
                    errors.append("Frame %d lacks unique sidecar/DB/DMD-index records" % idx)
                sync, db, label = sr[0] if sr else {}, dr[0] if dr else {}, ar[0] if ar else {}
                with path.open("rb") as f:
                    header = f.read(26)
                with Image.open(path) as im:
                    array = np.asarray(im).copy()
                if (array.ndim != 2 or list(array.shape) != stored["shape"] or len(header) < 26
                        or header[24] != bits or header[25] != 0):
                    raise ValueError("Shape/bit depth differs from stored manifest")
                if array.min() < 0 or array.max() > 2**bits-1:
                    raise ValueError("Pixels outside dtype range")
                a = array.astype(float)
                mean, peak, total = float(a.mean()), int(a.max()), float(a.sum())
                if db.get("png_path") != str(path.relative_to(run)):
                    errors.append("Frame %d DB filename differs" % idx)
                metadata = sync.get("png", {}).get("metadata", {})
                for name, value in (("stat_mean", mean), ("stat_max", peak), ("stat_sum", total)):
                    for origin, record in (("sidecar", metadata), ("DB", db)):
                        if record.get(name) is None or not np.isclose(record[name], value, rtol=0, atol=1e-8):
                            errors.append("Frame %d %s %s differs from pixels" % (idx, origin, name))
                received = sync.get("local_receive_ns")
                step = match_step(received, spans)
                if label.get("step_id") != step:
                    errors.append("Frame %d DMD label differs from event intervals" % idx)
                if label.get("local_receive_ns") != received:
                    errors.append("Frame %d DMD index receive time differs from sidecar" % idx)
                if db.get("sync_time_ns") != sync.get("image_ready_ns"):
                    errors.append("Frame %d DB/sidecar pairing timestamps differ" % idx)
                row = {"camera": cam, "index": idx, "sample_id": db.get("sample_id"), "step_id": step,
                    "time_s": (received-t0)/1e9 if received is not None else None,
                    "local_receive_ns": received, "image_ready_ns": sync.get("image_ready_ns"),
                    "acq_stamp_ns": sync.get("acq_stamp_ns"), "source_image_number": sync.get("source_image_number"),
                    "timing_user": sync.get("cycleNameGlob"), "min": int(a.min()), "mean": mean, "max": peak,
                    "p99": float(np.percentile(a, 99)), "total": total,
                    "zero_pixels": int(np.count_nonzero(a == 0)), "pixels_ge_4095": int(np.count_nonzero(a >= 4095))}
                maximum = np.maximum(maximum, array.astype(np.uint16))
                histogram += np.bincount(array.ravel().astype(np.int64), minlength=65536)
                frames.append(row); current.append(row)
                hashes.append(hashlib.sha256(array.astype("<u2").tobytes()).hexdigest())
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append(path.name + ": " + str(exc))
        products[cam] = {"maximum": maximum, "histogram": histogram}
        if not current:
            errors.append("No readable images")
        ids = set(indices)
        expected_ids = set(range(requested * quota))
        missing_ids = sorted(expected_ids - ids)
        if ids - expected_ids:
            errors.append("Frame indices outside requested sequence")
        if len(ids) != len(indices) or any(len(v) != 1 for v in side_map.values()):
            errors.append("Duplicate frame index")
        if ids != {k[0] for k in side_map} or ids != {k[1] for k in db_map if k[0] == cam}:
            errors.append("PNG/sidecar/DB frame index sets differ")
        if ids != {k[1] for k in label_map if k[0] == cam}:
            errors.append("PNG/DMD-index frame index sets differ")
        timestamp_fields = ("local_receive_ns", "image_ready_ns", "acq_stamp_ns")
        regressed = sum(sum(b < a for a, b in zip(ts, ts[1:])) for ts in
            ([f[k] for f in current if f[k] is not None] for k in timestamp_fields))
        missing_ts = sum(f[k] is None for f in current for k in timestamp_fields)
        source = [f["source_image_number"] for f in current if f["source_image_number"] is not None]
        received = [f["local_receive_ns"] for f in current if f["local_receive_ns"] is not None]
        per_step = {str(s["step_id"]): sum(f["step_id"] == s["step_id"] for f in current) for s in spans}
        c = {"expected_frames": requested*quota, "png_count": len(paths), "sidecar_count": len(sidecar),
            "db_count": sum(r["camera"] == cam for r in db_rows), "decoded_count": len(current),
            "shape": stored["shape"], "dtype": stored["dtype"], "errors": errors, "missing_frame_indices": missing_ids,
            "identical_duplicate_images": len(hashes)-len(set(hashes)), "missing_timestamps": missing_ts,
            "timestamp_regressions": regressed, "source_counter_non_increasing": sum(b <= a for a,b in zip(source,source[1:])),
            "unobserved_source_counters": sum(max(b-a-1,0) for a,b in zip(source,source[1:])),
            "frames_per_host_labelled_pattern": per_step, "frames_without_pattern_label": sum(f["step_id"] is None for f in current),
            "frame_interval_s": describe(np.diff(received)/1e9), "timing_users": dict(Counter(f["timing_user"] for f in current))}
        if errors or missing_ids or missing_ts or regressed or c["source_counter_non_increasing"]:
            failures.append(cam + ": integrity/timestamp errors")
        if any(c[k] != requested*quota for k in ("png_count", "sidecar_count", "db_count", "decoded_count")):
            failures.append(cam + ": count differs from requested quota")
        if c["frames_without_pattern_label"] or any(v != quota for v in per_step.values()):
            failures.append(cam + ": per-pattern count mismatch")
        checks[cam] = c
        statistics[cam] = {"role": ROLES.get(cam,cam), "mean_dn": describe(f["mean"] for f in current),
            "frame_peak_dn": describe(f["max"] for f in current),
            "pooled_pixel_quantiles_dn": dict(zip(("p1","p50","p99"), histogram_percentiles(histogram, [1,50,99]))) if current else {},
            "mean_first_to_last_percent": 100*(current[-1]["mean"]/current[0]["mean"]-1) if current and current[0]["mean"] else None,
            "zero_pixel_fraction": float(histogram[0] / histogram.sum()) if current else None,
            "stored_pixel_fraction_ge_4095": float(histogram[4095:].sum() / histogram.sum()) if current else None}

    sample_groups = grouped(db_rows, ("sample_id",))
    pairs, incomplete, ambiguous = [], [], []
    for (sid,), group in sample_groups.items():
        if sid is None:
            continue
        counts = Counter(r["camera"] for r in group)
        if any(v > 1 for v in counts.values()):
            ambiguous.append(sid)
        elif set(counts) != set(cameras) or any(r["sync_time_ns"] is None for r in group):
            incomplete.append(sid)
        else:
            ts = [r["sync_time_ns"] for r in group]
            pairs.append({"sample_id":sid,"span_us":(max(ts)-min(ts))/1000})
    pairs.sort(key=lambda r:r["sample_id"])
    paired_frames = grouped(frames, ("sample_id",))
    same_step_pairs = sum(
        len(paired_frames[(p["sample_id"],)]) == len(cameras)
        and len({f["step_id"] for f in paired_frames[(p["sample_id"],)]}) == 1
        and paired_frames[(p["sample_id"],)][0]["step_id"] is not None for p in pairs)
    alignment = cfg["output"]["png"]["sync_alignment"]
    pairing = {"complete_groups":len(pairs),"incomplete_ids":incomplete,"ambiguous_ids":ambiguous,
        "frames_without_sample_id":sum(r["sample_id"] is None for r in db_rows),
        "span_us":describe(p["span_us"] for p in pairs),"tolerance_us":alignment["tolerance_ns"]/1000,
        "same_host_pattern_pairs":same_step_pairs,
        "groups_outside_tolerance":sum(p["span_us"]*1000 > alignment["tolerance_ns"] for p in pairs),"source":alignment["source"]}
    if len(pairs) != requested*quota or incomplete or ambiguous or pairing["frames_without_sample_id"] or pairing["groups_outside_tolerance"] or same_step_pairs != len(pairs):
        failures.append("Camera pairing/count check failed")
    if len(spans) != requested:
        failures.append("Completed pattern count differs from requested")
    if {s["step_id"] for s in spans} != set(range(requested)):
        failures.append("Pattern IDs differ from requested sequence")
    pattern_count = len(list((run / "dmd_patterns").glob("*.png")))
    if pattern_count != requested:
        failures.append("Saved DMD pattern count differs from requested")
    if {p.name for p in (run / "dmd_patterns").glob("*.png")} != {"%06d.png" % i for i in range(requested)}:
        failures.append("DMD source pattern filenames differ from requested sequence")
    audit = {"run":run.name,"created_utc":datetime.now(timezone.utc).isoformat(),"validation_passed":not failures,
        "failures":failures,"database_integrity":db_status,"requested_patterns":requested,"completed_patterns":len(spans),
        "saved_dmd_patterns":pattern_count,"duration_s":duration,"complete_pairs_per_s":len(pairs)/duration if duration else None,
        "cameras":checks,"pairing":pairing,"statistics":statistics,"source_files_unchanged":True,"source_manifest":manifest,
        "code_sha256":code_hashes()}
    return audit, frames, pairs, products



def load_cache(path, manifest, code_hash):
    """Return checked numeric results, or None when a cache cannot be reused."""
    try:
        with np.load(path, allow_pickle=False) as cached:
            metadata = json.loads(cached["metadata"].item())
            if (metadata["schema_version"] != CACHE_SCHEMA_VERSION
                    or metadata["analysis_code_sha256"] != code_hash
                    or metadata["audit"]["source_manifest"] != manifest):
                return None
            audit, frames, pairs = metadata["audit"], metadata["frames"], metadata["pairs"]
            if not isinstance(frames, list) or not isinstance(pairs, list):
                return None
            products = {}
            for i, (camera, check) in enumerate(audit["cameras"].items()):
                maximum = cached["maximum_%d" % i]
                histogram = cached["histogram_%d" % i]
                if (maximum.dtype != np.uint16 or list(maximum.shape) != check["shape"]
                        or histogram.shape != (65536,) or histogram.dtype.kind not in "iu"
                        or np.any(histogram < 0)
                        or int(histogram.sum()) != int(np.prod(maximum.shape)) * check["decoded_count"]
                        or sum(row["camera"] == camera for row in frames) != check["decoded_count"]):
                    return None
                products[camera] = {"maximum": maximum.copy(), "histogram": histogram.copy()}
            if len(pairs) != audit["pairing"]["complete_groups"]:
                return None
            return audit, frames, pairs, products
    except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, zipfile.BadZipFile):
        return None


def save_cache(path, audit, frames, pairs, products, code_hash):
    """Write one non-pickle numerical cache; caller publishes the staged file."""
    metadata = {"schema_version": CACHE_SCHEMA_VERSION,
                "analysis_code_sha256": code_hash, "audit": audit,
                "frames": frames, "pairs": pairs}
    arrays = {"metadata": np.asarray(json.dumps(metadata, allow_nan=False))}
    for i, camera in enumerate(audit["cameras"]):
        arrays["maximum_%d" % i] = products[camera]["maximum"]
        arrays["histogram_%d" % i] = products[camera]["histogram"]
    with Path(path).open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _output_directory(run, out):
    candidate = Path(out) if out is not None else run / "data_analysis"
    if candidate.is_symlink():
        raise ValueError("Output directory must not be a symlink")
    result = candidate.resolve()
    if result == run or result in run.parents:
        raise ValueError("Output must not replace the run or its parent")
    if run in result.parents and result != run / "data_analysis":
        raise ValueError("Inside a run, output must be exactly data_analysis/")
    if any((result / name).is_symlink() for name in ("report.pdf", "results.npz")):
        raise ValueError("Output paths must not be symlinks")
    return result


def analyze_run(run, out=None, *, refresh=False):
    """Verify inputs, reuse/compute facts, then publish a PDF and numerical cache."""
    run = Path(run).resolve()
    out = _output_directory(run, out)
    manifest = source_manifest(run)
    hashes = code_hashes()
    cached = None if refresh else load_cache(out / "results.npz", manifest, hashes["analyze_run.py"])
    cache_used = cached is not None
    audit, frames, pairs, products = cached if cache_used else compute_results(run, manifest)
    audit["run"] = run.name
    audit["code_sha256"] = hashes
    if manifest != source_manifest(run):
        raise RuntimeError("Dataset changed during analysis; retry after collection finishes")
    out.mkdir(parents=True, exist_ok=True)
    # Nothing replaces the previous results until both outputs are ready and
    # source hashes have been checked again. Staging stays outside the inputs.
    with tempfile.TemporaryDirectory(prefix="data_analysis_stage_", dir=out) as temporary:
        staging = Path(temporary)
        from .clean_plots import write_report
        write_report(staging, audit, frames, pairs, products, cache_used=cache_used)
        save_cache(staging / "results.npz", audit, frames, pairs, products, hashes["analyze_run.py"])
        if manifest != source_manifest(run):
            raise RuntimeError("Dataset changed while creating the report; outputs were not replaced")
        _output_directory(run, out)
        (staging / "results.npz").replace(out / "results.npz")
        (staging / "report.pdf").replace(out / "report.pdf")
    return audit, cache_used


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, help="Optional staging output directory")
    parser.add_argument("--refresh", action="store_true", help="Recompute measurements even when the cache matches")
    args = parser.parse_args(argv)
    audit, cache_used = analyze_run(args.run_dir, args.out, refresh=args.refresh)
    print(json.dumps({"run": audit["run"], "validation_passed": audit["validation_passed"],
                      "failures": audit["failures"], "complete_pairs": audit["pairing"]["complete_groups"],
                      "cache_used": cache_used,
                      "out": str(args.out or args.run_dir / "data_analysis")}))
    if not audit["validation_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
