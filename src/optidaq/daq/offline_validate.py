"""Offline validation and timestamp clustering for raw DAQ streams."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Union

from optidaq.daq.contracts import NO_PULSE_ID
from optidaq.daq.raw_stream import (
    RawRecord,
    RawStreamParseError,
    RawStreamSpec,
    iter_stream_records,
    validate_stream,
)
from optidaq.daq.sync_tools import evaluate_sync_sources, sync_value


def validate_device_stream(
    run_dir: Union[Path, str],
    spec: RawStreamSpec,
    *,
    strict: bool = False,
) -> dict:
    """Validate one device stream and return a plain summary dict.

    ``frame_index`` is writer-owned and must be exactly 0..n-1. The upstream
    ``source_image_number`` may start anywhere, but must strictly increase;
    gaps are warnings. ``acq_stamp_ns`` must be non-decreasing.
    """

    report = validate_stream(run_dir, spec, strict=strict)
    errors = list(report.errors)
    warnings = list(report.warnings)
    headers = []

    for record in iter_stream_records(run_dir, spec):
        header = dict(record.header)
        headers.append(header)

    previous_frame_index = None
    previous_source_image_number = None
    previous_acq_stamp_ns = None

    for record_number, header in enumerate(headers):
        frame_index = header.get("frame_index")
        source_image_number = header.get("source_image_number")
        acq_stamp_ns = header.get("acq_stamp_ns")

        expected_frame_index = record_number
        if frame_index != expected_frame_index:
            if record_number == 0:
                errors.append("first frame_index is not 0: %r" % frame_index)
            elif previous_frame_index is not None and frame_index == previous_frame_index:
                errors.append(
                    "frame_index repeat at record %d: %r" % (record_number, frame_index)
                )
            elif previous_frame_index is not None and frame_index < previous_frame_index:
                errors.append(
                    "frame_index decreases at record %d: %r after %r"
                    % (record_number, frame_index, previous_frame_index)
                )
            else:
                errors.append(
                    "frame_index jump at record %d: expected %d, got %r"
                    % (record_number, expected_frame_index, frame_index)
                )

        if (
            previous_source_image_number is not None
            and source_image_number is not None
        ):
            if source_image_number <= previous_source_image_number:
                errors.append(
                    "source_image_number is not strictly increasing at record "
                    "%d: %r after %r"
                    % (
                        record_number,
                        source_image_number,
                        previous_source_image_number,
                    )
                )
            elif source_image_number > previous_source_image_number + 1:
                gap_size = source_image_number - previous_source_image_number - 1
                first_missing = previous_source_image_number + 1
                last_missing = source_image_number - 1
                warnings.append(
                    "source_image_number gap at record %d: previous=%d, "
                    "current=%d, gap size=%d missing source image numbers: %d..%d"
                    % (
                        record_number,
                        previous_source_image_number,
                        source_image_number,
                        gap_size,
                        first_missing,
                        last_missing,
                    )
                )

        if previous_acq_stamp_ns is not None and acq_stamp_ns is not None:
            if acq_stamp_ns < previous_acq_stamp_ns:
                errors.append(
                    "acq_stamp_ns decreases at record %d: %r after %r"
                    % (record_number, acq_stamp_ns, previous_acq_stamp_ns)
                )

        previous_frame_index = frame_index
        previous_source_image_number = source_image_number
        previous_acq_stamp_ns = acq_stamp_ns

    summary = {
        "device": spec.file_prefix,
        "file_prefix": spec.file_prefix,
        "n_records": len(headers),
        "n_segments": report.segment_file_count,
        "first_header": headers[0] if headers else None,
        "last_header": headers[-1] if headers else None,
        "errors": errors,
        "warnings": warnings,
    }
    if strict and errors:
        raise RawStreamParseError(
            "device stream validation failed for %s: %s"
            % (spec.file_prefix, "; ".join(errors))
        )
    return summary


def read_records_by_device(
    run_dir: Union[Path, str],
    specs: Mapping[str, RawStreamSpec],
) -> Dict[str, List[RawRecord]]:
    """Read valid raw records grouped by device key without pairing."""

    return {
        device: list(iter_stream_records(run_dir, spec))
        for device, spec in specs.items()
    }


def read_sync_sidecars(
    run_dir: Union[Path, str],
    aliases: List[str],
) -> Dict[str, List[dict]]:
    """Read per-camera sync audit sidecars."""

    root = Path(run_dir)
    records: Dict[str, List[dict]] = {}
    for alias in aliases:
        path = root / ("%s.sync.jsonl" % alias)
        if not path.exists():
            raise RuntimeError("sync sidecar missing for %s: %s" % (alias, path))
        alias_records = []
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        "invalid sync sidecar JSON %s:%d: %s"
                        % (path, line_number, exc)
                    ) from exc
                record.setdefault("alias", alias)
                alias_records.append(record)
        records[alias] = alias_records
    return records


def build_configured_aligned_index(
    per_device_records: Mapping[str, List[RawRecord]],
    aliases: List[str],
    sync_cfg: Mapping[str, object],
    sync_records_by_device: Mapping[str, List[dict]],
) -> dict:
    """Build aligned index using config-selected sync sidecar fields."""

    sync_report = evaluate_sync_sources(sync_records_by_device, aliases, sync_cfg)
    if not sync_report["passed"]:
        return {"entries": [], "sync_report": sync_report}

    selected_report = sync_report["selected_report"]
    selected_source = sync_report["selected_source"]
    raw_by_device = {
        alias: {
            int(record.global_record_index): record
            for record in per_device_records.get(alias, [])
        }
        for alias in aliases
    }

    entries = []
    for offline_shot_id, group in enumerate(selected_report["groups"]):
        entries.append(
            _configured_aligned_entry(
                offline_shot_id,
                str(selected_source),
                group,
                aliases,
                raw_by_device,
            )
        )
    invalid = [entry for entry in entries if not bool(entry["valid"])]
    if invalid:
        sync_report = dict(sync_report)
        sync_report["passed"] = False
        selected_report = dict(selected_report)
        selected_report["passed"] = False
        selected_report["reasons"] = list(selected_report.get("reasons", [])) + [
            "%d aligned index entry/entries invalid" % len(invalid)
        ]
        sync_report["selected_report"] = selected_report
    return {"entries": entries, "sync_report": sync_report}


def rewrite_sync_sidecars_with_selection(
    run_dir: Union[Path, str],
    aliases: List[str],
    selected_source: str,
) -> None:
    """Rewrite sync sidecars so chosen_sync_* matches final offline selection."""

    root = Path(run_dir)
    for alias in aliases:
        path = root / ("%s.sync.jsonl" % alias)
        tmp_path = root / ("%s.sync.jsonl.tmp" % alias)
        with open(path, "r", encoding="utf-8") as source_handle, open(
            tmp_path,
            "w",
            encoding="utf-8",
        ) as dest_handle:
            for line in source_handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                record["chosen_sync_source"] = selected_source
                record["chosen_sync_value"] = sync_value(record, selected_source)
                dest_handle.write(json.dumps(record, sort_keys=True))
                dest_handle.write("\n")
        tmp_path.replace(path)


def cluster_by_timestamp(
    per_device_records: Mapping[str, List[RawRecord]],
    *,
    gap_ns: int,
) -> List[dict]:
    """Cluster records offline by adjacent timestamp gaps."""

    if gap_ns < 0:
        raise ValueError("gap_ns must be non-negative")

    all_records = []
    for device, records in per_device_records.items():
        for record in records:
            all_records.append((record.header["acq_stamp_ns"], device, record))
    all_records.sort(key=lambda item: item[0])

    clusters: List[List[tuple]] = []
    current: List[tuple] = []
    previous_stamp = None
    for item in all_records:
        stamp = item[0]
        if previous_stamp is not None and stamp - previous_stamp > gap_ns:
            clusters.append(current)
            current = []
        current.append(item)
        previous_stamp = stamp
    if current:
        clusters.append(current)

    summaries = []
    for cluster_index, cluster in enumerate(clusters):
        stamps = [item[0] for item in cluster]
        counts: Dict[str, int] = {}
        for _stamp, device, _record in cluster:
            counts[device] = counts.get(device, 0) + 1
        summaries.append(
            {
                "cluster_index": cluster_index,
                "devices": sorted(counts),
                "min_stamp_ns": min(stamps),
                "max_stamp_ns": max(stamps),
                "n_records": len(cluster),
                "records_by_device_count": counts,
            }
        )
    return summaries


def build_aligned_index(
    per_device_records: Mapping[str, List[RawRecord]],
    aliases: List[str],
    trigger_period_ns: int,
    max_stamp_span_ns: int,
) -> List[dict]:
    """Build an offline aligned index from per-camera raw records."""

    if trigger_period_ns <= 0:
        raise ValueError("trigger_period_ns must be positive")
    if max_stamp_span_ns < 0:
        raise ValueError("max_stamp_span_ns must be non-negative")

    all_records = []
    for device, records in per_device_records.items():
        for record in records:
            pulse_id = int(record.header.get("pulse_id", NO_PULSE_ID))
            all_records.append(
                {
                    "device": device,
                    "record": record,
                    "pulse_id": pulse_id,
                    "acq_stamp_ns": int(record.header["acq_stamp_ns"]),
                }
            )

    use_pulse_id = bool(all_records) and all(
        item["pulse_id"] != NO_PULSE_ID for item in all_records
    )

    if use_pulse_id:
        buckets = _pulse_id_buckets(all_records)
        sync_key_type = "pulse_id"
    else:
        buckets = _timestamp_buckets(all_records, max(1, trigger_period_ns // 2))
        sync_key_type = "timestamp_cluster"

    entries = []
    for offline_shot_id, bucket in enumerate(buckets):
        entries.append(
            _aligned_entry(
                offline_shot_id,
                sync_key_type,
                bucket,
                aliases,
                max_stamp_span_ns,
            )
        )
    return entries


def write_aligned_index_jsonl(
    run_dir: Union[Path, str],
    entries: List[dict],
    filename: str = "aligned_index.jsonl",
) -> Path:
    """Write aligned index entries as newline-delimited JSON."""

    path = Path(run_dir) / filename
    with open(path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True))
            handle.write("\n")
    return path


def _pulse_id_buckets(all_records: List[dict]) -> List[List[dict]]:
    grouped: Dict[int, List[dict]] = {}
    for item in all_records:
        grouped.setdefault(int(item["pulse_id"]), []).append(item)
    return [grouped[key] for key in sorted(grouped)]


def _timestamp_buckets(all_records: List[dict], gap_ns: int) -> List[List[dict]]:
    ordered = sorted(all_records, key=lambda item: item["acq_stamp_ns"])
    buckets = []
    current = []
    previous_stamp = None
    for item in ordered:
        stamp = item["acq_stamp_ns"]
        if previous_stamp is not None and stamp - previous_stamp > gap_ns:
            buckets.append(current)
            current = []
        current.append(item)
        previous_stamp = stamp
    if current:
        buckets.append(current)
    return buckets


def _aligned_entry(
    offline_shot_id: int,
    sync_key_type: str,
    bucket: List[dict],
    aliases: List[str],
    max_stamp_span_ns: int,
) -> dict:
    errors = []
    records_by_device: Dict[str, List[dict]] = {}
    for item in bucket:
        records_by_device.setdefault(item["device"], []).append(item)

    for alias in aliases:
        count = len(records_by_device.get(alias, []))
        if count == 0:
            errors.append("missing camera %s" % alias)
        elif count > 1:
            errors.append("duplicate frames for camera %s" % alias)

    stamps = [item["acq_stamp_ns"] for item in bucket]
    stamp_span_ns = max(stamps) - min(stamps) if stamps else 0
    if stamp_span_ns > max_stamp_span_ns:
        errors.append(
            "timestamp span %d ns exceeds max %d ns"
            % (stamp_span_ns, max_stamp_span_ns)
        )

    if sync_key_type == "pulse_id":
        pulse_ids = sorted(set(item["pulse_id"] for item in bucket))
        if len(pulse_ids) != 1:
            errors.append("mixed pulse_id values: %s" % pulse_ids)
        if pulse_ids and pulse_ids[0] == NO_PULSE_ID:
            errors.append("unknown pulse_id in pulse_id bucket")
        sync_key = pulse_ids[0] if pulse_ids else NO_PULSE_ID
    else:
        pulse_ids = sorted(set(item["pulse_id"] for item in bucket))
        if any(pulse_id != NO_PULSE_ID for pulse_id in pulse_ids):
            errors.append("mixed/known pulse_id in timestamp bucket: %s" % pulse_ids)
        sync_key = min(stamps) if stamps else 0

    records = {}
    for alias in aliases:
        items = records_by_device.get(alias, [])
        if len(items) == 1:
            records[alias] = _record_index_entry(items[0]["record"])

    return {
        "offline_shot_id": offline_shot_id,
        "sync_key_type": sync_key_type,
        "sync_key": int(sync_key),
        "valid": not errors,
        "stamp_span_ns": int(stamp_span_ns),
        "records": records,
        "errors": errors,
    }


def _record_index_entry(record: RawRecord) -> dict:
    return {
        "segment_index": record.segment_index,
        "global_record_index": record.global_record_index,
        "source_image_number": record.header["source_image_number"],
        "acq_stamp_ns": record.header["acq_stamp_ns"],
        "pulse_id": record.header.get("pulse_id", NO_PULSE_ID),
    }


def _configured_aligned_entry(
    offline_shot_id: int,
    selected_source: str,
    group: Mapping[str, object],
    aliases: List[str],
    raw_by_device: Mapping[str, Mapping[int, RawRecord]],
) -> dict:
    errors = list(group.get("errors", []))
    sidecars_by_alias: Dict[str, List[dict]] = {}
    for record in group.get("records", []):
        alias = str(record["alias"])
        sidecars_by_alias.setdefault(alias, []).append(dict(record))

    records = {}
    stamps = []
    sync_values = []
    for alias in aliases:
        sidecars = sidecars_by_alias.get(alias, [])
        if len(sidecars) != 1:
            continue
        frame_index = int(sidecars[0]["frame_index"])
        raw_record = raw_by_device.get(alias, {}).get(frame_index)
        if raw_record is None:
            errors.append(
                "missing raw record for %s frame_index %d" % (alias, frame_index)
            )
            continue
        records[alias] = _record_index_entry(raw_record)
        stamps.append(int(raw_record.header["acq_stamp_ns"]))
        value = sync_value(sidecars[0], selected_source)
        sync_values.append(value)

    stamp_span_ns = max(stamps) - min(stamps) if stamps else 0
    sync_key = group.get("sync_key")
    if group.get("kind") == "exact":
        unique_values = sorted(set(sync_values), key=lambda value: repr(value))
        if len(unique_values) != 1:
            errors.append("mixed %s values: %s" % (selected_source, unique_values))
        sync_key = unique_values[0] if unique_values else sync_key

    return {
        "offline_shot_id": offline_shot_id,
        "sync_key_type": selected_source,
        "sync_key": sync_key,
        "valid": not errors,
        "stamp_span_ns": int(stamp_span_ns),
        "sync_span_ns": int(group.get("span_ns", 0)),
        "records": records,
        "errors": errors,
    }
