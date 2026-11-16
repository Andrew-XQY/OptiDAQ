"""Small synchronization policy helpers for OptiDAQ scripts."""

from __future__ import annotations

import statistics
from typing import Any, Iterable, Mapping, Optional, Sequence


DEFAULT_PULSE_ID_FIELDS = [
    "pulseId",
    "pulseID",
    "eventId",
    "eventID",
    "eventNumber",
    "shotId",
    "shotID",
    "triggerId",
    "triggerID",
    "cycleId",
    "cycleID",
]

SYNC_NUMERIC_FIELDS = (
    "cycleNbGlob",
    "cycleStampGlob",
    "imageTimeStamp",
    "imageTimeStampRaw",
    "acqImageTransmissionTime",
    "imageNumber",
    "frameNumber",
)
SYNC_TEXT_FIELDS = ("cycleNameGlob",)
DERIVED_SYNC_FIELDS = ("pulse_id", "image_ready_ns", "local_receive_ns")
SYNC_CANDIDATE_FIELDS = (
    tuple(DEFAULT_PULSE_ID_FIELDS)
    + SYNC_NUMERIC_FIELDS
    + SYNC_TEXT_FIELDS
    + DERIVED_SYNC_FIELDS
)
EXACT_SYNC_SOURCES = (
    "pulse_id",
    "cycleNbGlob",
    "cycleStampGlob",
    "cycleNameGlob",
    *DEFAULT_PULSE_ID_FIELDS,
)
LOCAL_CLOCK_SOURCES = ("local_receive_ns", "local_clock")
DEFAULT_SYNC_PRIORITY = [
    "pulse_id",
    "cycleNbGlob",
    "cycleStampGlob",
    "image_ready_ns",
    "imageTimeStampRaw",
    "imageTimeStamp",
]
DEFAULT_MAX_SPAN_NS_BY_SOURCE = {
    "pulse_id": 0,
    "cycleNbGlob": 0,
    "cycleStampGlob": 0,
    "cycleNameGlob": 0,
    "image_ready_ns": 1_000_000,
    "imageTimeStampRaw": 50_000_000,
    "imageTimeStamp": 300_000_000,
}


def pulse_id_candidate_fields(sync_cfg: Mapping[str, Any]) -> list[str]:
    pulse_cfg = sync_cfg.get("pulse_id")
    if pulse_cfg is None:
        return list(DEFAULT_PULSE_ID_FIELDS) if "pulse_id" in sync_priority(sync_cfg) else []
    if not isinstance(pulse_cfg, Mapping):
        raise RuntimeError("sync.pulse_id must be an object")
    if not bool(pulse_cfg.get("enabled", False)):
        return []
    fields = pulse_cfg.get("candidate_fields", DEFAULT_PULSE_ID_FIELDS)
    if not isinstance(fields, Sequence) or isinstance(fields, (str, bytes)):
        raise RuntimeError("sync.pulse_id.candidate_fields must be a list")
    return [str(field) for field in fields]


def sync_priority(sync_cfg: Mapping[str, Any]) -> list[str]:
    priority = sync_cfg.get("priority", DEFAULT_SYNC_PRIORITY)
    if not isinstance(priority, Sequence) or isinstance(priority, (str, bytes)):
        raise RuntimeError("sync.priority must be a list")
    result = [str(source) for source in priority]
    if not result:
        raise RuntimeError("sync.priority must not be empty")
    return result


def max_span_ns_for_source(sync_cfg: Mapping[str, Any], source: str) -> int:
    spans = sync_cfg.get("max_span_ns_by_source", {})
    if not isinstance(spans, Mapping):
        raise RuntimeError("sync.max_span_ns_by_source must be an object")
    if source in spans:
        value = spans[source]
    elif source in DEFAULT_MAX_SPAN_NS_BY_SOURCE:
        value = DEFAULT_MAX_SPAN_NS_BY_SOURCE[source]
    elif source in EXACT_SYNC_SOURCES:
        value = 0
    else:
        value = 300_000_000
    value = int(value)
    if value < 0:
        raise RuntimeError("sync max span for %s must be non-negative" % source)
    return value


def extract_sync_candidates(
    data: Mapping[str, Any],
    *,
    local_receive_ns: Optional[int] = None,
    trailing_zero_rows: Optional[int] = None,
    crc32: Optional[int] = None,
) -> dict:
    """Extract flat sync candidates from one FESA LastImage dict.

    The result is intentionally JSON-friendly and never includes ``image2D``.
    ``local_receive_ns`` is kept only for diagnostics; selection helpers reject
    it as a sync source.
    """

    out = {}
    pulse_candidates = {}
    for field in DEFAULT_PULSE_ID_FIELDS:
        value = _candidate_value(data.get(field)) if field in data else None
        out[field] = value
        if value is not None:
            pulse_candidates[field] = value

    out["pulse_id"] = None
    out["pulse_id_field"] = None
    for field in DEFAULT_PULSE_ID_FIELDS:
        value = out.get(field)
        if _non_empty(value):
            out["pulse_id"] = value
            out["pulse_id_field"] = field
            break
    out["pulse_id_candidates"] = pulse_candidates

    for field in SYNC_NUMERIC_FIELDS + SYNC_TEXT_FIELDS:
        out[field] = _candidate_value(data.get(field)) if field in data else None

    image_stamp = _optional_int_like(out.get("imageTimeStamp"))
    transmission = _optional_int_like(out.get("acqImageTransmissionTime"))
    out["image_ready_ns"] = (
        image_stamp + transmission
        if image_stamp is not None and transmission is not None
        else None
    )
    out["local_receive_ns"] = int(local_receive_ns) if local_receive_ns is not None else None
    out["trailing_zero_rows"] = (
        int(trailing_zero_rows) if trailing_zero_rows is not None else None
    )
    out["crc32"] = int(crc32) if crc32 is not None else None
    return out


def choose_frame_sync_candidate(
    candidates: Mapping[str, Any],
    sync_cfg: Mapping[str, Any],
) -> tuple[Optional[str], Any]:
    """Choose the first non-empty per-frame candidate from config priority."""

    for source in sync_priority(sync_cfg):
        if source in LOCAL_CLOCK_SOURCES:
            raise RuntimeError("local VM clock source %r is forbidden for sync" % source)
        value = sync_value(candidates, source)
        if _usable_source_value(source, value):
            return source, value
    return None, None


def evaluate_sync_sources(
    records_by_camera: Mapping[str, Sequence[Mapping[str, Any]]],
    aliases: Sequence[str],
    sync_cfg: Mapping[str, Any],
) -> dict:
    """Evaluate configured sync priority and return the first passing source."""

    priority = sync_priority(sync_cfg)
    fallback_reasons = []
    reports = {}
    selected_source = None
    selected_report = None

    for source in priority:
        if source in LOCAL_CLOCK_SOURCES:
            if bool(sync_cfg.get("forbid_local_clock", True)):
                reason = "local VM clock source %r is forbidden" % source
                fallback_reasons.append("%s: %s" % (source, reason))
                reports[source] = _failed_source_report(source, reason)
                continue

        if (
            not bool(sync_cfg.get("allow_timestamp_fallback", True))
            and source not in EXACT_SYNC_SOURCES
            and fallback_reasons
        ):
            reason = "timestamp fallback disabled"
            fallback_reasons.append("%s: %s" % (source, reason))
            reports[source] = _failed_source_report(source, reason)
            continue

        report = evaluate_sync_source(records_by_camera, aliases, sync_cfg, source)
        reports[source] = report
        if report["passed"]:
            selected_source = source
            selected_report = report
            break
        fallback_reasons.append("%s: %s" % (source, "; ".join(report["reasons"])))

    return {
        "passed": selected_source is not None,
        "selected_source": selected_source,
        "selected_report": selected_report,
        "fallback_used": bool(fallback_reasons and selected_source is not None),
        "fallback_reasons": fallback_reasons,
        "reports": reports,
    }


def evaluate_sync_source(
    records_by_camera: Mapping[str, Sequence[Mapping[str, Any]]],
    aliases: Sequence[str],
    sync_cfg: Mapping[str, Any],
    source: str,
    *,
    diagnostic_local: bool = False,
) -> dict:
    """Evaluate one candidate source across cameras."""

    aliases = [str(alias) for alias in aliases]
    max_span_ns = max_span_ns_for_source(sync_cfg, source)
    reject_trailing = bool(sync_cfg.get("reject_trailing_zero_rows", False))
    source_is_local = source in LOCAL_CLOCK_SOURCES
    kind = "exact" if source in EXACT_SYNC_SOURCES else "timestamp"
    reasons = []

    if source_is_local and not diagnostic_local:
        reasons.append("local VM clock is diagnostic only")

    values_by_alias = {}
    usable_by_alias = {}
    changing_by_alias = {}
    items = []
    for alias in aliases:
        records = list(records_by_camera.get(alias, ()))
        values = [sync_value(record, source) for record in records]
        values_by_alias[alias] = values
        usable_values = [value for value in values if _usable_source_value(source, value)]
        usable_by_alias[alias] = len(usable_values)
        changing_by_alias[alias] = _is_changing(usable_values)
        for record, value in zip(records, values):
            if _usable_source_value(source, value):
                items.append(
                    {
                        "alias": alias,
                        "value": value,
                        "record": record,
                    }
                )

    missing_aliases = [alias for alias in aliases if usable_by_alias.get(alias, 0) == 0]
    if missing_aliases:
        reasons.append("missing usable %s for %s" % (source, ", ".join(missing_aliases)))

    for alias in aliases:
        if usable_by_alias.get(alias, 0) >= 2 and not changing_by_alias.get(alias):
            reasons.append("%s is not changing on %s" % (source, alias))

    if kind == "exact":
        groups = _exact_groups(items, aliases, source, reject_trailing)
    else:
        groups = _timestamp_groups(items, aliases, source, max_span_ns, reject_trailing)

    if not groups:
        reasons.append("no groups formed")

    invalid_groups = [group for group in groups if not bool(group["valid"])]
    valid_groups = [group for group in groups if bool(group["valid"])]
    if invalid_groups:
        reasons.append("%d invalid group(s)" % len(invalid_groups))
    if not valid_groups:
        reasons.append("no valid complete groups")

    drift_ns = _drift_ns(valid_groups, aliases, kind)
    max_drift_ns = sync_cfg.get("max_drift_ns")
    if (
        max_drift_ns is not None
        and drift_ns is not None
        and drift_ns > int(max_drift_ns)
    ):
        reasons.append("drift %d ns exceeds max %d ns" % (drift_ns, int(max_drift_ns)))

    spans = [int(group.get("span_ns", 0)) for group in valid_groups]
    offset_stats = _offset_stats(valid_groups, aliases, kind)
    return {
        "source": source,
        "kind": kind,
        "max_span_ns": max_span_ns,
        "passed": not reasons,
        "reasons": reasons,
        "groups": groups,
        "valid_groups": len(valid_groups),
        "invalid_groups": len(invalid_groups),
        "span_stats": _int_stats(spans),
        "drift_ns": drift_ns,
        "offset_stats": offset_stats,
        "usable_records_by_camera": usable_by_alias,
        "changing_by_camera": changing_by_alias,
    }


def sync_value(record: Mapping[str, Any], source: str) -> Any:
    if source == "pulse_id":
        return record.get("pulse_id")
    return record.get(source)


POLL_TAP_MODES = ("poll", "polling")
SUBSCRIBE_TAP_MODES = ("subscribe", "subscribing", "push", "event")


def tap_capture_mode(tap_cfg: Mapping[str, Any]) -> str:
    """Return the normalized camera tap mode from config."""

    raw_mode = str(tap_cfg.get("mode", "poll")).strip().lower()
    if raw_mode in POLL_TAP_MODES:
        return "poll"
    if raw_mode in SUBSCRIBE_TAP_MODES:
        return "subscribe"
    raise RuntimeError("tap.mode must be 'poll' or 'subscribe', got %r" % raw_mode)


def tap_poll_interval_s(
    tap_cfg: Mapping[str, Any],
    *,
    required: bool = True,
) -> float:
    """Return the polling interval, accepting the legacy ``poll_s`` key."""

    value = tap_cfg.get("last_image_poll_s", tap_cfg.get("poll_s"))
    if value is None:
        if required:
            raise RuntimeError("tap.last_image_poll_s is required in poll mode")
        return 0.0
    return float(value)


def validate_poll_rate(scan_cfg: Mapping[str, Any], tap_cfg: Mapping[str, Any], sync_cfg: Mapping[str, Any]) -> None:
    if tap_capture_mode(tap_cfg) != "poll":
        return

    trigger_hz = float(scan_cfg["trigger_hz"])
    poll_s = tap_poll_interval_s(tap_cfg)
    if trigger_hz <= 0:
        raise RuntimeError("scan.trigger_hz must be positive")
    if poll_s <= 0:
        raise RuntimeError("tap.last_image_poll_s must be positive")

    trigger_period_s = 1.0 / trigger_hz
    if poll_s > trigger_period_s / 4.0 and not bool(
        sync_cfg.get("debug_allow_slow_poll", False)
    ):
        raise RuntimeError(
            "tap.last_image_poll_s too slow for trigger_hz; risk of missing LastImage updates"
        )


def validate_tap_stamp_sources(taps: Sequence[Any], sync_cfg: Mapping[str, Any]) -> None:
    sources = {getattr(tap, "device", "?"): getattr(tap, "stamp_source", None) for tap in taps}

    forbid_local_clock = bool(sync_cfg.get("forbid_local_clock", True))
    if bool(sync_cfg.get("debug_allow_local_clock", False)):
        forbid_local_clock = False
    if forbid_local_clock:
        local = [device for device, source in sources.items() if source == "local_clock"]
        if local:
            raise RuntimeError(
                "local_clock stamp_source forbidden for cameras: %s"
                % ", ".join(sorted(local))
            )

    required = sync_cfg.get("required_stamp_source")
    if required:
        required = str(required)
        mismatches = [
            "%s=%s" % (device, source)
            for device, source in sources.items()
            if source != required
        ]
        if mismatches:
            raise RuntimeError(
                "required stamp_source %r not met: %s"
                % (required, ", ".join(sorted(mismatches)))
            )

    if bool(sync_cfg.get("require_same_stamp_source_all_cameras", False)):
        unique = sorted(set(sources.values()))
        if len(unique) != 1:
            raise RuntimeError("mixed camera stamp_source values: %s" % sources)


def _failed_source_report(source: str, reason: str) -> dict:
    return {
        "source": source,
        "kind": "forbidden",
        "max_span_ns": 0,
        "passed": False,
        "reasons": [reason],
        "groups": [],
        "valid_groups": 0,
        "invalid_groups": 0,
        "span_stats": _int_stats([]),
        "drift_ns": None,
        "offset_stats": {},
        "usable_records_by_camera": {},
        "changing_by_camera": {},
    }


def _exact_groups(
    items: Sequence[Mapping[str, Any]],
    aliases: Sequence[str],
    source: str,
    reject_trailing: bool,
) -> list[dict]:
    grouped = {}
    for item in items:
        grouped.setdefault(item["value"], []).append(item)
    return [
        _group_summary(index, source, "exact", grouped[key], aliases, reject_trailing)
        for index, key in enumerate(sorted(grouped, key=lambda value: repr(value)))
    ]


def _timestamp_groups(
    items: Sequence[Mapping[str, Any]],
    aliases: Sequence[str],
    source: str,
    max_span_ns: int,
    reject_trailing: bool,
) -> list[dict]:
    ordered = sorted(items, key=lambda item: int(item["value"]))
    groups = []
    current = []
    current_min = None
    current_aliases = set()
    for item in ordered:
        value = int(item["value"])
        alias = str(item["alias"])
        if (
            current
            and (
                value - int(current_min) > max_span_ns
                or alias in current_aliases
            )
        ):
            groups.append(current)
            current = []
            current_min = None
            current_aliases = set()
        if current_min is None:
            current_min = value
        current.append(item)
        current_aliases.add(alias)
    if current:
        groups.append(current)

    return [
        _group_summary(index, source, "timestamp", group, aliases, reject_trailing)
        for index, group in enumerate(groups)
    ]


def _group_summary(
    index: int,
    source: str,
    kind: str,
    group: Sequence[Mapping[str, Any]],
    aliases: Sequence[str],
    reject_trailing: bool,
) -> dict:
    errors = []
    by_alias = {}
    records = []
    for item in group:
        alias = str(item["alias"])
        by_alias.setdefault(alias, []).append(item)
        record = dict(item["record"])
        record["_sync_value"] = item["value"]
        records.append(record)

    for alias in aliases:
        count = len(by_alias.get(alias, []))
        if count == 0:
            errors.append("missing camera %s" % alias)
        elif count > 1:
            errors.append("duplicate camera %s" % alias)

    values = [item["value"] for item in group]
    if kind == "exact":
        unique_values = sorted(set(values), key=lambda value: repr(value))
        span_ns = 0
        if len(unique_values) != 1:
            errors.append("mixed %s values: %s" % (source, unique_values))
        sync_key = unique_values[0] if unique_values else None
    else:
        numeric_values = [int(value) for value in values]
        span_ns = max(numeric_values) - min(numeric_values) if numeric_values else 0
        sync_key = min(numeric_values) if numeric_values else None

    if reject_trailing:
        for item in group:
            trailing = _optional_int_like(item["record"].get("trailing_zero_rows"))
            if trailing is not None and trailing > 0:
                errors.append(
                    "%s trailing_zero_rows=%d" % (item["alias"], trailing)
                )

    return {
        "group_index": index,
        "source": source,
        "kind": kind,
        "sync_key": sync_key,
        "valid": not errors,
        "span_ns": int(span_ns),
        "records": records,
        "errors": errors,
    }


def _offset_stats(groups: Sequence[Mapping[str, Any]], aliases: Sequence[str], kind: str) -> dict:
    if kind != "timestamp" or not aliases:
        return {}
    base = aliases[0]
    offsets = {}
    max_min = []
    for group in groups:
        by_alias = {
            str(record["alias"]): int(record["_sync_value"])
            for record in group.get("records", [])
            if record.get("_sync_value") is not None
        }
        if base not in by_alias:
            continue
        values = list(by_alias.values())
        if values:
            max_min.append(max(values) - min(values))
        for alias in aliases[1:]:
            if alias in by_alias:
                offsets.setdefault("%s-%s" % (alias, base), []).append(
                    by_alias[alias] - by_alias[base]
                )
    offsets["max-min"] = max_min
    return {name: _int_stats(nums) for name, nums in offsets.items()}


def _drift_ns(groups: Sequence[Mapping[str, Any]], aliases: Sequence[str], kind: str) -> Optional[int]:
    if kind != "timestamp" or len(groups) < 2 or not aliases:
        return None
    base = aliases[0]
    drifts = []
    offsets_by_alias = {alias: [] for alias in aliases[1:]}
    for group in groups:
        by_alias = {
            str(record["alias"]): int(record["_sync_value"])
            for record in group.get("records", [])
            if record.get("_sync_value") is not None
        }
        if base not in by_alias:
            continue
        for alias in aliases[1:]:
            if alias in by_alias:
                offsets_by_alias[alias].append(by_alias[alias] - by_alias[base])
    for offsets in offsets_by_alias.values():
        if len(offsets) >= 2:
            drifts.append(abs(offsets[-1] - offsets[0]))
    return max(drifts) if drifts else None


def _int_stats(values: Iterable[int]) -> dict:
    nums = [int(value) for value in values]
    if not nums:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(nums),
        "min": min(nums),
        "max": max(nums),
        "mean": statistics.fmean(nums),
    }


def _is_changing(values: Sequence[Any]) -> bool:
    return len(set(repr(value) for value in values)) > 1


def _usable_source_value(source: str, value: Any) -> bool:
    if not _non_empty(value):
        return False
    if source in EXACT_SYNC_SOURCES:
        if source in ("cycleNbGlob", "cycleStampGlob"):
            int_value = _optional_int_like(value)
            return int_value is not None and int_value != 0
        return True
    return _optional_int_like(value) is not None and int(value) > 0


def _candidate_value(value: Any) -> Any:
    int_value = _optional_int_like(value)
    if int_value is not None:
        return int_value
    value = _plain(value)
    shape = getattr(value, "shape", None)
    if shape not in (None, ()):
        return None
    if isinstance(value, (str, bytes)):
        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
    if value is None or isinstance(value, bool):
        return None
    return value


def _non_empty(value: Any) -> bool:
    return value is not None and value not in ("", b"")


def _optional_int_like(value: Any) -> Optional[int]:
    value = _plain(value)
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _plain(value: Any) -> Any:
    if hasattr(value, "value"):
        value = value.value
    shape = getattr(value, "shape", None)
    if shape == () and hasattr(value, "_value"):
        value = value._value
    if hasattr(value, "item") and shape in ((), None):
        try:
            return value.item()
        except ValueError:
            pass
    return value
