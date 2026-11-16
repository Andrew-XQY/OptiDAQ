"""DMD event facts and approximate offline associations with saved frames.

Version 1 uses host Unix nanoseconds, as camera ``local_receive_ns`` does.
``settle_complete`` means the software settling wait finished, not measured
optical readiness. Receive-time association never proves exposure alignment.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import json
from pathlib import Path
import tempfile
import time

from optidaq.daq.offline_validate import read_sync_sidecars


EVENT_LOG_FILENAME = "actuator.events.jsonl"
FRAME_INDEX_FILENAME = "actuator_frame_index.jsonl"
SCHEMA_VERSION = 1
CLOCK = "host_unix_ns"


def _nonnegative_int(value):
    return type(value) is int and value >= 0


def _validate_event(record):
    if not isinstance(record, dict):
        raise ValueError("event must be an object")
    version = record.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ValueError("unsupported actuator event schema_version")
    if record.get("actuator") != "dmd" or record.get("clock") != CLOCK:
        raise ValueError("unsupported actuator or event clock")
    event = record.get("event")
    if event not in ("change_begin", "settle_complete", "end"):
        raise ValueError("unsupported actuator event: %r" % event)
    valid_step = (
        record.get("step_id") is None
        if event == "end" else _nonnegative_int(record.get("step_id"))
    )
    if "step_id" not in record or not valid_step:
        raise ValueError("end requires a null step_id; other events require an integer step_id")
    if not _nonnegative_int(record.get("host_time_ns")):
        raise ValueError("host_time_ns must be a nonnegative integer")
    if not isinstance(record.get("metadata"), dict):
        raise ValueError("event metadata must be an object")


def append_actuator_event(
    event_path, event, step_id=None, metadata=None, host_time_ns=None
):
    """Append one event and close the file so it is visible immediately.

    Closing flushes Python buffers, but does not guarantee survival of power
    loss. Additional source facts live in metadata.
    """
    record = {
        "schema_version": SCHEMA_VERSION,
        "actuator": "dmd",
        "event": event,
        "step_id": step_id,
        "host_time_ns": time.time_ns() if host_time_ns is None else host_time_ns,
        "clock": CLOCK,
        "metadata": {} if metadata is None else dict(metadata),
    }
    _validate_event(record)
    line = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
    with Path(event_path).open("a", encoding="utf-8") as handle:
        handle.write(line)
    return record


def _read_intervals(path):
    intervals = []
    boundaries = set()
    previous_time = None
    active_step = None
    settled_at = None
    seen_steps = set()
    ended = False
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                _validate_event(record)
                timestamp = record["host_time_ns"]
                if previous_time is not None and timestamp < previous_time:
                    raise ValueError("host event clock moved backwards")
                if ended:
                    raise ValueError("event follows end")
                event, step_id = record["event"], record["step_id"]
                if event in ("change_begin", "end"):
                    boundaries.add(timestamp)
                    if settled_at is not None:
                        intervals.append((settled_at, timestamp, active_step))
                    settled_at = None
                    active_step = step_id
                    if event == "change_begin":
                        if step_id in seen_steps:
                            raise ValueError("repeated step_id")
                        seen_steps.add(step_id)
                    else:
                        ended = True
                else:
                    if active_step != step_id or settled_at is not None:
                        raise ValueError("settle_complete requires its preceding change_begin")
                    settled_at = timestamp
                previous_time = timestamp
            except (ValueError, TypeError) as exc:
                raise ValueError("%s:%d: %s" % (path, line_number, exc)) from exc
    # An unclosed final step has no safe upper boundary, even for approximation.
    return intervals, boundaries


def write_actuator_frame_index(run_dir, aliases=None):
    """Regenerate associations without modifying event or camera source files.

    Only completed software-stable intervals are considered. All matches are
    approximate: camera host receive/processing time is not exposure time.
    A receive-clock regression leaves the entire affected camera unmatched.
    """
    root = Path(run_dir)
    if aliases is None:
        aliases = sorted(
            path.name[:-len(".sync.jsonl")] for path in root.glob("*.sync.jsonl")
        )
    aliases = list(aliases)
    if not aliases or len(set(aliases)) != len(aliases):
        raise ValueError("expected at least one unique camera alias")
    intervals, boundaries = _read_intervals(root / EVENT_LOG_FILENAME)
    starts = [interval[0] for interval in intervals]
    cameras = read_sync_sidecars(root, aliases)
    output_path = root / FRAME_INDEX_FILENAME
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=root,
            prefix=".actuator_frame_index-", suffix=".tmp", delete=False,
        ) as output:
            temporary_path = Path(output.name)
            for alias in aliases:
                records = cameras[alias]
                times = [
                    record["local_receive_ns"] for record in records
                    if _nonnegative_int(record.get("local_receive_ns"))
                ]
                regressed = any(b < a for a, b in zip(times, times[1:]))
                for record in records:
                    if not _nonnegative_int(record.get("frame_index")):
                        raise ValueError("invalid frame_index in %s.sync.jsonl" % alias)
                    received = record.get("local_receive_ns")
                    step_id = None
                    reason = "no_closed_stable_interval"
                    if regressed:
                        reason = "receive_clock_regression"
                    elif not _nonnegative_int(received):
                        reason = "missing_or_invalid_receive_time"
                    elif received in boundaries:
                        reason = "transition_boundary"
                    else:
                        index = bisect_right(starts, received) - 1
                        if index >= 0 and received < intervals[index][1]:
                            step_id = intervals[index][2]
                    entry = {
                        "schema_version": SCHEMA_VERSION,
                        "actuator": "dmd",
                        "device": alias,
                        "frame_index": record["frame_index"],
                        "source_image_number": record.get(
                            "source_image_number", record.get("imageNumber")
                        ),
                        "acq_stamp_ns": record.get("acq_stamp_ns"),
                        "local_receive_ns": received,
                        "step_id": step_id,
                        "method": "host_receive_time",
                        "quality": "approximate" if step_id is not None else "unmatched",
                    }
                    if step_id is None:
                        entry["reason"] = reason
                    output.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(write_actuator_frame_index(args.run_dir))


if __name__ == "__main__":
    main()
