"""Index committed lab captures without inferring new camera pairs.

The acquisition coordinator owns pairing. Timestamps remain diagnostic metadata;
the shared capture ID and its durable commit determine each database sample.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from optidaq.data.scientific_png_export import (
    ExportResult,
    SyncAlignment,
    _SavedPngMetadata,
    _finalize_saved_png_dataset,
)


def _integer(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise RuntimeError("%s must be a nonnegative integer" % label)
    return value


def _rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError("%s:%d must contain an object" % (path, number))
            rows.append(row)
    return rows


def _capture_key(capture: Mapping[str, Any]) -> tuple[str, int]:
    capture_id = capture.get("capture_id")
    if not isinstance(capture_id, str) or not capture_id.strip():
        raise RuntimeError("capture_id must be a nonempty string")
    return capture_id, _integer(capture.get("step_id"), "capture step_id")


def finalize_lab_png_dataset(
    run_dir: Union[str, Path],
    *,
    cameras: Optional[Sequence[str]] = None,
) -> ExportResult:
    """Validate complete committed captures and atomically build ``dataset.db``.

    Each capture must contain exactly one saved image for every active camera,
    identical capture metadata, and a matching commit acknowledgement. Orphaned
    files or incomplete commits fail before an existing database is replaced.
    Rebuilding never modifies the PNGs or source metadata.
    """
    run_dir = Path(run_dir).expanduser()
    reader = _SavedPngMetadata(run_dir, cameras)
    if len(reader.cameras) < 2 or len(set(reader.cameras)) != len(reader.cameras):
        raise RuntimeError("lab capture finalization requires distinct paired cameras")

    groups = defaultdict(dict)
    captures = {}
    step_to_capture = {}
    for camera in reader.cameras:
        # The general reader supports older sidecars with repeated keys. Lab
        # captures are append-only and must never silently replace a record.
        raw_rows = _rows(run_dir / (camera + ".sync.jsonl"))
        indices = [_integer(row.get("frame_index"), "frame_index") for row in raw_rows]
        if len(indices) != len(set(indices)):
            raise RuntimeError("duplicate sidecar frame index for %s" % camera)
        image_numbers = set()
        for frame_index, sync in reader.sync_by_camera[camera].items():
            number = _integer(sync.get("source_image_number"), "source_image_number")
            if number in image_numbers:
                raise RuntimeError("duplicate source image number for %s" % camera)
            image_numbers.add(number)
            capture = sync.get("capture")
            if not isinstance(capture, dict):
                raise RuntimeError("saved lab image has no capture metadata")
            capture_id, step_id = _capture_key(capture)
            if capture_id in captures and captures[capture_id] != capture:
                raise RuntimeError("capture metadata differs across camera images: %s" % capture_id)
            if step_id in step_to_capture and step_to_capture[step_id] != capture_id:
                raise RuntimeError("multiple capture IDs use step_id %d" % step_id)
            if camera in groups[capture_id]:
                raise RuntimeError("duplicate camera image in capture %s" % capture_id)
            captures[capture_id] = capture
            step_to_capture[step_id] = capture_id
            groups[capture_id][camera] = frame_index

    if not groups:
        raise RuntimeError("no saved lab captures to finalize")
    camera_set = set(reader.cameras)
    for capture_id, frames in groups.items():
        if set(frames) != camera_set:
            raise RuntimeError("incomplete camera pair for capture %s" % capture_id)

    commit_path = run_dir / "capture_commits.jsonl"
    if not commit_path.exists():
        raise RuntimeError("capture commit log is missing")
    commits = {}
    for commit in _rows(commit_path):
        if commit.get("type") != "capture_commit":
            raise RuntimeError("unexpected record in capture commit log")
        capture_id, step_id = _capture_key(commit)
        if capture_id in commits:
            raise RuntimeError("duplicate capture commit: %s" % capture_id)
        if capture_id not in captures or captures[capture_id]["step_id"] != step_id:
            raise RuntimeError("commit has no matching saved capture: %s" % capture_id)
        if any(key not in commit or commit[key] != value
               for key, value in captures[capture_id].items()):
            raise RuntimeError("commit metadata differs from saved capture: %s" % capture_id)
        saved = commit.get("saved")
        if not isinstance(saved, list) or len(saved) != len(reader.cameras):
            raise RuntimeError("commit must acknowledge every camera once")
        acknowledged = set()
        for ack in saved:
            if not isinstance(ack, dict):
                raise RuntimeError("invalid saved-image acknowledgement")
            camera = ack.get("camera")
            if camera not in camera_set or camera in acknowledged:
                raise RuntimeError("duplicate or unknown camera in capture commit")
            acknowledged.add(camera)
            frame_index = _integer(ack.get("frame_index"), "ack frame_index")
            if frame_index != groups[capture_id][camera]:
                raise RuntimeError("commit frame index differs from captured image")
            sync = reader.sync_by_camera[camera][frame_index]
            if _integer(ack.get("imageNumber"), "ack imageNumber") != sync["source_image_number"]:
                raise RuntimeError("commit source image number differs from captured image")
            relative_path = str(reader.paths[camera][frame_index].relative_to(run_dir))
            if ack.get("png_path") != relative_path:
                raise RuntimeError("commit PNG path differs from captured image")
            if "capture_id" in ack and ack["capture_id"] != capture_id:
                raise RuntimeError("saved-image acknowledgement has a different capture ID")
        commits[capture_id] = commit
    if set(commits) != set(captures):
        raise RuntimeError("saved images exist without a completed capture commit")

    frame_to_sample = {camera: {} for camera in reader.cameras}
    frame_to_time = {camera: {} for camera in reader.cameras}
    frame_to_value = {camera: {} for camera in reader.cameras}
    spans = {}
    for capture_id, frames in groups.items():
        sample_id = captures[capture_id]["step_id"] + 1
        times = []
        for camera, frame_index in frames.items():
            frame_to_sample[camera][frame_index] = sample_id
            frame_to_value[camera][frame_index] = capture_id
            timestamp = reader.sync_by_camera[camera][frame_index].get("image_ready_ns")
            if type(timestamp) is int and timestamp > 0:
                frame_to_time[camera][frame_index] = timestamp
                times.append(timestamp)
        if len(times) == len(reader.cameras):
            spans[sample_id] = max(times) - min(times)
    alignment = SyncAlignment(
        source="capture_id", tolerance_ns=0,
        frame_to_sample_id=frame_to_sample,
        frame_to_sync_time_ns=frame_to_time,
        sample_to_span_ns=spans,
        group_count=len(groups), complete_group_count=len(groups),
        incomplete_group_count=0, dropped_group_count=0, dropped_frame_count=0,
    )
    report = {
        "mode": "committed_lab_capture", "selected_source": "capture_id", "passed": True,
        "capture_count": len(groups), "sample_id_rule": "step_id + 1",
        "diagnostic_timestamp_source": "image_ready_ns",
        "physical_exposure_synchronization_verified": False,
    }
    return _finalize_saved_png_dataset(
        reader, alignment, report, frame_to_value,
        contract="committed lab captures; sample_id = step_id + 1; pairing fixed by capture_id; timestamps are diagnostic only",
    )
