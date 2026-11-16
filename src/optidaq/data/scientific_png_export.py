"""Export raw runs as per-camera PNGs and SQLite metadata, with optional image transforms."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from optidaq.daq.sync_tools import EXACT_SYNC_SOURCES, evaluate_sync_sources
from optidaq.data.hooks import (
    HookContext,
    build_image_hooks,
    build_metadata_hooks,
)
from optidaq.data.png_io import write_png_gray
from optidaq.data.raw_run import (
    MISSING_U64,
    RawFrame,
    RawRunReader,
    _camera_configs_from_resolved,
    _read_json,
    _read_sync_sidecar,
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS run_info (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS cameras (
    camera             TEXT PRIMARY KEY,
    raw_dtype          TEXT,
    height             INTEGER,
    width              INTEGER,
    payload_nbytes     INTEGER,
    camera_pixel_format TEXT,
    settings_json      TEXT
);

CREATE TABLE IF NOT EXISTS images (
    image_id                 TEXT PRIMARY KEY,
    sample_id                INTEGER,
    frame_sample_id          INTEGER,
    run_id                   TEXT,
    camera                   TEXT REFERENCES cameras(camera),
    png_path                 TEXT,
    png_dtype                TEXT,
    raw_dtype                TEXT,
    height                   INTEGER,
    width                    INTEGER,
    frame_index              INTEGER,
    global_record_index      INTEGER,
    segment_index            INTEGER,
    record_index_in_segment  INTEGER,
    source_image_number      INTEGER,
    acq_stamp_ns             INTEGER,
    sync_time_ns             INTEGER,
    sync_group_span_ns       INTEGER,
    pulse_id                 INTEGER,
    chosen_sync_source       TEXT,
    chosen_sync_value        TEXT,
    raw_segment_path         TEXT,
    byte_offset              INTEGER,
    stat_min                 REAL,
    stat_max                 REAL,
    stat_mean                REAL,
    stat_sum                 REAL,
    centroid_x               REAL,
    centroid_y               REAL,
    intensity_in_roi         REAL,
    intensity_cropped_out    REAL,
    roi_energy_fraction      REAL,
    raw_header_json          TEXT,
    sync_json                TEXT,
    hook_metadata_json       TEXT,
    extra_json               TEXT,
    created_utc              TEXT
);

CREATE INDEX IF NOT EXISTS ix_images_run_camera ON images(run_id, camera);
CREATE INDEX IF NOT EXISTS ix_images_frame ON images(camera, frame_index);
CREATE INDEX IF NOT EXISTS ix_images_stamp ON images(acq_stamp_ns);
CREATE INDEX IF NOT EXISTS ix_images_sync_time ON images(sync_time_ns);
CREATE INDEX IF NOT EXISTS ix_images_sync ON images(chosen_sync_source, chosen_sync_value);
CREATE INDEX IF NOT EXISTS ix_images_camera_run_sample ON images(camera, run_id, sample_id);
"""


@dataclass
class ExportConfig:
    run_dir: Path
    out_dir: Path
    run_id: Optional[str] = None
    cameras: Optional[list[str]] = None
    overwrite: bool = False
    include_part_files: bool = False
    png_compress_level: int = 6
    show_progress: bool = True
    crop_enabled: bool = True
    sync_alignment_source: str = "image_ready_ns"
    sync_alignment_tolerance_ns: int = 50_000_000
    image_hooks: list[Any] = field(default_factory=list)
    metadata_hooks: list[Any] = field(default_factory=list)
    extra_info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SyncAlignment:
    """Frame-to-sample map built from one external-trigger sync timestamp."""

    source: str
    tolerance_ns: int
    frame_to_sample_id: Mapping[str, Mapping[int, int]]
    frame_to_sync_time_ns: Mapping[str, Mapping[int, int]]
    sample_to_span_ns: Mapping[int, int]
    group_count: int
    complete_group_count: int
    incomplete_group_count: int
    dropped_group_count: int
    dropped_frame_count: int


@dataclass(frozen=True)
class ExportResult:
    out_dir: Path
    db_path: Path
    run_id: str
    image_count: int
    counts_by_camera: Mapping[str, int]


def load_export_config(path: Union[str, Path], **overrides: Any) -> ExportConfig:
    """Load a standalone JSON config and apply non-None keyword overrides."""

    with open(Path(path).expanduser(), "r", encoding="utf-8") as handle:
        data = json.load(handle)
    for key, value in overrides.items():
        if value is not None:
            data[key] = value
    return config_from_mapping(data)


def config_from_mapping(data: Mapping[str, Any]) -> ExportConfig:
    png_cfg = data.get("png", {}) or {}
    hooks_cfg = data.get("hooks", {}) or {}
    sync_cfg = data.get("sync_alignment", {}) or {}
    return ExportConfig(
        run_dir=Path(_require(data, "run_dir")).expanduser(),
        out_dir=Path(_require(data, "out_dir")).expanduser(),
        run_id=data.get("run_id"),
        cameras=[str(camera) for camera in data["cameras"]] if data.get("cameras") else None,
        overwrite=bool(data.get("overwrite", False)),
        include_part_files=bool(data.get("include_part_files", False)),
        png_compress_level=int(png_cfg.get("compress_level", data.get("png_compress_level", 6))),
        show_progress=bool(data.get("show_progress", data.get("progress", True))),
        crop_enabled=bool(data.get("crop_enabled", True)),
        sync_alignment_source=str(
            sync_cfg.get("source", data.get("sync_alignment_source", "image_ready_ns"))
        ),
        sync_alignment_tolerance_ns=int(
            sync_cfg.get(
                "tolerance_ns",
                data.get("sync_alignment_tolerance_ns", 50_000_000),
            )
        ),
        image_hooks=hooks_cfg.get("image", data.get("image_hooks", [])) or [],
        metadata_hooks=list(hooks_cfg.get("metadata", data.get("metadata_hooks", [])) or []),
        extra_info=dict(data.get("extra_info", {}) or {}),
    )


def _resolve_camera_hooks(image_hooks: Any, camera: str) -> list:
    """Resolve the image-hook list for one camera.

    - ``list`` -> applied to every camera (default; current behavior).
    - ``dict`` -> per-camera: ``{"cam2": [...], "default": [...]}``. The camera's own
      key wins, else ``default``/``*``, else no image hooks for that camera.
    """

    if isinstance(image_hooks, Mapping):
        for key in (camera, "default", "*"):
            if key in image_hooks:
                return list(image_hooks[key] or [])
        return []
    return list(image_hooks or [])


def _extract_crop_box(specs: Any) -> Optional[tuple]:
    """Return the first configured crop_box for crop-energy metadata, or None."""

    for spec in specs or []:
        if isinstance(spec, Mapping) and spec.get("name") == "crop_box":
            p = spec.get("params", {}) or {}
            return (p.get("x1"), p.get("y1"), p.get("x2"), p.get("y2"))
    return None


def _build_sync_alignment(
    reader: RawRunReader,
    *,
    source: str,
    tolerance_ns: int,
    allow_missing_sync: bool = False,
) -> SyncAlignment:
    """Assign cross-camera sample IDs by clustering frames on a sync timestamp."""

    if tolerance_ns <= 0:
        raise ValueError("sync alignment tolerance must be positive")

    source = str(source)
    records = []
    raw_counts = {camera: reader.raw_record_count(camera) for camera in reader.cameras}
    for camera in reader.cameras:
        sync_rows = reader.sync_by_camera.get(camera, {})
        by_frame = {}
        for frame_index, sync in sync_rows.items():
            sync_time_ns = _sync_time_from_record(sync, source)
            if sync_time_ns is not None:
                by_frame[int(frame_index)] = int(sync_time_ns)

        expected = raw_counts[camera]
        if len(by_frame) != expected and not allow_missing_sync:
            raise RuntimeError(
                "camera %s has %d/%d frames with sync source %r; cannot build "
                "reliable cross-camera sample IDs"
                % (camera, len(by_frame), expected, source)
            )
        if allow_missing_sync:
            raw_counts[camera] = len(by_frame)

        for frame_index, sync_time_ns in by_frame.items():
            records.append((sync_time_ns, camera, frame_index))

    if not records and not allow_missing_sync:
        raise RuntimeError("no sync records found for source %r" % source)

    records.sort(key=lambda item: item[0])
    groups = []
    current = []
    group_start = None
    for record in records:
        sync_time_ns = record[0]
        if current and sync_time_ns - group_start > tolerance_ns:
            groups.append(current)
            current = []
            group_start = None
        if not current:
            group_start = sync_time_ns
        current.append(record)
    if current:
        groups.append(current)

    frame_to_sample_id = {camera: {} for camera in reader.cameras}
    frame_to_sync_time_ns = {camera: {} for camera in reader.cameras}
    sample_to_span_ns = {}
    complete_count = 0
    dropped_group_count = 0
    dropped_frame_count = 0
    dropped_counts = {camera: 0 for camera in reader.cameras}
    camera_set = set(reader.cameras)

    for sample_id, group in enumerate(groups, start=1):
        seen = {}
        times = [record[0] for record in group]
        span_ns = int(max(times) - min(times)) if times else 0
        if span_ns > tolerance_ns:
            raise RuntimeError(
                "internal sync grouping error: sample_id=%d span=%d ns exceeds "
                "tolerance=%d ns" % (sample_id, span_ns, tolerance_ns)
            )

        group_cameras = [record[1] for record in group]
        if len(group_cameras) != len(set(group_cameras)):
            dropped_group_count += 1
            dropped_frame_count += len(group)
            for _, camera, _ in group:
                dropped_counts[camera] += 1
            continue

        for sync_time_ns, camera, frame_index in group:
            seen[camera] = frame_index
            frame_to_sample_id[camera][frame_index] = sample_id
            frame_to_sync_time_ns[camera][frame_index] = sync_time_ns

        sample_to_span_ns[sample_id] = span_ns
        if set(seen) == camera_set:
            complete_count += 1

    for camera in reader.cameras:
        accounted = len(frame_to_sample_id[camera]) + dropped_counts[camera]
        if accounted != raw_counts[camera]:
            raise RuntimeError(
                "camera %s sync alignment accounted for %d/%d frames"
                % (camera, accounted, raw_counts[camera])
            )

    if dropped_group_count:
        sys.stderr.write(
            "warning: sync alignment dropped %d ambiguous group(s) containing "
            "%d frame(s)\n" % (dropped_group_count, dropped_frame_count)
        )

    return SyncAlignment(
        source=source,
        tolerance_ns=int(tolerance_ns),
        frame_to_sample_id=frame_to_sample_id,
        frame_to_sync_time_ns=frame_to_sync_time_ns,
        sample_to_span_ns=sample_to_span_ns,
        group_count=len(groups),
        complete_group_count=complete_count,
        incomplete_group_count=len(groups) - complete_count,
        dropped_group_count=dropped_group_count,
        dropped_frame_count=dropped_frame_count,
    )


def _sync_time_from_record(sync: Mapping[str, Any], source: str) -> Optional[int]:
    if not sync:
        return None
    if sync.get(source) is not None:
        return _sync_int_or_none(sync.get(source))
    if sync.get("chosen_sync_source") == source and sync.get("chosen_sync_value") is not None:
        return _sync_int_or_none(sync.get("chosen_sync_value"))
    return None


def _sync_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = text
    return int(value)


def export_run_to_png_dataset(config: ExportConfig) -> ExportResult:
    """Read a raw run, apply configured hooks, and write PNGs with SQLite metadata."""

    reader = RawRunReader(
        config.run_dir,
        cameras=config.cameras,
        include_part_files=config.include_part_files,
    )
    run_id = config.run_id or str(
        reader.resolved_config.get("run_id") or config.run_dir.name
    )
    out_dir = config.out_dir
    db_path = out_dir / "dataset.db"
    _prepare_output(out_dir, db_path, overwrite=config.overwrite)
    alignment = _build_sync_alignment(
        reader,
        source=config.sync_alignment_source,
        tolerance_ns=config.sync_alignment_tolerance_ns,
    )

    image_specs_by_camera = {
        camera: _resolve_camera_hooks(config.image_hooks, camera)
        if config.crop_enabled
        else []
        for camera in reader.cameras
    }
    image_hooks_by_camera = {
        camera: build_image_hooks(specs)
        for camera, specs in image_specs_by_camera.items()
    }
    crop_box_by_camera = {
        camera: _extract_crop_box(specs)
        for camera, specs in image_specs_by_camera.items()
    }
    metadata_hooks = build_metadata_hooks(config.metadata_hooks)
    progress = _ProgressBar(
        sum(reader.raw_record_count(camera) for camera in reader.cameras),
        enabled=config.show_progress,
    )

    conn = sqlite3.connect(str(db_path))
    counts_by_camera = {camera: 0 for camera in reader.cameras}
    image_count = 0
    processed_count = 0
    try:
        _init_db(conn, reader, config, run_id, alignment)
        created_utc = datetime.now(timezone.utc).isoformat()
        for camera in reader.cameras:
            for frame in reader.iter_camera_frames(camera):
                processed_count += 1
                frame_index = int(frame.header.get("frame_index", frame.global_record_index))
                if frame_index not in alignment.frame_to_sample_id[camera]:
                    progress.update(processed_count)
                    continue
                row = _write_frame(
                    out_dir=out_dir,
                    run_dir=reader.run_dir,
                    run_id=run_id,
                    frame=frame,
                    sample_id=alignment.frame_to_sample_id[camera][frame_index],
                    sync_time_ns=alignment.frame_to_sync_time_ns[camera][frame_index],
                    sync_group_span_ns=alignment.sample_to_span_ns[
                        alignment.frame_to_sample_id[camera][frame_index]
                    ],
                    image_hooks=image_hooks_by_camera[camera],
                    applied_crop=crop_box_by_camera[camera],
                    metadata_hooks=metadata_hooks,
                    compress_level=config.png_compress_level,
                    created_utc=created_utc,
                )
                _insert_image(conn, row)
                counts_by_camera[camera] += 1
                image_count += 1
                progress.update(processed_count)
        conn.commit()
    finally:
        progress.done(processed_count)
        conn.close()

    return ExportResult(
        out_dir=out_dir,
        db_path=db_path,
        run_id=run_id,
        image_count=image_count,
        counts_by_camera=counts_by_camera,
    )


class _SavedPngMetadata:
    """Metadata-only adapter for the existing alignment and database helpers."""

    def __init__(self, run_dir: Path, cameras: Optional[Sequence[str]]) -> None:
        self.run_dir = run_dir
        self.resolved_config = _read_json(run_dir / "resolved_config.json")
        configs = _camera_configs_from_resolved(self.resolved_config)
        self.cameras = list(cameras) if cameras is not None else list(configs)
        self.camera_configs = {}
        self.manifests = {}
        self.paths = {}
        self.sync_by_camera = {}
        pattern = re.compile(r"^(\d+)_(\d+)\.png$")
        for camera in self.cameras:
            if camera not in configs:
                raise RuntimeError("camera not in resolved_config.json: %s" % camera)
            cfg = configs[camera]
            shape = tuple(int(n) for n in cfg["shape"])
            dtype = np.dtype(cfg["dtype"])
            if len(shape) not in (2, 3) or (len(shape) == 3 and (shape[2] != 3 or dtype != np.dtype("uint8"))):
                raise RuntimeError("PNG source must be grayscale or uint8 RGB: %s" % camera)
            self.camera_configs[camera] = SimpleNamespace(
                dtype=dtype, shape=shape,
                payload_nbytes=int(cfg.get("payload_nbytes") or 0), settings=dict(cfg),
            )
            paths = {}
            for path in (run_dir / camera).glob("*.png"):
                match = pattern.fullmatch(path.name)
                if match is None:
                    raise RuntimeError("unrecognized PNG filename: %s" % path)
                frame_index = int(match.group(2))
                if frame_index in paths:
                    raise RuntimeError("duplicate PNG frame index for %s: %d" % (camera, frame_index))
                paths[frame_index] = path
            self.paths[camera] = paths
            rows = _read_sync_sidecar(run_dir, camera)
            if set(rows) != set(paths):
                raise RuntimeError(
                    "PNG/sidecar frame mismatch for %s: %d PNGs, %d sync rows"
                    % (camera, len(paths), len(rows))
                )
            for frame_index, path in paths.items():
                if int(path.stem.split("_")[0]) != int(rows[frame_index]["acq_stamp_ns"]):
                    raise RuntimeError("PNG timestamp differs from sidecar: %s" % path)
            self.sync_by_camera[camera] = rows
            manifest_path = run_dir / ("%s.png.json" % camera)
            if manifest_path.exists():
                manifest = _read_json(manifest_path)
                if manifest.get("schema_version") != 1:
                    raise RuntimeError("unsupported PNG manifest: %s" % manifest_path)
                stored = manifest["stored"]
                shape = stored["shape"]
                dtype = np.dtype(stored["dtype"])
                if len(shape) not in (2, 3) or any(int(n) <= 0 for n in shape):
                    raise RuntimeError("invalid stored PNG shape: %s" % manifest_path)
                if dtype not in (np.dtype("uint8"), np.dtype("uint16")) or (len(shape) == 3 and (shape[2] != 3 or dtype != np.dtype("uint8"))):
                    raise RuntimeError("invalid stored PNG dtype: %s" % manifest_path)
                self.manifests[camera] = manifest
            elif paths:
                raise RuntimeError("PNG manifest missing: %s" % manifest_path)

    def raw_record_count(self, camera: str) -> int:
        return len(self.paths[camera])


def _configured_png_alignment(reader, sync_cfg):
    """Use precisely the acquisition framework's configured source selection."""
    records = {
        camera: [dict(row, alias=camera) for _, row in sorted(rows.items())]
        for camera, rows in reader.sync_by_camera.items()
    }
    report = evaluate_sync_sources(records, reader.cameras, sync_cfg)
    selected = report.get("selected_report") or {}
    frame_to_sample = {camera: {} for camera in reader.cameras}
    frame_to_time = {camera: {} for camera in reader.cameras}
    frame_to_value = {camera: {} for camera in reader.cameras}
    spans = {}
    groups = selected.get("groups", []) if report["passed"] else []
    for sample_id, group in enumerate(groups, start=1):
        if not group["valid"]:
            continue
        spans[sample_id] = int(group["span_ns"])
        for row in group["records"]:
            camera, frame_index = str(row["alias"]), int(row["frame_index"])
            frame_to_sample[camera][frame_index] = sample_id
            frame_to_value[camera][frame_index] = row["_sync_value"]
            if group["kind"] == "timestamp":
                frame_to_time[camera][frame_index] = int(row["_sync_value"])
    alignment = SyncAlignment(
        source=str(report.get("selected_source") or ""),
        tolerance_ns=int(selected.get("max_span_ns", 0)),
        frame_to_sample_id=frame_to_sample,
        frame_to_sync_time_ns=frame_to_time,
        sample_to_span_ns=spans,
        group_count=len(groups), complete_group_count=len(spans),
        incomplete_group_count=0, dropped_group_count=0, dropped_frame_count=0,
    )
    return alignment, report, frame_to_value


def finalize_png_dataset(
    run_dir: Union[str, Path],
    *,
    cameras: Optional[Sequence[str]] = None,
    sync_alignment_source: Optional[str] = None,
    sync_alignment_tolerance_ns: Optional[int] = None,
) -> ExportResult:
    """Index already-saved PNGs without reading or transforming their pixels.

    Called after the writers close. The database is replaced atomically, so a
    failed rebuild leaves an earlier database intact. Missing or ambiguous sync
    timestamps produce NULL sample IDs; they never become cross-camera pairs.
    ``output.png.sync_alignment`` selects the exporter's timestamp matcher.
    Without that block, the run's resolved acquisition sync policy is retained.
    Explicit source/tolerance arguments override the PNG settings.
    """

    run_dir = Path(run_dir).expanduser()
    reader = _SavedPngMetadata(run_dir, cameras)
    run_id = str(reader.resolved_config.get("run_id") or run_dir.name)
    inner_config = reader.resolved_config.get("config") or {}
    png_cfg = inner_config.get("output", {}).get("png", {})
    if not isinstance(png_cfg, Mapping):
        raise ValueError("output.png must be an object")
    explicit_alignment = sync_alignment_source is not None or sync_alignment_tolerance_ns is not None
    if "sync_alignment" in png_cfg or explicit_alignment:
        pairing = png_cfg.get("sync_alignment", {
            "source": "image_ready_ns", "tolerance_ns": 50_000_000,
        })
        if not isinstance(pairing, Mapping) or not {"source", "tolerance_ns"}.issubset(pairing):
            raise ValueError("output.png.sync_alignment requires source and tolerance_ns")
        source = pairing["source"] if sync_alignment_source is None else sync_alignment_source
        tolerance = pairing["tolerance_ns"] if sync_alignment_tolerance_ns is None else sync_alignment_tolerance_ns
        if not isinstance(source, str) or not source.strip():
            raise ValueError("PNG sync alignment source must be a nonempty string")
        if type(tolerance) is not int or tolerance <= 0:
            raise ValueError("PNG sync alignment tolerance_ns must be a positive integer")
        if source in EXACT_SYNC_SOURCES:
            raise ValueError("exact sync sources must use the resolved sync policy, not timestamp tolerance")
        alignment = _build_sync_alignment(
            reader,
            source=source,
            tolerance_ns=tolerance,
            allow_missing_sync=True,
        )
        sync_report = {
            "mode": "explicit_timestamp_alignment" if explicit_alignment else "png_timestamp_alignment",
            "selected_source": alignment.source,
            "tolerance_ns": alignment.tolerance_ns,
        }
        sync_values = {
            camera: {index: _sync_time_from_record(row, alignment.source) for index, row in rows.items()}
            for camera, rows in reader.sync_by_camera.items()
        }
    else:
        sync_cfg = reader.resolved_config.get("sync", inner_config.get("sync", {})) or {}
        alignment, sync_report, sync_values = _configured_png_alignment(reader, sync_cfg)
    return _finalize_saved_png_dataset(reader, alignment, sync_report, sync_values)


def _finalize_saved_png_dataset(
    reader: _SavedPngMetadata,
    alignment: SyncAlignment,
    sync_report: Mapping[str, Any],
    sync_values: Mapping[str, Mapping[int, Any]],
    *,
    contract: str = "saved PNGs indexed without pixel processing; sample_id = sync-aligned group, NULL for missing/ambiguous sync; frame_sample_id = frame_index + 1",
) -> ExportResult:
    """Shared atomic index writer; callers supply their validated pairing map."""
    run_dir = reader.run_dir
    run_id = str(reader.resolved_config.get("run_id") or run_dir.name)
    inner_config = reader.resolved_config.get("config") or {}
    png_cfg = inner_config.get("output", {}).get("png", {})
    if not isinstance(png_cfg, Mapping):
        raise ValueError("output.png must be an object")
    actuator_rows = {}
    actuator_path = run_dir / "actuator_frame_index.jsonl"
    if actuator_path.exists():
        with open(actuator_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    actuator_rows[(row["device"], int(row["frame_index"]))] = row

    db_path = run_dir / "dataset.db"
    counts = {camera: 0 for camera in reader.cameras}
    # Record the processing that actually produced these PNGs, using the same
    # provenance format as offline export. The manifest takes precedence over
    # configuration so a later rebuild cannot relabel already-saved pixels.
    image_hooks = {
        camera: reader.manifests.get(camera, {}).get(
            "transforms", _resolve_camera_hooks(png_cfg.get("transforms"), camera)
        ) for camera in reader.cameras
    }
    metadata_by_camera = {
        camera: reader.manifests.get(camera, {}).get(
            "metadata_hooks", _resolve_camera_hooks(png_cfg.get("metadata_hooks"), camera)
        ) for camera in reader.cameras
    }
    metadata_hooks = next(iter(metadata_by_camera.values()), [])
    if any(specs != metadata_hooks for specs in metadata_by_camera.values()):
        metadata_hooks = metadata_by_camera
    levels = {manifest.get("compress_level", png_cfg.get("compress_level", 6))
              for manifest in reader.manifests.values()}
    config = ExportConfig(
        run_dir=run_dir, out_dir=run_dir, run_id=run_id, cameras=reader.cameras,
        show_progress=False, crop_enabled=any(image_hooks.values()),
        image_hooks=image_hooks, metadata_hooks=metadata_hooks,
        png_compress_level=next(iter(levels)) if len(levels) == 1 else png_cfg.get("compress_level", 6),
        sync_alignment_source=alignment.source,
        sync_alignment_tolerance_ns=alignment.tolerance_ns,
    )
    fd, temp_name = tempfile.mkstemp(prefix=".dataset-", suffix=".db", dir=run_dir)
    os.close(fd)
    temp_path = Path(temp_name)
    conn = sqlite3.connect(str(temp_path))
    try:
        _init_db(conn, reader, config, run_id, alignment)
        conn.executemany(
            "INSERT OR REPLACE INTO run_info(key, value) VALUES(?, ?)",
            [
                ("source_format", "clear2026_png"),
                ("contract", contract),
                ("png_manifests_json", json.dumps(reader.manifests, sort_keys=True)),
                ("sync_report_json", json.dumps(sync_report, sort_keys=True)),
                ("sync_alignment_unmatched_frame_count", str(
                    sum(len(paths) for paths in reader.paths.values())
                    - sum(len(rows) for rows in alignment.frame_to_sample_id.values())
                )),
            ],
        )
        created_utc = datetime.now(timezone.utc).isoformat()
        for camera in reader.cameras:
            for frame_index, path in sorted(reader.paths[camera].items()):
                sync = reader.sync_by_camera[camera][frame_index]
                stored = reader.manifests[camera]["stored"]
                png = sync.get("png", {})
                if png.get("shape", stored["shape"]) != stored["shape"] or png.get("dtype", stored["dtype"]) != stored["dtype"]:
                    raise RuntimeError("PNG descriptor changed within run: %s" % path)
                metadata = dict(png.get("metadata") or {})
                sample_id = alignment.frame_to_sample_id[camera].get(frame_index)
                stamp = int(sync["acq_stamp_ns"])
                raw_pulse_id = sync.get("raw_header_pulse_id", sync.get("pulse_id"))
                pulse_id = _pulse_or_none(raw_pulse_id)
                header = {key: sync.get(key) for key in (
                    "frame_index", "acq_stamp_ns", "source_image_number", "pulse_id"
                )}
                header["pulse_id"] = raw_pulse_id
                actuator = actuator_rows.get((camera, frame_index))
                row = {
                    "image_id": _image_id(camera, stamp, frame_index),
                    "sample_id": sample_id,
                    "frame_sample_id": frame_index + 1,
                    "run_id": run_id,
                    "camera": camera,
                    "png_path": str(path.relative_to(run_dir)),
                    "png_dtype": stored["dtype"],
                    "raw_dtype": str(reader.camera_configs[camera].dtype),
                    "height": int(stored["shape"][0]),
                    "width": int(stored["shape"][1]),
                    "frame_index": frame_index,
                    "global_record_index": frame_index,
                    "segment_index": 0,
                    "record_index_in_segment": frame_index,
                    "source_image_number": _int_or_none(sync.get("source_image_number")),
                    "acq_stamp_ns": stamp,
                    "sync_time_ns": alignment.frame_to_sync_time_ns[camera].get(frame_index),
                    "sync_group_span_ns": alignment.sample_to_span_ns.get(sample_id),
                    "pulse_id": pulse_id,
                    "chosen_sync_source": alignment.source or None,
                    "chosen_sync_value": _json_or_none(sync_values[camera].get(frame_index)),
                    "raw_header_json": json.dumps(header, sort_keys=True),
                    "sync_json": json.dumps(sync, sort_keys=True),
                    "hook_metadata_json": json.dumps(metadata, sort_keys=True),
                    "extra_json": json.dumps({"actuator": actuator}, sort_keys=True) if actuator else None,
                    "created_utc": created_utc,
                }
                for key in ("stat_min", "stat_max", "stat_mean", "stat_sum", "centroid_x", "centroid_y", "intensity_in_roi", "intensity_cropped_out", "roi_energy_fraction"):
                    row[key] = _float_or_none(metadata.get(key))
                _insert_image(conn, row)
                counts[camera] += 1
        conn.commit()
        conn.close()
        os.replace(temp_path, db_path)
    finally:
        conn.close()
        if temp_path.exists():
            temp_path.unlink()
    return ExportResult(run_dir, db_path, run_id, sum(counts.values()), counts)


def _prepare_output(out_dir: Path, db_path: Path, *, overwrite: bool) -> None:
    if out_dir.exists() and not out_dir.is_dir():
        raise RuntimeError("out_dir exists and is not a directory: %s" % out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise RuntimeError("out_dir is not empty; set overwrite=true: %s" % out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        if not overwrite:
            raise RuntimeError("dataset DB exists; set overwrite=true: %s" % db_path)
        db_path.unlink()


def _init_db(
    conn: sqlite3.Connection,
    reader: RawRunReader,
    config: ExportConfig,
    run_id: str,
    alignment: SyncAlignment,
) -> None:
    conn.executescript(SCHEMA)
    run_info = {
        "run_id": run_id,
        "run_dir_name": reader.run_dir.name,
        "run_dir_path": str(reader.run_dir),
        "source_format": "clear2026_raw",
        "contract": "lossless per-camera PNG cache; sample_id = sync-aligned external-trigger group; frame_sample_id = frame_index + 1; no intensity transform unless image hooks are configured",
        "sync_alignment_source": alignment.source,
        "sync_alignment_tolerance_ns": str(alignment.tolerance_ns),
        "sync_alignment_group_count": str(alignment.group_count),
        "sync_alignment_complete_group_count": str(alignment.complete_group_count),
        "sync_alignment_incomplete_group_count": str(alignment.incomplete_group_count),
        "sync_alignment_dropped_group_count": str(alignment.dropped_group_count),
        "sync_alignment_dropped_frame_count": str(alignment.dropped_frame_count),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "resolved_config_json": json.dumps(reader.resolved_config, sort_keys=True, default=str),
        "export_config_json": json.dumps(_config_to_jsonable(config), sort_keys=True, default=str),
        "extra_info_json": json.dumps(config.extra_info, sort_keys=True, default=str),
    }
    conn.executemany(
        "INSERT OR REPLACE INTO run_info(key, value) VALUES(?, ?)",
        list(run_info.items()),
    )

    for camera, cfg in reader.camera_configs.items():
        conn.execute(
            """
            INSERT OR REPLACE INTO cameras(
                camera, raw_dtype, height, width, payload_nbytes,
                camera_pixel_format, settings_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                camera,
                str(cfg.dtype),
                int(cfg.shape[0]),
                int(cfg.shape[1]),
                int(cfg.payload_nbytes),
                _text_or_none(cfg.settings.get("cameraPixelFormat")),
                json.dumps(cfg.settings, sort_keys=True, default=str),
            ),
        )
    conn.commit()


def _write_frame(
    *,
    out_dir: Path,
    run_dir: Path,
    run_id: str,
    frame: RawFrame,
    sample_id: int,
    sync_time_ns: int,
    sync_group_span_ns: int,
    image_hooks: Sequence[Any],
    applied_crop: Optional[tuple],
    metadata_hooks: Sequence[Any],
    compress_level: int,
    created_utc: str,
) -> dict[str, Any]:
    header = frame.header
    frame_index = int(header.get("frame_index", frame.global_record_index))
    acq_stamp_ns = _int_or_none(header.get("acq_stamp_ns"))
    source_image_number = _int_or_none(header.get("source_image_number"))
    pulse_id = _pulse_or_none(header.get("pulse_id"))
    image_id = _image_id(frame.camera, acq_stamp_ns, frame_index)
    filename = _png_filename(acq_stamp_ns, frame_index)
    rel_path = "%s/%s" % (frame.camera, filename)

    raw_segment_path = _relative_text(frame.source_path, run_dir)
    context = HookContext(
        run_id=run_id,
        camera=frame.camera,
        image_id=image_id,
        frame_index=frame_index,
        acq_stamp_ns=acq_stamp_ns,
        source_image_number=source_image_number,
        pulse_id=pulse_id,
        sync=frame.sync,
        raw_header=header,
        raw_segment_path=raw_segment_path,
        byte_offset=int(frame.byte_offset),
        source_image=np.asarray(frame.image),
        applied_crop=applied_crop,
    )

    image = np.asarray(frame.image)
    for hook in image_hooks:
        image = np.asarray(hook(image, context))
    png_image = _to_png_integer_image(image, frame.camera)

    hook_metadata: dict[str, Any] = {}
    for hook in metadata_hooks:
        hook_metadata.update(dict(hook(png_image, context)))

    write_png_gray(out_dir / rel_path, png_image, compress_level=compress_level)

    sync = frame.sync or {}
    return {
        "image_id": image_id,
        "sample_id": int(sample_id),
        "frame_sample_id": frame_index + 1,
        "run_id": run_id,
        "camera": frame.camera,
        "png_path": rel_path,
        "png_dtype": str(png_image.dtype),
        "raw_dtype": str(np.asarray(frame.image).dtype),
        "height": int(png_image.shape[0]),
        "width": int(png_image.shape[1]),
        "frame_index": frame_index,
        "global_record_index": int(frame.global_record_index),
        "segment_index": int(frame.segment_index),
        "record_index_in_segment": int(frame.record_index_in_segment),
        "source_image_number": source_image_number,
        "acq_stamp_ns": acq_stamp_ns,
        "sync_time_ns": int(sync_time_ns),
        "sync_group_span_ns": int(sync_group_span_ns),
        "pulse_id": pulse_id,
        "chosen_sync_source": _text_or_none(sync.get("chosen_sync_source")),
        "chosen_sync_value": _json_or_none(sync.get("chosen_sync_value")),
        "raw_segment_path": raw_segment_path,
        "byte_offset": int(frame.byte_offset),
        "stat_min": _float_or_none(hook_metadata.get("stat_min")),
        "stat_max": _float_or_none(hook_metadata.get("stat_max")),
        "stat_mean": _float_or_none(hook_metadata.get("stat_mean")),
        "stat_sum": _float_or_none(hook_metadata.get("stat_sum")),
        "centroid_x": _float_or_none(hook_metadata.get("centroid_x")),
        "centroid_y": _float_or_none(hook_metadata.get("centroid_y")),
        "intensity_in_roi": _float_or_none(hook_metadata.get("intensity_in_roi")),
        "intensity_cropped_out": _float_or_none(hook_metadata.get("intensity_cropped_out")),
        "roi_energy_fraction": _float_or_none(hook_metadata.get("roi_energy_fraction")),
        "raw_header_json": json.dumps(header, sort_keys=True, default=str),
        "sync_json": json.dumps(frame.sync, sort_keys=True, default=str) if frame.sync else None,
        "hook_metadata_json": json.dumps(hook_metadata, sort_keys=True, default=str),
        "extra_json": None,
        "created_utc": created_utc,
    }


def _insert_image(conn: sqlite3.Connection, row: Mapping[str, Any]) -> None:
    columns = [
        "image_id",
        "sample_id",
        "frame_sample_id",
        "run_id",
        "camera",
        "png_path",
        "png_dtype",
        "raw_dtype",
        "height",
        "width",
        "frame_index",
        "global_record_index",
        "segment_index",
        "record_index_in_segment",
        "source_image_number",
        "acq_stamp_ns",
        "sync_time_ns",
        "sync_group_span_ns",
        "pulse_id",
        "chosen_sync_source",
        "chosen_sync_value",
        "raw_segment_path",
        "byte_offset",
        "stat_min",
        "stat_max",
        "stat_mean",
        "stat_sum",
        "centroid_x",
        "centroid_y",
        "intensity_in_roi",
        "intensity_cropped_out",
        "roi_energy_fraction",
        "raw_header_json",
        "sync_json",
        "hook_metadata_json",
        "extra_json",
        "created_utc",
    ]
    placeholders = ", ".join(["?"] * len(columns))
    conn.execute(
        "INSERT INTO images(%s) VALUES(%s)" % (", ".join(columns), placeholders),
        [row.get(column) for column in columns],
    )


def _to_png_integer_image(image: np.ndarray, camera: str) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise RuntimeError("camera %s image must be 2D, got %s" % (camera, arr.shape))
    if arr.dtype == np.dtype("uint8") or arr.dtype == np.dtype("uint16"):
        return arr
    if not np.issubdtype(arr.dtype, np.integer):
        raise RuntimeError("camera %s image is not integer dtype: %s" % (camera, arr.dtype))
    if arr.size:
        amin = int(arr.min())
        amax = int(arr.max())
        if amin < 0:
            raise RuntimeError(
                "camera %s has negative values (min=%d); cannot store as source-truth PNG"
                % (camera, amin)
            )
        if amax > 65535:
            raise RuntimeError(
                "camera %s max=%d exceeds uint16 PNG range" % (camera, amax)
            )
    return arr.astype(np.uint16, copy=False)


def _image_id(camera: str, acq_stamp_ns: Optional[int], frame_index: int) -> str:
    stamp = acq_stamp_ns if acq_stamp_ns is not None else 0
    return "%s_%d_%d" % (camera, stamp, frame_index)


def _png_filename(acq_stamp_ns: Optional[int], frame_index: int) -> str:
    stamp = acq_stamp_ns if acq_stamp_ns is not None else 0
    return "%020d_%08d.png" % (stamp, frame_index)


def _relative_text(path: Path, parent: Path) -> str:
    try:
        return str(path.relative_to(parent))
    except ValueError:
        return str(path)


def _pulse_or_none(value: Any) -> Optional[int]:
    value = _int_or_none(value)
    if value == MISSING_U64:
        return None
    return value


def _int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _text_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _json_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, default=str)


def _require(data: Mapping[str, Any], key: str) -> Any:
    value = data.get(key)
    if value in (None, ""):
        raise ValueError("config is missing required field %r" % key)
    return value


def _config_to_jsonable(config: ExportConfig) -> Mapping[str, Any]:
    return {
        "run_dir": str(config.run_dir),
        "out_dir": str(config.out_dir),
        "run_id": config.run_id,
        "cameras": config.cameras,
        "overwrite": config.overwrite,
        "include_part_files": config.include_part_files,
        "png": {"compress_level": config.png_compress_level},
        "show_progress": config.show_progress,
        "crop_enabled": config.crop_enabled,
        "sync_alignment": {
            "source": config.sync_alignment_source,
            "tolerance_ns": config.sync_alignment_tolerance_ns,
        },
        "hooks": {
            "image": config.image_hooks,
            "metadata": config.metadata_hooks,
        },
        "extra_info": config.extra_info,
    }


class _ProgressBar:
    def __init__(self, total: int, *, enabled: bool = True, width: int = 30) -> None:
        self.total = int(total)
        self.enabled = bool(enabled)
        self.width = int(width)
        self.next_update = 0.0

    def update(self, count: int, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now < self.next_update:
            return
        self.next_update = now + 0.2

        if self.total > 0:
            fraction = min(1.0, float(count) / float(self.total))
            filled = int(round(self.width * fraction))
            bar = "#" * filled + "-" * (self.width - filled)
            text = "\r[%s] %d/%d images (%5.1f%%)" % (
                bar,
                count,
                self.total,
                fraction * 100.0,
            )
        else:
            text = "\r%d images" % count
        sys.stderr.write(text)
        sys.stderr.flush()

    def done(self, count: int) -> None:
        if not self.enabled:
            return
        self.update(count, force=True)
        sys.stderr.write("\n")
        sys.stderr.flush()
