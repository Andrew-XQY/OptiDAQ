"""Read an OptiDAQ raw run as per-camera image frames."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence, Union

import numpy as np

from optidaq.daq.raw_stream import iter_stream_records, load_sidecar_spec


MISSING_U64 = (1 << 64) - 1


@dataclass(frozen=True)
class CameraConfig:
    camera: str
    dtype: np.dtype
    shape: tuple[int, int]
    payload_nbytes: int
    settings: Mapping[str, Any]


@dataclass(frozen=True)
class RawFrame:
    camera: str
    image: np.ndarray
    header: Mapping[str, int]
    sync: Optional[Mapping[str, Any]]
    source_path: Path
    segment_index: int
    record_index_in_segment: int
    global_record_index: int
    byte_offset: int


class RawRunReader:
    """Decode image arrays and metadata from one OptiDAQ run folder."""

    def __init__(
        self,
        run_dir: Union[str, Path],
        *,
        cameras: Optional[Sequence[str]] = None,
        include_part_files: bool = False,
    ) -> None:
        self.run_dir = Path(run_dir).expanduser()
        if not self.run_dir.is_dir():
            raise RuntimeError("run_dir is not a directory: %s" % self.run_dir)

        self.include_part_files = bool(include_part_files)
        self.resolved_config = _read_json(self.run_dir / "resolved_config.json")
        all_camera_configs = _camera_configs_from_resolved(self.resolved_config)
        selected = [str(camera) for camera in cameras] if cameras else list(all_camera_configs)
        if not selected:
            raise RuntimeError("no cameras configured in %s" % self.run_dir)

        missing = [camera for camera in selected if camera not in all_camera_configs]
        if missing:
            raise RuntimeError("camera(s) not found in resolved_config.json: %s" % ", ".join(missing))

        self.cameras = selected
        self.camera_configs = {
            camera: _build_camera_config(camera, all_camera_configs[camera])
            for camera in self.cameras
        }
        self.specs = {
            camera: load_sidecar_spec(self.run_dir, camera)
            for camera in self.cameras
        }
        self.sync_by_camera = {
            camera: _read_sync_sidecar(self.run_dir, camera)
            for camera in self.cameras
        }
        self._validate_payloads()

    def iter_camera_frames(self, camera: str) -> Iterator[RawFrame]:
        cfg = self.camera_configs[camera]
        spec = self.specs[camera]
        sync_rows = self.sync_by_camera.get(camera, {})
        for record in iter_stream_records(
            self.run_dir,
            spec,
            include_part_files=self.include_part_files,
        ):
            frame_index = int(record.header.get("frame_index", record.global_record_index))
            image = np.frombuffer(record.payload, dtype=cfg.dtype).reshape(cfg.shape)
            yield RawFrame(
                camera=camera,
                image=image,
                header=dict(record.header),
                sync=sync_rows.get(frame_index),
                source_path=record.source_path,
                segment_index=record.segment_index,
                record_index_in_segment=record.record_index_in_segment,
                global_record_index=record.global_record_index,
                byte_offset=record.byte_offset,
            )

    def iter_frames(self) -> Iterator[RawFrame]:
        for camera in self.cameras:
            yield from self.iter_camera_frames(camera)

    def raw_record_count(self, camera: str) -> int:
        spec = self.specs[camera]
        count = 0
        for path in self.run_dir.glob("%s_*%s" % (spec.file_prefix, spec.file_extension)):
            count += path.stat().st_size // spec.record_nbytes
        return count

    def _validate_payloads(self) -> None:
        for camera in self.cameras:
            cfg = self.camera_configs[camera]
            spec = self.specs[camera]
            expected = int(np.prod(cfg.shape)) * cfg.dtype.itemsize
            if expected != spec.payload_nbytes:
                raise RuntimeError(
                    "shape/dtype payload mismatch for %s: shape=%s dtype=%s expected=%d spec=%d"
                    % (camera, cfg.shape, cfg.dtype, expected, spec.payload_nbytes)
                )
            if cfg.payload_nbytes and cfg.payload_nbytes != spec.payload_nbytes:
                raise RuntimeError(
                    "payload_nbytes mismatch for %s: resolved=%d spec=%d"
                    % (camera, cfg.payload_nbytes, spec.payload_nbytes)
                )


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _camera_configs_from_resolved(resolved_config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    cameras = resolved_config.get("cameras")
    if not isinstance(cameras, Mapping):
        raise RuntimeError("resolved_config.json cameras must be an object")

    inner = resolved_config.get("config")
    camera_setting = {}
    if isinstance(inner, Mapping) and isinstance(inner.get("camera_setting"), Mapping):
        camera_setting = inner["camera_setting"]

    out = {}
    for camera, value in cameras.items():
        cfg = dict(value) if isinstance(value, Mapping) else {}
        if cfg.get("cameraPixelFormat") is None and cfg.get("bit_depth") is None:
            setting = camera_setting.get(camera)
            if isinstance(setting, Mapping) and setting.get("cameraPixelFormat") is not None:
                cfg["cameraPixelFormat"] = setting["cameraPixelFormat"]
        out[str(camera)] = cfg
    return out


def _build_camera_config(camera: str, cfg: Mapping[str, Any]) -> CameraConfig:
    if "dtype" not in cfg:
        raise RuntimeError("camera %s is missing dtype" % camera)
    if "shape" not in cfg:
        raise RuntimeError("camera %s is missing shape" % camera)
    shape = tuple(int(part) for part in cfg["shape"])
    if len(shape) != 2:
        raise RuntimeError("camera %s shape must be 2D, got %s" % (camera, shape))
    return CameraConfig(
        camera=camera,
        dtype=np.dtype(cfg["dtype"]),
        shape=(shape[0], shape[1]),
        payload_nbytes=int(cfg.get("payload_nbytes") or 0),
        settings=dict(cfg),
    )


def _read_sync_sidecar(run_dir: Path, camera: str) -> dict[int, Mapping[str, Any]]:
    path = run_dir / ("%s.sync.jsonl" % camera)
    if not path.exists():
        return {}

    rows = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "frame_index" not in row:
                raise RuntimeError("%s:%d missing frame_index" % (path, line_number))
            rows[int(row["frame_index"])] = row
    return rows
