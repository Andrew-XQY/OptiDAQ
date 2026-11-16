CONFIG = {
    "dataset_dir": "/path/to/dataset",
    "out_path": "/path/to/output_png",
    "core_dtype": None,
    "verify_crc": False,
    "require_complete_cameras": True,
    "use_aligned_index_if_present": True,
    "transforms": [
        # {"name": "to_dtype", "kwargs": {"dtype": None}},
        # {"name": "crop", "kwargs": None},
        {"name": "to_uint16", "kwargs": {}},
        {"name": "stats", "kwargs": {}},
        # {"name": "centroid", "kwargs": {}},
        #{"name": "minmax_stretch", "kwargs": {}},
    ],
    "input_format": "auto",  # input run format to read: raw | png | auto (detect)
    "process_png": False,  # opt in to image transforms again on already-processed PNG runs
    "backend": "png",  # output backend: png | hdf5
    "unaligned": False,  # True -> skip cross-camera matching: save every frame as its own PNG row (use when alignment/sync is unavailable; requires backend "png")
    "on_alignment_missing": "sync_groups",  # error | per_image | sync_groups
    # Force the cross-camera alignment key order, overriding each run's resolved_config.json
    # sync.priority. Lets old runs (whose frozen config may list cycleStampGlob first) be
    # re-parsed with a selected ordering. Set to None to use the run's
    # own sync.priority unchanged. Affects only the sync-group fallback (when no
    # aligned_index.jsonl exists); the matching window (max_span_ns_by_source) and every other
    # sync setting still come from the run config.
    "sync_priority": [
        "pulse_id",          # exact per-shot id: used when present, cleanly skipped when absent
        "image_ready_ns",    # Per-frame timestamp; match cameras within the configured tolerance.
        "imageTimeStampRaw",
        "imageTimeStamp",
        "cycleStampGlob",    # exact, but only valid at <=1 frame/cycle (e.g. 0.833 Hz runs)
    ],
    "png": {
        "commit_every": 256,
        # PNG zlib level 0-9; lower levels trade file size for faster encoding.
        "compress_level": 1,
    },
    "hdf5": {
        "chunk": 64,
        "compression": "gzip",
        "compression_opts": 4,
        "flush_every": 128,
    },
}

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import struct
import sys
import time
import warnings
import zlib

import numpy as np

try:
    import h5py
except ImportError:  # optional: only needed for the hdf5 backend
    h5py = None


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from optidaq.data.png_io import write_png as _write_png_file  # noqa: E402
from optidaq.daq.sync_tools import (  # noqa: E402
    EXACT_SYNC_SOURCES,
    LOCAL_CLOCK_SOURCES,
    evaluate_sync_source,
    sync_priority,
)


MISSING_U64 = (1 << 64) - 1
UNKNOWN_CRC = np.int8(-1)
TYPE_FORMATS = {
    "uint64": "Q",
    "int64": "q",
    "uint32": "I",
    "int32": "i",
}
ENDIANNESS_FORMATS = {
    "little": "<",
    "big": ">",
}


class ProgressBar:
    def __init__(self, total=None, width=30, stream=None):
        self.total = int(total) if total else None
        self.width = int(width)
        self.stream = stream or sys.stderr
        self.next_update = 0.0

    def update(self, count, force=False):
        now = time.monotonic()
        if not force and now < self.next_update:
            return
        self.next_update = now + 0.2

        count = int(count)
        if self.total:
            fraction = min(1.0, float(count) / float(self.total))
            filled = int(round(self.width * fraction))
            bar = "#" * filled + "-" * (self.width - filled)
            text = "\r[%s] %d/%d samples (%5.1f%%)" % (
                bar,
                count,
                self.total,
                fraction * 100.0,
            )
        else:
            pos = count % self.width
            bar = "." * pos + ">" + "." * (self.width - pos - 1)
            text = "\r[%s] %d samples" % (bar, count)

        self.stream.write(text)
        self.stream.flush()

    def done(self, count):
        self.update(count, force=True)
        self.stream.write("\n")
        self.stream.flush()


@dataclass
class Sample:
    images: dict[str, np.ndarray]
    meta: dict


@dataclass
class RawRecord:
    camera: str
    header: dict
    image: np.ndarray
    sync: object
    crc_ok: object
    segment_path: Path
    segment_index: int
    record_index_in_segment: int
    global_record_index: int
    byte_offset: int


class TransformRegistry:
    _items: dict[str, object] = {}

    @classmethod
    def register(cls, name):
        def decorator(factory):
            if name in cls._items:
                raise RuntimeError("duplicate transform %r" % name)
            cls._items[name] = factory
            return factory

        return decorator

    @classmethod
    def build(cls, name, kwargs):
        if name not in cls._items:
            raise RuntimeError("unknown transform %r" % name)
        factory = cls._items[name]
        return factory(**(kwargs or {}))


def register(name):
    return TransformRegistry.register(name)


def build_transforms(transform_configs):
    transforms = []
    for item in transform_configs:
        if not isinstance(item, dict):
            raise RuntimeError("transform entry must be an object")
        transforms.append(TransformRegistry.build(item["name"], item.get("kwargs")))
    return transforms


@register("to_dtype")
def make_to_dtype(dtype=None):
    target_dtype = None if dtype is None else np.dtype(dtype)

    def transform(sample):
        if target_dtype is None:
            return sample
        sample.images = {
            camera: image.astype(target_dtype, copy=False)
            for camera, image in sample.images.items()
        }
        return sample

    return transform


@register("to_uint16")
def make_to_uint16(bits_by_camera=None):
    """Cast integer images to the unsigned dtype PNG needs, guarding against silent corruption.

    PNG holds only unsigned 8/16-bit. The target is chosen per camera from its real bit depth
    (``bits_by_camera``, derived from the pixel format): <= 8-bit data (e.g. mono8) becomes uint8
    so it fills an 8-bit PNG instead of looking near-black in a 16-bit one; deeper data becomes
    uint16. When the bit depth is unknown the original behaviour is kept (uint8/uint16 pass
    through, other integers -> uint16). Signed or over-range values raise instead of wrapping
    (-1 -> 65535).
    """
    bits_map = dict(bits_by_camera or {})

    def transform(sample):
        images = {}
        for camera, image in sample.images.items():
            arr = np.asarray(image)
            bits = bits_map.get(camera)
            if bits is not None and int(bits) <= 8:
                target = np.dtype(np.uint8)
            elif arr.dtype in (np.dtype(np.uint8), np.dtype(np.uint16)):
                images[camera] = arr
                continue
            else:
                target = np.dtype(np.uint16)
            if arr.dtype == target:
                images[camera] = arr
                continue
            if not np.issubdtype(arr.dtype, np.integer):
                raise RuntimeError(
                    "to_uint16 expects integer images; got %s for %s (PNG cannot store floats)"
                    % (arr.dtype, camera)
                )
            ceiling = 255 if target == np.dtype(np.uint8) else 65535
            if arr.size:
                amin = int(arr.min())
                amax = int(arr.max())
                if amin < 0:
                    raise RuntimeError(
                        "to_uint16: %s has negative values (min=%d) — data is genuinely signed; "
                        "casting would wrap. Resolve upstream before PNG export." % (camera, amin)
                    )
                if amax > ceiling:
                    raise RuntimeError(
                        "to_uint16: %s max=%d exceeds the %s range (%d); check the camera bit depth"
                        % (camera, amax, target, ceiling)
                    )
            images[camera] = arr.astype(target, copy=False)
        sample.images = images
        return sample

    return transform


@register("crop")
def make_crop(offset_y=None, offset_x=None, height=None, width=None):
    if None in (offset_y, offset_x, height, width):
        return lambda sample: sample

    offset_y = int(offset_y)
    offset_x = int(offset_x)
    height = int(height)
    width = int(width)
    if offset_y < 0 or offset_x < 0 or height <= 0 or width <= 0:
        raise RuntimeError("crop requires non-negative offsets and positive size")

    def transform(sample):
        cropped = {}
        for camera, image in sample.images.items():
            if offset_y + height > image.shape[0] or offset_x + width > image.shape[1]:
                raise RuntimeError(
                    "crop is outside image for %s: image=%s crop=%s"
                    % (camera, image.shape, (offset_y, offset_x, height, width))
                )
            cropped[camera] = image[offset_y:offset_y + height, offset_x:offset_x + width]
        sample.images = cropped
        return sample

    return transform


@register("subtract_dark")
def make_subtract_dark(path=None, value=None):
    return _make_subtract_transform(path=path, value=value)


@register("subtract_background")
def make_subtract_background(path=None, value=None):
    return _make_subtract_transform(path=path, value=value)


@register("clip_zero")
def make_clip_zero():
    def transform(sample):
        sample.images = {
            camera: np.maximum(image, 0)
            for camera, image in sample.images.items()
        }
        return sample

    return transform


@register("minmax_stretch")
def make_minmax_stretch(target_max=None, target_by_camera=None):
    """Per-image contrast stretch: rescale each camera image's own [min, max] to [0, target_max].

    target_max defaults per camera to the bit-depth ceiling (mono8 -> 255, mono12 -> 4095),
    sourced the same way as saturation. Output dtype is uint8 when the ceiling <= 255, else uint16.

    NOTE: this overwrites absolute pixel intensities (every frame is scaled independently), so
    place it AFTER `stats` if you want stats to reflect true sensor values. A flat image
    (max == min) maps to all zeros.
    """
    global_target = None if target_max is None else float(target_max)
    per_camera_target = target_by_camera or {}

    def transform(sample):
        images = {}
        for camera, image in sample.images.items():
            target = global_target
            if target is None:
                target = per_camera_target.get(camera)
            if target is None:
                target = 255.0  # safe fallback if no ceiling is known
            target = float(target)
            out_dtype = np.uint8 if target <= 255 else np.uint16

            arr = np.asarray(image)
            if arr.size == 0:
                images[camera] = arr.astype(out_dtype, copy=False)
                continue
            values = arr.astype(np.float64, copy=False)
            lo = float(values.min())
            hi = float(values.max())
            if hi <= lo:
                scaled = np.zeros_like(values)
            else:
                scaled = (values - lo) * (target / (hi - lo))
            scaled = np.clip(np.round(scaled), 0.0, target)
            images[camera] = scaled.astype(out_dtype, copy=False)
        sample.images = images
        return sample

    return transform


@register("stats")
def make_stats(saturation_value=None, saturation_by_camera=None):
    global_saturation = None if saturation_value is None else float(saturation_value)
    per_camera_saturation = saturation_by_camera or {}

    def transform(sample):
        stats_by_camera = sample.meta.setdefault("stats", {})
        for camera, image in sample.images.items():
            # explicit scalar kwarg overrides; else per-camera value derived from pixel format
            saturation = global_saturation
            if saturation is None:
                saturation = per_camera_saturation.get(camera)
            saturation = None if saturation is None else float(saturation)

            values = np.asarray(image)
            if values.size == 0:
                stats_by_camera[camera] = {
                    "min": float("nan"),
                    "max": float("nan"),
                    "mean": float("nan"),
                    "sum": float("nan"),
                    "saturated_fraction": float("nan"),
                    "saturated": False,
                }
                continue

            saturated_fraction = float("nan")
            saturated = False
            if saturation is not None:
                saturated_mask = values >= saturation
                saturated_fraction = float(np.count_nonzero(saturated_mask)) / float(values.size)
                saturated = bool(np.any(saturated_mask))

            values64 = values.astype(np.float64, copy=False)
            stats_by_camera[camera] = {
                "min": float(np.min(values64)),
                "max": float(np.max(values64)),
                "mean": float(np.mean(values64)),
                "sum": float(np.sum(values64)),
                "saturated_fraction": saturated_fraction,
                "saturated": saturated,
            }
        return sample

    return transform


@register("centroid")
def make_centroid():
    """Store per-camera intensity-weighted centroid as zero-based pixel coordinates."""

    def transform(sample):
        centroid_by_camera = sample.meta.setdefault("centroid", {})
        for camera, image in sample.images.items():
            values = np.asarray(image, dtype=np.float64)
            if values.ndim != 2 or values.size == 0:
                centroid_by_camera[camera] = {"x": float("nan"), "y": float("nan")}
                continue

            total = float(np.sum(values))
            if not np.isfinite(total) or total == 0.0:
                centroid_by_camera[camera] = {"x": float("nan"), "y": float("nan")}
                continue

            height, width = values.shape
            x_profile = np.sum(values, axis=0)
            y_profile = np.sum(values, axis=1)
            x = float(np.dot(x_profile, np.arange(width, dtype=np.float64)) / total)
            y = float(np.dot(y_profile, np.arange(height, dtype=np.float64)) / total)
            centroid_by_camera[camera] = {"x": x, "y": y}
        return sample

    return transform


def _make_subtract_transform(path=None, value=None):
    if path is None and value is None:
        return lambda sample: sample
    if path is not None and value is not None:
        raise RuntimeError("subtraction transform accepts either path or value, not both")

    correction = _load_correction(path) if path is not None else value

    def transform(sample):
        images = {}
        for camera, image in sample.images.items():
            camera_correction = _correction_for_camera(correction, camera)
            if camera_correction is None:
                images[camera] = image
            else:
                images[camera] = image.astype(np.float32, copy=False) - camera_correction
        sample.images = images
        return sample

    return transform


def _load_correction(path):
    path = Path(path).expanduser()
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix in (".h5", ".hdf5"):
        if h5py is None:
            raise RuntimeError("reading .h5/.hdf5 corrections requires h5py (pip install h5py)")
        out = {}
        with h5py.File(path, "r") as handle:
            group = handle["images"] if "images" in handle and isinstance(handle["images"], h5py.Group) else handle
            for name, value in group.items():
                if isinstance(value, h5py.Dataset):
                    out[str(name)] = np.asarray(value[()])
        if not out:
            raise RuntimeError("no top-level correction datasets found in %s" % path)
        return out
    raise RuntimeError("unsupported correction file type: %s" % path)


def _correction_for_camera(correction, camera):
    if isinstance(correction, dict):
        return correction.get(camera)
    return correction


class RawRecordCursor:
    def __init__(self, records):
        self._records = iter(records)
        self._last_index = -1

    def advance_to(self, target_index):
        target_index = int(target_index)
        if target_index <= self._last_index:
            raise RuntimeError(
                "raw record target moved backwards or repeated: %d after %d"
                % (target_index, self._last_index)
            )

        for record in self._records:
            if record.global_record_index == target_index:
                self._last_index = target_index
                return record
            if record.global_record_index > target_index:
                raise RuntimeError(
                    "raw stream passed target record %d at %s record %d"
                    % (target_index, record.camera, record.global_record_index)
                )

        raise RuntimeError("raw stream ended before target record %d" % target_index)


class RawSource:
    def __init__(self, dataset_dir, config):
        self.dataset_dir = Path(dataset_dir).expanduser()
        self.config = config
        self.verify_crc = bool(config.get("verify_crc", False))
        self.require_complete_cameras = bool(config.get("require_complete_cameras", True))
        self.use_aligned_index = bool(config.get("use_aligned_index_if_present", True))
        self.force_per_image = bool(config.get("unaligned", False))
        self.on_alignment_missing = str(config.get("on_alignment_missing", "error"))
        if self.on_alignment_missing not in ("error", "per_image", "sync_groups"):
            raise RuntimeError(
                "on_alignment_missing must be error/per_image/sync_groups, got %r"
                % self.on_alignment_missing
            )

        self.resolved_config = _read_json(self.dataset_dir / "resolved_config.json")
        self.sync_config = _resolve_sync_config(self.resolved_config, self.config)
        self.camera_configs = _camera_configs_from_resolved(self.resolved_config)
        self.cameras = list(self.camera_configs)
        if not self.cameras:
            raise RuntimeError("resolved_config.json has no cameras")

        self.specs = _load_specs(self.dataset_dir, self.cameras)
        self.sync_by_camera = {
            camera: self._read_sync_sidecar(camera)
            for camera in self.cameras
        }
        self._validate_camera_payloads()

    def __iter__(self):
        if self.force_per_image:
            yield from self._iter_per_image()
            return
        if len(self.cameras) == 1:
            yield from self._iter_single_camera()
            return

        aligned_path = self.dataset_dir / "aligned_index.jsonl"
        if self.use_aligned_index and aligned_path.exists() and _aligned_index_has_entries(aligned_path):
            yield from self._iter_aligned_index(aligned_path)
            return

        # Multi-camera, but no usable alignment. An empty index is treated as missing
        # (a present-but-empty file is the failure signal, not "an empty index to obey").
        if aligned_path.exists():
            reason = "aligned_index.jsonl is present but empty (alignment appears to have failed)"
        else:
            reason = "aligned_index.jsonl is absent"

        if self.on_alignment_missing == "sync_groups":
            warnings.warn("%s; falling back to sync-key grouping from the sidecars" % reason)
            yield from self._iter_chosen_sync_groups()
        elif self.on_alignment_missing == "per_image":
            warnings.warn("%s; storing every frame as its own row (align later via the .db)" % reason)
            yield from self._iter_per_image()
        else:  # "error" (default): fail loud instead of silently writing nothing
            raise RuntimeError(
                "%s. Re-run alignment, or pass --unaligned to store every frame and align "
                "later via the .db, or --on-missing sync_groups to align by sync key from the "
                "sidecars." % reason
            )

    def _iter_per_image(self):
        # Lossless: every camera frame becomes its own sample/row, carrying all of its
        # header stamps and sync keys. Cross-camera matching is deferred to a query on
        # the stored keys (e.g. pulse_id) rather than gated at parse time.
        sample_index = 0
        for camera in self.cameras:
            for record in self._iter_raw_records(camera):
                source, value = _chosen_sync_from_record(record)
                yield _sample_from_records(
                    {camera: record},
                    sample_index=sample_index,
                    chosen_sync_value=value,
                    chosen_sync_source=source,
                )
                sample_index += 1

    def has_raw_data(self):
        """True if any camera has a non-empty raw segment file (used as a safety net:
        producing zero samples while raw data exists on disk is an error, not success)."""
        for camera in self.cameras:
            try:
                paths = self._segment_paths(camera)
            except RuntimeError:
                continue
            for _index, path in paths:
                try:
                    if path.stat().st_size > 0:
                        return True
                except OSError:
                    continue
        return False

    def estimated_sample_count(self):
        if self.force_per_image:
            return sum(self._raw_record_count(camera) for camera in self.cameras)
        if len(self.cameras) == 1:
            return self._raw_record_count(self.cameras[0])

        aligned_path = self.dataset_dir / "aligned_index.jsonl"
        if self.use_aligned_index and aligned_path.exists() and _aligned_index_has_entries(aligned_path):
            return _count_nonempty_lines(aligned_path)
        if self.on_alignment_missing == "per_image":
            return sum(self._raw_record_count(camera) for camera in self.cameras)
        return None

    def _raw_record_count(self, camera):
        spec = self.specs[camera]
        payload_nbytes = int(spec["payload_nbytes"])
        header_nbytes = _header_struct(spec).size
        record_nbytes = header_nbytes + payload_nbytes
        return sum(
            path.stat().st_size // record_nbytes
            for _segment_index, path in self._segment_paths(camera)
        )

    def _iter_single_camera(self):
        camera = self.cameras[0]
        for sample_index, record in enumerate(self._iter_raw_records(camera)):
            source, value = _chosen_sync_from_record(record)
            yield _sample_from_records(
                {camera: record},
                sample_index=sample_index,
                chosen_sync_value=value,
                chosen_sync_source=source,
            )

    def _iter_aligned_index(self, aligned_path):
        cursors = {
            camera: RawRecordCursor(self._iter_raw_records(camera))
            for camera in self.cameras
        }
        sample_index = 0
        with open(aligned_path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if not entry.get("valid", True):
                    message = "invalid aligned_index entry at line %d: %s" % (
                        line_number,
                        entry.get("errors", []),
                    )
                    if self.require_complete_cameras:
                        raise RuntimeError(message)
                    warnings.warn(message)

                refs = entry.get("records", {})
                records = {}
                for camera in self.cameras:
                    ref = refs.get(camera)
                    if ref is None:
                        if self.require_complete_cameras:
                            raise RuntimeError(
                                "aligned_index line %d missing camera %s"
                                % (line_number, camera)
                            )
                        continue
                    target = ref.get("global_record_index", ref.get("frame_index"))
                    if target is None:
                        raise RuntimeError(
                            "aligned_index line %d camera %s has no record index"
                            % (line_number, camera)
                        )
                    records[camera] = cursors[camera].advance_to(target)

                self._require_or_warn_complete(records, "aligned_index line %d" % line_number)
                if not records:
                    continue

                yield _sample_from_records(
                    records,
                    sample_index=sample_index,
                    chosen_sync_value=entry.get("chosen_sync_value", entry.get("sync_key")),
                    chosen_sync_source=entry.get("chosen_sync_source", entry.get("sync_key_type")),
                )
                sample_index += 1

    def _iter_chosen_sync_groups(self):
        source, groups, reasons, skipped = _select_sync_groups(
            self.sync_by_camera,
            self.cameras,
            self.sync_config,
            require_complete=self.require_complete_cameras,
        )
        if source is None:
            raise RuntimeError(
                "no usable sync source from configured priority; tried: %s"
                % "; ".join(reasons)
            )
        if reasons:
            warnings.warn(
                "sync fallback selected %s after: %s"
                % (source, "; ".join(reasons))
            )
        if skipped:
            warnings.warn(
                "%s fallback skipped %d incomplete/duplicate sync group(s)"
                % (source, skipped)
            )

        cursors = {
            camera: RawRecordCursor(self._iter_raw_records(camera))
            for camera in self.cameras
        }
        sample_index = 0
        for group in groups:
            records = {}
            for camera, frame_index in group["frames"].items():
                records[camera] = cursors[camera].advance_to(frame_index)

            self._require_or_warn_complete(records, "%s %r" % (source, group["value"]))
            if not records:
                continue

            yield _sample_from_records(
                records,
                sample_index=sample_index,
                chosen_sync_value=group["value"],
                chosen_sync_source=source,
            )
            sample_index += 1

    def _iter_raw_records(self, camera):
        camera_cfg = self.camera_configs[camera]
        spec = self.specs[camera]
        dtype = np.dtype(camera_cfg["dtype"])
        shape = tuple(int(part) for part in camera_cfg["shape"])
        payload_nbytes = int(spec["payload_nbytes"])
        header_struct = _header_struct(spec)
        field_names = [field["name"] for field in spec["header_fields"]]
        record_nbytes = header_struct.size + payload_nbytes
        global_record_index = 0

        for segment_index, path in self._segment_paths(camera):
            record_index_in_segment = 0
            byte_offset = 0
            with open(path, "rb") as handle:
                while True:
                    header_bytes = handle.read(header_struct.size)
                    if not header_bytes:
                        break
                    if len(header_bytes) < header_struct.size:
                        warnings.warn(
                            "ignoring incomplete trailing header in %s at byte %d"
                            % (path, byte_offset)
                        )
                        break

                    payload = handle.read(payload_nbytes)
                    if len(payload) < payload_nbytes:
                        warnings.warn(
                            "ignoring incomplete trailing payload in %s at byte %d"
                            % (path, byte_offset)
                        )
                        break

                    values = header_struct.unpack(header_bytes)
                    header = {
                        name: int(value)
                        for name, value in zip(field_names, values)
                    }
                    frame_index = int(header.get("frame_index", global_record_index))
                    sync = self.sync_by_camera[camera].get(frame_index)
                    crc_ok = self._crc_ok(payload, sync)
                    image = np.frombuffer(payload, dtype=dtype).reshape(shape)

                    yield RawRecord(
                        camera=camera,
                        header=header,
                        image=image,
                        sync=sync,
                        crc_ok=crc_ok,
                        segment_path=path,
                        segment_index=segment_index,
                        record_index_in_segment=record_index_in_segment,
                        global_record_index=global_record_index,
                        byte_offset=byte_offset,
                    )

                    byte_offset += record_nbytes
                    record_index_in_segment += 1
                    global_record_index += 1

    def _read_sync_sidecar(self, camera):
        path = self.dataset_dir / ("%s.sync.jsonl" % camera)
        if not path.exists():
            raise RuntimeError("sync sidecar missing: %s" % path)

        rows = {}
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "frame_index" not in row:
                    raise RuntimeError("%s:%d missing frame_index" % (path, line_number))
                frame_index = int(row["frame_index"])
                if frame_index in rows:
                    warnings.warn("duplicate sync sidecar frame_index %d in %s" % (frame_index, path))
                rows[frame_index] = row
        return rows

    def _segment_paths(self, camera):
        spec = self.specs[camera]
        prefix = str(spec["file_prefix"])
        extension = str(spec["file_extension"])
        pattern = re.compile(r"^%s_(\d+)%s$" % (re.escape(prefix), re.escape(extension)))
        matches = []
        for path in self.dataset_dir.glob("%s_*%s" % (prefix, extension)):
            match = pattern.fullmatch(path.name)
            if match is None:
                raise RuntimeError("raw segment has non-numeric segment id: %s" % path)
            matches.append((int(match.group(1)), path))
        if not matches:
            raise RuntimeError("no raw segment files found for %s" % camera)
        matches.sort(key=lambda item: item[0])
        return matches

    def _crc_ok(self, payload, sync):
        if not self.verify_crc:
            return None
        if sync is None or sync.get("crc32") is None:
            return False
        expected = int(sync["crc32"]) & 0xFFFFFFFF
        actual = zlib.crc32(payload) & 0xFFFFFFFF
        return actual == expected

    def _validate_camera_payloads(self):
        for camera in self.cameras:
            camera_cfg = self.camera_configs[camera]
            spec = self.specs[camera]
            shape = tuple(int(part) for part in camera_cfg["shape"])
            if len(shape) != 2:
                raise RuntimeError("camera %s shape must be 2D for HDF5 images: %s" % (camera, shape))
            dtype = np.dtype(camera_cfg["dtype"])
            config_payload = int(camera_cfg["payload_nbytes"])
            spec_payload = int(spec["payload_nbytes"])
            expected_payload = int(np.prod(shape)) * dtype.itemsize
            if config_payload != spec_payload:
                raise RuntimeError(
                    "payload_nbytes mismatch for %s: resolved=%d spec=%d"
                    % (camera, config_payload, spec_payload)
                )
            if expected_payload != spec_payload:
                raise RuntimeError(
                    "shape/dtype payload mismatch for %s: shape=%s dtype=%s bytes=%d spec=%d"
                    % (camera, shape, dtype, expected_payload, spec_payload)
                )

    def _require_or_warn_complete(self, records, context):
        missing = [camera for camera in self.cameras if camera not in records]
        if not missing:
            return
        message = "%s missing camera(s): %s" % (context, ", ".join(missing))
        if self.require_complete_cameras:
            raise RuntimeError(message)
        warnings.warn(message)


class PngSource(RawSource):
    """Read a PNG-backend run through the exact same interface as ``RawSource``.

    The PNG backend writes one PNG per frame plus the identical
    ``{device}.sync.jsonl`` sidecar, so every bit of alignment / sync-grouping
    logic in ``RawSource`` is reused unchanged. Only the per-record *pixel*
    source differs: images come from self-describing PNG files (no
    ``{device}.spec.json`` needed) instead of raw binary segments. Record
    ``global_record_index`` is the frame index encoded in the PNG filename, which
    matches the sidecar/aligned-index keys the raw path uses.
    """

    def __init__(self, dataset_dir, config):
        self.dataset_dir = Path(dataset_dir).expanduser()
        self.config = config
        self.verify_crc = bool(config.get("verify_crc", False))
        self.require_complete_cameras = bool(config.get("require_complete_cameras", True))
        self.use_aligned_index = bool(config.get("use_aligned_index_if_present", True))
        self.force_per_image = bool(config.get("unaligned", False))
        self.on_alignment_missing = str(config.get("on_alignment_missing", "error"))
        if self.on_alignment_missing not in ("error", "per_image", "sync_groups"):
            raise RuntimeError(
                "on_alignment_missing must be error/per_image/sync_groups, got %r"
                % self.on_alignment_missing
            )

        self.resolved_config = _read_json(self.dataset_dir / "resolved_config.json")
        self.sync_config = _resolve_sync_config(self.resolved_config, self.config)
        self.camera_configs = _camera_configs_from_resolved(self.resolved_config)
        self.cameras = list(self.camera_configs)
        if not self.cameras:
            raise RuntimeError("resolved_config.json has no cameras")
        self.png_manifests = _load_png_manifests(self.dataset_dir, self.cameras)
        self.camera_configs = _png_camera_configs(self.camera_configs, self.png_manifests)

        self.specs = None  # PNG is self-describing; no .spec.json is required
        self.sync_by_camera = {
            camera: self._read_sync_sidecar(camera)
            for camera in self.cameras
        }
        self._validate_camera_payloads()

    def _png_paths(self, camera):
        """Return ``[(frame_index, path), ...]`` sorted by frame index.

        Filenames are ``{acq_stamp_ns:020d}_{frame_index:08d}.png`` (see
        ``PngImageSaveBackend``); the frame index is the alignment key.
        """
        cam_dir = self.dataset_dir / camera
        if not cam_dir.is_dir():
            raise RuntimeError("no png directory for camera %s: %s" % (camera, cam_dir))
        pattern = re.compile(r"^(\d+)_(\d+)\.png$")
        matches = []
        for path in cam_dir.glob("*.png"):
            match = pattern.fullmatch(path.name)
            if match is None:
                continue
            matches.append((int(match.group(2)), path))
        matches.sort(key=lambda item: item[0])
        return matches

    def _validate_camera_payloads(self):
        # PNG carries its own shape/dtype, so the raw payload-byte checks do not
        # apply. Enforce only the 2D image contract the HDF5 layout needs, which
        # is exactly what RawSource also requires.
        for camera in self.cameras:
            shape = tuple(int(part) for part in self.camera_configs[camera]["shape"])
            if len(shape) != 2:
                raise RuntimeError(
                    "camera %s shape must be 2D for HDF5 images: %s" % (camera, shape)
                )

    def _iter_raw_records(self, camera):
        camera_cfg = self.camera_configs[camera]
        dtype = np.dtype(camera_cfg["dtype"])
        shape = tuple(int(part) for part in camera_cfg["shape"])
        sidecar = self.sync_by_camera[camera]
        for frame_index, path in self._png_paths(camera):
            image = np.asarray(_read_png_file(path))
            if image.shape != shape:
                raise RuntimeError(
                    "PNG shape differs from recorded shape for %s: %s != %s"
                    % (path, image.shape, shape)
                )
            if camera in self.png_manifests:
                # Older Pillow versions decode 16-bit grayscale PNGs as int32.
                if dtype == np.dtype("uint16") and image.dtype == np.dtype("int32") and image.min() >= 0 and image.max() <= 65535:
                    image = image.astype(dtype)
                if image.dtype != dtype:
                    raise RuntimeError("PNG dtype differs from stored descriptor: %s" % path)
            elif image.dtype != dtype:
                # Legacy runs have no stored descriptor; keep their established
                # conversion to the resolved source dtype.
                image = image.astype(dtype, copy=False)
            sync = sidecar.get(frame_index)
            yield RawRecord(
                camera=camera,
                header=_header_from_sidecar(frame_index, sync),
                image=image,
                sync=sync,
                crc_ok=None,  # CRC is over raw payloads; re-encoded PNGs have none
                segment_path=path,
                segment_index=0,
                record_index_in_segment=frame_index,
                global_record_index=frame_index,
                byte_offset=0,
            )

    def _raw_record_count(self, camera):
        return len(self._png_paths(camera))

    def has_raw_data(self):
        for camera in self.cameras:
            try:
                if self._png_paths(camera):
                    return True
            except RuntimeError:
                continue
        return False


class Hdf5Sink:
    def __init__(self, out_path, config):
        if h5py is None:
            raise RuntimeError("the hdf5 backend requires h5py (pip install h5py)")
        self.out_path = Path(out_path).expanduser()
        self.config = config
        self.dataset_dir = Path(config["dataset_dir"]).expanduser()
        self.resolved_config = _read_json(self.dataset_dir / "resolved_config.json")
        self.camera_configs = _camera_configs_from_resolved(self.resolved_config)
        self.camera_configs = _png_camera_configs(
            self.camera_configs, _load_png_manifests(self.dataset_dir, self.camera_configs)
        )
        self.specs = _load_specs(self.dataset_dir, list(self.camera_configs), required=False)

        self.h5 = h5py.File(self.out_path, "w")
        self.h5.attrs["resolved_config_json"] = json.dumps(
            self.resolved_config,
            sort_keys=True,
        )
        self.h5.attrs["specs_json"] = json.dumps(self.specs, sort_keys=True)
        self.h5.attrs["processing_config_json"] = json.dumps(config, sort_keys=True)
        self.h5.attrs["script_name"] = Path(__file__).name
        self.h5.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()

        self.chunk = int(config.get("hdf5", {}).get("chunk", 64))
        self.flush_every = int(config.get("hdf5", {}).get("flush_every", 128))
        self.compression = config.get("hdf5", {}).get("compression")
        self.compression_opts = config.get("hdf5", {}).get("compression_opts")
        if self.compression in ("", "none", "None"):
            self.compression = None

        self.count = 0
        self.cameras = []
        self.datasets = {}
        self.closed = False

    def write(self, sample):
        if self.closed:
            raise RuntimeError("Hdf5Sink is closed")
        if not self.datasets:
            self._initialize(sample)
        if list(sample.images) != self.cameras:
            raise RuntimeError(
                "sample cameras changed: expected %s got %s"
                % (self.cameras, list(sample.images))
            )

        index = self.count
        self._resize_all(index + 1)

        self.datasets["sample_index"][index] = int(sample.meta.get("sample_index", index))
        self.datasets["chosen_sync_value"][index] = _json_text(sample.meta.get("chosen_sync_value"))
        self.datasets["chosen_sync_source"][index] = _text_or_empty(sample.meta.get("chosen_sync_source"))

        camera_meta = sample.meta.get("cameras", {})
        stats_meta = sample.meta.get("stats", {})
        centroid_meta = sample.meta.get("centroid", {})
        for camera in self.cameras:
            self.datasets["images"][camera][index, :, :] = sample.images[camera]
            meta = camera_meta.get(camera, {})
            self.datasets["frame_index"][camera][index] = _u64_or_missing(meta.get("frame_index"))
            self.datasets["source_image_number"][camera][index] = _u64_or_missing(
                meta.get("source_image_number")
            )
            self.datasets["acq_stamp_ns"][camera][index] = _u64_or_missing(meta.get("acq_stamp_ns"))
            self.datasets["pulse_id"][camera][index] = _u64_or_missing(meta.get("pulse_id"))
            self.datasets["crc_ok"][camera][index] = _crc_value(meta.get("crc_ok"))
            self.datasets["raw_segment_path"][camera][index] = _text_or_empty(
                meta.get("raw_segment_path")
            )
            self.datasets["byte_offset"][camera][index] = _u64_or_missing(meta.get("byte_offset"))

            stats = stats_meta.get(camera, {})
            self.datasets["stats_min"][camera][index] = float(stats.get("min", np.nan))
            self.datasets["stats_max"][camera][index] = float(stats.get("max", np.nan))
            self.datasets["stats_mean"][camera][index] = float(stats.get("mean", np.nan))
            self.datasets["stats_sum"][camera][index] = float(stats.get("sum", np.nan))
            self.datasets["stats_saturated_fraction"][camera][index] = float(
                stats.get("saturated_fraction", np.nan)
            )
            self.datasets["stats_saturated"][camera][index] = bool(stats.get("saturated", False))
            centroid = centroid_meta.get(camera, {})
            self.datasets["centroid_x"][camera][index] = float(centroid.get("x", np.nan))
            self.datasets["centroid_y"][camera][index] = float(centroid.get("y", np.nan))

        self.count += 1
        if self.flush_every > 0 and self.count % self.flush_every == 0:
            self.h5.flush()

    def close(self):
        if self.closed:
            return
        self.h5.flush()
        self.h5.close()
        self.closed = True

    def _initialize(self, sample):
        self.cameras = list(sample.images)
        self.h5.create_group("images")
        meta_group = self.h5.create_group("meta")
        stats_group = self.h5.create_group("stats")
        centroid_group = self.h5.create_group("centroid")
        string_dtype = h5py.string_dtype(encoding="utf-8")

        self.datasets = {
            "images": {},
            "sample_index": meta_group.create_dataset(
                "sample_index",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int64,
                chunks=(self.chunk,),
            ),
            "chosen_sync_value": meta_group.create_dataset(
                "chosen_sync_value",
                shape=(0,),
                maxshape=(None,),
                dtype=string_dtype,
                chunks=(self.chunk,),
            ),
            "chosen_sync_source": meta_group.create_dataset(
                "chosen_sync_source",
                shape=(0,),
                maxshape=(None,),
                dtype=string_dtype,
                chunks=(self.chunk,),
            ),
            "frame_index": {},
            "source_image_number": {},
            "acq_stamp_ns": {},
            "pulse_id": {},
            "crc_ok": {},
            "raw_segment_path": {},
            "byte_offset": {},
            "stats_min": {},
            "stats_max": {},
            "stats_mean": {},
            "stats_sum": {},
            "stats_saturated_fraction": {},
            "stats_saturated": {},
            "centroid_x": {},
            "centroid_y": {},
        }

        for camera, image in sample.images.items():
            if image.ndim != 2:
                raise RuntimeError("HDF5 sink supports 2D images, got %s for %s" % (image.shape, camera))
            height, width = image.shape
            image_chunks = (self.chunk, height, width)
            image_kwargs = {
                "shape": (0, height, width),
                "maxshape": (None, height, width),
                "dtype": image.dtype,
                "chunks": image_chunks,
            }
            if self.compression is not None:
                image_kwargs["compression"] = self.compression
                image_kwargs["compression_opts"] = self.compression_opts
            self.datasets["images"][camera] = self.h5["images"].create_dataset(
                camera,
                **image_kwargs,
            )

            camera_meta = meta_group.create_group(camera)
            self.datasets["frame_index"][camera] = _create_meta_dataset(camera_meta, "frame_index", np.uint64, self.chunk)
            self.datasets["source_image_number"][camera] = _create_meta_dataset(camera_meta, "source_image_number", np.uint64, self.chunk)
            self.datasets["acq_stamp_ns"][camera] = _create_meta_dataset(camera_meta, "acq_stamp_ns", np.uint64, self.chunk)
            self.datasets["pulse_id"][camera] = _create_meta_dataset(camera_meta, "pulse_id", np.uint64, self.chunk)
            self.datasets["crc_ok"][camera] = _create_meta_dataset(camera_meta, "crc_ok", np.int8, self.chunk)
            self.datasets["raw_segment_path"][camera] = camera_meta.create_dataset(
                "raw_segment_path",
                shape=(0,),
                maxshape=(None,),
                dtype=string_dtype,
                chunks=(self.chunk,),
            )
            self.datasets["byte_offset"][camera] = _create_meta_dataset(camera_meta, "byte_offset", np.uint64, self.chunk)

            camera_stats = stats_group.create_group(camera)
            self.datasets["stats_min"][camera] = _create_meta_dataset(camera_stats, "min", np.float64, self.chunk)
            self.datasets["stats_max"][camera] = _create_meta_dataset(camera_stats, "max", np.float64, self.chunk)
            self.datasets["stats_mean"][camera] = _create_meta_dataset(camera_stats, "mean", np.float64, self.chunk)
            self.datasets["stats_sum"][camera] = _create_meta_dataset(camera_stats, "sum", np.float64, self.chunk)
            self.datasets["stats_saturated_fraction"][camera] = _create_meta_dataset(
                camera_stats,
                "saturated_fraction",
                np.float64,
                self.chunk,
            )
            self.datasets["stats_saturated"][camera] = _create_meta_dataset(camera_stats, "saturated", np.bool_, self.chunk)

            camera_centroid = centroid_group.create_group(camera)
            self.datasets["centroid_x"][camera] = _create_meta_dataset(
                camera_centroid,
                "x",
                np.float64,
                self.chunk,
            )
            self.datasets["centroid_y"][camera] = _create_meta_dataset(
                camera_centroid,
                "y",
                np.float64,
                self.chunk,
            )

    def _resize_all(self, size):
        for value in self.datasets.values():
            if isinstance(value, dict):
                for dataset in value.values():
                    dataset.resize((size,) + dataset.shape[1:])
            else:
                value.resize((size,))


def _create_meta_dataset(group, name, dtype, chunk):
    return group.create_dataset(
        name,
        shape=(0,),
        maxshape=(None,),
        dtype=dtype,
        chunks=(chunk,),
    )


PNG_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_info (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS cameras (
    camera           TEXT PRIMARY KEY,
    dtype            TEXT,
    height           INTEGER,
    width            INTEGER,
    payload_nbytes   INTEGER,
    saturation_value REAL,
    settings_json    TEXT
);
CREATE TABLE IF NOT EXISTS images (
    image_id            TEXT PRIMARY KEY,
    camera              TEXT REFERENCES cameras(camera),
    sample_id           INTEGER,
    png_path            TEXT,
    png_dtype           TEXT,
    height              INTEGER,
    width               INTEGER,
    frame_index         INTEGER,
    source_image_number INTEGER,
    acq_stamp_ns        INTEGER,
    pulse_id            INTEGER,
    crc_ok              INTEGER,
    raw_segment_path    TEXT,
    byte_offset         INTEGER,
    chosen_sync_value   TEXT,
    chosen_sync_source  TEXT,
    stat_min            REAL,
    stat_max            REAL,
    stat_mean           REAL,
    stat_sum            REAL,
    saturated           INTEGER,
    saturated_fraction  REAL,
    centroid_x          REAL,
    centroid_y          REAL,
    created_utc         TEXT
);
CREATE INDEX IF NOT EXISTS ix_images_sample    ON images(sample_id);
CREATE INDEX IF NOT EXISTS ix_images_camera    ON images(camera);
CREATE INDEX IF NOT EXISTS ix_images_saturated ON images(saturated);
"""

_PNG_IMAGE_COLUMNS = (
    "image_id", "camera", "sample_id", "png_path", "png_dtype", "height", "width",
    "frame_index", "source_image_number", "acq_stamp_ns", "pulse_id", "crc_ok",
    "raw_segment_path", "byte_offset", "chosen_sync_value", "chosen_sync_source",
    "stat_min", "stat_max", "stat_mean", "stat_sum", "saturated", "saturated_fraction",
    "centroid_x", "centroid_y", "created_utc",
)


def _ensure_png_image_columns(conn):
    existing = {
        row[1]
        for row in conn.execute("PRAGMA table_info(images)")
    }
    for name, declaration in (
        ("centroid_x", "REAL"),
        ("centroid_y", "REAL"),
    ):
        if name not in existing:
            conn.execute("ALTER TABLE images ADD COLUMN %s %s" % (name, declaration))


def _write_png(path, image, compress_level=1):
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise RuntimeError("png backend supports 2D grayscale images; got shape %s" % (arr.shape,))
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8)
    if arr.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
        raise RuntimeError(
            "png backend supports only uint8/uint16 images; got %s. "
            "Float or signed data (e.g. after subtract_background) cannot be stored as PNG; "
            "use the hdf5 backend instead." % arr.dtype
        )
    _write_png_file(path, np.ascontiguousarray(arr), compress_level=compress_level)


# CERN pyda cameraPixelFormat enum -> bit depth (see src/optidaq/extensions/cern_pyda.py)
_PIXEL_FORMAT_BITS = {
    44: 8,    # mono8
    40: 12,   # mono12
}


def _pixel_format_bits(value):
    """Map a pixel-format hint (enum int or 'mono8'/'mono12'/...) to a bit depth."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return _PIXEL_FORMAT_BITS.get(int(value))
    text = str(value).strip().lower().replace("-", "").replace("_", "")
    match = re.fullmatch(r"mono(\d+)", text)
    return int(match.group(1)) if match else None


def _camera_bits(camera_cfg):
    """Real per-image bit depth: explicit bit_depth, else the pixel format (mono8 -> 8,
    mono12 -> 12). None when neither is recorded (e.g. only the int16 wire dtype is known)."""
    if camera_cfg.get("bit_depth") is not None:
        return int(camera_cfg["bit_depth"])
    for key in ("cameraPixelFormat", "pixel_format", "pixelFormat"):
        if camera_cfg.get(key) is not None:
            bits = _pixel_format_bits(camera_cfg[key])
            if bits:
                return bits
    return None


def _camera_saturation(camera_cfg):
    """Per-camera saturation ceiling from acquisition truth, not a hardcoded constant.

    Precedence: explicit max -> pixel format / bit depth -> dtype container max (with a
    warning for ambiguous uint16, since mono12 lives in a uint16 but saturates at 4095).
    """
    for key in ("saturation_value", "max_value"):
        if camera_cfg.get(key) is not None:
            return float(camera_cfg[key])

    bits = _camera_bits(camera_cfg)
    if bits:
        return float((1 << bits) - 1)

    dtype = np.dtype(camera_cfg.get("dtype", "uint16"))
    if dtype == np.dtype(np.uint8):
        return 255.0
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        warnings.warn(
            "camera config has no pixel format/bit depth; saturation falls back to %s max (%d). "
            "Add cameraPixelFormat or bit_depth for packed formats like mono12 (ceiling 4095)."
            % (dtype, info.max)
        )
        return float(info.max)
    return None


class PngDbSink:
    """One PNG per single-camera image on disk, plus a SQLite manifest with one row per image.

    Layout (paths in the DB are relative to the DB file, so the folder is portable):
        out_dir/dataset.db
        out_dir/<camera>/<image_id>.png
    image_id is deterministic: <camera>_<acq_stamp_ns>_<frame_index>.
    """

    def __init__(self, out_path, config):
        self.out_dir = Path(out_path).expanduser()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.dataset_dir = Path(config["dataset_dir"]).expanduser()
        self.resolved_config = _read_json(self.dataset_dir / "resolved_config.json")
        self.camera_configs = _camera_configs_from_resolved(self.resolved_config)
        self.camera_configs = _png_camera_configs(
            self.camera_configs, _load_png_manifests(self.dataset_dir, self.camera_configs)
        )
        self.specs = _load_specs(self.dataset_dir, list(self.camera_configs), required=False)
        self.run_id = str(self.resolved_config.get("run_id") or self.dataset_dir.name)

        self.commit_every = int(config.get("png", {}).get("commit_every", 256))
        self.compress_level = int(config.get("png", {}).get("compress_level", 1))

        self.count = 0
        self.closed = False
        self._uncommitted = 0
        self._seen_ids = set()

        self.db_path = self.out_dir / "dataset.db"
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_db()

    def _init_db(self):
        # Default rollback journal is used deliberately: WAL fails on some network/FUSE
        # filesystems where DAQ data is stored. synchronous=NORMAL is best-effort.
        try:
            self.conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.OperationalError:
            pass
        self.conn.executescript(PNG_DB_SCHEMA)
        _ensure_png_image_columns(self.conn)

        run_info = {
            "run_id": self.run_id,
            "resolved_config_json": json.dumps(self.resolved_config, sort_keys=True),
            "specs_json": json.dumps(self.specs, sort_keys=True),
            "processing_config_json": json.dumps(self.config, sort_keys=True),
            "script_name": Path(__file__).name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.conn.executemany(
            "INSERT OR REPLACE INTO run_info(key, value) VALUES(?, ?)",
            list(run_info.items()),
        )
        for camera, cfg in self.camera_configs.items():
            shape = tuple(int(part) for part in cfg["shape"])
            payload = cfg.get("payload_nbytes")
            self.conn.execute(
                "INSERT OR REPLACE INTO cameras"
                "(camera, dtype, height, width, payload_nbytes, saturation_value, settings_json)"
                " VALUES(?, ?, ?, ?, ?, ?, ?)",
                (
                    camera,
                    str(np.dtype(cfg["dtype"])),
                    shape[0],
                    shape[1],
                    int(payload) if payload is not None else None,
                    _camera_saturation(cfg),
                    json.dumps(cfg, sort_keys=True),
                ),
            )
        self.conn.commit()

    def _image_id(self, camera, meta):
        missing = [
            key for key in ("acq_stamp_ns", "frame_index")
            if meta.get(key) is None
        ]
        if missing:
            raise RuntimeError(
                "missing %s for PNG image id for camera %s"
                % (", ".join(missing), camera)
            )
        timestamp_ns = int(meta["acq_stamp_ns"])
        frame_index = int(meta["frame_index"])
        return "%s_%d_%d" % (camera, timestamp_ns, frame_index)

    def write(self, sample):
        if self.closed:
            raise RuntimeError("PngDbSink is closed")
        sample_id = int(sample.meta.get("sample_index", self.count))
        camera_meta = sample.meta.get("cameras", {})
        stats_meta = sample.meta.get("stats", {})
        centroid_meta = sample.meta.get("centroid", {})
        chosen_value = _json_or_none(sample.meta.get("chosen_sync_value"))
        chosen_source = _text_or_none(sample.meta.get("chosen_sync_source"))
        now = datetime.now(timezone.utc).isoformat()

        rows = []
        for camera, image in sample.images.items():
            meta = camera_meta.get(camera, {})
            image_id = self._image_id(camera, meta)
            if image_id in self._seen_ids:
                raise RuntimeError(
                    "duplicate image id for sample %d camera %s: %s"
                    % (sample_id, camera, image_id)
                )
            self._seen_ids.add(image_id)

            rel_path = "%s/%s.png" % (camera, image_id)
            abs_path = self.out_dir / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            _write_png(abs_path, image, self.compress_level)

            arr = np.asarray(image)
            stats = stats_meta.get(camera, {})
            centroid = centroid_meta.get(camera, {})
            rows.append((
                image_id,
                camera,
                sample_id,
                rel_path,
                str(arr.dtype),
                int(arr.shape[0]),
                int(arr.shape[1]),
                _sqlite_int(meta.get("frame_index")),
                _sqlite_int(meta.get("source_image_number")),
                _sqlite_int(meta.get("acq_stamp_ns")),
                _sqlite_int(meta.get("pulse_id")),
                _sqlite_crc(meta.get("crc_ok")),
                _text_or_none(meta.get("raw_segment_path")),
                _sqlite_int(meta.get("byte_offset")),
                chosen_value,
                chosen_source,
                _sqlite_float(stats.get("min")),
                _sqlite_float(stats.get("max")),
                _sqlite_float(stats.get("mean")),
                _sqlite_float(stats.get("sum")),
                int(bool(stats.get("saturated"))) if stats else None,
                _sqlite_float(stats.get("saturated_fraction")),
                _sqlite_float(centroid.get("x")),
                _sqlite_float(centroid.get("y")),
                now,
            ))

        placeholders = ", ".join(["?"] * len(_PNG_IMAGE_COLUMNS))
        self.conn.executemany(
            "INSERT INTO images(%s) VALUES(%s)" % (", ".join(_PNG_IMAGE_COLUMNS), placeholders),
            rows,
        )
        self.count += 1
        self._uncommitted += 1
        if self.commit_every > 0 and self._uncommitted >= self.commit_every:
            self.conn.commit()
            self._uncommitted = 0

    def close(self):
        if self.closed:
            return
        self.conn.commit()
        self.conn.close()
        self.closed = True


def _build_sink(backend, out_path, config):
    if backend == "hdf5":
        return Hdf5Sink(out_path, config)
    if backend == "png":
        return PngDbSink(out_path, config)
    raise RuntimeError("unknown backend %r (expected 'hdf5' or 'png')" % backend)


def _sqlite_int(value):
    return None if value is None else int(value)


def _sqlite_float(value):
    return None if value is None else float(value)


def _sqlite_crc(value):
    return None if value is None else int(bool(value))


def _json_or_none(value):
    return None if value is None else json.dumps(value, sort_keys=True)


def _text_or_none(value):
    return None if value is None else str(value)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_sync_config(resolved_config, config):
    """Return the run's sync config, optionally forcing the alignment key order.

    ``config['sync_priority']`` (when not None) overrides ``resolved_config.json``'s
    ``sync.priority`` so a run whose frozen config predates the recommended ordering
    can be re-parsed without editing it. Every other field in the run's sync block
    (windows, drift limits, ...) is preserved untouched.
    """
    sync_config = resolved_config.get("sync", {})
    if not isinstance(sync_config, dict):
        raise RuntimeError("resolved_config.json sync must be an object")
    override = config.get("sync_priority")
    if override is not None:
        if not isinstance(override, (list, tuple)) or not override:
            raise RuntimeError("CONFIG['sync_priority'] must be a non-empty list or None")
        sync_config = dict(sync_config)
        sync_config["priority"] = [str(source) for source in override]
    return sync_config


def _header_from_sidecar(frame_index, sync):
    """Rebuild the raw-style integer header from a sync sidecar row.

    PNG runs do not carry the binary per-record header, but the sidecar records
    the same fields, so downstream meta (frame_index / source_image_number /
    acq_stamp_ns / pulse_id) is identical to the raw path.
    """
    header = {"frame_index": int(frame_index)}
    if sync:
        if sync.get("source_image_number") is not None:
            header["source_image_number"] = int(sync["source_image_number"])
        if sync.get("acq_stamp_ns") is not None:
            header["acq_stamp_ns"] = int(sync["acq_stamp_ns"])
        if sync.get("raw_header_pulse_id") is not None:
            header["pulse_id"] = int(sync["raw_header_pulse_id"])
    return header


def _read_png_file(path):
    """Read a PNG into a numpy array.

    Prefers Pillow / imageio / cv2 when available, then falls back to a minimal
    stdlib decoder for the grayscale / RGB PNGs the stdlib writer emits.
    """
    try:
        from PIL import Image

        with Image.open(path) as img:
            return np.asarray(img)
    except ImportError:
        pass

    try:
        import imageio.v2 as imageio

        return np.asarray(imageio.imread(path))
    except ImportError:
        pass

    try:
        import cv2

        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise RuntimeError("cv2 failed to read png: %s" % path)
        return np.asarray(arr)
    except ImportError:
        pass

    return _read_png_file_stdlib(path)


def _read_png_file_stdlib(path):
    """Decode PNGs written by the shared ``optidaq.data.png_io`` encoder.

    Handles 8/16-bit grayscale and 8-bit RGB, non-interlaced, filter-type 0
    (what the stdlib writer emits). Raises a clear error otherwise so the caller
    knows to install Pillow.
    """
    with open(path, "rb") as handle:
        data = handle.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError("not a PNG file: %s" % path)

    pos = 8
    width = height = bit_depth = color_type = None
    idat = bytearray()
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        kind = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length  # length(4) + type(4) + data + crc(4)
        if kind == b"IHDR":
            width, height, bit_depth, color_type, _comp, _filter, interlace = (
                struct.unpack(">IIBBBBB", body)
            )
            if interlace != 0:
                raise RuntimeError("interlaced PNG not supported: %s" % path)
        elif kind == b"IDAT":
            idat += body
        elif kind == b"IEND":
            break

    if width is None:
        raise RuntimeError("PNG missing IHDR: %s" % path)
    if color_type == 0:
        channels = 1
    elif color_type == 2:
        channels = 3
    else:
        raise RuntimeError("unsupported PNG color_type %d in %s" % (color_type, path))
    if bit_depth not in (8, 16):
        raise RuntimeError("unsupported PNG bit depth %d in %s" % (bit_depth, path))

    sample_bytes = bit_depth // 8
    raw = zlib.decompress(bytes(idat))
    row_bytes = width * channels * sample_bytes
    stride = row_bytes + 1  # one filter-type byte per row
    if len(raw) != stride * height:
        raise RuntimeError("unexpected PNG data size in %s" % path)

    out = bytearray(row_bytes * height)
    for row in range(height):
        filter_type = raw[row * stride]
        if filter_type != 0:
            raise RuntimeError(
                "PNG row filter %d not supported by the stdlib reader "
                "(install Pillow to read this file): %s" % (filter_type, path)
            )
        start = row * stride + 1
        out[row * row_bytes:(row + 1) * row_bytes] = raw[start:start + row_bytes]

    if bit_depth == 8:
        arr = np.frombuffer(bytes(out), dtype=np.uint8)
    else:
        arr = np.frombuffer(bytes(out), dtype=">u2").astype(np.uint16)
    if channels == 1:
        return arr.reshape((height, width))
    return arr.reshape((height, width, channels))


def _make_source(dataset_dir, config):
    """Pick the input reader.

    ``input_format`` of ``raw`` / ``png`` forces a backend; ``auto`` (default)
    detects a PNG run by the presence of per-camera PNG directories (and the
    absence of raw segment files at the run root).
    """
    requested = str(config.get("input_format", "auto")).lower()
    if requested == "raw":
        return RawSource(dataset_dir, config)
    if requested == "png":
        return PngSource(dataset_dir, config)
    if requested != "auto":
        raise RuntimeError("input_format must be raw/png/auto, got %r" % requested)

    dataset_dir = Path(dataset_dir).expanduser()
    cameras = list(_camera_configs_from_resolved(
        _read_json(dataset_dir / "resolved_config.json")
    ))
    has_raw_segments = any(
        any(dataset_dir.glob("%s_*" % camera)) for camera in cameras
    )
    if has_raw_segments:
        return RawSource(dataset_dir, config)
    has_png_dirs = any((dataset_dir / camera).is_dir() for camera in cameras)
    if has_png_dirs:
        return PngSource(dataset_dir, config)
    # Nothing recognizable: fall back to raw so its existing error path fires.
    return RawSource(dataset_dir, config)


def _aligned_index_has_entries(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return True
    return False


def _count_nonempty_lines(path):
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _camera_configs_from_resolved(resolved_config):
    cameras = resolved_config.get("cameras")
    if not isinstance(cameras, dict):
        raise RuntimeError("resolved_config.json cameras must be an object")

    # Older runs recorded only the int16 wire dtype in each camera entry and left the pixel
    # format out. Recover it from the embedded input config's camera_setting so the real bit
    # depth (mono8 = 8-bit) is known without re-acquiring.
    camera_setting = {}
    inner = resolved_config.get("config")
    if isinstance(inner, dict) and isinstance(inner.get("camera_setting"), dict):
        camera_setting = inner["camera_setting"]

    out = {}
    for camera, value in cameras.items():
        cfg = dict(value) if isinstance(value, dict) else value
        if (
            isinstance(cfg, dict)
            and cfg.get("cameraPixelFormat") is None
            and cfg.get("bit_depth") is None
        ):
            setting = camera_setting.get(camera)
            if isinstance(setting, dict) and setting.get("cameraPixelFormat") is not None:
                cfg["cameraPixelFormat"] = setting["cameraPixelFormat"]
        out[str(camera)] = cfg
    return out


def _load_png_manifests(dataset_dir, cameras):
    manifests = {}
    for camera in cameras:
        path = Path(dataset_dir) / ("%s.png.json" % camera)
        if path.exists():
            manifest = _read_json(path)
            if manifest.get("schema_version") != 1:
                raise RuntimeError("unsupported PNG manifest: %s" % path)
            manifests[camera] = manifest
    return manifests


def _png_camera_configs(camera_configs, manifests):
    """Describe the pixels on disk while retaining original settings as provenance."""
    configs = {camera: dict(cfg) for camera, cfg in camera_configs.items()}
    for camera, manifest in manifests.items():
        cfg = configs[camera]
        stored = manifest["stored"]
        shape = [int(n) for n in stored["shape"]]
        dtype = np.dtype(stored["dtype"])
        if len(shape) != 2 or any(n <= 0 for n in shape) or dtype not in (np.dtype("uint8"), np.dtype("uint16")):
            raise RuntimeError("invalid stored PNG descriptor for %s" % camera)
        cfg["source"] = dict(cfg)
        saturation = _camera_saturation(cfg)
        cfg.update(shape=shape, dtype=str(dtype), payload_nbytes=int(np.prod(shape)) * dtype.itemsize)
        cfg["png"] = manifest
        if dtype == np.dtype("uint8"):
            cfg["bit_depth"] = 8
            cfg["saturation_value"] = 255
        elif saturation is not None:
            cfg["saturation_value"] = saturation * float(stored.get("value_scale", 1))
    return configs


def _load_specs(dataset_dir, cameras, *, required=True):
    specs = {}
    for camera in cameras:
        path = Path(dataset_dir) / ("%s.spec.json" % camera)
        if required or path.exists():
            specs[camera] = _read_json(path)
    return specs


def _header_struct(spec):
    if spec["endianness"] not in ENDIANNESS_FORMATS:
        raise RuntimeError("unsupported endianness %r" % spec["endianness"])
    formats = []
    for field in spec["header_fields"]:
        field_type = field["type"]
        if field_type not in TYPE_FORMATS:
            raise RuntimeError("unsupported header field type %r" % field_type)
        formats.append(TYPE_FORMATS[field_type])
    return struct.Struct(ENDIANNESS_FORMATS[spec["endianness"]] + "".join(formats))


def _sample_from_records(records, *, sample_index, chosen_sync_value, chosen_sync_source):
    images = {
        camera: record.image
        for camera, record in records.items()
    }
    meta = {
        "sample_index": int(sample_index),
        "chosen_sync_value": chosen_sync_value,
        "chosen_sync_source": chosen_sync_source,
        "cameras": {},
    }
    for camera, record in records.items():
        header = record.header
        pulse_id = header.get("pulse_id")
        if pulse_id == MISSING_U64:
            pulse_id = None
        meta["cameras"][camera] = {
            "frame_index": int(header.get("frame_index", record.global_record_index)),
            "source_image_number": int(header.get("source_image_number", 0)),
            "acq_stamp_ns": int(header.get("acq_stamp_ns", 0)),
            "pulse_id": pulse_id,
            "crc_ok": record.crc_ok,
            "raw_segment_path": str(record.segment_path),
            "byte_offset": int(record.byte_offset),
        }
    return Sample(images=images, meta=meta)


def _chosen_sync_from_record(record):
    if not record.sync:
        return None, None
    return record.sync.get("chosen_sync_source"), record.sync.get("chosen_sync_value")


def _select_sync_groups(sync_by_camera, cameras, sync_cfg, *, require_complete):
    reasons = []
    for source in sync_priority(sync_cfg):
        if source in LOCAL_CLOCK_SOURCES and bool(sync_cfg.get("forbid_local_clock", True)):
            reasons.append("%s: local clock is forbidden" % source)
            continue
        if source not in EXACT_SYNC_SOURCES and not bool(sync_cfg.get("allow_timestamp_fallback", True)):
            reasons.append("%s: timestamp fallback disabled" % source)
            continue

        records_by_camera = _sync_records_for_source(sync_by_camera, cameras, source)
        report = evaluate_sync_source(records_by_camera, cameras, sync_cfg, source)
        blocking = _blocking_sync_reasons(report)
        selected = [
            _parser_sync_group(group)
            for group in report.get("groups", [])
            if _usable_report_group(group, require_complete=require_complete)
        ]
        selected = [group for group in selected if group["frames"]]
        if selected and not blocking:
            skipped = int(len(report.get("groups", [])) - len(selected))
            return source, selected, reasons, skipped

        report_reasons = report.get("reasons", [])
        reasons.append(
            "%s: %s" % (
                source,
                "; ".join(report_reasons) if report_reasons else "no usable complete groups",
            )
        )

    return None, [], reasons, 0


def _sync_records_for_source(sync_by_camera, cameras, source):
    records = {camera: [] for camera in cameras}
    for camera in cameras:
        for frame_index, row in sync_by_camera.get(camera, {}).items():
            record = dict(row)
            record.setdefault("alias", camera)
            record.setdefault("frame_index", int(frame_index))
            if source == "pulse_id":
                _fill_raw_header_pulse_id(record)
            if record.get(source) is None and record.get("chosen_sync_source") == source:
                record[source] = record.get("chosen_sync_value")
            records[camera].append(record)
    return records


def _fill_raw_header_pulse_id(record):
    if record.get("pulse_id") is not None:
        return
    raw_pulse = record.get("raw_header_pulse_id")
    if raw_pulse is None:
        return
    try:
        if int(raw_pulse) == MISSING_U64:
            return
    except (TypeError, ValueError):
        pass
    record["pulse_id"] = raw_pulse


def _blocking_sync_reasons(report):
    # A few dropped/invalid groups should not block writing the complete matched samples.
    return [
        reason
        for reason in report.get("reasons", [])
        if "invalid group" not in reason
    ]


def _usable_report_group(group, *, require_complete):
    if require_complete:
        return bool(group.get("valid"))
    errors = group.get("errors", [])
    return not any("duplicate" in error for error in errors)


def _parser_sync_group(group):
    frames = {}
    for record in group.get("records", []):
        alias = str(record["alias"])
        if alias not in frames:
            frames[alias] = int(record["frame_index"])
    return {
        "value": group.get("sync_key"),
        "frames": frames,
    }


def _sync_sort_key(value):
    if isinstance(value, bool):
        return (2, str(value))
    if isinstance(value, (int, float)):
        return (0, float(value))
    return (1, json.dumps(value, sort_keys=True))


def _u64_or_missing(value):
    if value is None:
        return np.uint64(MISSING_U64)
    return np.uint64(int(value))


def _crc_value(value):
    if value is None:
        return UNKNOWN_CRC
    return np.int8(1 if bool(value) else 0)


def _text_or_empty(value):
    if value is None:
        return ""
    return str(value)


def _json_text(value):
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True)


def _effective_config():
    config = json.loads(json.dumps(CONFIG))
    core_dtype = config.get("core_dtype")
    if core_dtype is None:
        return config

    for transform in config.get("transforms", []):
        if transform.get("name") != "to_dtype":
            continue
        kwargs = transform.setdefault("kwargs", {})
        if kwargs.get("dtype") is None:
            kwargs["dtype"] = core_dtype
        break
    else:
        config.setdefault("transforms", []).insert(
            0,
            {"name": "to_dtype", "kwargs": {"dtype": core_dtype}},
        )
    return config


def _source_transform_configs(source, config):
    transforms = config.get("transforms", [])
    manifests = getattr(source, "png_manifests", {})
    already_processed = any(
        manifest.get("transforms") or manifest["stored"].get("value_scale", 1) != 1
        for manifest in manifests.values()
    )
    if already_processed and not config.get("process_png", False):
        # Metadata remains useful; image transforms require an explicit opt-in
        # so a run cropped/binned during acquisition is never processed twice.
        retained = [item for item in transforms if item.get("name") in ("stats", "centroid")]
        if len(retained) != len(transforms):
            warnings.warn("PNG pixels were processed during saving; skipping image transforms. Set process_png=true or --process-png to apply them again.")
        return retained
    return transforms


def main():
    parser = argparse.ArgumentParser(
        description="Parse a completed OptiDAQ run into an ML-ready dataset (HDF5, or PNG+SQLite)."
    )
    parser.add_argument("dataset_dir", nargs="?")
    parser.add_argument(
        "--backend",
        choices=["hdf5", "png"],
        default=None,
        help="output backend (default: CONFIG['backend'])",
    )
    parser.add_argument(
        "--input-format",
        dest="input_format",
        choices=["raw", "png", "auto"],
        default=None,
        help="input run format to read (default: CONFIG['input_format'], auto-detected)",
    )
    parser.add_argument(
        "--unaligned",
        action="store_true",
        help="ignore aligned_index and store every frame as its own row (align later via the .db)",
    )
    parser.add_argument(
        "--process-png",
        action="store_true",
        help="explicitly apply image transforms again to already-processed PNG input",
    )
    parser.add_argument(
        "--on-missing",
        dest="on_missing",
        choices=["error", "per_image", "sync_groups"],
        default=None,
        help="what to do when multi-camera alignment is missing/empty (default: CONFIG['on_alignment_missing'])",
    )
    args = parser.parse_args()

    config = _effective_config()
    if args.dataset_dir is not None:
        config["dataset_dir"] = args.dataset_dir
    if args.backend is not None:
        config["backend"] = args.backend
    if args.input_format is not None:
        config["input_format"] = args.input_format
    if args.unaligned:
        config["unaligned"] = True
    if args.process_png:
        config["process_png"] = True
    if args.on_missing is not None:
        config["on_alignment_missing"] = args.on_missing
    backend = config.get("backend", "hdf5")

    dataset_dir = Path(config["dataset_dir"]).expanduser()
    if not dataset_dir.is_dir():
        raise RuntimeError("dataset_dir is not a directory: %s" % dataset_dir)
    config["dataset_dir"] = str(dataset_dir)

    source = _make_source(dataset_dir, config)
    config["transforms"] = _source_transform_configs(source, config)
    # Use the stored PNG descriptor when present; raw runs keep acquisition truth.
    resolved_cameras = source.camera_configs
    saturation_by_camera = {
        camera: _camera_saturation(cfg) for camera, cfg in resolved_cameras.items()
    }
    bits_by_camera = {
        camera: _camera_bits(cfg) for camera, cfg in resolved_cameras.items()
    }
    for transform in config.get("transforms", []):
        name = transform.get("name")
        if name == "stats":
            transform.setdefault("kwargs", {})
            transform["kwargs"].setdefault("saturation_by_camera", saturation_by_camera)
        elif name == "minmax_stretch":
            transform.setdefault("kwargs", {})
            transform["kwargs"].setdefault("target_by_camera", saturation_by_camera)
        elif name == "to_uint16":
            transform.setdefault("kwargs", {})
            transform["kwargs"].setdefault("bits_by_camera", bits_by_camera)

    # Per-image / unaligned mode yields one camera per sample, which the hdf5 layout
    # (fixed camera set per sample) cannot represent.
    per_image_requested = config.get("unaligned") or config.get("on_alignment_missing") == "per_image"
    if backend == "hdf5" and len(saturation_by_camera) > 1 and per_image_requested:
        raise RuntimeError(
            "per-image / --unaligned mode is only supported by the png backend; "
            "the hdf5 backend needs aligned multi-camera samples (use --backend png)"
        )

    out_path = config.get("out_path")
    if out_path is None:
        out_path = dataset_dir / ("ml_dataset.h5" if backend == "hdf5" else "png_dataset")
    else:
        out_path = Path(out_path).expanduser()

    total_samples = source.estimated_sample_count()
    sink = _build_sink(backend, out_path, config)
    transforms = build_transforms(config["transforms"])
    progress = ProgressBar(total_samples)
    processed = 0

    try:
        progress.update(processed, force=True)
        for sample in source:
            processed += 1
            for transform in transforms:
                sample = transform(sample)
                if sample is None:
                    break
            if sample is not None:
                sink.write(sample)
            progress.update(processed)
    finally:
        progress.done(processed)
        sink.close()

    # Safety net: never report success for an empty dataset when raw frames exist on disk.
    if sink.count == 0 and source.has_raw_data():
        raise RuntimeError(
            "no samples were written, but the raw segments contain data — alignment likely "
            "failed. Pass --unaligned to store every frame (align later via the .db), or check "
            "aligned_index.jsonl / the sync sidecars."
        )

    print("wrote %d samples to %s" % (sink.count, out_path))


if __name__ == "__main__":
    main()
