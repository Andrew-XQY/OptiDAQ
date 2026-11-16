"""Save backends for the upstream DAQ pipeline."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from optidaq.daq.contracts import Frame
from optidaq.daq.raw_stream import RawStreamSpec, RawStreamWriter
from optidaq.data.hooks import HookContext, build_image_hooks, build_metadata_hooks
from optidaq.data.png_io import write_png


class _SyncSidecarWriter:
    """Write the per-device ``{device}.sync.jsonl`` sidecar.

    Original timing fields are shared by raw and PNG. PNG adds stored-image
    metadata; raw frames without ``sync_metadata`` remain a no-op.
    """

    def __init__(
        self,
        device: str,
        output_dir: Union[Path, str],
        *,
        overwrite: bool,
    ) -> None:
        self.device = device
        self.output_dir = Path(output_dir)
        self.overwrite = overwrite
        self._handle: Optional[Any] = None

    def write(self, frame: Frame, frame_index: int, *, extra: Optional[Mapping[str, Any]] = None) -> None:
        metadata = getattr(frame, "sync_metadata", None)
        if metadata is None and extra is None:
            return
        if self._handle is None:
            path = self.output_dir / ("%s.sync.jsonl" % self.device)
            mode = "w" if self.overwrite else "x"
            self._handle = open(path, mode, encoding="utf-8")

        entry = dict(metadata or {})
        entry.update(extra or {})
        entry.update(
            {
                "alias": self.device,
                "device": self.device,
                "frame_index": int(frame_index),
                "imageNumber": int(frame.source_image_number),
                "source_image_number": int(frame.source_image_number),
                "acq_stamp_ns": int(frame.acq_stamp_ns),
                "raw_header_pulse_id": int(frame.pulse_id),
            }
        )
        # default=str: never let an unexpected value abort the per-frame sidecar write.
        self._handle.write(json.dumps(entry, sort_keys=True, default=str))
        self._handle.write("\n")

    def close(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is not None:
            handle.flush()
            handle.close()


class RawStreamSaveBackend:
    """Save one device stream through ``RawStreamWriter``.

    This backend implements the ``SaveBackend`` protocol without adding pairing,
    ``shot_id`` values, image transforms, compression, or pickle. It records
    only upstream source facts plus the raw payload bytes.
    """

    def __init__(
        self,
        device: str,
        output_dir: Union[Path, str],
        spec: Union[RawStreamSpec, Mapping[str, Any]],
        *,
        overwrite: bool = False,
        fsync_on_close: bool = True,
        fsync_every_records: Optional[int] = None,
    ) -> None:
        if not isinstance(device, str) or not device:
            raise ValueError("device must be a non-empty str")
        self.device = device
        self.output_dir = Path(output_dir)
        self.spec = spec if isinstance(spec, RawStreamSpec) else RawStreamSpec.from_dict(spec)
        if self.device != self.spec.file_prefix:
            raise ValueError("device must match spec.file_prefix")
        self._validate_header_contract()
        self.overwrite = overwrite
        self.fsync_on_close = fsync_on_close
        self.fsync_every_records = fsync_every_records
        self._writer: Optional[RawStreamWriter] = None
        self._sidecar = _SyncSidecarWriter(
            self.device, self.output_dir, overwrite=self.overwrite
        )

    def open(self) -> None:
        """Open the underlying raw stream writer."""

        if self._writer is not None:
            raise RuntimeError("raw stream backend is already open")
        self._writer = RawStreamWriter(
            self.output_dir,
            self.spec,
            overwrite=self.overwrite,
            fsync_on_close=self.fsync_on_close,
            fsync_every_records=self.fsync_every_records,
        )

    def save(self, frame: Frame) -> None:
        """Append one frame to this device's raw stream."""

        if self._writer is None:
            raise RuntimeError("raw stream backend is not open")
        if frame.device != self.device:
            raise RuntimeError(
                "frame device %r does not match backend device %r"
                % (frame.device, self.device)
            )
        header_values = {
            "source_image_number": frame.source_image_number,
            "acq_stamp_ns": frame.acq_stamp_ns,
        }
        if "pulse_id" in self.spec.header_field_names:
            header_values["pulse_id"] = frame.pulse_id
        location = self._writer.append(frame.payload_bytes, **header_values)
        self._sidecar.write(frame, location.global_record_index)

    def close(self) -> None:
        """Close the underlying writer if it was opened."""

        writer = self._writer
        self._writer = None
        try:
            if writer is not None:
                writer.close()
        finally:
            self._sidecar.close()

    def _validate_header_contract(self) -> None:
        expected_v1 = [
            {"name": "frame_index", "type": "uint64"},
            {"name": "source_image_number", "type": "uint64"},
            {"name": "acq_stamp_ns", "type": "uint64"},
        ]
        expected_v2 = expected_v1 + [
            {"name": "pulse_id", "type": "uint64"},
        ]
        if self.spec.to_dict()["header_fields"] not in (expected_v1, expected_v2):
            raise ValueError(
                "RawStreamSaveBackend requires frame_index, "
                "source_image_number, acq_stamp_ns, optional pulse_id "
                "uint64 headers"
            )

class PngImageSaveBackend:
    """Save one device stream as lossless PNG images."""

    def __init__(
        self,
        device: str,
        output_dir: Union[Path, str],
        *,
        height: int,
        width: int,
        dtype: Any,
        channels: int = 1,
        value_scale: int = 1,
        transforms: Optional[Sequence[Any]] = None,
        metadata_hooks: Optional[Sequence[Any]] = None,
        compress_level: int = 6,
        source_dtype: Any = None,
        overwrite: bool = False,
        fsync_on_close: bool = True,
    ) -> None:
        if not isinstance(device, str) or not device:
            raise ValueError("device must be a non-empty str")
        if type(height) is not int or height <= 0:
            raise ValueError("height must be a positive int")
        if type(width) is not int or width <= 0:
            raise ValueError("width must be a positive int")
        if type(channels) is not int or channels <= 0:
            raise ValueError("channels must be a positive int")

        self._dtype = np.dtype(dtype)
        if self._dtype not in (np.dtype("uint8"), np.dtype("uint16")):
            raise ValueError("PngImageSaveBackend supports only uint8 or uint16")
        if channels not in (1, 3):
            raise ValueError("channels must be 1 or 3")
        if channels == 3 and self._dtype != np.dtype("uint8"):
            raise ValueError("channels=3 requires uint8")
        if type(value_scale) is not int or value_scale <= 0:
            raise ValueError("value_scale must be a positive int")
        if value_scale != 1 and (
            self._dtype != np.dtype("uint16") or channels != 1
        ):
            raise ValueError("value_scale requires uint16 grayscale")

        self.device = device
        self.height = height
        self.width = width
        self.channels = channels
        self.value_scale = value_scale
        self.overwrite = overwrite
        self.fsync_on_close = fsync_on_close
        self.expected_nbytes = height * width * channels * self._dtype.itemsize
        if type(compress_level) is not int or not 0 <= compress_level <= 9:
            raise ValueError("PNG compress_level must be an integer in 0..9")
        self.compress_level = compress_level
        self.transforms = list(transforms or [])
        self.metadata_hooks = list(metadata_hooks or [])
        self._image_hooks = build_image_hooks(self.transforms)
        self._metadata_hooks = build_metadata_hooks(self.metadata_hooks)
        names = [s if isinstance(s, str) else s.get("name") for s in self.transforms]
        metadata_names = [s if isinstance(s, str) else s.get("name") for s in self.metadata_hooks]
        self._crop_spec = None
        if "crop_energy" in metadata_names and names:
            # This hook measures a source-coordinate ROI. Do not invent one for
            # custom geometry or for cropping after binning.
            tail = names
            if names[0] in ("center_crop", "crop_box"):
                self._crop_spec = self.transforms[0]
                tail = names[1:]
            if any(name not in ("avg_bin", "normalize", "to_uint8") for name in tail):
                raise ValueError("crop_energy supports one initial crop followed by bin/value transforms")
        self._source_dtype = str(np.dtype(source_dtype if source_dtype is not None else dtype))
        self._stored: Optional[dict] = None
        self._manifest_path = Path(output_dir) / (device + ".png.json")
        self._dir = Path(output_dir) / device
        # Sidecar lives at the run-dir level ({output_dir}/{device}.sync.jsonl),
        # identical to the raw backend, so downstream alignment is unchanged.
        self._sidecar = _SyncSidecarWriter(device, Path(output_dir), overwrite=overwrite)
        self._frame_index: Optional[int] = None

    def open(self) -> None:
        """Open the PNG output directory."""

        if self._frame_index is not None:
            raise RuntimeError("png backend is already open")
        if self._dir.exists():
            if not self._dir.is_dir():
                raise ValueError("png output path exists and is not a directory")
            if not self.overwrite and any(self._dir.iterdir()):
                raise ValueError("png output directory is not empty")
        self._dir.mkdir(parents=True, exist_ok=True)
        if self._manifest_path.exists() and not self.overwrite:
            raise ValueError("png manifest already exists")
        self._stored = None
        self._frame_index = 0

    def save(self, frame: Frame) -> None:
        """Write one frame payload as one PNG image."""

        if self._frame_index is None:
            raise RuntimeError("png backend is not open")
        if frame.device != self.device:
            raise RuntimeError(
                "frame device %r does not match backend device %r"
                % (frame.device, self.device)
            )
        if len(frame.payload_bytes) != self.expected_nbytes:
            raise RuntimeError(
                "payload byte count mismatch: expected %d, got %d"
                % (self.expected_nbytes, len(frame.payload_bytes))
            )

        arr = np.frombuffer(frame.payload_bytes, dtype=self._dtype)
        if self.channels == 1:
            arr = arr.reshape((self.height, self.width))
        else:
            arr = arr.reshape((self.height, self.width, self.channels))
        source = arr
        context = HookContext(
            run_id=self._dir.parent.name, camera=self.device,
            image_id="%s_%d_%d" % (self.device, frame.acq_stamp_ns, self._frame_index),
            frame_index=self._frame_index, acq_stamp_ns=frame.acq_stamp_ns,
            source_image_number=frame.source_image_number, pulse_id=frame.pulse_id,
            sync=deepcopy(getattr(frame, "sync_metadata", {}) or {}),
            raw_header={"frame_index": self._frame_index,
                        "source_image_number": frame.source_image_number,
                        "acq_stamp_ns": frame.acq_stamp_ns, "pulse_id": frame.pulse_id},
            raw_segment_path="", byte_offset=0, source_image=source,
            applied_crop=self._source_crop(),
        )
        # A hook may work in place, but never on the original Frame payload.
        if self._image_hooks:
            arr = arr.copy()
        for hook in self._image_hooks:
            arr = np.asarray(hook(arr, context))
        if arr.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
            raise ValueError("PNG hooks must return uint8 or uint16; choose an explicit conversion")
        if not arr.size or not (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 3 and arr.dtype == np.dtype("uint8"))):
            raise ValueError("PNG hooks must return nonempty grayscale or uint8 RGB")
        if self.value_scale != 1:
            if arr.dtype != np.dtype("uint16") or arr.ndim != 2:
                raise ValueError("display scaling requires uint16 grayscale hook output")
            arr = np.minimum(arr.astype(np.uint32) * self.value_scale, 65535).astype(np.uint16)

        stored = {"shape": list(arr.shape), "dtype": str(arr.dtype), "value_scale": self.value_scale}
        if self._stored is not None and self._stored != stored:
            raise ValueError("PNG hook output shape/dtype must stay fixed within a camera run")
        metadata = {}
        readonly = arr.view()
        readonly.setflags(write=False)
        for hook in self._metadata_hooks:
            result = hook(readonly, context)
            if not isinstance(result, Mapping):
                raise ValueError("PNG metadata hooks must return a mapping")
            metadata.update(result)
        # Validate hook metadata before publishing a PNG (do not silently stringify).
        json.dumps(metadata, allow_nan=False)
        if self._stored is None:
            manifest = {"schema_version": 1,
                        "source": {"shape": list(source.shape), "dtype": self._source_dtype},
                        "stored": stored, "transforms": self.transforms,
                        "metadata_hooks": self.metadata_hooks,
                        "compress_level": self.compress_level}
            _write_json_atomic(self._manifest_path, manifest)
            self._stored = stored

        name = "%020d_%08d.png" % (frame.acq_stamp_ns, self._frame_index)
        final_path = self._dir / name
        fd, tmp_name = tempfile.mkstemp(
            prefix=".tmp_",
            suffix=".png",
            dir=str(self._dir),
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            write_png(tmp_path, arr, compress_level=self.compress_level)
            _fsync_file(tmp_path)
            os.replace(tmp_path, final_path)
        except BaseException:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

        # Same key as the PNG filename's frame index, so the sidecar lines up
        # one-to-one with the image files for the downstream reader.
        self._sidecar.write(frame, self._frame_index, extra={"png": {
            "shape": stored["shape"], "dtype": stored["dtype"], "metadata": metadata,
        }})
        self._frame_index += 1

    def _source_crop(self) -> Optional[tuple]:
        if self._crop_spec is None:
            return None
        params = self._crop_spec.get("params", {})
        if self._crop_spec["name"] == "crop_box":
            return tuple(params[key] for key in ("x1", "y1", "x2", "y2"))
        size = int(params["size"])
        x = (self.width - size) // 2
        y = (self.height - size) // 2
        return (x, y, x + size, y + size)

    @property
    def frame_count(self) -> int:
        """Number saved by this open writer (used for explicit lab capture IDs)."""
        if self._frame_index is None:
            raise RuntimeError("png backend is not open")
        return self._frame_index

    def flush(self) -> None:
        """Make PNG names and their sidecar durable before a lab pair commits."""
        if self._frame_index is None:
            raise RuntimeError("png backend is not open")
        handle = self._sidecar._handle
        if handle is not None:
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(self._dir)
        _fsync_directory(self._dir.parent)

    def close(self) -> None:
        """Close the PNG backend if it was opened."""

        if self._frame_index is None:
            return
        try:
            if self.fsync_on_close:
                _fsync_directory(self._dir)
        finally:
            self._sidecar.close()
            self._frame_index = None


class MultiSaveBackend:
    """Tee one device stream to multiple save backends."""

    def __init__(self, device: str, backends: Sequence[Any]) -> None:
        if not isinstance(device, str) or not device:
            raise ValueError("device must be a non-empty str")
        self.device = device
        self.backends = list(backends)
        if not self.backends:
            raise ValueError("backends must not be empty")
        for backend in self.backends:
            if getattr(backend, "device", None) != device:
                raise ValueError("child backend device must match device")
        self._opened_backends: Optional[list] = None

    def open(self) -> None:
        """Open all child backends."""

        if self._opened_backends is not None:
            raise RuntimeError("multi backend is already open")

        opened = []
        try:
            for backend in self.backends:
                backend.open()
                opened.append(backend)
        except BaseException:
            for backend in reversed(opened):
                try:
                    backend.close()
                except BaseException:
                    pass
            raise

        self._opened_backends = opened

    def save(self, frame: Frame) -> None:
        """Save one frame to all child backends in order."""

        if self._opened_backends is None:
            raise RuntimeError("multi backend is not open")
        if frame.device != self.device:
            raise RuntimeError(
                "frame device %r does not match backend device %r"
                % (frame.device, self.device)
            )
        for backend in self.backends:
            backend.save(frame)

    def close(self) -> None:
        """Close all child backends, preserving the first close error."""

        if self._opened_backends is None:
            return

        first_error: Optional[BaseException] = None
        try:
            for backend in reversed(self._opened_backends):
                try:
                    backend.close()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
        finally:
            self._opened_backends = None

        if first_error is not None:
            raise first_error


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    fd, name = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def output_format(config: Mapping[str, Any]) -> str:
    output = config.get("output", {})
    if not isinstance(output, Mapping):
        raise RuntimeError("output must be an object")
    name = str(output.get("format", "raw")).lower()
    if name not in ("raw", "png"):
        raise RuntimeError("output format must be 'raw' or 'png'")
    return name


def png_settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    settings = config.get("output", {}).get("png", {})
    if not isinstance(settings, Mapping):
        raise RuntimeError("output.png must be an object")
    return settings


def png_mode(config: Mapping[str, Any]) -> str:
    # Keep old configs valid; the nested setting takes precedence.
    mode = str(png_settings(config).get("mode", config.get("output", {}).get("png_mode", "raw16"))).lower()
    if mode not in ("raw16", "mono12_display16"):
        raise RuntimeError("PNG mode must be 'raw16' or 'mono12_display16'")
    return mode


def _camera_hook_specs(settings: Mapping[str, Any], key: str, device: str) -> list:
    specs = settings.get(key, [])
    if isinstance(specs, Mapping):
        specs = specs.get(device, [])
    if not isinstance(specs, list):
        raise ValueError("output.png.%s must be a hook list or per-camera hook lists" % key)
    return specs


def make_save_backend(
    device: str,
    output_dir: Union[Path, str],
    spec: Union[RawStreamSpec, Mapping[str, Any]],
    *,
    image_shape: Sequence[int],
    image_dtype: Any,
    config: Mapping[str, Any],
    fsync_every_records: Optional[int] = None,
) -> Any:
    """The single backend selector shared by experiment and background capture."""

    if output_format(config) == "raw":
        return RawStreamSaveBackend(device, output_dir, spec,
                                    fsync_every_records=fsync_every_records)
    settings = png_settings(config)
    shape = tuple(image_shape)
    if len(shape) not in (2, 3):
        raise ValueError("camera %s has unsupported PNG shape %s" % (device, shape))
    dtype = np.dtype(image_dtype)
    # CERN Mono12 can arrive in a signed 16-bit container. Preserve its bits.
    if dtype.kind == "i" and dtype.itemsize == 2:
        dtype = np.dtype("uint16")
    return PngImageSaveBackend(
        device, output_dir, height=int(shape[0]), width=int(shape[1]),
        channels=int(shape[2]) if len(shape) == 3 else 1,
        dtype=dtype, source_dtype=image_dtype,
        value_scale=16 if png_mode(config) == "mono12_display16" else 1,
        transforms=_camera_hook_specs(settings, "transforms", device),
        metadata_hooks=_camera_hook_specs(settings, "metadata_hooks", device),
        compress_level=settings.get("compress_level", 6),
    )


def _fsync_file(path: Path) -> None:
    with open(path, "rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
