"""Append-only fixed-record raw byte streams for DAQ image saving.

This module saves and parses raw payload bytes according to a fixed binary
record layout:

    [fixed-width header fields][raw payload bytes]

The binary segment files are deliberately not self-describing. The sidecar
``{file_prefix}.spec.json`` is the source of truth for decoding and must be
preserved with the ``.raw`` segment files for long-term readability. The same
canonical spec must be used for writing and reading.

Higher-level experiment details such as image shape, dtype, pixel format,
camera identity, clock meaning, or run metadata can live in an external run
configuration. This module only persists the low-level binary decoding spec:
``schema_version``, ``endianness``, ``header_fields``, ``payload_nbytes``,
``file_prefix``, ``file_extension``, and ``segment_max_bytes``.

For multi-camera acquisition, write one stream per camera/device. Pairing is
offline. A raw record should store source facts such as ``frame_index``,
``source_image_number``, ``acq_stamp_ns``, and payload bytes. Offline tools may
later derive ``cluster_id`` or ``offline_shot_id`` without changing raw files.

Parsers ignore incomplete final records after a crash unless strict validation
is requested; incomplete bytes are never interpreted as a valid frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import struct
import threading
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)


class RawStreamError(Exception):
    """Base class for raw stream errors."""


class RawStreamSpecError(RawStreamError):
    """Raised when a raw stream spec is invalid."""


class RawStreamWriteError(RawStreamError):
    """Raised when raw stream writing fails."""


class RawStreamParseError(RawStreamError):
    """Raised when raw stream parsing or validation fails."""


_TYPE_FORMATS: Dict[str, Tuple[str, int, int]] = {
    "uint64": ("Q", 0, (1 << 64) - 1),
    "int64": ("q", -(1 << 63), (1 << 63) - 1),
    "uint32": ("I", 0, (1 << 32) - 1),
    "int32": ("i", -(1 << 31), (1 << 31) - 1),
}

_ENDIANNESS_FORMATS = {
    "little": "<",
    "big": ">",
}

_REQUIRED_SPEC_KEYS = {
    "schema_version",
    "endianness",
    "header_fields",
    "payload_nbytes",
    "segment_max_bytes",
    "file_prefix",
    "file_extension",
}

_REQUIRED_FIELD_KEYS = {"name", "type"}


@dataclass(frozen=True)
class HeaderField:
    """One fixed-width integer field in a raw stream record header."""

    name: str
    type: str


@dataclass(frozen=True)
class RawStreamSpec:
    """Validated fixed-record stream specification.

    The binary stream is not self-describing. The writer persists the canonical
    dictionary returned by :meth:`to_dict` to ``{file_prefix}.spec.json`` before
    writing any segment data. That sidecar is mandatory for long-term decoding
    of the raw segments.

    The spec describes only the byte-level record contract. Shape, dtype,
    pixel format, camera details, and experiment metadata belong in external
    run configuration. The same spec must be used for reading and writing.
    """

    schema_version: int
    endianness: str
    header_fields: Tuple[HeaderField, ...]
    payload_nbytes: int
    segment_max_bytes: int
    file_prefix: str
    file_extension: str
    header_nbytes: int
    record_nbytes: int
    header_struct: struct.Struct
    _canonical_spec: Mapping[str, Any]

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RawStreamSpec":
        """Build and validate a spec from a plain mapping."""

        if not isinstance(d, Mapping):
            raise RawStreamSpecError("spec must be a mapping")

        keys = set(d.keys())
        missing = sorted(_REQUIRED_SPEC_KEYS - keys)
        if missing:
            raise RawStreamSpecError(
                "spec is missing required keys: " + ", ".join(missing)
            )

        unknown = sorted(keys - _REQUIRED_SPEC_KEYS)
        if unknown:
            raise RawStreamSpecError(
                "spec has unknown keys: " + ", ".join(unknown)
            )

        schema_version = _require_int(d["schema_version"], "schema_version")
        if schema_version < 1:
            raise RawStreamSpecError("schema_version must be positive")

        endianness = d["endianness"]
        if endianness not in _ENDIANNESS_FORMATS:
            raise RawStreamSpecError('endianness must be "little" or "big"')

        payload_nbytes = _require_int(d["payload_nbytes"], "payload_nbytes")
        if payload_nbytes <= 0:
            raise RawStreamSpecError("payload_nbytes must be positive")

        segment_max_bytes = _require_int(
            d["segment_max_bytes"], "segment_max_bytes"
        )

        file_prefix = _require_non_empty_str(d["file_prefix"], "file_prefix")
        if "/" in file_prefix or "\\" in file_prefix:
            raise RawStreamSpecError("file_prefix must not contain path separators")

        file_extension = _require_non_empty_str(
            d["file_extension"], "file_extension"
        )
        if not file_extension.startswith("."):
            raise RawStreamSpecError("file_extension must start with '.'")
        if "/" in file_extension or "\\" in file_extension:
            raise RawStreamSpecError("file_extension must not contain path separators")

        raw_fields = d["header_fields"]
        if not isinstance(raw_fields, Sequence) or isinstance(
            raw_fields, (str, bytes, bytearray)
        ):
            raise RawStreamSpecError("header_fields must be a non-empty list")
        if not raw_fields:
            raise RawStreamSpecError("header_fields must be a non-empty list")

        fields: List[HeaderField] = []
        names = set()
        formats: List[str] = []
        canonical_fields: List[Dict[str, str]] = []

        for index, raw_field in enumerate(raw_fields):
            if not isinstance(raw_field, Mapping):
                raise RawStreamSpecError(
                    "header_fields[%d] must be a mapping" % index
                )
            field_keys = set(raw_field.keys())
            missing_field_keys = sorted(_REQUIRED_FIELD_KEYS - field_keys)
            if missing_field_keys:
                raise RawStreamSpecError(
                    "header_fields[%d] is missing required keys: %s"
                    % (index, ", ".join(missing_field_keys))
                )
            unknown_field_keys = sorted(field_keys - _REQUIRED_FIELD_KEYS)
            if unknown_field_keys:
                raise RawStreamSpecError(
                    "header_fields[%d] has unknown keys: %s"
                    % (index, ", ".join(unknown_field_keys))
                )

            name = _require_non_empty_str(raw_field["name"], "header field name")
            field_type = _require_non_empty_str(
                raw_field["type"], "header field type"
            )
            if field_type not in _TYPE_FORMATS:
                supported = ", ".join(sorted(_TYPE_FORMATS))
                raise RawStreamSpecError(
                    "unsupported header field type %r; supported: %s"
                    % (field_type, supported)
                )
            if name in names:
                raise RawStreamSpecError("duplicate header field name %r" % name)

            names.add(name)
            fields.append(HeaderField(name=name, type=field_type))
            formats.append(_TYPE_FORMATS[field_type][0])
            canonical_fields.append({"name": name, "type": field_type})

        header_struct = struct.Struct(_ENDIANNESS_FORMATS[endianness] + "".join(formats))
        header_nbytes = header_struct.size
        record_nbytes = header_nbytes + payload_nbytes

        if segment_max_bytes < record_nbytes:
            raise RawStreamSpecError(
                "segment_max_bytes must fit at least one full record "
                "(%d bytes)" % record_nbytes
            )

        canonical_spec: Dict[str, Any] = {
            "schema_version": schema_version,
            "endianness": endianness,
            "header_fields": canonical_fields,
            "payload_nbytes": payload_nbytes,
            "segment_max_bytes": segment_max_bytes,
            "file_prefix": file_prefix,
            "file_extension": file_extension,
        }

        return cls(
            schema_version=schema_version,
            endianness=endianness,
            header_fields=tuple(fields),
            payload_nbytes=payload_nbytes,
            segment_max_bytes=segment_max_bytes,
            file_prefix=file_prefix,
            file_extension=file_extension,
            header_nbytes=header_nbytes,
            record_nbytes=record_nbytes,
            header_struct=header_struct,
            _canonical_spec=canonical_spec,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return the canonical spec dict persisted to the sidecar."""

        return {
            "schema_version": self._canonical_spec["schema_version"],
            "endianness": self._canonical_spec["endianness"],
            "header_fields": [
                {"name": field["name"], "type": field["type"]}
                for field in self._canonical_spec["header_fields"]
            ],
            "payload_nbytes": self._canonical_spec["payload_nbytes"],
            "segment_max_bytes": self._canonical_spec["segment_max_bytes"],
            "file_prefix": self._canonical_spec["file_prefix"],
            "file_extension": self._canonical_spec["file_extension"],
        }

    @property
    def header_field_names(self) -> Tuple[str, ...]:
        """Header field names in binary packing order."""

        return tuple(field.name for field in self.header_fields)

    def pack_header(self, values: Mapping[str, int]) -> bytes:
        """Pack a header mapping according to this spec."""

        packed_values = []
        for field in self.header_fields:
            if field.name not in values:
                raise RawStreamWriteError(
                    "missing header value for field %r" % field.name
                )
            value = values[field.name]
            _validate_int_for_type(value, field.name, field.type)
            packed_values.append(value)
        try:
            return self.header_struct.pack(*packed_values)
        except struct.error as exc:
            raise RawStreamWriteError("failed to pack header: %s" % exc) from exc

    def unpack_header(self, data: bytes) -> Dict[str, int]:
        """Unpack exactly ``header_nbytes`` bytes into a header dict."""

        if len(data) != self.header_nbytes:
            raise RawStreamParseError(
                "header byte count mismatch: expected %d, got %d"
                % (self.header_nbytes, len(data))
            )
        try:
            values = self.header_struct.unpack(data)
        except struct.error as exc:
            raise RawStreamParseError("failed to unpack header: %s" % exc) from exc
        return {
            field.name: int(value)
            for field, value in zip(self.header_fields, values)
        }


@dataclass(frozen=True)
class RecordLocation:
    """Location returned after a successful synchronous append."""

    segment_index: int
    segment_path: Path
    record_index_in_segment: int
    global_record_index: int
    byte_offset: int
    record_nbytes: int


@dataclass(frozen=True)
class RawRecord:
    """One parsed fixed-size raw stream record."""

    header: Dict[str, int]
    payload: bytes
    source_path: Path
    segment_index: int
    record_index_in_segment: int
    global_record_index: int
    byte_offset: int


@dataclass(frozen=True)
class ValidationReport:
    """Summary from validating a raw stream without loading all payload bytes."""

    segment_file_count: int
    valid_record_count: int
    incomplete_tail_bytes: Dict[str, int]
    sidecar_exists: bool
    sidecar_matches: Optional[bool]
    first_header: Optional[Dict[str, int]]
    last_header: Optional[Dict[str, int]]
    warnings: List[str]
    errors: List[str]


class RawStreamWriter:
    """Synchronous append-only writer for one fixed-record raw stream.

    The writer creates ``output_dir`` if needed, writes
    ``{file_prefix}.spec.json`` before any segment data, writes open segments
    with a ``.part`` suffix, and finalizes them to ``file_extension`` on
    rollover or normal close. Existing sidecars or segment files are never
    overwritten unless ``overwrite=True`` is passed.

    Payload bytes are written exactly as provided; this class does not transform
    images, compress data, parse JSON per frame, or know anything about cameras.
    If a write error occurs, the current ``.part`` file is left in place for
    recovery and the writer refuses further appends.
    """

    def __init__(
        self,
        output_dir: Union[Path, str],
        spec: RawStreamSpec,
        start_segment_index: int = 0,
        overwrite: bool = False,
        fsync_on_close: bool = True,
        fsync_every_records: Optional[int] = None,
    ) -> None:
        if start_segment_index < 0:
            raise RawStreamWriteError("start_segment_index must be non-negative")
        if fsync_every_records is not None and fsync_every_records <= 0:
            raise RawStreamWriteError("fsync_every_records must be positive or None")

        self.output_dir = Path(output_dir)
        self.spec = spec
        self.start_segment_index = start_segment_index
        self.overwrite = overwrite
        self.fsync_on_close = fsync_on_close
        self.fsync_every_records = fsync_every_records

        self._segment_index = start_segment_index
        self._record_index_in_segment = 0
        self._global_record_index = 0
        self._bytes_in_segment = 0
        self._next_frame_index = 0
        self._file: Optional[Any] = None
        self._part_path: Optional[Path] = None
        self._closed = False
        self._failed: Optional[BaseException] = None

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._write_sidecar()
        except RawStreamError:
            raise
        except Exception as exc:
            raise RawStreamWriteError("failed to initialize writer: %s" % exc) from exc

    def __enter__(self) -> "RawStreamWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if exc_type is not None:
            try:
                self._close_open_file(finalize=False)
            except RawStreamError:
                pass
            finally:
                self._closed = True
            return False
        self.close()
        return False

    def append(
        self,
        payload: Union[bytes, bytearray, memoryview],
        **header_values: int,
    ) -> RecordLocation:
        """Append one record and return its location.

        ``payload`` must be exactly ``spec.payload_nbytes`` bytes. All header
        fields must be supplied except ``frame_index``, which is auto-filled
        from zero when present in the spec and omitted by the caller. All other
        header fields are supplied by upstream code.
        """

        self._ensure_usable()
        payload_view = _payload_memoryview(payload)
        if payload_view.nbytes != self.spec.payload_nbytes:
            raise RawStreamWriteError(
                "payload byte count mismatch: expected %d, got %d"
                % (self.spec.payload_nbytes, payload_view.nbytes)
            )

        complete_header = self._prepare_header_values(header_values)
        header = self.spec.pack_header(complete_header)

        if self._file is not None and (
            self._bytes_in_segment + self.spec.record_nbytes
            > self.spec.segment_max_bytes
        ):
            self._rollover()

        self._ensure_open_segment()
        assert self._file is not None
        assert self._part_path is not None

        byte_offset = self._bytes_in_segment
        record_index_in_segment = self._record_index_in_segment
        global_record_index = self._global_record_index
        segment_index = self._segment_index
        segment_path = self._part_path

        try:
            self._file.write(header)
            self._file.write(payload_view)
            self._bytes_in_segment += self.spec.record_nbytes
            self._record_index_in_segment += 1
            self._global_record_index += 1
            if "frame_index" in self.spec.header_field_names:
                self._next_frame_index += 1

            if (
                self.fsync_every_records is not None
                and self._global_record_index % self.fsync_every_records == 0
            ):
                self._file.flush()
                os.fsync(self._file.fileno())
        except Exception as exc:
            self._failed = exc
            raise RawStreamWriteError("failed to append raw record: %s" % exc) from exc

        return RecordLocation(
            segment_index=segment_index,
            segment_path=segment_path,
            record_index_in_segment=record_index_in_segment,
            global_record_index=global_record_index,
            byte_offset=byte_offset,
            record_nbytes=self.spec.record_nbytes,
        )

    def close(self) -> None:
        """Finalize the open ``.part`` file to ``.raw`` on normal close."""

        if self._closed:
            return
        if self._failed is not None:
            self._close_open_file(finalize=False)
            self._closed = True
            raise RawStreamWriteError(
                "writer is failed; current .part file was left for recovery"
            ) from self._failed
        self._close_open_file(finalize=True)
        self._closed = True

    def _ensure_usable(self) -> None:
        if self._closed:
            raise RawStreamWriteError("writer is closed")
        if self._failed is not None:
            raise RawStreamWriteError("writer is failed") from self._failed

    def _prepare_header_values(
        self, header_values: Mapping[str, int]
    ) -> Dict[str, int]:
        field_names = set(self.spec.header_field_names)
        unknown = sorted(set(header_values.keys()) - field_names)
        if unknown:
            raise RawStreamWriteError(
                "unknown header values: " + ", ".join(unknown)
            )

        complete = dict(header_values)
        if "frame_index" in field_names and "frame_index" not in complete:
            complete["frame_index"] = self._next_frame_index

        missing = sorted(field_names - set(complete.keys()))
        if missing:
            raise RawStreamWriteError(
                "missing header values: " + ", ".join(missing)
            )
        return complete

    def _write_sidecar(self) -> None:
        sidecar_path = self.output_dir / ("%s.spec.json" % self.spec.file_prefix)
        spec_text = json.dumps(self.spec.to_dict(), indent=2, sort_keys=True)

        if sidecar_path.exists() and not self.overwrite:
            raise RawStreamWriteError(
                "sidecar already exists: %s" % sidecar_path
            )

        tmp_path = self.output_dir / (
            "%s.spec.json.tmp.%d" % (self.spec.file_prefix, os.getpid())
        )
        try:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                handle.write(spec_text)
                handle.flush()
                os.fsync(handle.fileno())

            if self.overwrite:
                os.replace(tmp_path, sidecar_path)
            else:
                try:
                    os.link(tmp_path, sidecar_path)
                except FileExistsError as exc:
                    raise RawStreamWriteError(
                        "sidecar already exists: %s" % sidecar_path
                    ) from exc
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()
        except RawStreamError:
            raise
        except Exception as exc:
            raise RawStreamWriteError("failed to write sidecar: %s" % exc) from exc

    def _ensure_open_segment(self) -> None:
        if self._file is not None:
            return
        part_path = _segment_path(self.output_dir, self.spec, self._segment_index, True)
        final_path = _segment_path(
            self.output_dir, self.spec, self._segment_index, False
        )
        if not self.overwrite:
            existing = [path for path in (part_path, final_path) if path.exists()]
            if existing:
                raise RawStreamWriteError(
                    "segment file already exists: %s" % existing[0]
                )
        try:
            self._file = open(part_path, "wb")
        except Exception as exc:
            raise RawStreamWriteError(
                "failed to open segment %s: %s" % (part_path, exc)
            ) from exc
        self._part_path = part_path
        self._bytes_in_segment = 0
        self._record_index_in_segment = 0

    def _rollover(self) -> None:
        self._close_open_file(finalize=True)
        self._segment_index += 1
        self._bytes_in_segment = 0
        self._record_index_in_segment = 0
        self._ensure_open_segment()

    def _close_open_file(self, finalize: bool) -> None:
        if self._file is None:
            return

        handle = self._file
        part_path = self._part_path
        self._file = None
        self._part_path = None

        try:
            handle.flush()
            if self.fsync_on_close:
                os.fsync(handle.fileno())
            handle.close()
        except Exception as exc:
            self._failed = exc
            raise RawStreamWriteError("failed to close segment: %s" % exc) from exc

        if not finalize:
            return

        assert part_path is not None
        final_path = _segment_path(self.output_dir, self.spec, self._segment_index, False)
        if final_path.exists() and not self.overwrite:
            self._failed = RawStreamWriteError(
                "final segment already exists: %s" % final_path
            )
            raise self._failed
        try:
            if self.overwrite:
                os.replace(part_path, final_path)
            else:
                os.rename(part_path, final_path)
        except Exception as exc:
            self._failed = exc
            raise RawStreamWriteError(
                "failed to finalize segment %s: %s" % (part_path, exc)
            ) from exc


class RawStreamAsyncWriter:
    """Background-thread wrapper around :class:`RawStreamWriter`.

    ``submit`` enqueues records without silently dropping them. By default,
    payloads are copied before enqueueing because upstream camera buffers may be
    reused or mutated before the writer thread consumes them. If
    ``copy_payload=False``, the caller must guarantee the buffer remains
    immutable until the background writer has written it.

    Background writer errors are stored and re-raised by future ``submit`` calls
    and by ``close``.
    """

    def __init__(
        self,
        writer: RawStreamWriter,
        max_queue_size: int = 128,
        copy_payload: bool = True,
    ) -> None:
        if max_queue_size <= 0:
            raise RawStreamWriteError("max_queue_size must be positive")
        self.writer = writer
        self.copy_payload = copy_payload
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=max_queue_size)
        self._sentinel = object()
        self._error: Optional[BaseException] = None
        self._error_lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="RawStreamAsyncWriter",
            daemon=True,
        )
        self._thread.start()

    def __enter__(self) -> "RawStreamAsyncWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        try:
            self.close()
        except RawStreamError:
            if exc_type is None:
                raise
        return False

    def submit(
        self,
        payload: Union[bytes, bytearray, memoryview],
        block: bool = True,
        timeout: Optional[float] = None,
        **header_values: int,
    ) -> None:
        """Enqueue one record for background writing.

        The default blocking mode waits for queue space. With ``block=False``,
        a full queue raises :class:`RawStreamWriteError`; frames are never
        silently dropped.
        """

        if self._closed:
            raise RawStreamWriteError("async writer is closed")
        self._raise_background_error()

        queued_payload: Union[bytes, bytearray, memoryview]
        if self.copy_payload:
            queued_payload = bytes(_payload_memoryview(payload))
        else:
            queued_payload = payload

        item = (queued_payload, dict(header_values))
        try:
            self._queue.put(item, block=block, timeout=timeout)
        except queue.Full as exc:
            raise RawStreamWriteError("raw stream async queue is full") from exc
        self._raise_background_error()

    def close(self) -> None:
        """Drain queued records, stop the thread, close the wrapped writer."""

        if not self._closed:
            self._closed = True
            sentinel_sent = False
            while self._get_background_error() is None and self._thread.is_alive():
                try:
                    self._queue.put(self._sentinel, timeout=0.1)
                    sentinel_sent = True
                    break
                except queue.Full:
                    continue
            if not sentinel_sent and self._get_background_error() is None:
                self._set_background_error(
                    RawStreamWriteError("background writer stopped before close")
                )
            self._thread.join()

        close_error: Optional[BaseException] = None
        try:
            self.writer.close()
        except BaseException as exc:
            close_error = exc

        background_error = self._get_background_error()
        if background_error is not None:
            raise RawStreamWriteError(
                "background raw stream writer failed: %s" % background_error
            ) from background_error
        if close_error is not None:
            raise RawStreamWriteError(
                "raw stream writer close failed: %s" % close_error
            ) from close_error

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._sentinel:
                    return
                payload, header_values = item
                self.writer.append(payload, **header_values)
            except BaseException as exc:
                self._set_background_error(exc)
                return
            finally:
                self._queue.task_done()

    def _set_background_error(self, exc: BaseException) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = exc

    def _get_background_error(self) -> Optional[BaseException]:
        with self._error_lock:
            return self._error

    def _raise_background_error(self) -> None:
        error = self._get_background_error()
        if error is not None:
            raise RawStreamWriteError(
                "background raw stream writer failed: %s" % error
            ) from error


def iter_stream_records(
    input_dir: Union[Path, str],
    spec: RawStreamSpec,
    include_part_files: bool = True,
) -> Iterator[RawRecord]:
    """Iterate valid records from matching segment files in ``input_dir``.

    Finalized ``.raw`` files and, when requested, recoverable ``.raw.part``
    files are sorted by segment index. Incomplete final bytes are ignored and
    are never yielded as a record; use :func:`validate_stream` to report them.
    """

    paths = [
        info[2]
        for info in _find_segment_files(Path(input_dir), spec, include_part_files)
    ]
    return iter_records(paths, spec, include_part_files=include_part_files)


def iter_records(
    paths: Iterable[Union[Path, str]],
    spec: RawStreamSpec,
    include_part_files: bool = True,
) -> Iterator[RawRecord]:
    """Iterate valid records from explicit segment paths.

    The parser reads fixed-size records only. A trailing incomplete record after
    a crash is ignored by default; strict handling is available through
    :func:`validate_stream`.
    """

    path_infos = []
    for raw_path in paths:
        path = Path(raw_path)
        parsed = _parse_segment_filename(path.name, spec)
        if parsed is None:
            raise RawStreamParseError("not a segment file for spec: %s" % path)
        segment_index, is_part = parsed
        if is_part and not include_part_files:
            continue
        path_infos.append((segment_index, is_part, path))

    path_infos.sort(key=lambda item: (item[0], item[1], item[2].name))
    global_record_index = 0
    for segment_index, _is_part, path in path_infos:
        record_index_in_segment = 0
        with open(path, "rb") as handle:
            while True:
                byte_offset = handle.tell()
                header = handle.read(spec.header_nbytes)
                if not header:
                    break
                if len(header) < spec.header_nbytes:
                    break
                payload = handle.read(spec.payload_nbytes)
                if len(payload) < spec.payload_nbytes:
                    break
                yield RawRecord(
                    header=spec.unpack_header(header),
                    payload=payload,
                    source_path=path,
                    segment_index=segment_index,
                    record_index_in_segment=record_index_in_segment,
                    global_record_index=global_record_index,
                    byte_offset=byte_offset,
                )
                record_index_in_segment += 1
                global_record_index += 1


def load_sidecar_spec(input_dir: Union[Path, str], file_prefix: str) -> RawStreamSpec:
    """Load ``{file_prefix}.spec.json`` from ``input_dir``."""

    sidecar_path = Path(input_dir) / ("%s.spec.json" % file_prefix)
    try:
        with open(sidecar_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise RawStreamParseError("sidecar not found: %s" % sidecar_path) from exc
    except json.JSONDecodeError as exc:
        raise RawStreamParseError(
            "sidecar is not valid JSON: %s" % sidecar_path
        ) from exc
    return RawStreamSpec.from_dict(data)


def validate_stream(
    input_dir: Union[Path, str],
    spec: RawStreamSpec,
    include_part_files: bool = True,
    strict: bool = False,
) -> ValidationReport:
    """Validate a raw stream without loading payload bytes.

    Validation checks segment file sizes, incomplete tails, sidecar agreement
    with the explicitly supplied spec, and monotonic header fields where
    present. Incomplete tail bytes are warnings by default and errors in strict
    mode. If strict mode sees any error, :class:`RawStreamParseError` is raised.
    """

    directory = Path(input_dir)
    warnings: List[str] = []
    errors: List[str] = []
    incomplete_tail_bytes: Dict[str, int] = {}
    first_header: Optional[Dict[str, int]] = None
    last_header: Optional[Dict[str, int]] = None
    valid_record_count = 0

    sidecar_path = directory / ("%s.spec.json" % spec.file_prefix)
    sidecar_exists = sidecar_path.exists()
    sidecar_matches: Optional[bool] = None
    if sidecar_exists:
        try:
            sidecar_spec = load_sidecar_spec(directory, spec.file_prefix)
            sidecar_matches = sidecar_spec.to_dict() == spec.to_dict()
            if not sidecar_matches:
                errors.append(
                    "sidecar spec does not match explicitly supplied spec"
                )
        except RawStreamError as exc:
            sidecar_matches = False
            errors.append("failed to load sidecar: %s" % exc)
    else:
        sidecar_matches = None
        warnings.append("sidecar is missing: %s" % sidecar_path)

    segment_infos = _find_segment_files(directory, spec, include_part_files)
    previous_frame_index: Optional[int] = None
    previous_shot_id: Optional[int] = None

    for segment_index, _is_part, path in segment_infos:
        try:
            size = path.stat().st_size
        except OSError as exc:
            errors.append("failed to stat %s: %s" % (path, exc))
            continue

        tail = size % spec.record_nbytes
        incomplete_tail_bytes[path.name] = tail
        if tail:
            message = "%s has %d incomplete trailing bytes" % (path.name, tail)
            if strict:
                errors.append(message)
            else:
                warnings.append(message)

        complete_records = size // spec.record_nbytes
        try:
            with open(path, "rb") as handle:
                for _record_index in range(complete_records):
                    header_offset = handle.tell()
                    header_bytes = handle.read(spec.header_nbytes)
                    if len(header_bytes) != spec.header_nbytes:
                        errors.append(
                            "%s has incomplete header at byte offset %d"
                            % (path.name, header_offset)
                        )
                        break
                    try:
                        header = spec.unpack_header(header_bytes)
                    except RawStreamParseError as exc:
                        errors.append("%s: %s" % (path.name, exc))
                        break

                    if first_header is None:
                        first_header = dict(header)
                    last_header = dict(header)
                    valid_record_count += 1

                    if "frame_index" in header:
                        frame_index = header["frame_index"]
                        if (
                            previous_frame_index is not None
                            and frame_index <= previous_frame_index
                        ):
                            errors.append(
                                "frame_index is not monotonically increasing "
                                "at %s segment %d: %d after %d"
                                % (
                                    path.name,
                                    segment_index,
                                    frame_index,
                                    previous_frame_index,
                                )
                            )
                        previous_frame_index = frame_index

                    if "shot_id" in header:
                        shot_id = header["shot_id"]
                        if (
                            previous_shot_id is not None
                            and shot_id < previous_shot_id
                        ):
                            warnings.append(
                                "shot_id decreased at %s segment %d: %d after %d"
                                % (
                                    path.name,
                                    segment_index,
                                    shot_id,
                                    previous_shot_id,
                                )
                            )
                        previous_shot_id = shot_id

                    handle.seek(spec.payload_nbytes, os.SEEK_CUR)
        except OSError as exc:
            errors.append("failed to validate %s: %s" % (path, exc))

    report = ValidationReport(
        segment_file_count=len(segment_infos),
        valid_record_count=valid_record_count,
        incomplete_tail_bytes=incomplete_tail_bytes,
        sidecar_exists=sidecar_exists,
        sidecar_matches=sidecar_matches,
        first_header=first_header,
        last_header=last_header,
        warnings=warnings,
        errors=errors,
    )

    if strict and report.errors:
        raise RawStreamParseError(
            "raw stream validation failed: " + "; ".join(report.errors)
        )
    return report


def _require_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise RawStreamSpecError("%s must be an int" % name)
    return value


def _require_non_empty_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RawStreamSpecError("%s must be a non-empty string" % name)
    return value


def _validate_int_for_type(value: Any, name: str, field_type: str) -> None:
    if type(value) is not int:
        raise RawStreamWriteError("header field %r must be an int" % name)
    _fmt, minimum, maximum = _TYPE_FORMATS[field_type]
    if value < minimum or value > maximum:
        raise RawStreamWriteError(
            "header field %r=%r does not fit %s" % (name, value, field_type)
        )


def _payload_memoryview(
    payload: Union[bytes, bytearray, memoryview]
) -> memoryview:
    try:
        view = memoryview(payload)
    except TypeError as exc:
        raise RawStreamWriteError("payload must be bytes-like") from exc
    if not view.contiguous:
        raise RawStreamWriteError("payload memoryview must be contiguous")
    return view.cast("B")


def _segment_path(
    directory: Path,
    spec: RawStreamSpec,
    segment_index: int,
    part: bool,
) -> Path:
    name = "%s_%06d%s" % (
        spec.file_prefix,
        segment_index,
        spec.file_extension,
    )
    if part:
        name += ".part"
    return directory / name


def _parse_segment_filename(
    name: str,
    spec: RawStreamSpec,
) -> Optional[Tuple[int, bool]]:
    prefix = "%s_" % spec.file_prefix
    if not name.startswith(prefix):
        return None

    tail = name[len(prefix) :]
    is_part = False
    part_suffix = "%s.part" % spec.file_extension
    final_suffix = spec.file_extension
    if tail.endswith(part_suffix):
        index_text = tail[: -len(part_suffix)]
        is_part = True
    elif tail.endswith(final_suffix):
        index_text = tail[: -len(final_suffix)]
    else:
        return None

    if len(index_text) != 6 or not index_text.isdigit():
        return None
    return int(index_text), is_part


def _find_segment_files(
    directory: Path,
    spec: RawStreamSpec,
    include_part_files: bool,
) -> List[Tuple[int, bool, Path]]:
    infos: List[Tuple[int, bool, Path]] = []
    if not directory.exists():
        return infos
    for path in directory.iterdir():
        if not path.is_file():
            continue
        parsed = _parse_segment_filename(path.name, spec)
        if parsed is None:
            continue
        segment_index, is_part = parsed
        if is_part and not include_part_files:
            continue
        infos.append((segment_index, is_part, path))
    infos.sort(key=lambda item: (item[0], item[1], item[2].name))
    return infos
