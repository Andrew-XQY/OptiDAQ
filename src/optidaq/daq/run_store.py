"""Create run folders and persist acquisition metadata."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from optidaq.daq.raw_stream import RawStreamSpec


def create_run_dir(root: Union[Path, str], run_id: Optional[str] = None) -> Path:
    """Create a fresh run directory and refuse to reuse existing folders."""

    root_path = Path(root)
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty str")
    if "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must not contain path separators")
    run_dir = root_path / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_resolved_config(
    run_dir: Union[Path, str],
    config: Mapping[str, Any],
) -> Path:
    """Write ``resolved_config.json`` in ``run_dir`` without overwriting."""

    config_path = Path(run_dir) / "resolved_config.json"
    # Never let an unexpected, non-JSON value (e.g. a FESA device object) abort the run:
    # values should already be coerced by jsonable(), but anything that slips through is
    # written as its text form instead of raising.
    text = json.dumps(dict(config), indent=2, sort_keys=True, default=str)
    with open(config_path, "x", encoding="utf-8") as handle:
        handle.write(text)
    return config_path


def build_raw_spec(
    *,
    file_prefix: str,
    payload_nbytes: int,
    segment_max_bytes: int,
    endianness: str = "little",
    schema_version: int = 2,
) -> RawStreamSpec:
    """Build a versioned raw stream specification for one camera."""

    if schema_version == 1:
        header_fields = [
            {"name": "frame_index", "type": "uint64"},
            {"name": "source_image_number", "type": "uint64"},
            {"name": "acq_stamp_ns", "type": "uint64"},
        ]
    elif schema_version == 2:
        header_fields = [
            {"name": "frame_index", "type": "uint64"},
            {"name": "source_image_number", "type": "uint64"},
            {"name": "acq_stamp_ns", "type": "uint64"},
            {"name": "pulse_id", "type": "uint64"},
        ]
    else:
        raise ValueError("schema_version must be 1 or 2")

    return RawStreamSpec.from_dict(
        {
            "schema_version": schema_version,
            "endianness": endianness,
            "header_fields": header_fields,
            "payload_nbytes": payload_nbytes,
            "segment_max_bytes": segment_max_bytes,
            "file_prefix": file_prefix,
            "file_extension": ".raw",
        }
    )
