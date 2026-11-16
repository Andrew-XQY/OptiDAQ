"""Frame, camera-tap and storage contracts for the DAQ pipeline.

Adapters copy mutable camera buffers to bytes before emitting a Frame.
Save backends persist each camera stream independently; matching is offline.
The raw backend assigns frame_index and stores the source counter, timestamp
and payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Protocol


NO_PULSE_ID = (1 << 64) - 1


@dataclass(frozen=True)
class Frame:
    """One immutable upstream image fact.

    ``device`` identifies the source stream. ``source_image_number`` is the
    upstream camera/source counter. ``acq_stamp_ns`` is the acquisition timestamp
    later used by offline code for clustering. The upstream pipeline does not
    add ``shot_id`` or perform online pairing.
    """

    device: str
    source_image_number: int
    acq_stamp_ns: int
    payload_bytes: bytes
    pulse_id: int = NO_PULSE_ID

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty str")
        if type(self.source_image_number) is not int or self.source_image_number < 0:
            raise ValueError("source_image_number must be an int >= 0")
        if type(self.acq_stamp_ns) is not int or self.acq_stamp_ns < 0:
            raise ValueError("acq_stamp_ns must be an int >= 0")
        if type(self.payload_bytes) is not bytes:
            raise ValueError("payload_bytes must be immutable bytes")
        if type(self.pulse_id) is not int or not (0 <= self.pulse_id <= NO_PULSE_ID):
            raise ValueError("pulse_id must be an int in [0, NO_PULSE_ID]")


class CameraTap(Protocol):
    """Source adapter that emits :class:`Frame` objects into the pipeline."""

    device: str

    def start(self, emit: Callable[[Frame], None]) -> None:
        """Start producing frames through ``emit``."""

    def stop(self) -> None:
        """Stop producing frames."""


class SaveBackend(Protocol):
    """Per-device save target used by the upstream pipeline."""

    device: str

    def open(self) -> None:
        """Prepare the backend before frames are routed."""

    def save(self, frame: Frame) -> None:
        """Persist one frame fact for this backend's device."""

    def close(self) -> None:
        """Release backend resources."""


@dataclass
class Problem:
    """Health or lifecycle problem reported by the pipeline/watchdog."""

    source: str
    message: str
    severity: str = "warning"
    exception: Optional[BaseException] = None


@dataclass
class PipelineStats:
    """Thread-safe snapshot of upstream pipeline counters and state."""

    accepted_count: int
    saved_count: int
    queue_size: int
    accepting: bool
    running: bool
    per_device_accepted: Dict[str, int]
    per_device_saved: Dict[str, int]
    latest_source_image_number: Dict[str, int]
    latest_acq_stamp_ns: Dict[str, int]
    error: Optional[str]
