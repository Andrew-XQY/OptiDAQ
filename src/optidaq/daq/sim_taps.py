"""Deterministic fake camera/source taps for integration tests."""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Callable, Optional

from optidaq.daq.contracts import Frame


class FakeTap:
    """CameraTap implementation that emits deterministic fake frames."""

    def __init__(
        self,
        device: str,
        n_frames: int,
        payload_nbytes: int,
        *,
        base_stamp_ns: int = 1_000_000_000,
        period_ns: int = 100_000_000,
        payload_seed: int = 0,
        sleep_s: float = 0.0,
        start_image_number: int = 0,
    ) -> None:
        if not isinstance(device, str) or not device:
            raise ValueError("device must be a non-empty str")
        if n_frames < 0:
            raise ValueError("n_frames must be non-negative")
        if payload_nbytes <= 0:
            raise ValueError("payload_nbytes must be positive")
        if base_stamp_ns < 0 or period_ns < 0:
            raise ValueError("timestamps must be non-negative")
        if sleep_s < 0:
            raise ValueError("sleep_s must be non-negative")
        if start_image_number < 0:
            raise ValueError("start_image_number must be non-negative")

        self.device = device
        self.n_frames = n_frames
        self.payload_nbytes = payload_nbytes
        self.base_stamp_ns = base_stamp_ns
        self.period_ns = period_ns
        self.payload_seed = payload_seed
        self.sleep_s = sleep_s
        self.start_image_number = start_image_number
        self.error: Optional[BaseException] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self, emit: Callable[[Frame], None]) -> None:
        """Start emitting frames from one background thread."""

        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("fake tap is already running")
        self.error = None
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(emit,),
            name="FakeTap:%s" % self.device,
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Ask the fake tap to stop emitting."""

        self._stop_event.set()

    def join(self, timeout: Optional[float] = None) -> None:
        """Wait for the fake emitter thread to finish."""

        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self, emit: Callable[[Frame], None]) -> None:
        try:
            for i in range(self.n_frames):
                if self._stop_event.is_set():
                    break
                source_image_number = self.start_image_number + i
                emit(
                    Frame(
                        device=self.device,
                        source_image_number=source_image_number,
                        acq_stamp_ns=self.base_stamp_ns + i * self.period_ns,
                        payload_bytes=expected_fake_payload(
                            self.device,
                            source_image_number,
                            self.payload_nbytes,
                            self.payload_seed,
                        ),
                    )
                )
                if self.sleep_s:
                    time.sleep(self.sleep_s)
        except BaseException as exc:
            self.error = exc


def expected_fake_payload(
    device: str,
    source_image_number: int,
    payload_nbytes: int,
    payload_seed: int = 0,
) -> bytes:
    """Return deterministic fake payload bytes for a device/source counter."""

    if payload_nbytes <= 0:
        raise ValueError("payload_nbytes must be positive")
    seed = ("%s:%d:%d:%d" % (
        device,
        source_image_number,
        payload_nbytes,
        payload_seed,
    )).encode("utf-8")
    chunks = []
    counter = 0
    total = 0
    while total < payload_nbytes:
        digest = hashlib.sha256(seed + b":" + str(counter).encode("ascii")).digest()
        chunks.append(digest)
        total += len(digest)
        counter += 1
    return b"".join(chunks)[:payload_nbytes]
