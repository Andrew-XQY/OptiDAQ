"""Sample magnet currents in a background thread and append a JSONL sidecar.

Each row contains t_ns from time.time_ns() and the mapping returned by read_fn.
Compare t_ns with frame local_receive_ns for offline association. Read/write
errors are recorded on the logger; they are not raised to the acquisition loop.
The callback has no dependency on CERN-specific types.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union


class MagnetCurrentLogger:
    """Sample magnet currents on a timer and append them to a JSONL sidecar.

    Parameters
    ----------
    path:
        Destination ``magnets.samples.jsonl`` file (one JSON object per line).
    read_fn:
        Zero-arg callable returning a mapping ``{alias: {...}}`` for one sample.
        Typically ``ClearMagnetSet.read_sample``.
    period_s:
        Seconds between samples. Should be a few times the camera frame period so
        every frame has a close-in-time reading.
    on_problem:
        Optional callback invoked with a short message string on a read/write
        failure. Purely informational; the logger keeps running.
    """

    def __init__(
        self,
        path: Union[Path, str],
        read_fn: Callable[[], Mapping[str, Any]],
        *,
        period_s: float,
        on_problem: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.path = Path(path)
        self._read_fn = read_fn
        self.period_s = float(period_s)
        if self.period_s <= 0:
            raise ValueError("magnet logger period_s must be positive")
        self._on_problem = on_problem

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handle: Optional[Any] = None

        self.sample_count = 0
        self.error: Optional[BaseException] = None

    def start(self) -> None:
        """Open the sidecar and start the sampling thread."""

        if self._thread is not None:
            raise RuntimeError("magnet logger already started")
        # Open eagerly so a bad path fails fast (before the scan), not silently
        # mid-run. Opening is the one operation allowed to raise to the caller.
        self._handle = open(self.path, "w", encoding="utf-8")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="MagnetCurrentLogger",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                t_ns = time.time_ns()
                try:
                    magnets = self._read_fn()
                    line = json.dumps(
                        {"t_ns": int(t_ns), "magnets": magnets},
                        sort_keys=True,
                        default=str,
                    )
                    self._handle.write(line)
                    self._handle.write("\n")
                    self._handle.flush()
                    self.sample_count += 1
                except BaseException as exc:  # never propagate into acquisition
                    self.error = exc
                    self._report("magnet sample failed: %s" % exc)
                self._stop_event.wait(self.period_s)
        except BaseException as exc:  # pragma: no cover - thread-fatal guard
            self.error = exc
            self._report("magnet logger thread stopped: %s" % exc)

    def stop(self, timeout_s: float = 5.0) -> None:
        """Stop sampling, join the thread, and close the sidecar."""

        self._stop_event.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=max(0.0, float(timeout_s)))
        handle = self._handle
        self._handle = None
        if handle is not None:
            try:
                handle.flush()
                handle.close()
            except BaseException as exc:
                if self.error is None:
                    self.error = exc

    def _report(self, message: str) -> None:
        if self._on_problem is None:
            return
        try:
            self._on_problem(message)
        except BaseException:
            pass
