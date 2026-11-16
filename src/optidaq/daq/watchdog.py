"""Health-plane helpers for upstream DAQ."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence
import threading
import time

from optidaq.daq.contracts import PipelineStats, Problem


class CounterFreezeWatchdog:
    """Report devices whose source image counter stops advancing.

    The watchdog observes ``PipelineStats.latest_source_image_number`` only. It
    does not close writers, stop taps, touch devices, inspect frames, perform
    pairing, or handle router errors already owned by the pipeline.
    """

    def __init__(
        self,
        get_stats: Callable[[], PipelineStats],
        devices: Sequence[str],
        on_problem: Callable[[Problem], None],
        *,
        freeze_timeout_s: float,
        poll_interval_s: float = 1.0,
    ) -> None:
        if freeze_timeout_s <= 0:
            raise ValueError("freeze_timeout_s must be positive")
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")

        self._get_stats = get_stats
        self._devices = tuple(devices)
        self._on_problem = on_problem
        self._freeze_timeout_s = freeze_timeout_s
        self._poll_interval_s = poll_interval_s
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_value: Dict[str, Optional[int]] = {}
        self._last_change: Dict[str, float] = {}
        self._reported: Dict[str, bool] = {}

    def start(self) -> None:
        """Start the background watchdog thread."""

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            now = time.monotonic()
            for device in self._devices:
                self._last_value.setdefault(device, None)
                self._last_change.setdefault(device, now)
                self._reported.setdefault(device, False)
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="CounterFreezeWatchdog",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop the background watchdog thread."""

        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join()

    def check_once(self, now: Optional[float] = None) -> None:
        """Run one freeze check, useful for demos and tests."""

        if now is None:
            now = time.monotonic()
        stats = self._get_stats()
        latest = stats.latest_source_image_number
        problems = []

        with self._lock:
            for device in self._devices:
                value = latest.get(device)
                if device not in self._last_change:
                    self._last_change[device] = now
                    self._last_value[device] = value
                    self._reported[device] = False
                    continue

                previous = self._last_value.get(device)
                if value != previous:
                    self._last_value[device] = value
                    self._last_change[device] = now
                    self._reported[device] = False
                    continue

                elapsed = now - self._last_change[device]
                if elapsed >= self._freeze_timeout_s and not self._reported[device]:
                    self._reported[device] = True
                    if value is None:
                        message = (
                            "device %r has not produced a source image counter"
                            " for %.3f s"
                        ) % (device, elapsed)
                    else:
                        message = (
                            "device %r source image counter froze at %d"
                            " for %.3f s"
                        ) % (device, value, elapsed)
                    problems.append(
                        Problem(
                            source="watchdog.counter_freeze",
                            message=message,
                            severity="warning",
                        )
                    )

        for problem in problems:
            self._on_problem(problem)

    def _run(self) -> None:
        while not self._stop_event.wait(self._poll_interval_s):
            self.check_once()
