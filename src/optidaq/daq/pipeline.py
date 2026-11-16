"""Bounded-queue upstream DAQ data plane.

``DAQPipeline`` connects camera/source taps to per-device save backends:

    CameraTap(s) -> Frame -> bounded queue -> router thread -> SaveBackend

The pipeline stores facts only. It does not pair frames, merge images, gate
acquisition, create ``shot_id`` values, inspect payload contents, or understand
camera/save-format internals. Payload modification happens only if the caller
explicitly injects a transform.
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Sequence
import queue
import threading
import time

from optidaq.daq.contracts import (
    CameraTap,
    Frame,
    PipelineStats,
    Problem,
    SaveBackend,
)


class DAQPipelineError(RuntimeError):
    """Raised when the upstream DAQ pipeline cannot continue safely."""


class DAQPipeline:
    """Route immutable ``Frame`` facts from taps to save backends.

    Lifecycle contract:
    - ``start`` opens backends, starts the router thread, and starts taps.
    - ``submit`` blocks by default when the bounded queue is full and never
      silently drops a frame.
    - ``stop`` stops accepting, stops taps, drains queued frames up to a bounded
      timeout, joins the router, closes all backends, and then raises any stored
      lifecycle/router/backend error.
    - Any router/save error poisons the pipeline; future calls surface it.
    """

    def __init__(
        self,
        taps: Sequence[CameraTap],
        backends: Mapping[str, SaveBackend],
        *,
        queue_maxsize: int = 128,
        transform: Optional[Callable[[Frame], Frame]] = None,
        on_problem: Optional[Callable[[Problem], None]] = None,
    ) -> None:
        if queue_maxsize <= 0:
            raise ValueError("queue_maxsize must be positive")

        self._taps = list(taps)
        self._backends: Dict[str, SaveBackend] = dict(backends)
        self._queue: "queue.Queue[Frame]" = queue.Queue(maxsize=queue_maxsize)
        self._transform = transform if transform is not None else _identity
        self._on_problem = on_problem

        self._lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._router_thread: Optional[threading.Thread] = None
        self._started_taps: list[CameraTap] = []
        self._opened_backends: list[SaveBackend] = []

        self._accepting = False
        self._running = False
        self._closed = False
        self._error: Optional[BaseException] = None
        self._accepted_count = 0
        self._saved_count = 0
        self._per_device_accepted: Dict[str, int] = {}
        self._per_device_saved: Dict[str, int] = {}
        self._latest_source_image_number: Dict[str, int] = {}
        self._latest_acq_stamp_ns: Dict[str, int] = {}

    def __enter__(self) -> "DAQPipeline":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        try:
            self.stop()
        except DAQPipelineError:
            if exc_type is None:
                raise
        return False

    def start(self) -> None:
        """Open backends, start the router, and start all taps."""

        with self._lock:
            if self._running:
                raise DAQPipelineError("pipeline is already running")
            if self._closed:
                raise DAQPipelineError("pipeline is closed")
            self._validate_routes_locked()
            self._stop_requested.clear()

        try:
            for backend in self._backends.values():
                backend.open()
                self._opened_backends.append(backend)

            with self._lock:
                self._running = True
                self._accepting = True

            self._router_thread = threading.Thread(
                target=self._router_loop,
                name="DAQPipelineRouter",
                daemon=True,
            )
            self._router_thread.start()

            for tap in self._taps:
                tap.start(self.submit)
                self._started_taps.append(tap)
        except BaseException as exc:
            self._record_problem(
                Problem(
                    source="pipeline.start",
                    message="failed to start pipeline: %s" % exc,
                    severity="error",
                    exception=exc,
                )
            )
            self._set_error(exc)
            self._accepting_off()
            self._stop_requested.set()
            self._stop_started_taps()
            self._join_router(timeout=1.0)
            self._close_opened_backends()
            raise DAQPipelineError("failed to start pipeline: %s" % exc) from exc

    def submit(
        self,
        frame: Frame,
        block: bool = True,
        timeout: Optional[float] = None,
    ) -> None:
        """Submit one frame into the bounded queue.

        The call blocks by default when the queue is full. With ``block=False``
        or an expired timeout, a clear error is raised; frames are never silently
        dropped.
        """

        if not isinstance(frame, Frame):
            raise DAQPipelineError("submit expects a Frame")

        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            self.raise_if_failed()
            with self._lock:
                if not self._accepting:
                    raise DAQPipelineError("pipeline is not accepting frames")

            put_timeout = None
            if block:
                put_timeout = 0.1
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        put_timeout = 0
                    else:
                        put_timeout = min(0.1, remaining)

            try:
                self._queue.put(frame, block=block, timeout=put_timeout)
                break
            except queue.Full as exc:
                if not block:
                    raise DAQPipelineError("pipeline queue is full") from exc
                if deadline is not None and time.monotonic() >= deadline:
                    raise DAQPipelineError(
                        "pipeline queue is full (timeout)"
                    ) from exc
                continue

        with self._lock:
            self._accepted_count += 1
            self._per_device_accepted[frame.device] = (
                self._per_device_accepted.get(frame.device, 0) + 1
            )
            self._latest_source_image_number[frame.device] = (
                frame.source_image_number
            )
            self._latest_acq_stamp_ns[frame.device] = frame.acq_stamp_ns

    def stop(self, drain_timeout_s: float = 30.0) -> None:
        """Stop taps, drain queued frames, close backends, and surface errors."""

        if drain_timeout_s < 0:
            raise ValueError("drain_timeout_s must be non-negative")

        self._accepting_off()
        self._stop_started_taps()
        self._stop_requested.set()

        remaining = self._drain_until(deadline=time.monotonic() + drain_timeout_s)
        if remaining:
            self._record_problem(
                Problem(
                    source="pipeline.stop",
                    message="%d queued frame(s) remained after drain timeout"
                    % remaining,
                    severity="error",
                )
            )

        self._join_router(timeout=drain_timeout_s)
        close_error = self._close_opened_backends()

        with self._lock:
            self._running = False
            self._closed = True
            stored_error = self._error

        if stored_error is not None:
            raise DAQPipelineError("pipeline failed: %s" % stored_error) from stored_error
        if close_error is not None:
            raise DAQPipelineError("pipeline close failed: %s" % close_error) from close_error
        if remaining:
            raise DAQPipelineError(
                "%d queued frame(s) remained after drain timeout" % remaining
            )

    def close(self) -> None:
        """Alias for ``stop`` for context-manager style cleanup."""

        self.stop()

    def stats(self) -> PipelineStats:
        """Return a thread-safe snapshot of counters and lifecycle state."""

        with self._lock:
            return PipelineStats(
                accepted_count=self._accepted_count,
                saved_count=self._saved_count,
                queue_size=self._queue.qsize(),
                accepting=self._accepting,
                running=self._running,
                per_device_accepted=dict(self._per_device_accepted),
                per_device_saved=dict(self._per_device_saved),
                latest_source_image_number=dict(
                    self._latest_source_image_number
                ),
                latest_acq_stamp_ns=dict(self._latest_acq_stamp_ns),
                error=str(self._error) if self._error is not None else None,
            )

    def raise_if_failed(self) -> None:
        """Raise the stored pipeline error if the router/backend failed."""

        with self._lock:
            error = self._error
        if error is not None:
            raise DAQPipelineError("pipeline failed: %s" % error) from error

    def _router_loop(self) -> None:
        while True:
            if self._stop_requested.is_set() and self._queue.empty():
                break
            try:
                frame = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                routed = self._transform(frame)
                if not isinstance(routed, Frame):
                    raise DAQPipelineError("transform must return a Frame")
                backend = self._backends.get(routed.device)
                if backend is None:
                    raise DAQPipelineError(
                        "no save backend for device %r" % routed.device
                    )
                backend.save(routed)
                with self._lock:
                    self._saved_count += 1
                    self._per_device_saved[routed.device] = (
                        self._per_device_saved.get(routed.device, 0) + 1
                    )
            except BaseException as exc:
                self._set_error(exc)
                self._accepting_off()
                self._record_problem(
                    Problem(
                        source="pipeline.router",
                        message="router failed: %s" % exc,
                        severity="error",
                        exception=exc,
                    )
                )
                self._queue.task_done()
                break
            else:
                self._queue.task_done()

        with self._lock:
            self._running = False

    def _validate_routes_locked(self) -> None:
        seen = set()
        for tap in self._taps:
            device = getattr(tap, "device", None)
            if not isinstance(device, str) or not device:
                raise DAQPipelineError("tap device must be a non-empty str")
            if device in seen:
                raise DAQPipelineError("duplicate tap device %r" % device)
            seen.add(device)
            if device not in self._backends:
                raise DAQPipelineError("missing backend for tap device %r" % device)
        for key, backend in self._backends.items():
            device = getattr(backend, "device", None)
            if device != key:
                raise DAQPipelineError(
                    "backend mapping key %r does not match backend.device %r"
                    % (key, device)
                )

    def _accepting_off(self) -> None:
        with self._lock:
            self._accepting = False

    def _set_error(self, exc: BaseException) -> None:
        with self._lock:
            if self._error is None:
                self._error = exc

    def _stop_started_taps(self) -> None:
        for tap in list(self._started_taps):
            try:
                tap.stop()
            except BaseException as exc:
                self._set_error(exc)
                self._record_problem(
                    Problem(
                        source="pipeline.stop",
                        message="tap %r stop failed: %s"
                        % (getattr(tap, "device", "?"), exc),
                        severity="error",
                        exception=exc,
                    )
                )
        self._started_taps = []

    def _drain_until(self, deadline: float) -> int:
        while time.monotonic() < deadline:
            with self._lock:
                error = self._error
            if error is not None:
                break
            if self._queue.empty():
                break
            time.sleep(0.01)
        return self._queue.qsize()

    def _join_router(self, timeout: float) -> None:
        thread = self._router_thread
        if thread is None:
            return
        thread.join(timeout=max(timeout, 0.0))
        if thread.is_alive():
            exc = DAQPipelineError("router thread did not stop")
            self._set_error(exc)
            self._record_problem(
                Problem(
                    source="pipeline.stop",
                    message=str(exc),
                    severity="error",
                    exception=exc,
                )
            )

    def _close_opened_backends(self) -> Optional[BaseException]:
        first_error: Optional[BaseException] = None
        for backend in list(self._opened_backends):
            try:
                backend.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                self._record_problem(
                    Problem(
                        source="pipeline.close",
                        message="backend %r close failed: %s"
                        % (getattr(backend, "device", "?"), exc),
                        severity="error",
                        exception=exc,
                    )
                )
        self._opened_backends = []
        return first_error

    def _record_problem(self, problem: Problem) -> None:
        if self._on_problem is None:
            return
        try:
            self._on_problem(problem)
        except BaseException:
            pass


def _identity(frame: Frame) -> Frame:
    return frame
