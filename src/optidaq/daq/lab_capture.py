"""Select one fresh pair after a lab actuator settles; cameras stay on Line1.

This is a save gate, not exposure gating. The FESA image-ready timestamp is a
timing proxy, not a shared trigger ID or a measurement of exposure simultaneity.
Frames outside an active request are deliberately discarded, including during
long motor moves. The active raw pipeline does not use this module.
"""
from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass
import math
import threading
import time

from optidaq.daq.contracts import Frame


QUALITY = "software_selected_pair; exposure_simultaneity_not_measured"


def integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError("%s must be a nonnegative integer" % name)
    return value


def validate_capture_config(cfg):
    capture = cfg["capture"]
    if len(cfg["cameras"]) != 2 or len(set(cfg["cameras"])) != 2:
        raise ValueError("lab point capture requires two distinct cameras")
    if cfg["tap"].get("mode") != "subscribe" or cfg["tap"].get("subscribe_selector") != "SPS.USER.ALL":
        raise ValueError("lab point capture requires subscribe mode with SPS.USER.ALL")
    if cfg["scan"].get("frames_per_point") != 1:
        raise ValueError("lab motor raster requires one paired sample per position")
    if cfg.get("postprocess", {}).get("actions"):
        raise ValueError("motor raster uses capture-ID finalization; DMD-only analysis actions must be empty")
    for key in ("guard_s", "pair_tolerance_ms", "max_frame_age_s"):
        value = capture.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("capture.%s must be finite and positive" % key)
    size = capture.get("queue_size", 16)
    if type(size) is not int or size < 2:
        raise ValueError("capture.queue_size must be an integer >= 2")
    timeout = cfg["scan"]["point_timeout_s"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= capture["guard_s"]:
        raise ValueError("point_timeout_s must exceed capture.guard_s")


def validate_exposure_guard(capture_config, settings):
    """The empirical guard must at least cover the longest applied exposure."""
    longest_s = max(float(value["AcquiredParameters"]["acqCameraExposureTimeUs"])
                    for value in settings.values()) / 1e6
    if capture_config["guard_s"] < longest_s:
        raise ValueError("capture.guard_s must cover the longest applied camera exposure")


@dataclass(frozen=True)
class CaptureRequest:
    step_id: int
    host_ns: int
    mono_ns: int
    baseline: dict
    guard_ns: int
    actuator: dict

    def metadata(self):
        return {
            "capture_id": "capture_%06d" % self.step_id, "step_id": self.step_id,
            "barrier_host_time_ns": self.host_ns, "barrier_monotonic_ns": self.mono_ns,
            "baseline_image_numbers": dict(self.baseline), "guard_ns": self.guard_ns,
            "actuator": deepcopy(self.actuator), "quality": QUALITY,
            "timing_method": "imageTimeStamp_plus_acqImageTransmissionTime",
        }


def metadata(frame):
    values = getattr(frame, "sync_metadata", None)
    if not isinstance(values, dict):
        raise ValueError("camera frame has no timing metadata")
    return values


def proxy_ns(frame):
    values = metadata(frame)
    stamp = integer(values.get("imageTimeStamp"), "imageTimeStamp")
    transmission = integer(values.get("acqImageTransmissionTime"), "acqImageTransmissionTime")
    proxy = stamp + transmission
    if values.get("image_ready_ns") != proxy:
        raise ValueError("image-ready proxy disagrees with source fields")
    return proxy


class FreshPairGate:
    def __init__(self, aliases, config, *, clock=time):
        self.aliases = tuple(aliases)
        if len(self.aliases) != 2 or len(set(self.aliases)) != 2:
            raise ValueError("exactly two distinct camera aliases required")
        self.clock = clock
        self.guard_ns = round(config["guard_s"] * 1e9)
        self.tolerance_ns = round(config["pair_tolerance_ms"] * 1e6)
        self.max_age_ns = round(config["max_frame_age_s"] * 1e9)
        self.queue_size = config.get("queue_size", 16)
        self.condition = threading.Condition()
        self.queues = {cam: deque() for cam in self.aliases}
        self.latest = {}
        self.request = None
        self.error = None
        self.received, self.discarded = Counter(), Counter()

    def receive(self, frame):
        with self.condition:
            if self.error is not None:
                return
            try:
                cam = frame.device
                if cam not in self.queues:
                    raise ValueError("unexpected camera %s" % cam)
                data = metadata(frame)
                if data.get("triggerMode") != 3:
                    raise RuntimeError("%s changed from external Line1 triggering" % cam)
                for key in ("local_receive_ns", "local_receive_monotonic_ns", "frameNumber", "imageTimeStampRaw"):
                    integer(data.get(key), key)
                proxy_ns(frame)
                previous = self.latest.get(cam)
                if previous is not None:
                    old = metadata(previous)
                    if frame.source_image_number <= previous.source_image_number:
                        raise RuntimeError("%s source counter reset/duplicate; restart the run" % cam)
                    for key in ("frameNumber", "imageTimeStamp", "imageTimeStampRaw", "image_ready_ns",
                                "local_receive_ns", "local_receive_monotonic_ns"):
                        if data[key] <= old[key]:
                            raise RuntimeError("%s %s regressed; restart the run" % (cam, key))
                    wall_delta = data["local_receive_ns"] - old["local_receive_ns"]
                    mono_delta = data["local_receive_monotonic_ns"] - old["local_receive_monotonic_ns"]
                    if abs(wall_delta - mono_delta) > 100_000_000:
                        raise RuntimeError("host clock jumped; restart the run")
                self.latest[cam] = frame
                self.received[cam] += 1
                if self.request is None:
                    self.discarded["outside_capture"] += 1
                elif len(self.queues[cam]) >= self.queue_size:
                    raise RuntimeError("%s active capture queue overflow" % cam)
                else:
                    self.queues[cam].append(frame)
            except Exception as exc:
                self.error = exc
            self.condition.notify_all()

    def raise_if_failed(self):
        if self.error is not None:
            raise RuntimeError("camera capture failed: %s" % self.error) from self.error

    def wait_initial(self, timeout_s, check):
        deadline = self.clock.monotonic() + timeout_s
        with self.condition:
            while len(self.latest) != len(self.aliases):
                check()
                self.raise_if_failed()
                if self.clock.monotonic() >= deadline:
                    raise TimeoutError("no fresh initial frame from both cameras")
                self.condition.wait(0.1)

    def begin(self, step_id, actuator):
        with self.condition:
            self.raise_if_failed()
            if self.request is not None or len(self.latest) != len(self.aliases):
                raise RuntimeError("capture request requires idle gate and both cameras")
            for queue in self.queues.values():
                queue.clear()
            self.request = CaptureRequest(
                step_id, self.clock.time_ns(), self.clock.monotonic_ns(),
                {cam: self.latest[cam].source_image_number for cam in self.aliases},
                self.guard_ns, deepcopy(actuator),
            )
            return self.request

    def _reason(self, frame, now):
        data, request = metadata(frame), self.request
        if data["local_receive_monotonic_ns"] <= request.mono_ns:
            return "received_before_ready"
        if now - data["local_receive_monotonic_ns"] > self.max_age_ns:
            return "stale_receipt"
        if data["local_receive_monotonic_ns"] <= request.mono_ns + request.guard_ns:
            return "inside_host_transition_guard"
        if frame.source_image_number <= request.baseline[frame.device]:
            return "old_counter"
        if data["imageTimeStamp"] <= request.host_ns:
            return "reported_start_before_ready"
        if proxy_ns(frame) <= request.host_ns + request.guard_ns:
            return "inside_transition_guard"
        return None

    def _select(self, now):
        while True:
            for cam in self.aliases:
                while self.queues[cam]:
                    reason = self._reason(self.queues[cam][0], now)
                    if reason is None:
                        break
                    self.queues[cam].popleft()
                    self.discarded[reason] += 1
            if not all(self.queues.values()):
                return None
            a, b = self.aliases
            delta = proxy_ns(self.queues[b][0]) - proxy_ns(self.queues[a][0])
            if abs(delta) <= self.tolerance_ns:
                pair = {cam: self.queues[cam].popleft() for cam in self.aliases}
                self.end()
                return pair
            self.queues[a if delta > 0 else b].popleft()
            self.discarded["unmatched_timing"] += 1

    def wait_pair(self, timeout_s, check):
        if self.request is None:
            raise RuntimeError("begin capture before waiting")
        deadline = self.request.mono_ns / 1e9 + timeout_s
        try:
            with self.condition:
                while True:
                    check()
                    self.raise_if_failed()
                    now = self.clock.monotonic_ns()
                    if now / 1e9 >= deadline:
                        raise TimeoutError("no eligible camera pair before point timeout")
                    pair = self._select(now)
                    if pair is not None:
                        return pair, now
                    self.condition.wait(0.1)
        finally:
            self.end()

    def end(self):
        with self.condition:
            self.request = None
            for queue in self.queues.values():
                self.discarded["after_capture"] += len(queue)
                queue.clear()

    def stats(self):
        with self.condition:
            return {"received": dict(self.received), "discarded": dict(self.discarded)}
