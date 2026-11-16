"""The save gate must reject transitions and never pair by callback arrival order."""
import pytest

from optidaq.daq.contracts import Frame
from optidaq.daq.lab_capture import FreshPairGate, validate_exposure_guard


NS = 1_000_000_000


class Clock:
    wall = 10 * NS

    def time_ns(self):
        return self.wall

    def monotonic_ns(self):
        return self.wall - 5 * NS

    def monotonic(self):
        return self.monotonic_ns() / NS


def frame(cam, number, proxy, **overrides):
    stamp = proxy - 1_000_000
    item = Frame(cam, number, stamp, b"\0\0")
    data = dict(triggerMode=3, frameNumber=number, imageTimeStampRaw=stamp,
                imageTimeStamp=stamp, acqImageTransmissionTime=1_000_000,
                image_ready_ns=proxy, local_receive_ns=proxy + 1_000_000,
                local_receive_monotonic_ns=proxy + 1_000_000 - 5 * NS)
    data.update(overrides)
    object.__setattr__(item, "sync_metadata", data)
    return item


def gate():
    clock = Clock()
    item = FreshPairGate(["cam1", "cam2"], dict(guard_s=2, pair_tolerance_ms=1,
                                             max_frame_age_s=2, queue_size=2), clock=clock)
    for cam in item.aliases:
        item.receive(frame(cam, 1, 9 * NS))
    return item, clock


def test_motion_frames_are_discarded_without_filling_queue():
    item, clock = gate()
    for number in range(2, 100):
        for cam in item.aliases:
            item.receive(frame(cam, number, (number + 8) * NS))
    item.raise_if_failed()
    assert not any(item.queues.values())
    assert item.stats()["discarded"]["outside_capture"] == 198
    clock.wall = 108 * NS
    request = item.begin(0, {"target_counters": {"x": 500, "y": 1000}})
    assert request.baseline == {"cam1": 99, "cam2": 99}


def test_first_callbacks_can_mismatch_but_selected_pair_matches_timing():
    item, clock = gate()
    request = item.begin(7, {"target_counters": {"x": 500, "y": 1000}})
    item.receive(frame("cam1", 2, 13 * NS))
    item.receive(frame("cam2", 2, 14 * NS + 500_000))
    item.receive(frame("cam1", 3, 14 * NS))
    clock.wall = 15 * NS
    pair, _ = item.wait_pair(10, lambda: None)
    assert [pair[cam].source_image_number for cam in item.aliases] == [3, 2]
    assert item.stats()["discarded"]["unmatched_timing"] == 1
    assert request.metadata()["capture_id"] == "capture_000007"
    assert item.request is None
    assert not any(item.queues.values())


@pytest.mark.parametrize("proxy,overrides,reason", [
    (11 * NS, {}, "inside_host_transition_guard"),
    (11 * NS, {"local_receive_monotonic_ns": 8 * NS, "local_receive_ns": 13 * NS}, "inside_transition_guard"),
    (13 * NS, {"imageTimeStamp": 9 * NS, "acqImageTransmissionTime": 4 * NS}, "reported_start_before_ready"),
    (13 * NS, {"local_receive_monotonic_ns": 5 * NS, "local_receive_ns": 10 * NS}, "received_before_ready"),
    (13 * NS, {}, "stale_receipt"),
])
def test_rejects_old_or_transition_frames(proxy, overrides, reason):
    item, clock = gate()
    item.begin(0, {})
    item.receive(frame("cam1", 2, proxy, **overrides))
    item.raise_if_failed()
    now = 11 * NS if reason == "stale_receipt" else 8 * NS
    with item.condition:
        assert item._select(now) is None
    assert item.discarded[reason] == 1


@pytest.mark.parametrize("overrides", [
    {"triggerMode": 1}, {"image_ready_ns": 123}, {"frameNumber": None},
    {"imageTimeStampRaw": 1}, {"local_receive_ns": 15 * NS},
])
def test_invalid_frame_or_clock_regression_aborts(overrides):
    item, _ = gate()
    item.receive(frame("cam1", 2, 13 * NS, **overrides))
    with pytest.raises(RuntimeError, match="camera capture failed"):
        item.raise_if_failed()


def test_ahead_source_clock_cannot_bypass_elapsed_host_guard():
    item, clock = gate()
    item.begin(0, {})
    for cam in item.aliases:
        item.receive(frame(cam, 2, 13 * NS, local_receive_ns=10 * NS + 50_000_000,
                           local_receive_monotonic_ns=5 * NS + 50_000_000))
    item.raise_if_failed()
    with item.condition:
        assert item._select(5 * NS + 100_000_000) is None
    assert item.discarded["inside_host_transition_guard"] == 2


def test_source_counter_reset_aborts_even_when_idle():
    item, _ = gate()
    item.receive(frame("cam1", 1, 13 * NS))
    with pytest.raises(RuntimeError, match="counter reset/duplicate"):
        item.raise_if_failed()


def test_active_overflow_aborts_instead_of_dropping_arbitrary_frames():
    item, _ = gate()
    item.begin(0, {})
    for number in range(2, 5):
        item.receive(frame("cam1", number, (number + 11) * NS))
    with pytest.raises(RuntimeError, match="queue overflow"):
        item.raise_if_failed()


def test_timeout_and_cancellation_reset_request():
    item, clock = gate()
    item.begin(0, {})
    clock.wall = 15 * NS
    with pytest.raises(TimeoutError):
        item.wait_pair(4, lambda: None)
    assert item.request is None
    item.begin(1, {})

    def stop():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        item.wait_pair(10, stop)
    assert item.request is None
    assert not any(item.queues.values())


def test_guard_cannot_be_shorter_than_an_applied_exposure():
    settings = {"cam1": {"AcquiredParameters": {"acqCameraExposureTimeUs": 3_000_000}},
                "cam2": {"AcquiredParameters": {"acqCameraExposureTimeUs": 1_000}}}
    with pytest.raises(ValueError, match="longest applied camera exposure"):
        validate_exposure_guard({"guard_s": 2}, settings)
    validate_exposure_guard({"guard_s": 3}, settings)
