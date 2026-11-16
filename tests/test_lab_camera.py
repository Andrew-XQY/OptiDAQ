"""Lab camera adapter tests with no CERN connection or device commands."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from optidaq.extensions import lab_camera


@pytest.fixture
def adapter(monkeypatch):
    # Load the real shared adapter under an isolated name, substituting only
    # its import-time SDK dependencies. Exercise its real frame decoding.
    for name, values in {
        "pyda": {"SimpleClient": object},
        "pyda_rda3": {"RdaProvider": object},
        "pyrbac": {"AuthenticationListener": object},
    }.items():
        fake = ModuleType(name)
        fake.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, fake)
    path = Path(__file__).resolve().parents[1] / "src/optidaq/extensions/cern_pyda.py"
    spec = importlib.util.spec_from_file_location("_lab_camera_test_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(lab_camera, "_adapter", lambda: module)
    return module


def config():
    return {"cameras": ["cam1"], "camera_setting": {"cam1": {"device": "CAM1"}},
            "tap": {"mode": "subscribe", "subscribe_selector": "SPS.USER.ALL",
                    "stamp_fields": ["imageTimeStamp"]}}


def response(number=2, image=None, trigger=3, first=False):
    return SimpleNamespace(
        exception=None, first_update=first,
        header=SimpleNamespace(acquisition_timestamp=1_900_000_000_100_000_000, cycle_timestamp=0),
        data={"image2D": np.array([[0, 1], [2, 4095]], dtype=np.int16) if image is None else image,
              "imageNumber": number, "frameNumber": number, "triggerMode": trigger,
              "imageTimeStamp": 1_900_000_000_000_000_000 + number * 1_000_000_000,
              "imageTimeStampRaw": number*125_000_000, "acqImageTransmissionTime": 1000})


def prepared_tap(adapter, client=None):
    tap = lab_camera.build_lab_taps(client, config())[0]
    tap.image_shape, tap.image_dtype, tap.payload_nbytes = (2, 2), "int16", 8
    tap._last_image_number = 1
    return tap


def test_real_frame_builder_retains_timing_and_receipt_metadata(adapter):
    tap = prepared_tap(adapter)
    frames = []
    source = response()
    start = time.monotonic_ns()
    tap._emit_subscription_response(source, frames.append)
    assert len(frames) == 1
    frame = frames[0]
    sync = frame.sync_metadata
    assert sync["image_ready_ns"] == source.data["imageTimeStamp"]+1000
    assert start <= sync["local_receive_monotonic_ns"] <= time.monotonic_ns()
    assert sync["triggerMode"] == 3 and sync["saturated_pixels"] == 1
    assert (sync["image_min"], sync["image_max"]) == (0, 4095)
    assert sync["header"]["acquisition_timestamp"] == source.header.acquisition_timestamp
    source.data["image2D"][:] = 44
    assert np.frombuffer(frame.payload_bytes, np.int16).tolist() == [0, 1, 2, 4095]


def test_cached_updates_duplicates_and_counter_reset(adapter):
    tap = prepared_tap(adapter)
    frames = []
    tap._emit_subscription_response(response(number=99, first=True), frames.append)
    assert not frames
    tap._emit_subscription_response(response(number=2), frames.append)
    tap._emit_subscription_response(response(number=2), frames.append)
    assert len(frames) == 1
    with pytest.raises(RuntimeError, match="went backwards"):
        tap._emit_subscription_response(response(number=1), frames.append)


@pytest.mark.parametrize("bad", [
    response(trigger=2),
    response(image=np.zeros((1, 4), dtype=np.int16)),
    response(image=np.zeros((2, 2), dtype=np.uint16)),
    response(image=np.full((2, 2), -1, dtype=np.int16)),
    response(image=np.full((2, 2), 4096, dtype=np.int16)),
])
def test_changed_trigger_geometry_or_invalid_counts_are_rejected(adapter, bad):
    tap = prepared_tap(adapter)
    with pytest.raises(RuntimeError):
        tap._emit_subscription_response(bad, lambda frame: pytest.fail("invalid frame emitted"))


def test_iterator_stop_does_not_wait_for_next_hardware_frame(adapter):
    class SubscriptionTimeoutError(Exception):
        pass

    class Subscription:
        def __init__(self):
            self.calls, self.closed = [], False
            self.waited = threading.Event()

        def __iter__(self):
            return self

        def __next__(self):
            raise AssertionError("blocking iteration must not be used")

        def next(self, *, timeout):
            self.calls.append(timeout)
            self.waited.set()
            time.sleep(0.01)
            raise SubscriptionTimeoutError()

        def close(self):
            self.closed = True

    subscription = Subscription()
    def subscribe(endpoint, **kwargs):
        assert kwargs["receive_first_updates"] is False
        if "callback" in kwargs:
            raise TypeError("this SDK uses an iterator")
        return subscription

    tap = prepared_tap(adapter, SimpleNamespace(subscribe=subscribe))
    tap.start(lambda frame: pytest.fail("no frames were provided"))
    assert subscription.waited.wait(1)
    tap.stop()
    tap.join(1)
    assert not tap._thread.is_alive() and tap.error is None
    assert subscription.closed and all(t == 0.25 for t in subscription.calls)


def test_failed_iterator_releases_monitor_despite_saved_exception(adapter):
    released = threading.Event()

    class Iterator:
        def __iter__(self):
            return self

        def __next__(self):
            raise AssertionError("blocking iteration must not be used")

        def next(self, *, timeout):
            raise RuntimeError("subscription disconnected")

        def __del__(self):
            released.set()

    class Subscription:
        closed = False

        def __iter__(self):
            return Iterator()

        def close(self):
            self.closed = True

    subscription = Subscription()
    def subscribe(endpoint, **kwargs):
        if "callback" in kwargs:
            raise TypeError("this SDK uses an iterator")
        return subscription

    tap = prepared_tap(adapter, SimpleNamespace(subscribe=subscribe))
    tap.start(lambda frame: pytest.fail("no frames were provided"))
    tap.join(1)
    assert not tap._thread.is_alive() and subscription.closed
    assert isinstance(tap.error, RuntimeError) and "disconnected" in str(tap.error)
    assert released.wait(.2), "saved exception must not keep the SDK monitor alive"


def settings():
    requested = {"acqTrigger": 2, "cameraPixelFormat": 40, "cameraExposureAuto": 1,
                 "cameraGainAuto": 1, "cameraGammaEnable": False,
                 "cameraExposureTimeUs": 100000, "cameraGain": 0.0, "cameraTriggerDelay": 0.0}
    applied = {"acq"+k[0].upper()+k[1:]: v for k, v in requested.items()}
    return {"cam1": {"CameraSetting": requested, "AcquiredParameters": applied,
                     "TriggerConfiguration": {"trigAwakeSource": 0}}}


def test_external_settings_are_applied_and_unchanged():
    before = settings()
    lab_camera.verify_external_settings(before, deepcopy(before))
    after = deepcopy(before)
    after["cam1"]["CameraSetting"]["cameraGain"] = 0.5
    with pytest.raises(RuntimeError, match="applied readback"):
        lab_camera.verify_external_settings(after)
    after["cam1"]["AcquiredParameters"]["acqCameraGain"] = 0.5
    with pytest.raises(RuntimeError, match="changed during"):
        lab_camera.verify_external_settings(before, after)
    after = deepcopy(before)
    after["cam1"]["CameraSetting"]["cameraGammaEnable"] = True
    with pytest.raises(RuntimeError, match="cameraGammaEnable"):
        lab_camera.verify_external_settings(after)


def auth_adapter(tmp_path, token=None, *, cached=None, kerberos=None, location=None):
    path = tmp_path/"fake-token"
    path.write_bytes(b"fake credential for offline test")
    events = []
    def factory(listener):
        events.append("kerberos")
        listener.token = kerberos
        return object()
    def location_login(provider, client):
        events.append("location")
        return location
    module = SimpleNamespace(
        RdaProvider=lambda: SimpleNamespace(rbac_token=None),
        pyda=SimpleNamespace(SimpleClient=lambda provider: SimpleNamespace(provider=provider)),
        _load_cached_token=lambda: cached, _token_cache_path=lambda: path,
        _token_from_bytes=lambda value: token,
        pyrbac=SimpleNamespace(LoginService=SimpleNamespace(create_for_kerberos=factory)),
        _RbacListener=lambda callback: SimpleNamespace(token=None),
        _await_login=lambda listener, required: listener.token is not None,
        _login_location=location_login, _save_token=lambda value: None,
    )
    return module, events


def test_authoritative_unexpired_cache_survives_old_helper_rejection(monkeypatch, tmp_path):
    token = SimpleNamespace(is_expired=False)
    adapter, events = auth_adapter(tmp_path, token)
    monkeypatch.setattr(lab_camera, "_adapter", lambda: adapter)
    assert lab_camera.make_lab_client().provider.rbac_token is token
    assert events == []


def test_expired_cache_falls_back_to_existing_kerberos(monkeypatch, tmp_path):
    expired, valid = SimpleNamespace(is_expired=True), SimpleNamespace(is_expired=False)
    adapter, events = auth_adapter(tmp_path, expired, cached=expired, kerberos=valid)
    monkeypatch.setattr(lab_camera, "_adapter", lambda: adapter)
    client = lab_camera.make_lab_client()
    assert client.provider.rbac_token is valid and events == ["kerberos"]
    assert client._optidaq_rbac_listener.token is valid


def test_auth_failure_is_noninteractive(monkeypatch, tmp_path):
    adapter, events = auth_adapter(tmp_path)
    monkeypatch.setattr(lab_camera, "_adapter", lambda: adapter)
    with pytest.raises(RuntimeError, match="refresh your CERN login"):
        lab_camera.make_lab_client()
    assert events == ["kerberos", "location"]


def test_camera_settings_keep_nested_numeric_values(adapter):
    class Numeric:
        shape = ()
        _value = 100000
    client = SimpleNamespace(get=lambda endpoint: SimpleNamespace(
        exception=None, data={"exposure": Numeric(), "nested": {"counter": np.int64(8)}}))
    result = lab_camera.read_camera_settings(client, {"cam1": "CAM1"})
    assert result["cam1"]["CameraSetting"] == {"exposure": 100000, "nested": {"counter": 8}}
