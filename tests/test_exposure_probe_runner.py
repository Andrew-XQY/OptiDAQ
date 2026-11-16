"""Exposure-probe lifecycle tests; every hardware operation is an offline fake."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


def load_runner():
    path = Path(__file__).resolve().parents[1] / "experiments" / "exposure_probe.py"
    spec = importlib.util.spec_from_file_location("exposure_probe_runner_test_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner():
    return load_runner()


@pytest.fixture
def plan():
    return {
        "source": "laser", "exposures_us": {"cam1": [1000, 2000, 4000], "cam2": [50, 100, 200]},
        "scenes": ["bright"], "repeats": 3, "dmd_period_us": None,
        "target_low": 0.7, "target_high": 0.85,
    }


@pytest.fixture
def cfg():
    path = Path(__file__).resolve().parents[1] / "configs" / "experiment_lab_motor_raster.json"
    config = json.loads(path.read_text())
    for camera in ("cam1", "cam2"):
        config["output"]["png"]["transforms"][camera] = [
            {"name": "crop_box", "params": {"x1": 0, "y1": 0, "x2": 32, "y2": 32}}
        ]
    return config


class FakeHardware:
    """Models setters that may mutate hardware and still raise an exception."""

    original = {"cam1": 1234.0, "cam2": 567.0}

    def __init__(self):
        self.exposures = dict(self.original)
        self.calls = []
        self.phase = "background_before"
        self.prompt_count = 0
        self.capture_count = 0
        self.fail_set_once = None
        self.capture_error = None
        self.restore_error = None
        self.close_error = None

    def snapshot(self):
        self.calls.append(("snapshot",))
        result = {}
        for camera, exposure in self.exposures.items():
            setting = {
                "acqTrigger": 2, "cameraPixelFormat": 40,
                "cameraExposureAuto": 1, "cameraGainAuto": 1,
                "cameraGammaEnable": False, "cameraExposureTimeUs": exposure,
                "cameraExposureTimeUs_min": 40, "cameraExposureTimeUs_max": 10_000_000,
                "cameraGain": 0.0, "cameraTriggerDelay": 0.0,
            }
            acquired = {"acq" + key[0].upper() + key[1:]: value for key, value in setting.items()}
            result[camera] = {"CameraSetting": setting, "AcquiredParameters": acquired,
                              "TriggerConfiguration": {"trigAwakeSource": 0}}
        return result

    def set_exposure(self, camera, exposure):
        self.calls.append(("set", camera, exposure))
        self.exposures[camera] = float(exposure)
        if self.fail_set_once == camera:
            self.fail_set_once = None
            raise RuntimeError("setter failed after applying exposure")
        if self.restore_error == camera and exposure == self.original[camera]:
            raise RuntimeError("restore failed for " + camera)

    def wait_settings(self, exposures):
        self.calls.append(("wait", dict(exposures)))
        assert self.exposures == {key: float(value) for key, value in exposures.items()}
        return self.snapshot()

    def capture(self, exposures, repeats):
        self.calls.append(("capture", dict(exposures), repeats))
        self.capture_count += 1
        if self.capture_error:
            raise self.capture_error
        yy, xx = np.mgrid[:32, :32]
        shape = np.exp(-((xx - 16) ** 2 + (yy - 16) ** 2) / 32)
        images = {}
        for camera, exposure in exposures.items():
            amplitude = 0 if self.phase.startswith("background") else exposure / (1 if camera == "cam1" else .05)
            images[camera] = np.stack([
                np.rint(30 + frame - 1 + shape * amplitude).clip(0, 4095).astype(np.uint16)
                for frame in range(repeats)
            ])
        return images, {"offline_fake": True, "phase": self.phase, "capture_number": self.capture_count}

    def prompt(self, message):
        self.calls.append(("prompt", str(message)))
        self.phase = ("background_before", "bright", "background_after")[self.prompt_count]
        self.prompt_count += 1
        return ""

    def close(self):
        self.calls.append(("close",))
        if self.close_error:
            raise self.close_error


def report(out):
    return json.loads((out / "probe.json").read_text())


def restore_calls(hardware):
    return [call for call in hardware.calls if call[0] == "set" and call[2] == hardware.original[call[1]]]


def test_success_restores_original_exposures_and_never_applies_recommendation(runner, cfg, plan, tmp_path):
    hardware = FakeHardware()
    out = tmp_path / "probe"
    runner.run_probe(cfg, plan, out, hardware=hardware, prompt=hardware.prompt)
    assert hardware.exposures == hardware.original
    assert restore_calls(hardware) == [("set", "cam1", 1234.0), ("set", "cam2", 567.0)]
    assert hardware.prompt_count == 3
    assert hardware.capture_count == 9
    allowed = {camera: set(values) | {hardware.original[camera]} for camera, values in plan["exposures_us"].items()}
    assert all(call[2] in allowed[call[1]] for call in hardware.calls if call[0] == "set")
    assert report(out)["status"] == "complete"
    assert report(out)["original_settings_verified"] is True
    assert len(report(out)["captures"]) == 9
    with np.load(out / report(out)["captures"][0]["file"], allow_pickle=False) as stored:
        assert stored["cam1"].shape == (3, 32, 32)
        assert stored["cam1"].dtype == np.uint16


def test_partial_apply_then_raise_still_restores_both_cameras(runner, cfg, plan, tmp_path):
    hardware = FakeHardware()
    hardware.fail_set_once = "cam1"
    out = tmp_path / "probe"
    with pytest.raises(RuntimeError, match="setter failed after applying"):
        runner.run_probe(cfg, plan, out, hardware=hardware, prompt=hardware.prompt)
    assert hardware.exposures == hardware.original
    assert {item[1] for item in restore_calls(hardware)} == {"cam1", "cam2"}
    assert hardware.capture_count == 0
    assert "setter failed after applying" in json.dumps(report(out))
    assert report(out)["status"] == "failed"


def test_capture_error_is_logged_and_restores_both_cameras(runner, cfg, plan, tmp_path):
    hardware = FakeHardware()
    hardware.capture_error = ValueError("capture stream failed")
    hardware.capture_error.probe_cleanup_errors = ["capture cleanup: late subscription error"]
    out = tmp_path / "probe"
    with pytest.raises(ValueError, match="capture stream failed"):
        runner.run_probe(cfg, plan, out, hardware=hardware, prompt=hardware.prompt)
    assert hardware.exposures == hardware.original
    assert {item[1] for item in restore_calls(hardware)} == {"cam1", "cam2"}
    assert "capture stream failed" in json.dumps(report(out))
    assert "late subscription error" in json.dumps(report(out))


def test_keyboard_interrupt_is_logged_and_restores_both_cameras(runner, cfg, plan, tmp_path):
    hardware = FakeHardware()
    hardware.capture_error = KeyboardInterrupt("user stopped capture")
    out = tmp_path / "probe"
    with pytest.raises(KeyboardInterrupt, match="user stopped capture"):
        runner.run_probe(cfg, plan, out, hardware=hardware, prompt=hardware.prompt)
    assert hardware.exposures == hardware.original
    assert {item[1] for item in restore_calls(hardware)} == {"cam1", "cam2"}
    assert "KeyboardInterrupt" in json.dumps(report(out))
    assert report(out)["status"] == "failed"


def test_cleanup_failures_do_not_mask_capture_error_or_skip_other_restore(runner, cfg, plan, tmp_path):
    hardware = FakeHardware()
    hardware.capture_error = ValueError("primary capture failure")
    hardware.restore_error = "cam1"
    hardware.close_error = RuntimeError("stream close failure")
    out = tmp_path / "probe"
    with pytest.raises(ValueError, match="primary capture failure"):
        runner.run_probe(cfg, plan, out, hardware=hardware, prompt=hardware.prompt)
    assert {item[1] for item in restore_calls(hardware)} == {"cam1", "cam2"}
    assert hardware.exposures["cam2"] == hardware.original["cam2"]
    text = json.dumps(report(out))
    assert "primary capture failure" in text
    assert "restore failed for cam1" in text
    assert "stream close failure" in text


def test_existing_output_is_preserved_without_hardware_access(runner, cfg, plan, tmp_path):
    out = tmp_path / "probe"
    out.mkdir()
    sentinel = out / "probe.json"
    sentinel.write_text('{"existing": true}\n')
    hardware = FakeHardware()
    with pytest.raises((FileExistsError, ValueError, RuntimeError)):
        runner.run_probe(cfg, plan, out, hardware=hardware, prompt=hardware.prompt)
    assert sentinel.read_text() == '{"existing": true}\n'
    assert not hardware.calls


def test_ladder_limits_are_checked_before_any_exposure_write(runner, cfg, plan, tmp_path):
    hardware = FakeHardware()
    snapshot = hardware.snapshot

    def limited_snapshot():
        values = snapshot()
        values["cam2"]["CameraSetting"]["cameraExposureTimeUs_max"] = 150
        return values

    hardware.snapshot = limited_snapshot
    out = tmp_path / "probe"
    with pytest.raises(ValueError, match="exceeds live camera limits"):
        runner.run_probe(cfg, plan, out, hardware=hardware, prompt=hardware.prompt)
    assert not any(call[0] == "set" for call in hardware.calls)
    assert report(out)["status"] == "failed"


def test_nonzero_gain_is_rejected_without_mutating_settings(runner, cfg, plan, tmp_path):
    hardware = FakeHardware()
    snapshot = hardware.snapshot

    def nonzero_gain():
        values = snapshot()
        values["cam1"]["CameraSetting"]["cameraGain"] = 1
        values["cam1"]["AcquiredParameters"]["acqCameraGain"] = 1
        return values

    hardware.snapshot = nonzero_gain
    with pytest.raises(RuntimeError, match="gain must already be zero"):
        runner.run_probe(cfg, plan, tmp_path / "probe", hardware=hardware, prompt=hardware.prompt)
    assert not any(call[0] == "set" for call in hardware.calls)


def test_restoration_failure_invalidates_otherwise_successful_probe(runner, cfg, plan, tmp_path):
    hardware = FakeHardware()
    hardware.restore_error = "cam1"
    out = tmp_path / "probe"
    with pytest.raises(RuntimeError, match="restoration/cleanup failed"):
        runner.run_probe(cfg, plan, out, hardware=hardware, prompt=hardware.prompt)
    assert hardware.capture_count == 9
    assert {item[1] for item in restore_calls(hardware)} == {"cam1", "cam2"}
    assert report(out)["status"] == "failed"


@pytest.mark.parametrize("values", [[1, 2], [1, 1, 2], [3, 2, 1], [0, 1, 2], [-1, 1, 2], [1, 2, float("nan")], [1, 2, float("inf")]])
def test_invalid_exposure_ladders_rejected(runner, plan, values):
    plan["exposures_us"]["cam1"] = values
    with pytest.raises((ValueError, RuntimeError)):
        runner.validate_plan(plan)


def test_valid_asymmetric_camera_ladders_are_allowed(runner, plan):
    runner.validate_plan(plan)


def test_unequal_ladder_lengths_rejected(runner, plan):
    plan["exposures_us"]["cam2"].append(400)
    with pytest.raises((ValueError, RuntimeError)):
        runner.validate_plan(plan)


def test_live_adapter_writes_only_exposure_and_refuses_active_streams(runner, monkeypatch):
    calls = []
    adapter = ModuleType("optidaq.extensions.cern_pyda")
    adapter.set_value = lambda client, endpoint, values: calls.append((endpoint, values))
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    hardware = runner.LabHardware.__new__(runner.LabHardware)
    hardware.client = object()
    hardware.camera_map = {"cam1": "CAMERA_1", "cam2": "CAMERA_2"}
    hardware.taps = []
    hardware.set_exposure("cam1", 1000)
    assert len(calls) == 1
    assert calls[0][0] == "CAMERA_1/CameraSetting"
    assert set(calls[0][1]) == {"cameraExposureTimeUs"}
    assert int(calls[0][1]["cameraExposureTimeUs"]) == 1000
    hardware.taps.append(object())
    with pytest.raises(RuntimeError, match="close subscriptions"):
        hardware.set_exposure("cam2", 2000)
    assert len(calls) == 1


@pytest.fixture
def live_boundary(runner, cfg, monkeypatch):
    from optidaq.daq import lab_capture
    from optidaq.extensions import lab_camera
    from optidaq.daq.contracts import Frame

    state = SimpleNamespace(taps=[], gates=[], failure=None, stop_failure=None)

    class Gate:
        def __init__(self, aliases, config):
            self.frames = {}
            self.ended = False
            state.gates.append(self)

        def receive(self, frame):
            self.frames[frame.device] = frame

        def raise_if_failed(self):
            return None

        def wait_initial(self, timeout, check):
            check()
            assert set(self.frames) == {"cam1", "cam2"}

        def begin(self, index, actuator):
            return SimpleNamespace(metadata=lambda: {"index": index, "actuator": actuator})

        def wait_pair(self, timeout, check):
            check()
            if state.failure:
                raise state.failure
            return dict(self.frames), 100

        def end(self):
            self.ended = True

    class Tap:
        image_dtype = np.uint16
        image_shape = (2, 2)
        error = None
        _thread = None

        def __init__(self, camera):
            self.device = camera
            self.probed = self.started = self.stopped = self.joined = False
            state.taps.append(self)

        def probe(self):
            self.probed = True

        def start(self, receive):
            self.started = True
            # Every new exposure session starts at source counter 1, like the
            # camera after a reconfiguration. No old gate may survive.
            frame = Frame(self.device, 1, 1000, np.full((2, 2), 50, np.uint16).tobytes())
            object.__setattr__(frame, "sync_metadata", {"imageNumber": 1})
            receive(frame)

        def stop(self):
            self.stopped = True
            if state.stop_failure:
                raise state.stop_failure

        def join(self, timeout):
            self.joined = True

    monkeypatch.setattr(lab_capture, "FreshPairGate", Gate)
    monkeypatch.setattr(lab_camera, "build_lab_taps", lambda client, config: [Tap("cam1"), Tap("cam2")])
    hardware = runner.LabHardware.__new__(runner.LabHardware)
    hardware.client, hardware.cfg, hardware.taps = object(), cfg, []
    fake = FakeHardware()

    def wait_settings(exposures):
        fake.exposures = dict(exposures)
        return fake.snapshot()

    hardware.wait_settings = wait_settings
    return hardware, state


def test_each_exposure_capture_uses_new_gate_and_streams(live_boundary):
    hardware, state = live_boundary
    for exposures in ({"cam1": 1000, "cam2": 50}, {"cam1": 2000, "cam2": 100}):
        images, metadata = hardware.capture(exposures, 1)
        assert images["cam1"].shape == (1, 2, 2)
        assert not hardware.taps
    assert len(state.gates) == 2
    assert all(gate.ended for gate in state.gates)
    assert len(state.taps) == 4
    assert all(tap.probed and tap.started and tap.stopped and tap.joined for tap in state.taps)


def test_live_capture_cleanup_does_not_mask_original_exception(live_boundary):
    hardware, state = live_boundary
    state.failure = KeyboardInterrupt("original capture interrupt")
    state.stop_failure = RuntimeError("subscription stop failed")
    with pytest.raises(KeyboardInterrupt, match="original capture interrupt"):
        hardware.capture({"cam1": 1000, "cam2": 50}, 1)
    assert all(tap.stopped and tap.joined for tap in state.taps)
    assert "subscription stop failed" in " ".join(state.failure.probe_cleanup_errors)
