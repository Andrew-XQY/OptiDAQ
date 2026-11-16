"""Offline lab lifecycle: real raster, PNG storage and capture-ID database."""

from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import sys
import time
from types import ModuleType, SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from optidaq.actuators.motor_point_raster import MotorPointRasterController
from optidaq.daq.contracts import Frame
from optidaq.daq.lab_capture import CaptureRequest, QUALITY
from optidaq.daq import lab_motor_run as runner
from optidaq.extensions import lab_camera


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class Axis:
    """Instant motion at the SDK boundary; the controller performs real checks."""

    def __init__(self, name, client, cfg, *, require_valid_position):
        self.name, self.device = name, cfg["device"]
        self.counter = {"x": 79912, "y": 19908}[name]
        self.units, self.units_scale, self.units_offset = "Steps", 1.0, 0.0
        self.position_min, self.position_max = -1000000, 1000000
        self.tol_counts, self.settle_s = cfg["tol_counts"], cfg["settle_s"]
        self.timeout_s, self.poll_s = cfg["timeout_s"], cfg["poll_s"]
        self.moves, self.stops = [], 0

    def acquisition(self):
        return {"position_counter": self.counter, "position": self.counter,
                "position_valid": False, "motor_busy": False, "status_channel": 1032}

    def driver_acquisition(self):
        return {"ps_power_state": True, "ps_error": 0}

    def preflight(self):
        self.acquisition()

    def _counter(self, values):
        return values["position_counter"]

    def _position(self, values):
        return values["position"]

    def position_for_counter(self, counter):
        return counter

    def range_check_absolute_counters(self, values):
        assert all(self.position_min <= value <= self.position_max for value in values)

    def move_abs_raw(self, target):
        self.acquisition()
        if abs(target - self.counter) > self.tol_counts:
            self.moves.append(target)
            self.counter = target
        self.acquisition()

    def stop(self):
        self.stops += 1


def source_frame(camera, step):
    stamp = 1_900_000_000_000_000_000 + (step + 1) * 4_000_000_000
    offset = 0 if camera == "cam1" else 1000
    pixels = np.array([[step + 1, 2], [3, 4095]], dtype=np.int16)
    frame = Frame(camera, offset + step + 100, stamp, pixels.tobytes())
    object.__setattr__(frame, "sync_metadata", {
        "imageTimeStamp": stamp, "imageTimeStampRaw": (step + 1) * 125_000_000,
        "frameNumber": offset + step + 500, "triggerMode": 3,
        "acqImageTransmissionTime": 1000 + offset,
        "image_ready_ns": stamp + 1000 + offset,
        "local_receive_ns": stamp + 100_000_000,
        "local_receive_monotonic_ns": (step + 1) * 4_000_000_000,
        "image_min": 2 if step > 0 else 1, "image_max": 4095,
        "saturated_pixels": 1,
    })
    return frame


class ScriptedGate:
    """Deterministic selection only; real selector timing has separate tests."""

    def __init__(self, aliases, cfg):
        self.aliases, self.error, self.current = tuple(aliases), None, None
        self.seen, self.steps = set(), []

    def receive(self, frame):
        self.seen.add(frame.device)

    def raise_if_failed(self):
        if self.error:
            raise self.error

    def wait_initial(self, timeout, check):
        check()
        assert self.seen == set(self.aliases)

    def begin(self, step_id, actuator):
        assert self.current is None
        self.steps.append(step_id)
        self.current = CaptureRequest(step_id, time.time_ns(), time.monotonic_ns(),
                                      {camera: step_id for camera in self.aliases},
                                      2_000_000_000, deepcopy(actuator))
        return self.current

    def wait_pair(self, timeout, check):
        check()
        step = self.current.step_id
        pair = {camera: source_frame(camera, step) for camera in self.aliases}
        self.end()
        return pair, time.monotonic_ns()

    def end(self):
        self.current = None

    def stats(self):
        return {"scripted_test_pairs": len(self.steps)}


@pytest.fixture
def lab(tmp_path, monkeypatch):
    cfg = {
        "run": {"data_root": str(tmp_path), "runs_subdir": "runs", "free_disk_margin": 2},
        "rbac": {"enabled": True}, "cameras": ["cam1", "cam2"],
        "camera_setting": {camera: {"device": camera.upper(), "cameraExposureTimeUs": 100000,
                                    "cameraTriggerDelay": 0.0} for camera in ("cam1", "cam2")},
        "output": {"format": "png", "png": {"mode": "raw16", "pairing": "capture_id",
                   "transforms": [], "metadata_hooks": ["stats"], "compress_level": 1}},
        "writer": {"configure_cameras": True}, "postprocess": {"actions": []},
        "tap": {"mode": "subscribe", "subscribe_selector": "SPS.USER.ALL"},
        "capture": {"guard_s": 2.0, "pair_tolerance_ms": 50.0, "max_frame_age_s": 4.0},
        "scan": {"nx": 10, "ny": 10, "stride_counts": {"x": 100, "y": 100},
                 "start_offset_counts": {"x": -400, "y": -400}, "serpentine": True,
                 "frames_per_point": 1, "point_timeout_s": 15.0},
        "motors": {"position_mode": "raw_counter", "return_to_start": True,
                   **{name: {"device": "MOTOR." + name, "tol_counts": 5, "poll_s": .01,
                             "settle_s": 0, "timeout_s": 1} for name in ("x", "y")}},
    }
    requested = {"acqTrigger": 2, "cameraPixelFormat": 40, "cameraExposureAuto": 1,
                 "cameraGainAuto": 1, "cameraGammaEnable": False, "cameraExposureTimeUs": 100000,
                 "cameraGain": 0.0, "cameraTriggerDelay": 0.0}
    snapshot = {"CameraSetting": requested,
                "AcquiredParameters": {"acq" + key[0].upper() + key[1:]: value
                                       for key, value in requested.items()},
                "TriggerConfiguration": {"trigAwakeSource": 0}}
    state = SimpleNamespace(cfg=cfg, axes={}, controllers=[], taps=[], backends=[], gates=[],
                            initial_counters={"x": 79912, "y": 19908},
                            calls=[], settings={camera.upper(): deepcopy(snapshot) for camera in cfg["cameras"]},
                            on_start=lambda camera: None, on_stop=lambda camera: None)

    # No SDK imports/connections: only the adapter boundary is replaced. The
    # actual setting reader and validator still run against these FESA values.
    adapter = ModuleType("optidaq.extensions.cern_pyda")
    adapter.camera_map_from_config = lambda config: {
        camera: config["camera_setting"][camera]["device"] for camera in config["cameras"]}
    adapter.get_value = lambda client, endpoint: deepcopy(state.settings[endpoint.split("/")[0]][endpoint.split("/")[1]])
    adapter.jsonable = lambda value: deepcopy(value)
    adapter.ensure_cameras_alive = lambda *args, **kwargs: state.calls.append("alive")
    adapter.configure_cameras = lambda *args, **kwargs: state.calls.append("configure")
    adapter.collect_camera_source_provenance = lambda *args: {"test": "offline"}
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    monkeypatch.setattr(lab_camera, "_adapter", lambda: adapter)
    monkeypatch.setattr(runner, "make_lab_client", lambda **kwargs: object())

    def controller_factory(client, config, should_stop):
        def axis_factory(*args, **kwargs):
            axis = Axis(*args, **kwargs)
            axis.counter = state.initial_counters[axis.name]
            state.axes[axis.name] = axis
            return axis
        controller = MotorPointRasterController(client, config, should_stop, axis_factory=axis_factory)
        state.controllers.append(controller)
        return controller

    class Tap:
        def __init__(self, camera):
            self.device, self.fesa_device, self.error = camera, camera.upper(), None
            self.image_shape, self.image_dtype, self.payload_nbytes = (2, 2), "int16", 8
            self.started, self.stopped, self.joined, self._thread = False, False, False, None

        def probe(self):
            pass

        def start(self, emit):
            self.started = True
            self._thread = SimpleNamespace(is_alive=lambda: False)
            emit(source_frame(self.device, 0))
            state.on_start(self.device)

        def stop(self):
            self.stopped = True
            state.on_stop(self.device)

        def join(self, timeout):
            self.joined = True

    def taps(client, config):
        state.taps = [Tap(camera) for camera in config["cameras"]]
        return state.taps

    def gate_factory(*args):
        gate = ScriptedGate(*args)
        state.gates.append(gate)
        return gate

    make_backend = runner.make_save_backend
    def backend_factory(*args, **kwargs):
        backend = make_backend(*args, **kwargs)
        state.backends.append(backend)
        return backend

    monkeypatch.setattr(runner, "MotorPointRasterController", controller_factory)
    monkeypatch.setattr(runner, "build_lab_taps", taps)
    monkeypatch.setattr(runner, "FreshPairGate", gate_factory)
    monkeypatch.setattr(runner, "make_save_backend", backend_factory)
    state.run = lambda **kwargs: runner.run_motor_raster(cfg, tmp_path / "lab.json", **kwargs)
    state.run_dir = lambda: next((tmp_path / "runs").iterdir())
    return state


def assert_closed(lab):
    assert all(tap.started and tap.stopped and tap.joined for tap in lab.taps)
    assert all(backend._frame_index is None for backend in lab.backends)
    assert lab.gates[0].current is None


@pytest.mark.parametrize("stride", [100, 500, "span"])
def test_full_hundred_point_run_preserves_pairs_coordinates_pixels_and_returns(lab, stride):
    if stride == "span":
        lab.cfg["scan"].pop("stride_counts")
        lab.cfg["scan"].pop("start_offset_counts")
        lab.cfg["scan"]["span_counts"] = {"x": 10000, "y": 10000}
        lab.cfg["motors"]["reference_counters"] = {"x": 79912, "y": 19908}
        lab.initial_counters["x"] = 77896
        expected_offsets = [-5000, -3889, -2778, -1667, -556, 556, 1667, 2778, 3889, 5000]
    else:
        lab.cfg["scan"]["stride_counts"] = {"x": stride, "y": stride}
        lab.cfg["scan"]["start_offset_counts"] = {"x": -4 * stride, "y": -4 * stride}
        expected_offsets = [(index - 4) * stride for index in range(10)]
    initial = dict(lab.initial_counters)
    reference = lab.cfg["motors"].get("reference_counters", initial)
    original = deepcopy(lab.cfg)
    run = lab.run()
    assert lab.cfg == original
    assert lab.calls == ["alive", "configure"]
    assert_closed(lab)
    controller = lab.controllers[0]
    assert controller.restored
    assert {name: axis.counter for name, axis in lab.axes.items()} == initial
    assert lab.gates[0].steps == list(range(100))
    assert len(list(run.rglob("*.png"))) == 200
    commits = rows(run / "capture_commits.jsonl")
    assert len(commits) == 100
    resolved = json.loads((run / "resolved_config.json").read_text())
    assert resolved["actuator"]["initial_counters"] == initial
    assert resolved["actuator"]["reference_counters"] == reference
    assert resolved["scan_plan"] == [point.to_dict() for point in controller.plan]
    assert len({(p.x_offset_counts, p.y_offset_counts) for p in controller.plan}) == 100
    assert sorted({p.x_offset_counts for p in controller.plan}) == expected_offsets
    assert sorted({p.y_offset_counts for p in controller.plan}) == expected_offsets
    for a, b in zip(controller.plan, controller.plan[1:]):
        delta = [b.x_offset_counts - a.x_offset_counts, b.y_offset_counts - a.y_offset_counts]
        assert sum(value != 0 for value in delta) == 1
        assert sum(abs(value) for value in delta) in ({1111, 1112} if stride == "span" else {stride})

    with sqlite3.connect(run / "dataset.db") as database:
        images = database.execute(
            "SELECT camera,frame_index,sample_id,png_path,sync_json,chosen_sync_source "
            "FROM images ORDER BY frame_index,camera").fetchall()
        assert len(images) == 200
        for camera, index, sample, path, sync_text, source in images:
            sync, point = json.loads(sync_text), controller.plan[index]
            capture = sync["capture"]
            target = {"x": reference["x"] + point.x_offset_counts, "y": reference["y"] + point.y_offset_counts}
            assert (sample, capture["step_id"], capture["capture_id"], source) == (
                index + 1, index, "capture_%06d" % index, "capture_id")
            assert capture["quality"] == QUALITY
            assert capture["actuator"]["target_counters"] == target
            assert capture["actuator"]["initial_counters"] == initial
            assert capture["actuator"]["reference_counters"] == reference
            for name in ("x", "y"):
                assert capture["actuator"]["readbacks"][name]["position_counter"] == target[name]
                assert capture["actuator_held_readbacks"][name]["position_counter"] == target[name]
                assert commits[index]["post_save_readbacks"][name]["position_counter"] == target[name]
            assert sync["saturated_pixels"] == 1 and sync["image_max"] == 4095
            expected = np.frombuffer(source_frame(camera, index).payload_bytes, dtype=np.int16).reshape(2, 2)
            with Image.open(run / path) as image:
                np.testing.assert_array_equal(np.asarray(image), expected)
            assert commits[index]["saved"][0 if camera == "cam1" else 1]["frame_index"] == index
        for first, second in zip(images[::2], images[1::2]):
            assert json.loads(first[4])["capture"] == json.loads(second[4])["capture"]
    summary = json.loads((run / "run_summary.json").read_text())
    assert summary["ok"] and summary["points_committed"] == summary["points_requested"] == 100


@pytest.mark.parametrize("failed_step", [0, 1])
def test_second_camera_save_failure_keeps_orphan_evidence_without_commit_or_advance(lab, monkeypatch, failed_step):
    from optidaq.daq.save_backends import PngImageSaveBackend
    save = PngImageSaveBackend.save
    def fail_second_camera(self, frame):
        if self.device == "cam2" and self.frame_count == failed_step:
            raise OSError("simulated camera-2 disk failure")
        save(self, frame)
    monkeypatch.setattr(PngImageSaveBackend, "save", fail_second_camera)
    with pytest.raises(OSError, match="camera-2 disk failure"):
        lab.run()
    run = lab.run_dir()
    assert_closed(lab)
    assert len(rows(run / "capture_commits.jsonl")) == failed_step
    assert len(list(run.rglob("*.png"))) == 2 * failed_step + 1 and not (run / "dataset.db").exists()
    assert lab.gates[0].steps == list(range(failed_step + 1)) and not lab.controllers[0].restored
    assert lab.axes["x"].moves == [79512 + 100 * step for step in range(failed_step + 1)]
    assert lab.axes["y"].moves == [19508]
    summary = json.loads((run / "run_summary.json").read_text())
    assert not summary["ok"] and summary["points_committed"] == failed_step
    assert "OSError" in summary["error"]


def test_settings_change_after_capture_blocks_database_publication(lab):
    lab.cfg["scan"].update(nx=2, ny=1)
    def change_setting(camera):
        if camera == "cam2":
            for values in lab.settings.values():
                values["CameraSetting"]["cameraGain"] = .5
                values["AcquiredParameters"]["acqCameraGain"] = .5
    lab.on_stop = change_setting
    with pytest.raises(RuntimeError, match="settings changed"):
        lab.run()
    run = lab.run_dir()
    assert_closed(lab)
    assert len(rows(run / "capture_commits.jsonl")) == 2
    assert not (run / "dataset.db").exists() and not lab.controllers[0].restored
    summary = json.loads((run / "run_summary.json").read_text())
    assert not summary["ok"] and summary["points_committed"] == 2


def test_late_gate_failure_during_subscription_shutdown_blocks_return_and_database(lab):
    lab.cfg["scan"].update(nx=2, ny=1)
    def late_failure(camera):
        if camera == "cam2":
            lab.gates[0].error = RuntimeError("late camera timestamp regression")
    lab.on_stop = late_failure
    with pytest.raises(RuntimeError, match="late camera timestamp regression"):
        lab.run()
    run = lab.run_dir()
    assert_closed(lab)
    assert len(rows(run / "capture_commits.jsonl")) == 2
    assert not (run / "dataset.db").exists() and not lab.controllers[0].restored
    assert lab.axes["x"].moves == [79512, 79612] and lab.axes["y"].moves == [19508]
    summary = json.loads((run / "run_summary.json").read_text())
    assert not summary["ok"] and summary["points_committed"] == 2


def test_cancellation_before_first_move_closes_resources_without_motion(lab):
    stop = [False]
    lab.on_start = lambda camera: stop.__setitem__(0, True)
    with pytest.raises(KeyboardInterrupt):
        lab.run(should_stop=lambda: stop[0])
    assert_closed(lab)
    assert all(not axis.moves and not axis.stops for axis in lab.axes.values())
    assert not list(lab.run_dir().rglob("*.png"))
    assert not (lab.run_dir() / "dataset.db").exists()
    assert lab.gates[0].steps == []


def test_cancellation_before_preflight_does_not_create_a_run_or_touch_axes(lab):
    with pytest.raises(KeyboardInterrupt):
        lab.run(should_stop=lambda: True)
    assert not lab.axes and not lab.calls and not lab.taps and not lab.backends
    assert not Path(lab.cfg["run"]["data_root"], "runs").exists()
