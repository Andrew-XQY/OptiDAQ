"""Keep raw CERN and PNG lab entry points separate before touching hardware."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import types
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for name in ("pyda", "pyrbac", "pyda_rda3"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["pyda"].SimpleClient = type("SimpleClient", (), {})
sys.modules["pyrbac"].AuthenticationListener = type("AuthenticationListener", (), {})
sys.modules["pyrbac"].LoginService = type("LoginService", (), {})
sys.modules["pyda_rda3"].RdaProvider = type("RdaProvider", (), {})


@pytest.fixture(scope="module")
def runners():
    result = {}
    for kind, filename in (("raw", "experiment_v1.py"), ("lab", "experiment_lab_v1.py")):
        spec = importlib.util.spec_from_file_location(
            "optidaq_runner_split_test_" + kind, ROOT / "experiments" / filename
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        result[kind] = module
    return result


@pytest.mark.parametrize("kind, config_name, expected_format", [
    ("raw", "experiment_v1.json", "raw"),
    ("lab", "experiment_lab_v1.json", "png"),
])
def test_each_runner_loads_its_own_default_profile(runners, kind, config_name, expected_format):
    runner = runners[kind]
    assert runner.DEFAULT_CONFIG_PATH == ROOT / "configs" / config_name
    cfg = runner.load_config(runner.DEFAULT_CONFIG_PATH)
    assert str(cfg.get("output", {}).get("format", "raw")).lower() == expected_format
    runner.postprocess_config(cfg)


@pytest.mark.parametrize("kind, other_config", [
    ("raw", "experiment_lab_v1.json"),
    ("lab", "experiment_v1.json"),
])
def test_cross_runner_configuration_is_rejected_before_client_creation(
    runners, monkeypatch, kind, other_config
):
    runner = runners[kind]
    client = Mock(side_effect=AssertionError("Hardware must not be reached"))
    monkeypatch.setattr(runner, "make_client", client)
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)
    monkeypatch.setattr(sys, "argv", [
        str(runner.__file__), "--config", str(ROOT / "configs" / other_config),
    ])

    with pytest.raises((RuntimeError, ValueError)):
        runner.main()

    client.assert_not_called()


def test_lab_camera_only_mode_remains_available(runners):
    runner = runners["lab"]
    cfg = deepcopy(runner.load_config(runner.DEFAULT_CONFIG_PATH))
    cfg["actuator"] = {"mode": "noop"}
    cfg["postprocess"]["actions"] = []

    runner.validate_config_minimal(cfg)
    assert runner.actuator_mode(cfg) == "noop"
    assert runner.postprocess_config(cfg)["actions"] == []


def test_checked_in_motor_profile_routes_to_coordinated_lab_runner(runners, monkeypatch):
    from optidaq.daq import lab_motor_run
    from optidaq.actuators.motor_point_raster import make_motor_raster_plan

    runner = runners["lab"]
    config_path = ROOT / "configs" / "experiment_lab_motor_raster.json"
    config = runner.load_config(config_path)
    assert runner.actuator_mode(config) == "motor_raster"
    assert config["output"]["png"]["pairing"] == "capture_id"
    assert config["output"]["png"]["mode"] == "raw16"
    assert runner.postprocess_config(config)["actions"] == []
    assert config["tap"]["subscribe_selector"] == "SPS.USER.ALL"
    assert config["scan"]["frames_per_point"] == 1
    assert config["scan"]["span_counts"] == {"x": 11000, "y": 30000}
    assert "stride_counts" not in config["scan"] and "start_offset_counts" not in config["scan"]
    assert config["motors"]["reference_counters"] == {"x": 79912, "y": 19908}
    assert len(make_motor_raster_plan(config)) == 10000

    dispatch = Mock()
    independent_client = Mock(side_effect=AssertionError("Independent camera acquisition must not start"))
    monkeypatch.setattr(lab_motor_run, "run_motor_raster", dispatch)
    monkeypatch.setattr(runner, "make_client", independent_client)
    monkeypatch.setattr(runner, "STOP_REQUESTED", False)
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)
    monkeypatch.setattr(sys, "argv", [str(runner.__file__), "--config", str(config_path)])
    runner.main()
    dispatch.assert_called_once()
    assert dispatch.call_args.args == (config, config_path)
    assert not dispatch.call_args.kwargs["should_stop"]()
    monkeypatch.setattr(runner, "STOP_REQUESTED", True)
    assert dispatch.call_args.kwargs["should_stop"]()
    independent_client.assert_not_called()


@pytest.mark.parametrize("mode", ["motor", "magnet_current_scan"])
def test_lab_rejects_cern_scan_modes_before_client_creation(runners, monkeypatch, tmp_path, mode):
    runner = runners["lab"]
    cfg = json.loads((ROOT / "configs" / "experiment_lab_v1.json").read_text())
    cfg["actuator"] = {"mode": mode}
    cfg["postprocess"]["actions"] = []
    path = tmp_path / "unsupported_scan.json"
    path.write_text(json.dumps(cfg))
    client = Mock(side_effect=AssertionError("Hardware must not be reached"))
    monkeypatch.setattr(runner, "make_client", client)
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)
    monkeypatch.setattr(sys, "argv", [str(runner.__file__), "--config", str(path)])

    with pytest.raises((RuntimeError, ValueError)):
        runner.main()

    client.assert_not_called()


@pytest.mark.parametrize("kind", ["raw", "lab"])
@pytest.mark.parametrize("acquisition_fails", [False, True])
def test_runner_closes_real_files_before_cleanup_and_finalization(
    runners, monkeypatch, tmp_path, kind, acquisition_fails
):
    from optidaq.daq.contracts import Frame
    from optidaq.daq.save_backends import RawStreamSaveBackend

    runner = runners[kind]
    cfg = deepcopy(runner.load_config(runner.DEFAULT_CONFIG_PATH))
    cfg["run"]["data_root"] = str(tmp_path)
    cfg["actuator"] = {"mode": "noop", "points": 1}
    cfg["scan"]["frames_per_point"] = 1
    cfg["scan"]["point_timeout_s"] = 2.0
    cfg["writer"]["configure_cameras"] = False
    cfg["postprocess"] = {"build_aligned_index": False, "validate_after_scan": True}
    if kind == "lab":
        cfg["postprocess"]["actions"] = []
        cfg["output"]["png"]["transforms"] = {}
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(cfg))
    (tmp_path / cfg["run"]["runs_subdir"]).mkdir(parents=True)
    events, taps, backends, summaries = [], [], [], []
    payload = b"\x01\x00\x02\x00\x03\x00\x04\x00"

    class FakeTap:
        def __init__(self, alias, device, client, **kwargs):
            self.device, self.fesa_device = alias, device
            self.image_shape, self.image_dtype = (2, 2), "uint16"
            self.payload_nbytes = len(payload)
            self.stamp_source, self.probe_pulse_id = "image_ready_ns", None
            self.error = None
            taps.append(self)

        def probe(self):
            pass

        def start(self, emit):
            self.emit = emit

        def stop(self):
            pass

    class TrackedBackend(RawStreamSaveBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            backends.append(self)

        def close(self):
            super().close()
            events.append("close:" + self.device)

    class TrackedNoop(runner.NoopActuatorController):
        def cleanup(self):
            assert set(events) == {"close:cam1", "close:cam2"}
            events.append("cleanup")

    class InjectedFrameWaiter(runner.FrameWaiter):
        def wait(self, markers, item, extra=None):
            for tap in taps:
                frame = Frame(tap.device, 41, 1_000_000_000_000_000_000, payload)
                object.__setattr__(frame, "sync_metadata", {
                    "image_ready_ns": frame.acq_stamp_ns,
                    "chosen_sync_source": "image_ready_ns",
                    "chosen_sync_value": frame.acq_stamp_ns,
                })
                tap.emit(frame)
            super().wait(markers, item, extra)
            if acquisition_fails:
                raise RuntimeError("injected acquisition failure")

    validator_name = "validate_device_stream" if kind == "raw" else "validate_png_device"
    validate_files = getattr(runner, validator_name)

    def validate_after_close(run_dir, spec_or_alias, *args, **kwargs):
        assert "cleanup" in events
        summary = validate_files(run_dir, spec_or_alias, *args, **kwargs)
        summaries.append(summary)
        events.append("validate")
        return summary

    patches = {
        "STOP_REQUESTED": False,
        "make_client": Mock(return_value=object()),
        "SubscribingCameraTap": FakeTap,
        "PollingCameraTap": FakeTap,
        "collect_camera_source_provenance": Mock(return_value={}),
        "validate_tap_stamp_sources": Mock(),
        "NoopActuatorController": TrackedNoop,
        "CounterFreezeWatchdog": Mock(),
        "FrameWaiter": InjectedFrameWaiter,
        validator_name: validate_after_close,
    }
    if kind == "raw":
        patches["RawStreamSaveBackend"] = TrackedBackend
        patches["maybe_launch_quad_view"] = Mock(return_value=None)
    else:
        make_backend = runner.make_save_backend
        finalize_dataset = runner.finalize_png_dataset

        def tracked_png_backend(*args, **kwargs):
            backend = make_backend(*args, **kwargs)
            backends.append(backend)
            close = backend.close

            def tracked_close():
                close()
                events.append("close:" + backend.device)

            backend.close = tracked_close
            return backend

        def finalize_after_close(*args, **kwargs):
            assert "cleanup" in events
            result = finalize_dataset(*args, **kwargs)
            events.append("finalize")
            return result

        patches["make_save_backend"] = tracked_png_backend
        patches["finalize_png_dataset"] = finalize_after_close
    for name, value in patches.items():
        monkeypatch.setattr(runner, name, value)
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)
    monkeypatch.setattr(sys, "argv", [str(runner.__file__), "--config", str(config_path)])

    if acquisition_fails:
        with pytest.raises(RuntimeError, match="injected acquisition failure"):
            runner.main()
    else:
        runner.main()

    assert set(events[:2]) == {"close:cam1", "close:cam2"}
    assert events[2] == "cleanup"
    assert len(summaries) == (0 if acquisition_fails else 2)
    assert all(summary["n_records"] == 1 and not summary["errors"] for summary in summaries)
    run_dir, = (tmp_path / cfg["run"]["runs_subdir"]).glob("run_*")
    if kind == "raw":
        raw_files = list(run_dir.glob("*.raw"))
        assert len(raw_files) == 2
        assert all(path.read_bytes().endswith(payload) for path in raw_files)
        assert not list(run_dir.rglob("*.png"))
    else:
        from PIL import Image

        png_files = list(run_dir.rglob("*.png"))
        assert len(png_files) == 2
        for path in png_files:
            with Image.open(path) as image:
                assert list(image.getdata()) == [1, 2, 3, 4]
        assert not list(run_dir.glob("*.raw"))
        assert (run_dir / "dataset.db").exists() == (not acquisition_fails)
        if not acquisition_fails:
            assert events.index("cleanup") < events.index("finalize") < events.index("validate")
            with sqlite3.connect(run_dir / "dataset.db") as db:
                assert db.execute("SELECT count(*), count(DISTINCT sample_id) FROM images").fetchone() == (2, 1)
