"""Offline runtime-control checks; all CERN GETs and SETs use a fake client."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Match the existing offline CERN tests without importing any hardware SDK.
for name in ("pyda", "pyrbac", "pyda_rda3"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["pyda"].SimpleClient = type("SimpleClient", (), {})
sys.modules["pyrbac"].AuthenticationListener = type("AuthenticationListener", (), {})
sys.modules["pyrbac"].LoginService = type("LoginService", (), {})
sys.modules["pyda_rda3"].RdaProvider = type("RdaProvider", (), {})

_spec = importlib.util.spec_from_file_location(
    "optidaq_camera_preview_controls_test", REPO_ROOT / "experiments" / "camera_preview.py"
)
preview = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = preview
_spec.loader.exec_module(preview)


class FakeClient:
    """Expose mutable live bounds and applied settings, recording actual SET payloads."""

    def __init__(self, limits, *, apply_exposure=True):
        self.limits = limits
        self.apply_exposure = apply_exposure
        self.applied = {
            device: {"acqCameraExposureTimeUs": 50000, "acqCameraGain": 0.0}
            for device in limits
        }
        self.gets = []
        self.sets = []

    def get(self, endpoint, context=None):
        self.gets.append(endpoint)
        device, prop = endpoint.split("/")
        if prop == "CameraSetting":
            data = self.limits[device]
            if isinstance(data, Exception):
                raise data
        elif prop == "AcquiredParameters":
            data = self.applied[device]
        else:
            raise AssertionError("Unexpected GET: " + endpoint)
        return types.SimpleNamespace(data=dict(data), exception=None)

    def set(self, endpoint, data=None, context=None):
        device, prop = endpoint.split("/")
        assert prop == "CameraSetting"
        payload = dict(data)
        self.sets.append((device, payload))
        if self.apply_exposure:
            self.applied[device]["acqCameraExposureTimeUs"] = int(payload["cameraExposureTimeUs"])
        self.applied[device]["acqCameraGain"] = float(payload["cameraGain"])
        return types.SimpleNamespace(data={}, exception=None)


def bounds(minimum, maximum):
    return {"cameraExposureTimeUs_min": minimum, "cameraExposureTimeUs_max": maximum}


@pytest.fixture
def make_controls(monkeypatch):
    # Exercise readback polling without waiting or reading pytest's stdin.
    clock = [10.0]
    monkeypatch.setattr(preview.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(preview.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(preview.RuntimeCameraControl, "_input_loop", lambda self: None)

    def build(limits, **client_kwargs):
        client = FakeClient(limits, **client_kwargs)
        taps = [
            types.SimpleNamespace(device="cam%d" % (index + 1), fesa_device=device)
            for index, device in enumerate(limits)
        ]
        cfg = {
            "tap": {"mode": "poll"},
            "camera_setting": {
                tap.device: {
                    "cameraExposureTimeUs": 50000,
                    "cameraGain": 0.0,
                    "cameraPixelFormat": "mono12",
                    "roi_enabled": False,
                }
                for tap in taps
            },
        }
        controls = preview.RuntimeCameraControl(
            client, taps, cfg,
            boxes={tap.device: None for tap in taps},
            display_shapes={tap.device: (16, 16) for tap in taps},
            source_boxes={tap.device: None for tap in taps},
            control_readbacks={},
            apply_camera_roi=False,
        )
        controls.write_interval_s = 0.0
        return client, controls

    return build


def command(controls, text):
    controls.commands.put(text)
    controls.update()


@pytest.mark.parametrize("requested", [100, 500000])
def test_accepts_live_supported_exposures_outside_old_ui_limits(make_controls, requested):
    client, controls = make_controls({"DEV1": bounds(25, 1000000)})

    command(controls, "1 exp %d" % requested)

    assert len(client.sets) == 1
    assert int(client.sets[0][1]["cameraExposureTimeUs"]) == requested
    assert controls.snapshot()["cam1"]["exposure_us"] == requested
    assert controls.snapshot()["cam1"]["last_error"] is None
    assert client.gets.count("DEV1/CameraSetting") == 2  # Startup and command.
    assert client.gets.count("DEV1/AcquiredParameters") >= 2  # Initial and post-SET.


@pytest.mark.parametrize("requested, expected", [(500, [500, 1500]), (700000, [700000, 300000])])
def test_all_exposure_clamps_each_camera_to_its_own_limits(make_controls, requested, expected):
    client, controls = make_controls({
        "DEV1": bounds(25, 1000000),
        "DEV2": bounds(1500, 300000),
    })

    command(controls, "all %d" % requested)

    assert [int(payload["cameraExposureTimeUs"]) for _, payload in client.sets] == expected
    assert [controls.snapshot()[name]["exposure_us"] for name in ("cam1", "cam2")] == expected


def test_exposure_command_refreshes_changed_camera_limits(make_controls):
    client, controls = make_controls({"DEV1": bounds(25, 300000)})
    command(controls, "1 exp 500000")
    assert controls.snapshot()["cam1"]["exposure_us"] == 300000

    client.limits["DEV1"] = bounds(10, 900000)
    command(controls, "1 exp 800000")

    assert int(client.sets[-1][1]["cameraExposureTimeUs"]) == 800000
    assert controls.snapshot()["cam1"]["exposure_limits_us"] == (10, 900000)
    assert controls.snapshot()["cam1"]["exposure_us"] == 800000


@pytest.mark.parametrize("invalid_limits", [
    {},
    bounds(1000, 100),
    RuntimeError("CameraSetting unavailable"),
], ids=["missing", "reversed", "get-fails"])
def test_unavailable_exposure_limits_block_exposure_but_allow_gain(make_controls, invalid_limits):
    client, controls = make_controls({"DEV1": invalid_limits})

    command(controls, "1 exp 500000")

    assert client.sets == []
    assert controls.snapshot()["cam1"]["pending_exposure_us"] is None
    assert "exposure limits" in controls.snapshot()["cam1"]["last_error"]

    command(controls, "1 gain 0.3")

    assert len(client.sets) == 1
    assert float(client.sets[0][1]["cameraGain"]) == pytest.approx(0.3)
    assert int(client.sets[0][1]["cameraExposureTimeUs"]) == 50000
    assert controls.snapshot()["cam1"]["gain"] == pytest.approx(0.3)


def test_readback_mismatch_is_reported_without_claiming_requested_exposure(make_controls, capsys):
    client, controls = make_controls({"DEV1": bounds(25, 1000000)}, apply_exposure=False)

    command(controls, "1 exp 500000")

    assert int(client.sets[0][1]["cameraExposureTimeUs"]) == 500000
    state = controls.snapshot()["cam1"]
    assert state["exposure_us"] == 50000
    assert state["pending_exposure_us"] is None
    assert "readback mismatch" in state["last_error"]
    assert "readback mismatch" in capsys.readouterr().out
