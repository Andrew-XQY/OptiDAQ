"""Offline checks for ensure_camera_alive (fake CERN SDKs, no hardware).

Run: python3 -m pytest tests/test_ensure_camera_alive.py -q
"""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Minimal stand-ins so optidaq.extensions.cern_pyda imports without acc-py.
for name in ("pyda", "pyrbac", "pyda_rda3"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["pyda"].SimpleClient = type("SimpleClient", (), {})
sys.modules["pyrbac"].AuthenticationListener = type("AuthenticationListener", (), {})
sys.modules["pyrbac"].LoginService = type("LoginService", (), {})
sys.modules["pyda_rda3"].RdaProvider = type("RdaProvider", (), {})

from optidaq.extensions import cern_pyda  # noqa: E402


class _Response:
    def __init__(self, data):
        self.data = data
        self.exception = None


class FakeClient:
    """Serves HardwareStatus / AcquisitionStatus from scripted values, records SETs."""

    def __init__(self, power_on=1, statuses=(0,), switch_connected=1, refuse=()):
        self.power_on = power_on
        self.switch_connected = switch_connected
        self.statuses = list(statuses)      # acqCameraStatus returned on successive GETs
        self.refuse = set(refuse)           # commands the fake server rejects
        self.sets = []

    def get(self, endpoint, context=None):
        prop = endpoint.split("/")[1]
        if prop == "HardwareStatus":
            return _Response({"cameraSwitchStatus": self.power_on,
                              "cameraSwitchConnectionStatus": self.switch_connected})
        if prop == "AcquisitionStatus":
            code = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            return _Response({"acqCameraStatus": code, "acqMsg": "msg %s" % code})
        raise AssertionError("unexpected GET " + endpoint)

    def set(self, endpoint, data=None, context=None):
        prop = endpoint.split("/")[1]
        self.sets.append((prop, int(data["operation"])))
        if prop in self.refuse:
            raise RuntimeError("USER_ERROR: The power switch seems to be disconnected")
        return _Response({})


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(cern_pyda.time, "sleep", lambda s: None)


def test_already_ok_sends_nothing():
    client = FakeClient(power_on=0, switch_connected=0, statuses=(0,))   # switch fields unreliable
    report = cern_pyda.ensure_camera_alive(client, "cam1", "DEV")
    assert client.sets == []
    assert report["actions"] == [] and report["acqCameraStatus"] == "OK"


def test_unreachable_sends_connect_only():
    client = FakeClient(power_on=1, statuses=(2, 2, 0))   # UNREACHABLE, then OK after CONNECT
    report = cern_pyda.ensure_camera_alive(client, "cam1", "DEV")
    assert client.sets == [("ManageCameraConnection", 1)]
    assert report["camera_was_ok"] is False and report["acqCameraStatus"] == "OK"


def test_power_off_with_switch_connected_sends_on_then_connect():
    client = FakeClient(power_on=0, switch_connected=1, statuses=(4, 4, 0))
    report = cern_pyda.ensure_camera_alive(client, "cam2", "DEV")
    assert client.sets == [("ManageCameraPower", 1), ("ManageCameraConnection", 1)]
    assert report["power_switch"] == {"controller_connected": True, "port_on": False}


def test_switch_controller_down_skips_power_command():
    client = FakeClient(power_on=0, switch_connected=0, statuses=(2, 2, 0))
    cern_pyda.ensure_camera_alive(client, "cam1", "DEV")
    assert client.sets == [("ManageCameraConnection", 1)]


def test_refused_command_is_a_warning_when_camera_recovers():
    client = FakeClient(power_on=0, switch_connected=1, statuses=(2, 2, 0), refuse={"ManageCameraPower"})
    report = cern_pyda.ensure_camera_alive(client, "cam1", "DEV")
    assert report["actions"] == ["ManageCameraConnection=CONNECT"]
    assert report["warnings"] and "ManageCameraPower=ON refused" in report["warnings"][0]


def test_never_sends_off_or_disconnect_even_when_forced():
    client = FakeClient(power_on=1, switch_connected=1, statuses=(0,))
    cern_pyda.ensure_camera_alive(client, "cam1", "DEV", force=True)
    assert all(op == 1 for _, op in client.sets)
    assert [name for name, _ in client.sets] == ["ManageCameraPower", "ManageCameraConnection"]


def test_timeout_raises_with_server_message(monkeypatch):
    client = FakeClient(power_on=1, statuses=(2,))        # stays UNREACHABLE
    clock = iter([0.0, 0.0, 100.0])
    monkeypatch.setattr(cern_pyda.time, "monotonic", lambda: next(clock, 100.0))
    with pytest.raises(RuntimeError, match="UNREACHABLE"):
        cern_pyda.ensure_camera_alive(client, "cam1", "DEV", timeout_s=30.0)


def test_writer_flag_disables_it():
    client = FakeClient(power_on=0, statuses=(2,))
    assert cern_pyda.ensure_cameras_alive(client, {"cam1": "DEV"}, {"ensure_camera_alive": False}) == {}
    assert client.sets == []
