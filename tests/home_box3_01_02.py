"""Configure homing fields and home the two BOX3 scan motors.

Run from the repository root in the CERN acc-py environment after selecting
the homing direction:

    python tests/home_box3_01_02.py

Writes ExpertSetting homing fields, then sends Home. No counters are rewritten.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from optidaq.extensions.cern_pyda import get_value, make_client, plain, set_value  # noqa: E402


DEVICES = (
    ("x", "CA.AIR-TS.BOX3.01"),
    ("y", "CA.AIR-TS.BOX3.02"),
)

CONFIG = {
    "home_steps": {
        "x": -50000.0,
        "y": -50000.0,
    },
    "home_speed": 1000.0,
    "timeout_s": 180.0,
    "poll_s": 0.5,
    "require_rbac": True,
    "stop_all": False,
}


def get_acq(client: Any, device: str) -> dict[str, Any]:
    return get_value(client, "%s/Acquisition" % device)


def get_driver(client: Any, device: str) -> dict[str, Any]:
    return get_value(client, "%s/DriverAcquisition" % device)


def get_expert_acq(client: Any, device: str) -> dict[str, Any]:
    return get_value(client, "%s/ExpertAcquisition" % device)


def get_expert_setting(client: Any, device: str) -> dict[str, Any]:
    return get_value(client, "%s/ExpertSetting" % device)


def counter(acq: dict[str, Any]) -> Any:
    for key in ("position_counter", "positionCounter"):
        if key in acq:
            return plain(acq[key])
    return "--"


def print_state(client: Any, name: str, device: str) -> None:
    acq = get_acq(client, device)
    expert = get_expert_acq(client, device)
    setting = get_expert_setting(client, device)
    driver = get_driver(client, device)
    print(
        "%s %s pos=%s counter=%s valid=%s busy=%s plus=%s minus=%s "
        "homing_state=%s home_steps=%s home_stop=%s home_speed=%s "
        "driver_power=%s ps_error=%s"
        % (
            name,
            device,
            plain(acq.get("position", "--")),
            counter(acq),
            bool(acq.get("position_valid")),
            bool(acq.get("motor_busy")),
            bool(acq.get("end_switch_plus_active")),
            bool(acq.get("end_switch_minus_active")),
            plain(expert.get("homing_state", "--")),
            plain(setting.get("home_steps", "--")),
            plain(setting.get("home_stop_at_switch", "--")),
            plain(setting.get("home_speed", "--")),
            bool(driver.get("ps_power_state")),
            plain(driver.get("ps_error", "--")),
        )
    )


def require_ready(client: Any, name: str, device: str) -> None:
    acq = get_acq(client, device)
    driver = get_driver(client, device)
    if bool(acq.get("motor_busy")):
        raise RuntimeError("%s %s is already moving" % (name, device))
    if not bool(driver.get("ps_power_state")):
        raise RuntimeError("%s %s driver power is off" % (name, device))
    ps_error = int(plain(driver.get("ps_error", 0)))
    if ps_error != 0:
        raise RuntimeError("%s %s driver ps_error=%d" % (name, device, ps_error))


def configure_home(client: Any, device: str, steps: float, speed: float) -> None:
    if steps == 0:
        raise RuntimeError("home_steps must be nonzero")
    payload = {
        "home_steps": np.float64(steps),
        "home_stop_at_switch": True,
        "home_speed": np.float64(speed),
    }
    set_value(client, "%s/ExpertSetting" % device, payload)


def stop(client: Any, device: str) -> None:
    try:
        set_value(client, "%s/Stop" % device, {"stop_all": bool(CONFIG["stop_all"])})
    except BaseException as exc:
        print("  STOP failed: %s" % str(exc).split("\n", 1)[0])


def wait_home(client: Any, name: str, device: str, timeout_s: float, poll_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        acq = get_acq(client, device)
        expert = get_expert_acq(client, device)
        print(
            "  pos=%s counter=%s valid=%s busy=%s homing_state=%s plus=%s minus=%s"
            % (
                plain(acq.get("position", "--")),
                counter(acq),
                bool(acq.get("position_valid")),
                bool(acq.get("motor_busy")),
                plain(expert.get("homing_state", "--")),
                bool(acq.get("end_switch_plus_active")),
                bool(acq.get("end_switch_minus_active")),
            )
        )
        if bool(acq.get("position_valid")) and not bool(acq.get("motor_busy")):
            print("  %s homed OK" % name)
            return
        time.sleep(poll_s)

    stop(client, device)
    raise RuntimeError("%s %s home timeout; STOP sent" % (name, device))


def home_one(
    client: Any,
    name: str,
    device: str,
    *,
    steps: float,
    speed: float,
    timeout_s: float,
    poll_s: float,
) -> None:
    print("")
    print("=" * 88)
    print("HOME %s %s steps=%s speed=%s" % (name, device, steps, speed))
    print("before:")
    print_state(client, name, device)
    require_ready(client, name, device)
    configure_home(client, device, steps, speed)
    print("after ExpertSetting write:")
    print_state(client, name, device)
    set_value(client, "%s/Home" % device, {"doHome": True})
    print("Home fired")
    wait_home(client, name, device, timeout_s, poll_s)
    print("final:")
    print_state(client, name, device)


def main() -> None:
    client = make_client(require_rbac=bool(CONFIG["require_rbac"]))
    print("BOX3 .01/.02 HOME ATTEMPT")
    print("Writes ExpertSetting home_steps/home_stop_at_switch/home_speed, then Home.")
    print("No Move/MvtAbs/MvtRel/Reset/counter writes are sent.")
    print("Config: %s" % CONFIG)

    try:
        for name, device in DEVICES:
            home_one(
                client,
                name,
                device,
                steps=float(CONFIG["home_steps"][name]),
                speed=float(CONFIG["home_speed"]),
                timeout_s=float(CONFIG["timeout_s"]),
                poll_s=float(CONFIG["poll_s"]),
            )
    except KeyboardInterrupt:
        print("\nCtrl-C: sending Stop to both BOX3 axes.")
        for _name, device in DEVICES:
            stop(client, device)
        raise
    except BaseException:
        print("\nError: sending Stop to both BOX3 axes before re-raising.")
        for _name, device in DEVICES:
            stop(client, device)
        raise


if __name__ == "__main__":
    main()
