#!/usr/bin/env python3
"""Read selected CLEAR magnet Acquisition properties with FESA GET requests.

Run from the repository root in the CERN acc-py environment:

    python experiments/read_magnets.py

Optional device-name arguments replace the defaults. No settings are written.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from optidaq.extensions.cern_pyda import get_value, make_client  # noqa: E402

# Default FESA device names to read.
DEFAULT_DEVICES = [
    "CS.QFG0525",
    "CS.QDG0535",
    "CS.QFG0545",
    "CS.DHJ0505",
    "CS.DVJ0505",
]

SELECTOR = "SCT.USER.SETUP"  # cycle selector used for Acquisition readback


def read_current(client, device: str) -> float:
    """One GET of <device>/Acquisition; return measured currentAverage (A)."""
    acq = get_value(client, "%s/Acquisition" % device, context=SELECTOR)
    if "currentAverage" not in acq:
        raise RuntimeError("%s/Acquisition has no 'currentAverage'" % device)
    return float(acq["currentAverage"])


def main(devices=None) -> None:
    devices = devices or DEFAULT_DEVICES
    client = make_client(require_rbac=False)  # GET-only; reads need no RBAC token
    print("READ-ONLY magnet currents  (selector=%s)" % SELECTOR)
    for device in devices:
        try:
            print("  %-12s = %+10.4f A" % (device, read_current(client, device)))
        except Exception as exc:  # report and continue if one device is unreachable
            print("  %-12s = ERROR: %s" % (device, exc))


if __name__ == "__main__":
    main(sys.argv[1:] or None)
