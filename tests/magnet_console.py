#!/usr/bin/env python3
"""Tiny interactive command-line control for the CLEAR magnets.

Prints a numbered menu (device, role, allowed range, reference, live setpoint/measured),
then reads commands:

    <id> <current>   set magnet <id> to <current> A     e.g.  "1 3.2"   or   "4 -3.3"
    r                read & print all magnet currents
    ref              set ALL magnets to the REFERENCE preset below
    scan             run the experiment's magnet random-walk (same backend, no cameras)
    m                reprint the menu (re-reads live values)
    q                quit

Writes to live hardware (SettingPPM). Two limits are enforced before any write:
  1. a HARD GUARD (quads +/-80 A, correctors +/-3 A) defined below, and
  2. the device's FESA SettingPPM range.
The allowed range is the intersection of the two; anything outside is refused. Device is
also preflighted (OK / ON / REMOTE). Use --dry-run to preview without writing.

Run on the CERN VM (same env as experiment_v1.py):
    ... python tests/magnet_console.py            # live control
    ... python tests/magnet_console.py --dry-run  # preview only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from optidaq.extensions.cern_pyda import (  # noqa: E402
    get_value,
    make_client,
    optional_int_like,
    plain,
    set_value,
)

# (device, role, hard_min A, hard_max A) — id is the position in this list (1-based).
# HARD GUARD (hard_min, hard_max), enforced ON TOP of the FESA range.
DEVICES = [
    ("CS.QFG0525", "QF size",    0.0,  100.0),
    ("CS.QDG0535", "QD size",    0.0,  100.0),
    ("CS.QFG0545", "QF size",    0.0,  100.0),
    ("CS.DHJ0505", "DH x-pos",  -5.0,   10.0),
    ("CS.DVJ0505", "DV y-pos",  -5.0,   10.0),
]
HARD = {dev: (lo, hi) for dev, _role, lo, hi in DEVICES}

# Reference currents (A); the "ref" command applies this preset.
# NOTE: values outside the hard guard above are refused (flagged at startup).
REFERENCE = {
    "CS.QFG0525": 57.5,
    "CS.QDG0535": 90.0,
    "CS.QFG0545": 32.5,
    "CS.DHJ0505": 3.5,
    "CS.DVJ0505": 3.0,
}

READ_SELECTOR = "SCT.USER.SETUP"   # Acquisition is cycle-bound
SET_CONTEXT = None                 # SettingPPM write uses no selector
BUSY_BIT = 4
SETTLE_TOL = 0.05                  # A, for the post-set readback wait


def _f(value):
    return float(plain(value))


def read_measured(client, dev):
    acq = get_value(client, "%s/Acquisition" % dev, context=READ_SELECTOR)
    return _f(acq["currentAverage"]), optional_int_like(acq.get("current_status"))


def read_setpoint(client, dev):
    sp = get_value(client, "%s/SettingPPM" % dev)
    return {
        "current": _f(sp["current"]),
        "min": _f(sp["current_min"]),
        "max": _f(sp["current_max"]),
        "tol_abs": _f(sp["current_tolAbs"]),
        "tol_rel": _f(sp["current_tolRel"]),
    }


def allowed_range(dev, sp):
    """Intersection of FESA SettingPPM range and the hard guard."""
    hlo, hhi = HARD.get(dev, (-1e30, 1e30))
    return max(sp["min"], hlo), min(sp["max"], hhi)


def preflight(client, dev):
    st = get_value(client, "%s/Status" % dev)
    bad = []
    if optional_int_like(st.get("status")) != 0:
        bad.append("status!=OK")
    if optional_int_like(st.get("control")) != 5:
        bad.append("not REMOTE")
    if optional_int_like(st.get("mode")) != 3:
        bad.append("not ON")
    return bad


def is_busy(current_status):
    return current_status is not None and bool(current_status & (1 << BUSY_BIT))


def write_current(client, dev, current, tol_abs, tol_rel):
    set_value(
        client,
        "%s/SettingPPM" % dev,
        {
            "current": np.float32(current),
            "current_tolAbs": np.float32(tol_abs),
            "current_tolRel": np.float32(tol_rel),
        },
        context=SET_CONTEXT,
    )


def set_one(client, dev, current, *, dry_run, timeout_s=5.0, poll_s=0.3):
    try:
        sp = read_setpoint(client, dev)
    except Exception as exc:
        print("  %s: read error: %s" % (dev, exc))
        return
    lo, hi = allowed_range(dev, sp)
    if not (lo <= current <= hi):
        print("  %s: %.4f A OUT OF ALLOWED RANGE [%.3f, %.3f] — REFUSED" % (dev, current, lo, hi))
        return
    bad = preflight(client, dev)
    if bad:
        print("  %s: not writable (%s) — refused" % (dev, ", ".join(bad)))
        return
    if dry_run:
        print("  %s: DRY-RUN would set -> %.4f A  (allowed [%.3f, %.3f])" % (dev, current, lo, hi))
        return
    write_current(client, dev, current, sp["tol_abs"], sp["tol_rel"])
    meas = None
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        meas, cs = read_measured(client, dev)
        if not is_busy(cs) and abs(meas - current) <= SETTLE_TOL:
            break
        time.sleep(poll_s)
    back = read_setpoint(client, dev)["current"]
    settled = meas is not None and abs(meas - current) <= SETTLE_TOL
    print("  %s: setpoint=%.4f A  measured=%.4f A%s"
          % (dev, back, meas if meas is not None else float("nan"), "" if settled else "  (not settled)"))


def read_all(client):
    print("Currents (A):")
    for i, (dev, role, _lo, _hi) in enumerate(DEVICES, 1):
        try:
            meas, _ = read_measured(client, dev)
            setp = read_setpoint(client, dev)["current"]
            print("  %2d  %-12s  %-8s  measured=%9.3f  setpoint=%9.3f" % (i, dev, role, meas, setp))
        except Exception as exc:
            print("  %2d  %-12s  read error: %s" % (i, dev, exc))


def set_reference(client, dry_run):
    print("Reference restore:")
    for dev, _role, _lo, _hi in DEVICES:
        if dev not in REFERENCE:
            print("  %s: no reference defined — skipped" % dev)
            continue
        set_one(client, dev, float(REFERENCE[dev]), dry_run=dry_run)


def print_menu(client, dry_run):
    print()
    print("CLEAR magnet control — %s" % ("DRY-RUN (no writes)" if dry_run else "LIVE (writes to hardware)"))
    print("  id  device        role      allowed (A)      ref       setpoint   measured")
    for i, (dev, role, _lo, _hi) in enumerate(DEVICES, 1):
        try:
            sp = read_setpoint(client, dev)
            meas, _ = read_measured(client, dev)
            lo, hi = allowed_range(dev, sp)
            rng = "[%g, %g]" % (lo, hi)
            ref = REFERENCE.get(dev)
            refs = "%.3f" % ref if ref is not None else "--"
            print("  %2d  %-12s  %-8s  %-14s  %-8s  %9.3f  %8.3f"
                  % (i, dev, role, rng, refs, sp["current"], meas))
        except Exception as exc:
            print("  %2d  %-12s  %-8s  read error: %s" % (i, dev, role, exc))
    print("  (allowed = FESA range ∩ per-device hard guard)")
    print('Commands:  "<id> <A>" (e.g. 1 3.2)  |  r (read all)  |  ref  |  scan  |  m (menu)  |  q (quit)')


def run_random_walk_scan(client, config_path, dry_run):
    """Run the experiment's magnet random-walk backend (same controller), no cameras.

    Builds MagnetRandomWalkController from the experiment config and drives its plan:
    preflight -> prepare (move to reference) -> apply each point (set + wait settle) ->
    cleanup (return to reference). Identical magnet behaviour to experiment_v1.py.
    """
    from optidaq.actuators.magnet_random_walk import MagnetRandomWalkController
    try:
        with open(config_path) as handle:
            cfg = json.load(handle)
    except Exception as exc:
        print("  scan: cannot load config %s: %s" % (config_path, exc))
        return
    try:
        controller = MagnetRandomWalkController(client, cfg)
        controller.preflight()  # reads starts, builds the walk plan, prints a summary
    except Exception as exc:
        print("  scan: setup failed: %s" % exc)
        return
    n = len(controller.plan)
    if dry_run:
        print("  DRY-RUN: %d-point random walk (not applying). First points:" % n)
        for point in controller.plan[:8]:
            print("    %03d %s" % (point.index, point.currents))
        return
    if input("  run %d-point random-walk scan (writes magnets)? [y/N] " % n).strip().lower() != "y":
        print("  cancelled")
        return
    controller.prepare()  # move to reference
    print("  running %d-point random-walk scan..." % n)
    try:
        for point in controller.plan:
            controller.apply(point)  # set magnets + wait for settle (same backend)
    except KeyboardInterrupt:
        print("  interrupted")
    finally:
        controller.cleanup()  # return to reference
    print("  scan done")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="preview, do not write")
    ap.add_argument("--config", type=Path,
                    default=REPO_ROOT / "configs" / "experiment_v1.json",
                    help="experiment config used by the 'scan' command")
    args = ap.parse_args()

    # Static safety checks on the REFERENCE preset vs the hard guard.
    for dev, _role, lo, hi in DEVICES:
        if dev not in REFERENCE:
            print("WARNING: no REFERENCE for %s" % dev)
        elif not (lo <= REFERENCE[dev] <= hi):
            print("WARNING: REFERENCE %s=%.3f A exceeds hard guard [%.0f, %.0f] — 'ref' will REFUSE it"
                  % (dev, REFERENCE[dev], lo, hi))

    client = make_client(require_rbac=True)
    print_menu(client, args.dry_run)

    while True:
        try:
            line = input("magnet> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        low = line.lower()
        if low in ("q", "quit", "exit"):
            break
        if low in ("m", "menu"):
            print_menu(client, args.dry_run)
            continue
        if low in ("r", "read"):
            read_all(client)
            continue
        if low == "ref":
            if not args.dry_run:
                if input("  set ALL magnets to reference? [y/N] ").strip().lower() != "y":
                    print("  cancelled")
                    continue
            set_reference(client, args.dry_run)
            continue
        if low == "scan":
            run_random_walk_scan(client, args.config, args.dry_run)
            continue
        parts = line.split()
        if len(parts) != 2:
            print('  usage: "<id> <current>"  e.g. "1 3.2"   (or ref / m / q)')
            continue
        try:
            idx = int(parts[0])
            current = float(parts[1])
        except ValueError:
            print("  bad input — need: <id:int> <current:float>")
            continue
        if not (1 <= idx <= len(DEVICES)):
            print("  id must be 1..%d" % len(DEVICES))
            continue
        set_one(client, DEVICES[idx - 1][0], current, dry_run=args.dry_run)

    print("bye")


if __name__ == "__main__":
    main()
