#!/usr/bin/env python3
"""Validate CLEAR magnet actuation, one device at a time.

For each magnet: preflight (must be OK/ON/REMOTE), read its present setpoint + range,
set a SMALL in-range drift on the setpoint, confirm the write landed, watch the measured
current move by that drift (robust to any standing setpoint<->readback offset), then
RESTORE the original setpoint. Devices are done strictly one by one (restored before the
next), so at most one magnet is ever off its nominal value.

Writes SettingPPM with --commit. The default dry run reads state without writing.
PowRs422 recipe: write SettingPPM (no selector),
read Acquisition.currentAverage with context=SCT.USER.SETUP, health from Status.

Run on the CERN VM (same env as experiment_v1.py):
    ... python tests/validate_clear_magnets.py            # dry-run, shows the plan
    ... python tests/validate_clear_magnets.py --commit   # set + verify + restore

If interrupted (Ctrl-C) mid-device, that one magnet may be left at the test current;
re-run, or restore it manually.
"""

from __future__ import annotations

import argparse
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

# 3 quadrupoles (beam size) and 2 correctors (beam position).
DEVICES = [
    "CS.QFG0525",
    "CS.QDG0535",
    "CS.QFG0545",
    "CS.DHJ0505",
    "CS.DVJ0505",
]

READ_SELECTOR = "SCT.USER.SETUP"   # Acquisition is cycle-bound -> needs a selector
SET_CONTEXT = None                 # SettingPPM write uses no selector (config setting_selector="")
BUSY_BIT = 4                       # current_status bit 4 = BUSY


def _f(value) -> float:
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


def preflight(client, dev):
    st = get_value(client, "%s/Status" % dev)
    problems = []
    if optional_int_like(st.get("status")) != 0:
        problems.append("status=%s(need 0/OK)" % optional_int_like(st.get("status")))
    if optional_int_like(st.get("control")) != 5:
        problems.append("control=%s(need 5/REMOTE)" % optional_int_like(st.get("control")))
    if optional_int_like(st.get("mode")) != 3:
        problems.append("mode=%s(need 3/ON)" % optional_int_like(st.get("mode")))
    return problems


def is_busy(current_status) -> bool:
    return current_status is not None and bool(current_status & (1 << BUSY_BIT))


def write_current(client, dev, current, tol_abs, tol_rel) -> None:
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


def watch(client, dev, goal, tol, poll_s, timeout_s, baseline, verbose):
    """Poll measured current until within tol of goal (A) and not busy, or timeout.

    ``goal`` is the expected measured endpoint; ``baseline`` is the measured value before
    the move, so the printed drift = cur - baseline shows actuation directly.
    """
    t0 = time.time()
    final = None
    settled_t = None
    while time.time() - t0 < timeout_s:
        cur, cs = read_measured(client, dev)
        final = cur
        busy = is_busy(cs)
        if verbose:
            print("      t=%4.1fs  I=%+10.4f A  drift=%+8.4f  to_goal=%+8.4f  busy=%s"
                  % (time.time() - t0, cur, cur - baseline, cur - goal, busy))
        if not busy and abs(cur - goal) <= tol:
            settled_t = time.time() - t0
            break
        time.sleep(poll_s)
    return final, settled_t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", action="store_true", help="actually WRITE (default: dry-run)")
    ap.add_argument("--delta", type=float, default=0.5, help="test step from present setpoint (A)")
    ap.add_argument("--abs-tol", type=float, default=0.05, help="convergence tol, absolute (A)")
    ap.add_argument("--rel-tol", type=float, default=0.0, help="convergence tol, relative (fraction)")
    ap.add_argument("--poll", type=float, default=0.3, help="poll interval (s)")
    ap.add_argument("--timeout", type=float, default=10.0, help="settle timeout per move (s)")
    args = ap.parse_args()

    if abs(args.delta) > 5.0:
        print("WARNING: --delta %.3f A is large for a validation step." % args.delta)

    client = make_client(require_rbac=True)
    print("Magnet validation — %s\n" % ("COMMIT (writing)" if args.commit else "DRY-RUN (no writes)"))

    results = []
    for dev in DEVICES:
        print("=" * 72)
        print("DEVICE %s" % dev)
        try:
            problems = preflight(client, dev)
            sp = read_setpoint(client, dev)
            meas0, _ = read_measured(client, dev)
        except Exception as exc:
            print("  ERROR reading device: %s" % exc)
            results.append((dev, "READ-ERROR"))
            continue

        start = sp["current"]
        print("  start : setpoint=%+.4f A  measured=%+.4f A  range=[%.3f, %.3f]"
              % (start, meas0, sp["min"], sp["max"]))

        if problems:
            print("  SKIP (not safe to write): %s" % "; ".join(problems))
            results.append((dev, "SKIP:" + ",".join(problems)))
            continue

        gap = meas0 - start
        if abs(gap) > 0.2:
            print("  NOTE  : measured-setpoint gap %+.3f A — validation tracks measured DRIFT, not absolute" % gap)

        target = start + args.delta
        if target > sp["max"] or target < sp["min"]:
            target = start - args.delta
        target = min(max(target, sp["min"]), sp["max"])
        commanded = target - start              # actual drift after range-clamp
        goal = meas0 + commanded                # expected measured endpoint
        tol = max(args.abs_tol, args.rel_tol * abs(target))
        print("  plan  : set -> %+.4f A  (drift %+.3f A; expect measured %+.4f -> %+.4f, tol %.4f)"
              % (target, commanded, meas0, goal, tol))

        if abs(commanded) <= tol:
            print("  SKIP: drift within tolerance / no room in range — not a meaningful test")
            results.append((dev, "SKIP:no-room"))
            continue

        if not args.commit:
            print("  DRY-RUN: not writing.")
            results.append((dev, "DRY-RUN"))
            continue

        verdict = "FAIL"
        try:
            write_current(client, dev, target, sp["tol_abs"], sp["tol_rel"])
            sp_after = read_setpoint(client, dev)["current"]
            wrote_ok = abs(sp_after - target) <= max(1e-3, tol)
            print("  write : setpoint readback=%+.4f A  %s"
                  % (sp_after, "OK" if wrote_ok else "MISMATCH"))
            final, settled_t = watch(client, dev, goal, tol, args.poll, args.timeout,
                                     baseline=meas0, verbose=True)
            drift = final - meas0
            converged = settled_t is not None
            verdict = "PASS" if (wrote_ok and converged) else "FAIL"
            print("  result: measured drift %+.4f A (wanted %+.4f, miss %+.4f)  %s%s"
                  % (drift, commanded, drift - commanded, verdict,
                     "" if settled_t is None else "  (settled %.1fs)" % settled_t))
        except Exception as exc:
            print("  ERROR during set/watch: %s" % exc)
            verdict = "SET-ERROR"
        finally:
            try:
                write_current(client, dev, start, sp["tol_abs"], sp["tol_rel"])
                back, _ = watch(client, dev, meas0, tol, args.poll, args.timeout,
                                baseline=meas0, verbose=False)
                ok = abs(back - meas0) <= tol
                print("  restore: setpoint=%+.4f A  measured=%+.4f A  %s"
                      % (start, back, "OK" if ok else "NOT BACK"))
                if not ok:
                    verdict += "+RESTORE-OFF"
            except Exception as exc:
                print("  WARNING: restore FAILED: %s  (device left at test current!)" % exc)
                verdict += "+RESTORE-FAIL"
        results.append((dev, verdict))

    print("\n" + "=" * 72)
    print("SUMMARY")
    for dev, verdict in results:
        print("  %-14s %s" % (dev, verdict))
    if not args.commit:
        print("\nDry-run only. Re-run with --commit to set + verify + restore.")


if __name__ == "__main__":
    main()
