#!/usr/bin/env python3
"""Control the CS.BTV0520 filter wheel from the CERN acc-py environment.

    python experiments/filter_wheel_control.py
    python experiments/filter_wheel_control.py 1000

Accepts menu index 0..3 or attenuation factor 1, 10, 100, 1000.
Interactive mode continues until q, quit or exit.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


DEVICE = "CS.BTV0520"
FILTERS = (
    {"index": 0, "label": "T100%", "factor": 1},
    {"index": 1, "label": "T10%", "factor": 10},
    {"index": 2, "label": "T1%", "factor": 100},
    {"index": 3, "label": "T0.1%", "factor": 1000},
)


def filter_name(index: int) -> str:
    for item in FILTERS:
        if item["index"] == index:
            return "%s (attenuation x%s)" % (item["label"], item["factor"])
    return "unknown index %s" % index


def parse_choice(raw: str) -> int:
    choice = raw.strip().lower().replace("x", "")
    if choice.startswith("t"):
        choice = choice[1:]
    choice = choice.replace("%", "")

    for item in FILTERS:
        if choice == str(item["index"]):
            return int(item["index"])
        if choice == str(item["factor"]):
            return int(item["index"])
        if choice == item["label"].lower().replace("t", "").replace("%", ""):
            return int(item["index"])
    raise RuntimeError("unknown filter choice %r" % raw)


def print_menu(current: int) -> None:
    print("CS.BTV0520 filter wheel")
    print("current: %d -> %s" % (current, filter_name(current)))
    print("")
    print("Choose filter:")
    for item in FILTERS:
        marker = "*" if item["index"] == current else " "
        print(
            "  %s %d  %-6s  attenuation x%s"
            % (marker, item["index"], item["label"], item["factor"])
        )
    print("")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "choice",
        nargs="?",
        help="filter index 0..3 or attenuation factor 1, 10, 100, 1000",
    )
    args = parser.parse_args()

    from optidaq.extensions.cern_pyda import get_value, make_client, plain, set_value  # noqa: E402

    def read_setting() -> dict:
        return get_value(client, "%s/Setting" % DEVICE)

    def current_filter(setting: dict) -> int:
        return int(plain(setting["filterSelect"]))

    def apply_filter(target: int, current: int) -> int:
        if target == current:
            print("already at %d -> %s" % (target, filter_name(target)))
            return current

        print("setting filter: %d -> %s" % (target, filter_name(target)))
        set_value(client, "%s/Setting" % DEVICE, {"filterSelect": np.int32(target)})
        time.sleep(0.7)

        after_filter = current_filter(read_setting())
        print("readback: %d -> %s" % (after_filter, filter_name(after_filter)))
        if after_filter != target:
            raise RuntimeError(
                "filter readback mismatch: requested %d got %d"
                % (target, after_filter)
            )
        return after_filter

    client = make_client(require_rbac=True)
    current = current_filter(read_setting())

    if args.choice is not None:
        print_menu(current)
        apply_filter(parse_choice(args.choice), current)
        return

    while True:
        print_menu(current)
        raw_choice = input("filter> ").strip()
        if raw_choice.lower() in ("q", "quit", "exit"):
            print("leaving filter at %d -> %s" % (current, filter_name(current)))
            return
        if not raw_choice:
            continue
        current = apply_filter(parse_choice(raw_choice), current)


if __name__ == "__main__":
    main()
