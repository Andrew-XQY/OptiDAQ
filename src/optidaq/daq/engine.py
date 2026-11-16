"""Tiny upstream control-plane loop.

The engine never sees frames, never saves data, and never performs online
pairing. It only applies physical setpoints from a plan and lets an injected
``dwell`` callable decide how long each state remains meaningful.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional


def run_engine(
    plan: Iterable[Any],
    apply: Callable[[Any], None],
    dwell: Callable[[], None],
    abort: Optional[Callable[[], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> None:
    """Run a simple apply/dwell control plan.

    Cameras free-run independently of this loop. ``dwell`` may inspect injected
    counters or health metrics, but this engine does not inspect image payloads,
    save frames, or derive pairings.
    """

    try:
        for sp in plan:
            if should_stop is not None and should_stop():
                break
            apply(sp)
            dwell()
    finally:
        if abort is not None:
            abort()
