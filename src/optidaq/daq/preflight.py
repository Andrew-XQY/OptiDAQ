"""Generic standard-library preflight checks for DAQ setup."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Callable, Sequence


@dataclass(frozen=True)
class PreflightResult:
    """Result from one generic preflight check."""

    name: str
    ok: bool
    message: str = ""


PreflightCheck = Callable[[], PreflightResult]


def run_preflight(checks: Sequence[PreflightCheck]) -> list[PreflightResult]:
    """Run checks in order, converting raised exceptions into failures."""

    results: list[PreflightResult] = []
    for check in checks:
        try:
            results.append(check())
        except BaseException as exc:
            results.append(
                PreflightResult(
                    name=getattr(check, "__name__", check.__class__.__name__),
                    ok=False,
                    message="raised %s: %s" % (exc.__class__.__name__, exc),
                )
            )
    return results


def make_write_access_check(path: Path | str) -> PreflightCheck:
    """Return a check that creates and removes a small temp file in ``path``."""

    directory = Path(path)
    name = "write_access:%s" % directory

    def check() -> PreflightResult:
        tmp_path = None
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix=".optidaq_preflight_",
                suffix=".tmp",
                dir=str(directory),
                delete=False,
            ) as handle:
                tmp_path = Path(handle.name)
                handle.write(b"ok")
                handle.flush()
                os.fsync(handle.fileno())
            if tmp_path is not None:
                tmp_path.unlink()
            return PreflightResult(name=name, ok=True, message="write access ok")
        except BaseException as exc:
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            return PreflightResult(
                name=name,
                ok=False,
                message="%s: %s" % (exc.__class__.__name__, exc),
            )

    check.__name__ = name
    return check
