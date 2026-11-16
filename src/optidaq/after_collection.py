"""Run optional actions against a finished dataset, without hardware access.

Only the standard library is imported here. Each action runs in a separate
Python process and can also be called directly on an existing run directory.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


_ACTIONS = {"data_analysis": "optidaq.analysis.analyze_run"}


def _action_names(value) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("postprocess.actions must be a list")
    if any(not isinstance(name, str) or name not in _ACTIONS for name in value):
        raise ValueError("postprocess.actions supports only: %s" % ", ".join(_ACTIONS))
    if len(set(value)) != len(value):
        raise ValueError("postprocess.actions must not contain duplicate names")
    return list(value)


def _has_stats(hooks) -> bool:
    return isinstance(hooks, list) and any(
        hook == "stats" or isinstance(hook, Mapping) and hook.get("name") == "stats"
        for hook in hooks
    )


def configured_actions(cfg) -> list[str]:
    """Check the selected actions before camera or actuator setup begins."""
    postprocess = cfg.get("postprocess", {})
    if not isinstance(postprocess, Mapping):
        raise ValueError("postprocess must be an object")
    actions = _action_names(postprocess.get("actions", []))
    if "data_analysis" not in actions:
        return actions

    output = cfg.get("output", {})
    png = output.get("png", {}) if isinstance(output, Mapping) else {}
    actuator = cfg.get("actuator", {})
    mode = actuator.get("mode", "motor") if isinstance(actuator, Mapping) else actuator
    dmd = cfg.get("dmd", {})
    alignment = png.get("sync_alignment", {}) if isinstance(png, Mapping) else {}
    hooks = png.get("metadata_hooks", []) if isinstance(png, Mapping) else []
    if isinstance(hooks, Mapping):
        cameras = cfg.get("cameras", [])
        has_stats = bool(cameras) and all(_has_stats(hooks.get(str(camera), [])) for camera in cameras)
    else:
        has_stats = _has_stats(hooks)
    requirements = (
        (isinstance(output, Mapping) and str(output.get("format", "raw")).lower() == "png", "output.format=png"),
        (str(mode).lower() in ("dmd", "dmd_pattern_scan"), "actuator.mode=dmd"),
        (isinstance(dmd, Mapping) and bool(dmd.get("save_patterns", True)), "dmd.save_patterns=true"),
        (postprocess.get("build_aligned_index", True) is True, "postprocess.build_aligned_index=true"),
        (isinstance(alignment, Mapping) and alignment.get("source") == "image_ready_ns",
         "output.png.sync_alignment.source=image_ready_ns"),
        (has_stats, "stats in output.png.metadata_hooks"),
    )
    missing = [description for passed, description in requirements if not passed]
    if missing:
        raise ValueError("data_analysis currently requires " + "; ".join(missing))
    return actions


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_paths(out: Path) -> None:
    if out.is_symlink() or any((out / name).is_symlink()
                               for name in ("action.log", "action_status.json")):
        raise ValueError("Action output paths must not be symlinks")


def _save_status(out: Path, result: dict) -> None:
    _check_paths(out)
    (out / "action_status.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def run_after_collection(run_dir, actions, *, completed: bool) -> list[dict]:
    """Run selected actions in order; keep their failures separate from collection.

    The caller must confirm all planned points finished, writers closed and
    required dataset files were finalized before passing ``completed=True``.
    A cancellation is recorded and re-raised. Other action errors are reported
    and returned, so they cannot undo or relabel the acquisition result.
    """
    names = _action_names(actions)
    if not names:
        return []
    run = Path(run_dir).resolve()
    results = []
    for name in names:
        out = run / name
        result = {"action": name, "status": "running", "started_utc": _now(),
                  "finished_utc": None, "returncode": None, "error": None}
        try:
            if not run.is_dir():
                raise ValueError("Run directory does not exist: %s" % run)
            _check_paths(out)
            out.mkdir(exist_ok=True)
            if not completed:
                result.update(status="skipped", error="Collection did not complete or required dataset files are not ready")
                (out / "action.log").write_text(result["error"] + "\n", encoding="utf-8")
            else:
                _save_status(out, result)
                print("After collection: %s running; log: %s" % (name, out / "action.log"), flush=True)
                env = os.environ.copy()
                source_root = str(Path(__file__).resolve().parents[1])
                env["PYTHONPATH"] = os.pathsep.join(filter(None, (source_root, env.get("PYTHONPATH", ""))))
                with (out / "action.log").open("w", encoding="utf-8") as log:
                    process = subprocess.run(
                        [sys.executable, "-m", _ACTIONS[name], str(run)],
                        stdout=log, stderr=subprocess.STDOUT, env=env, check=False,
                    )
                result["returncode"] = process.returncode
                if process.returncode == 0:
                    result["status"] = "success"
                else:
                    result.update(status="failed", error="Action exited with code %d; see action.log" % process.returncode)
        except KeyboardInterrupt:
            result.update(status="interrupted", error="Action interrupted", finished_utc=_now())
            try:
                _save_status(out, result)
            except Exception as exc:
                print("WARNING: could not save %s status: %s" % (name, exc))
            raise
        except Exception as exc:
            result.update(status="failed", error=str(exc))
        result["finished_utc"] = _now()
        try:
            _save_status(out, result)
        except Exception as exc:
            result.update(status="failed", error="%s; could not save action status: %s" % (result["error"] or "Action finished", exc))
        results.append(result)
        print("After collection: %s %s%s" % (
            name, result["status"], ": " + result["error"] if result["error"] else "",
        ))
    return results
