"""After-collection actions operate on saved files without controlling hardware."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from optidaq.after_collection import configured_actions, run_after_collection


ROOT = Path(__file__).resolve().parents[1]
MODULE = "optidaq.after_collection"


class ActionConfigurationTests(unittest.TestCase):
    def config(self):
        cfg = json.loads((ROOT / "configs/experiment_lab_v1.json").read_text())
        cfg["postprocess"]["actions"] = ["data_analysis"]
        return cfg

    def test_missing_or_empty_actions_leave_collection_independent(self):
        self.assertEqual(configured_actions({}), [])
        self.assertEqual(configured_actions({"postprocess": {"actions": []}}), [])

    def test_current_profile_and_named_stats_hook_are_supported(self):
        cfg = self.config()
        original = deepcopy(cfg)
        self.assertEqual(configured_actions(cfg), ["data_analysis"])
        self.assertEqual(cfg, original)
        cfg["output"]["png"]["metadata_hooks"] = [{"name": "stats"}]
        self.assertEqual(configured_actions(cfg), ["data_analysis"])

    def test_existing_actuator_aliases_and_defaults_remain_supported(self):
        for actuator in ("dmd", "DMD_PATTERN_SCAN", {"mode": "dmd_pattern_scan"}):
            with self.subTest(actuator=actuator):
                cfg = self.config()
                cfg["actuator"] = actuator
                cfg["output"]["format"] = "PNG"
                del cfg["dmd"]["save_patterns"]
                self.assertEqual(configured_actions(cfg), ["data_analysis"])

    def test_per_camera_metadata_hooks_require_stats_for_each_active_camera(self):
        cfg = self.config()
        cfg["output"]["png"]["metadata_hooks"] = {
            "cam1": ["stats"], "cam2": [{"name": "stats"}],
        }
        self.assertEqual(configured_actions(cfg), ["data_analysis"])
        for hooks in ({"cam1": ["stats"]}, {"cam1": ["stats"], "cam2": []}):
            with self.subTest(hooks=hooks):
                cfg["output"]["png"]["metadata_hooks"] = hooks
                with self.assertRaises(ValueError):
                    configured_actions(cfg)

    def test_invalid_action_selection_is_rejected_before_collection(self):
        for actions in ("data_analysis", {}, 1, [1], ["unknown"],
                        ["data_analysis", "data_analysis"]):
            with self.subTest(actions=actions):
                cfg = self.config()
                cfg["postprocess"]["actions"] = actions
                with self.assertRaises(ValueError):
                    configured_actions(cfg)

    def test_analyzer_requirements_are_checked_before_collection(self):
        changes = (
            (("output", "format"), "raw"),
            (("actuator", "mode"), "noop"),
            (("dmd", "save_patterns"), False),
            (("postprocess", "build_aligned_index"), False),
            (("output", "png", "sync_alignment", "source"), "acq_stamp_ns"),
            (("output", "png", "metadata_hooks"), []),
        )
        for keys, value in changes:
            with self.subTest(setting=".".join(keys)):
                cfg = self.config()
                target = cfg
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                with self.assertRaises(ValueError):
                    configured_actions(cfg)


class AfterCollectionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        self.original = self.run_dir / "original.png"
        self.original.write_bytes(b"original acquisition bytes")
        self.output = self.run_dir / "data_analysis"
        quiet = patch("builtins.print")
        quiet.start()
        self.addCleanup(quiet.stop)

    def read_status(self):
        return json.loads((self.output / "action_status.json").read_text())

    def assert_original_unchanged(self):
        self.assertEqual(self.original.read_bytes(), b"original acquisition bytes")

    def test_disabled_actions_create_nothing(self):
        with patch(MODULE + ".subprocess.run") as launch:
            self.assertEqual(run_after_collection(self.run_dir, [], completed=True), [])
        launch.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_incomplete_collection_does_not_launch_analysis(self):
        with patch(MODULE + ".subprocess.run") as launch:
            results = run_after_collection(self.run_dir, ["data_analysis"], completed=False)
        launch.assert_not_called()
        self.assertEqual(results[0]["status"], "skipped")
        self.assertEqual(self.read_status()["status"], "skipped")
        self.assert_original_unchanged()

    def test_complete_collection_launches_separate_python_with_run_path(self):
        observed = {}

        def child(command, **kwargs):
            observed.update(command=command, kwargs=kwargs)
            self.assertEqual(self.read_status()["status"], "running")
            kwargs["stdout"].write("analysis completed\n")
            return subprocess.CompletedProcess(command, 0)

        with patch.dict(os.environ, {"PYTHONPATH": "/existing/python/path"}), \
                patch(MODULE + ".subprocess.run", side_effect=child):
            results = run_after_collection(self.run_dir, ["data_analysis"], completed=True)
        self.assertEqual(observed["command"], [sys.executable, "-m",
                         "optidaq.analysis.analyze_run", str(self.run_dir.resolve())])
        self.assertFalse(observed["kwargs"].get("shell", False))
        child_paths = observed["kwargs"]["env"]["PYTHONPATH"].split(os.pathsep)
        self.assertIn(str(ROOT / "src"), child_paths)
        self.assertIn("/existing/python/path", child_paths)
        self.assertEqual(results[0]["status"], "success")
        self.assertEqual(self.read_status()["returncode"], 0)
        self.assertIn("analysis completed", (self.output / "action.log").read_text())
        self.assert_original_unchanged()

    def test_analysis_exit_failure_does_not_become_collection_failure(self):
        with patch(MODULE + ".subprocess.run", return_value=subprocess.CompletedProcess([], 1)):
            results = run_after_collection(self.run_dir, ["data_analysis"], completed=True)
        self.assertEqual(results[0]["status"], "failed")
        self.assertEqual(self.read_status()["returncode"], 1)
        self.assert_original_unchanged()

    def test_launch_error_is_recorded_and_original_files_are_preserved(self):
        with patch(MODULE + ".subprocess.run", side_effect=OSError("cannot start analysis")):
            results = run_after_collection(self.run_dir, ["data_analysis"], completed=True)
        self.assertEqual(results[0]["status"], "failed")
        self.assertIn("cannot start analysis", self.read_status()["error"])
        self.assert_original_unchanged()

    def test_user_interrupt_is_recorded_and_propagated(self):
        with patch(MODULE + ".subprocess.run", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                run_after_collection(self.run_dir, ["data_analysis"], completed=True)
        self.assertEqual(self.read_status()["status"], "interrupted")
        self.assert_original_unchanged()

    def test_output_directory_symlink_is_not_followed(self):
        outside = self.root / "outside"
        outside.mkdir()
        sentinel = outside / "sentinel"
        sentinel.write_text("keep")
        self.output.symlink_to(outside, target_is_directory=True)
        with patch(MODULE + ".subprocess.run") as launch:
            results = run_after_collection(self.run_dir, ["data_analysis"], completed=True)
        launch.assert_not_called()
        self.assertEqual(results[0]["status"], "failed")
        self.assertEqual(list(outside.iterdir()), [sentinel])
        self.assertEqual(sentinel.read_text(), "keep")

    def test_output_file_symlinks_are_not_overwritten(self):
        self.output.mkdir()
        outside = self.root / "outside.txt"
        outside.write_text("keep")
        for name in ("action.log", "action_status.json"):
            with self.subTest(name=name):
                for path in self.output.iterdir():
                    path.unlink()
                (self.output / name).symlink_to(outside)
                with patch(MODULE + ".subprocess.run") as launch:
                    results = run_after_collection(self.run_dir, ["data_analysis"], completed=True)
                launch.assert_not_called()
                self.assertEqual(results[0]["status"], "failed")
                self.assertEqual(outside.read_text(), "keep")

    def test_runner_import_works_without_analysis_or_hardware_packages(self):
        code = (
            "import sys; import optidaq.after_collection; "
            "assert not any(n == p or n.startswith(p + '.') "
            "for n in sys.modules for p in "
            "('numpy', 'PIL', 'matplotlib', 'optidaq.analysis', 'optidaq.extensions'))"
        )
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        completed = subprocess.run([sys.executable, "-S", "-c", code], env=env,
                                   text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
