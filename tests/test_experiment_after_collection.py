"""Exercise the lab experiment's completion boundary without devices or analysis."""

from contextlib import ExitStack, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]

# The acquisition adapter imports CERN SDK types, but these tests never create
# a real client or device. Match the existing offline adapter tests.
for name in ("pyda", "pyrbac", "pyda_rda3"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["pyda"].SimpleClient = type("SimpleClient", (), {})
sys.modules["pyrbac"].AuthenticationListener = type("AuthenticationListener", (), {})
sys.modules["pyrbac"].LoginService = type("LoginService", (), {})
sys.modules["pyda_rda3"].RdaProvider = type("RdaProvider", (), {})

_spec = importlib.util.spec_from_file_location(
    "optidaq_experiment_after_collection_test", ROOT / "experiments/experiment_lab_v1.py"
)
experiment = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = experiment
_spec.loader.exec_module(experiment)


class FakeTap:
    def __init__(self, alias, device, client, **kwargs):
        self.device = alias
        self.fesa_device = device
        self.image_shape = (2, 2)
        self.image_dtype = "uint16"
        self.payload_nbytes = 8
        self.stamp_source = "image_ready_ns"
        self.probe_pulse_id = None
        self.error = None

    def probe(self):
        pass


class ExperimentAfterCollectionTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.config_path = self.root / "config.json"
        self.cfg = json.loads((ROOT / "configs/experiment_lab_v1.json").read_text())
        self.cfg["run"]["data_root"] = str(self.root)
        self.cfg["writer"]["configure_cameras"] = False
        self.cfg["dmd"]["count"] = 2
        self.cfg["postprocess"]["actions"] = ["data_analysis"]
        self.cfg["postprocess"]["validate_after_scan"] = False
        (self.root / self.cfg["run"]["runs_subdir"]).mkdir(parents=True)
        self.events = []
        self.stop_after_first = False
        self.acquisition_error = None

        test = self

        class FakeController:
            plan = [experiment.DmdPoint(0), experiment.DmdPoint(1)]
            frames_per_point = 1

            def preflight(self):
                pass

            def prepare(self):
                pass

            def apply_point(self, point):
                if test.acquisition_error:
                    raise test.acquisition_error
                return {}, None

            def dwell(self):
                pass

            def abort(self):
                pass

            def cleanup(self):
                test.events.append("hardware_cleanup")

            def actuator_metadata(self):
                return {}

            motor_metadata = actuator_metadata
            magnet_metadata = actuator_metadata

        class FakeFrameWaiter:
            def __init__(self, *, point_log, **kwargs):
                self.point_log = point_log

            def markers(self):
                return {}

            def wait(self, markers, item, extra=None):
                self.point_log.append(item.to_dict())
                if test.stop_after_first:
                    experiment.STOP_REQUESTED = True

        self.pipeline = Mock()
        self.pipeline.stop.side_effect = lambda timeout: self.events.append("close_files")
        self.client = Mock(return_value=object())
        self.action = Mock(side_effect=lambda *args, **kwargs: self.events.append("after_collection"))
        self.finalize = Mock(side_effect=self.finalize_dataset)
        patches = {
            "STOP_REQUESTED": False,
            "make_client": self.client,
            "SubscribingCameraTap": FakeTap,
            "PollingCameraTap": FakeTap,
            "collect_camera_source_provenance": Mock(return_value={}),
            "validate_tap_stamp_sources": Mock(),
            "DmdPatternScanController": FakeController,
            "make_controller": Mock(return_value=FakeController()),
            "run_preflight": Mock(return_value=[]),
            "make_save_backend": Mock(return_value=object()),
            "DAQPipeline": Mock(return_value=self.pipeline),
            "CounterFreezeWatchdog": Mock(),
            "FrameWaiter": FakeFrameWaiter,
            "write_png_gray": Mock(),
            "write_actuator_frame_index": Mock(side_effect=self.build_pattern_index),
            "finalize_png_dataset": self.finalize,
            "run_after_collection": self.action,
        }
        for name, value in patches.items():
            self.stack.enter_context(patch.object(experiment, name, value))
        self.stack.enter_context(patch.object(experiment.signal, "signal"))
        self.stack.enter_context(patch.object(sys, "argv", ["experiment_lab_v1.py", "--config", str(self.config_path)]))
        self.stack.enter_context(redirect_stdout(io.StringIO()))

    def build_pattern_index(self, run_dir, aliases):
        self.events.append("pattern_index")
        return run_dir / "actuator_frame_index.jsonl"

    def finalize_dataset(self, run_dir, **kwargs):
        self.events.append("dataset_database")
        return types.SimpleNamespace(db_path=run_dir / "dataset.db")

    def run_experiment(self):
        self.config_path.write_text(json.dumps(self.cfg))
        experiment.main()

    def test_action_runs_after_files_hardware_and_dataset_are_finished(self):
        self.run_experiment()
        self.assertEqual(self.events, ["close_files", "hardware_cleanup", "pattern_index",
                                      "dataset_database", "after_collection"])
        self.action.assert_called_once()
        args, kwargs = self.action.call_args
        self.assertTrue(args[0].is_dir())
        self.assertEqual(args[1], ["data_analysis"])
        self.assertEqual(kwargs, {"completed": True})

    def test_stop_between_points_is_not_success_even_without_an_exception(self):
        self.stop_after_first = True
        self.run_experiment()
        self.action.assert_called_once()
        self.assertEqual(self.action.call_args.kwargs, {"completed": False})
        self.assertEqual(self.events[:2], ["close_files", "hardware_cleanup"])

    def test_acquisition_failure_cleans_up_and_never_launches_action(self):
        self.acquisition_error = RuntimeError("acquisition failed")
        with self.assertRaisesRegex(RuntimeError, "acquisition failed"):
            self.run_experiment()
        self.assertEqual(self.events, ["close_files", "hardware_cleanup"])
        self.finalize.assert_not_called()
        self.action.assert_not_called()

    def test_setup_failure_after_dmd_preflight_still_cleans_up(self):
        experiment.run_preflight.return_value = [types.SimpleNamespace(
            ok=False, name="output_directory", message="write access denied",
        )]
        with self.assertRaisesRegex(RuntimeError, "preflight failed"):
            self.run_experiment()
        self.assertEqual(self.events, ["hardware_cleanup"])
        self.pipeline.start.assert_not_called()
        self.finalize.assert_not_called()
        self.action.assert_not_called()

    def test_stop_during_finalization_prevents_analysis(self):
        def stop_while_finalizing(run_dir, **kwargs):
            experiment.request_stop(None, None)
            return self.finalize_dataset(run_dir, **kwargs)

        self.finalize.side_effect = stop_while_finalizing
        self.run_experiment()
        self.action.assert_called_once()
        self.assertEqual(self.action.call_args.kwargs, {"completed": False})

    def test_writer_close_failure_never_launches_action(self):
        self.pipeline.stop.side_effect = RuntimeError("writer close failed")
        with self.assertRaisesRegex(RuntimeError, "writer close failed"):
            self.run_experiment()
        self.assertEqual(self.events, ["hardware_cleanup"])
        self.finalize.assert_not_called()
        self.action.assert_not_called()

    def test_finalizer_failure_never_launches_action(self):
        self.finalize.side_effect = RuntimeError("database failed")
        with self.assertRaisesRegex(RuntimeError, "database failed"):
            self.run_experiment()
        self.assertEqual(self.events, ["close_files", "hardware_cleanup", "pattern_index"])
        self.action.assert_not_called()

    def test_unknown_action_is_rejected_before_creating_a_client(self):
        self.cfg["postprocess"]["actions"] = ["misspelled_action"]
        with self.assertRaisesRegex(ValueError, "postprocess.actions"):
            self.run_experiment()
        self.client.assert_not_called()
        self.action.assert_not_called()


if __name__ == "__main__":
    unittest.main()
