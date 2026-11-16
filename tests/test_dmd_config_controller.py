"""DMD profile and controller checks without ALP hardware or XFlow imports."""

from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from optidaq.actuators.dmd_pattern_scan import (
    DmdPatternScanController,
    make_dmd_plan,
    validate_dmd_config,
)


MODULE = "optidaq.actuators.dmd_pattern_scan"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "experiment_lab_v1.json"


class FakeDmd:
    """Expose native geometry only after opening, like the real driver."""

    def __init__(self, **settings):
        self.settings = settings
        self.height = None
        self.width = None
        self.opened = False
        self.closed = False
        self.checked = []
        self.shown = []

    def open(self):
        self.opened = True
        self.height, self.width = 720, 1280

    def check_frame(self, frame):
        if not self.opened or frame.shape != (self.height, self.width) or frame.dtype != np.uint8:
            raise ValueError("bad native DMD frame")
        self.checked.append(np.array(frame, copy=True))

    def show(self, frame):
        self.check_frame(frame)
        self.shown.append(np.array(frame, copy=True))

    def close(self):
        self.closed = True
        self.opened = False


class DmdConfigControllerTest(unittest.TestCase):
    def config(self):
        return json.loads(CONFIG_PATH.read_text())

    def test_main_config_is_complete_sgm_dmd_profile(self):
        cfg = self.config()
        self.assertEqual(cfg["actuator"]["mode"], "dmd")
        self.assertTrue(cfg["postprocess"]["build_aligned_index"])
        self.assertNotIn("<FILL", json.dumps(cfg["dmd"]))
        validate_dmd_config(cfg["dmd"])
        plan = make_dmd_plan(cfg["dmd"])
        self.assertEqual([point.index for point in plan], list(range(10)))
        self.assertEqual(cfg["scan"]["frames_per_point"], 10)
        self.assertEqual(len(cfg["dmd"]["streams"]), 1)
        stream = cfg["dmd"]["streams"][0]
        self.assertEqual(stream["name"], "dynamic_patterns")
        self.assertEqual(stream["params"]["canvas_size"], [256, 256])
        self.assertEqual(stream["params"]["num_gaussians"], 1)
        self.assertEqual(stream["params"]["update"]["intensity_range"], [0.07, 0.16])
        self.assertEqual(stream["transforms"][-1], {
            "name": "to_uint8", "params": {"max_value": 0.16},
        })
        self.assertEqual(cfg["dmd"]["device"]["picture_time_us"], 25000)
        self.assertEqual([cfg["camera_setting"][camera]["cameraExposureTimeUs"]
                          for camera in cfg["cameras"]], [50000, 50000])

    def test_native_display_preserves_first_pattern_and_source_config(self):
        cfg = self.config()
        # Keep this test about the native-size behavior rather than one physical
        # DMD's dimensions or a particular generator installation.
        cfg["dmd"]["display_transforms"] = [
            {"name": "upscale", "params": {"factor": 1}},
            {"name": "fit_center", "params": {"height": None, "width": None}},
            {"name": "rotate", "params": {"degrees": 45}},
        ]
        original_config = deepcopy(cfg)
        patterns = [np.full((256, 256), value, dtype=np.uint8) for value in (73, 151)]
        originals = [pattern.copy() for pattern in patterns]
        items = iter((0, str(index), pattern) for index, pattern in enumerate(patterns))
        with patch(MODULE + ".AlpDmd", FakeDmd), \
             patch(MODULE + ".build_pattern_stream", return_value=items) as build_stream, \
             patch(MODULE + ".time.sleep") as sleep:
            controller = DmdPatternScanController(None, cfg)
            with self.assertRaisesRegex(RuntimeError, "preflight"):
                controller.apply_point(controller.plan[0])
            controller.preflight()
            device = controller.device
            build_stream.assert_called_once_with(cfg["dmd"]["streams"], seed=cfg["dmd"]["seed"])
            self.assertEqual(device.shown, [])
            self.assertEqual(device.checked[0].shape, (720, 1280))
            for index, point in enumerate(controller.plan[:2]):
                info, source = controller.apply_point(point)
                self.assertEqual(info["dmd_source"], "dynamic_patterns")
                self.assertEqual(info["dmd_source_id"], str(index))
                np.testing.assert_array_equal(source, originals[index])
                shown = device.shown[index]
                self.assertEqual(shown.shape, (720, 1280))
                self.assertEqual(shown.dtype, np.dtype("uint8"))
                self.assertEqual(set(np.unique(shown)), {0, (73, 151)[index]})
                rows, cols = np.nonzero(shown)
                # Rotation of a centered 256-square spans more than 256 pixels
                # while remaining centered in the device's native frame.
                self.assertGreater(int(np.ptp(rows)), 256)
                self.assertGreater(int(np.ptp(cols)), 256)
                self.assertAlmostEqual(float(rows.mean()), 359.5, places=1)
                self.assertAlmostEqual(float(cols.mean()), 639.5, places=1)
            self.assertEqual(sleep.call_count, 2)
            sleep.assert_called_with(cfg["dmd"]["settle_s"])
            controller.cleanup()
            self.assertTrue(device.closed)
        self.assertEqual(cfg, original_config)
        for pattern, original in zip(patterns, originals):
            np.testing.assert_array_equal(pattern, original)

    def test_preflight_display_failure_closes_device(self):
        cfg = self.config()
        cfg["dmd"]["display_transforms"] = [
            {"name": "fit_center", "params": {"height": 12, "width": 16}},
        ]
        items = iter([(0, None, np.ones((256, 256), dtype=np.uint8))])
        with patch(MODULE + ".AlpDmd", FakeDmd), \
             patch(MODULE + ".build_pattern_stream", return_value=items):
            controller = DmdPatternScanController(None, cfg)
            with self.assertRaisesRegex(ValueError, "bad native DMD frame"):
                controller.preflight()
            self.assertTrue(controller.device.closed)
            self.assertEqual(controller.device.shown, [])


if __name__ == "__main__":
    unittest.main()
