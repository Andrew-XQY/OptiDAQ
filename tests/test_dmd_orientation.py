"""Orientation checks against asymmetric markers; no device connections."""
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from optidaq.data.transforms import build_pipeline, flip_vertical


REPO = Path(__file__).resolve().parents[1]


def helper_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO / "tests" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DmdOrientationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.png = helper_module("png_orientation_helper", "dmd_show_png_left_half.py")

    @staticmethod
    def marker():
        source = np.zeros((9, 9), dtype=np.uint8)
        source[4, 7] = 201       # right
        source[1, 4] = 127       # above
        source[:2, :2] = 53      # corner: lost if rotated before native padding
        return source

    def test_original_api_keeps_identity_and_center_padding(self):
        source = self.marker()
        expected = np.zeros((19, 29), dtype=np.uint8)
        expected[5:14, 10:19] = source
        np.testing.assert_array_equal(self.png.to_dmd_frame(source, 19, 29, 1), expected)

    def test_correction_cancels_measured_optical_parity_and_angle(self):
        source = self.marker()
        frame = self.png.to_dmd_frame(source, 19, 29, 1, rotation_deg=45, flip_vertical=True)
        # Idealized measured optical mapping: x-right/y-down, +45 deg clockwise
        # with negative parity. Test marker positions, not another raster helper.
        optical = np.array([[1, 1], [1, -1]], dtype=float) / np.sqrt(2)
        for intensity in (201, 127, 53):
            original_y, original_x = np.nonzero(source == intensity)
            displayed_y, displayed_x = np.nonzero(frame == intensity)
            self.assertGreater(len(displayed_x), 0)
            camera_offset = optical @ (np.array([displayed_x.mean(), displayed_y.mean()]) - [14, 9])
            source_offset = np.array([original_x.mean(), original_y.mean()]) - [4, 4]
            np.testing.assert_allclose(camera_offset, source_offset, atol=.6)
        self.assertEqual(set(np.unique(frame)), {0, 53, 127, 201})
        self.assertTrue(frame.flags.c_contiguous)

    def test_lab_sgm_and_standalone_png_apply_same_correction_once(self):
        config = json.loads((REPO / "configs/experiment_lab_v1.json").read_text())
        specs = config["dmd"]["display_transforms"]
        names = [step["name"] for step in specs]
        self.assertEqual(names, ["upscale", "fit_center", "rotate", "flip_vertical"])
        self.assertEqual(specs[-2]["params"]["degrees"], self.png.CONFIG["rotation_deg"])
        self.assertTrue(self.png.CONFIG["flip_vertical"])
        sgm = helper_module("sgm_orientation_helper", "dmd_show_random_sgm.py")
        frame = sgm.display_pipeline(config, 19, 29)(self.marker())
        expected = self.png.to_dmd_frame(self.marker(), 19, 29, 1,
            rotation_deg=self.png.CONFIG["rotation_deg"], flip_vertical=self.png.CONFIG["flip_vertical"])
        np.testing.assert_array_equal(frame, expected)
        config, _ = sgm.load_test_config(sgm.CONFIG)
        names = [step["name"] for step in config["dmd"]["display_transforms"]]
        self.assertEqual(names.count("rotate"), 1)
        self.assertEqual(names.count("flip_vertical"), 1)

    def test_vertical_flip_preserves_rows_and_rejects_nonimages(self):
        image = np.array([[11, 12], [21, 22], [31, 32]], dtype=np.uint16)
        expected = np.array([[31, 32], [21, 22], [11, 12]], dtype=np.uint16)
        np.testing.assert_array_equal(build_pipeline(["flip_vertical"])(image), expected)
        with self.assertRaises(ValueError):
            flip_vertical(np.ones((3, 2, 2)))


if __name__ == "__main__":
    unittest.main()
