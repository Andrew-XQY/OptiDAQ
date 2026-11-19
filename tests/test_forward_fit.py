"""Forward fit and analysis pipeline on synthetic datasets whose fibre images are
linear in the beam light."""

import csv
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest

import numpy as np
from PIL import Image
from scipy.optimize import nnls

from optidaq.analysis import pipeline
from optidaq.analysis.config import fill, strip
from optidaq.analysis.figures import fit_samples
from optidaq.analysis.forward_fit import FITS, SELECTORS, FibreMetrics, beam_centre, locate_spot, nnls_gram
from optidaq.analysis.forward_fit.run import resolve
from optidaq.data.png_io import write_png_gray

SIZE, GRID, AMPLITUDE = 48, 8, 20000.0
BACKGROUND = {"cam2": 100.0, "cam3": 50.0, "cam1": 70.0}
SPOTS = [(2.5 + 6 * j, 2.5 + 6 * i) for i in range(8) for j in range(8)]
REPO = Path(__file__).resolve().parents[1]


def gaussian(x0, y0, sigma=0.8, size=SIZE):
    yy, xx = np.indices((size, size))
    return np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))


def frames(light):
    """Beam camera sees the light; the two fibre cameras see two linear maps of it."""
    images = {"cam2": light, "cam3": light[::-1, :], "cam1": 0.5 * light.T}
    return {c: np.rint(BACKGROUND[c] + AMPLITUDE * v).astype(np.uint16) for c, v in images.items()}


def write_dataset(folder, samples):
    """samples: (sample_id, images by camera, sync offsets by camera, roi_energy_fraction)."""
    folder.mkdir(parents=True)
    con = sqlite3.connect(folder / "dataset.db")
    con.execute(
        "CREATE TABLE images (camera TEXT, png_path TEXT, run_id TEXT, sample_id INTEGER, sync_time_ns INTEGER,"
        " acq_stamp_ns INTEGER, stat_max REAL, stat_mean REAL, roi_energy_fraction REAL)"
    )
    for sample_id, images, offsets, roi in samples:
        t = sample_id * 1_000_000_000
        for cam, img in images.items():
            rel = "%s/%05d.png" % (cam, sample_id)
            write_png_gray(folder / rel, img)
            con.execute(
                "INSERT INTO images VALUES (?, ?, 'run', ?, ?, ?, ?, ?, ?)",
                (cam, rel, sample_id, t + offsets.get(cam, 0), t, float(img.max()), float(img.mean()), roi),
            )
    con.commit()
    con.close()


def write_datasets(root):
    """A background, a laser scan and a beam run; returns the config sections for them."""
    empty = frames(np.zeros((SIZE, SIZE)))
    write_dataset(root / "background", [(i, empty, {}, 1.0) for i in range(3)])

    basis = [(i, frames(gaussian(*xy)), {}, 1.0) for i, xy in enumerate(SPOTS)]
    basis += [(100 + i, empty, {}, 1.0) for i in range(6)]                       # no spot: dropped by otsu
    basis.append((200, frames(gaussian(*SPOTS[9])), {"cam3": 5_000_000}, 1.0))   # out of sync: dropped
    partial = frames(gaussian(*SPOTS[10]))
    del partial["cam1"]
    basis.append((201, partial, {}, 1.0))                                        # camera missing: dropped
    write_dataset(root / "basis", basis)

    rng = np.random.default_rng(2)
    target = []
    for i in range(6):
        chosen = rng.choice(len(SPOTS), 4, replace=False)
        light = sum(rng.uniform(0.3, 1.0) * gaussian(*SPOTS[k]) for k in chosen)
        target.append((i, frames(light), {}, 0.5 if i == 5 else 1.0))           # last: dropped by where
    target += [(10, empty, {}, 1.0), (11, empty, {}, 1.0)]                       # beam off: dropped by otsu
    write_dataset(root / "target", target)

    minus = [{"name": "subtract_background", "params": {"dataset": "background"}}]
    return {
        "background": {"path": str(root / "background")},
        "basis": {"path": str(root / "basis"), "otsu": "cam2.stat_max", "preprocess": minus},
        "target": {"path": str(root / "target"), "otsu": "cam2.stat_mean",
                   "where": "cam2.roi_energy_fraction >= 0.8", "preprocess": minus},
    }


class ModelTest(unittest.TestCase):
    def test_nnls_gram_matches_scipy(self):
        rng = np.random.default_rng(0)
        a, b = rng.random((40, 12)), rng.random(40) - 0.2
        np.testing.assert_allclose(nnls_gram(a.T @ a, a.T @ b), nnls(a, b)[0], atol=1e-7)

    def test_locate_spot_is_not_pulled_to_frame_centre(self):
        rng = np.random.default_rng(1)
        true = np.array([70.3, 20.6])
        frame = 1000.0 * gaussian(*true, sigma=2.0, size=96) + rng.normal(0.0, 5.0, (96, 96))
        position, image = locate_spot(frame, half=12, far_px=30)
        clipped = np.maximum(frame, 0.0)
        yy, xx = np.indices(frame.shape)
        moment = np.array([(xx * clipped).sum(), (yy * clipped).sum()]) / clipped.sum()
        self.assertLess(np.hypot(*(position - true)), 0.1)
        self.assertGreater(np.hypot(*(moment - true)), 1.0)
        self.assertAlmostEqual(image.sum(), 1.0)
        wide = 1000.0 * gaussian(*true, sigma=6.0, size=96) + rng.normal(0.0, 5.0, (96, 96))
        self.assertLess(np.hypot(*(beam_centre(np.maximum(wide, 0.0)) - true)), 0.3)

    def test_spot_selection_rules(self):
        positions = np.array([(x, y) for y in range(2, 60, 4) for x in range(2, 40, 4)], dtype=float)
        every, spacing = SELECTORS["all"](positions, (60, 40))
        self.assertEqual((len(every), spacing), (len(positions), None))
        kept, cell = SELECTORS["grid"](positions, (60, 40), cells=6)      # 10 px cells on a 60 x 40 image
        self.assertEqual((cell, len(kept)), (10.0, 6 * 4))

    def test_metrics_of_identical_and_scaled_images(self):
        yy, xx = np.indices((64, 64))
        ring = np.exp(-((np.hypot(xx - 31, yy - 33) - 12) ** 2) / 18) + 0.3 * gaussian(25, 30, 3, 64)
        metrics = FibreMetrics(ring)
        same = metrics(ring, 3.0 * ring)
        self.assertAlmostEqual(same["shape_rel_l2"], 0.0)
        self.assertAlmostEqual(same["lowpass_rel_l2"], 0.0)
        self.assertAlmostEqual(same["radial_rel_l2"], 0.0)
        for key in ("pearson", "ssim", "highpass_pearson"):
            self.assertAlmostEqual(same[key], 1.0)
        moved = metrics(ring, np.roll(ring, 3, axis=1))
        self.assertGreater(moved["shape_rel_l2"], 0.1)
        self.assertLess(moved["ssim"], same["ssim"])


class PipelineTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.datasets = write_datasets(self.root)
        self.out = self.root / "out"
        self.quiet = lambda *args: None

    def config(self, cameras, step, figure=None):
        return {
            "output_dir": str(self.out),
            "cameras": cameras,
            "datasets": self.datasets,
            "pipeline": {"fit": {"module": "forward_fit", **step}},
            "figures": {"samples": {"figure": "fit_samples", "source": "fit", "dpi": 100, **(figure or {})}},
        }

    def test_two_fibres_from_datasets_to_figures(self):
        config = self.config(
            {"beam": "cam2", "fibre": {"MMF1": "cam3", "MMF2": "cam1"}},
            {"spots": {"half_px": 4, "far_px": 10, "select": {"name": "grid", "params": {"cells": GRID}}},
             "samples": {"ids": [1], "quantiles": [0.5]}},
        )
        summary = pipeline.run(config, log=self.quiet)["fit"]

        self.assertEqual(summary["basis"]["samples"], len(SPOTS))
        self.assertEqual(summary["basis"]["used"], len(SPOTS))
        self.assertEqual(summary["target"]["samples"], 5)
        for fibre in ("MMF1", "MMF2"):
            result = summary["results"][fibre]
            self.assertEqual(result["covered"], 5)
            for fit in ("corrected", "decomposed"):
                q = result["fits"][fit]
                self.assertLess(q["shape_rel_l2"]["q75"], 2e-3, (fibre, fit))
                self.assertLess(q["beam_rel_l2"]["q75"], 2e-3, (fibre, fit))
                self.assertGreater(q["pearson"]["q25"], 0.999, (fibre, fit))

        step = self.out / "fit"
        with open(step / "frames.csv") as f:
            self.assertEqual(len(list(csv.DictReader(f))), 2 * 2 * 5)
        with np.load(step / "weights.npz") as weights:
            self.assertEqual(weights["weights/decomposed"].shape, (5, len(SPOTS)))
        with np.load(step / "samples.npz") as saved:
            n = len(saved["sample_id"])
            self.assertIn(1, saved["sample_id"])
            self.assertEqual(list(saved["fibres"]), ["MMF1", "MMF2"])
            self.assertEqual(saved["synthetic/MMF2/corrected"].shape, (n, SIZE, SIZE))
            self.assertEqual(saved["light/decomposed"].shape, (n, SIZE, SIZE))
            self.assertEqual(saved["axis/MMF1"].shape, (2,))
        figures = self.out / "figures" / "samples"
        self.assertEqual(len(list(figures.glob("*.png"))), n)

        # Figures are drawn from the artefacts alone: no dataset, no step.
        for name in ("background", "basis", "target"):
            shutil.rmtree(self.root / name)
        shutil.rmtree(figures)
        config["figures"]["samples"].update(fits=["decomposed"], fibres=["MMF2"], samples=[1],
                                            style={"font_size": 9, "cmap": "magma"})
        pipeline.run(config, steps=[], log=self.quiet)
        self.assertEqual([p.name for p in figures.iterdir()], ["run_1.png"])
        with self.assertRaises(FileNotFoundError):
            pipeline.run(config, figures=[], log=self.quiet)

    def test_one_fibre_all_spots_and_an_added_fit(self):
        class Even:
            """Every spot gets the same weight."""

            def __init__(self, basis):
                self.n = len(basis.positions)

            def __call__(self, beam):
                return np.full(self.n, float(np.sum(beam)) / self.n)

        FITS["even"] = Even
        self.addCleanup(FITS.pop, "even")
        self.datasets["basis"]["sync_guard_ns"] = None    # no sync check: the out-of-sync spot counts too
        config = self.config(
            {"beam": "cam2", "fibre": "cam3"},
            {"spots": {"half_px": 4, "far_px": 10, "select": "all", "reach_px": 6}, "fits": ["corrected", "even"],
             "samples": {"quantiles": [0.5]}},
            {"dpi": 200},
        )
        summary = pipeline.run(config, log=self.quiet)["fit"]
        self.assertEqual(summary["basis"]["used"], len(SPOTS) + 2)    # and the spot that lacks only cam1
        fits = summary["results"]["cam3"]["fits"]
        self.assertLess(fits["corrected"]["shape_rel_l2"]["median"], 0.01)
        self.assertGreater(fits["even"]["shape_rel_l2"]["median"], 0.5)
        (figure,) = (self.out / "figures" / "samples").glob("*.png")
        with Image.open(figure) as image:
            self.assertEqual(image.size[0], 675)             # one fibre: single-column width, 3.375 in at 200 dpi

    def test_config_errors_name_the_problem(self):
        cameras = {"beam": "cam2", "fibre": ["cam3"]}
        with self.assertRaisesRegex(ValueError, "unknown keys \\['basis'"):
            pipeline.run({"cameras": cameras, "basis": {}, "target": {}}, log=self.quiet)
        with self.assertRaisesRegex(ValueError, "reach_px"):
            pipeline.run(self.config(cameras, {"spots": {"half_px": 4, "far_px": 10, "select": "all"}}), log=self.quiet)
        with self.assertRaisesRegex(ValueError, "unknown fit 'nearest'; known"):
            pipeline.run(self.config(cameras, {"fits": ["nearest"]}), log=self.quiet)
        with self.assertRaisesRegex(ValueError, "unknown dataset adapter"):
            self.datasets["target"]["adapter"] = "hdf5"
            pipeline.run(self.config(cameras, {"spots": {"half_px": 4, "far_px": 10}}), log=self.quiet)

    def test_example_config_is_valid(self):
        config = strip(json.loads((REPO / "configs" / "analysis_v1.example.json").read_text()))
        fill(config, pipeline.DEFAULTS, "example config", required=("cameras",))
        for block in config["pipeline"].values():
            self.assertIn(block["module"], pipeline.MODULES)
            resolve(block)
        for block in config["figures"].values():
            self.assertIn(block["figure"], pipeline.FIGURES)
            fill(block, fit_samples.DEFAULTS, "figure", required=("source",))
            self.assertIn(block["source"], config["pipeline"])


if __name__ == "__main__":
    unittest.main()
