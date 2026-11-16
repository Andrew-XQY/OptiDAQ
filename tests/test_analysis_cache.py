"""Cached analysis keeps exact numerical facts and never trusts changed inputs."""

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from optidaq.analysis import analyze_run as analysis


class AnalysisCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name) / "run"
        self.run.mkdir()
        self.out = self.run / "data_analysis"
        self.images = {}
        cfg = {"cameras": ["cam1", "cam2"], "dmd": {"count": 2},
               "scan": {"frames_per_point": 1},
               "output": {"png": {"sync_alignment": {"source": "image_ready_ns", "tolerance_ns": 100}}}}
        (self.run / "resolved_config.json").write_text(json.dumps({"config": cfg}))
        events = [{"event": kind, "step_id": step, "host_time_ns": stamp}
                  for kind, step, stamp in (("change_begin", 0, 100), ("settle_complete", 0, 110),
                                             ("change_begin", 1, 200), ("settle_complete", 1, 210),
                                             ("end", None, 300))]
        self.write_lines("actuator.events.jsonl", events)
        patterns = self.run / "dmd_patterns"
        patterns.mkdir()
        for index in range(2):
            Image.fromarray(np.ones((2, 2), np.uint8)).save(patterns / ("%06d.png" % index))
        labels = []
        with sqlite3.connect(self.run / "dataset.db") as con:
            con.execute("CREATE TABLE images (camera, frame_index, png_path, sample_id, sync_time_ns, stat_mean, stat_max, stat_sum)")
            for c, camera in enumerate(cfg["cameras"]):
                (self.run / camera).mkdir()
                (self.run / (camera + ".png.json")).write_text(json.dumps({"stored": {"shape": [2, 2], "dtype": "uint16"}}))
                sidecars = []
                self.images[camera] = []
                for index in range(2):
                    image = np.array([[1, 8], [3, 4]], np.uint16) + c * 20 + index * 3
                    # Different pixels peak in different images.
                    if index:
                        image[0, 1] = 0
                    self.images[camera].append(image)
                    name = "%020d_%08d.png" % (1000 + index * 1000, index)
                    Image.fromarray(image).save(self.run / camera / name)
                    ready = 1000 + index * 1000 + c * 10
                    received = 150 + index * 100
                    metrics = {"stat_mean": float(image.mean()), "stat_max": int(image.max()), "stat_sum": int(image.sum())}
                    sidecars.append({"frame_index": index, "local_receive_ns": received, "image_ready_ns": ready,
                                     "acq_stamp_ns": ready - 100, "source_image_number": index,
                                     "cycleNameGlob": "test", "png": {"metadata": metrics}})
                    labels.append({"device": camera, "frame_index": index, "step_id": index,
                                   "local_receive_ns": received})
                    con.execute("INSERT INTO images VALUES (?,?,?,?,?,?,?,?)", (camera, index, camera + "/" + name,
                                index, ready, metrics["stat_mean"], metrics["stat_max"], metrics["stat_sum"]))
                self.write_lines(camera + ".sync.jsonl", sidecars)
        self.write_lines("actuator_frame_index.jsonl", labels)

    def write_lines(self, name, rows):
        (self.run / name).write_text("".join(json.dumps(row) + "\n" for row in rows))

    @contextmanager
    def fake_renderer(self, effect=None):
        module = types.ModuleType("optidaq.analysis.clean_plots")

        def render(out, *args, **kwargs):
            # A test placeholder, not PDF authoring or a deliverable.
            (out / "report.pdf").write_bytes(b"report placeholder")
            if effect:
                effect()

        module.write_report = render
        with patch.dict(sys.modules, {module.__name__: module}):
            yield

    def test_streamed_maxima_histograms_and_percentiles_match_pixels(self):
        audit, frames, pairs, products = analysis.compute_results(self.run)
        self.assertTrue(audit["validation_passed"], audit["failures"])
        self.assertEqual(len(frames), 4)
        self.assertEqual(len(pairs), 2)
        for camera, images in self.images.items():
            stack = np.stack(images)
            np.testing.assert_array_equal(products[camera]["maximum"], stack.max(axis=0))
            np.testing.assert_array_equal(products[camera]["histogram"], np.bincount(stack.ravel(), minlength=65536))
            np.testing.assert_allclose(list(audit["statistics"][camera]["pooled_pixel_quantiles_dn"].values()),
                                       np.percentile(stack, [1, 50, 99]), rtol=0, atol=1e-14)

    def test_histogram_percentiles_match_numpy_at_interpolated_values(self):
        values = np.array([0, 0, 9, 10, 65535], dtype=np.uint16)
        levels = [0, 1, 25, 50, 71, 99, 100]
        actual = analysis.histogram_percentiles(np.bincount(values, minlength=65536), levels)
        np.testing.assert_allclose(actual, np.percentile(values, levels), rtol=0, atol=1e-10)

    def test_cache_hit_does_not_decode_or_query_database(self):
        with self.fake_renderer():
            first, cached = analysis.analyze_run(self.run)
            self.assertFalse(cached)
            with patch.object(analysis.Image, "open", side_effect=AssertionError("must reuse cache")), \
                    patch.object(analysis.sqlite3, "connect", side_effect=AssertionError("must reuse cache")):
                second, cached = analysis.analyze_run(self.run)
        self.assertTrue(cached)
        self.assertEqual(first, second)
        self.assertEqual({p.name for p in self.out.iterdir()}, {"results.npz", "report.pdf"})

    def test_source_change_invalidates_cache_even_with_same_size_and_timestamp(self):
        with self.fake_renderer():
            analysis.analyze_run(self.run)
            path = self.run / "resolved_config.json"
            old_stat = path.stat()
            text = path.read_text().replace('"count": 2', '"count": 3')
            path.write_text(text)
            import os
            os.utime(path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
            with patch.object(analysis, "compute_results", wraps=analysis.compute_results) as compute:
                audit, cached = analysis.analyze_run(self.run)
            compute.assert_called_once()
        self.assertFalse(cached)
        self.assertFalse(audit["validation_passed"])

    def test_refresh_recomputes_and_corrupt_cache_is_rebuilt(self):
        with self.fake_renderer():
            analysis.analyze_run(self.run)
            with patch.object(analysis, "compute_results", wraps=analysis.compute_results) as compute:
                _, cached = analysis.analyze_run(self.run, refresh=True)
            self.assertFalse(cached)
            compute.assert_called_once()
            (self.out / "results.npz").write_bytes(b"damaged cache")
            audit, cached = analysis.analyze_run(self.run)
        self.assertFalse(cached)
        self.assertTrue(audit["validation_passed"])

    def test_changed_sources_during_render_do_not_replace_prior_outputs(self):
        with self.fake_renderer():
            analysis.analyze_run(self.run)
        before = {p.name: p.read_bytes() for p in self.out.iterdir()}
        with self.fake_renderer(lambda: (self.run / "unexpected.txt").write_text("changed")):
            with self.assertRaisesRegex(RuntimeError, "Dataset changed"):
                analysis.analyze_run(self.run)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.out.iterdir()})

    def test_corrupt_database_and_empty_camera_are_reported_as_failures(self):
        (self.run / "dataset.db").write_bytes(b"corrupt database")
        for path in (self.run / "cam1").glob("*.png"):
            path.write_bytes(b"corrupt png")
        audit, frames, pairs, products = analysis.compute_results(self.run)
        self.assertFalse(audit["validation_passed"])
        self.assertEqual(audit["cameras"]["cam1"]["decoded_count"], 0)
        self.assertEqual(audit["statistics"]["cam1"]["mean_dn"], {"n": 0})
        self.assertEqual(products["cam1"]["histogram"].sum(), 0)
        self.assertEqual(len(frames), 2)
        self.assertEqual(pairs, [])


if __name__ == "__main__":
    unittest.main()
