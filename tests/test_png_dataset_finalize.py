"""PNG finalization indexes saved pixels without processing them a second time."""

import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from optidaq.data.png_io import write_png, write_png_gray
from optidaq.data.scientific_png_export import finalize_png_dataset


_PARSER_PATH = Path(__file__).resolve().parents[1] / "experiments" / "parse_process.py"
_SPEC = importlib.util.spec_from_file_location("clear26_test_parse_process", _PARSER_PATH)
parser = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = parser
_SPEC.loader.exec_module(parser)


class PngDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "run"
        self.run.mkdir()
        self.cameras = ["cam1", "cam2"]
        self.image = np.array([[1, 4095], [24, 81]], dtype=np.uint16)
        resolved = {
            "run_id": "test-run",
            "cameras": {
                camera: {"shape": [8, 8], "dtype": "int16", "payload_nbytes": 128, "cameraPixelFormat": "mono12"}
                for camera in self.cameras
            },
        }
        self.write_json("resolved_config.json", resolved)
        for camera in self.cameras:
            self.write_json(camera + ".png.json", {
                "schema_version": 1,
                "source": {"shape": [8, 8], "dtype": "int16"},
                "stored": {"shape": [2, 2], "dtype": "uint16", "value_scale": 1},
                "transforms": [{"name": "center_crop", "params": {"height": 4, "width": 4}}, {"name": "avg_bin", "params": {"factor": 2, "dtype": "uint16"}}],
                "metadata_hooks": [{"name": "stats"}],
            })

    def write_json(self, name, value):
        (self.run / name).write_text(json.dumps(value), encoding="utf-8")

    def write_camera(self, camera, times):
        rows = []
        for index, timestamp in enumerate(times):
            stamp = 1_000_000 + index * 1000
            write_png_gray(self.run / camera / ("%020d_%08d.png" % (stamp, index)), self.image)
            row = {
                "device": camera, "frame_index": index, "acq_stamp_ns": stamp,
                "source_image_number": index + 200, "raw_header_pulse_id": index + 300,
                "png": {"shape": [2, 2], "dtype": "uint16", "metadata": {
                    "stat_min": 1, "stat_max": 4095, "stat_mean": float(self.image.mean()),
                    "custom": {"label": "stored pixels"},
                }},
            }
            if timestamp is not None:
                row["image_ready_ns"] = timestamp
            rows.append(row)
        (self.run / (camera + ".sync.jsonl")).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return rows

    def finalize(self):
        return finalize_png_dataset(self.run, sync_alignment_tolerance_ns=50)

    def test_finalizer_preserves_pixels_and_aligns_only_unambiguous_frames(self):
        self.write_camera("cam1", [100, 130, 300, 700, None])
        self.write_camera("cam2", [110, 320, 510])
        actuator = {"device": "cam1", "frame_index": 2, "step_id": 4, "quality": "approximate", "method": "host_receive_time"}
        (self.run / "actuator_frame_index.jsonl").write_text(json.dumps(actuator) + "\n", encoding="utf-8")
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in self.run.glob("*/*.png")}
        with patch("optidaq.data.scientific_png_export.write_png_gray", side_effect=AssertionError("must not rewrite PNGs")):
            result = self.finalize()
        self.assertEqual(result.image_count, 8)
        self.assertEqual(result.counts_by_camera, {"cam1": 5, "cam2": 3})
        for path, snapshot in before.items():
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), snapshot)
        with sqlite3.connect(result.db_path) as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM images WHERE sample_id IS NULL").fetchone()[0], 4)
            pairs = conn.execute("SELECT a.frame_index, b.frame_index FROM images a JOIN images b ON a.sample_id=b.sample_id WHERE a.camera='cam1' AND b.camera='cam2'").fetchall()
            self.assertEqual(pairs, [(2, 1)])
            row = conn.execute("SELECT height,width,png_dtype,raw_dtype,pulse_id,hook_metadata_json,extra_json FROM images WHERE camera='cam1' AND frame_index=2").fetchone()
            self.assertEqual(row[:5], (2, 2, "uint16", "int16", 302))
            self.assertEqual(json.loads(row[5])["custom"]["label"], "stored pixels")
            self.assertEqual(json.loads(row[6])["actuator"], actuator)
            run_info = dict(conn.execute("SELECT key,value FROM run_info"))
            self.assertEqual(run_info["source_format"], "clear2026_png")
            self.assertEqual(run_info["sync_alignment_unmatched_frame_count"], "4")

    def test_missing_sync_and_empty_camera_still_get_database(self):
        self.write_camera("cam1", [None])
        self.write_camera("cam2", [])
        result = self.finalize()
        with sqlite3.connect(result.db_path) as conn:
            self.assertEqual(conn.execute("SELECT sample_id,sync_time_ns FROM images").fetchall(), [(None, None)])

    def test_bad_sidecar_does_not_replace_existing_database(self):
        self.write_camera("cam1", [100])
        self.write_camera("cam2", [110])
        result = self.finalize()
        previous = result.db_path.read_bytes()
        (self.run / "cam1.sync.jsonl").write_text("", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "PNG/sidecar frame mismatch"):
            self.finalize()
        self.assertEqual(result.db_path.read_bytes(), previous)

    def test_configured_pulse_priority_preserves_exact_ids_not_fake_timestamps(self):
        resolved = json.loads((self.run / "resolved_config.json").read_text())
        resolved["config"] = {"sync": {
            "priority": ["pulse_id", "image_ready_ns"],
            "max_span_ns_by_source": {"image_ready_ns": 1_000_000},
        }}
        self.write_json("resolved_config.json", resolved)
        for camera, times in (("cam1", [1_000_000, 2_000_000]), ("cam2", [30_000_000, 40_000_000])):
            rows = self.write_camera(camera, times)
            for index, row in enumerate(rows):
                row["pulse_id"] = 11 + index
                row["chosen_sync_source"] = "old_source"
            (self.run / (camera + ".sync.jsonl")).write_text("".join(json.dumps(row) + "\n" for row in rows))
        result = finalize_png_dataset(self.run)
        with sqlite3.connect(result.db_path) as conn:
            rows = conn.execute("SELECT sample_id,chosen_sync_source,chosen_sync_value,sync_time_ns FROM images ORDER BY camera,frame_index").fetchall()
            self.assertEqual(rows, [(1, "pulse_id", "11", None), (2, "pulse_id", "12", None)] * 2)
            report = json.loads(dict(conn.execute("SELECT key,value FROM run_info"))["sync_report_json"])
            self.assertTrue(report["passed"])
            self.assertEqual(report["selected_source"], "pulse_id")

    def test_configured_one_ms_tolerance_rejects_false_pair(self):
        resolved = json.loads((self.run / "resolved_config.json").read_text())
        resolved["sync"] = {
            "priority": ["image_ready_ns"],
            "max_span_ns_by_source": {"image_ready_ns": 1_000_000},
        }
        self.write_json("resolved_config.json", resolved)
        self.write_camera("cam1", [1_000_000_000])
        self.write_camera("cam2", [1_002_000_000])
        result = finalize_png_dataset(self.run)
        with sqlite3.connect(result.db_path) as conn:
            self.assertEqual(conn.execute("SELECT sample_id,chosen_sync_source FROM images").fetchall(), [(None, None), (None, None)])
            report = json.loads(dict(conn.execute("SELECT key,value FROM run_info"))["sync_report_json"])
            self.assertFalse(report["passed"])
            self.assertEqual(report["reports"]["image_ready_ns"]["max_span_ns"], 1_000_000)

    def set_png_alignment(self, value):
        resolved = json.loads((self.run / "resolved_config.json").read_text())
        resolved["sync"] = {
            "priority": ["image_ready_ns"],
            "max_span_ns_by_source": {"image_ready_ns": 1_000_000},
        }
        resolved["config"] = {"output": {"png": {"sync_alignment": value}}}
        self.write_json("resolved_config.json", resolved)

    def test_png_pairing_uses_exporter_groups_without_rejecting_whole_run(self):
        self.set_png_alignment({"source": "image_ready_ns", "tolerance_ns": 50_000_000})
        self.write_camera("cam1", [1_000_000_000, 1_100_000_000, 1_300_000_000,
                                   1_330_000_000, 1_500_000_000, None])
        self.write_camera("cam2", [1_002_000_000, 1_502_000_000, 1_700_000_000])
        result = finalize_png_dataset(self.run)
        with sqlite3.connect(result.db_path) as conn:
            rows = conn.execute("SELECT camera,frame_index,sample_id FROM images ORDER BY camera,frame_index").fetchall()
            self.assertEqual(rows, [
                ("cam1", 0, 1), ("cam1", 1, 2), ("cam1", 2, None),
                ("cam1", 3, None), ("cam1", 4, 4), ("cam1", 5, None),
                ("cam2", 0, 1), ("cam2", 1, 4), ("cam2", 2, 5),
            ])
            info = dict(conn.execute("SELECT key,value FROM run_info"))
            self.assertEqual(info["sync_alignment_tolerance_ns"], "50000000")
            self.assertEqual(json.loads(info["sync_report_json"])["mode"], "png_timestamp_alignment")
        self.assertEqual(result.image_count, 9)

    def test_explicit_alignment_arguments_override_png_config(self):
        self.set_png_alignment({"source": "image_ready_ns", "tolerance_ns": 50_000_000})
        self.write_camera("cam1", [1_000_000_000])
        self.write_camera("cam2", [1_002_000_000])
        result = finalize_png_dataset(self.run, sync_alignment_tolerance_ns=1_000_000)
        with sqlite3.connect(result.db_path) as conn:
            self.assertEqual(conn.execute("SELECT sample_id FROM images ORDER BY camera").fetchall(), [(1,), (2,)])
        result = finalize_png_dataset(self.run, sync_alignment_source="acq_stamp_ns",
                                      sync_alignment_tolerance_ns=1)
        with sqlite3.connect(result.db_path) as conn:
            self.assertEqual(conn.execute("SELECT sample_id,chosen_sync_source FROM images").fetchall(),
                             [(1, "acq_stamp_ns"), (1, "acq_stamp_ns")])

    def test_invalid_png_pairing_never_silently_uses_acquisition_policy(self):
        self.write_camera("cam1", [100])
        self.write_camera("cam2", [110])
        previous = self.finalize().db_path.read_bytes()
        for invalid in (None, [], {}, {"source": "image_ready_ns"},
                        {"source": "", "tolerance_ns": 50},
                        {"source": "image_ready_ns", "tolerance_ns": 0},
                        {"source": "image_ready_ns", "tolerance_ns": True},
                        {"source": "pulse_id", "tolerance_ns": 50}):
            with self.subTest(invalid=invalid):
                self.set_png_alignment(invalid)
                with self.assertRaises(ValueError):
                    finalize_png_dataset(self.run)
                self.assertEqual((self.run / "dataset.db").read_bytes(), previous)

    def test_export_provenance_uses_saved_manifests_and_effective_pairing(self):
        self.set_png_alignment({"source": "image_ready_ns", "tolerance_ns": 50_000_000})
        resolved = json.loads((self.run / "resolved_config.json").read_text())
        resolved["config"]["output"]["png"].update(
            transforms=[], metadata_hooks=[], compress_level=9,
        )
        self.write_json("resolved_config.json", resolved)
        manifests = {}
        for camera in self.cameras:
            self.write_camera(camera, [100])
            path = self.run / (camera + ".png.json")
            manifest = json.loads(path.read_text())
            manifest.update(metadata_hooks=["stats", "centroid", "crop_energy"], compress_level=1)
            self.write_json(path.name, manifest)
            manifests[camera] = manifest
        result = finalize_png_dataset(self.run)
        with sqlite3.connect(result.db_path) as conn:
            info = dict(conn.execute("SELECT key,value FROM run_info"))
        config = json.loads(info["export_config_json"])
        self.assertEqual(config["hooks"]["image"], {camera: manifest["transforms"] for camera, manifest in manifests.items()})
        self.assertEqual(config["hooks"]["metadata"], ["stats", "centroid", "crop_energy"])
        self.assertEqual(config["png"], {"compress_level": 1})
        self.assertEqual(config["sync_alignment"], {"source": "image_ready_ns", "tolerance_ns": 50_000_000})
        self.assertEqual(json.loads(info["png_manifests_json"]), manifests)

    def test_rgb_saved_run_can_be_indexed_without_decoding_pixels(self):
        self.write_camera("cam1", [100])
        self.write_camera("cam2", [110])
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        resolved = json.loads((self.run / "resolved_config.json").read_text())
        for camera in self.cameras:
            resolved["cameras"][camera].update(shape=[2, 2, 3], dtype="uint8", payload_nbytes=12)
            manifest = json.loads((self.run / (camera + ".png.json")).read_text())
            manifest["source"] = {"shape": [2, 2, 3], "dtype": "uint8"}
            manifest["stored"].update(shape=[2, 2, 3], dtype="uint8")
            self.write_json(camera + ".png.json", manifest)
            sidecar = self.run / (camera + ".sync.jsonl")
            row = json.loads(sidecar.read_text())
            row["png"].update(shape=[2, 2, 3], dtype="uint8")
            sidecar.write_text(json.dumps(row) + "\n")
            write_png(next((self.run / camera).glob("*.png")), rgb)
        self.write_json("resolved_config.json", resolved)
        result = self.finalize()
        with sqlite3.connect(result.db_path) as conn:
            self.assertEqual(conn.execute("SELECT height,width,png_dtype FROM images").fetchall(), [(2, 2, "uint8")] * 2)

    def test_parser_uses_stored_descriptor_and_requires_reprocessing_opt_in(self):
        self.write_camera("cam1", [100])
        self.write_camera("cam2", [110])
        config = {"dataset_dir": str(self.run), "unaligned": True, "transforms": [
            {"name": "crop", "kwargs": {"offset_y": 0, "offset_x": 0, "height": 1, "width": 1}},
            {"name": "stats"},
        ]}
        source = parser.PngSource(self.run, config)
        samples = list(source)
        self.assertEqual(len(samples), 2)
        np.testing.assert_array_equal(samples[0].images["cam1"], self.image)
        self.assertEqual(source.camera_configs["cam1"]["shape"], [2, 2])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            safe_specs = parser._source_transform_configs(source, config)
        self.assertEqual([item["name"] for item in safe_specs], ["stats"])
        for transform in parser.build_transforms(safe_specs):
            samples[0] = transform(samples[0])
        self.assertEqual(samples[0].images["cam1"].shape, (2, 2))
        explicit_specs = parser._source_transform_configs(source, dict(config, process_png=True))
        for transform in parser.build_transforms(explicit_specs):
            samples[1] = transform(samples[1])
        self.assertEqual(samples[1].images["cam2"].shape, (1, 1))
        sink = parser.PngDbSink(Path(self.temp.name) / "parsed", config)
        try:
            sink.write(samples[0])
            self.assertEqual(sink.count, 1)
            self.assertEqual(sink.camera_configs["cam1"]["shape"], [2, 2])
        finally:
            sink.close()

    def test_parser_legacy_run_without_manifest(self):
        self.image = np.arange(64, dtype=np.uint16).reshape(8, 8)
        for camera in self.cameras:
            (self.run / (camera + ".png.json")).unlink()
            self.write_camera(camera, [100])
        source = parser.PngSource(self.run, {"unaligned": True})
        sample = next(iter(source))
        np.testing.assert_array_equal(sample.images["cam1"], self.image)
        self.assertEqual(sample.images["cam1"].dtype, np.dtype("int16"))


if __name__ == "__main__":
    unittest.main()
