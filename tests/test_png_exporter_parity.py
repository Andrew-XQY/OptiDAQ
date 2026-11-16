"""Direct PNG acquisition matches the existing offline export for the same frames."""

from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
from PIL import Image

from optidaq.daq.contracts import Frame
from optidaq.daq.run_store import build_raw_spec, write_resolved_config
from optidaq.daq.save_backends import make_save_backend
from optidaq.data.scientific_png_export import (
    config_from_mapping,
    export_run_to_png_dataset,
    finalize_png_dataset,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ("cam1", "cam2")
# Recorded in the example dataset's export_config_json, not today's standalone
# export_config_full.json (whose cam2 crop has since changed).
EXAMPLE_TRANSFORMS = {
    "cam1": [
        {"name": "crop_box", "params": {"x1": 96, "y1": 96, "x2": 1120, "y2": 1120}},
        {"name": "avg_bin", "params": {"factor": 4, "dtype": "uint16"}},
    ],
    "cam2": [
        {"name": "crop_box", "params": {"x1": 335, "y1": 345, "x2": 847, "y2": 857}},
        {"name": "avg_bin", "params": {"factor": 2, "dtype": "uint16"}},
    ],
}
METADATA_HOOKS = ["stats", "centroid", "crop_energy"]
ALIGNMENT = {"source": "image_ready_ns", "tolerance_ns": 50_000_000}


class PngExporterParityTest(unittest.TestCase):
    def config(self):
        return json.loads((REPO_ROOT / "configs" / "experiment_lab_v1.json").read_text())

    def test_main_config_matches_example_for_active_cameras(self):
        cfg = self.config()
        self.assertEqual(cfg["cameras"], list(CAMERAS))
        self.assertEqual(cfg["output"]["format"], "png")
        png = cfg["output"]["png"]
        self.assertEqual(png["mode"], "raw16")
        self.assertEqual(png["compress_level"], 1)
        self.assertEqual(png["transforms"], EXAMPLE_TRANSFORMS)
        self.assertEqual(png["metadata_hooks"], METADATA_HOOKS)
        self.assertEqual({key: png["sync_alignment"][key] for key in ALIGNMENT}, ALIGNMENT)

    def test_direct_output_matches_offline_pixels_metadata_and_pairing(self):
        cfg = self.config()
        png_cfg = cfg["output"]["png"]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw_dir, direct_dir, exported_dir = (root / name for name in ("raw", "direct", "exported"))
            raw_dir.mkdir()
            direct_dir.mkdir()
            raw_cfg = deepcopy(cfg)
            raw_cfg["output"]["format"] = "raw"
            camera_config = {
                camera: {"shape": [1216, 1216], "dtype": "int16",
                         "payload_nbytes": 1216 * 1216 * 2, "cameraPixelFormat": "mono12"}
                for camera in CAMERAS
            }
            for directory, run_cfg in ((raw_dir, raw_cfg), (direct_dir, cfg)):
                write_resolved_config(directory, {
                    "run_id": "parity-run", "config": run_cfg,
                    "sync": cfg["sync"], "cameras": camera_config,
                    "output_format": run_cfg["output"]["format"],
                })

            # Two complete groups, one ambiguous group, and two incomplete groups.
            # The ambiguous group must not invalidate otherwise usable samples.
            times_ms = {"cam1": [0, 100, 130, 300, 500], "cam2": [20, 110, 320, 700]}
            grid = np.arange(1216 * 1216, dtype=np.int32).reshape(1216, 1216)
            originals = []
            for camera_number, camera in enumerate(CAMERAS):
                spec = build_raw_spec(
                    file_prefix=camera, payload_nbytes=1216 * 1216 * 2,
                    segment_max_bytes=1216 * 1216 * 4 + 128,
                )
                backends = [make_save_backend(
                    camera, directory, spec, image_shape=(1216, 1216),
                    image_dtype="int16", config=run_cfg,
                ) for directory, run_cfg in ((raw_dir, raw_cfg), (direct_dir, cfg))]
                for backend in backends:
                    backend.open()
                try:
                    for index, milliseconds in enumerate(times_ms[camera]):
                        source = ((grid + camera_number * 179 + index * 31) % 4096).astype(np.int16)
                        timestamp = 1_000_000_000 + milliseconds * 1_000_000
                        frame = Frame(camera, 200 + index, timestamp + 123,
                                      source.tobytes(), pulse_id=400 + index)
                        sync = {"image_ready_ns": timestamp,
                                "local_receive_ns": timestamp + 456,
                                "pulse_id": frame.pulse_id,
                                "chosen_sync_source": "pulse_id",
                                "chosen_sync_value": frame.pulse_id,
                                "source": {"camera": camera}}
                        object.__setattr__(frame, "sync_metadata", sync)
                        originals.append((frame, deepcopy(vars(frame))))
                        for backend in backends:
                            backend.save(frame)
                finally:
                    for backend in backends:
                        backend.close()
            for frame, snapshot in originals:
                self.assertEqual(vars(frame), snapshot)

            offline = export_run_to_png_dataset(config_from_mapping({
                "run_dir": str(raw_dir), "out_dir": str(exported_dir),
                "cameras": list(CAMERAS), "show_progress": False,
                "png": {"compress_level": png_cfg["compress_level"]},
                "hooks": {"image": png_cfg["transforms"], "metadata": png_cfg["metadata_hooks"]},
                "sync_alignment": png_cfg["sync_alignment"],
            }))
            direct = finalize_png_dataset(direct_dir)
            self.assertEqual(offline.image_count, 6)
            self.assertEqual(direct.image_count, 9)

            columns = (
                "image_id", "sample_id", "frame_sample_id", "camera", "png_path",
                "png_dtype", "raw_dtype", "height", "width", "frame_index",
                "source_image_number", "acq_stamp_ns", "sync_time_ns", "sync_group_span_ns",
                "pulse_id", "stat_min", "stat_max", "stat_mean", "stat_sum",
                "centroid_x", "centroid_y", "intensity_in_roi", "intensity_cropped_out",
                "roi_energy_fraction", "hook_metadata_json",
            )
            select = "SELECT %s FROM images" % ",".join(columns)
            with sqlite3.connect(offline.db_path) as offline_db, sqlite3.connect(direct.db_path) as direct_db:
                expected_rows = offline_db.execute(select + " ORDER BY camera,frame_index").fetchall()
                actual_rows = direct_db.execute(select + " WHERE sample_id IS NOT NULL ORDER BY camera,frame_index").fetchall()
                self.assertEqual(actual_rows, expected_rows)
                self.assertEqual(direct_db.execute("SELECT count(*) FROM images WHERE sample_id IS NULL").fetchone()[0], 3)
                self.assertEqual(direct_db.execute(
                    "SELECT sample_id,count(*) FROM images WHERE sample_id IS NOT NULL GROUP BY sample_id"
                ).fetchall(), [(1, 2), (3, 2), (4, 1), (5, 1)])
                self.assertEqual(direct_db.execute(
                    "SELECT DISTINCT chosen_sync_source FROM images WHERE sample_id IS NOT NULL"
                ).fetchall(), [("image_ready_ns",)])
                self.assertEqual(direct_db.execute(
                    "SELECT count(*) FROM images WHERE raw_segment_path IS NOT NULL OR byte_offset IS NOT NULL"
                ).fetchone()[0], 0)
                run_info = dict(direct_db.execute("SELECT key,value FROM run_info"))
                provenance = json.loads(run_info["export_config_json"])
                self.assertEqual(provenance["hooks"]["image"], EXAMPLE_TRANSFORMS)
                self.assertEqual(provenance["hooks"]["metadata"], METADATA_HOOKS)
                self.assertEqual(provenance["png"]["compress_level"], 1)
                self.assertEqual(provenance["sync_alignment"], ALIGNMENT)

            for row in expected_rows:
                path = row[columns.index("png_path")]
                self.assertEqual((direct_dir / path).read_bytes(), (exported_dir / path).read_bytes())
                with Image.open(direct_dir / path) as image:
                    pixels = np.asarray(image)
                    self.assertEqual(pixels.shape, (256, 256))
                    self.assertEqual(pixels.dtype, np.dtype("uint16"))


if __name__ == "__main__":
    unittest.main()
