"""Committed capture identity, crash evidence and compatibility of lab PNGs."""

from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from optidaq.daq.contracts import Frame
from optidaq.daq.run_store import build_raw_spec, write_resolved_config
from optidaq.daq.save_backends import make_save_backend
from optidaq.data.lab_png_finalize import finalize_lab_png_dataset


class LabPngFinalizeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name)
        self.cameras = ("cam1", "cam2")
        self.config = {"output": {"format": "png", "png": {
            "mode": "raw16", "transforms": [], "metadata_hooks": ["stats"],
            # Deliberately contradictory timestamps must not override IDs.
            "sync_alignment": {"source": "image_ready_ns", "tolerance_ns": 2},
        }}}
        write_resolved_config(self.run, {
            "config": self.config,
            "cameras": {camera: {"shape": [2, 2], "dtype": "int16", "payload_nbytes": 8}
                        for camera in self.cameras},
        })
        self.commits = [{"type": "capture_commit", "capture_id": "capture_%06d" % step,
                         "step_id": step, "saved": []} for step in range(2)]
        for camera in self.cameras:
            backend = make_save_backend(
                camera, self.run, build_raw_spec(file_prefix=camera, payload_nbytes=8, segment_max_bytes=4096),
                image_shape=(2, 2), image_dtype="int16", config=self.config,
            )
            backend.open()
            try:
                for step in range(2):
                    stamp = 10_000 + step * 100 + (0 if camera == "cam1" else 1)
                    frame = Frame(camera, 100 + step, stamp,
                                  np.array([[1, 2], [3, step + 4]], dtype=np.int16).tobytes())
                    object.__setattr__(frame, "sync_metadata", {
                        "image_ready_ns": (100, 200)[step] if camera == "cam1" else (201, 101)[step],
                        "capture": {"capture_id": self.commits[step]["capture_id"], "step_id": step,
                                    "actuator": {"target_counters": {"x": step * 100, "y": 0},
                                                 "readbacks": {"x": {"position_counter": step * 100}}}},
                    })
                    self.commits[step].update(deepcopy(frame.sync_metadata["capture"]))
                    backend.save(frame)
                    self.commits[step]["saved"].append({
                        "camera": camera, "capture_id": self.commits[step]["capture_id"],
                        "frame_index": step, "imageNumber": 100 + step,
                        "png_path": "%s/%020d_%08d.png" % (camera, stamp, step),
                    })
            finally:
                backend.close()
        self.write_commits(self.commits)

    def write_commits(self, rows):
        self.write_rows(self.run / "capture_commits.jsonl", rows)

    @staticmethod
    def write_rows(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def sidecar(self, camera):
        path = self.run / (camera + ".sync.jsonl")
        return path, [json.loads(line) for line in path.read_text().splitlines()]

    def test_committed_ids_survive_crossed_timestamps_and_preserve_pixels(self):
        snapshots = {path: path.read_bytes() for path in self.run.rglob("*.png")}
        result = finalize_lab_png_dataset(self.run)
        self.assertEqual(result.image_count, 4)
        self.assertEqual(result.counts_by_camera, {"cam1": 2, "cam2": 2})
        self.assertEqual({path: path.read_bytes() for path in snapshots}, snapshots)
        with sqlite3.connect(result.db_path) as db:
            self.assertEqual(db.execute(
                "SELECT camera,frame_index,sample_id,chosen_sync_source FROM images ORDER BY camera,frame_index"
            ).fetchall(), [(camera, step, step + 1, "capture_id") for camera in self.cameras for step in range(2)])
            rows = db.execute("SELECT sample_id,sync_json,chosen_sync_value FROM images ORDER BY camera,frame_index").fetchall()
            for sample_id, sync_json, value in rows:
                capture = json.loads(sync_json)["capture"]
                self.assertEqual(capture["step_id"] + 1, sample_id)
                self.assertEqual(json.loads(value), capture["capture_id"])
                self.assertEqual(capture["actuator"]["target_counters"]["x"], (sample_id - 1) * 100)
            report = json.loads(dict(db.execute("SELECT key,value FROM run_info"))["sync_report_json"])
            self.assertEqual(report["mode"], "committed_lab_capture")
            self.assertFalse(report["physical_exposure_synchronization_verified"])

    def test_uncommitted_image_preserves_existing_database(self):
        previous = finalize_lab_png_dataset(self.run).db_path.read_bytes()
        self.write_commits(self.commits[:1])
        with self.assertRaisesRegex(RuntimeError, "without a completed capture commit"):
            finalize_lab_png_dataset(self.run)
        self.assertEqual((self.run / "dataset.db").read_bytes(), previous)

    def test_duplicate_sidecar_is_not_silently_overwritten(self):
        path, rows = self.sidecar("cam1")
        self.write_rows(path, rows + rows[:1])
        with self.assertRaisesRegex(RuntimeError, "duplicate sidecar"):
            finalize_lab_png_dataset(self.run)

    def test_capture_metadata_must_agree_across_cameras(self):
        path, rows = self.sidecar("cam2")
        rows[0]["capture"]["actuator"]["target_counters"]["x"] = 999
        self.write_rows(path, rows)
        with self.assertRaisesRegex(RuntimeError, "capture metadata differs"):
            finalize_lab_png_dataset(self.run)

    def test_commit_coordinates_must_match_both_saved_images(self):
        previous = finalize_lab_png_dataset(self.run).db_path.read_bytes()
        commits = deepcopy(self.commits)
        commits[0]["actuator"]["target_counters"]["x"] = 999
        self.write_commits(commits)
        with self.assertRaisesRegex(RuntimeError, "commit metadata differs"):
            finalize_lab_png_dataset(self.run)
        self.assertEqual((self.run / "dataset.db").read_bytes(), previous)

    def test_incomplete_pair_is_not_indexed(self):
        path, rows = self.sidecar("cam2")
        (self.run / self.commits[1]["saved"][1]["png_path"]).unlink()
        self.write_rows(path, rows[:1])
        with self.assertRaisesRegex(RuntimeError, "incomplete camera pair"):
            finalize_lab_png_dataset(self.run)

    def test_duplicate_camera_or_source_frame_is_rejected(self):
        path, original = self.sidecar("cam1")
        for field in ("capture", "source_image_number"):
            rows = deepcopy(original)
            rows[1][field] = rows[0][field]
            self.write_rows(path, rows)
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "duplicate"):
                finalize_lab_png_dataset(self.run)

    def test_commit_must_acknowledge_exact_saved_pair(self):
        previous = finalize_lab_png_dataset(self.run).db_path.read_bytes()
        cases = []
        for field, value in (("frame_index", 1), ("imageNumber", 999),
                             ("png_path", "../other.png"), ("capture_id", "other"),
                             ("camera", "cam1")):
            rows = deepcopy(self.commits)
            rows[0]["saved"][1][field] = value
            cases.append(rows)
        rows = deepcopy(self.commits)
        rows[0]["saved"] = rows[0]["saved"][:1]
        cases.append(rows)
        cases.append(deepcopy(self.commits) + deepcopy(self.commits[:1]))
        for index, rows in enumerate(cases):
            self.write_commits(rows)
            with self.subTest(case=index), self.assertRaises(RuntimeError):
                finalize_lab_png_dataset(self.run)
            self.assertEqual((self.run / "dataset.db").read_bytes(), previous)

    def test_missing_png_fails_before_database_replacement(self):
        previous = finalize_lab_png_dataset(self.run).db_path.read_bytes()
        next((self.run / "cam1").glob("*.png")).unlink()
        with self.assertRaisesRegex(RuntimeError, "PNG/sidecar frame mismatch"):
            finalize_lab_png_dataset(self.run)
        self.assertEqual((self.run / "dataset.db").read_bytes(), previous)

    def test_missing_commit_log_and_invalid_step_are_rejected(self):
        (self.run / "capture_commits.jsonl").unlink()
        with self.assertRaisesRegex(RuntimeError, "commit log is missing"):
            finalize_lab_png_dataset(self.run)
        self.write_commits(self.commits)
        path, rows = self.sidecar("cam1")
        rows[0]["capture"]["step_id"] = True
        self.write_rows(path, rows)
        with self.assertRaisesRegex(RuntimeError, "nonnegative integer"):
            finalize_lab_png_dataset(self.run)


if __name__ == "__main__":
    unittest.main()
