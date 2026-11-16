import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from optidaq.daq.actuator_events import (
    EVENT_LOG_FILENAME,
    FRAME_INDEX_FILENAME,
    append_actuator_event,
    write_actuator_frame_index,
)
from optidaq.daq.contracts import Frame
from optidaq.daq.raw_stream import iter_stream_records
from optidaq.daq.save_backends import PngImageSaveBackend, RawStreamSaveBackend


class ActuatorEventsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.events = self.root / EVENT_LOG_FILENAME

    def event(self, event, timestamp, step_id=None, metadata=None):
        return append_actuator_event(
            self.events, event, step_id, metadata, host_time_ns=timestamp
        )

    def camera(self, alias, times):
        records = [
            {
                "device": alias,
                "frame_index": index,
                "source_image_number": index + 900,
                "acq_stamp_ns": index + 10_000,
                "local_receive_ns": timestamp,
            }
            for index, timestamp in enumerate(times)
        ]
        with (self.root / (alias + ".sync.jsonl")).open("w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        return records

    def index(self, aliases=None):
        path = write_actuator_frame_index(self.root, aliases)
        return [json.loads(line) for line in path.read_text().splitlines()]

    def closed_step(self):
        self.event("change_begin", 10, 1)
        self.event("settle_complete", 20, 1)
        self.event("end", 100)

    def test_event_is_persisted_immediately_with_extensible_metadata(self):
        with patch("optidaq.daq.actuator_events.time.time_ns", return_value=123):
            written = append_actuator_event(
                self.events, "change_begin", 1, {"source": {"future_field": [1, 2]}}
            )
        self.assertEqual(json.loads(self.events.read_text()), written)
        self.assertEqual(written["host_time_ns"], 123)
        self.assertEqual(written["clock"], "host_unix_ns")
        self.event("settle_complete", 124, 1)
        self.assertEqual(len(self.events.read_text().splitlines()), 2)
        before = self.events.read_bytes()
        with self.assertRaises(ValueError):
            self.event("unknown", 125, 1)
        self.assertEqual(self.events.read_bytes(), before)

    def test_many_frames_and_cameras_boundaries_and_regeneration(self):
        self.event("change_begin", 10, 1)
        self.event("settle_complete", 20, 1)
        self.event("change_begin", 50, 2)
        self.event("settle_complete", 60, 2)
        self.event("end", 90)
        camera1 = self.camera("cam1", [19, 20, 40, 50, 59, 60, 89, 90])
        self.camera("cam2", [22, 42, 72])
        original_files = {path: path.read_bytes() for path in self.root.iterdir()}
        result = self.index()
        self.assertEqual([row["step_id"] for row in result],
                         [None, 1, 1, None, None, 2, 2, None, 1, 1, 2])
        for row in result:
            self.assertEqual(row["method"], "host_receive_time")
            self.assertEqual(row["quality"], "unmatched" if row["step_id"] is None else "approximate")
        self.assertEqual(result[3]["reason"], "transition_boundary")
        self.assertEqual(result[7]["reason"], "transition_boundary")
        for expected, actual in zip(camera1, result):
            for key, value in expected.items():
                self.assertEqual(actual[key], value)
        self.assertEqual(self.index(), result)
        for path, content in original_files.items():
            self.assertEqual(path.read_bytes(), content)

    def test_incomplete_final_interval_stays_unmatched(self):
        self.event("change_begin", 10, 1)
        self.event("settle_complete", 20, 1)
        self.event("change_begin", 50, 2)
        self.event("settle_complete", 60, 2)
        self.camera("cam1", [30, 70])
        self.assertEqual([row["step_id"] for row in self.index()], [1, None])

    def test_missing_invalid_or_backwards_receive_times(self):
        self.closed_step()
        self.camera("cam1", [40, 30])
        self.camera("cam2", [None, "30", -1, 31])
        rows = self.index()
        self.assertEqual([row.get("reason") for row in rows[:2]],
                         ["receive_clock_regression"] * 2)
        self.assertTrue(all(row["step_id"] is None for row in rows[:5]))
        self.assertEqual(rows[-1]["step_id"], 1)
        self.assertEqual(rows[2]["reason"], "missing_or_invalid_receive_time")

    def test_event_schema_and_clock_errors_leave_old_index_intact(self):
        self.closed_step()
        self.camera("cam1", [30])
        good_events = [json.loads(line) for line in self.events.read_text().splitlines()]
        original_index = self.index()
        cases = [
            (0, "schema_version", 2, "schema_version"),
            (0, "clock", "camera_clock", "clock"),
            (1, "step_id", 999, "preceding change_begin"),
            (2, "host_time_ns", 15, "backwards"),
        ]
        for index, field, value, error in cases:
            with self.subTest(field=field):
                bad_events = [dict(record) for record in good_events]
                bad_events[index][field] = value
                self.events.write_text("".join(json.dumps(record) + "\n" for record in bad_events))
                with self.assertRaisesRegex(ValueError, error):
                    self.index()
                saved = self.root / FRAME_INDEX_FILENAME
                self.assertEqual([json.loads(line) for line in saved.read_text().splitlines()], original_index)
        self.events.write_text('{"schema_version":')
        with self.assertRaisesRegex(ValueError, r"actuator.events.jsonl:1"):
            self.index()

    def test_raw_and_png_backend_frame_references(self):
        self.closed_step()
        spec = {
            "schema_version": 1,
            "endianness": "little",
            "header_fields": [{"name": name, "type": "uint64"} for name in
                              ("frame_index", "source_image_number", "acq_stamp_ns")],
            "payload_nbytes": 1,
            "segment_max_bytes": 1000,
            "file_prefix": "raw_cam",
            "file_extension": ".raw",
        }
        backends = [
            RawStreamSaveBackend("raw_cam", self.root, spec, fsync_on_close=False),
            PngImageSaveBackend("png_cam", self.root, height=1, width=1, dtype="uint8",
                                fsync_on_close=False),
        ]
        for backend in backends:
            frame = Frame(backend.device, 321, 987654, b"\x7f")
            object.__setattr__(frame, "sync_metadata", {"local_receive_ns": 30})
            backend.open()
            try:
                backend.save(frame)
            finally:
                backend.close()
        rows = self.index()
        self.assertEqual({row["device"] for row in rows}, {"raw_cam", "png_cam"})
        for row in rows:
            self.assertEqual((row["frame_index"], row["source_image_number"], row["acq_stamp_ns"]),
                             (0, 321, 987654))
            self.assertEqual(row["step_id"], 1)
        raw_record = next(iter_stream_records(self.root, backends[0].spec))
        self.assertEqual(raw_record.header["frame_index"], 0)
        self.assertTrue((self.root / "png_cam" / "00000000000000987654_00000000.png").is_file())

    def test_cli_discovers_camera_sidecars(self):
        self.closed_step()
        self.camera("cam1", [30])
        result = subprocess.run(
            [sys.executable, "-m", "optidaq.daq.actuator_events", str(self.root)],
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(result.stdout.strip(), str(self.root / FRAME_INDEX_FILENAME))
        self.assertEqual(self.index()[0]["step_id"], 1)


if __name__ == "__main__":
    unittest.main()
