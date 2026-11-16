"""Storage contract tests using synthetic frames; no CERN hardware required."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from optidaq.daq.contracts import Frame
from optidaq.daq.pipeline import DAQPipeline, DAQPipelineError
from optidaq.daq.raw_stream import iter_stream_records
from optidaq.daq.save_backends import (
    PngImageSaveBackend,
    RawStreamSaveBackend,
    make_save_backend,
)


class PngSaveBackendTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    @staticmethod
    def frame(image, camera="cam1", index=0):
        frame = Frame(camera, 900 + index, 123456789 + index,
                      image.tobytes(), pulse_id=42 + index)
        object.__setattr__(frame, "sync_metadata", {
            "local_receive_ns": 200000000 + index,
            "image_ready_ns": 190000000 + index,
            "source": {"unchanged": True},
        })
        return frame

    @staticmethod
    def raw_spec(camera, payload_nbytes):
        return {
            "schema_version": 2,
            "endianness": "little",
            "header_fields": [{"name": name, "type": "uint64"} for name in
                              ("frame_index", "source_image_number", "acq_stamp_ns", "pulse_id")],
            "payload_nbytes": payload_nbytes,
            "segment_max_bytes": 4096,
            "file_prefix": camera,
            "file_extension": ".raw",
        }

    def save_frames(self, backend, *frames):
        backend.open()
        try:
            for frame in frames:
                backend.save(frame)
        finally:
            backend.close()

    def pixels(self, camera="cam1", index=0):
        paths = sorted((self.root / camera).glob("*.png"))
        with Image.open(paths[index]) as image:
            return np.array(image)

    def sidecar(self, camera="cam1"):
        return [json.loads(line) for line in
                (self.root / (camera + ".sync.jsonl")).read_text().splitlines()]

    def manifest(self, camera="cam1"):
        return json.loads((self.root / (camera + ".png.json")).read_text())

    def configured_backend(self, image, config, camera="cam1", image_dtype=None):
        return make_save_backend(
            camera, self.root, self.raw_spec(camera, image.nbytes),
            image_shape=image.shape,
            image_dtype=image_dtype or str(image.dtype),
            config=config,
        )

    def hook_module(self, **functions):
        module = types.ModuleType("_clear_png_test_hooks")
        for name, function in functions.items():
            setattr(module, name, function)
        module_patch = patch.dict(sys.modules, {module.__name__: module})
        module_patch.start()
        self.addCleanup(module_patch.stop)
        return module.__name__

    def assert_frame_reference(self, row, frame, index=0):
        self.assertEqual(row["frame_index"], index)
        self.assertEqual(row["device"], frame.device)
        self.assertEqual(row["source_image_number"], frame.source_image_number)
        self.assertEqual(row["imageNumber"], frame.source_image_number)
        self.assertEqual(row["acq_stamp_ns"], frame.acq_stamp_ns)
        self.assertEqual(row["raw_header_pulse_id"], frame.pulse_id)
        for key, value in frame.sync_metadata.items():
            self.assertEqual(row[key], value)

    def test_legacy_png_constructor_preserves_pixels_and_frame_identity(self):
        cases = [
            ("uint16", 1, np.array([[0, 1], [2048, 4095]], dtype=np.uint16)),
            ("uint16", 16, np.array([[0, 1], [2048, 4095]], dtype=np.uint16)),
            ("uint8", 1, np.array([[0, 1], [128, 255]], dtype=np.uint8)),
        ]
        for number, (dtype, scale, source) in enumerate(cases):
            with self.subTest(dtype=dtype, scale=scale):
                camera = "cam%d" % number
                frame = self.frame(source, camera)
                before = deepcopy(vars(frame))
                backend = PngImageSaveBackend(
                    camera, self.root, height=2, width=2, dtype=dtype,
                    value_scale=scale, fsync_on_close=False,
                )
                self.save_frames(backend, frame)
                np.testing.assert_array_equal(self.pixels(camera), source * scale)
                self.assertEqual(vars(frame), before)
                self.assert_frame_reference(self.sidecar(camera)[0], frame)
                expected_name = "%020d_%08d.png" % (frame.acq_stamp_ns, 0)
                self.assertTrue((self.root / camera / expected_name).is_file())

    def test_legacy_rgb_constructor_preserves_pixels(self):
        source = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        frame = self.frame(source)
        backend = PngImageSaveBackend(
            "cam1", self.root, height=2, width=3, channels=3, dtype="uint8",
            fsync_on_close=False,
        )
        self.save_frames(backend, frame)
        np.testing.assert_array_equal(self.pixels(), source)
        self.assert_frame_reference(self.sidecar()[0], frame)

    def test_raw_backend_keeps_original_bytes_and_metadata(self):
        source = np.array([[0, 4095], [2048, 123]], dtype=np.uint16)
        frames = [self.frame(source, index=index) for index in range(2)]
        before = [deepcopy(vars(frame)) for frame in frames]
        backend = RawStreamSaveBackend(
            "cam1", self.root, self.raw_spec("cam1", source.nbytes),
            fsync_on_close=False,
        )
        self.save_frames(backend, *frames)
        records = list(iter_stream_records(self.root, backend.spec))
        self.assertEqual(len(records), 2)
        for index, (frame, record, row) in enumerate(zip(frames, records, self.sidecar())):
            self.assertEqual(record.payload, source.tobytes())
            self.assertEqual(record.header, {
                "frame_index": index,
                "source_image_number": frame.source_image_number,
                "acq_stamp_ns": frame.acq_stamp_ns,
                "pulse_id": frame.pulse_id,
            })
            self.assert_frame_reference(row, frame, index)
            self.assertNotIn("png", row)
            self.assertEqual(vars(frame), before[index])

    def test_configured_raw_does_not_load_png_hooks(self):
        source = np.arange(16, dtype=np.uint16).reshape(4, 4)
        frame = self.frame(source)
        config = {"output": {
            "format": "raw",
            "png": {"transforms": ["nonexistent_example_module:transform"]},
        }}
        backend = self.configured_backend(source, config)
        self.assertIsInstance(backend, RawStreamSaveBackend)
        self.save_frames(backend, frame)
        record = next(iter_stream_records(self.root, backend.spec))
        self.assertEqual(record.payload, source.tobytes())
        self.assertFalse((self.root / "cam1.png.json").exists())
        self.assertFalse((self.root / "cam1").exists())

    def test_per_camera_crop_bin_records_stored_geometry_and_metadata(self):
        source = np.arange(64, dtype=np.uint16).reshape(8, 8)
        transforms = [
            {"name": "center_crop", "params": {"size": 4}},
            {"name": "avg_bin", "params": {"factor": 2, "dtype": "uint16"}},
        ]
        config = {"output": {"format": "png", "png": {
            "mode": "raw16", "compress_level": 6,
            "transforms": {"cam1": transforms, "cam2": []},
            "metadata_hooks": {"cam1": ["stats"], "cam2": []},
        }}}
        expected = np.array([[22, 24], [38, 40]], dtype=np.uint16)
        for camera, pixels in (("cam1", expected), ("cam2", source)):
            with self.subTest(camera=camera):
                frame = self.frame(source, camera)
                before = deepcopy(vars(frame))
                # PyDA can describe mono12 payloads using signed int16 storage.
                backend = self.configured_backend(source, config, camera, "int16")
                self.assertIsInstance(backend, PngImageSaveBackend)
                self.save_frames(backend, frame)
                np.testing.assert_array_equal(self.pixels(camera), pixels)
                manifest = self.manifest(camera)
                self.assertEqual(manifest["schema_version"], 1)
                self.assertEqual(manifest["source"]["shape"], [8, 8])
                self.assertEqual(manifest["source"]["dtype"], "int16")
                self.assertEqual(manifest["stored"]["shape"], list(pixels.shape))
                self.assertEqual(manifest["stored"]["dtype"], "uint16")
                self.assertEqual(manifest["stored"]["value_scale"], 1)
                self.assertEqual(manifest["transforms"], transforms if camera == "cam1" else [])
                row = self.sidecar(camera)[0]
                self.assert_frame_reference(row, frame)
                self.assertEqual(row["png"]["shape"], list(pixels.shape))
                self.assertEqual(row["png"]["dtype"], "uint16")
                if camera == "cam1":
                    self.assertEqual(row["png"]["metadata"]["stat_sum"], float(expected.sum()))
                    self.assertEqual(row["png"]["metadata"]["stat_mean"], float(expected.mean()))
                else:
                    self.assertEqual(row["png"]["metadata"], {})
                self.assertEqual(vars(frame), before)

    def test_custom_hooks_get_context_and_cannot_change_source_frame(self):
        source = np.arange(16, dtype=np.uint16).reshape(4, 4)
        calls = []

        def add_offset(image, context, *, offset):
            calls.append(context)
            image += offset  # A custom in-place transform must use a working copy.
            return image

        def measure(image, context):
            self.assertFalse(image.flags.writeable)
            self.assertFalse(context.source_image.flags.writeable)
            np.testing.assert_array_equal(context.source_image, source)
            return {"source_sum": int(context.source_image.sum()),
                    "stored_sum": int(image.sum())}

        module = self.hook_module(add_offset=add_offset, measure=measure)
        transforms = [{"name": module + ":add_offset", "params": {"offset": 7}}]
        config = {"output": {"format": "png", "png": {
            "transforms": transforms,
            "metadata_hooks": [module + ":measure"],
        }}}
        frame = self.frame(source)
        before = deepcopy(vars(frame))
        self.save_frames(self.configured_backend(source, config), frame)
        np.testing.assert_array_equal(self.pixels(), source + 7)
        self.assertEqual(vars(frame), before)
        self.assertEqual(len(calls), 1)
        context = calls[0]
        self.assertEqual(context.camera, frame.device)
        self.assertEqual(context.frame_index, 0)
        self.assertEqual(context.source_image_number, frame.source_image_number)
        self.assertEqual(context.acq_stamp_ns, frame.acq_stamp_ns)
        self.assertEqual(context.pulse_id, frame.pulse_id)
        self.assertEqual(context.sync, frame.sync_metadata)
        self.assertEqual(self.sidecar()[0]["png"]["metadata"], {
            "source_sum": int(source.sum()), "stored_sum": int((source + 7).sum()),
        })
        self.assertEqual(self.manifest()["transforms"], transforms)

    def test_hook_context_does_not_share_nested_sync_metadata_with_frame(self):
        source = np.arange(4, dtype=np.uint16).reshape(2, 2)

        def change_context(image, context):
            context.sync["source"]["unchanged"] = False
            return image

        module = self.hook_module(change_context=change_context)
        config = {"output": {"format": "png", "png": {
            "transforms": [module + ":change_context"],
        }}}
        frame = self.frame(source)
        before = deepcopy(vars(frame))
        self.save_frames(self.configured_backend(source, config), frame)
        self.assertEqual(vars(frame), before)
        self.assertEqual(self.sidecar()[0]["source"], {"unchanged": True})

    def test_crop_energy_measures_source_roi_before_binning(self):
        source = np.arange(64, dtype=np.uint16).reshape(8, 8)
        cases = [
            ({"name": "center_crop", "params": {"size": 4}}, source[2:6, 2:6]),
            ({"name": "crop_box", "params": {"x1": 2, "y1": 1, "x2": 6, "y2": 5}},
             source[1:5, 2:6]),
        ]
        for index, (crop, roi) in enumerate(cases):
            with self.subTest(crop=crop["name"]):
                camera = "cam%d" % index
                config = {"output": {"format": "png", "png": {
                    "transforms": [crop, {"name": "avg_bin", "params": {
                        "factor": 2, "dtype": "uint16",
                    }}],
                    "metadata_hooks": ["crop_energy"],
                }}}
                self.save_frames(self.configured_backend(source, config, camera),
                                 self.frame(source, camera))
                metadata = self.sidecar(camera)[0]["png"]["metadata"]
                self.assertEqual(metadata["intensity_in_roi"], float(roi.sum()))
                self.assertEqual(metadata["intensity_cropped_out"],
                                 float(source.sum() - roi.sum()))
                self.assertAlmostEqual(metadata["roi_energy_fraction"],
                                       float(roi.sum()) / float(source.sum()))
                self.assertEqual(self.pixels(camera).shape, (2, 2))

    def test_explicit_uint8_hook_uses_fixed_scale(self):
        source = np.array([[0, 1365], [2730, 4095]], dtype=np.uint16)
        config = {"output": {"format": "png", "png": {
            "transforms": [{"name": "to_uint8", "params": {"max_value": 4095}}],
        }}}
        frame = self.frame(source)
        self.save_frames(self.configured_backend(source, config), frame)
        np.testing.assert_array_equal(self.pixels(), np.array([[0, 85], [170, 255]], dtype=np.uint8))
        self.assertEqual(self.manifest()["stored"]["dtype"], "uint8")

    def test_legacy_flat_png_mode_and_nested_mode_precedence(self):
        source = np.array([[1, 4095]], dtype=np.uint16)
        cases = [
            ({"format": "png", "png_mode": "mono12_display16"}, 16),
            ({"format": "png", "png_mode": "mono12_display16",
              "png": {"mode": "raw16"}}, 1),
        ]
        for index, (output, scale) in enumerate(cases):
            with self.subTest(scale=scale):
                camera = "cam%d" % index
                frame = self.frame(source, camera)
                self.save_frames(self.configured_backend(source, {"output": output}, camera), frame)
                np.testing.assert_array_equal(self.pixels(camera), source * scale)
                self.assertEqual(self.manifest(camera)["stored"]["value_scale"], scale)

    def test_invalid_processed_images_do_not_publish_or_consume_frame_index(self):
        source = np.arange(16, dtype=np.uint16).reshape(4, 4)
        invalid_outputs = [
            source.astype(np.float32),
            source[:0, :],
            source.reshape(2, 2, 4),
        ]
        for index, invalid in enumerate(invalid_outputs):
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype):
                camera = "cam%d" % index
                selected = [invalid]

                def transform(image, context):
                    return selected[0]

                module = self.hook_module(transform=transform)
                config = {"output": {"format": "png", "png": {
                    "transforms": [module + ":transform"],
                }}}
                frame = self.frame(source, camera)
                before = deepcopy(vars(frame))
                backend = self.configured_backend(source, config, camera)
                backend.open()
                try:
                    with self.assertRaises((ValueError, RuntimeError)):
                        backend.save(frame)
                    self.assertEqual(list((self.root / camera).glob("*.png")), [])
                    self.assertFalse((self.root / (camera + ".sync.jsonl")).exists())
                    self.assertFalse((self.root / (camera + ".png.json")).exists())
                    self.assertEqual(vars(frame), before)
                    selected[0] = source
                    backend.save(frame)
                finally:
                    backend.close()
                self.assertEqual(len(self.sidecar(camera)), 1)
                self.assert_frame_reference(self.sidecar(camera)[0], frame, 0)
                np.testing.assert_array_equal(self.pixels(camera), source)

    def test_changing_output_geometry_rejects_second_frame(self):
        source = np.arange(16, dtype=np.uint16).reshape(4, 4)

        def variable_crop(image, context):
            return image if context.frame_index == 0 else image[:2, :2]

        module = self.hook_module(variable_crop=variable_crop)
        config = {"output": {"format": "png", "png": {
            "transforms": [module + ":variable_crop"],
        }}}
        backend = self.configured_backend(source, config)
        first, second = self.frame(source), self.frame(source, index=1)
        backend.open()
        try:
            backend.save(first)
            with self.assertRaises((ValueError, RuntimeError)):
                backend.save(second)
        finally:
            backend.close()
        self.assertEqual(len(list((self.root / "cam1").glob("*.png"))), 1)
        self.assertEqual(len(self.sidecar()), 1)
        self.assert_frame_reference(self.sidecar()[0], first)
        self.assertEqual(self.manifest()["stored"]["shape"], [4, 4])

    def test_metadata_hook_failure_does_not_publish_image(self):
        source = np.arange(4, dtype=np.uint16).reshape(2, 2)

        def fail(image, context):
            raise RuntimeError("metadata failed")

        module = self.hook_module(fail=fail)
        config = {"output": {"format": "png", "png": {
            "metadata_hooks": [module + ":fail"],
        }}}
        backend = self.configured_backend(source, config)
        backend.open()
        try:
            with self.assertRaisesRegex(RuntimeError, "metadata failed"):
                backend.save(self.frame(source))
        finally:
            backend.close()
        self.assertEqual(list((self.root / "cam1").glob("*.png")), [])
        self.assertFalse((self.root / "cam1.sync.jsonl").exists())

    def test_png_without_optional_sync_still_records_source_identity(self):
        source = np.arange(4, dtype=np.uint16).reshape(2, 2)
        frame = Frame("cam1", 54, 987654, source.tobytes(), pulse_id=765)
        config = {"output": {"format": "png"}}
        self.save_frames(self.configured_backend(source, config), frame)
        row = self.sidecar()[0]
        self.assertEqual(row["frame_index"], 0)
        self.assertEqual(row["source_image_number"], 54)
        self.assertEqual(row["acq_stamp_ns"], 987654)
        self.assertEqual(row["raw_header_pulse_id"], 765)
        self.assertEqual(row["png"]["shape"], [2, 2])
        np.testing.assert_array_equal(self.pixels(), source)

    def test_pipeline_surfaces_hook_failure_without_marking_frame_saved(self):
        source = np.arange(4, dtype=np.uint16).reshape(2, 2)
        config = {"output": {"format": "png", "png": {
            # normalize returns float32; PNG needs an explicit integer conversion.
            "transforms": ["normalize"],
        }}}
        backend = self.configured_backend(source, config)
        pipeline = DAQPipeline(taps=[], backends={"cam1": backend}, queue_maxsize=2)
        pipeline.start()
        pipeline.submit(self.frame(source))
        with self.assertRaisesRegex(DAQPipelineError, "PNG hooks must return"):
            pipeline.stop(drain_timeout_s=2)
        self.assertEqual(pipeline.stats().accepted_count, 1)
        self.assertEqual(pipeline.stats().saved_count, 0)
        self.assertIsNotNone(pipeline.stats().error)
        self.assertEqual(list((self.root / "cam1").glob("*.png")), [])


if __name__ == "__main__":
    unittest.main()
