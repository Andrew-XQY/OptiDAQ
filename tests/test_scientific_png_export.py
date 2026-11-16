from optidaq.data.scientific_png_export import _build_sync_alignment


class _FakeReader:
    cameras = ["cam1", "cam2"]
    sync_by_camera = {
        "cam1": {
            0: {"image_ready_ns": 100},
            1: {"image_ready_ns": 130},
            2: {"image_ready_ns": 300},
        },
        "cam2": {
            0: {"image_ready_ns": 110},
            1: {"image_ready_ns": 320},
        },
    }

    def raw_record_count(self, camera):
        return len(self.sync_by_camera[camera])


def test_duplicate_camera_drops_entire_sync_group():
    alignment = _build_sync_alignment(
        _FakeReader(), source="image_ready_ns", tolerance_ns=50
    )

    assert alignment.frame_to_sample_id == {"cam1": {2: 2}, "cam2": {1: 2}}
    assert alignment.dropped_group_count == 1
    assert alignment.dropped_frame_count == 3
    assert alignment.complete_group_count == 1
