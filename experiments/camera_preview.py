"""Preview cameras using the experiment config.

Run from the repository root in the CERN acc-py environment:

    python experiments/camera_preview.py --config configs/experiment_v1.json

The default preview applies camera settings and offers exposure/gain controls.
Use both --no-configure and --no-control for a preview without camera writes.
No run directory or image files are saved.
Runtime exposure controls use each camera's live CameraSetting min/max limits.

By default frames are shown with true values on the sensor's full scale (0 to
4095 for mono12); the runtime command ``scale auto`` stretches each frame from
its own min to max for viewing only.
The P99.9 readout is the 99.9th percentile of the acquired image, alongside the
true max. It reduces sensitivity to isolated hot pixels but can miss tiny spots.
"""

from __future__ import annotations

import argparse
import base64
from collections import deque
from collections.abc import Callable, Mapping as MappingABC
import json
from pathlib import Path
import queue
import signal
import struct
import sys
import threading
import time
from typing import Any, Optional
import zlib

import numpy as np

try:
    import tkinter as tk
except Exception:  # pragma: no cover - depends on the CERN VM GUI stack
    tk = None


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from optidaq.daq.contracts import Frame  # noqa: E402
from optidaq.daq.sync_tools import (  # noqa: E402
    pulse_id_candidate_fields,
    tap_capture_mode,
    tap_poll_interval_s,
    validate_poll_rate,
    validate_tap_stamp_sources,
)
from optidaq.extensions.cern_pyda import (  # noqa: E402
    PollingCameraTap,
    SubscribingCameraTap,
    camera_map_from_config,
    camera_binning_factors,
    camera_pixel_format_value,
    camera_setting_for_alias,
    camera_setting_payload,
    ensure_cameras_alive,
    camera_roi_enabled,
    fesa_roi_from_camera_box,
    get_value,
    make_client,
    read_last_image,
    set_value,
)


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "experiment_v1.json"
DISPLAY_CONFIG = {
    "scale": 0.8,
    "max_width": 1400,
    "refresh_hz": 20.0,
    "window": "OptiDAQ camera preview",
    "text_color": "yellow",
    "fixed_scale": True,  # True: 0..sensor max (true values); False: per-frame min-max
}
GRID_CONFIG = {
    "enabled": True,
    "size_px": 200,
    "dash_px": 12,
    "gap_px": 12,
    "thickness_px": 1,
    "gray_value": 150,
    "opacity": 0.18,
    "show_label": True,
}
ROI_PREVIEW_CONFIG = {
    "apply_camera_roi": True,
    "show_box": True,
    "shade_outside": True,
    "outside_value": 40,
}
GRID_MAX_COLS = 2  # tiles fill up to this many columns, then add rows
STOP_REQUESTED = False
GAIN_MIN = 0.0
GAIN_MAX = 1.0
CAMERA_AUTO_OFF_VALUE = 1
PIXEL_FORMAT_FULL_SCALE = {40: 4095, 44: 255}  # BTVDC MONO_12, MONO_8
CAMERA_READBACK_TIMEOUT_S = 5.0
CAMERA_READBACK_POLL_S = 0.2
GAIN_READBACK_TOLERANCE = 1.0e-4


def request_stop(signum: int, frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    for section in ("rbac", "cameras", "camera_setting", "tap", "sync"):
        if section not in cfg:
            raise RuntimeError("config missing section %r" % section)
    if "scan" in cfg:
        validate_poll_rate(cfg["scan"], cfg["tap"], cfg["sync"])
    return cfg


def output_configures_cameras(cfg: MappingABC[str, Any]) -> bool:
    writer = cfg.get("writer", {})
    if not isinstance(writer, MappingABC):
        return True
    return bool(writer.get("configure_cameras", True))


class PreviewDynamicShapeMixin:
    """Preview-only tap behavior for interactive ROI size changes."""

    def _build_and_emit(
        self,
        data: MappingABC[str, Any],
        emit: Callable[[Frame], None],
    ) -> None:
        with self._preview_shape_lock:
            self._build_and_emit_dynamic_shape(data, emit)

    def _build_and_emit_dynamic_shape(
        self,
        data: MappingABC[str, Any],
        emit: Callable[[Frame], None],
    ) -> None:
        image = np.ascontiguousarray(np.asarray(data["image2D"]))
        shape = tuple(int(part) for part in image.shape)
        dtype = str(image.dtype)
        payload_nbytes = int(image.nbytes)
        changed = (
            self.image_shape is not None
            and (
                tuple(int(part) for part in self.image_shape) != shape
                or self.image_dtype != dtype
                or self.payload_nbytes != payload_nbytes
            )
        )

        self.image_shape = shape
        self.image_dtype = dtype
        self.payload_nbytes = payload_nbytes
        if changed:
            print(
                "camera preview tap %s image shape updated: shape=%s dtype=%s "
                "payload_nbytes=%d"
                % (self.device, shape, dtype, payload_nbytes),
                flush=True,
            )

        def emit_with_shape(frame: Frame) -> None:
            object.__setattr__(frame, "image_shape", shape)
            object.__setattr__(frame, "image_dtype", dtype)
            emit(frame)

        super()._build_and_emit(data, emit_with_shape)


class PreviewPollingCameraTap(PreviewDynamicShapeMixin, PollingCameraTap):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._preview_shape_lock = threading.Lock()


class PreviewSubscribingCameraTap(PreviewDynamicShapeMixin, SubscribingCameraTap):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._preview_shape_lock = threading.Lock()


def build_camera_taps(
    cfg: dict,
    client: Any,
    *,
    allow_image_number_reset: bool = False,
) -> list[Any]:
    camera_map = camera_map_from_config(cfg)

    sync_cfg = cfg["sync"]
    pulse_id_fields = pulse_id_candidate_fields(sync_cfg)
    tap_mode = tap_capture_mode(cfg["tap"])
    use_subscribe = tap_mode == "subscribe"
    common_tap_kwargs = dict(
        get_selector=cfg["tap"].get("get_selector"),
        poll_s=tap_poll_interval_s(cfg["tap"], required=not use_subscribe),
        stamp_fields=cfg["tap"]["stamp_fields"],
        min_ns_epoch=int(cfg["tap"]["min_ns_epoch"]),
        pulse_id_fields=pulse_id_fields,
        settle_poll_s=float(cfg["tap"].get("settle_poll_s", 0.005)),
        settle_timeout_s=float(cfg["tap"].get("settle_timeout_s", 0.25)),
        max_trailing_zero_rows=cfg["tap"].get("max_trailing_zero_rows", 0),
        sync_cfg=sync_cfg,
        allow_image_number_reset=allow_image_number_reset,
    )

    taps = []
    for alias, fesa in camera_map.items():
        if use_subscribe:
            tap = PreviewSubscribingCameraTap(
                alias,
                fesa,
                client,
                selector=cfg["tap"].get("subscribe_selector"),
                **common_tap_kwargs,
            )
        else:
            tap = PreviewPollingCameraTap(alias, fesa, client, **common_tap_kwargs)
        taps.append(tap)

    print("camera preview tap mode: %s" % tap_mode)
    return taps


def read_camera_exposure_limits(client: Any, fesa: str) -> tuple[int, int]:
    """Read the allowed exposure range in microseconds; never guess a limit."""
    setting = get_value(client, "%s/CameraSetting" % fesa)
    try:
        minimum = int(setting["cameraExposureTimeUs_min"])
        maximum = int(setting["cameraExposureTimeUs_max"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("camera exposure limits are missing or invalid") from exc
    if not 0 < minimum <= maximum <= np.iinfo(np.int64).max:
        raise RuntimeError("invalid camera exposure range: %s..%s us" % (minimum, maximum))
    return minimum, maximum


def read_current_camera_controls(
    client: Any,
    fesa: str,
    fallback_exposure_us: int,
    fallback_gain: float,
) -> tuple[int, float]:
    readback = read_camera_control_readback(
        client,
        fesa,
        fallback_exposure_us,
        fallback_gain,
    )
    return int(readback["exposure_us"]), float(readback["gain"])


def read_camera_control_readback(
    client: Any,
    fesa: str,
    fallback_exposure_us: int,
    fallback_gain: float,
) -> dict[str, Any]:
    try:
        params = get_value(client, "%s/AcquiredParameters" % fesa)
    except Exception as exc:
        return {
            "exposure_us": int(fallback_exposure_us),
            "gain": float(fallback_gain),
            "params": {},
            "error": str(exc),
            "matched": False,
        }

    exposure = params.get("acqCameraExposureTimeUs", fallback_exposure_us)
    gain = params.get("acqCameraGain", fallback_gain)
    try:
        exposure_us = int(exposure)
    except (TypeError, ValueError):
        exposure_us = int(fallback_exposure_us)
    try:
        gain_value = float(gain)
    except (TypeError, ValueError):
        gain_value = float(fallback_gain)
    return {
        "exposure_us": exposure_us,
        "gain": gain_value,
        "params": params,
        "error": None,
        "matched": False,
    }


def camera_control_readback_matches(
    readback: MappingABC[str, Any],
    expected_exposure_us: int,
    expected_gain: float,
) -> bool:
    if readback.get("error"):
        return False
    exposure_ok = abs(int(readback["exposure_us"]) - int(expected_exposure_us)) <= 1
    gain_ok = (
        abs(float(readback["gain"]) - float(expected_gain))
        <= GAIN_READBACK_TOLERANCE
    )
    return exposure_ok and gain_ok


def wait_camera_control_readback(
    client: Any,
    alias: str,
    fesa: str,
    *,
    expected_exposure_us: int,
    expected_gain: float,
    timeout_s: float = CAMERA_READBACK_TIMEOUT_S,
) -> dict[str, Any]:
    deadline = time.monotonic() + float(timeout_s)
    last = read_camera_control_readback(
        client,
        fesa,
        expected_exposure_us,
        expected_gain,
    )
    while True:
        if camera_control_readback_matches(
            last,
            expected_exposure_us,
            expected_gain,
        ):
            last["matched"] = True
            last["warning"] = None
            return last
        if time.monotonic() >= deadline:
            break
        time.sleep(CAMERA_READBACK_POLL_S)
        last = read_camera_control_readback(
            client,
            fesa,
            expected_exposure_us,
            expected_gain,
        )

    warning = (
        "camera %s readback mismatch after CameraSetting: requested "
        "exposure=%d us gain=%.4g, got exposure=%d us gain=%.4g"
        % (
            alias,
            int(expected_exposure_us),
            float(expected_gain),
            int(last["exposure_us"]),
            float(last["gain"]),
        )
    )
    if last.get("error"):
        warning += " (%s)" % last["error"]
    print("WARNING: %s" % warning, flush=True)
    last["matched"] = False
    last["warning"] = warning
    return last


def camera_control_expected(setting: MappingABC[str, Any]) -> tuple[int, float]:
    return (
        int(setting.get("cameraExposureTimeUs", 10000)),
        float(setting.get("cameraGain", 0.0)),
    )



def config_roi_box(setting: MappingABC[str, Any]) -> Optional[tuple[int, int, int, int]]:
    if not camera_roi_enabled(setting):
        return None
    roi = setting.get("roi")
    if not isinstance(roi, MappingABC):
        return None
    width = int(roi["width"])
    height = int(roi["height"])
    center_x = int(roi["offsetX"])
    center_y = int(roi["offsetY"])
    x1 = center_x - width // 2
    y1 = center_y - height // 2
    return normalize_box(x1, y1, x1 + width, y1 + height)


def normalize_box(x1: Any, y1: Any, x2: Any, y2: Any) -> tuple[int, int, int, int]:
    left = int(round(float(x1)))
    top = int(round(float(y1)))
    right = int(round(float(x2)))
    bottom = int(round(float(y2)))
    if right < left:
        left, right = right, left
    if bottom < top:
        top, bottom = bottom, top
    if right == left or bottom == top:
        raise RuntimeError("box must have non-zero width and height")
    return left, top, right, bottom


def clip_box_to_shape(
    box: Optional[tuple[int, int, int, int]],
    shape: Optional[tuple[int, ...]],
) -> Optional[tuple[int, int, int, int]]:
    if box is None:
        return None
    if not shape or len(shape) < 2:
        return box
    height, width = int(shape[0]), int(shape[1])
    left, top, right, bottom = box
    left = max(0, min(width, left))
    right = max(0, min(width, right))
    top = max(0, min(height, top))
    bottom = max(0, min(height, bottom))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def initial_roi_boxes(
    cfg: dict,
    taps: list[Any],
    *,
    apply_camera_roi: bool,
    display_shapes: MappingABC[str, tuple[int, int]],
) -> dict[str, Optional[tuple[int, int, int, int]]]:
    aliases = [tap.device for tap in taps]
    boxes = {}
    for tap in taps:
        setting = camera_setting_for_alias(tap.device, cfg["camera_setting"], aliases)
        config_box = config_roi_box(setting)
        display_shape = display_shapes[tap.device]
        boxes[tap.device] = clip_box_to_shape(config_box, display_shape)
    return boxes


def initial_display_shapes(
    cfg: dict,
    taps: list[Any],
    full_shapes: MappingABC[str, Optional[tuple[int, int]]],
) -> dict[str, tuple[int, int]]:
    aliases = [tap.device for tap in taps]
    shapes = {}
    for tap in taps:
        shape = full_shapes.get(tap.device)
        if shape is None:
            shape = tuple(int(part) for part in (tap.image_shape or (512, 512))[:2])
            setting = camera_setting_for_alias(tap.device, cfg["camera_setting"], aliases)
            bin_x, bin_y = camera_binning_factors(setting)
            shape = (int(shape[0]) * bin_y, int(shape[1]) * bin_x)
        shapes[tap.device] = (int(shape[0]), int(shape[1]))
    return shapes


def initial_source_boxes(
    cfg: dict,
    taps: list[Any],
    *,
    apply_camera_roi: bool,
    display_shapes: MappingABC[str, tuple[int, int]],
) -> dict[str, Optional[tuple[int, int, int, int]]]:
    aliases = [tap.device for tap in taps]
    boxes = {}
    for tap in taps:
        display_shape = display_shapes[tap.device]
        height, width = display_shape
        if apply_camera_roi:
            setting = camera_setting_for_alias(tap.device, cfg["camera_setting"], aliases)
            source_box = clip_box_to_shape(config_roi_box(setting), display_shape)
        else:
            source_box = (0, 0, width, height)
        boxes[tap.device] = source_box
    return boxes


def format_box(box: Optional[tuple[int, int, int, int]]) -> str:
    if box is None:
        return "none"
    return "%d,%d -> %d,%d" % box


def full_shape_from_setting(setting: MappingABC[str, Any]) -> Optional[tuple[int, int]]:
    window = np.asarray(setting.get("imageWindow"), dtype=np.int32).reshape(-1)
    if window.size < 4:
        return None
    width = int(window[2])
    height = int(window[3])
    if width <= 0 or height <= 0:
        return None
    return height, width


def display_shape_from_binned_shape(
    shape: Optional[tuple[int, int]],
    setting: MappingABC[str, Any],
) -> Optional[tuple[int, int]]:
    if shape is None:
        return None
    bin_x, bin_y = camera_binning_factors(setting)
    return int(shape[0]) * bin_y, int(shape[1]) * bin_x


def box_inside_shape(
    box: Optional[tuple[int, int, int, int]],
    shape: tuple[int, int],
) -> bool:
    if box is None:
        return False
    height, width = int(shape[0]), int(shape[1])
    left, top, right, bottom = box
    return 0 <= left < right <= width and 0 <= top < bottom <= height


def display_shape_from_fesa_window(
    shape: Optional[tuple[int, int]],
    setting: MappingABC[str, Any],
) -> Optional[tuple[int, int]]:
    if shape is None:
        return None
    display_shape = (int(shape[0]), int(shape[1]))
    if box_inside_shape(config_roi_box(setting), display_shape):
        return display_shape
    return display_shape_from_binned_shape(display_shape, setting)


def wait_preview_image_shape(
    client: Any,
    alias: str,
    fesa: str,
    expected_shape: Optional[tuple[int, int]],
    label: str,
    *,
    get_selector: Optional[str] = None,
    use_subscription: bool = False,
    subscribe_selector: Optional[str] = None,
) -> None:
    if expected_shape is None:
        return

    if use_subscription:
        read_last_image(
            client, fesa, use_subscription=True,
            subscribe_selector=subscribe_selector, expected_shape=expected_shape,
        )
        return

    deadline = time.monotonic() + 10.0
    last_shape: tuple[int, ...] = ()
    last_image_number: Any = None
    while time.monotonic() < deadline:
        data = get_value(client, "%s/LastImage" % fesa, context=get_selector)
        image = np.asarray(data["image2D"])
        last_shape = tuple(int(part) for part in image.shape)
        last_image_number = data.get("imageNumber")
        if len(last_shape) >= 2 and last_shape[:2] == expected_shape:
            return
        time.sleep(0.2)

    raise RuntimeError(
        "camera %s LastImage %s shape mismatch after preview ROI update: "
        "expected %s, last imageNumber=%s shape=%s"
        % (alias, label, expected_shape, last_image_number, last_shape)
    )


def apply_preview_roi(
    client: Any,
    alias: str,
    fesa: str,
    setting: MappingABC[str, Any],
) -> Optional[tuple[int, int]]:
    box = config_roi_box(setting)
    if box is None:
        return None

    fesa_roi = fesa_roi_from_camera_box(alias, setting, box)
    _fesa_left, _fesa_top, fesa_width, fesa_height = fesa_roi
    current = get_value(client, "%s/Setting" % fesa)
    payload = {
        "imageROIEnable": True,
        "imageWindow": np.asarray(current["imageWindow"], dtype=np.int32),
        "imageROI": np.asarray(fesa_roi, dtype=np.int32),
    }
    set_value(client, "%s/Setting" % fesa, payload)

    deadline = time.monotonic() + 10.0
    last_enabled: Optional[bool] = None
    last_roi: tuple[int, ...] = ()
    while time.monotonic() < deadline:
        params = get_value(client, "%s/AcquiredParameters" % fesa)
        if "acqImageROIEnable" not in params or "acqImageROI" not in params:
            return fesa_height, fesa_width
        last_enabled = bool(params.get("acqImageROIEnable"))
        last_roi = tuple(
            int(value) for value in np.asarray(params.get("acqImageROI")).reshape(-1)
        )
        if last_enabled and last_roi == fesa_roi:
            return fesa_height, fesa_width
        time.sleep(0.2)

    raise RuntimeError(
        "camera %s ROI apply readback mismatch: enabled=%s roi=%s"
        % (alias, last_enabled, list(last_roi))
    )


def set_preview_roi_box(
    client: Any,
    alias: str,
    fesa: str,
    box: tuple[int, int, int, int],
    camera_setting: MappingABC[str, Any],
    *,
    get_selector: Optional[str] = None,
    use_subscription: bool = False,
    subscribe_selector: Optional[str] = None,
) -> tuple[int, int, int, int]:
    fesa_roi = fesa_roi_from_camera_box(alias, camera_setting, box)
    _left, _top, width, height = fesa_roi
    expected_shape = (height, width)
    setting = get_value(client, "%s/Setting" % fesa)
    payload = {
        "imageROIEnable": True,
        "imageWindow": np.asarray(setting["imageWindow"], dtype=np.int32),
        "imageROI": np.asarray(fesa_roi, dtype=np.int32),
    }
    set_value(client, "%s/Setting" % fesa, payload)

    deadline = time.monotonic() + 10.0
    last_enabled: Optional[bool] = None
    last_roi: tuple[int, ...] = ()
    while time.monotonic() < deadline:
        params = get_value(client, "%s/AcquiredParameters" % fesa)
        if "acqImageROIEnable" not in params or "acqImageROI" not in params:
            wait_preview_image_shape(
                client, alias, fesa, expected_shape, "runtime ROI",
                get_selector=get_selector,
                use_subscription=use_subscription, subscribe_selector=subscribe_selector,
            )
            return box
        last_enabled = bool(params.get("acqImageROIEnable"))
        last_roi = tuple(
            int(value) for value in np.asarray(params.get("acqImageROI")).reshape(-1)
        )
        if last_enabled and last_roi == fesa_roi:
            wait_preview_image_shape(
                client, alias, fesa, expected_shape, "runtime ROI",
                get_selector=get_selector,
                use_subscription=use_subscription, subscribe_selector=subscribe_selector,
            )
            return box
        time.sleep(0.2)

    raise RuntimeError(
        "camera %s runtime ROI readback mismatch: enabled=%s roi=%s"
        % (alias, last_enabled, list(last_roi))
    )


def disable_preview_roi(client: Any, alias: str, fesa: str) -> Optional[tuple[int, int]]:
    setting = get_value(client, "%s/Setting" % fesa)
    expected_shape = full_shape_from_setting(setting)
    payload = {
        "imageROIEnable": False,
        "imageWindow": np.asarray(setting["imageWindow"], dtype=np.int32),
        "imageROI": np.asarray(setting["imageROI"], dtype=np.int32),
    }
    set_value(client, "%s/Setting" % fesa, payload)

    deadline = time.monotonic() + 10.0
    last_enabled: Optional[bool] = None
    while time.monotonic() < deadline:
        params = get_value(client, "%s/AcquiredParameters" % fesa)
        if "acqImageROIEnable" not in params:
            return expected_shape
        last_enabled = bool(params.get("acqImageROIEnable"))
        if not last_enabled:
            return expected_shape
        time.sleep(0.2)

    raise RuntimeError(
        "camera %s ROI disable readback mismatch: enabled=%s"
        % (alias, last_enabled)
    )


def configure_preview_cameras(
    client: Any,
    camera_map: MappingABC[str, str],
    camera_setting: MappingABC[str, Any],
    *,
    apply_camera_roi: bool,
    get_selector: Optional[str] = None,
    use_subscription: bool = False,
    subscribe_selector: Optional[str] = None,
) -> tuple[
    dict[str, Optional[tuple[int, int]]],
    dict[str, dict[str, Any]],
]:
    aliases = list(camera_map)
    full_shapes: dict[str, Optional[tuple[int, int]]] = {}
    control_readbacks: dict[str, dict[str, Any]] = {}
    for alias, fesa in camera_map.items():
        setting = camera_setting_for_alias(alias, camera_setting, aliases)
        payload = camera_setting_payload(setting)
        expected_exposure_us, expected_gain = camera_control_expected(setting)
        if payload:
            set_value(client, "%s/CameraSetting" % fesa, payload)
        fesa_setting = get_value(client, "%s/Setting" % fesa)
        fesa_full_shape = full_shape_from_setting(fesa_setting)
        full_shapes[alias] = display_shape_from_fesa_window(fesa_full_shape, setting)
        use_camera_roi = apply_camera_roi and camera_roi_enabled(setting)
        mode = "camera_roi" if use_camera_roi else "full_frame"
        print(
            "configure preview camera %s %s %s fesa_shape=%s display_shape=%s overlay_box=%s"
            % (
                alias,
                fesa,
                mode,
                fesa_full_shape,
                full_shapes[alias],
                format_box(config_roi_box(setting)),
            )
        )
        if use_camera_roi:
            expected_shape = apply_preview_roi(client, alias, fesa, setting)
            wait_preview_image_shape(
                client, alias, fesa, expected_shape, "ROI", get_selector=get_selector,
                use_subscription=use_subscription, subscribe_selector=subscribe_selector,
            )
        else:
            expected_shape = disable_preview_roi(client, alias, fesa)
            wait_preview_image_shape(
                client, alias, fesa, expected_shape, "full-frame",
                get_selector=get_selector,
                use_subscription=use_subscription, subscribe_selector=subscribe_selector,
            )

        readback = wait_camera_control_readback(
            client,
            alias,
            fesa,
            expected_exposure_us=expected_exposure_us,
            expected_gain=expected_gain,
        )
        control_readbacks[alias] = readback
        params = readback.get("params", {})
        print(
            "  roi_enabled=%s exposure_us=%s gain=%s pixel_format=%s"
            % (
                params.get("acqImageROIEnable"),
                readback["exposure_us"],
                readback["gain"],
                params.get("acqCameraPixelFormat"),
            )
        )
    return full_shapes, control_readbacks


class RuntimeCameraControl:
    def __init__(
        self,
        client: Any,
        taps: list[Any],
        cfg: dict,
        *,
        boxes: MappingABC[str, Optional[tuple[int, int, int, int]]],
        display_shapes: MappingABC[str, tuple[int, int]],
        source_boxes: MappingABC[str, Optional[tuple[int, int, int, int]]],
        control_readbacks: MappingABC[str, MappingABC[str, Any]],
        apply_camera_roi: bool,
        frame_emit: Optional[Callable[[Frame], None]] = None,
    ) -> None:
        self.client = client
        self.frame_emit = frame_emit
        self.commands: "queue.Queue[str]" = queue.Queue()
        self.quit_requested = False
        self.write_interval_s = 0.25
        self.failure_backoff_s = 2.0
        self.apply_camera_roi = bool(apply_camera_roi)
        self.get_selector = cfg["tap"].get("get_selector")
        self.use_subscription = tap_capture_mode(cfg["tap"]) == "subscribe"
        self.subscribe_selector = cfg["tap"].get("subscribe_selector")
        self.fixed_scale = bool(DISPLAY_CONFIG["fixed_scale"])

        aliases = [tap.device for tap in taps]
        self.states = []
        for tap in taps:
            base = camera_setting_for_alias(tap.device, cfg["camera_setting"], aliases)
            fallback_exposure = int(base.get("cameraExposureTimeUs", 10000))
            fallback_gain = float(base.get("cameraGain", 0.0))
            readback = control_readbacks.get(tap.device)
            if readback is None:
                readback = read_camera_control_readback(
                    client,
                    tap.fesa_device,
                    fallback_exposure,
                    fallback_gain,
                )
            self.states.append(
                {
                    "alias": tap.device,
                    "fesa": tap.fesa_device,
                    "tap": tap,
                    "base": base,
                    "roi_writes_enabled": self.apply_camera_roi
                    and camera_roi_enabled(base),
                    "exposure_us": int(readback["exposure_us"]),
                    "exposure_limits_us": None,
                    "gain": float(readback["gain"]),
                    "full_scale": camera_full_scale(base),
                    "display_shape": display_shapes[tap.device],
                    "source_box": source_boxes.get(tap.device),
                    "box": boxes.get(tap.device),
                    "pending_exposure_us": None,
                    "pending_gain": None,
                    "last_write_t": 0.0,
                    "backoff_until": 0.0,
                    "last_error": readback.get("warning") or readback.get("error"),
                }
            )
            self._refresh_exposure_limits(self.states[-1])

        print("")
        print("Runtime camera controls:")
        print("  1 20000           set camera 1 exposure in us")
        print("  1 exp 20000       set camera 1 exposure in us")
        print("  2 gain 0.3        set camera 2 gain")
        print("  all 30000         set all exposures")
        print("  all exp 30000     set all exposures")
        print("  all gain 0        set all gains")
        print("  1 box 564 573 1000 1000")
        print("  all box 564 573 1000 1000")
        print("  box 564 573 1000 1000")
        print("  1 box off")
        print("  box coordinates use full-frame display pixels")
        if self.apply_camera_roi:
            print("  ROI mode: box updates camera crop only when roi_enabled=true")
        else:
            print("  full-frame mode: box is visual only")
        print("  scale fixed       display true values, 0 to sensor max (default)")
        print("  scale auto        stretch each frame min-max for viewing")
        print("  show              print current readback values")
        print("  quit              close preview")
        self.print_status()
        print("")

        thread = threading.Thread(target=self._input_loop, daemon=True)
        thread.start()

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            str(state["alias"]): {
                "exposure_us": int(state["exposure_us"]),
                "exposure_limits_us": state["exposure_limits_us"],
                "gain": float(state["gain"]),
                "pending_exposure_us": state["pending_exposure_us"],
                "pending_gain": state["pending_gain"],
                "last_error": state["last_error"],
                "display_full_scale": state["full_scale"] if self.fixed_scale else None,
                "box": state["box"],
                "display_shape": state["display_shape"],
                "source_box": state["source_box"],
            }
            for state in self.states
        }

    def update(self) -> None:
        while True:
            try:
                line = self.commands.get_nowait()
            except queue.Empty:
                break
            self._handle_line(line)

        now = time.monotonic()
        for state in self.states:
            if state["pending_exposure_us"] is None and state["pending_gain"] is None:
                continue
            if now < float(state["backoff_until"]):
                continue
            if now - float(state["last_write_t"]) < self.write_interval_s:
                continue
            self._write_state(state, now)

    def _input_loop(self) -> None:
        while True:
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if not line:
                break
            self.commands.put(line.strip())

    def _handle_line(self, line: str) -> None:
        if not line:
            return
        parts = line.replace("=", " ").replace(",", " ").split()
        cmd = parts[0].lower()
        if cmd in {"q", "quit", "exit"}:
            self.quit_requested = True
            return
        if cmd == "show":
            self.print_status()
            return
        if cmd == "scale":
            mode = parts[1].lower() if len(parts) == 2 else ""
            if mode not in {"auto", "fixed"}:
                print("Invalid scale. Use: scale auto, scale fixed")
                return
            self.fixed_scale = mode == "fixed"
            print("display scale: %s (camera settings unchanged)" % mode)
            return
        if len(parts) == 5 and cmd in {"box", "roi"}:
            self._handle_box(list(range(len(self.states))), parts[1:])
            return
        if len(parts) >= 2 and parts[1].lower() in {"box", "roi"}:
            targets = self._target_indices(parts[0])
            if not targets:
                return
            self._handle_box(targets, parts[2:])
            return
        if len(parts) == 2:
            target_text, value_text = parts
            field = "exp"
        elif len(parts) == 3:
            target_text, field, value_text = parts
            field = field.lower()
        else:
            print("Invalid command. Use: 1 20000, 1 exp 20000, 2 gain 0.3")
            return

        targets = self._target_indices(target_text)
        if not targets:
            return

        if field in {"exp", "exposure", "exposure_us"}:
            try:
                requested = int(float(value_text))
            except (ValueError, OverflowError):
                print("Invalid exposure: %s" % value_text)
                return
            for idx in targets:
                state = self.states[idx]
                # ROI, acquisition mode or server settings may have changed since startup.
                if not self._refresh_exposure_limits(state):
                    continue
                minimum, maximum = state["exposure_limits_us"]
                value = max(minimum, min(maximum, requested))
                state["pending_exposure_us"] = value
                print(
                    "queued exposure %s: %d us (requested %d; camera range %d..%d us)"
                    % (state["alias"], value, requested, minimum, maximum)
                )
            return

        if field == "gain":
            try:
                value = self.clip_gain(float(value_text))
            except ValueError:
                print("Invalid gain: %s" % value_text)
                return
            for idx in targets:
                self.states[idx]["pending_gain"] = value
            print("queued gain %s: %.4g" % (self._target_names(targets), value))
            return

        print("Invalid field %r. Use exp or gain." % field)

    def _refresh_exposure_limits(self, state: dict[str, Any]) -> bool:
        try:
            state["exposure_limits_us"] = read_camera_exposure_limits(
                self.client, state["fesa"]
            )
        except Exception as exc:
            state["exposure_limits_us"] = None
            state["pending_exposure_us"] = None
            state["last_error"] = "Cannot read camera exposure limits: %s" % exc
            print(
                "WARNING: %s: %s; exposure edit unavailable"
                % (state["alias"], state["last_error"]),
                flush=True,
            )
            return False
        return True

    def _handle_box(self, targets: list[int], values: list[str]) -> None:
        if len(values) == 1 and values[0].lower() in {"off", "none", "clear"}:
            applied = []
            for idx in targets:
                state = self.states[idx]
                if state["roi_writes_enabled"]:
                    try:
                        expected_shape = disable_preview_roi(
                            self.client,
                            state["alias"],
                            state["fesa"],
                        )
                        wait_preview_image_shape(
                            self.client,
                            state["alias"],
                            state["fesa"],
                            expected_shape,
                            "runtime full-frame",
                            get_selector=self.get_selector,
                            use_subscription=self.use_subscription,
                            subscribe_selector=self.subscribe_selector,
                        )
                    except Exception as exc:
                        state["last_error"] = str(exc)
                        print(
                            "WARNING: runtime ROI disable failed for %s: %s"
                            % (state["alias"], exc),
                            flush=True,
                        )
                        continue
                    state["source_box"] = None
                    state["last_error"] = None
                    self._refresh_current_frame(state)
                state["box"] = None
                applied.append("%s=%s" % (state["alias"], format_box(None)))
            print("cleared box %s" % "; ".join(applied))
            return

        if len(values) != 4:
            print("Invalid box. Use: 1 box x1 y1 x2 y2")
            return

        try:
            box = normalize_box(values[0], values[1], values[2], values[3])
        except (RuntimeError, TypeError, ValueError) as exc:
            print("Invalid box: %s" % exc)
            return

        applied = []
        for idx in targets:
            clipped = clip_box_to_shape(box, self.states[idx]["display_shape"])
            state = self.states[idx]
            if clipped is not None and state["roi_writes_enabled"]:
                try:
                    clipped = set_preview_roi_box(
                        self.client,
                        state["alias"],
                        state["fesa"],
                        clipped,
                        state["base"],
                        get_selector=self.get_selector,
                        use_subscription=self.use_subscription,
                        subscribe_selector=self.subscribe_selector,
                    )
                except Exception as exc:
                    state["last_error"] = str(exc)
                    print(
                        "WARNING: runtime ROI write failed for %s: %s"
                        % (state["alias"], exc),
                        flush=True,
                    )
                    continue
                state["source_box"] = clipped
                state["last_error"] = None
                self._refresh_current_frame(state)
            state["box"] = clipped
            applied.append("%s=%s" % (state["alias"], format_box(clipped)))
        print("set box %s" % "; ".join(applied))

    def _refresh_current_frame(self, state: dict[str, Any]) -> None:
        if self.frame_emit is None:
            return
        tap = state.get("tap")
        if tap is None:
            return
        try:
            tap._build_and_emit(tap._read_last_image(), self.frame_emit)  # noqa: SLF001
        except Exception as exc:
            state["last_error"] = str(exc)
            print(
                "WARNING: runtime ROI preview refresh failed for %s: %s"
                % (state["alias"], exc),
                flush=True,
            )

    def _target_indices(self, text: str) -> list[int]:
        target = text.lower().replace("camera", "").replace("cam", "").replace("c", "")
        if target == "all":
            return list(range(len(self.states)))
        try:
            idx = int(target) - 1
        except ValueError:
            aliases = [str(state["alias"]).lower() for state in self.states]
            if text.lower() in aliases:
                return [aliases.index(text.lower())]
            print("Invalid camera target: %s" % text)
            return []
        if not 0 <= idx < len(self.states):
            print("Camera target out of range: %s" % text)
            return []
        return [idx]

    def _target_names(self, targets: list[int]) -> str:
        return ", ".join(str(self.states[idx]["alias"]) for idx in targets)

    def _write_state(self, state: dict[str, Any], now: float) -> None:
        exposure_us = (
            int(state["pending_exposure_us"])
            if state["pending_exposure_us"] is not None
            else int(state["exposure_us"])
        )
        gain = (
            float(state["pending_gain"])
            if state["pending_gain"] is not None
            else float(state["gain"])
        )

        setting = dict(state["base"])
        setting["cameraExposureTimeUs"] = exposure_us
        setting["cameraExposureAuto"] = CAMERA_AUTO_OFF_VALUE
        setting["cameraGain"] = gain
        setting["cameraGainAuto"] = CAMERA_AUTO_OFF_VALUE

        try:
            set_value(
                self.client,
                "%s/CameraSetting" % state["fesa"],
                camera_setting_payload(setting),
            )
        except Exception as exc:
            state["last_error"] = str(exc)
            state["backoff_until"] = now + self.failure_backoff_s
            print(
                "WARNING: camera setting write failed for %s: %s"
                % (state["alias"], exc),
                flush=True,
            )
            return

        readback = wait_camera_control_readback(
            self.client,
            state["alias"],
            state["fesa"],
            expected_exposure_us=exposure_us,
            expected_gain=gain,
        )
        state["exposure_us"] = int(readback["exposure_us"])
        state["gain"] = float(readback["gain"])
        state["pending_exposure_us"] = None
        state["pending_gain"] = None
        state["last_write_t"] = now
        state["last_error"] = readback.get("warning") or readback.get("error")
        print(
            "set %s exposure=%d us gain=%.4g readback exposure=%d us gain=%.4g"
            % (
                state["alias"],
                exposure_us,
                gain,
                int(readback["exposure_us"]),
                float(readback["gain"]),
            ),
            flush=True,
        )

    def print_status(self) -> None:
        for state in self.states:
            pending = []
            if state["pending_exposure_us"] is not None:
                pending.append("pending_exp=%s" % state["pending_exposure_us"])
            if state["pending_gain"] is not None:
                pending.append("pending_gain=%.4g" % state["pending_gain"])
            suffix = "" if not pending else "  " + " ".join(pending)
            limits = state["exposure_limits_us"]
            exposure_range = "unknown" if limits is None else "%d..%d us" % limits
            print(
                "%s: readback exposure=%d us gain=%.4g exposure_range=%s box=%s%s"
                % (
                    state["alias"],
                    int(state["exposure_us"]),
                    float(state["gain"]),
                    exposure_range,
                    format_box(state["box"]),
                    suffix,
                )
            )

    @staticmethod
    def clip_gain(value: float) -> float:
        return max(GAIN_MIN, min(GAIN_MAX, float(value)))


class LatestFrames:
    def __init__(self, taps: list[Any]) -> None:
        self._lock = threading.Lock()
        self._records = {
            tap.device: {
                "frame": None,
                "received_at": None,
                "last_image_number": None,
                "frames": 0,
                "gaps": 0,
                "times": deque(maxlen=32),
            }
            for tap in taps
        }

    def emit(self, frame: Frame) -> None:
        now = time.monotonic()
        with self._lock:
            record = self._records[frame.device]
            previous = record["last_image_number"]
            current = int(frame.source_image_number)
            if previous is not None and current > int(previous) + 1:
                record["gaps"] += current - int(previous) - 1
            record["frame"] = frame
            record["received_at"] = now
            record["last_image_number"] = current
            record["frames"] += 1
            record["times"].append(now)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            out = {}
            for alias, record in self._records.items():
                copy = dict(record)
                copy["times"] = list(record["times"])
                out[alias] = copy
            return out


def seed_current_last_images(taps: list[Any], emit: Any) -> None:
    for tap in taps:
        # Use the tap's own frame builder so the preview starts from the same
        # Frame metadata/path that the DAQ writer would receive.
        tap._build_and_emit(tap._read_last_image(), emit)  # noqa: SLF001


def frame_to_image(frame: Frame, tap: Any) -> np.ndarray:
    image_dtype = getattr(frame, "image_dtype", None) or tap.image_dtype
    image_shape = getattr(frame, "image_shape", None) or tap.image_shape
    if not image_dtype or not image_shape:
        raise RuntimeError("camera %s was not probed" % tap.device)
    dtype = np.dtype(image_dtype)
    shape = tuple(int(part) for part in image_shape)
    image = np.frombuffer(frame.payload_bytes, dtype=dtype)
    expected = int(np.prod(shape))
    if image.size != expected:
        raise RuntimeError(
            "camera %s frame payload has %d elements, expected %d"
            % (tap.device, image.size, expected)
        )
    return image.reshape(shape)


def grayscale_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    if arr.ndim == 3:
        return arr.astype(np.float32).mean(axis=2)
    return arr.reshape(1, -1)


def image_stats(image: np.ndarray) -> dict[str, float]:
    gray = grayscale_image(image).astype(np.float64, copy=False)
    values = gray[np.isfinite(gray)]
    if values.size == 0:
        return {key: float("nan") for key in ("min", "max", "p99_9", "sum")}
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "p99_9": float(np.percentile(values, 99.9)),
        "sum": float(values.sum()),
    }


def camera_full_scale(setting: MappingABC[str, Any]) -> int:
    """Largest pixel value the camera sends: format max, times bin count in SUM binning."""

    full_scale = PIXEL_FORMAT_FULL_SCALE[
        camera_pixel_format_value(setting.get("cameraPixelFormat", "mono12"))
    ]
    if int(setting.get("cameraBinningMode", 0)) == 1:  # SUM
        bin_x, bin_y = camera_binning_factors(setting)
        full_scale *= bin_x * bin_y
    return full_scale


def image_to_u8(image: np.ndarray, full_scale: Optional[float] = None) -> np.ndarray:
    """Map to 8-bit for display: fixed 0..full_scale, or per-frame min-max when None."""

    gray = grayscale_image(image).astype(np.float32, copy=False)
    if full_scale is not None:
        return np.clip(gray * (255.0 / float(full_scale)), 0, 255).astype(np.uint8)
    values = gray[np.isfinite(gray)]
    if values.size == 0:
        return np.zeros(gray.shape, dtype=np.uint8)

    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((gray - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def empty_tile(shape: Optional[tuple[int, int]] = None) -> np.ndarray:
    if shape is None:
        shape = (512, 512)
    return np.full(
        (int(shape[0]), int(shape[1])),
        int(ROI_PREVIEW_CONFIG["outside_value"]),
        dtype=np.uint8,
    )


def resize_nearest_to_shape(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        return np.zeros((0, 0), dtype=np.asarray(image).dtype)
    image = np.asarray(image)
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        return np.zeros((height, width), dtype=image.dtype)
    if image.shape[:2] == (height, width):
        return np.ascontiguousarray(image)
    y_idx = np.linspace(0, image.shape[0] - 1, height).astype(np.int32)
    x_idx = np.linspace(0, image.shape[1] - 1, width).astype(np.int32)
    return np.ascontiguousarray(image[y_idx[:, None], x_idx])


def display_tile(
    image: np.ndarray,
    display_shape: tuple[int, int],
    source_box: Optional[tuple[int, int, int, int]],
) -> np.ndarray:
    out = empty_tile(display_shape)
    if source_box is None:
        resized = resize_nearest_to_shape(image, out.shape[:2])
        out[:resized.shape[0], :resized.shape[1]] = resized
        return out

    clipped = clip_box_to_shape(source_box, display_shape)
    if clipped is None:
        return out

    left, top, right, bottom = clipped
    height = bottom - top
    width = right - left
    if height <= 0 or width <= 0:
        return out
    out[top:bottom, left:right] = resize_nearest_to_shape(image, (height, width))
    return out


def pad_to_shape(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.uint8)
    height = min(shape[0], image.shape[0])
    width = min(shape[1], image.shape[1])
    out[:height, :width] = image[:height, :width]
    return out


def resize_for_screen(image: np.ndarray, scale: float, max_width: int) -> np.ndarray:
    image = np.asarray(image)
    scale = max(0.05, min(1.0, float(scale)))
    if max_width > 0:
        scale = min(scale, float(max_width) / max(1, image.shape[1]))
    if scale >= 0.999:
        return np.ascontiguousarray(image)
    new_h = max(1, int(round(image.shape[0] * scale)))
    new_w = max(1, int(round(image.shape[1] * scale)))
    y_idx = np.linspace(0, image.shape[0] - 1, new_h).astype(np.int32)
    x_idx = np.linspace(0, image.shape[1] - 1, new_w).astype(np.int32)
    return np.ascontiguousarray(image[y_idx[:, None], x_idx])


def grid_shape(count: int) -> tuple[int, int]:
    """Tile rows and columns for count cameras: 1 -> 1x1, 2 -> 1x2, 3-4 -> 2x2, ..."""

    cols = max(1, min(count, GRID_MAX_COLS))
    return -(-max(count, 1) // cols), cols


def draw_display_dashed_grid(
    mosaic: np.ndarray,
    raw_tile_h: int,
    raw_tile_w: int,
    rows: int,
    cols: int,
) -> np.ndarray:
    if not GRID_CONFIG["enabled"]:
        return mosaic

    size = int(GRID_CONFIG["size_px"])
    if size <= 0 or raw_tile_h <= 0 or raw_tile_w <= 0:
        return mosaic

    base = np.ascontiguousarray(mosaic, dtype=np.uint8)
    out = base.astype(np.float32)
    height, width = base.shape[:2]
    mask = np.zeros((height, width), dtype=bool)

    dash = max(1, int(GRID_CONFIG["dash_px"]))
    gap = max(0, int(GRID_CONFIG["gap_px"]))
    period = max(1, dash + gap)
    thickness = max(1, int(GRID_CONFIG["thickness_px"]))

    for row in range(rows):
        y_tile0 = int(round(row * height / rows))
        y_tile1 = int(round((row + 1) * height / rows))
        tile_h = y_tile1 - y_tile0
        if tile_h <= 0:
            continue
        scale_y = tile_h / float(raw_tile_h)

        for col in range(cols):
            x_tile0 = int(round(col * width / cols))
            x_tile1 = int(round((col + 1) * width / cols))
            tile_w = x_tile1 - x_tile0
            if tile_w <= 0:
                continue
            scale_x = tile_w / float(raw_tile_w)

            n = 0
            while True:
                x = int(round(n * size * scale_x))
                if x >= tile_w:
                    break
                x0 = x_tile0 + x
                x1 = min(x_tile1, x0 + thickness)

                k = 0
                while True:
                    y0_local = int(round(k * period * scale_y))
                    if y0_local >= tile_h:
                        break
                    y1_local = int(round((k * period + dash) * scale_y))
                    y1_local = max(y0_local + 1, y1_local)
                    y0 = y_tile0 + y0_local
                    y1 = min(y_tile1, y_tile0 + y1_local)
                    mask[y0:y1, x0:x1] = True
                    k += 1
                n += 1

            n = 0
            while True:
                y = int(round(n * size * scale_y))
                if y >= tile_h:
                    break
                y0 = y_tile0 + y
                y1 = min(y_tile1, y0 + thickness)

                k = 0
                while True:
                    x0_local = int(round(k * period * scale_x))
                    if x0_local >= tile_w:
                        break
                    x1_local = int(round((k * period + dash) * scale_x))
                    x1_local = max(x0_local + 1, x1_local)
                    x0 = x_tile0 + x0_local
                    x1 = min(x_tile1, x_tile0 + x1_local)
                    mask[y0:y1, x0:x1] = True
                    k += 1
                n += 1

    value = float(max(0, min(255, int(GRID_CONFIG["gray_value"]))))
    opacity = max(0.0, min(1.0, float(GRID_CONFIG["opacity"])))
    out[mask] = out[mask] + opacity * (value - out[mask])
    return np.clip(out, 0, 255).astype(np.uint8)


def shade_outside_boxes(
    mosaic: np.ndarray,
    taps: list[Any],
    control_snapshot: Optional[dict[str, dict[str, Any]]],
    raw_tile_h: int,
    raw_tile_w: int,
) -> np.ndarray:
    if not ROI_PREVIEW_CONFIG["shade_outside"]:
        return mosaic

    controls = control_snapshot or {}
    out = np.ascontiguousarray(mosaic).copy()
    rows, cols = grid_shape(len(taps))
    tile_h = out.shape[0] // rows
    tile_w = out.shape[1] // cols
    outside_value = max(0, min(255, int(ROI_PREVIEW_CONFIG["outside_value"])))

    for idx, tap in enumerate(taps):
        row = idx // cols
        col = idx % cols
        box = controls.get(tap.device, {}).get("box")
        box = clip_box_to_shape(box, (raw_tile_h, raw_tile_w))
        if box is None or raw_tile_h <= 0 or raw_tile_w <= 0:
            continue

        x0 = col * tile_w
        y0 = row * tile_h
        tile = out[y0:y0 + tile_h, x0:x0 + tile_w]
        keep = tile.copy()

        left, top, right, bottom = box
        left = int(round(left * tile_w / float(raw_tile_w)))
        right = int(round(right * tile_w / float(raw_tile_w)))
        top = int(round(top * tile_h / float(raw_tile_h)))
        bottom = int(round(bottom * tile_h / float(raw_tile_h)))
        left = max(0, min(tile_w, left))
        right = max(0, min(tile_w, right))
        top = max(0, min(tile_h, top))
        bottom = max(0, min(tile_h, bottom))
        if right <= left or bottom <= top:
            continue

        tile[:, :] = outside_value
        tile[top:bottom, left:right] = keep[top:bottom, left:right]

    return out


def recent_fps(times: list[float]) -> float:
    if len(times) < 2:
        return 0.0
    span = times[-1] - times[0]
    if span <= 0:
        return 0.0
    return float(len(times) - 1) / span


def sync_source(frame: Optional[Frame]) -> str:
    if frame is None:
        return "waiting"
    metadata = getattr(frame, "sync_metadata", {}) or {}
    return str(metadata.get("chosen_sync_source") or metadata.get("legacy_stamp_source") or "none")


def png_chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(kind)
    crc = zlib.crc32(data, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def gray_png_bytes(image: np.ndarray, palette: Optional[np.ndarray] = None) -> bytes:
    """Encode an 8-bit display image as grayscale or with an indexed RGB palette."""
    image = np.ascontiguousarray(image, dtype=np.uint8)
    height, width = image.shape
    rows = [b"\x00" + image[row].tobytes() for row in range(height)]
    raw = b"".join(rows)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0 if palette is None else 3, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + (b"" if palette is None else png_chunk(b"PLTE", palette.tobytes()))
        + png_chunk(b"IDAT", zlib.compress(raw, 1))
        + png_chunk(b"IEND", b"")
    )


def image_to_tk_put(image: np.ndarray, palette: Optional[np.ndarray] = None) -> Any:
    if tk is None:
        raise RuntimeError("tkinter is unavailable")
    image = np.asarray(image, dtype=np.uint8)
    photo = tk.PhotoImage(width=image.shape[1], height=image.shape[0])
    colors = (
        ["#%02x%02x%02x" % (i, i, i) for i in range(256)]
        if palette is None else ["#%02x%02x%02x" % tuple(rgb) for rgb in palette]
    )
    rows = [
        "{" + " ".join(colors[int(pixel)] for pixel in row) + "}"
        for row in image
    ]
    photo.put(" ".join(rows))
    return photo


class TkPreview:
    def __init__(self, title: str) -> None:
        if tk is None:
            raise RuntimeError("camera_preview.py requires tkinter for display")

        self.root = tk.Tk()
        self.root.title(title)
        self.canvas = tk.Canvas(self.root, bd=0, highlightthickness=0)
        self.canvas.pack()
        self.photo = None
        self.closed = False
        self.use_put = False

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Escape>", lambda event: self.close())
        self.root.bind("q", lambda event: self.close())

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.root.destroy()
        except Exception:
            pass

    def show(
        self,
        image: np.ndarray,
        overlays: list[dict[str, Any]],
        title: str,
        *,
        palette: Optional[np.ndarray] = None,
    ) -> bool:
        if self.closed:
            return False

        if self.use_put:
            self.photo = image_to_tk_put(image, palette)
        else:
            try:
                png = gray_png_bytes(image, palette)
                self.photo = tk.PhotoImage(
                    data=base64.b64encode(png).decode("ascii"),
                    format="png",
                )
            except tk.TclError:
                self.use_put = True
                self.photo = image_to_tk_put(image, palette)

        height, width = image.shape
        self.canvas.configure(width=width, height=height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

        for item in overlays:
            if item.get("kind") == "rectangle":
                x1 = int(item["x1"])
                y1 = int(item["y1"])
                x2 = int(item["x2"])
                y2 = int(item["y2"])
                color = str(item.get("color", "cyan"))
                width_px = int(item.get("width", 2))
                self.canvas.create_rectangle(
                    x1 + 1,
                    y1 + 1,
                    x2 + 1,
                    y2 + 1,
                    outline="black",
                    width=width_px + 1,
                )
                self.canvas.create_rectangle(
                    x1,
                    y1,
                    x2,
                    y2,
                    outline=color,
                    width=width_px,
                )
                continue

            x = int(item["x"])
            y = int(item["y"])
            text = str(item["text"])
            color = str(item.get("color", "white"))
            self.canvas.create_text(
                x + 1,
                y + 1,
                anchor="nw",
                text=text,
                fill="black",
                font=("TkFixedFont", 10),
            )
            self.canvas.create_text(
                x,
                y,
                anchor="nw",
                text=text,
                fill=color,
                font=("TkFixedFont", 10),
            )

        self.root.title(title)
        self.root.update_idletasks()
        self.root.update()
        return True


def render_mosaic(
    taps: list[Any],
    snapshot: dict[str, dict[str, Any]],
    *,
    control_snapshot: Optional[dict[str, dict[str, Any]]] = None,
    display_scale: float,
    max_width: int,
) -> tuple[np.ndarray, list[dict[str, Any]], str]:
    tiles = []
    stats_by_alias = {}
    now = time.monotonic()
    controls = control_snapshot or {}

    for tap in taps:
        control = controls.get(tap.device, {})
        display_shape = control.get("display_shape")
        if display_shape is None:
            display_shape = tuple(int(part) for part in (tap.image_shape or (512, 512))[:2])
        else:
            display_shape = (int(display_shape[0]), int(display_shape[1]))
        source_box = control.get("source_box")
        record = snapshot[tap.device]
        frame = record["frame"]
        if frame is None:
            tile = empty_tile(display_shape)
            stats = image_stats(np.empty(0))
        else:
            image = frame_to_image(frame, tap)
            tile = display_tile(
                image_to_u8(image, control.get("display_full_scale")),
                display_shape,
                source_box,
            )
            stats = image_stats(image)
        tiles.append(tile)
        stats_by_alias[tap.device] = stats

    height = max(tile.shape[0] for tile in tiles) if tiles else 512
    width = max(tile.shape[1] for tile in tiles) if tiles else 512
    grid_rows, grid_cols = grid_shape(len(taps))
    while len(tiles) < grid_rows * grid_cols:
        tiles.append(np.zeros((height, width), dtype=np.uint8))
    tiles = [pad_to_shape(tile, (height, width)) for tile in tiles]

    rows = []
    for row in range(grid_rows):
        start = row * grid_cols
        rows.append(np.hstack(tiles[start:start + grid_cols]))
    mosaic = np.vstack(rows)
    raw_tile_h = mosaic.shape[0] // grid_rows
    raw_tile_w = mosaic.shape[1] // grid_cols
    mosaic = resize_for_screen(mosaic, display_scale, max_width)
    mosaic = shade_outside_boxes(mosaic, taps, control_snapshot, raw_tile_h, raw_tile_w)
    mosaic = draw_display_dashed_grid(
        mosaic, raw_tile_h, raw_tile_w, grid_rows, grid_cols
    )

    tile_h = mosaic.shape[0] // grid_rows
    tile_w = mosaic.shape[1] // grid_cols
    for col in range(1, grid_cols):
        mosaic[:, col * tile_w:col * tile_w + 1] = 255
    for row in range(1, grid_rows):
        mosaic[row * tile_h:row * tile_h + 1, :] = 255

    overlays: list[dict[str, Any]] = []
    title_parts = []
    for idx, tap in enumerate(taps):
        row = idx // grid_cols
        col = idx % grid_cols
        x = col * tile_w + 10
        y = row * tile_h + 22
        record = snapshot[tap.device]
        frame = record["frame"]
        age = None
        if record["received_at"] is not None:
            age = now - float(record["received_at"])
        stats = stats_by_alias[tap.device]
        fps = recent_fps(record["times"])
        control = controls.get(tap.device, {})
        color = str(DISPLAY_CONFIG["text_color"])
        box = control.get("box")
        if (
            ROI_PREVIEW_CONFIG["show_box"]
            and box is not None
            and width > 0
            and height > 0
        ):
            x_scale = tile_w / float(raw_tile_w)
            y_scale = tile_h / float(raw_tile_h)
            x1, y1, x2, y2 = box
            overlays.append(
                {
                    "kind": "rectangle",
                    "x1": col * tile_w + int(round(x1 * x_scale)),
                    "y1": row * tile_h + int(round(y1 * y_scale)),
                    "x2": col * tile_w + int(round(x2 * x_scale)),
                    "y2": row * tile_h + int(round(y2 * y_scale)),
                    "color": "cyan",
                    "width": 2,
                }
            )
        image_number = "..." if frame is None else str(frame.source_image_number)
        age_text = "--" if age is None else "%.2fs" % age
        title_parts.append("%s:%s" % (tap.device, image_number))
        exposure = control.get("exposure_us")
        gain = control.get("gain")
        control_line = "rb exp=-- gain=--"
        if exposure is not None and gain is not None:
            control_line = "rb exp=%dus  gain=%.4g" % (
                int(exposure),
                float(gain),
            )
        pending = []
        if control.get("pending_exposure_us") is not None:
            pending.append("exp")
        if control.get("pending_gain") is not None:
            pending.append("gain")
        if pending:
            control_line += "  pending:%s" % ",".join(pending)
        box_line = "box=%s" % format_box(box)
        lines = [
            "%s  %s" % (tap.device, tap.fesa_device),
            "img=%s  fps=%.1f  age=%s" % (image_number, fps, age_text),
            control_line,
            box_line,
            "min=%.0f  max=%.0f  gaps=%d" % (
                stats["min"],
                stats["max"],
                record["gaps"],
            ),
            "P99.9=%.0f  sum=%.0f" % (stats["p99_9"], stats["sum"]),
            "sync=%s" % sync_source(frame),
        ]
        for line in lines:
            overlays.append({"x": x, "y": y, "text": line, "color": color})
            y += 18

        full_scale = control.get("display_full_scale")
        bottom = [
            "display: auto min-max"
            if full_scale is None
            else "display: fixed 0-%d" % int(full_scale)
        ]
        if GRID_CONFIG["enabled"] and GRID_CONFIG["show_label"]:
            bottom.insert(0, "grid:%dpx" % int(GRID_CONFIG["size_px"]))
        y = row * tile_h + max(0, tile_h - 24 - 18 * (len(bottom) - 1))
        for line in bottom:
            overlays.append({"x": col * tile_w + 10, "y": y, "text": line, "color": color})
            y += 18

    title = "OptiDAQ camera preview  " + "  ".join(title_parts)
    return mosaic, overlays, title


def print_tap_probe(tap: Any) -> None:
    print(
        "camera %s fesa=%s shape=%s dtype=%s payload_nbytes=%s "
        "stamp_source=%s probe_pulse_id=%s"
        % (
            tap.device,
            tap.fesa_device,
            tap.image_shape,
            tap.image_dtype,
            tap.payload_nbytes,
            tap.stamp_source,
            tap.probe_pulse_id,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_arg",
        nargs="?",
        type=Path,
        help="Optional config JSON path. Usually use --config instead.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--no-configure",
        action="store_true",
        help="Do not apply camera_setting or preview ROI policy before previewing.",
    )
    parser.add_argument(
        "--no-seed-current",
        action="store_true",
        help="Wait for new tap updates instead of displaying current LastImage first.",
    )
    parser.add_argument("--display-scale", type=float, default=DISPLAY_CONFIG["scale"])
    parser.add_argument("--max-width", type=int, default=DISPLAY_CONFIG["max_width"])
    parser.add_argument("--refresh-hz", type=float, default=DISPLAY_CONFIG["refresh_hz"])
    parser.add_argument("--window", default=DISPLAY_CONFIG["window"])
    parser.add_argument(
        "--no-control",
        action="store_true",
        help="Disable terminal runtime exposure/gain writes.",
    )
    parser.add_argument(
        "--strict-image-number",
        action="store_true",
        help="Stop on imageNumber reset, matching real DAQ strictness.",
    )
    args = parser.parse_args()

    if args.config_arg is not None:
        try:
            duplicate_script = args.config_arg.resolve() == Path(__file__).resolve()
        except OSError:
            duplicate_script = False
        if duplicate_script:
            print("ignoring duplicate script path argument: %s" % args.config_arg)
        elif args.config != DEFAULT_CONFIG_PATH:
            raise RuntimeError("use either positional config path or --config, not both")
        else:
            args.config = args.config_arg

    return args


def main() -> None:
    args = parse_args()
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    cfg = load_config(args.config)
    camera_map = camera_map_from_config(cfg)
    should_configure = output_configures_cameras(cfg) and not args.no_configure
    apply_camera_roi = bool(ROI_PREVIEW_CONFIG["apply_camera_roi"])
    require_rbac = bool(cfg["rbac"]["enabled"]) and (
        should_configure or not args.no_control
    )
    client = make_client(require_rbac=require_rbac)
    full_shapes: dict[str, Optional[tuple[int, int]]] = {}
    control_readbacks: dict[str, dict[str, Any]] = {}

    if should_configure:
        ensure_cameras_alive(client, camera_map, cfg.get("writer", {}))
        full_shapes, control_readbacks = configure_preview_cameras(
            client,
            camera_map,
            cfg["camera_setting"],
            apply_camera_roi=apply_camera_roi,
            get_selector=cfg["tap"].get("get_selector"),
            use_subscription=tap_capture_mode(cfg["tap"]) == "subscribe",
            subscribe_selector=cfg["tap"].get("subscribe_selector"),
        )

    taps = build_camera_taps(
        cfg,
        client,
        allow_image_number_reset=not args.strict_image_number,
    )
    for tap in taps:
        tap.probe()
        print_tap_probe(tap)
    validate_tap_stamp_sources(taps, cfg["sync"])
    display_shapes = initial_display_shapes(cfg, taps, full_shapes)
    source_boxes = initial_source_boxes(
        cfg,
        taps,
        apply_camera_roi=apply_camera_roi,
        display_shapes=display_shapes,
    )
    roi_boxes = initial_roi_boxes(
        cfg,
        taps,
        apply_camera_roi=apply_camera_roi,
        display_shapes=display_shapes,
    )

    latest = LatestFrames(taps)
    if not args.no_seed_current:
        seed_current_last_images(taps, latest.emit)

    controls = None if args.no_control else RuntimeCameraControl(
        client,
        taps,
        cfg,
        boxes=roi_boxes,
        display_shapes=display_shapes,
        source_boxes=source_boxes,
        control_readbacks=control_readbacks,
        apply_camera_roi=apply_camera_roi,
        frame_emit=latest.emit,
    )

    aliases = [tap.device for tap in taps]
    fixed_full_scales = {
        alias: camera_full_scale(
            camera_setting_for_alias(alias, cfg["camera_setting"], aliases)
        )
        if DISPLAY_CONFIG["fixed_scale"]
        else None
        for alias in aliases
    }

    viewer = TkPreview(args.window)
    for tap in taps:
        tap.start(latest.emit)

    wait_s = 1.0 / max(1.0, float(args.refresh_hz))
    try:
        while not STOP_REQUESTED:
            if controls is not None:
                controls.update()
                if controls.quit_requested:
                    break

            for tap in taps:
                if tap.error is not None:
                    raise RuntimeError("camera tap %s failed: %s" % (tap.device, tap.error))

            mosaic, overlays, title = render_mosaic(
                taps,
                latest.snapshot(),
                control_snapshot=(
                    {
                        alias: {
                            "box": box,
                            "display_shape": display_shapes[alias],
                            "source_box": source_boxes.get(alias),
                            "display_full_scale": fixed_full_scales[alias],
                        }
                        for alias, box in roi_boxes.items()
                    }
                    if controls is None
                    else controls.snapshot()
                ),
                display_scale=args.display_scale,
                max_width=args.max_width,
            )
            if not viewer.show(mosaic, overlays, title):
                break
            time.sleep(wait_s)
    finally:
        for tap in taps:
            tap.stop()
        for tap in taps:
            join = getattr(tap, "join", None)
            if callable(join):
                join(timeout=2.0)
        viewer.close()


if __name__ == "__main__":
    main()
