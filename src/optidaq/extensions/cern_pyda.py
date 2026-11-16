"""CERN/PyDA adapters for OptiDAQ VM smoke scans.

This module is intentionally outside ``optidaq.daq`` because it depends on
CERN-specific packages: pyda, pyrbac, pyda_rda3, and numpy.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
import getpass
import os
from pathlib import Path
import threading
import time
import traceback
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple
import zlib

import numpy as np
import pyda
import pyrbac
from pyda_rda3 import RdaProvider

from optidaq.daq.contracts import NO_PULSE_ID, Frame
from optidaq.daq.sync_tools import (
    choose_frame_sync_candidate,
    extract_sync_candidates,
)


class _RbacListener(pyrbac.AuthenticationListener):
    def __init__(self, on_token: Callable[[object], None]) -> None:
        super().__init__()
        self._on_token = on_token
        self.event = threading.Event()
        self.error: Optional[BaseException] = None
        self.token: Optional[object] = None

    def authentication_done(self, token: object) -> None:
        self.token = token
        self._on_token(token)
        self.event.set()

    def authentication_error(self, error: BaseException) -> None:
        self.error = error
        self.event.set()

    def token_expired(self, token: object) -> None:
        self.token = None


# --- RBAC token cache -------------------------------------------------------
# Reuse an unexpired RBAC token. CLEAR_RBAC_TOKEN_FILE overrides the cache path.
# The cache holds a live credential and is created with mode 0600. Cache
# failures fall back to the normal login path.
_TOKEN_REFRESH_MARGIN_S = 300        # re-login if < 5 min of validity is left
_TOKEN_FALLBACK_TTL_S = 6 * 3600     # assumed lifetime when expiry can't be read


def _token_cache_path() -> Path:
    override = os.environ.get("CLEAR_RBAC_TOKEN_FILE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "clear2026" / "rbac_token"


def _token_to_bytes(token: object) -> Optional[bytes]:
    """Serialize a pyrbac token to bytes (the accessor name varies by version)."""
    for attr in ("encode", "get_encoded", "encode_token"):
        fn = getattr(token, attr, None)
        if not callable(fn):
            continue
        try:
            data = fn()
        except Exception:
            continue
        if data is None:
            continue
        try:
            return bytes(data)
        except (TypeError, ValueError):
            continue
    return None


def _token_from_bytes(data: bytes) -> Optional[object]:
    ctor = getattr(pyrbac, "Token", None)
    if ctor is None:
        return None
    for arg in (data, bytearray(data), list(data)):
        try:
            return ctor(arg)
        except Exception:
            continue
    return None


def _token_expiry_epoch(token: object) -> Optional[float]:
    for attr in (
        "get_expiration_time",
        "expiration_time",
        "get_expiration_timestamp",
        "expiration_timestamp",
    ):
        value = getattr(token, attr, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                value = None
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return None


def _token_is_usable(token: Optional[object], saved_mtime: Optional[float]) -> bool:
    if token is None:
        return False
    empty = getattr(token, "empty", None)
    try:
        if callable(empty) and empty():
            return False
    except Exception:
        pass
    expiry = _token_expiry_epoch(token)
    now = time.time()
    if expiry is not None:
        return (expiry - now) > _TOKEN_REFRESH_MARGIN_S
    # No readable expiry: trust the file only for a conservative window.
    if saved_mtime is not None:
        return (now - saved_mtime) < _TOKEN_FALLBACK_TTL_S
    return False


def _load_cached_token() -> Optional[object]:
    path = _token_cache_path()
    try:
        data = path.read_bytes()
        mtime = path.stat().st_mtime
    except OSError:
        return None
    token = _token_from_bytes(data)
    if not _token_is_usable(token, mtime):
        return None
    return token


def _save_token(token: Optional[object]) -> None:
    data = _token_to_bytes(token) if token is not None else None
    if not data:
        return
    path = _token_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(path.parent, 0o700)
        except OSError:
            pass
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
    except OSError:
        pass


def _await_login(listener: "_RbacListener", *, required: bool) -> bool:
    if not listener.event.wait(30):
        if required:
            raise RuntimeError("RBAC login timed out")
        return False
    if listener.error is not None:
        if required:
            raise RuntimeError("RBAC login failed: %s" % listener.error)
        return False
    return True


def _login_location(provider: object, client: object) -> Optional[object]:
    """Passwordless Kerberos/location login. Returns a token, or None when the
    site / pyrbac build doesn't support it (caller then prompts for a password)."""
    factory = getattr(pyrbac.LoginService, "create_for_location", None)
    if factory is None:
        return None
    listener = _RbacListener(lambda token: setattr(provider, "rbac_token", token))
    service = None
    for args in ((listener,), ("CLEAR2026", listener)):  # signature varies by version
        try:
            service = factory(*args)
            break
        except Exception:
            service = None
    if service is None or not _await_login(listener, required=False):
        return None
    client._optidaq_rbac_service = service
    client._optidaq_rbac_listener = listener
    return listener.token


def _login_explicit(provider: object, client: object) -> Optional[object]:
    default_user = os.environ.get("CLEAR_RBAC_USER") or getpass.getuser()
    username = input("RBAC username [%s]: " % default_user).strip() or default_user
    password = getpass.getpass("RBAC password: ")
    listener = _RbacListener(lambda token: setattr(provider, "rbac_token", token))
    service = pyrbac.LoginService.create_for_explicit(username, password, listener)
    _await_login(listener, required=True)
    client._optidaq_rbac_service = service
    client._optidaq_rbac_listener = listener
    return listener.token


def make_client(require_rbac: bool) -> pyda.SimpleClient:
    provider = RdaProvider()
    client = pyda.SimpleClient(provider=provider)

    if require_rbac:
        # 1) Reuse a still-valid cached token: no prompt at all.
        token = _load_cached_token()
        if token is not None:
            provider.rbac_token = token
            print("RBAC OK (reused cached token)")
            return client

        # 2) No usable cache: try passwordless location login, else prompt once.
        token = _login_location(provider, client)
        if token is None:
            token = _login_explicit(provider, client)

        # 3) Cache the fresh token so the next run skips the prompt.
        _save_token(token)
        print("RBAC OK")

    return client


def plain(value: Any) -> Any:
    if hasattr(value, "value"):
        value = value.value
    shape = getattr(value, "shape", None)
    if shape == () and hasattr(value, "_value"):
        value = value._value
    if hasattr(value, "item") and shape in ((), None):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def optional_int_like(value: Any) -> Optional[int]:
    value = plain(value)
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def require_int_like(value: Any, name: str) -> int:
    result = optional_int_like(value)
    if result is None:
        raise RuntimeError("%s is missing or not int-like: %r" % (name, value))
    return result


def jsonable(value: Any) -> Any:
    value = plain(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, MappingABC):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    # FESA bit-enums (e.g. IntBitEnum for faultBits/current_status) are not JSON
    # native; they carry an int .value. Coerce to int, then float, then str so the
    # the resolved-config dump can serialize these wrapper values.
    inner = getattr(value, "value", None)
    if inner is not None and inner is not value:
        return jsonable(inner)
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def get_value(
    client: pyda.SimpleClient,
    endpoint: str,
    context: Optional[str] = None,
) -> Dict[str, Any]:
    if context:
        response = client.get(endpoint, context=context)
    else:
        response = client.get(endpoint)
    if response.exception:
        raise response.exception
    return {key: plain(value) for key, value in response.data.items()}


def _is_first_subscription_update(response: Any) -> bool:
    for obj in (response, getattr(response, "header", None)):
        if obj is None:
            continue
        if any(bool(getattr(obj, name, False)) for name in ("first_update", "is_first_update")):
            return True
        notice = getattr(obj, "notification_type", None)
        name = str(getattr(notice, "name", notice)).upper().replace(" ", "_")
        if "FIRST_UPDATE" in name:
            return True
    return False


def read_last_image(
    client: pyda.SimpleClient,
    fesa: str,
    *,
    get_selector: Optional[str] = None,
    use_subscription: bool = False,
    subscribe_selector: Optional[str] = None,
    timeout_s: float = 10.0,
    expected_shape: Optional[Tuple[int, ...]] = None,
) -> Dict[str, Any]:
    """Read a startup image using GET or a bounded live subscription.

    Some BTVDC deployments publish live images while their LastImage GET returns
    NO_DATA_AVAILABLE_FOR_USER. In subscription mode, wait for a live, nonempty
    2-D image instead. Bound setup as well as reception because SDK monitor
    creation can itself block. One worker owns and releases the SDK iterator.
    """
    endpoint = "%s/LastImage" % fesa
    if not use_subscription:
        return get_value(client, endpoint, context=get_selector)

    timeout_s = float(timeout_s)
    if not np.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("LastImage subscription timeout_s must be positive and finite")
    expected = None if expected_shape is None else tuple(expected_shape)
    deadline = time.monotonic() + timeout_s
    done = threading.Event()
    abandoned = threading.Event()
    result: Dict[str, Any] = {"last_reason": "no live image received"}

    def receive() -> None:
        subscription = iterator = response = None
        try:
            kwargs: Dict[str, Any] = {"receive_first_updates": False}
            if subscribe_selector:
                kwargs["context"] = subscribe_selector
            subscription = client.subscribe(endpoint, **kwargs)
            if abandoned.is_set():
                return
            # PyDA 1.0.10 starts monitoring in __iter__, not via start().
            iterator = iter(subscription)
            while not abandoned.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                try:
                    response = iterator.next(timeout=min(remaining, 0.25))
                except Exception as exc:
                    if type(exc).__name__ == "SubscriptionTimeoutError":
                        continue
                    raise
                if abandoned.is_set():
                    return
                exc = getattr(response, "exception", None)
                if exc is not None:
                    raise exc
                if _is_first_subscription_update(response):
                    result["last_reason"] = "only cached FIRST_UPDATE received"
                    continue
                raw_data = getattr(response, "data", None)
                if raw_data is None or "image2D" not in raw_data:
                    result["last_reason"] = "live update missing image2D"
                    continue
                data = {key: plain(value) for key, value in raw_data.items()}
                image = np.asarray(data["image2D"])
                shape = tuple(int(part) for part in image.shape)
                if image.ndim != 2 or image.size == 0:
                    result["last_reason"] = "live image2D is empty or not 2-D (shape=%s)" % (shape,)
                    continue
                if expected is not None and shape != expected:
                    result["last_reason"] = "expected shape=%s, last imageNumber=%s shape=%s" % (
                        expected, optional_int_like(data.get("imageNumber")), shape,
                    )
                    continue
                result["data"] = data
                return
        except BaseException as exc:  # surfaced unchanged to the caller
            # A failed iterator.next() traceback can retain its monitor via
            # the frame's ``self``. Keep the exception and stack locations,
            # but release locals in completed SDK frames (including causes).
            current: Optional[BaseException] = exc
            seen = set()
            while current is not None and id(current) not in seen:
                seen.add(id(current))
                traceback.clear_frames(current.__traceback__)
                current = current.__cause__ or current.__context__
            result["error"] = exc
        finally:
            # The PyDA iterator finalizer stops its monitor when released.
            response = iterator = subscription = None
            done.set()

    worker = threading.Thread(
        target=receive, name="LastImageStartup:%s" % fesa, daemon=True,
    )
    worker.start()
    completed = done.wait(max(0.0, deadline - time.monotonic()))
    if completed and "error" in result:
        raise result["error"]
    if completed and "data" in result:
        return result["data"]
    abandoned.set()
    raise TimeoutError(
        "camera %s LastImage subscription timed out after %.1fs (selector=%r): %s; "
        "check camera acquisition/trigger and selector"
        % (fesa, timeout_s, subscribe_selector, result["last_reason"])
    )


def set_value(
    client: pyda.SimpleClient,
    endpoint: str,
    payload: Mapping[str, Any],
    context: Optional[str] = None,
) -> Any:
    if context:
        response = client.set(endpoint, data=dict(payload), context=context)
    else:
        response = client.set(endpoint, data=dict(payload))
    if response.exception:
        raise response.exception
    return response


_CAMERA_PIXEL_FORMAT_VALUES = {
    "mono8": 44,
    "monochrome8": 44,
    "mono12": 40,
    "monochrome12": 40,
}
_CAMERA_PIXEL_FORMAT_NAMES = {
    44: "mono8",
    40: "mono12",
}


def _camera_pixel_format_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_").replace("_", "")


def camera_pixel_format_value(value: Any) -> int:
    value = plain(value)
    if isinstance(value, bool):
        raise RuntimeError("cameraPixelFormat must be mono8/mono12 or an integer enum")
    if isinstance(value, str):
        key = _camera_pixel_format_key(value)
        if key in _CAMERA_PIXEL_FORMAT_VALUES:
            return _CAMERA_PIXEL_FORMAT_VALUES[key]
        try:
            return int(value, 0)
        except ValueError:
            raise RuntimeError(
                "cameraPixelFormat must be mono8/mono12 or an integer enum, got %r"
                % value
            )
    try:
        return int(value)
    except (TypeError, ValueError):
        raise RuntimeError(
            "cameraPixelFormat must be mono8/mono12 or an integer enum, got %r"
            % value
        )


def _camera_pixel_format_name(value: Any) -> Optional[str]:
    try:
        return _CAMERA_PIXEL_FORMAT_NAMES[int(plain(value))]
    except (TypeError, ValueError, KeyError):
        return None


def camera_setting_payload(config_setting: Mapping[str, Any]) -> Dict[str, Any]:
    int64_fields = {
        "cameraExposureTimeUs",
    }
    bool_fields = {
        "cameraGammaEnable",
    }
    int_fields = {
        "acqTrigger",
        "cameraExposureAuto",
        "cameraGainAuto",
        "cameraBinningHor",
        "cameraBinningVer",
        "cameraBinningMode",
        "cameraPixelFormat",
    }
    float_fields = {
        "cameraTriggerDelay",
        "cameraFrameRate",
        "cameraGain",
    }

    payload: Dict[str, Any] = {}
    for key, value in config_setting.items():
        if key in ("roi", "roi_enabled", "device", "fesa") or key.startswith("_"):
            continue
        if key in int64_fields:
            payload[key] = np.int64(value)
        elif key in bool_fields:
            payload[key] = bool(value)
        elif key == "cameraPixelFormat":
            payload[key] = np.int32(camera_pixel_format_value(value))
        elif key in int_fields:
            payload[key] = np.int32(value)
        elif key in float_fields:
            payload[key] = np.float64(value)
        else:
            payload[key] = value
    return payload


def camera_roi_enabled(config_setting: Mapping[str, Any]) -> bool:
    return bool(config_setting.get("roi_enabled", True))


def camera_map_from_config(config: Mapping[str, Any]) -> Dict[str, str]:
    """Resolve active camera aliases to FESA device names from config."""

    cameras = config.get("cameras")
    camera_setting = config.get("camera_setting", {})
    if not cameras:
        raise RuntimeError("config cameras must not be empty")

    if isinstance(cameras, MappingABC):
        resolved = {}
        for alias, value in cameras.items():
            alias = str(alias)
            if isinstance(value, str):
                resolved[alias] = value
            elif isinstance(value, MappingABC):
                device = value.get("device", value.get("fesa"))
                if not device:
                    raise RuntimeError("cameras.%s missing device" % alias)
                resolved[alias] = str(device)
            else:
                raise RuntimeError("cameras.%s must be a device string or object" % alias)
        return resolved

    if isinstance(cameras, (list, tuple)):
        if not isinstance(camera_setting, MappingABC):
            raise RuntimeError("camera_setting must be an object")
        resolved = {}
        for alias in cameras:
            alias = str(alias)
            setting = camera_setting.get(alias)
            if not isinstance(setting, MappingABC):
                raise RuntimeError("camera_setting.%s must define active camera" % alias)
            device = setting.get("device", setting.get("fesa"))
            if not device:
                raise RuntimeError("camera_setting.%s missing device" % alias)
            resolved[alias] = str(device)
        return resolved

    raise RuntimeError("config cameras must be an object or list")


def camera_setting_for_alias(
    alias: str,
    camera_setting: Mapping[str, Any],
    camera_aliases: Iterable[str],
) -> Dict[str, Any]:
    if "default" in camera_setting or "per_camera" in camera_setting:
        default = camera_setting.get("default", {})
        per_camera = camera_setting.get("per_camera", {})
        if not isinstance(default, MappingABC):
            raise RuntimeError("camera_setting.default must be an object")
        if not isinstance(per_camera, MappingABC):
            raise RuntimeError("camera_setting.per_camera must be an object")
        specific = per_camera.get(alias, {})
        if not isinstance(specific, MappingABC):
            raise RuntimeError(
                "camera_setting.per_camera.%s must be an object" % alias
            )
        merged = {
            key: value
            for key, value in default.items()
            if not str(key).startswith("_")
        }
        merged.update(
            {
                key: value
                for key, value in specific.items()
                if not str(key).startswith("_")
            }
        )
        return merged

    alias_keys = set(camera_aliases).intersection(camera_setting)
    if alias_keys:
        if alias not in camera_setting:
            raise RuntimeError("camera_setting missing camera %s" % alias)
        specific = camera_setting[alias]
        if not isinstance(specific, MappingABC):
            raise RuntimeError("camera_setting.%s must be an object" % alias)
        return {
            key: value
            for key, value in specific.items()
            if not str(key).startswith("_")
        }

    return {
        key: value
        for key, value in camera_setting.items()
        if not str(key).startswith("_")
    }


def _roi_values(config_roi: Mapping[str, Any]) -> Tuple[int, int, int, int]:
    required = ("offsetX", "offsetY", "width", "height")
    missing = [key for key in required if key not in config_roi]
    if missing:
        raise RuntimeError("camera roi missing keys: %s" % ", ".join(missing))

    center_x = int(config_roi["offsetX"])
    center_y = int(config_roi["offsetY"])
    width = int(config_roi["width"])
    height = int(config_roi["height"])
    if center_x < 0 or center_y < 0 or width <= 0 or height <= 0:
        raise RuntimeError("camera roi must have non-negative center and positive size")

    offset_x = center_x - width // 2
    offset_y = center_y - height // 2
    return offset_x, offset_y, width, height


def _camera_binning_factors(setting: Mapping[str, Any]) -> Tuple[int, int]:
    factors = []
    for key in ("cameraBinningHor", "cameraBinningVer"):
        value = plain(setting.get(key, 1))
        if isinstance(value, bool):
            raise RuntimeError("%s must be an integer >= 1, got %r" % (key, value))
        try:
            factor = int(value)
        except (TypeError, ValueError):
            raise RuntimeError("%s must be an integer >= 1, got %r" % (key, value))
        if factor < 1:
            raise RuntimeError("%s must be an integer >= 1, got %r" % (key, value))
        factors.append(factor)
    return factors[0], factors[1]


def camera_binning_factors(setting: Mapping[str, Any]) -> Tuple[int, int]:
    """Return horizontal/vertical camera binning factors from a CameraSetting map."""

    return _camera_binning_factors(setting)


def _scale_roi_axis(
    *,
    alias: str,
    name: str,
    value: int,
    factor: int,
) -> int:
    if value % factor != 0:
        raise RuntimeError(
            "camera %s ROI %s=%d is not divisible by binning factor %d"
            % (alias, name, value, factor)
        )
    return value // factor


def fesa_roi_from_camera_box(
    alias: str,
    effective_setting: Mapping[str, Any],
    box: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    left, top, right, bottom = box
    width = int(right) - int(left)
    height = int(bottom) - int(top)
    if width <= 0 or height <= 0:
        raise RuntimeError("camera %s ROI box must have positive size" % alias)

    bin_x, bin_y = _camera_binning_factors(effective_setting)
    if bin_x == 1 and bin_y == 1:
        return int(left), int(top), width, height
    return (
        _scale_roi_axis(alias=alias, name="offset_x", value=int(left), factor=bin_x),
        _scale_roi_axis(alias=alias, name="offset_y", value=int(top), factor=bin_y),
        _scale_roi_axis(alias=alias, name="width", value=width, factor=bin_x),
        _scale_roi_axis(alias=alias, name="height", value=height, factor=bin_y),
    )


def _fesa_roi_values(
    alias: str,
    effective_setting: Mapping[str, Any],
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    config_roi = effective_setting["roi"]
    roi = _roi_values(config_roi)
    offset_x, offset_y, width, height = roi
    fesa_roi = fesa_roi_from_camera_box(
        alias,
        effective_setting,
        (offset_x, offset_y, offset_x + width, offset_y + height),
    )
    return roi, fesa_roi


def _camera_max_size(
    setting: Mapping[str, Any],
    hardware: Mapping[str, Any],
) -> Tuple[int, int]:
    window = np.asarray(setting.get("imageWindow"), dtype=np.int32).reshape(-1)
    if window.size >= 4:
        max_width = int(window[2])
        max_height = int(window[3])
    else:
        max_width = int(hardware.get("maxFrameWidth", 0) or 0)
        max_height = int(hardware.get("maxFrameHeight", 0) or 0)

    if max_width <= 0:
        max_width = int(hardware.get("maxFrameWidth", 0) or 0)
    if max_height <= 0:
        max_height = int(hardware.get("maxFrameHeight", 0) or 0)
    if max_width <= 0 or max_height <= 0:
        raise RuntimeError("camera max frame size is not available")

    return max_width, max_height


def _validate_roi_inside_window(
    roi: Tuple[int, int, int, int],
    max_width: int,
    max_height: int,
) -> None:
    offset_x, offset_y, width, height = roi
    if (
        offset_x < 0
        or offset_y < 0
        or offset_x + width > max_width
        or offset_y + height > max_height
    ):
        raise RuntimeError(
            "camera roi %s is outside max window %dx%d" % (
                list(roi),
                max_width,
                max_height,
            )
        )


def _wait_last_image_shape(
    client: pyda.SimpleClient,
    alias: str,
    fesa: str,
    roi: Tuple[int, int, int, int],
    get_selector: Optional[str] = None,
    *,
    use_subscription: bool = False,
    subscribe_selector: Optional[str] = None,
) -> None:
    _offset_x, _offset_y, width, height = roi
    expected_shape = (height, width)
    if use_subscription:
        read_last_image(
            client, fesa, use_subscription=True,
            subscribe_selector=subscribe_selector, expected_shape=expected_shape,
        )
        return
    deadline = time.monotonic() + 10.0
    last_shape: Tuple[int, ...] = ()
    last_image_number: Optional[int] = None

    while time.monotonic() < deadline:
        data = get_value(client, "%s/LastImage" % fesa, context=get_selector)
        image = np.asarray(data["image2D"])
        last_shape = tuple(int(part) for part in image.shape)
        last_image_number = optional_int_like(data.get("imageNumber"))
        if last_shape == expected_shape:
            return
        time.sleep(0.2)

    raise RuntimeError(
        "camera %s LastImage shape mismatch after ROI: expected %s, "
        "last imageNumber=%s shape=%s"
        % (alias, expected_shape, last_image_number, last_shape)
    )


def _configure_camera_roi(
    client: pyda.SimpleClient,
    alias: str,
    fesa: str,
    effective_setting: Mapping[str, Any],
    get_selector: Optional[str] = None,
    *,
    use_subscription: bool = False,
    subscribe_selector: Optional[str] = None,
) -> None:
    setting = get_value(client, "%s/Setting" % fesa)
    hardware = get_value(client, "%s/HardwareStatus" % fesa)
    config_roi, fesa_roi = _fesa_roi_values(alias, effective_setting)
    max_width, max_height = _camera_max_size(setting, hardware)
    try:
        _validate_roi_inside_window(config_roi, max_width, max_height)
    except RuntimeError as config_exc:
        try:
            _validate_roi_inside_window(fesa_roi, max_width, max_height)
        except RuntimeError:
            raise config_exc

    image_window = np.asarray(setting["imageWindow"], dtype=np.int32)
    payload = {
        "imageROIEnable": True,
        "imageWindow": image_window,
        "imageROI": np.asarray(fesa_roi, dtype=np.int32),
    }

    bin_x, bin_y = _camera_binning_factors(effective_setting)
    if config_roi == fesa_roi:
        print("configure camera %s ROI=%s max=%dx%d" % (
            alias,
            list(fesa_roi),
            max_width,
            max_height,
        ))
    else:
        print("configure camera %s ROI=%s -> fesaROI=%s binning=%dx%d max=%dx%d" % (
            alias,
            list(config_roi),
            list(fesa_roi),
            bin_x,
            bin_y,
            max_width,
            max_height,
        ))
    set_value(client, "%s/Setting" % fesa, payload)

    deadline = time.monotonic() + 5.0
    applied_enabled = False
    applied_roi: Tuple[int, ...] = ()
    while time.monotonic() < deadline:
        params = get_value(client, "%s/AcquiredParameters" % fesa)
        applied_enabled = bool(params.get("acqImageROIEnable"))
        applied_roi = tuple(
            int(value) for value in np.asarray(params.get("acqImageROI")).reshape(-1)
        )
        if applied_enabled and applied_roi == fesa_roi:
            _wait_last_image_shape(
                client, alias, fesa, fesa_roi, get_selector=get_selector,
                use_subscription=use_subscription, subscribe_selector=subscribe_selector,
            )
            return
        time.sleep(0.2)

    raise RuntimeError(
        "camera %s ROI readback mismatch: requested enabled=True roi=%s, "
        "applied enabled=%s roi=%s"
        % (alias, list(fesa_roi), applied_enabled, list(applied_roi))
    )


def _disable_camera_roi(
    client: pyda.SimpleClient,
    alias: str,
    fesa: str,
    get_selector: Optional[str] = None,
    *,
    use_subscription: bool = False,
    subscribe_selector: Optional[str] = None,
) -> None:
    setting = get_value(client, "%s/Setting" % fesa)
    hardware = get_value(client, "%s/HardwareStatus" % fesa)
    max_width, max_height = _camera_max_size(setting, hardware)
    image_window = np.asarray(setting["imageWindow"], dtype=np.int32)
    image_roi = np.asarray(
        setting.get("imageROI", [0, 0, max_width, max_height]),
        dtype=np.int32,
    )
    payload = {
        "imageROIEnable": False,
        "imageWindow": image_window,
        "imageROI": image_roi,
    }

    print("configure camera %s full-frame ROI disabled max=%dx%d" % (
        alias,
        max_width,
        max_height,
    ))
    set_value(client, "%s/Setting" % fesa, payload)

    deadline = time.monotonic() + 5.0
    applied_enabled: Optional[bool] = None
    while time.monotonic() < deadline:
        params = get_value(client, "%s/AcquiredParameters" % fesa)
        applied_enabled = bool(params.get("acqImageROIEnable"))
        if not applied_enabled:
            _wait_last_image_shape(
                client, alias, fesa, (0, 0, max_width, max_height),
                get_selector=get_selector,
                use_subscription=use_subscription, subscribe_selector=subscribe_selector,
            )
            return
        time.sleep(0.2)

    raise RuntimeError(
        "camera %s ROI disable readback mismatch: enabled=%s"
        % (alias, applied_enabled)
    )


_CAMERA_SETTING_READBACK_FIELDS = {
    "cameraGainAuto": "acqCameraGainAuto",
    "cameraGain": "acqCameraGain",
    "cameraExposureAuto": "acqCameraExposureAuto",
    "cameraExposureTimeUs": "acqCameraExposureTimeUs",
    "cameraPixelFormat": "acqCameraPixelFormat",
    "cameraGammaEnable": "acqCameraGammaEnable",
    "cameraBinningHor": "acqCameraBinningHor",
    "cameraBinningVer": "acqCameraBinningVer",
    "cameraBinningMode": "acqCameraBinningMode",
}
# Existing BTVDC VM helper scripts document enum value 1 as OFF/manual.
_CAMERA_AUTO_OFF_VALUE = 1
_REQUIRED_SCIENCE_CAMERA_SETTING_KEYS = (
    "cameraGainAuto",
    "cameraGain",
    "cameraExposureAuto",
    "cameraExposureTimeUs",
    "cameraPixelFormat",
    "cameraGammaEnable",
)
_CAMERA_SOURCE_ACQUIRED_PARAMETER_FIELDS = (
    "acqCameraBlackLevel",
    "acqCameraDigitalShift",
    "acqCameraFlipHor",
    "acqCameraFlipVer",
)


def _require_science_camera_setting(
    alias: str,
    effective_setting: Mapping[str, Any],
) -> None:
    missing = [
        key for key in _REQUIRED_SCIENCE_CAMERA_SETTING_KEYS
        if key not in effective_setting
    ]
    if missing:
        raise RuntimeError(
            "camera %s setting missing required source-data keys: %s"
            % (alias, ", ".join(missing))
        )

    for config_key in ("cameraGainAuto", "cameraExposureAuto"):
        try:
            requested = int(plain(effective_setting[config_key]))
        except (TypeError, ValueError):
            raise RuntimeError(
                "camera %s %s config is not an integer enum: %r"
                % (alias, config_key, effective_setting[config_key])
            )
        if requested != _CAMERA_AUTO_OFF_VALUE:
            raise RuntimeError(
                "camera %s %s config must request manual/off value %d, got %d"
                % (alias, config_key, _CAMERA_AUTO_OFF_VALUE, requested)
            )

    if bool(effective_setting["cameraGammaEnable"]):
        raise RuntimeError("camera %s cameraGammaEnable must be false" % alias)

    try:
        camera_pixel_format_value(effective_setting["cameraPixelFormat"])
    except RuntimeError as exc:
        raise RuntimeError("camera %s %s" % (alias, exc)) from exc


def _readback_equal(requested: Any, actual: Any) -> bool:
    requested = plain(requested)
    actual = plain(actual)

    if isinstance(requested, bool) or isinstance(actual, bool):
        return bool(requested) == bool(actual)

    try:
        requested_float = float(requested)
        actual_float = float(actual)
    except (TypeError, ValueError):
        return requested == actual

    return abs(requested_float - actual_float) <= 1e-9


def _require_readback_field(
    *,
    alias: str,
    config_key: str,
    readback_key: str,
    requested: Any,
    actual: Any,
) -> Any:
    if config_key == "cameraPixelFormat":
        requested = camera_pixel_format_value(requested)

    if actual is None:
        raise RuntimeError(
            "camera %s readback missing %s for %s"
            % (alias, readback_key, config_key)
        )
    if not _readback_equal(requested, actual):
        raise RuntimeError(
            "camera %s %s readback mismatch: requested %r, applied %r"
            % (alias, config_key, requested, actual)
        )
    return jsonable(actual)


def _verify_auto_disabled(alias: str, config_key: str, actual: Any) -> None:
    try:
        actual_int = int(plain(actual))
    except (TypeError, ValueError):
        raise RuntimeError(
            "camera %s %s readback is not an integer enum: %r"
            % (alias, config_key, actual)
        )
    if actual_int != _CAMERA_AUTO_OFF_VALUE:
        raise RuntimeError(
            "camera %s %s is not in manual/off state: expected %d, got %d"
            % (alias, config_key, _CAMERA_AUTO_OFF_VALUE, actual_int)
        )


def _camera_readback_metadata(
    client: pyda.SimpleClient,
    alias: str,
    fesa: str,
    effective_setting: Mapping[str, Any],
) -> Dict[str, Any]:
    deadline = time.monotonic() + 5.0
    last_error: Optional[RuntimeError] = None

    while time.monotonic() < deadline:
        try:
            params = get_value(client, "%s/AcquiredParameters" % fesa)
            readback: Dict[str, Any] = {}

            for config_key, readback_key in _CAMERA_SETTING_READBACK_FIELDS.items():
                if config_key not in effective_setting:
                    continue
                readback[config_key] = _require_readback_field(
                    alias=alias,
                    config_key=config_key,
                    readback_key=readback_key,
                    requested=effective_setting[config_key],
                    actual=params.get(readback_key),
                )

            if "cameraPixelFormat" in readback:
                name = _camera_pixel_format_name(readback["cameraPixelFormat"])
                if name is not None:
                    readback["cameraPixelFormatName"] = name

            for config_key in ("cameraGainAuto", "cameraExposureAuto"):
                if config_key in readback:
                    _verify_auto_disabled(alias, config_key, readback[config_key])

            if "roi" in effective_setting or not camera_roi_enabled(effective_setting):
                requested_roi, requested_fesa_roi = _fesa_roi_values(
                    alias,
                    effective_setting,
                ) if "roi" in effective_setting else (None, None)
                applied_enabled = params.get("acqImageROIEnable")
                applied_roi_value = params.get("acqImageROI")
                if applied_enabled is None:
                    raise RuntimeError("camera %s ROI readback is missing" % alias)
                applied_roi = ()
                if applied_roi_value is not None:
                    applied_roi = tuple(
                        int(value)
                        for value in np.asarray(applied_roi_value).reshape(-1)
                    )
                if not camera_roi_enabled(effective_setting):
                    if bool(applied_enabled):
                        raise RuntimeError(
                            "camera %s ROI readback mismatch: requested enabled=False, "
                            "applied enabled=%s roi=%s"
                            % (alias, bool(applied_enabled), list(applied_roi))
                        )
                    readback["roi"] = {
                        "enabled": False,
                    }
                elif (
                    applied_roi_value is None
                    or not bool(applied_enabled)
                    or applied_roi != requested_fesa_roi
                ):
                    raise RuntimeError(
                        "camera %s ROI readback mismatch: requested enabled=True roi=%s, "
                        "applied enabled=%s roi=%s"
                        % (
                            alias,
                            list(requested_fesa_roi),
                            bool(applied_enabled),
                            list(applied_roi),
                        )
                    )
                else:
                    readback["roi"] = {
                        "enabled": bool(applied_enabled),
                        "offsetX": applied_roi[0],
                        "offsetY": applied_roi[1],
                        "width": applied_roi[2],
                        "height": applied_roi[3],
                    }
                    if requested_roi != requested_fesa_roi:
                        readback["roiUnbinned"] = {
                            "offsetX": requested_roi[0],
                            "offsetY": requested_roi[1],
                            "width": requested_roi[2],
                            "height": requested_roi[3],
                        }

            print(
                "camera %s readback gain_auto=%s gain=%s exposure_auto=%s "
                "exposure_us=%s pixel_format=%s binning=%sx%s/%s "
                "gamma_enable=%s roi=%s"
                % (
                    alias,
                    readback.get("cameraGainAuto"),
                    readback.get("cameraGain"),
                    readback.get("cameraExposureAuto"),
                    readback.get("cameraExposureTimeUs"),
                    readback.get("cameraPixelFormatName")
                    or readback.get("cameraPixelFormat"),
                    readback.get("cameraBinningHor"),
                    readback.get("cameraBinningVer"),
                    readback.get("cameraBinningMode"),
                    readback.get("cameraGammaEnable"),
                    readback.get("roi"),
                )
            )
            return {
                "fesa": fesa,
                "requested": jsonable(effective_setting),
                "readback": readback,
            }
        except RuntimeError as exc:
            last_error = exc
            time.sleep(0.2)

    raise RuntimeError(
        "camera %s readback did not match requested setting within 5.0s: %s"
        % (alias, last_error)
    ) from last_error


def configure_cameras(
    client: pyda.SimpleClient,
    cameras: Mapping[str, str],
    camera_setting: Mapping[str, Any],
    *,
    get_selector: Optional[str] = None,
    use_subscription: bool = False,
    subscribe_selector: Optional[str] = None,
) -> Dict[str, Any]:
    readbacks: Dict[str, Any] = {}
    for alias, fesa in cameras.items():
        effective_setting = camera_setting_for_alias(alias, camera_setting, cameras)
        _require_science_camera_setting(alias, effective_setting)
        payload = camera_setting_payload(effective_setting)
        roi = effective_setting.get("roi")
        print("configure camera %s %s" % (alias, fesa))
        if payload:
            set_value(client, "%s/CameraSetting" % fesa, payload)
        if roi is not None and camera_roi_enabled(effective_setting):
            _configure_camera_roi(
                client, alias, fesa, effective_setting, get_selector=get_selector,
                use_subscription=use_subscription, subscribe_selector=subscribe_selector,
            )
        elif not camera_roi_enabled(effective_setting):
            _disable_camera_roi(
                client, alias, fesa, get_selector=get_selector,
                use_subscription=use_subscription, subscribe_selector=subscribe_selector,
            )
        readbacks[alias] = _camera_readback_metadata(
            client,
            alias,
            fesa,
            effective_setting,
        )
    return readbacks


_CAMERA_STATUS_NAMES = {
    -1: "MISSING", 0: "OK", 1: "ERROR", 2: "UNREACHABLE", 3: "REBOOTING",
    4: "DISCONNECTED", 5: "SYNCHRONIZING_PARAMETERS",
}


def ensure_camera_alive(
    client: pyda.SimpleClient,
    alias: str,
    fesa: str,
    *,
    timeout_s: float = 30.0,
    poll_s: float = 1.0,
    power_settle_s: float = 10.0,
    force: bool = False,
) -> Dict[str, Any]:
    """Make sure the camera behind a BTVDC device is powered and connected.

    This is what operators do in NavPy before a session: ManageCameraPower=ON and
    ManageCameraConnection=CONNECT. Only the ON / CONNECT directions are ever sent.

    A camera that already reports acqCameraStatus=OK is left alone (unless ``force``).
    The power command is sent only when the server is actually talking to the power
    switch (cameraSwitchConnectionStatus=CONNECTED) and reports the port OFF: when the
    switch controller (BTVDCPC) is disconnected those fields read OFF/NOT_CONNECTED
    even while the camera streams, and the command is refused by the server.
    Command errors are collected as warnings; what decides is the final
    acqCameraStatus poll, which raises with the server's message on timeout.
    """
    status = get_value(client, "%s/AcquisitionStatus" % fesa)
    camera_was_ok = int(plain(status.get("acqCameraStatus", -1))) == 0
    report: Dict[str, Any] = {"fesa": fesa, "camera_was_ok": camera_was_ok, "actions": [],
                              "warnings": []}
    if camera_was_ok and not force:
        report["acqCameraStatus"] = "OK"
        print("camera %s %s alive: already OK, no command sent" % (alias, fesa))
        return report

    hardware = get_value(client, "%s/HardwareStatus" % fesa)
    switch_connected = int(plain(hardware.get("cameraSwitchConnectionStatus", 0))) == 1
    power_on = int(plain(hardware.get("cameraSwitchStatus", 1))) == 1
    report["power_switch"] = {"controller_connected": switch_connected, "port_on": power_on}

    def command(prop: str, label: str) -> bool:
        try:
            set_value(client, "%s/%s" % (fesa, prop), {"operation": np.int32(1)})
            report["actions"].append(label)
            return True
        except Exception as exc:  # the server refuses, e.g. power switch controller down
            report["warnings"].append("%s refused: %s" % (label, str(exc).splitlines()[0][:200]))
            return False

    if switch_connected and (force or not power_on):
        if command("ManageCameraPower", "ManageCameraPower=ON") and not power_on:
            time.sleep(power_settle_s)  # camera boot time before a CONNECT can succeed
    command("ManageCameraConnection", "ManageCameraConnection=CONNECT")

    deadline = time.monotonic() + timeout_s
    while True:
        status = get_value(client, "%s/AcquisitionStatus" % fesa)
        code = int(plain(status.get("acqCameraStatus", -1)))
        if code == 0:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "camera %s (%s) not usable after %.0fs: acqCameraStatus=%s (%s) %s%s"
                % (alias, fesa, timeout_s, code, _CAMERA_STATUS_NAMES.get(code, "?"),
                   plain(status.get("acqMsg", "")),
                   (" | " + "; ".join(report["warnings"])) if report["warnings"] else "")
            )
        time.sleep(poll_s)

    report["acqCameraStatus"] = "OK"
    print("camera %s %s alive: %s -> OK%s" % (
        alias, fesa, ", ".join(report["actions"]) or "no command accepted",
        (" (warnings: " + "; ".join(report["warnings"]) + ")") if report["warnings"] else ""))
    return report


def ensure_cameras_alive(
    client: pyda.SimpleClient,
    cameras: Mapping[str, str],
    writer_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    """Run ensure_camera_alive for every camera unless writer.ensure_camera_alive is false."""
    if not bool(writer_cfg.get("ensure_camera_alive", True)):
        return {}
    return {
        alias: ensure_camera_alive(
            client, alias, fesa,
            timeout_s=float(writer_cfg.get("camera_alive_timeout_s", 30.0)),
            force=bool(writer_cfg.get("camera_alive_force", False)),
        )
        for alias, fesa in cameras.items()
    }


def collect_camera_source_provenance(
    client: pyda.SimpleClient,
    cameras: Mapping[str, str],
) -> Dict[str, Any]:
    """Read camera-side provenance for source-of-truth image decoding.

    This is metadata only. It does not configure cameras or change image bytes.
    """

    provenance: Dict[str, Any] = {}
    for alias, fesa in cameras.items():
        params = get_value(client, "%s/AcquiredParameters" % fesa)
        expert = get_value(client, "%s/ExpertSetting" % fesa)

        acquired_parameters: Dict[str, Any] = {}
        missing = []
        for field in _CAMERA_SOURCE_ACQUIRED_PARAMETER_FIELDS:
            if field not in params:
                missing.append(field)
            else:
                acquired_parameters[field] = jsonable(params[field])
        if missing:
            raise RuntimeError(
                "camera %s AcquiredParameters missing source provenance fields: %s"
                % (alias, ", ".join(missing))
            )

        provenance[alias] = {
            "fesa": fesa,
            "acquired_parameters": acquired_parameters,
            "expert_setting": jsonable(expert),
        }
        print(
            "camera %s source provenance black_level=%s digital_shift=%s "
            "flip_h=%s flip_v=%s cleaning=%s software_binning=%s"
            % (
                alias,
                acquired_parameters.get("acqCameraBlackLevel"),
                acquired_parameters.get("acqCameraDigitalShift"),
                acquired_parameters.get("acqCameraFlipHor"),
                acquired_parameters.get("acqCameraFlipVer"),
                expert.get("enableImageCleaning"),
                expert.get("softwareBinningFactor"),
            )
        )

    return provenance


class PollingCameraTap:
    """Poll one BTVDC ``LastImage`` and emit raw Frames when imageNumber changes."""

    def __init__(
        self,
        alias: str,
        fesa_device: str,
        client: pyda.SimpleClient,
        *,
        poll_s: float,
        stamp_fields: Iterable[str],
        min_ns_epoch: int,
        pulse_id_fields: Iterable[str] = (),
        settle_poll_s: float = 0.005,
        settle_timeout_s: float = 0.25,
        max_trailing_zero_rows: Optional[int] = 0,
        sync_cfg: Optional[Mapping[str, Any]] = None,
        allow_image_number_reset: bool = False,
        get_selector: Optional[str] = None,
    ) -> None:
        self.device = alias
        self.fesa_device = fesa_device
        self._client = client
        self._poll_s = float(poll_s)
        self._stamp_fields = list(stamp_fields)
        self._min_ns_epoch = int(min_ns_epoch)
        self._pulse_id_fields = list(pulse_id_fields)
        self._settle_poll_s = float(settle_poll_s)
        self._settle_timeout_s = float(settle_timeout_s)
        self._max_trailing_zero_rows = max_trailing_zero_rows
        self._sync_cfg = dict(sync_cfg or {})
        self._allow_image_number_reset = bool(allow_image_number_reset)
        self._get_selector = get_selector

        self.image_shape: Optional[Tuple[int, ...]] = None
        self.image_dtype: Optional[str] = None
        self.payload_nbytes: Optional[int] = None
        self.stamp_source: Optional[str] = None
        self.probe_pulse_id: int = NO_PULSE_ID
        self.error: Optional[BaseException] = None
        self.skipped_incomplete = 0
        self.last_incomplete_reason: Optional[str] = None
        self.image_number_resets = 0
        self.last_image_number_reset: Optional[Tuple[int, int]] = None

        self._last_image_number: Optional[int] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def probe(self) -> None:
        data = self._read_last_image()
        image = np.ascontiguousarray(np.asarray(data["image2D"]))
        image_number = require_int_like(data.get("imageNumber"), "imageNumber")
        _stamp, stamp_source = self._detect_stamp(data)
        self.probe_pulse_id = self._detect_pulse_id(data)

        self.image_shape = tuple(int(part) for part in image.shape)
        self.image_dtype = str(image.dtype)
        self.payload_nbytes = int(image.nbytes)
        self.stamp_source = stamp_source
        self._last_image_number = image_number

    def start(self, emit: Callable[[Frame], None]) -> None:
        if self.payload_nbytes is None:
            raise RuntimeError("camera %s must be probed before start" % self.device)
        self.error = None
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(emit,),
            name="PollingCameraTap:%s" % self.device,
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self, emit: Callable[[Frame], None]) -> None:
        try:
            while not self._stop_event.is_set():
                data = self._read_last_image()
                image_number = require_int_like(data.get("imageNumber"), "imageNumber")

                if image_number == self._last_image_number:
                    self._stop_event.wait(self._poll_s)
                    continue

                data = self._settle_last_image(data)
                if data is None:
                    self._stop_event.wait(self._poll_s)
                    continue

                self._build_and_emit(data, emit)
                self._stop_event.wait(self._poll_s)
        except BaseException as exc:
            self.error = exc
        finally:
            print(
                "camera tap %s skipped_incomplete=%d last_incomplete_reason=%s"
                % (self.device, self.skipped_incomplete, self.last_incomplete_reason)
            )

    def _build_and_emit(self, data: Mapping[str, Any], emit: Callable[[Frame], None]) -> None:
        """Build one Frame from a complete LastImage payload and emit it.

        Shared by the polling loop (:meth:`_run`) and the subscription callback
        (:class:`SubscribingCameraTap`). ``data`` must already be a complete
        image dict (settled, for the polling path).
        """
        image_number = require_int_like(data.get("imageNumber"), "imageNumber")
        image = np.ascontiguousarray(np.asarray(data["image2D"]))
        payload = image.tobytes(order="C")

        if len(payload) != self.payload_nbytes:
            raise RuntimeError(
                "%s payload size changed: expected %d, got %d"
                % (self.device, self.payload_nbytes, len(payload))
            )

        stamp_ns, _source = self._detect_stamp(data)
        pulse_id = self._detect_pulse_id(data)
        checksum = self._image_checksum(image)
        trailing_zero_rows = self._trailing_zero_rows(image)
        sync_metadata = extract_sync_candidates(
            data,
            local_receive_ns=time.time_ns(),
            trailing_zero_rows=trailing_zero_rows,
            crc32=checksum,
        )
        chosen_source, chosen_value = choose_frame_sync_candidate(
            sync_metadata,
            self._sync_cfg,
        )
        sync_metadata["chosen_sync_source"] = chosen_source
        sync_metadata["chosen_sync_value"] = chosen_value
        sync_metadata["legacy_stamp_source"] = _source

        frame = Frame(
            device=self.device,
            source_image_number=image_number,
            acq_stamp_ns=stamp_ns,
            payload_bytes=payload,
            pulse_id=pulse_id,
        )
        object.__setattr__(frame, "sync_metadata", sync_metadata)
        emit(frame)
        self._last_image_number = image_number

    def _check_image_number_order(self, image_number: int) -> None:
        previous = self._last_image_number
        if previous is None or image_number >= previous:
            return
        if not self._allow_image_number_reset:
            raise RuntimeError(
                "%s imageNumber went backwards: %d after %d"
                % (self.device, image_number, previous)
            )

        self.image_number_resets += 1
        self.last_image_number_reset = (previous, image_number)
        print(
            "camera tap %s accepted imageNumber reset: %d -> %d"
            % (self.device, previous, image_number),
            flush=True,
        )

    def _read_last_image(self) -> Dict[str, Any]:
        return get_value(self._client, "%s/LastImage" % self.fesa_device, context=self._get_selector)

    def _image_checksum(self, image: Any) -> int:
        arr = np.ascontiguousarray(np.asarray(image))
        return zlib.crc32(arr.tobytes(order="C")) & 0xFFFFFFFF

    def _trailing_zero_rows(self, image: Any) -> int:
        arr = np.asarray(image)
        if arr.ndim == 0:
            return int(arr == 0)
        if arr.shape[0] == 0:
            return 0

        axes = tuple(range(1, arr.ndim))
        row_nonzero = np.any(arr != 0, axis=axes) if axes else (arr != 0)
        row_nonzero = np.asarray(row_nonzero).reshape(-1)
        if not np.any(row_nonzero):
            return int(row_nonzero.size)
        return int(row_nonzero.size - int(np.flatnonzero(row_nonzero)[-1]) - 1)

    def _settle_snapshot(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        image = np.asarray(data["image2D"])
        stamp, _source = self._detect_stamp(data)
        trailing = (
            None
            if self._max_trailing_zero_rows is None
            else self._trailing_zero_rows(image)
        )
        return {
            "image_number": require_int_like(data.get("imageNumber"), "imageNumber"),
            "stamp": stamp,
            "shape": tuple(int(part) for part in image.shape),
            "checksum": self._image_checksum(image),
            "trailing_zero_rows": trailing,
        }

    def _settle_status(
        self,
        previous: Mapping[str, Any],
        current: Mapping[str, Any],
    ) -> Tuple[bool, str]:
        problems = []
        for key in ("image_number", "stamp", "shape", "checksum"):
            if current[key] != previous[key]:
                problems.append("%s changed" % key)

        trailing = current["trailing_zero_rows"]
        if (
            trailing is not None
            and self._max_trailing_zero_rows is not None
            and trailing > self._max_trailing_zero_rows
        ):
            problems.append(
                "trailing_zero_rows=%d > %d"
                % (trailing, self._max_trailing_zero_rows)
            )

        if problems:
            return False, ", ".join(problems)
        return True, "stable"

    def _settle_last_image(self, initial_data: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        current = self._settle_snapshot(initial_data)
        image_number = int(current["image_number"])
        started = time.monotonic()
        last_reason = "not checked"

        while not self._stop_event.is_set():
            elapsed = time.monotonic() - started
            if elapsed >= self._settle_timeout_s:
                self.skipped_incomplete += 1
                self.last_incomplete_reason = (
                    "imageNumber %s did not settle within %.3fs: %s"
                    % (image_number, self._settle_timeout_s, last_reason)
                )
                self._last_image_number = image_number
                return None

            wait_s = min(self._settle_poll_s, self._settle_timeout_s - elapsed)
            if wait_s > 0:
                self._stop_event.wait(wait_s)
            if self._stop_event.is_set():
                return None

            next_data = self._read_last_image()
            next_snapshot = self._settle_snapshot(next_data)
            next_number = int(next_snapshot["image_number"])
            if next_number != image_number:
                current = next_snapshot
                image_number = next_number
                started = time.monotonic()
                last_reason = "newer imageNumber appeared while settling"
                continue

            stable, last_reason = self._settle_status(current, next_snapshot)
            if stable:
                return next_data
            current = next_snapshot

    def _detect_stamp(self, data: Mapping[str, Any]) -> Tuple[int, str]:
        for field in self._stamp_fields:
            stamp = optional_int_like(data.get(field))
            if stamp is not None and stamp >= self._min_ns_epoch:
                return stamp, field
        return time.time_ns(), "local_clock"

    def _detect_pulse_id(self, data: Mapping[str, Any]) -> int:
        for field in self._pulse_id_fields:
            pulse_id = optional_int_like(data.get(field))
            if pulse_id is not None and pulse_id >= 0:
                return pulse_id
        return NO_PULSE_ID


class SubscribingCameraTap(PollingCameraTap):
    """Receive FESA subscription updates and emit Frames.

    Reuses the polling tap probe and frame builder. The polling interval,
    poll-rate guard and settle re-read loop are unused. Iterator and callback
    subscription APIs share the same Frame interface.
    """

    def __init__(self, *args: Any, selector: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._selector = selector
        self._subscription: Any = None

    def _read_last_image(self) -> Dict[str, Any]:
        return read_last_image(
            self._client, self.fesa_device, use_subscription=True,
            subscribe_selector=self._selector,
        )

    def start(self, emit: Callable[[Frame], None]) -> None:
        if self.payload_nbytes is None:
            raise RuntimeError("camera %s must be probed before start" % self.device)
        self.error = None
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_subscription,
            args=(emit,),
            name="SubscribingCameraTap:%s" % self.device,
            daemon=True,
        )
        self._thread.start()

    def _run_subscription(self, emit: Callable[[Frame], None]) -> None:
        endpoint = "%s/LastImage" % self.fesa_device
        try:
            if not self._run_callback_subscription(endpoint, emit):
                self._run_iterator_subscription(endpoint, emit)
        except BaseException as exc:  # noqa: BLE001 - surfaced via self.error
            if self.error is None:
                self.error = exc
        finally:
            self._close_subscription()
            print("camera tap %s (subscribe) stopped" % self.device)

    def stop(self) -> None:
        self._stop_event.set()
        self._close_subscription()

    def _close_subscription(self) -> None:
        subscription = self._subscription
        self._subscription = None
        if subscription is None:
            return
        for name in ("stop", "cancel", "close", "unsubscribe"):
            closer = getattr(subscription, name, None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass
                return

    def _run_callback_subscription(
        self,
        endpoint: str,
        emit: Callable[[Frame], None],
    ) -> bool:
        def on_update(response: Any) -> None:
            try:
                self._emit_subscription_response(response, emit)
            except BaseException as exc:  # noqa: BLE001 - stored for main tap thread
                if self.error is None:
                    self.error = exc
                self._stop_event.set()

        try:
            if self._selector:
                subscription = self._client.subscribe(
                    endpoint,
                    context=self._selector,
                    callback=on_update,
                    receive_first_updates=False,
                )
            else:
                subscription = self._client.subscribe(endpoint, callback=on_update, receive_first_updates=False)
        except TypeError:
            return False

        self._subscription = subscription
        self._start_subscription_if_needed(subscription)
        while not self._stop_event.wait(0.1):
            if self.error is not None:
                raise self.error
        if self.error is not None:
            raise self.error
        return True

    def _run_iterator_subscription(
        self,
        endpoint: str,
        emit: Callable[[Frame], None],
    ) -> None:
        if self._selector:
            subscription = self._client.subscribe(endpoint, context=self._selector, receive_first_updates=False)
        else:
            subscription = self._client.subscribe(endpoint, receive_first_updates=False)
        self._subscription = subscription
        self._start_subscription_if_needed(subscription)
        for response in subscription:  # blocks until the server pushes a frame
            if self._stop_event.is_set():
                break
            self._emit_subscription_response(response, emit)

    def _emit_subscription_response(
        self,
        response: Any,
        emit: Callable[[Frame], None],
    ) -> None:
        if _is_first_subscription_update(response):
            return

        exc = getattr(response, "exception", None)
        if exc is not None:
            raise exc

        raw_data = getattr(response, "data", None)
        if raw_data is None:
            raise RuntimeError("subscription response has no data")
        data = {key: plain(value) for key, value in raw_data.items()}
        image_number = require_int_like(data.get("imageNumber"), "imageNumber")
        if image_number == self._last_image_number:
            return
        self._check_image_number_order(image_number)
        self._build_and_emit(data, emit)

    def _start_subscription_if_needed(self, subscription: Any) -> None:
        starter = getattr(subscription, "start", None)
        if callable(starter):
            starter()


class ClearMagnetSet:
    """Small PowRs422 current-set adapter for CLEAR magnets.

    Current changes write ``SettingPPM``. Readback uses cycle-bound
    ``Acquisition`` with the configured SCT selector.
    """

    BUSY_BIT = 4

    def __init__(
        self,
        client: pyda.SimpleClient,
        cfg: Mapping[str, Any],
    ) -> None:
        self.client = client
        devices = cfg.get("devices", {})
        if not isinstance(devices, MappingABC) or not devices:
            raise RuntimeError("magnets.devices must be a non-empty object")
        self.devices = {str(alias): str(device) for alias, device in devices.items()}
        self.aliases = list(self.devices)
        self.selector = str(cfg.get("selector", "SCT.USER.SETUP"))
        setting_selector = cfg.get("setting_selector", None)
        self.setting_selector = (
            None if setting_selector in (None, "") else str(setting_selector)
        )
        self.poll_s = float(cfg.get("poll_s", 0.2))
        self.timeout_s = float(cfg.get("timeout_s", 20.0))
        self.settle_s = float(cfg.get("settle_s", 0.0))
        self.settle_tolerance_a = float(cfg.get("settle_tolerance_a", 0.02))
        # settle_mode: "tolerance" (wait until |measured-target| <= settle_tolerance_a
        # AND not BUSY) or "busy_only" (wait only until not BUSY, then proceed).
        self.settle_mode = str(cfg.get("settle_mode", "tolerance")).lower()
        if self.settle_mode not in ("tolerance", "busy_only"):
            raise RuntimeError("magnets.settle_mode must be 'tolerance' or 'busy_only'")
        current_tol_abs = cfg.get("current_tolAbs", None)
        current_tol_rel = cfg.get("current_tolRel", None)
        self.current_tol_abs = (
            None if current_tol_abs is None else float(current_tol_abs)
        )
        self.current_tol_rel = (
            None if current_tol_rel is None else float(current_tol_rel)
        )
        self.start_currents: Dict[str, float] = {}
        # Optional per-alias HARD limit [min, max] (A). Enforced on every write below,
        # in addition to the FESA range.
        self.limits: Dict[str, Tuple[float, float]] = {}
        limits_cfg = cfg.get("limits", {})
        if isinstance(limits_cfg, MappingABC):
            for alias, rng in limits_cfg.items():
                alias = str(alias)
                if alias not in self.devices:
                    continue
                lo, hi = float(rng[0]), float(rng[1])
                if lo > hi:
                    raise RuntimeError("magnets.limits.%s requires min <= max" % alias)
                self.limits[alias] = (lo, hi)

    def _clamp_to_limit(self, alias: str, value: float) -> float:
        rng = self.limits.get(alias)
        if rng is None:
            return value
        lo, hi = rng
        clamped = min(max(value, lo), hi)
        if clamped != value:
            print(
                "WARNING: magnet %s requested %.4f A outside hard limit [%.3f, %.3f]; "
                "clamped to %.4f A" % (alias, value, lo, hi, clamped)
            )
        return clamped

    def preflight(self) -> Dict[str, Dict[str, Any]]:
        readback: Dict[str, Dict[str, Any]] = {}
        for alias in self.aliases:
            status = self.status(alias)
            acq = self.acquisition(alias)
            status_code = optional_int_like(status.get("status"))
            control = optional_int_like(status.get("control"))
            if status_code == 2:
                raise RuntimeError(
                    "magnet %s %s Status.status=ERROR" % (alias, self.devices[alias])
                )
            if control is not None and control != 5:
                raise RuntimeError(
                    "magnet %s %s is not in REMOTE control (control=%s)"
                    % (alias, self.devices[alias], control)
                )
            if self._busy(acq):
                raise RuntimeError(
                    "magnet %s %s is already busy" % (alias, self.devices[alias])
                )
            readback[alias] = {
                "device": self.devices[alias],
                "status": jsonable(status),
                "currentAverage": self._current_average(alias, acq),
                "current_status": optional_int_like(acq.get("current_status")),
            }
        self.start_currents = {
            alias: float(readback[alias]["currentAverage"])
            for alias in self.aliases
        }
        return readback

    def status(self, alias: str) -> Dict[str, Any]:
        return get_value(self.client, "%s/Status" % self._device(alias))

    def acquisition(self, alias: str) -> Dict[str, Any]:
        return get_value(
            self.client,
            "%s/Acquisition" % self._device(alias),
            context=self.selector,
        )

    def setting_ppm(self, alias: str) -> Dict[str, Any]:
        return get_value(
            self.client,
            "%s/SettingPPM" % self._device(alias),
            context=self.setting_selector,
        )

    def read_currents(self) -> Dict[str, float]:
        return {
            alias: self._current_average(alias, self.acquisition(alias))
            for alias in self.aliases
        }

    def read_sample(
        self,
        aliases: Optional[Iterable[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """One-shot readback of magnets: currentAverage + current_status.

        ``aliases`` selects which magnets to read (default: all configured
        devices). Used by the decoupled magnet-current sample logger. One
        cycle-bound ``Acquisition`` get per selected alias; no writes, no settle,
        no side effects.
        """
        if aliases is None:
            selected = list(self.aliases)
        else:
            selected = [str(alias) for alias in aliases]
            unknown = sorted(set(selected) - set(self.devices))
            if unknown:
                raise RuntimeError(
                    "unknown magnet alias(es): %s" % ", ".join(unknown)
                )
        out: Dict[str, Dict[str, Any]] = {}
        for alias in selected:
            acq = self.acquisition(alias)
            out[alias] = {
                "currentAverage": self._current_average(alias, acq),
                "current_status": optional_int_like(acq.get("current_status")),
            }
        return out

    def apply_currents(self, currents: Mapping[str, float]) -> Dict[str, float]:
        unknown = sorted(set(currents) - set(self.devices))
        if unknown:
            raise RuntimeError("unknown magnet alias(es): %s" % ", ".join(unknown))
        # Clamp to the hard limit BEFORE setting and waiting, so the wait target
        # matches what was actually sent (an unreachable target would never settle).
        targets = {
            alias: self._clamp_to_limit(alias, float(current))
            for alias, current in currents.items()
        }
        for alias, current in targets.items():
            self._set_current(alias, current)
        return self.wait_for_currents(targets)

    def apply_currents_when_idle(self, currents: Mapping[str, float]) -> Dict[str, float]:
        unknown = sorted(set(currents) - set(self.devices))
        if unknown:
            raise RuntimeError("unknown magnet alias(es): %s" % ", ".join(unknown))
        targets = {
            alias: self._clamp_to_limit(alias, float(current))
            for alias, current in currents.items()
        }
        busy = [
            alias
            for alias in targets
            if self._busy(self.acquisition(alias))
        ]
        if busy:
            raise RuntimeError("magnet busy, refusing next write: %s" % ", ".join(busy))
        for alias, current in targets.items():
            self._set_current_fast(alias, current)
        return dict(targets)

    def wait_for_currents(self, targets: Mapping[str, float]) -> Dict[str, float]:
        deadline = time.monotonic() + self.timeout_s
        last_reason = "not checked"
        while time.monotonic() < deadline:
            readback: Dict[str, float] = {}
            busy = []
            off_target = []
            for alias, target in targets.items():
                acq = self.acquisition(alias)
                current = self._current_average(alias, acq)
                readback[alias] = current
                if self._busy(acq):
                    busy.append(alias)
                if (
                    self.settle_mode != "busy_only"
                    and abs(current - float(target)) > self.settle_tolerance_a
                ):
                    off_target.append(alias)
            if not busy and not off_target:
                if self.settle_s:
                    time.sleep(self.settle_s)
                return readback
            last_reason = "busy=%s off_target=%s readback=%s" % (
                busy,
                off_target,
                readback,
            )
            time.sleep(self.poll_s)
        raise RuntimeError(
            "magnet current settle timeout after %.1f s: %s"
            % (self.timeout_s, last_reason)
        )

    def restore_start_currents(self) -> None:
        if self.start_currents:
            self.apply_currents(self.start_currents)

    def stop(self) -> None:
        return None

    def _set_current(self, alias: str, current: float) -> None:
        current = self._clamp_to_limit(alias, current)  # backstop for any direct caller
        setting = self.setting_ppm(alias)
        tol_abs = self._setting_float(alias, setting, "current_tolAbs")
        tol_rel = self._setting_float(alias, setting, "current_tolRel")
        if self.current_tol_abs is not None:
            tol_abs = self.current_tol_abs
        if self.current_tol_rel is not None:
            tol_rel = self.current_tol_rel
        set_value(
            self.client,
            "%s/SettingPPM" % self._device(alias),
            {
                "current": np.float32(current),
                "current_tolAbs": np.float32(tol_abs),
                "current_tolRel": np.float32(tol_rel),
            },
            context=self.setting_selector,
        )

    def _set_current_fast(self, alias: str, current: float) -> None:
        current = self._clamp_to_limit(alias, current)
        payload: Dict[str, Any] = {"current": np.float32(current)}
        if self.current_tol_abs is not None:
            payload["current_tolAbs"] = np.float32(self.current_tol_abs)
        if self.current_tol_rel is not None:
            payload["current_tolRel"] = np.float32(self.current_tol_rel)
        set_value(
            self.client,
            "%s/SettingPPM" % self._device(alias),
            payload,
            context=self.setting_selector,
        )

    def _device(self, alias: str) -> str:
        if alias not in self.devices:
            raise RuntimeError("unknown magnet alias %r" % alias)
        return self.devices[alias]

    def _current_average(self, alias: str, acq: Mapping[str, Any]) -> float:
        if "currentAverage" not in acq:
            raise RuntimeError("magnet %s Acquisition has no currentAverage" % alias)
        return float(plain(acq["currentAverage"]))

    def _setting_float(
        self,
        alias: str,
        setting: Mapping[str, Any],
        key: str,
    ) -> float:
        if key not in setting:
            raise RuntimeError("magnet %s SettingPPM has no %s" % (alias, key))
        return float(plain(setting[key]))

    def _busy(self, acq: Mapping[str, Any]) -> bool:
        status = optional_int_like(acq.get("current_status"))
        if status is None:
            return False
        return bool(status & (1 << self.BUSY_BIT))


class MotorAxis:
    """Small BStepMotorVME relative-move adapter.

    Movement command uses raw steps with FESA Move payload:
        {"value": steps, "mode": 1, "units": 2}

    Physical position checks use:
        target_position = current_position + steps * units_scale

    Completion checks use:
        target_counter = current_counter + steps
    """

    def __init__(
        self,
        name: str,
        client: pyda.SimpleClient,
        cfg: Mapping[str, Any],
        *,
        require_valid_position: bool = True,
    ) -> None:
        self.name = name
        self.client = client
        self.device = str(cfg["device"])
        self.step_raw = int(cfg.get("step_raw", 0))
        self.timeout_s = float(cfg["timeout_s"])
        self.poll_s = float(cfg["poll_s"])
        self.tol_counts = float(cfg["tol_counts"])
        self.settle_s = float(cfg["settle_s"])
        self.require_valid_position = require_valid_position

        expert = get_value(client, "%s/ExpertSetting" % self.device)
        self.units = str(plain(expert.get("units", "")))
        self.units_scale = float(plain(expert["units_scale"]))
        units_offset = plain(expert.get("units_offset", 0.0))
        self.units_offset = 0.0 if units_offset is None else float(units_offset)
        self.position_min = float(plain(expert["position_min"]))
        self.position_max = float(plain(expert["position_max"]))

        self.start_position = 0.0
        self.start_counter = 0.0
        self.net_steps = 0.0

    def acquisition(self) -> Dict[str, Any]:
        return get_value(self.client, "%s/Acquisition" % self.device)

    def driver_acquisition(self) -> Dict[str, Any]:
        return get_value(self.client, "%s/DriverAcquisition" % self.device)

    def preflight(self) -> None:
        acq = self.acquisition()
        driver = self.driver_acquisition()

        if self.require_valid_position and not bool(acq.get("position_valid")):
            raise RuntimeError(
                "%s motor %s position_valid is false; home it first or disable it"
                % (self.name, self.device)
            )
        if bool(acq.get("motor_busy")):
            raise RuntimeError("%s motor %s is already busy" % (self.name, self.device))
        if not bool(driver.get("ps_power_state")):
            raise RuntimeError(
                "%s motor %s driver power is off/faulted" % (self.name, self.device)
            )
        if int(plain(driver.get("ps_error", 0))) != 0:
            raise RuntimeError(
                "%s motor %s driver ps_error=%s"
                % (self.name, self.device, driver.get("ps_error"))
            )

        self.start_position = self._position(acq)
        self.start_counter = self._counter(acq)

    def range_check_relative(self, min_steps: float, max_steps: float) -> None:
        lo = self.start_position + float(min_steps) * self.units_scale
        hi = self.start_position + float(max_steps) * self.units_scale
        span_min = min(lo, hi)
        span_max = max(lo, hi)

        if span_min < self.position_min or span_max > self.position_max:
            raise RuntimeError(
                "%s motor scan span %.6g..%.6g exceeds limits %.6g..%.6g"
                % (self.name, span_min, span_max, self.position_min, self.position_max)
            )

    def range_check_absolute_counters(self, counters: Iterable[float]) -> None:
        positions = [self.position_for_counter(counter) for counter in counters]
        if not positions:
            return

        span_min = min(positions)
        span_max = max(positions)
        if span_min < self.position_min or span_max > self.position_max:
            raise RuntimeError(
                "%s motor absolute span %.6g..%.6g exceeds limits %.6g..%.6g"
                % (self.name, span_min, span_max, self.position_min, self.position_max)
            )

    def counter_for_offset(self, reference_counter: float, offset: float) -> int:
        if self.units_scale == 0:
            raise RuntimeError("%s motor units_scale is zero" % self.name)
        return int(round(float(reference_counter) + float(offset) / self.units_scale))

    def range_check_reference_offsets(
        self,
        reference_counter: float,
        offsets: Iterable[float],
    ) -> None:
        self.range_check_absolute_counters(
            self.counter_for_offset(reference_counter, offset) for offset in offsets
        )

    def position_for_counter(self, counter: float) -> float:
        return float(counter) * self.units_scale - self.units_offset

    def current_counter(self) -> float:
        return self._counter(self.acquisition())

    def require_counter_near(self, target_counter: float, tolerance: float) -> None:
        acq = self.acquisition()
        self._require_ready(acq)
        current = self._counter(acq)
        if abs(current - float(target_counter)) > float(tolerance):
            raise RuntimeError(
                "%s motor counter %.1f is not near reference %.1f +/- %.1f"
                % (self.name, current, float(target_counter), float(tolerance))
            )

    def move_rel(self, steps: float) -> None:
        steps = float(steps)
        if steps == 0:
            return

        acq = self.acquisition()
        self._require_ready(acq)

        pos0 = self._position(acq)
        counter0 = self._counter(acq)

        delta_position = steps * self.units_scale
        target_position = pos0 + delta_position
        target_counter = counter0 + steps

        if target_position < self.position_min or target_position > self.position_max:
            raise RuntimeError(
                "%s motor target %.6g outside limits %.6g..%.6g"
                % (self.name, target_position, self.position_min, self.position_max)
            )

        if self._relevant_limit_active(acq, delta_position):
            direction = "+" if delta_position > 0 else "-"
            raise RuntimeError("%s motor already on %s limit switch" % (self.name, direction))

        set_value(
            self.client,
            "%s/Move" % self.device,
            {
                "value": np.float64(steps),
                "mode": np.int32(1),
                "units": np.int32(2),
            },
        )

        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            acq = self.acquisition()

            if self._relevant_limit_active(acq, delta_position):
                self.stop()
                raise RuntimeError("%s motor hit limit switch; STOP sent" % self.name)

            counter = self._counter(acq)
            done = abs(counter - target_counter) <= self.tol_counts

            if not bool(acq.get("motor_busy")) and done:
                self.net_steps += steps
                if self.settle_s:
                    time.sleep(self.settle_s)
                return

            time.sleep(self.poll_s)

        self.stop()
        raise RuntimeError("%s motor move timeout; STOP sent" % self.name)

    def move_abs_raw(self, target_counter: float) -> None:
        target_counter = int(round(float(target_counter)))

        acq = self.acquisition()
        self._require_ready(acq)

        current_position = self._position(acq)
        current_counter = self._counter(acq)
        if abs(current_counter - target_counter) <= self.tol_counts:
            return

        target_position = self.position_for_counter(target_counter)
        delta_position = target_position - current_position

        if target_position < self.position_min or target_position > self.position_max:
            raise RuntimeError(
                "%s motor absolute target %.6g outside limits %.6g..%.6g"
                % (self.name, target_position, self.position_min, self.position_max)
            )

        if self._relevant_limit_active(acq, delta_position):
            direction = "+" if delta_position > 0 else "-"
            raise RuntimeError("%s motor already on %s limit switch" % (self.name, direction))

        set_value(
            self.client,
            "%s/Move" % self.device,
            {
                "value": np.float64(target_counter),
                "mode": np.int32(2),
                "units": np.int32(2),
            },
        )

        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            acq = self.acquisition()

            if self._relevant_limit_active(acq, delta_position):
                self.stop()
                raise RuntimeError("%s motor hit limit switch; STOP sent" % self.name)

            counter = self._counter(acq)
            done = abs(counter - target_counter) <= self.tol_counts

            if not bool(acq.get("motor_busy")) and done:
                self.net_steps = target_counter - self.start_counter
                if self.settle_s:
                    time.sleep(self.settle_s)
                return

            time.sleep(self.poll_s)

        self.stop()
        raise RuntimeError("%s motor absolute move timeout; STOP sent" % self.name)

    def move_to_offset(self, reference_counter: float, offset: float) -> None:
        self.move_abs_raw(self.counter_for_offset(reference_counter, offset))

    def stop(self) -> None:
        try:
            set_value(self.client, "%s/Stop" % self.device, {"stop_all": False})
        except BaseException as exc:
            print("WARNING: %s motor Stop failed: %s" % (self.name, exc))

    def _require_ready(self, acq: Mapping[str, Any]) -> None:
        if self.require_valid_position and not bool(acq.get("position_valid")):
            raise RuntimeError("%s motor %s position is invalid" % (self.name, self.device))
        if bool(acq.get("motor_busy")):
            raise RuntimeError("%s motor %s is already moving" % (self.name, self.device))

    def _position(self, acq: Mapping[str, Any]) -> float:
        if "position" not in acq:
            raise RuntimeError("%s motor Acquisition has no position field" % self.name)
        return float(plain(acq["position"]))

    def _counter(self, acq: Mapping[str, Any]) -> float:
        for key in ("position_counter", "positionCounter"):
            if key in acq:
                return float(plain(acq[key]))
        raise RuntimeError("%s motor Acquisition has no position_counter field" % self.name)

    def _relevant_limit_active(self, acq: Mapping[str, Any], delta_position: float) -> bool:
        if delta_position > 0:
            return bool(acq.get("end_switch_plus_active"))
        if delta_position < 0:
            return bool(acq.get("end_switch_minus_active"))
        return False
