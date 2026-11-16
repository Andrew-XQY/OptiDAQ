"""Lab-only camera receipt metadata and passwordless CERN authentication.

The shared beam-time adapters remain unchanged. Imports of the CERN SDK are
delayed so configuration and offline capture tests do not need that SDK.
"""

from __future__ import annotations

from copy import deepcopy
import gc
import importlib
import math
import time
import traceback
from typing import Any, Mapping


def _adapter():
    return importlib.import_module("optidaq.extensions.cern_pyda")


def _cached_token(adapter):
    token = adapter._load_cached_token()
    if token is None:
        path = adapter._token_cache_path()
        try:
            token = adapter._token_from_bytes(path.read_bytes())
        except OSError:
            return None
        # The installed SDK exposes datetime expiry, which the older shared
        # helper cannot read. Its authoritative expired flag is usable here.
        try:
            expired = getattr(token, "is_expired", None)
            expired = expired() if callable(expired) else expired
        except Exception:
            return None
        return token if expired is False else None
    try:
        expired = getattr(token, "is_expired", None)
        expired = expired() if callable(expired) else expired
    except Exception:
        return None
    return None if expired is True else token


def make_lab_client(require_rbac: bool = True):
    """Use an existing token, Kerberos ticket or location login; never prompt."""
    adapter = _adapter()
    provider = adapter.RdaProvider()
    client = adapter.pyda.SimpleClient(provider=provider)
    if not require_rbac:
        return client
    token = _cached_token(adapter)
    reused = token is not None
    if token is None:
        factory = getattr(adapter.pyrbac.LoginService, "create_for_kerberos", None)
        if factory is not None:
            listener = adapter._RbacListener(lambda value: setattr(provider, "rbac_token", value))
            try:
                service = factory(listener)
                if adapter._await_login(listener, required=False):
                    token = listener.token
                    client._optidaq_rbac_service = service
                    client._optidaq_rbac_listener = listener
            except Exception:
                token = None
    if token is None:
        token = adapter._login_location(provider, client)
    if token is None:
        raise RuntimeError("Lab RBAC login failed; refresh your CERN login/ticket before acquisition")
    provider.rbac_token = token
    if not reused:
        adapter._save_token(token)
    return client


def read_camera_settings(client, camera_map: Mapping[str, str]) -> dict:
    """Read requested/applied camera settings and external-trigger selection."""
    adapter = _adapter()
    return {
        alias: {
            prop: adapter.jsonable(adapter.get_value(client, "%s/%s" % (device, prop)))
            for prop in ("CameraSetting", "AcquiredParameters", "TriggerConfiguration")
        }
        for alias, device in camera_map.items()
    }


def verify_external_settings(before: Mapping[str, Any], after=None) -> None:
    """Require the tested linear Mono12/Line1 mode, then detect settings changes."""
    if not before:
        raise RuntimeError("No camera settings were read")
    for camera, snapshot in before.items():
        requested = snapshot["CameraSetting"]
        applied = snapshot["AcquiredParameters"]
        for requested_key, applied_key, expected in (
            ("acqTrigger", "acqAcqTrigger", 2),
            ("cameraPixelFormat", "acqCameraPixelFormat", 40),
            ("cameraExposureAuto", "acqCameraExposureAuto", 1),
            ("cameraGainAuto", "acqCameraGainAuto", 1),
            ("cameraGammaEnable", "acqCameraGammaEnable", False),
        ):
            if requested.get(requested_key) != expected or applied.get(applied_key) != expected:
                raise RuntimeError("%s requires %s=%r in requested and applied settings" %
                                   (camera, requested_key, expected))
        for requested_key, applied_key in (
            ("cameraExposureTimeUs", "acqCameraExposureTimeUs"),
            ("cameraGain", "acqCameraGain"),
            ("cameraTriggerDelay", "acqCameraTriggerDelay"),
        ):
            try:
                desired = float(requested[requested_key])
                actual = float(applied[applied_key])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("%s missing/invalid %s readback" % (camera, requested_key)) from exc
            if not math.isfinite(desired) or not math.isfinite(actual) or not math.isclose(desired, actual, rel_tol=0, abs_tol=1e-4):
                raise RuntimeError("%s %s differs from its applied readback" % (camera, requested_key))
        if float(applied["acqCameraExposureTimeUs"]) <= 0 or float(applied["acqCameraGain"]) < 0:
            raise RuntimeError("%s has invalid exposure/gain" % camera)
    if after is not None:
        verify_external_settings(after)
        if before != after:
            raise RuntimeError("Camera settings changed during the lab run; timing baselines are invalid")


def _lab_tap_class(adapter):
    class LabSubscribingCameraTap(adapter.SubscribingCameraTap):
        """Reuse frame decoding with monotonic receipt times and bounded waits."""

        def _build_and_emit(self, data, emit):
            import numpy as np
            image = np.asarray(data["image2D"])
            if tuple(image.shape) != tuple(self.image_shape) or str(image.dtype) != self.image_dtype:
                raise RuntimeError("%s camera shape/dtype changed during acquisition" % self.device)
            super()._build_and_emit(data, emit)

        def _emit_subscription_response(self, response, emit):
            receipt_ns, receipt_mono_ns = time.time_ns(), time.monotonic_ns()

            def enriched(frame):
                import numpy as np
                metadata = deepcopy(frame.sync_metadata)
                trigger = adapter.require_int_like(response.data.get("triggerMode"), "triggerMode")
                if trigger != 3:
                    raise RuntimeError("%s frame is not externally triggered on Line1" % self.device)
                image = np.frombuffer(frame.payload_bytes, dtype=self.image_dtype)
                minimum, maximum = int(image.min()), int(image.max())
                if minimum < 0 or maximum > 4095:
                    raise RuntimeError("%s Mono12 frame contains counts outside 0..4095" % self.device)
                header = getattr(response, "header", None)
                metadata.update(
                    local_receive_ns=receipt_ns,
                    local_receive_monotonic_ns=receipt_mono_ns,
                    triggerMode=trigger,
                    image_min=minimum,
                    image_max=maximum,
                    saturated_pixels=int(np.count_nonzero(image >= 4095)),
                    header={name: adapter.jsonable(getattr(header, name))
                            for name in ("acquisition_timestamp", "cycle_timestamp", "selector")
                            if hasattr(header, name)},
                )
                object.__setattr__(frame, "sync_metadata", metadata)
                emit(frame)

            super()._emit_subscription_response(response, enriched)

        def _run_iterator_subscription(self, endpoint, emit):
            kwargs = {"receive_first_updates": False}
            if self._selector:
                kwargs["context"] = self._selector
            subscription = self._client.subscribe(endpoint, **kwargs)
            self._subscription = subscription
            self._start_subscription_if_needed(subscription)
            iterator = iter(subscription)
            response = next_response = None
            try:
                next_response = getattr(iterator, "next", None)
                if not callable(next_response):
                    raise RuntimeError("Lab subscription requires the PyDA iterator's timed next() API")
                while not self._stop_event.is_set():
                    try:
                        response = next_response(timeout=0.25)
                    except Exception as exc:
                        if type(exc).__name__ == "SubscriptionTimeoutError":
                            continue
                        if self._stop_event.is_set():
                            break
                        raise
                    if not self._stop_event.is_set():
                        self._emit_subscription_response(response, emit)
            except BaseException as exc:
                # A failed SDK next() traceback can own its iterator/monitor.
                # Preserve the exception and stack locations, release locals
                # in completed frames just as the shared startup probe does.
                current, seen = exc, set()
                while current is not None and id(current) not in seen:
                    seen.add(id(current))
                    traceback.clear_frames(current.__traceback__)
                    current = current.__cause__ or current.__context__
                raise
            finally:
                response = iterator = next_response = subscription = None
                gc.collect()

    return LabSubscribingCameraTap


def build_lab_taps(client, cfg: Mapping[str, Any]) -> list:
    """Build unprobed camera taps; the caller owns probe/start/stop/join."""
    adapter = _adapter()
    tap_cfg = cfg["tap"]
    if tap_cfg.get("mode", "subscribe") != "subscribe":
        raise ValueError("Coordinated lab acquisition requires subscription camera taps")
    if tap_cfg.get("subscribe_selector") != "SPS.USER.ALL":
        raise ValueError("Coordinated lab acquisition requires tap.subscribe_selector=SPS.USER.ALL")
    tap_class = _lab_tap_class(adapter)
    return [tap_class(
        alias, device, client,
        poll_s=float(tap_cfg.get("last_image_poll_s", 0.02)),
        stamp_fields=tap_cfg.get("stamp_fields", ["imageTimeStamp"]),
        min_ns_epoch=int(tap_cfg.get("min_ns_epoch", 100_000_000_000_000_000)),
        pulse_id_fields=(),
        settle_poll_s=float(tap_cfg.get("settle_poll_s", 0.005)),
        settle_timeout_s=float(tap_cfg.get("settle_timeout_s", 0.25)),
        max_trailing_zero_rows=tap_cfg.get("max_trailing_zero_rows", 0),
        sync_cfg={"priority": ["image_ready_ns"]},
        allow_image_number_reset=False,
        get_selector=tap_cfg.get("get_selector"),
        selector=tap_cfg["subscribe_selector"],
    ) for alias, device in adapter.camera_map_from_config(cfg).items()]
