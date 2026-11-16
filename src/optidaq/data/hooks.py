"""Optional image and metadata hooks for dataset export.

Image hooks return transformed pixels. Metadata hooks return a mapping merged
into the database row and must not mutate the image. Both receive the image,
context and configured keyword parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from optidaq.data.transforms import avg_bin, center_crop, crop_box, normalize, to_uint8


ImageHook = Callable[[np.ndarray, "HookContext"], np.ndarray]
MetadataHook = Callable[[np.ndarray, "HookContext"], Mapping[str, Any]]


@dataclass(frozen=True)
class HookContext:
    run_id: str
    camera: str
    image_id: str
    frame_index: int
    acq_stamp_ns: Optional[int]
    source_image_number: Optional[int]
    pulse_id: Optional[int]
    sync: Optional[Mapping[str, Any]]
    raw_header: Mapping[str, int]
    raw_segment_path: str
    byte_offset: int
    # Original full-resolution frame, before any image hooks (crop/bin). Lets
    # metadata hooks measure the source even when the stored image was transformed.
    source_image: Optional[np.ndarray] = None
    # The exact crop box (x1, y1, x2, y2) this camera's crop_box applied, in source
    # pixel coords (None if no crop). Single source of truth for crop_energy.
    applied_crop: Optional[tuple] = None


def build_image_hooks(hook_specs: Optional[Sequence[Any]]) -> list[ImageHook]:
    return [
        _make_hook(spec, _IMAGE_HOOKS, "image")
        for spec in (hook_specs or [])
    ]


def build_metadata_hooks(hook_specs: Optional[Sequence[Any]]) -> list[MetadataHook]:
    return [
        _make_hook(spec, _METADATA_HOOKS, "metadata")
        for spec in (hook_specs or [])
    ]


def stats_hook(image: np.ndarray, context: HookContext) -> Mapping[str, Any]:
    """Return stored-image intensity stats, including P99.9 alongside true max.

    P99.9 is less sensitive to isolated hot pixels, but may miss small bright
    features. It is saved in hook metadata; neither statistic changes the pixels.
    """

    arr = np.asarray(image)
    if arr.size == 0:
        return {
            "stat_min": None,
            "stat_max": None,
            "stat_p99_9": None,
            "stat_mean": None,
            "stat_sum": None,
        }
    values64 = arr.astype(np.float64, copy=False)
    return {
        "stat_min": float(values64.min()),
        "stat_max": float(values64.max()),
        "stat_p99_9": float(np.percentile(values64, 99.9)),
        "stat_mean": float(values64.mean()),
        "stat_sum": float(values64.sum()),
    }


def centroid_hook(image: np.ndarray, context: HookContext) -> Mapping[str, Any]:
    """Return intensity-weighted centroid in zero-based pixel coordinates."""

    values = np.asarray(image, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        return {"centroid_x": None, "centroid_y": None}

    total = float(values.sum())
    if not np.isfinite(total) or total == 0.0:
        return {"centroid_x": None, "centroid_y": None}

    height, width = values.shape
    x_profile = values.sum(axis=0)
    y_profile = values.sum(axis=1)
    return {
        "centroid_x": float(np.dot(x_profile, np.arange(width)) / total),
        "centroid_y": float(np.dot(y_profile, np.arange(height)) / total),
    }


def crop_energy_hook(image: np.ndarray, context: HookContext) -> Mapping[str, Any]:
    """Total pixel energy inside vs outside this camera's ACTUAL crop ROI.

    The ROI is ``context.applied_crop`` — the exact box the camera's ``crop_box``
    hook used — so the numbers always match the real per-camera crop (no separate
    ``crop_size`` to drift). Computed on the full-res source frame. If no crop was
    applied, the whole image is the ROI (nothing cropped out).
    """

    source = context.source_image if context.source_image is not None else image
    arr = np.asarray(source, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return {
            "intensity_in_roi": None,
            "intensity_cropped_out": None,
            "roi_energy_fraction": None,
        }
    total = float(arr.sum())
    box = context.applied_crop
    if box is None:
        inside = total  # no crop -> whole image is the ROI
    else:
        height, width = arr.shape
        xa, xb = sorted((int(round(box[0])), int(round(box[2]))))
        ya, yb = sorted((int(round(box[1])), int(round(box[3]))))
        xa, xb = max(0, xa), min(width, xb)
        ya, yb = max(0, ya), min(height, yb)
        inside = float(arr[ya:yb, xa:xb].sum())
    outside = total - inside
    return {
        "intensity_in_roi": inside,
        "intensity_cropped_out": outside,
        "roi_energy_fraction": (inside / total) if total > 0 else None,
    }


# Image hooks mutate pixels (chained: output -> input of next). Defaults are empty
# in the lossless export; the crop/bin/normalize transforms are available here for
# derived / runtime pipelines via the same contract.
_IMAGE_HOOKS: dict[str, Callable[..., Any]] = {
    "center_crop": center_crop,
    "crop_box": crop_box,
    "avg_bin": avg_bin,
    "normalize": normalize,
    "to_uint8": to_uint8,
}
# Metadata hooks are read-only and return a dict merged into the DB row.
_METADATA_HOOKS: dict[str, Callable[..., Any]] = {
    "stats": stats_hook,
    "centroid": centroid_hook,
    "crop_energy": crop_energy_hook,
}


def _make_hook(
    spec: Any,
    registry: Mapping[str, Callable[..., Any]],
    kind: str,
) -> Callable[..., Any]:
    if isinstance(spec, str):
        name = spec
        params: Mapping[str, Any] = {}
    elif isinstance(spec, Mapping):
        name = str(spec.get("name", ""))
        params = spec.get("params", {}) or {}
    else:
        raise ValueError("%s hook spec must be a string or object" % kind)

    if not name:
        raise ValueError("%s hook spec is missing a name" % kind)
    if not isinstance(params, Mapping):
        raise ValueError("%s hook params must be an object: %s" % (kind, name))

    fn = registry.get(name)
    if fn is None:
        fn = _load_external_hook(name)

    def wrapped(image: np.ndarray, context: HookContext) -> Any:
        return fn(image, context, **dict(params))

    return wrapped


def _load_external_hook(name: str) -> Callable[..., Any]:
    """Load hooks named as ``module.path:function_name``."""

    if ":" not in name:
        raise ValueError(
            "unknown hook %r; use a built-in name or module.path:function" % name
        )
    module_name, function_name = name.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, function_name)
    if not callable(fn):
        raise ValueError("hook is not callable: %s" % name)
    return fn
