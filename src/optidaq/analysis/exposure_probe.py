"""Pure numerical checks for a stationary, independently exposed camera probe.

Inputs are native frames, before any crop/bin/export operation. Recommendations
apply only to the supplied scenes; this is neither photon-SNR calibration nor
verification of optical superposition or camera exposure synchrony.
"""
from collections import defaultdict
import math

import numpy as np


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + " must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be a finite positive number")
    return float(value)


def _frames(value, name):
    if value is None:
        raise ValueError(name + " is required")
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3 or not all(array.shape):
        raise ValueError(name + " must contain nonempty native 2D frames")
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(name + " contains nonfinite or negative raw counts")
    return array


def summarize_exposure(camera, exposure_us, signal_frames, background_before_frames,
                       background_after_frames, *, full_scale=4095, target_low=.70,
                       target_high=.85, roi=None, min_contrast_to_noise=10.0,
                       max_drift_fraction=.10, min_corrected_contrast_fraction=.20,
                       max_signal_variation_fraction=.10):
    """Summarize one camera/exposure/held scene; caller adds its ``scene`` ID.

    ``roi`` is optional (x1, y1, x2, y2), half-open native pixel coordinates.
    It restricts signal/noise measurements, never saturation checks. Backgrounds
    must have the same exposure/settings as the signal frames (caller verifies
    readbacks); they bracket the held-scene acquisition. Negative background-
    subtracted values are retained for integrated-signal/linearity measurements.
    """
    if not isinstance(camera, str) or not camera:
        raise ValueError("camera must be a nonempty string")
    exposure_us = _positive(exposure_us, "exposure_us")
    full_scale = _positive(full_scale, "full_scale")
    min_contrast_to_noise = _positive(min_contrast_to_noise, "min_contrast_to_noise")
    max_drift_fraction = _positive(max_drift_fraction, "max_drift_fraction")
    min_corrected_contrast_fraction = _positive(min_corrected_contrast_fraction, "min_corrected_contrast_fraction")
    max_signal_variation_fraction = _positive(max_signal_variation_fraction, "max_signal_variation_fraction")
    if min_corrected_contrast_fraction >= 1:
        raise ValueError("min_corrected_contrast_fraction must be less than 1")
    if not 0 < target_low < target_high < 1:
        raise ValueError("require 0 < target_low < target_high < 1")
    signal = _frames(signal_frames, "signal_frames")
    before = _frames(background_before_frames, "background_before_frames")
    after = _frames(background_after_frames, "background_after_frames")
    shape = signal.shape[1:]
    if before.shape[1:] != shape or after.shape[1:] != shape:
        raise ValueError("signal and both backgrounds must have identical native shapes")
    if roi is None:
        roi = (0, 0, shape[1], shape[0])
    if (len(roi) != 4 or any(type(v) is not int for v in roi)
            or not 0 <= roi[0] < roi[2] <= shape[1]
            or not 0 <= roi[1] < roi[3] <= shape[0]):
        raise ValueError("roi must be integer (x1,y1,x2,y2) within the native frame")
    region = (slice(roi[1], roi[3]), slice(roi[0], roi[2]))
    before_map, after_map = np.median(before, axis=0), np.median(after, axis=0)
    corrected = signal.mean(axis=0) - (before_map + after_map) / 2
    useful = corrected[region]
    contrast = float(np.percentile(useful, 99.9))
    drift = (after_map - before_map)[region]
    drift_rms = float(np.sqrt(np.mean(drift ** 2)))
    repeated = all(len(frames) >= 2 for frames in (signal, before, after))
    noise = None
    signal_noise = None
    if repeated:
        variance = (before.var(axis=0, ddof=1) + after.var(axis=0, ddof=1)) / 2
        noise = float(np.sqrt(np.mean(variance[region])))
        signal_noise = float(np.sqrt(np.mean(signal.var(axis=0, ddof=1)[region])))
    ratio = contrast / noise if noise is not None and noise > 0 else None
    drift_fraction = drift_rms / contrast if contrast > 0 else None
    signal_variation = signal_noise / contrast if signal_noise is not None and contrast > 0 else None
    signal_max = float(signal.max())
    background_max = float(max(before.max(), after.max()))
    native_max = max(signal_max, background_max)
    clipped = {name: int(np.count_nonzero(frames >= full_scale))
               for name, frames in (("signal", signal), ("background_before", before),
                                    ("background_after", after))}
    flags = []
    if sum(clipped.values()):
        flags.append("native_clipping")
    if native_max > target_high * full_scale:
        flags.append("above_headroom_limit")
    if not repeated:
        flags.append("insufficient_repeats")
    elif noise == 0:
        flags.append("zero_repeat_noise_estimate")
    if contrast <= 0 or float(useful.sum()) <= 0 or ratio is not None and ratio < min_contrast_to_noise:
        flags.append("weak_corrected_signal")
    if contrast < min_corrected_contrast_fraction * full_scale:
        flags.append("below_useful_signal_target")
    if drift_fraction is None or drift_fraction > max_drift_fraction:
        flags.append("background_drift")
    if signal_variation is not None and signal_variation > max_signal_variation_fraction:
        flags.append("unstable_signal")
    return {
        "camera": camera, "exposure_us": exposure_us, "native_shape": list(shape),
        "roi_xyxy": list(roi), "full_scale_counts": full_scale,
        "target_low": float(target_low), "target_high": float(target_high),
        "min_contrast_to_noise": min_contrast_to_noise,
        "max_drift_fraction": max_drift_fraction,
        "min_corrected_contrast_fraction": min_corrected_contrast_fraction,
        "max_signal_variation_fraction": max_signal_variation_fraction,
        "frame_counts": {"signal": len(signal), "background_before": len(before),
                         "background_after": len(after)},
        "native_max_counts": native_max, "signal_native_max_counts": signal_max,
        "background_native_max_counts": background_max,
        "native_peak_fraction": native_max / full_scale,
        "native_saturated_pixels": clipped,
        "native_saturated_pixels_total": sum(clipped.values()),
        "in_target_band": bool(target_low <= signal_max / full_scale <= target_high
                               and native_max <= target_high * full_scale),
        "corrected_signal_p99_9_counts": contrast,
        "corrected_contrast_fraction": contrast / full_scale,
        "corrected_signal_mean_counts": float(useful.mean()),
        "corrected_signal_sum_counts": float(useful.sum()),
        "signal_rate_mean_counts_per_s": float(useful.mean() * 1e6 / exposure_us),
        "repeat_background_noise_rms_counts": noise,
        "repeat_signal_noise_rms_counts": signal_noise,
        "signal_variation_fraction": signal_variation,
        "contrast_over_repeat_background_noise": ratio,
        "background_drift_mean_counts": float(drift.mean()),
        "background_drift_rms_counts": drift_rms,
        "background_drift_fraction_of_signal": drift_fraction,
        "flags": flags,
    }


def recommend_exposures(points_by_camera, *, fibre_camera="cam1", reference_camera="cam2",
                        linearity_tolerance=.10):
    """Select independently, requiring every supplied scene to pass the checks.

    Each point comes from :func:`summarize_exposure` with a ``scene`` string added.
    Linearity compares signed integrated ROI signal/exposure across at least
    three distinct usable exposures per scene. A maximum relative deviation
    from their median greater than ``linearity_tolerance`` prevents a recommendation.
    There is no interpolation or extrapolation to untested exposure settings.
    """
    linearity_tolerance = _positive(linearity_tolerance, "linearity_tolerance")
    if not points_by_camera:
        raise ValueError("points_by_camera must not be empty")
    cameras = {}
    for camera, points in points_by_camera.items():
        if not points:
            raise ValueError("each camera needs exposure points")
        by_scene, by_exposure = defaultdict(dict), defaultdict(dict)
        for point in points:
            if point["camera"] != camera:
                raise ValueError("point camera disagrees with its mapping key")
            scene = point.get("scene", "default")
            if not isinstance(scene, str) or not scene:
                raise ValueError("scene must be a nonempty string")
            exposure = _positive(point["exposure_us"], "exposure_us")
            if exposure in by_scene[scene]:
                raise ValueError("duplicate camera/scene/exposure point")
            by_scene[scene][exposure] = point
            by_exposure[exposure][scene] = point
        scenes = sorted(by_scene)
        linearity = {}
        for scene, scene_points in sorted(by_scene.items()):
            identities = {tuple(p["native_shape"]) + tuple(p["roi_xyxy"]) + (p["full_scale_counts"],)
                          for p in scene_points.values()}
            if len(identities) != 1:
                raise ValueError("native shape, ROI and full scale must stay fixed within each scene")
            usable = [p for p in scene_points.values()
                      if not set(p["flags"]) - {"above_headroom_limit", "below_useful_signal_target"}]
            rates = [p["signal_rate_mean_counts_per_s"] for p in usable]
            deviation = None
            if len(rates) >= 3 and min(rates) > 0:
                median = float(np.median(rates))
                deviation = max(abs(rate / median - 1) for rate in rates)
            linearity[scene] = {
                "status": ("insufficient_candidates" if deviation is None else
                           "within_tolerance" if deviation <= linearity_tolerance else "nonlinear"),
                "usable_exposures_us": sorted(p["exposure_us"] for p in usable),
                "max_relative_rate_deviation": deviation,
                "relative_tolerance": linearity_tolerance,
            }
        candidates = []
        for exposure, members in sorted(by_exposure.items()):
            missing = sorted(set(scenes) - set(members))
            reasons = sorted({flag for p in members.values() for flag in p["flags"]})
            if missing:
                reasons.append("missing_scenes")
            peak = max(p["native_peak_fraction"] for p in members.values())
            target_low = max(p["target_low"] for p in members.values())
            target_high = min(p["target_high"] for p in members.values())
            signal_peak = max(p["signal_native_max_counts"] / p["full_scale_counts"]
                              for p in members.values())
            candidates.append({
                "exposure_us": exposure, "worst_native_peak_fraction": peak,
                "missing_scenes": missing, "flags": reasons,
                "in_target_band": bool(target_low <= signal_peak <= target_high and peak <= target_high),
                "within_headroom": bool(peak <= target_high),
            })
        unsaturated = [c for c in candidates if not c["missing_scenes"] and "native_clipping" not in c["flags"]]
        with_headroom = [c for c in unsaturated if c["within_headroom"]]
        provisional = (max(with_headroom, key=lambda c: c["exposure_us"]) if with_headroom else
                       min(unsaturated, key=lambda c: c["exposure_us"]) if unsaturated else None)
        eligible = [c for c in candidates if not c["flags"] and c["in_target_band"]]
        flags = []
        for scene, check in linearity.items():
            if check["status"] != "within_tolerance":
                flags.append("linearity_" + check["status"] + ":" + scene)
        if not eligible:
            flags.append("no_valid_candidate_in_target_band")
        selected = max(eligible, key=lambda c: c["exposure_us"]) if eligible and not flags else None
        cameras[camera] = {
            "status": "recommended_for_tested_scenes" if selected else "review_required",
            "recommended_exposure_us": selected["exposure_us"] if selected else None,
            "best_unsaturated_provisional_exposure_us": provisional["exposure_us"] if provisional else None,
            "provisional_flags": (sorted(set(provisional["flags"] + flags)) if provisional else
                                  ["no_unsaturated_candidate_for_all_scenes"]),
            "flags": flags, "scenes": scenes, "linearity": linearity, "candidates": candidates,
        }
    chosen = {camera: value["recommended_exposure_us"] for camera, value in cameras.items()}
    ratio = (chosen[fibre_camera] / chosen[reference_camera]
             if chosen.get(fibre_camera) is not None and chosen.get(reference_camera) is not None else None)
    return {
        "status": "recommended_for_tested_scenes" if all(chosen.values()) else "review_required",
        "cameras": cameras,
        "normalization": {
            "formula": "(raw_counts - matched_background_counts) * (1000000 / exposure_us)",
            "units": "DN/s; not calibrated photons or optical power",
            "gain_assumption": "gain fixed at 0, with otherwise unchanged linear camera response",
            "recommended_factors_to_counts_per_s": {
                camera: 1e6 / exposure if exposure is not None else None
                for camera, exposure in chosen.items()},
            "fibre_camera": fibre_camera, "reference_camera": reference_camera,
            "fibre_to_reference_exposure_ratio": ratio,
            "exposure_correction_applied_to_raw_data": False,
        },
        "limitations": [
            "Applies only to tested held scenes and current fixed camera settings; later brighter scenes can clip.",
            "Corrected brightness, contrast/noise, drift and repeatability limits are screening heuristics, not calibrated photon SNR.",
            "Linearity checks integrated ROI signal only; source drift can also change the measured rate.",
            "Optical superposition, cross-source calibration transfer and simultaneous exposures are not verified.",
            "Exposure normalization cannot repair saturation, background drift or noise differences.",
        ],
    }
