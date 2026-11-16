"""Exposure selection must reject clipped, unstable or uncalibrated candidates."""
import json

import numpy as np
import pytest

from optidaq.analysis.exposure_probe import recommend_exposures, summarize_exposure


def frames(level, shape=(12, 12)):
    return np.stack([np.full(shape, level + delta) for delta in (-1., 0., 1.)])


def point(camera, exposure, rate=10., scene="held", drift=0., roi=None):
    result = summarize_exposure(camera, exposure, frames(100 + rate * exposure),
                                frames(100), frames(100 + drift), roi=roi)
    result["scene"] = scene
    return result


def test_independent_exposure_selection_and_explicit_normalization():
    report = recommend_exposures({
        "cam1": [point("cam1", exposure) for exposure in (100, 200, 300)],
        "cam2": [point("cam2", exposure, rate=100.) for exposure in (10, 20, 30)],
    })
    assert report["cameras"]["cam1"]["recommended_exposure_us"] == 300
    assert report["cameras"]["cam2"]["recommended_exposure_us"] == 30
    assert report["normalization"]["fibre_to_reference_exposure_ratio"] == 10
    assert report["normalization"]["recommended_factors_to_counts_per_s"]["cam1"] == pytest.approx(1e6 / 300)
    json.dumps(report, allow_nan=False)


def test_native_clipping_outside_useful_roi_is_never_hidden():
    signal = frames(3100)
    signal[0, -1, -1] = 4095
    result = summarize_exposure("cam1", 300, signal, frames(100), frames(100), roi=(0, 0, 3, 3))
    result["scene"] = "held"
    assert result["native_saturated_pixels_total"] == 1
    assert "native_clipping" in result["flags"]
    report = recommend_exposures({"cam1": [point("cam1", 100, roi=(0, 0, 3, 3)),
                                         point("cam1", 200, roi=(0, 0, 3, 3)), result]})
    assert report["cameras"]["cam1"]["recommended_exposure_us"] is None
    assert report["cameras"]["cam1"]["best_unsaturated_provisional_exposure_us"] == 200


def test_bright_background_does_not_masquerade_as_good_signal():
    result = summarize_exposure("cam1", 100, frames(3102), frames(3100), frames(3100))
    assert result["in_target_band"]
    assert result["contrast_over_repeat_background_noise"] == pytest.approx(2.)
    assert "weak_corrected_signal" in result["flags"]
    report = recommend_exposures({"cam1": [result]})
    assert report["cameras"]["cam1"]["recommended_exposure_us"] is None
    assert "weak_corrected_signal" in report["cameras"]["cam1"]["provisional_flags"]


def test_drift_and_nonlinearity_each_prevent_recommendation():
    drifted = [point("cam1", exposure, drift=800.) for exposure in (100, 200, 300)]
    report = recommend_exposures({"cam1": drifted})
    assert report["cameras"]["cam1"]["recommended_exposure_us"] is None
    assert "background_drift" in report["cameras"]["cam1"]["provisional_flags"]
    nonlinear = [point("cam1", 100), point("cam1", 200, rate=7.), point("cam1", 300)]
    report = recommend_exposures({"cam1": nonlinear})
    result = report["cameras"]["cam1"]
    assert result["linearity"]["held"]["status"] == "nonlinear"
    assert result["recommended_exposure_us"] is None
    assert result["best_unsaturated_provisional_exposure_us"] == 300


def test_three_distinct_exposures_required_for_each_scene():
    report = recommend_exposures({"cam1": [point("cam1", 200), point("cam1", 300)]})
    result = report["cameras"]["cam1"]
    assert result["recommended_exposure_us"] is None
    assert result["best_unsaturated_provisional_exposure_us"] == 300
    assert result["linearity"]["held"]["status"] == "insufficient_candidates"


def test_multiple_scenes_require_shared_safe_exposure_and_all_scene_linearity():
    points = [point("cam1", exposure, rate=rate, scene=scene)
              for scene, rate in (("bright", 10.), ("dim", 7.))
              for exposure in (100, 200, 300)]
    report = recommend_exposures({"cam1": points})
    assert report["cameras"]["cam1"]["recommended_exposure_us"] == 300
    points[-1] = point("cam1", 300, rate=14., scene="dim")
    report = recommend_exposures({"cam1": points})
    assert report["cameras"]["cam1"]["recommended_exposure_us"] is None
    assert report["cameras"]["cam1"]["best_unsaturated_provisional_exposure_us"] == 200


def test_background_saturation_and_insufficient_repeats_are_reported():
    background = frames(100)
    background[0, 0, 0] = 4095
    result = summarize_exposure("cam1", 300, frames(3100), background, frames(100))
    assert "native_clipping" in result["flags"]
    single = summarize_exposure("cam1", 300, frames(3100)[:1], frames(100), frames(100))
    assert "insufficient_repeats" in single["flags"]
    assert single["contrast_over_repeat_background_noise"] is None
    json.dumps(single, allow_nan=False)


def test_missing_background_invalid_roi_and_duplicate_points_fail_explicitly():
    with pytest.raises(ValueError, match="required"):
        summarize_exposure("cam1", 100, frames(1000), None, frames(100))
    with pytest.raises(ValueError, match="roi"):
        summarize_exposure("cam1", 100, frames(1000), frames(100), frames(100), roi=(0, 0, 99, 99))
    with pytest.raises(ValueError, match="duplicate"):
        recommend_exposures({"cam1": [point("cam1", 100), point("cam1", 100)]})


def test_linearity_uses_subtracted_rates_not_bias_inflated_raw_counts():
    points = [point("cam1", exposure) for exposure in (100, 200, 300)]
    assert len({p["signal_rate_mean_counts_per_s"] for p in points}) == 1
    assert points[0]["native_max_counts"] / 100 != points[-1]["native_max_counts"] / 300
    report = recommend_exposures({"cam1": points})
    assert report["cameras"]["cam1"]["linearity"]["held"]["max_relative_rate_deviation"] == 0


def test_all_clipped_returns_no_provisional_and_no_recommendation():
    report = recommend_exposures({"cam1": [point("cam1", exposure) for exposure in (500, 600, 700)]})
    assert report["cameras"]["cam1"]["recommended_exposure_us"] is None
    assert report["cameras"]["cam1"]["best_unsaturated_provisional_exposure_us"] is None


def test_background_dominated_linear_high_snr_data_is_not_recommended():
    points = []
    for exposure in (100, 200, 300):
        result = summarize_exposure("cam1", exposure, frames(3100 + .2 * exposure),
                                    frames(3100), frames(3100))
        assert result["in_target_band"]
        assert result["contrast_over_repeat_background_noise"] >= 20
        assert "below_useful_signal_target" in result["flags"]
        points.append(result)
    report = recommend_exposures({"cam1": points})["cameras"]["cam1"]
    assert report["linearity"]["default"]["status"] == "within_tolerance"
    assert report["recommended_exposure_us"] is None
    assert "below_useful_signal_target" in report["provisional_flags"]


def test_unstable_signal_repeats_block_recommendation_and_linearity():
    points = [point("cam1", 100), point("cam1", 200)]
    signal = np.stack([np.full((12, 12), value) for value in (2400., 2800., 3200.)])
    result = summarize_exposure("cam1", 300, signal, frames(100), frames(100))
    result["scene"] = "held"
    assert result["in_target_band"]
    assert result["signal_variation_fraction"] == pytest.approx(400 / 2700)
    assert "unstable_signal" in result["flags"]
    report = recommend_exposures({"cam1": points + [result]})["cameras"]["cam1"]
    assert report["recommended_exposure_us"] is None
    assert report["linearity"]["held"]["status"] == "insufficient_candidates"
    assert "unstable_signal" in report["provisional_flags"]
