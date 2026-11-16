#!/usr/bin/env python3
"""Show random SGM patterns on the ViALUX DMD, with a simple timed display loop.

Edit CONFIG below, then run on the DMD PC:

    python tests/dmd_show_random_sgm.py

The SGM generator, canvas, seed, rotation, and DMD device settings come from the
main experiment JSON. CONFIG supplies bright, elongated mixtures for this test.
Components are summed in floating point, then each pattern
is scaled to peak 255. This replaces the main config's peak-intensity prior;
there is no clipping of overlapping components or hard threshold on the tails.
A fixed seed repeats the same random sequence. No camera or CERN client is used.
Requires NumPy and ALP4lib; local repository imports work without installation.

Defaults use 2-4 components (expected mean 3), with random orientations and
short/long axis ratios of 0.25-0.65 on the square canvas. Centers span a disk of
radius 0.60 times the canvas's shorter side; edge cutoffs are intentional.
Normalization uses the visible mixture after truncation. For reconstruction,
use the actual truncated input within the calibrated field of view as the target.

interval_s is the hold after each display update; preparing the next frame adds
some overhead. This script does not synchronize camera exposures. Ctrl-C stops
early and releases the DMD.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from optidaq.data.transforms import build_pipeline  # noqa: E402
from optidaq.extensions.sgm import dynamic_patterns  # noqa: E402
from optidaq.extensions.vialux_alp import AlpDmd  # noqa: E402


CONFIG = {
    "experiment_config": "configs/experiment_lab_v1.json",  # relative to repository root
    "count": 100,             # number of random SGM samples to show
    "interval_s": 5.0,        # seconds to hold each sample
    "num_gaussians_range": [2, 4],  # expected mean: 3; [1, 1] for single spots
    "std_range": [0.08, 0.13],     # broad spots, with more size variation
    "aspect_range": [0.25, 0.65],  # short/long axis ratio: elongated, randomly rotated
    "center_radius_range": [0.0, 0.60],  # wide positions; allow edge cutoffs
    "peak_value": 255,        # full DMD brightness at each pattern's peak
    "gamma": 1.0,             # 1 preserves Gaussian decay; <1 lifts/broadens the tails
    "upscale": 3,             # each source pixel uses 3x3 mirrors
}


def positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("%s must be a positive integer" % name)
    return value


def ordered_range(values: object, name: str, *, allow_zero: bool = False) -> list[float]:
    result = [float(value) for value in values]
    if (len(result) != 2 or not all(map(math.isfinite, result))
            or result[0] > result[1]
            or (result[0] < 0 if allow_zero else result[0] <= 0)):
        raise ValueError("%s must contain two finite, ordered values" % name)
    return result


def load_test_config(settings: dict) -> tuple[dict, float]:
    count = positive_integer(settings["count"], "count")
    interval_s = float(settings["interval_s"])
    if not math.isfinite(interval_s) or interval_s <= 0:
        raise ValueError("interval_s must be positive and finite")
    upscale = positive_integer(settings["upscale"], "upscale")
    if positive_integer(settings["peak_value"], "peak_value") > 255:
        raise ValueError("peak_value must be <= 255")
    if not math.isfinite(settings["gamma"]) or settings["gamma"] <= 0:
        raise ValueError("gamma must be positive and finite")
    counts = settings["num_gaussians_range"]
    if len(counts) != 2:
        raise ValueError("num_gaussians_range must contain two positive integers")
    low, high = [positive_integer(value, "num_gaussians_range") for value in counts]
    if low > high:
        raise ValueError("num_gaussians_range must be ordered")
    widths = ordered_range(settings["std_range"], "std_range")
    aspects = ordered_range(settings["aspect_range"], "aspect_range")
    radii = ordered_range(settings["center_radius_range"], "center_radius_range", allow_zero=True)

    path = Path(settings["experiment_config"]).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    with path.open(encoding="utf-8") as handle:
        cfg = json.load(handle)

    sgm_streams = [
        spec for spec in cfg["dmd"]["streams"]
        if spec.get("name") == "dynamic_patterns"
    ]
    if len(sgm_streams) != 1:
        raise ValueError("main config must contain exactly one dynamic_patterns SGM stream")
    source = sgm_streams[0]
    params = source.setdefault("params", {})
    update = params.setdefault("update", {})
    # Equal-brightness components, with no clipping or threshold before normalization.
    update.update(
        std_1=widths[0], std_2=widths[1], distribution="other",
        aspect_range=aspects, center_radius_range=radii, intensity_range=[1.0, 1.0],
        area_boost_scale=0.0, max_peak_intensity=None,
    )
    update.pop("component_params", None)
    update.pop("shared_center", None)
    params["minimum_pixel_threshold"] = None
    params["max_pixel_value"] = float(high)  # Sum cannot exceed the number of components.
    source["transforms"] = []  # Convert to uint8 only after normalizing the complete sum.
    cfg["dmd"]["streams"] = sgm_streams
    cfg["dmd"]["display_transforms"] = [
        {"name": "upscale", "params": {"factor": upscale}},
        *[spec for spec in cfg["dmd"].get("display_transforms", [])
          if not (isinstance(spec, dict) and spec.get("name") == "upscale")],
    ]
    cfg["dmd"]["count"] = count
    cfg["dmd"]["settle_s"] = 0.0  # This display-only loop controls its own hold time.
    return cfg, interval_s


def bright_patterns(cfg: dict, settings: dict):
    """Yield (component count, uint8 image) with independent reproducible streams."""
    params = cfg["dmd"]["streams"][0]["params"]
    seed = params.get("seed")
    rng = np.random.RandomState(seed)
    low, high = settings["num_gaussians_range"]
    streams = {}
    for count in range(low, high + 1):
        component_params = deepcopy(params)
        component_params["num_gaussians"] = count
        component_params["seed"] = int(rng.randint(0, 2**31 - 1))
        streams[count] = dynamic_patterns(**component_params)
    while True:
        count = int(rng.randint(low, high + 1))
        raw = next(streams[count])
        peak = float(raw.max())
        if not np.isfinite(raw).all() or peak <= 0:
            raise ValueError("SGM generator produced an empty or invalid pattern")
        image = np.rint(settings["peak_value"] * (raw / peak) ** settings["gamma"])
        yield count, image.astype(np.uint8)


def display_pipeline(cfg: dict, height: int, width: int):
    specs = deepcopy(cfg["dmd"].get("display_transforms", []))
    for spec in specs:
        if isinstance(spec, dict) and spec.get("name") == "fit_center":
            params = spec.setdefault("params", {})
            for key, size in (("height", height), ("width", width)):
                if params.get(key) is None:
                    params[key] = size
    return build_pipeline(specs)


def main(settings=None) -> None:
    settings = CONFIG if settings is None else settings
    cfg, interval_s = load_test_config(settings)
    patterns = bright_patterns(cfg, settings)
    first = next(patterns)  # Validate generation before opening the DMD.
    device = AlpDmd(**dict(cfg["dmd"].get("device", {}) or {}))
    try:
        device.open()
        display = display_pipeline(cfg, device.height, device.width)
        seed = cfg["dmd"]["streams"][0].get("params", {}).get("seed")
        print("Showing %d SGM samples; hold %.3g s each; source seed %s" % (
            settings["count"], interval_s, seed
        ))
        print("Components %s; peak %d; widths %s; display upscale %dx" % (
            settings["num_gaussians_range"], settings["peak_value"],
            settings["std_range"], settings["upscale"]
        ))
        for index in range(settings["count"]):
            components, pattern = first if index == 0 else next(patterns)
            frame = display(pattern)
            device.show(frame)
            print("SGM %d/%d  components %d  DMD peak %d  hold %.3g s" % (
                index + 1, settings["count"], components, int(frame.max()), interval_s
            ), flush=True)
            time.sleep(interval_s)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        try:
            device.close()
        except Exception as exc:
            print("WARNING: DMD cleanup failed: %s" % exc)


if __name__ == "__main__":
    main()
