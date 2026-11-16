"""Standalone NumPy Gaussian pattern source, with no hardware dependencies.

Extracted from Andrew Xu's Gaussian generator (MIT), retaining its sampling
and rendering conventions. Each stream owns a legacy RandomState so seeded
single-stream output is preserved without changing NumPy's global RNG.
Brightness conversion and display transforms belong to the caller.

Copyright (c) 2025 Andrew Xu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Optional

import numpy as np


def _center_offset(rng, width, height, radius_range, angle_range=None):
    lo, hi = map(float, radius_range)
    if not (np.isfinite(lo) and np.isfinite(hi) and 0.0 <= lo <= hi):
        raise ValueError("center_radius_range must satisfy 0 <= lo <= hi and be finite")
    radius = float(min(width, height)) * np.sqrt(rng.uniform(lo**2, hi**2))
    if angle_range is None:
        angle = rng.uniform(0.0, 2.0 * np.pi)
        return float(radius * np.cos(angle)), float(radius * np.sin(angle))
    lo, hi = map(float, angle_range)
    if not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi):
        raise ValueError("center_angle_range must contain finite degrees with lo <= hi")
    angle = np.deg2rad(rng.uniform(lo, hi))
    # Explicit angles use Cartesian semantics: 90 degrees points up.
    return float(radius * np.cos(angle)), float(-radius * np.sin(angle))


def _beta_scaled(rng, mode):
    mode = float(np.clip(mode, 1e-6, 1.0 - 1e-6))
    a = max(1e-3, 5.0 * 2.0 * mode)
    b = max(1e-3, 15.0 * 2.0 * (1.0 - mode))
    return 0.01 + 0.99 * rng.beta(a, b)


def _gaussian(
    rng, x, y, width, height, *,
    std_1=0.15, std_2=0.12, max_intensity=10, fade_rate=0.5,
    area_boost_scale=0.25, max_peak_intensity=None, distribution="",
    intensity_range=None, center_radius_range=None, aspect_range=None,
    orientation_range=None, center_angle_range=None, rendered_center_offset=None,
):
    if distribution == "beta":
        std_x, std_y = _beta_scaled(rng, std_1), _beta_scaled(rng, std_2)
    else:
        aspect_lo, aspect_hi = aspect_range if aspect_range is not None else (0.5, 2.0)
        std = rng.uniform(low=float(min(std_1, std_2)), high=float(max(std_1, std_2)))
        if rng.uniform(-1.0, 1.0) < 0:
            std_x = std
            std_y = std_x * rng.uniform(aspect_lo, aspect_hi)
        else:
            std_y = std
            std_x = std_y * rng.uniform(aspect_lo, aspect_hi)
    std_x = max(std_x * float(width), 1e-6)
    std_y = max(std_y * float(height), 1e-6)

    if intensity_range is not None:
        lo, hi = map(float, intensity_range)
        if not (np.isfinite(lo) and np.isfinite(hi) and 0.0 < lo <= hi):
            raise ValueError("intensity_range must satisfy 0 < lo <= hi and be finite")
        intensity = float(rng.uniform(lo, hi))
    else:
        low = (-float(max_intensity) if fade_rate == 1 else
               float(fade_rate) * float(max_intensity) / (float(fade_rate) - 1.0))
        intensity = float(rng.uniform(low, float(max_intensity)))
    if intensity <= 0:
        return np.zeros((height, width), dtype=float)

    boost = max(0.0, float(area_boost_scale))
    if boost > 0.0:
        area = (width * height) / (std_x * std_y)
        intensity += float(rng.uniform(0.0, area * boost))
    if max_peak_intensity is not None:
        cap = float(max_peak_intensity)
        if not np.isfinite(cap) or cap <= 0:
            raise ValueError("max_peak_intensity must be finite and > 0 when set")
        intensity = min(intensity, cap)

    if orientation_range is None:
        rotation = np.deg2rad(rng.uniform(0.0, 360.0))
    else:
        lo, hi = map(float, orientation_range)
        if not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi):
            raise ValueError("orientation_range must contain finite degrees with lo <= hi")
        angle = np.deg2rad(rng.uniform(lo, hi))
        rotation = -angle if std_x >= std_y else np.pi / 2.0 - angle
        rotation = float(rotation % (2.0 * np.pi))
    cos_t, sin_t = np.cos(rotation), np.sin(rotation)

    if rendered_center_offset is not None:
        ox, oy = map(float, rendered_center_offset)
        if not (np.isfinite(ox) and np.isfinite(oy)):
            raise ValueError("rendered_center_offset must contain finite pixel offsets")
    elif center_radius_range is not None:
        ox, oy = _center_offset(rng, width, height, center_radius_range, center_angle_range)
    else:
        if center_angle_range is not None:
            raise ValueError("center_angle_range requires center_radius_range")
        ox = oy = None
    if ox is None:
        dx = float(rng.uniform(0.0, width / 2.25))
        dy = float(rng.uniform(0.0, height / 2.25))
    else:
        # Rotate the offset too, so the rendered peak lands at (ox, oy).
        dx = float(-(cos_t * ox - sin_t * oy))
        dy = float(-(sin_t * ox + cos_t * oy))

    x_new = x * cos_t - y * sin_t
    y_new = x * sin_t + y * cos_t
    x_new += dx
    y_new += dy
    image = np.exp(-((x_new**2) / (2.0 * std_x**2) + (y_new**2) / (2.0 * std_y**2)))
    image *= intensity
    return image


def dynamic_patterns(
    canvas_size: Sequence[int],
    num_gaussians: int,
    max_pixel_value: float = 255.0,
    minimum_pixel_threshold: Optional[float] = None,
    seed: Optional[int] = None,
    update: Optional[Mapping[str, Any]] = None,
) -> Iterator[np.ndarray]:
    """Yield independent float64 Gaussian canvases forever.

    ``update`` selects the Gaussian priors; ``component_params`` can override
    them per Gaussian, and ``shared_center`` gives components one sampled
    center. Sum components, clip to ``max_pixel_value``, then threshold.
    """
    try:
        shape = tuple(canvas_size)
    except TypeError as exc:
        raise ValueError("dynamic_patterns canvas_size must contain two positive integers") from exc
    if len(shape) != 2 or any(
        isinstance(size, (bool, np.bool_))
        or not isinstance(size, (int, np.integer)) or size <= 0
        for size in shape
    ):
        raise ValueError("dynamic_patterns canvas_size must contain two positive integers")
    if (isinstance(num_gaussians, (bool, np.bool_))
            or not isinstance(num_gaussians, (int, np.integer)) or num_gaussians <= 0):
        raise ValueError("dynamic_patterns num_gaussians must be a positive integer")
    ceiling = float(max_pixel_value)
    if not np.isfinite(ceiling) or ceiling <= 0:
        raise ValueError("max_pixel_value must be finite and > 0")
    threshold = None if minimum_pixel_threshold is None else float(minimum_pixel_threshold)
    if threshold is not None and not np.isfinite(threshold):
        raise ValueError("minimum_pixel_threshold must be finite or None")

    common = dict(update or {})
    shared_center = common.pop("shared_center", False)
    if not isinstance(shared_center, (bool, np.bool_)):
        raise TypeError("shared_center must be a boolean")
    if shared_center and common.get("center_radius_range") is None:
        raise ValueError("shared_center=True requires center_radius_range")
    overrides = common.pop("component_params", None)
    if overrides is None:
        components = [dict(common) for _ in range(num_gaussians)]
    else:
        if isinstance(overrides, (str, bytes)) or not isinstance(overrides, Sequence):
            raise TypeError("component_params must be a sequence of mappings or None")
        if len(overrides) != num_gaussians:
            raise ValueError("component_params length must match number of Gaussians")
        components = []
        for params in overrides:
            if not isinstance(params, Mapping):
                raise TypeError("component_params entries must be mappings")
            if {"shared_center", "rendered_center_offset"}.intersection(params):
                raise ValueError("component_params cannot override shared_center or rendered_center_offset")
            components.append({**common, **params})

    height, width = map(int, shape)
    x, y = np.meshgrid(np.linspace(0, width - 1, width, dtype=float),
                       np.linspace(0, height - 1, height, dtype=float))
    x -= width / 2.0
    y -= height / 2.0
    rng = np.random.RandomState(None if seed is None else int(seed))

    def stream():
        while True:
            offset = (_center_offset(rng, width, height, common["center_radius_range"],
                                     common.get("center_angle_range")) if shared_center else None)
            image = np.zeros((height, width), dtype=float)
            for params in components:
                effective = {**params, "rendered_center_offset": offset} if shared_center else params
                image += _gaussian(rng, x, y, width, height, **effective)
            image = np.clip(image, 0.0, ceiling)
            if threshold is not None:
                image[image < threshold] = 0.0
            yield image

    return stream()
