"""Comparison of a measured and a synthetic fibre image, and two reference levels.

Every metric compares unit-sum images inside the fibre mask: the pixels where the
mean measured image, smoothed by 2 px, is above ``mask_level`` of its peak.

    shape_rel_l2      |real - synthetic| / |real|, relative L2 (0 = same shape)
    pearson           correlation coefficient
    ssim              mean local structural similarity, 7 px window
    lowpass_rel_l2    relative L2 after a Gaussian blur of lowpass_px (coarse structure)
    highpass_pearson  correlation of what that blur removes (fine structure)
    radial_rel_l2     relative L2 of the radial profiles in 4 px rings about the
                      centroid of the mean image

METRICS holds one function per metric; a new metric is one entry. Each function
gets (m, real, synthetic, real_blurred, synthetic_blurred): the FibreMetrics
object, the two unit-sum images and their blurs of lowpass_px.

Reference levels from the measured frames alone: ``floor`` is every metric between
consecutive frames; ``camera_noise`` is the noise of one frame in shape_rel_l2 units.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

METRICS = {
    "shape_rel_l2": lambda m, p, s, lp, ls: m.rel_l2(p, s),
    "pearson": lambda m, p, s, lp, ls: m.pearson(p, s),
    "ssim": lambda m, p, s, lp, ls: m.ssim(p, s),
    "lowpass_rel_l2": lambda m, p, s, lp, ls: m.rel_l2(lp, ls),
    "highpass_pearson": lambda m, p, s, lp, ls: m.pearson(p - lp, s - ls),
    "radial_rel_l2": lambda m, p, s, lp, ls: m.radial_rel_l2(p, s),
}


class FibreMetrics:
    """Metrics of one fibre camera; the mask and the centre come from the mean
    measured image of the frames under study."""

    def __init__(self, mean_image: np.ndarray, mask_level: float = 0.02, lowpass_px: float = 4.0):
        mean = np.asarray(mean_image, dtype=np.float64)
        smooth = gaussian_filter(mean, 2.0)
        self.mask = smooth > mask_level * smooth.max()
        self.lowpass_px = float(lowpass_px)
        yy, xx = np.indices(mean.shape)
        total = mean.sum()
        self.centre = ((xx * mean).sum() / total, (yy * mean).sum() / total)
        self.rings = (np.hypot(xx - self.centre[0], yy - self.centre[1]) / 4.0).astype(int).ravel()

    def __call__(self, real: np.ndarray, synthetic: np.ndarray) -> Dict[str, float]:
        p, s = self.unit(real), self.unit(synthetic)
        lp, ls = gaussian_filter(p, self.lowpass_px), gaussian_filter(s, self.lowpass_px)
        return {name: metric(self, p, s, lp, ls) for name, metric in METRICS.items()}

    def floor(self, frames: Sequence[np.ndarray]) -> Dict[str, float]:
        """Median of every metric between consecutive measured frames: a synthetic
        image at this level is as close to the measured one as the next frame is."""

        rows = [self(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]
        return {k: float(np.median([r[k] for r in rows])) if rows else float("nan") for k in METRICS}

    def camera_noise(self, frames: Sequence[np.ndarray], sigma_px: float = 3.0) -> float:
        """Noise of one measured frame in shape_rel_l2 units, median over pairs.

        The difference of consecutive frames, divided by sqrt(2), is high-pass
        filtered (minus its Gaussian blur of sigma_px), which removes most of the
        slow beam change; its norm is corrected for the share of white noise the
        filter keeps and divided by the norm of the image.
        """

        z = np.random.default_rng(0).standard_normal(self.mask.shape)
        kept = (z - gaussian_filter(z, sigma_px)).var() / z.var()
        out = []
        for a, b in zip(frames[:-1], frames[1:]):
            a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
            d = (a - b) / np.sqrt(2.0)
            high = d - gaussian_filter(d, sigma_px)
            out.append(np.sqrt((high[self.mask] ** 2).sum() / kept) / np.linalg.norm((0.5 * (a + b))[self.mask]))
        return float(np.median(out)) if out else float("nan")

    def unit(self, image):
        """The image clipped at zero, zero outside the mask, scaled to unit sum."""

        a = np.maximum(np.asarray(image, dtype=np.float64), 0.0) * self.mask
        return a / max(a.sum(), 1e-30)

    def rel_l2(self, a, b):
        return float(np.linalg.norm((a - b)[self.mask]) / max(np.linalg.norm(a[self.mask]), 1e-30))

    def pearson(self, a, b):
        a, b = a[self.mask] - a[self.mask].mean(), b[self.mask] - b[self.mask].mean()
        return float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))

    def ssim(self, a, b, win=7, k1=0.01, k2=0.03):
        # Definition of skimage.metrics.structural_similarity with a uniform window,
        # sample covariance and data range = peak of a; mean of the map in the mask.
        c1, c2 = (k1 * max(a.max(), 1e-30)) ** 2, (k2 * max(a.max(), 1e-30)) ** 2
        ua, ub = uniform_filter(a, win), uniform_filter(b, win)
        norm = win * win / (win * win - 1.0)
        va = norm * (uniform_filter(a * a, win) - ua * ua)
        vb = norm * (uniform_filter(b * b, win) - ub * ub)
        cov = norm * (uniform_filter(a * b, win) - ua * ub)
        s = (2 * ua * ub + c1) * (2 * cov + c2) / ((ua * ua + ub * ub + c1) * (va + vb + c2))
        return float(s[self.mask].mean())

    def radial_rel_l2(self, a, b):
        ra, rb = self._radial(a), self._radial(b)
        return float(np.linalg.norm(ra - rb) / max(np.linalg.norm(ra), 1e-30))

    def _radial(self, a):
        profile = np.bincount(self.rings, weights=a.ravel())
        return profile / max(profile.sum(), 1e-30)


def unit_rel_l2(reference: np.ndarray, other: np.ndarray) -> float:
    """Relative L2 distance of two images scaled to unit sum, over the whole frame."""

    a = np.ravel(reference) / max(float(np.sum(reference)), 1e-30)
    b = np.ravel(other) / max(float(np.sum(other)), 1e-30)
    return float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-30))
