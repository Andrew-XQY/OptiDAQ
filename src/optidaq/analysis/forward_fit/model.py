"""Linear forward model of a laser basis, on arrays only (no file access).

A laser basis is a scan of single laser spots on the screen. Per spot it holds the
spot's position and cleaned image in the beam camera, and the spot's image in every
fibre camera. A fit turns one beam image into one non-negative weight per spot; the
synthetic fibre image is the sum of the spots' fibre images times these weights.
Pixel coordinates are (x, y) = (column, row).

Two parts are chosen by name in the config, and a new one is one entry here:

    SELECTORS  which of the located spots form the basis
               f(positions, shape, **params) -> (spot indices, spacing in px or None)
    FITS       how a beam image becomes weights
               Fit(basis, **params), then fit(beam image) -> one weight per spot
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter, label, uniform_filter
from scipy.optimize import minimize
from scipy.spatial import cKDTree


def locate_spot(
    signed: np.ndarray, half: int = 20, far_px: float = 60.0
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Position and cleaned image of the laser spot in one beam-camera frame.

    ``signed`` is the background-subtracted frame, not clipped. The far level and
    noise are the median and standard deviation of the pixels farther than
    ``far_px`` from the brightest 5 x 5 neighbourhood. Kept pixels: within
    ``half`` px of the spot and more than two noise sigmas above the far level.
    Position: intensity-weighted mean of the kept pixels, taken twice, the second
    time around the first result. Image: the kept pixels of the first window minus
    the far level, scaled to unit sum. Returns None if no pixel is above the noise.

    The first moment of the whole frame after clipping at zero is pulled towards
    the frame centre by the clipped noise of the empty pixels; this position is not.
    """

    img = np.asarray(signed, dtype=np.float64)
    yy, xx = np.indices(img.shape)
    iy, ix = np.unravel_index(np.argmax(uniform_filter(img, 5)), img.shape)
    far = np.hypot(xx - ix, yy - iy) > far_px
    level = img - np.median(img[far])
    sigma = img[far].std()
    centre = np.array([ix, iy], dtype=np.float64)
    image = None
    for _ in range(2):
        keep = (np.abs(xx - centre[0]) <= half) & (np.abs(yy - centre[1]) <= half) & (level > 2 * sigma)
        total = level[keep].sum()
        if total <= 0:
            return None
        if image is None:
            image = np.where(keep, level, 0.0) / total
        centre = np.array([(xx[keep] * level[keep]).sum(), (yy[keep] * level[keep]).sum()]) / total
    return centre, image


def beam_centre(image: np.ndarray, level: float = 0.15) -> np.ndarray:
    """Centre of a beam image: intensity-weighted mean over the connected region
    around the brightest point where the image, smoothed by 2 px and with the
    frame median removed, is above ``level`` of its peak. The empty part of the
    frame does not enter, so its noise cannot pull the result."""

    img = np.asarray(image, dtype=np.float64)
    smooth = gaussian_filter(img, 2.0)
    smooth = smooth - np.median(smooth)
    peak = np.unravel_index(np.argmax(smooth), smooth.shape)
    regions, _ = label(smooth > level * smooth[peak])
    keep = regions == regions[peak]
    value = np.maximum(img - np.median(img), 0.0)[keep]
    yy, xx = np.nonzero(keep)
    total = value.sum()
    if total <= 0:
        return np.array([float(peak[1]), float(peak[0])])
    return np.array([(xx * value).sum(), (yy * value).sum()]) / total


def select_all(positions: np.ndarray, shape: Tuple[int, int]):
    """Every located spot. No spacing is implied, so the config must give reach_px."""

    return np.arange(len(positions)), None


def select_grid(positions: np.ndarray, shape: Tuple[int, int], cells: int):
    """One spot per cell of a grid of square cells; ``cells`` of them span the
    longer side of the image. Per cell: the spot nearest to the cell centre, if
    it lies within one cell size. Returns the sorted indices (each spot once) and
    the cell size in px."""

    height, width = shape
    cell = max(height, width) / cells
    cx = (np.arange(int(np.ceil(width / cell - 1e-9))) + 0.5) * cell - 0.5
    cy = (np.arange(int(np.ceil(height / cell - 1e-9))) + 0.5) * cell - 0.5
    gx, gy = np.meshgrid(cx, cy)
    dist, nearest = cKDTree(positions).query(np.stack([gx.ravel(), gy.ravel()], 1))
    return np.unique(nearest[dist <= cell]), cell


SELECTORS = {"all": select_all, "grid": select_grid}


@dataclass
class Basis:
    """Laser basis.

    positions: (K, 2) spot positions in beam-camera px.
    spots: (K, H*W) sparse cleaned unit-sum spot images (beam camera).
    responses: fibre -> (K, H', W') fibre images of the spots, background
        subtracted and clipped at zero.
    shape: (H, W) of the beam camera.
    reach: px; a beam pixel farther than this from every spot is outside the basis.
    """

    positions: np.ndarray
    spots: sparse.csr_matrix
    responses: Dict[str, np.ndarray]
    shape: Tuple[int, int]
    reach: float

    def synthesize(self, weights: np.ndarray, fibre: str) -> np.ndarray:
        """Synthetic fibre image: the spots' fibre images times the weights, summed."""

        r = self.responses[fibre]
        flat = np.asarray(weights, dtype=np.float32) @ r.reshape(len(r), -1)
        return flat.reshape(r.shape[1:]).astype(np.float64)

    def light(self, weights: np.ndarray) -> np.ndarray:
        """Beam-camera picture of the light a fit uses: spot images times weights."""

        return (self.spots.T @ np.asarray(weights, dtype=np.float64)).reshape(self.shape)

    @cached_property
    def nearest(self) -> Tuple[np.ndarray, np.ndarray]:
        """Per beam-camera pixel (flat): whether a spot lies within ``reach``, and
        for those pixels the index of the nearest spot."""

        yy, xx = np.indices(self.shape)
        dist, owner = cKDTree(self.positions).query(np.stack([xx.ravel(), yy.ravel()], 1))
        inside = dist <= self.reach
        return inside, owner[inside]

    def coverage(self, beam: np.ndarray) -> float:
        """Share of the beam light that falls within ``reach`` of a spot."""

        flat = np.ravel(beam).astype(np.float64)
        return float(flat[self.nearest[0]].sum() / max(float(flat.sum()), 1e-30))


class Corrected:
    """Corrected fit: every beam pixel gives its value to the nearest spot within
    the basis reach; a spot's weight is the sum it receives. Light outside the
    reach is not used. Spots sit at their local centroids."""

    def __init__(self, basis: Basis):
        self.inside, self.owner = basis.nearest
        self.n = len(basis.positions)

    def __call__(self, beam: np.ndarray) -> np.ndarray:
        return np.bincount(self.owner, weights=np.ravel(beam)[self.inside], minlength=self.n)


class Decomposed:
    """Decomposed fit: non-negative least squares of the beam image on the cleaned
    spot images; the fitted amounts are the weights. No spot position is used."""

    def __init__(self, basis: Basis):
        self.spots = basis.spots
        self.gram = (basis.spots @ basis.spots.T).toarray()

    def __call__(self, beam: np.ndarray) -> np.ndarray:
        return nnls_gram(self.gram, self.spots @ np.ravel(beam).astype(np.float64))


FITS = {"corrected": Corrected, "decomposed": Decomposed}


def nnls_gram(gram: np.ndarray, b: np.ndarray, ridge: float = 1e-9, max_iter: int = 3000) -> np.ndarray:
    """Exact active-set NNLS (Bro and De Jong) in Gram form.

    Minimises 0.5 x.G.x - x.b subject to x >= 0. A negligible ridge keeps the
    sub-systems solvable when spot images are nearly identical.
    """

    k = len(b)
    g = gram + ridge * np.trace(gram) / k * np.eye(k)
    active = np.zeros(k, dtype=bool)
    x = np.zeros(k)
    w = b.copy()
    tol = 1e-10 * np.abs(b).max()
    it = 0
    while (~active).any() and it < max_iter:
        j = int(np.argmax(np.where(active, -np.inf, w)))
        if w[j] <= tol:
            break
        active[j] = True
        while True:
            it += 1
            idx = np.flatnonzero(active)
            s = np.zeros(k)
            s[idx] = np.linalg.solve(g[np.ix_(idx, idx)], b[idx])
            if s[idx].min() > 0 or it >= max_iter:
                break
            bad = idx[s[idx] <= 0]
            alpha = np.min(x[bad] / np.maximum(x[bad] - s[bad], 1e-300))
            x = x + alpha * (s - x)
            active &= x > 1e-14 * max(x.max(), 1e-300)
            x[~active] = 0
        x = np.maximum(s, 0)
        w = b - g @ x
    return x


def fibre_axis(
    positions: np.ndarray, responses: np.ndarray, mask: np.ndarray, centre: Tuple[float, float]
) -> Tuple[np.ndarray, float]:
    """Fibre axis on the screen, in beam-camera px, from a laser basis.

    Per spot, the rms radius of its fibre image about ``centre`` (fibre-camera
    px) inside ``mask`` is measured. The axis is the point about which this rms
    radius is closest to a function of the spot's distance alone (a polynomial
    of degree 5 in the distance, fitted by least squares). Spots with less than
    0.2 of the median fibre signal are left out. Returns the axis and the share
    of the variance of the rms radius that the distance explains.
    """

    rows, cols = np.nonzero(mask)
    r2 = (cols - centre[0]) ** 2 + (rows - centre[1]) ** 2
    total, second = np.empty(len(responses)), np.empty(len(responses))
    for i in range(0, len(responses), 256):
        part = responses[i:i + 256][:, rows, cols].astype(np.float64)
        total[i:i + 256], second[i:i + 256] = part.sum(1), part @ r2
    keep = total > 0.2 * np.median(total)
    if keep.sum() < 12:
        return np.full(2, np.nan), float("nan")
    points, rms = positions[keep], np.sqrt(second[keep] / total[keep])
    span = float(np.ptp(points, axis=0).max())

    def loss(c):
        design = np.vander(np.hypot(*(points - c).T) / span, 6)
        return float(((rms - design @ np.linalg.lstsq(design, rms, rcond=None)[0]) ** 2).mean())

    start = (points * total[keep, None]).sum(0) / total[keep].sum()
    offsets = 0.05 * span * np.array([[0, 0], [-1, 1], [1, -1]])
    best = min(
        (minimize(loss, start + d, method="Nelder-Mead", options={"xatol": 0.02, "fatol": 1e-9}) for d in offsets),
        key=lambda result: result.fun,
    )
    return best.x, float(1 - best.fun / max(rms.var(), 1e-30))
