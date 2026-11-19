"""Forward fit: rebuild the fibre image of a measured beam from a laser-spot scan.

A laser basis is a scan of single laser spots on the screen, recorded by the beam
camera and by the fibre cameras. For a measured beam image, a fit gives one
non-negative weight per spot:

- corrected: every beam pixel gives its value to the nearest spot within reach,
  with each spot at its local centroid;
- decomposed: the beam image is fitted as a non-negative sum of the spot images.

The synthetic fibre image is the sum of the spots' fibre images times these weights.
It is compared with the fibre image recorded for the same beam frame.

Files: model (spots, basis, selection rules, fits; arrays only), metrics
(comparisons and reference levels; arrays only), run (the pipeline module: one
config block to artefacts). Datasets are read through optidaq.analysis.datasets.
"""

from optidaq.analysis.forward_fit.metrics import METRICS, FibreMetrics, unit_rel_l2
from optidaq.analysis.forward_fit.model import (
    FITS,
    SELECTORS,
    Basis,
    Corrected,
    Decomposed,
    beam_centre,
    fibre_axis,
    locate_spot,
    nnls_gram,
)

__all__ = [
    "FITS",
    "METRICS",
    "SELECTORS",
    "Basis",
    "Corrected",
    "Decomposed",
    "FibreMetrics",
    "beam_centre",
    "fibre_axis",
    "locate_spot",
    "nnls_gram",
    "unit_rel_l2",
]
