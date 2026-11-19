"""Dataset access of the analysis; the only analysis code that reads input files.

Analysis modules see every dataset as a ``Dataset`` with three calls:

    samples(cameras) -> (samples, info)   samples that have every listed camera,
                                          in time order
    image(sample, camera) -> array        2-D float64 counts, preprocessed
    mean_image(camera) -> array           mean of all images of one camera

A sample is a dict with "run_id", "sample_id" and "time_ns"; ``info`` is a dict of
what the adapter did (counts, thresholds) and is stored with the results.

Config, section "datasets": one block per dataset, under a name of your choice:

    {"adapter": "optidaq_db", "path": folder, <adapter options>, "preprocess": [...]}

Two things can be plugged in by name:

ADAPTERS, one per storage format. An adapter is a class with
``__init__(path, **options)``, ``samples(cameras)`` and ``image(sample, camera)``
as above, without preprocessing. Pairing of the cameras and filtering of samples
are the adapter's business; its options are the remaining keys of the block.

PREPROCESS, steps applied to every image in the listed order. Each entry is
"name" or {"name": ..., "params": {...}}:

    subtract_background  params {"dataset": name}: minus the mean image of the
                         same camera in that dataset. The result is signed.

Adapter "optidaq_db": a folder with ``dataset.db`` and the PNG files it lists, as
written by the OptiDAQ export. Only table ``images`` is read, with the columns

    camera, png_path (relative to the folder, or absolute), run_id, sample_id,
    sync_time_ns, acq_stamp_ns

plus any column named in ``where`` or ``otsu``. Rows with the same run_id and
sample_id form one sample. PNGs are 8- or 16-bit grayscale, and all images of one
camera have the same shape. Options:

    sync_guard_ns  a sample is used when every camera's sync_time_ns is within
                   this many ns of the first camera's (default 1e6); null: no check
    where          extra SQL condition; every camera is a table alias of its own
                   name, e.g. "cam2.roi_energy_fraction >= 0.8"
    otsu           one column, e.g. "cam2.stat_mean": samples below the valley
                   between its two modes are dropped (frames without beam or spot)
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from optidaq.analysis.config import named, pick


class OptidaqDb:
    """Adapter for a folder with dataset.db and PNGs (contract in the module text)."""

    def __init__(self, path, *, sync_guard_ns: Optional[int] = 1_000_000,
                 where: Optional[str] = None, otsu: Optional[str] = None):
        self.folder = Path(path)
        self.sync_guard_ns, self.where, self.otsu = sync_guard_ns, where, otsu

    def samples(self, cameras: Sequence[str]) -> Tuple[List[dict], dict]:
        cams = ['"%s"' % c for c in cameras]
        first = cams[0]
        columns = ["%s.png_path" % c for c in cams]
        columns += ["%s.run_id" % first, "%s.sample_id" % first, "%s.acq_stamp_ns" % first]
        if self.otsu:
            columns.append(self.otsu)
        joins = "".join(
            " JOIN images %s ON %s.run_id = %s.run_id AND %s.sample_id = %s.sample_id" % (c, c, first, c, first)
            for c in cams[1:]
        )
        conditions = ["%s.camera = ?" % c for c in cams]
        conditions += ["%s.png_path IS NOT NULL" % c for c in cams]
        params = list(cameras)
        if self.sync_guard_ns is not None:
            conditions += ["abs(%s.sync_time_ns - %s.sync_time_ns) <= ?" % (c, first) for c in cams[1:]]
            params += [int(self.sync_guard_ns)] * (len(cams) - 1)
        if self.where:
            conditions.append("(%s)" % self.where)
        sql = "SELECT %s FROM images %s%s WHERE %s ORDER BY %s.acq_stamp_ns" % (
            ", ".join(columns), first, joins, " AND ".join(conditions), first,
        )
        rows = self._query(sql, params)

        n = len(cameras)
        threshold = None
        if self.otsu and rows:
            values = np.array([np.nan if r[n + 3] is None else r[n + 3] for r in rows], dtype=float)
            threshold = otsu_threshold(values[np.isfinite(values)])
            rows = [r for r, v in zip(rows, values) if v >= threshold]
        samples = [
            {
                "run_id": r[n],
                "sample_id": r[n + 1],
                "time_ns": r[n + 2],
                "png": {c: self._resolve(p) for c, p in zip(cameras, r[:n])},
            }
            for r in rows
        ]
        return samples, {"samples": len(samples), "threshold": threshold}

    def image(self, sample: dict, camera: str) -> np.ndarray:
        path = sample["png"][camera]
        with Image.open(path) as img:
            arr = np.asarray(img)
        if arr.ndim != 2:
            raise ValueError("expected a grayscale PNG, got shape %s for %s" % (arr.shape, path))
        return arr.astype(np.float64)

    def _query(self, sql: str, params=()):
        db = self.folder / "dataset.db"
        if not db.is_file():
            raise FileNotFoundError(db)
        con = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            return con.execute(sql, params).fetchall()
        finally:
            con.close()

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.folder / p


def otsu_threshold(values) -> float:
    """Valley between the two modes of a positive quantity: Otsu on log10(v + 1)."""

    x = np.log10(np.asarray(values, dtype=float) + 1.0)
    hist, edges = np.histogram(x, bins=64)
    centres = 0.5 * (edges[1:] + edges[:-1])
    weights = hist.astype(float)
    total = weights.sum()
    best, threshold = -1.0, centres[len(centres) // 2]
    for k in range(1, len(centres)):
        w0, w1 = weights[:k].sum(), weights[k:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0 = (weights[:k] * centres[:k]).sum() / w0
        m1 = (weights[k:] * centres[k:]).sum() / w1
        between = w0 * w1 * (m0 - m1) ** 2 / total ** 2
        if between > best:
            best, threshold = between, edges[k]
    return float(10 ** threshold - 1.0)


def subtract_background(open_dataset: Callable[[str], "Dataset"], dataset: str):
    """Step: image minus the mean image of the same camera in ``dataset``."""

    background = open_dataset(dataset)
    return lambda image, camera: image - background.mean_image(camera)


ADAPTERS = {"optidaq_db": OptidaqDb}
PREPROCESS = {"subtract_background": subtract_background}


class Dataset:
    """What analysis modules see of a dataset: an adapter plus preprocessing steps."""

    def __init__(self, source, steps: Sequence[Callable] = ()):
        self.source, self.steps = source, list(steps)
        self._means: Dict[str, np.ndarray] = {}

    def samples(self, cameras: Sequence[str]) -> Tuple[List[dict], dict]:
        return self.source.samples(list(cameras))

    def image(self, sample: dict, camera: str) -> np.ndarray:
        image = self.source.image(sample, camera)
        for step in self.steps:
            image = step(image, camera)
        return image

    def mean_image(self, camera: str) -> np.ndarray:
        if camera not in self._means:
            samples, _ = self.source.samples([camera])
            if not samples:
                raise ValueError("no %s images for a mean image" % camera)
            total = None
            for s in samples:
                image = self.image(s, camera)
                total = image if total is None else total + image
            mean = total / len(samples)
            mean.setflags(write=False)
            self._means[camera] = mean
        return self._means[camera]


def opener(blocks: Mapping[str, Mapping]) -> Callable[[str], Dataset]:
    """For the "datasets" section of a config: a function name -> Dataset.
    Every dataset is opened once, so its mean images are computed once."""

    opened: Dict[str, Dataset] = {}

    def open_dataset(name: str) -> Dataset:
        if name not in opened:
            block = dict(pick(blocks, name, "dataset"))
            adapter = pick(ADAPTERS, block.pop("adapter", "optidaq_db"), "dataset adapter")
            entries = block.pop("preprocess", None) or []
            try:
                source = adapter(block.pop("path"), **block)
            except (KeyError, TypeError) as error:
                raise ValueError('dataset "%s": needs "path"; adapter options: %s' % (name, error)) from error
            opened[name] = Dataset(source)
            for entry in entries:
                step, params = named(entry)
                opened[name].steps.append(pick(PREPROCESS, step, "preprocessing step")(open_dataset, **params))
        return opened[name]

    return open_dataset
