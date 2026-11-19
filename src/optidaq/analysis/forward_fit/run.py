"""Forward-fit module of the analysis pipeline: laser basis -> synthetic fibre
images of beam frames -> metrics and artefacts.

Block in the "pipeline" section of the config; every key but "module" is optional:

    module        "forward_fit"
    basis         name of the dataset with the laser-spot scan (default "basis")
    target        name of the dataset with the beam frames to rebuild (default "target")
    max_frames    keep this many target frames, evenly spaced in time; null: all
    spots         how the scan becomes a basis
        half_px, far_px  window and far region of locate_spot
        select           which located spots are used, "name" or {"name", "params"}:
                         "all", or "grid" with params {"cells": n} (one spot per
                         square cell, n cells along the longer image side)
        reach_px         beam light farther than this from every used spot is
                         outside the basis; null: the cell size of "grid"
    fits          list of fits, each "name" or {"name", "params"}:
                  "corrected", "decomposed"
    coverage_min  frames with a smaller share of the beam light inside the basis
                  stay in frames.csv but are left out of the summary
    metrics       {"mask_level", "lowpass_px"} of FibreMetrics
    samples       frames whose images are saved for figures
        ids        sample ids to save (always fitted, also with max_frames)
        quantiles  also the frames at these quantiles of shape_rel_l2, averaged
                   over fibres and fits, among the covered frames

Preprocessing is set per dataset (datasets.py). On top of it this module clips
beam frames, target fibre images and spot fibre images at zero; the beam-camera
spot frames stay signed for locate_spot.

Artefacts in <output_dir>/<block name>/, the only thing figures read:

    frames.csv    one row per fibre, fit and frame: fibre, camera, fit, run_id,
                  sample_id, time_ns, coverage, covered, beam_rel_l2 and every
                  metric of metrics.py
    summary.json  the resolved block, counts, and per fibre: floor, camera noise,
                  fibre axis, and quartiles of every metric per fit over the
                  covered frames
    weights.npz   all frames: weights/<fit> (frames x spots), positions
                  (spots x 2), coverage, run_id, sample_id, time_ns, fits
    samples.npz   the saved frames: beam, beam_centre, light/<fit>,
                  measured/<fibre>, synthetic/<fibre>/<fit>, mask/<fibre>,
                  axis/<fibre>, run_id, sample_id, time_ns, fits, fibres

Floor and camera noise compare consecutive kept frames; target "step" in
summary.json is how many recorded frames apart they are (1 without thinning).
Large steps add beam motion to both levels.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Callable, Dict, Mapping

import numpy as np
from scipy import sparse

from optidaq.analysis.config import fill, named, pick
from optidaq.analysis.forward_fit.metrics import FibreMetrics, unit_rel_l2
from optidaq.analysis.forward_fit.model import (
    FITS,
    SELECTORS,
    Basis,
    beam_centre,
    fibre_axis,
    locate_spot,
)

DEFAULTS = {
    "module": "forward_fit",
    "basis": "basis",
    "target": "target",
    "max_frames": None,
    "spots": {
        "half_px": 20,
        "far_px": 60,
        "select": {"name": "grid", "params": {"cells": 34}},
        "reach_px": None,
    },
    "fits": ["corrected", "decomposed"],
    "coverage_min": 0.9,
    "metrics": {"mask_level": 0.02, "lowpass_px": 4.0},
    "samples": {"ids": [], "quantiles": [0.1, 0.5, 0.9]},
}


def resolve(params: Mapping) -> dict:
    """The block with defaults filled in; unknown keys and unknown fits are errors."""

    p = fill(params, DEFAULTS, "forward_fit")
    for key in ("spots", "metrics", "samples"):
        p[key] = fill(p[key], DEFAULTS[key], "forward_fit." + key)
    name, select = named(p["spots"]["select"])
    pick(SELECTORS, name, "spot selection")
    p["spots"]["select"] = {"name": name, "params": select}
    fits = [named(entry) for entry in p["fits"]]
    names = [name for name, _ in fits]
    if not fits or len(set(names)) != len(names):
        raise ValueError("forward_fit.fits: expected one or more different fits, got %s" % names)
    for name in names:
        pick(FITS, name, "fit")
    p["fits"] = [{"name": name, "params": fit} for name, fit in fits]
    return p


def load_basis(dataset, beam: str, fibres: Mapping[str, str], spots: Mapping, log=print):
    """Locate every laser spot in the beam camera, select the spots of the basis,
    and load their fibre images."""

    samples, info = dataset.samples([beam, *fibres.values()])
    found, shape = [], None
    for s in samples:
        frame = dataset.image(s, beam)
        shape = frame.shape
        located = locate_spot(frame, spots["half_px"], spots["far_px"])
        if located is not None:
            found.append((s, located[0], sparse.csr_matrix(located[1].ravel())))
    if not found:
        raise ValueError("no laser spot found in the basis dataset")
    positions = np.array([position for _, position, _ in found])
    select = spots["select"]
    keep, spacing = SELECTORS[select["name"]](positions, shape, **select["params"])
    reach = spots["reach_px"] if spots["reach_px"] is not None else spacing
    if reach is None:
        raise ValueError('spot selection "%s" implies no spacing: set spots.reach_px' % select["name"])
    basis = Basis(
        positions=positions[keep],
        spots=sparse.vstack([found[i][2] for i in keep], format="csr"),
        responses={
            fibre: np.stack([np.maximum(dataset.image(found[i][0], camera), 0.0).astype(np.float32) for i in keep])
            for fibre, camera in fibres.items()
        },
        shape=shape,
        reach=float(reach),
    )
    info = {**info, "located": len(found), "used": len(keep), "reach_px": float(reach)}
    log("basis: %d samples, %d spots located, %d used (%s), reach %.2f px"
        % (info["samples"], len(found), len(keep), select["name"], reach))
    return basis, info


def load_target(dataset, beam: str, fibres: Mapping[str, str], max_frames=None, always=(), log=print):
    """Beam frames and their measured fibre images, clipped at zero. With
    ``max_frames`` the frames are thinned evenly; ``always`` lists sample ids
    that are kept in any case."""

    samples, info = dataset.samples([beam, *fibres.values()])
    if not samples:
        raise ValueError("no frames in the target dataset")
    index = {s["sample_id"]: i for i, s in enumerate(samples)}
    missing = [i for i in always if i not in index]
    if missing:
        raise ValueError("samples.ids not in the target dataset after its filters: %s" % missing)
    keep = np.arange(len(samples))
    if max_frames and len(samples) > max_frames:
        keep = np.round(np.linspace(0, len(samples) - 1, max_frames)).astype(int)
        keep = np.unique(np.r_[keep, [index[i] for i in always]].astype(int))
    samples = [samples[i] for i in keep]

    def stack(camera):
        return np.stack([np.maximum(dataset.image(s, camera), 0.0).astype(np.float32) for s in samples])

    beams, real = stack(beam), {fibre: stack(camera) for fibre, camera in fibres.items()}
    step = float(np.median(np.diff(keep))) if len(keep) > 1 else 1.0
    info = {**info, "used": len(samples), "step": step}
    log("target: %d samples, %d used, %g recorded frames apart" % (info["samples"], len(samples), step))
    return samples, beams, real, info


def choose(samples, covered: np.ndarray, score: np.ndarray, ids, quantiles) -> list:
    """Indices of the frames whose images are saved: the listed sample ids, and
    per quantile the covered frame whose score is closest to that quantile."""

    index = {s["sample_id"]: i for i, s in enumerate(samples)}
    chosen = {index[i] for i in ids}
    pool = np.flatnonzero(covered & np.isfinite(score))
    if not len(pool):
        pool = np.flatnonzero(np.isfinite(score))
    for q in quantiles if len(pool) else ():
        chosen.add(int(pool[np.argmin(np.abs(score[pool] - np.quantile(score[pool], q)))]))
    return sorted(chosen)


def run(params: Mapping, *, cameras: Mapping, dataset: Callable, out_dir, log=print) -> dict:
    """Run the forward fit of one config block; write the artefacts and return the summary.

    cameras: {"beam": camera, "fibre": {label: camera}}; dataset: name -> Dataset.
    """

    p = resolve(params)
    beam, fibres = cameras["beam"], dict(cameras["fibre"])
    basis, basis_info = load_basis(dataset(p["basis"]), beam, fibres, p["spots"], log)
    samples, beams, real, target_info = load_target(
        dataset(p["target"]), beam, fibres, p["max_frames"], p["samples"]["ids"], log
    )
    if beams.shape[1:] != tuple(basis.shape):
        raise ValueError("beam camera: basis frames are %s, target frames %s" % (basis.shape, beams.shape[1:]))
    coverage = np.array([basis.coverage(b) for b in beams])
    covered = coverage >= p["coverage_min"]

    weights: Dict[str, np.ndarray] = {}
    for entry in p["fits"]:
        fit = FITS[entry["name"]](basis, **entry["params"])
        weights[entry["name"]] = _each(fit, beams, entry["name"], log)
    beam_error = {f: [unit_rel_l2(b, basis.light(w)) for b, w in zip(beams, weights[f])] for f in weights}

    rows, results, masks, axes = [], {}, {}, {}
    for fibre, camera in fibres.items():
        frames = real[fibre]
        metrics = FibreMetrics(frames.mean(0, dtype=np.float64), **p["metrics"])
        axis, explained = fibre_axis(basis.positions, basis.responses[fibre], metrics.mask, metrics.centre)
        masks[fibre], axes[fibre] = metrics.mask, axis
        result = {
            "camera": camera,
            "frames": len(frames),
            "covered": int(covered.sum()),
            "camera_noise": metrics.camera_noise(frames),
            "floor": metrics.floor(frames),
            "axis_px": [float(v) for v in axis],
            "axis_explained": explained,
            "fits": {},
        }
        for fit in weights:
            values = []
            for i, s in enumerate(samples):
                m = {"beam_rel_l2": beam_error[fit][i], **metrics(frames[i], basis.synthesize(weights[fit][i], fibre))}
                values.append(m)
                rows.append({
                    "fibre": fibre, "camera": camera, "fit": fit, "run_id": s["run_id"],
                    "sample_id": s["sample_id"], "time_ns": s["time_ns"],
                    "coverage": float(coverage[i]), "covered": int(covered[i]), **m,
                })
            result["fits"][fit] = {k: _quartiles([v[k] for v, c in zip(values, covered) if c]) for k in values[0]}
        results[fibre] = result

    score = np.mean([[r["shape_rel_l2"] for r in rows[i:i + len(samples)]]
                     for i in range(0, len(rows), len(samples))], axis=0)
    saved = choose(samples, covered, score, p["samples"]["ids"], p["samples"]["quantiles"])

    summary = {
        "module": "forward_fit", "params": p, "cameras": {"beam": beam, "fibre": fibres},
        "basis": basis_info, "target": target_info,
        "saved_samples": [samples[i]["sample_id"] for i in saved], "results": results,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "frames.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    ids = _ids(samples, range(len(samples)))
    np.savez_compressed(
        out_dir / "weights.npz", fits=np.array(list(weights)), positions=basis.positions, coverage=coverage,
        **ids, **{"weights/" + fit: w.astype(np.float32) for fit, w in weights.items()},
    )
    np.savez_compressed(out_dir / "samples.npz", **_sample_images(saved, samples, beams, real, basis, weights, masks, axes))

    for fibre, r in results.items():
        log("%s (%s): %d of %d frames covered; floor %.4f, camera noise %.4f (shape_rel_l2)"
            % (fibre, r["camera"], r["covered"], r["frames"], r["floor"]["shape_rel_l2"], r["camera_noise"]))
        for fit, q in r["fits"].items():
            if q["shape_rel_l2"] is not None:
                med = q["shape_rel_l2"]["median"]
                log("  %-10s shape_rel_l2 %.4f (%.2f x camera noise), pearson %.4f, ssim %.4f"
                    % (fit, med, med / r["camera_noise"], q["pearson"]["median"], q["ssim"]["median"]))
    log("written: %s (images of %d frames saved)" % (out_dir, len(saved)))
    return summary


def _each(fit, beams, name, log):
    out, start, last = [], time.time(), time.time()
    for i, b in enumerate(beams):
        out.append(fit(b))
        if time.time() - last > 15:
            last = time.time()
            log("%s: %d of %d frames, %.0f s" % (name, i + 1, len(beams), last - start))
    log("%s: %d frames, %.0f s" % (name, len(beams), time.time() - start))
    return np.stack(out)


def _ids(samples, idx):
    return {
        "run_id": np.array([str(samples[i]["run_id"]) for i in idx]),
        "sample_id": np.array([samples[i]["sample_id"] for i in idx]),
        "time_ns": np.array([samples[i]["time_ns"] for i in idx], dtype=np.int64),
    }


def _sample_images(idx, samples, beams, real, basis, weights, masks, axes):
    """Everything a figure needs of the frames ``idx`` (keys in the module text)."""

    def stack(images):
        return np.stack(list(images)).astype(np.float32) if idx else np.zeros((0,), np.float32)

    out = {
        "fits": np.array(list(weights)), "fibres": np.array(list(real)), **_ids(samples, idx),
        "beam": stack(beams[i] for i in idx),
        "beam_centre": np.array([beam_centre(beams[i]) for i in idx]).reshape(len(idx), 2),
    }
    for fit, w in weights.items():
        out["light/" + fit] = stack(basis.light(w[i]) for i in idx)
    for fibre, frames in real.items():
        out["measured/" + fibre] = stack(frames[i] for i in idx)
        out["mask/" + fibre], out["axis/" + fibre] = masks[fibre], axes[fibre]
        for fit, w in weights.items():
            out["synthetic/%s/%s" % (fibre, fit)] = stack(basis.synthesize(w[i], fibre) for i in idx)
    return out


def _quartiles(values):
    if not values:
        return None
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    return {"median": float(median), "q25": float(q25), "q75": float(q75)}
