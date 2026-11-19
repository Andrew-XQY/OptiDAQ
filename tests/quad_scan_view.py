"""Plot beam size and a thin-lens quadrupole-scan reconstruction.

The viewer reads magnets.samples.jsonl and saved image streams in a separate
process. It does not issue hardware commands. It requires a display and
matplotlib; the numerical helpers can be imported without a display.

Run with a config that defines quad_scan_view and a completed or active run:

    python tests/quad_scan_view.py --config configs/experiment_v1.json --run RUN_DIR

The fit uses sigma^2(u) = a*u^2 + b*u + c, with u = 1/f and drift length L.
Pixel size and optics are needed to reconstruct the beam matrix and emittance;
without them the viewer plots beam size versus current.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from optidaq.daq.raw_stream import (  # noqa: E402
    iter_stream_records,
    load_sidecar_spec,
)


# --- pure helpers (no matplotlib) -------------------------------------------

def beam_sigma_px(image: np.ndarray) -> Tuple[float, float]:
    """RMS beam size (sigma_x, sigma_y) in pixels via intensity moments.

    A flat background is removed with the image median and negatives clipped, so
    a roughly uniform pedestal does not inflate the width.
    """
    a = np.asarray(image, dtype=np.float64)
    if a.ndim == 3:
        a = a.mean(axis=2)
    a = a - np.median(a)
    a[a < 0] = 0.0
    total = a.sum()
    if total <= 0:
        return float("nan"), float("nan")
    ys, xs = np.indices(a.shape)
    xbar = float((a * xs).sum() / total)
    ybar = float((a * ys).sum() / total)
    sx = float(np.sqrt(max((a * (xs - xbar) ** 2).sum() / total, 0.0)))
    sy = float(np.sqrt(max((a * (ys - ybar) ** 2).sum() / total, 0.0)))
    return sx, sy


def quad_u(current_a: float, optics: Mapping[str, Any]) -> Optional[float]:
    """Thin-lens inverse focal length u = 1/f (1/m) from magnet current.

    u = (g / Brho) * L_quad, with g = grad_T_per_m_per_A * current. Returns None
    when the optics needed for reconstruction are not fully configured.
    """
    grad = float(optics.get("grad_T_per_m_per_A", 0.0) or 0.0)
    brho = float(optics.get("brho_T_m", 0.0) or 0.0)
    lq = float(optics.get("quad_length_m", 0.0) or 0.0)
    if grad == 0.0 or brho == 0.0 or lq == 0.0:
        return None
    return (grad * float(current_a) / brho) * lq


def reconstruct(u: np.ndarray, sigma_sq_m2: np.ndarray, drift_m: float) -> dict:
    """Quad-scan reconstruction at the quad (thin-lens + drift L).

    Fits sigma^2(u) = a u^2 + b u + c and inverts to the beam matrix
    s11=<x^2>, s12=<xx'>, s22=<x'^2>, then emittance/Twiss.
    """
    u = np.asarray(u, dtype=np.float64)
    y = np.asarray(sigma_sq_m2, dtype=np.float64)
    if u.size < 3:
        raise ValueError("need >= 3 points to fit a parabola")
    a, b, c = np.polyfit(u, y, 2)
    L = float(drift_m)
    s11 = a / L ** 2
    s12 = (-b - 2.0 * L * s11) / (2.0 * L ** 2)
    s22 = (c - s11 - 2.0 * L * s12) / L ** 2
    det = s11 * s22 - s12 ** 2
    eps = float(np.sqrt(det)) if det > 0 else float("nan")
    beta = s11 / eps if eps and np.isfinite(eps) and eps > 0 else float("nan")
    alpha = -s12 / eps if eps and np.isfinite(eps) and eps > 0 else float("nan")
    gamma = s22 / eps if eps and np.isfinite(eps) and eps > 0 else float("nan")
    return {
        "a": float(a), "b": float(b), "c": float(c),
        "s11": float(s11), "s12": float(s12), "s22": float(s22),
        "emittance": eps, "beta": beta, "alpha": alpha, "gamma": gamma,
    }


# --- read-only run-directory access -----------------------------------------

def latest_current(run_dir: Path, quad: str) -> Optional[float]:
    """currentAverage of ``quad`` from the last line of magnets.samples.jsonl."""
    path = run_dir / "magnets.samples.jsonl"
    if not path.exists():
        return None
    last = None
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                last = line
    if last is None:
        return None
    try:
        magnets = json.loads(last).get("magnets", {})
        return float(magnets[quad]["currentAverage"])
    except (ValueError, KeyError, TypeError):
        return None


def _resolved_config(run_dir: Path) -> dict:
    path = run_dir / "resolved_config.json"
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_image(
    run_dir: Path,
    camera: str,
    resolved: Mapping[str, Any],
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Newest fully-written image for ``camera`` as (array, frame_id), or (None, None).

    Supports the raw stream (via the spec sidecar) and png output. Read-only.
    """
    fmt = str(resolved.get("output_format", "raw"))
    cam_meta = (resolved.get("cameras") or {}).get(camera, {})
    shape = tuple(int(p) for p in cam_meta.get("shape", ()))
    dtype = str(cam_meta.get("dtype", "uint16"))

    if fmt == "png":
        manifest_path = run_dir / (camera + ".png.json")
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            if manifest.get("transforms"):
                raise RuntimeError(
                    "Quad reconstruction requires unprocessed images: PNG hooks "
                    "can change pixel calibration or the beam profile. Use empty "
                    "output.png.transforms for this viewer."
                )
        files = sorted((run_dir / camera).glob("*.png"))
        if not files:
            return None, None
        newest = files[-1]
        frame_id = int(newest.stem.split("_")[-1])
        try:
            from PIL import Image
            return np.asarray(Image.open(newest)), frame_id
        except Exception:
            return None, frame_id

    # raw: iterate the stream and keep the last complete record
    try:
        spec = load_sidecar_spec(run_dir, camera)
    except Exception:
        return None, None
    last = None
    for record in iter_stream_records(run_dir, spec):
        last = record
    if last is None or not shape:
        return None, None
    arr = np.frombuffer(last.payload, dtype=np.dtype(dtype)).reshape(shape)
    return arr, int(last.header.get("frame_index", 0))


# --- live viewer -------------------------------------------------------------

class QuadScanView:
    """Two-panel live figure: sigma^2 vs strength parabola, and phase ellipse."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        view = cfg.get("quad_scan_view", {})
        self.quad = str(view.get("quad", "qd"))
        self.plane = str(view.get("plane", "x")).lower()
        self.pixel_size_m = float(view.get("pixel_size_m", 0.0) or 0.0)
        self.optics = dict(view.get("optics", {}) or {})
        self.drift_m = float(self.optics.get("drift_m", 0.0) or 0.0)
        self._currents: list[float] = []
        self._sigmas_m: list[float] = []      # metres if pixel size set, else pixels
        self.fig = None
        self._ax_curve = None
        self._ax_ellipse = None

    def _ensure_fig(self) -> None:
        if self.fig is not None:
            return
        import matplotlib.pyplot as plt
        plt.ion()
        self.fig, (self._ax_curve, self._ax_ellipse) = plt.subplots(
            1, 2, figsize=(11, 4.6)
        )
        self.fig.canvas.manager.set_window_title("OptiDAQ quad scan - %s" % self.quad)

    def add(self, current_a: float, sigma_px: float) -> None:
        sigma = sigma_px * self.pixel_size_m if self.pixel_size_m else sigma_px
        self._currents.append(float(current_a))
        self._sigmas_m.append(float(sigma))

    def refresh(self) -> None:
        self._ensure_fig()
        cur = np.asarray(self._currents, dtype=np.float64)
        sig = np.asarray(self._sigmas_m, dtype=np.float64)
        unit = "m" if self.pixel_size_m else "px"

        self._ax_curve.clear()
        self._ax_curve.scatter(cur, sig ** 2, s=18, color="#1f77b4", label="measured")
        self._ax_curve.set_xlabel("%s current (A)" % self.quad)
        self._ax_curve.set_ylabel("sigma_%s^2 (%s^2)" % (self.plane, unit))
        self._ax_curve.set_title("Quad scan: sigma^2 vs current")
        self._ax_curve.grid(True, alpha=0.3)

        rec = None
        us = [quad_u(c, self.optics) for c in self._currents]
        have_optics = self.pixel_size_m > 0 and all(u is not None for u in us)
        if cur.size >= 3:
            xfit = np.asarray(us, dtype=np.float64) if have_optics else cur
            try:
                coeffs = np.polyfit(xfit, sig ** 2, 2)
                order = np.argsort(cur)
                xs = np.linspace(cur.min(), cur.max(), 200)
                xs_fit = (
                    np.asarray([quad_u(x, self.optics) for x in xs], dtype=np.float64)
                    if have_optics else xs
                )
                self._ax_curve.plot(
                    xs, np.polyval(coeffs, xs_fit), "-", color="#d62728", label="fit"
                )
                if have_optics and self.drift_m > 0:
                    rec = reconstruct(xfit, sig ** 2, self.drift_m)
            except Exception:
                rec = None
        self._ax_curve.legend(loc="upper center", fontsize=8)

        self._draw_ellipse(rec)
        title = "n=%d" % cur.size
        if rec and np.isfinite(rec["emittance"]):
            title += "   eps=%.3e m.rad   beta=%.2f m   alpha=%.2f" % (
                rec["emittance"], rec["beta"], rec["alpha"],
            )
        self.fig.suptitle(title, fontsize=10)
        self.fig.tight_layout(rect=(0, 0, 1, 0.95))
        self.fig.canvas.draw_idle()

    def _draw_ellipse(self, rec: Optional[dict]) -> None:
        ax = self._ax_ellipse
        ax.clear()
        ax.set_title("Phase-space ellipse (at quad)")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("x' (mrad)")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.6)
        if not rec or not np.isfinite(rec.get("emittance", float("nan"))):
            ax.text(0.5, 0.5, "set pixel_size_m + optics\nto reconstruct",
                    ha="center", va="center", transform=ax.transAxes, color="0.4")
            return
        sigma = np.array([[rec["s11"], rec["s12"]], [rec["s12"], rec["s22"]]])
        w, V = np.linalg.eigh(sigma)
        w = np.clip(w, 0, None)
        t = np.linspace(0, 2 * np.pi, 256)
        pts = V @ (np.sqrt(w)[:, None] * np.vstack([np.cos(t), np.sin(t)]))
        x_mm = pts[0] * 1e3
        xp_mrad = pts[1] * 1e3
        ax.plot(x_mm, xp_mrad, "-", color="#1f77b4", lw=1.8)
        sx = np.sqrt(rec["s11"]) * 1e3      # mm
        sxp = np.sqrt(rec["s22"]) * 1e3     # mrad
        for x in (sx, -sx):
            ax.axvline(x, ls="--", color="#d62728", lw=0.9)
        for y in (sxp, -sxp):
            ax.axhline(y, ls="--", color="#d62728", lw=0.9)
        ax.set_xlim(-1.6 * sx, 1.6 * sx)
        ax.set_ylim(-1.6 * sxp, 1.6 * sxp)
        ax.text(0.02, 0.98, "sigma_x=%.3f mm\nsigma_x'=%.3f mrad" % (sx, sxp),
                transform=ax.transAxes, va="top", fontsize=8, color="#d62728")


def run_live(cfg: Mapping[str, Any], run_dir: Path) -> None:
    import matplotlib.pyplot as plt
    view_cfg = cfg.get("quad_scan_view", {})
    camera = str(view_cfg.get("camera", "cam1"))
    quad = str(view_cfg.get("quad", "qd"))
    plane = str(view_cfg.get("plane", "x")).lower()
    update_s = float(view_cfg.get("update_s", 0.5) or 0.5)

    view = QuadScanView(cfg)
    resolved: Optional[dict] = None
    last_frame_id: Optional[int] = None
    print("quad scan view: run_dir=%s camera=%s quad=%s" % (run_dir, camera, quad))

    while True:
        if view.fig is not None and not plt.fignum_exists(view.fig.number):
            break
        if resolved is None and (run_dir / "resolved_config.json").exists():
            resolved = _resolved_config(run_dir)
        if resolved is not None:
            current = latest_current(run_dir, quad)
            image, frame_id = latest_image(run_dir, camera, resolved)
            if current is not None and image is not None and frame_id != last_frame_id:
                sx, sy = beam_sigma_px(image)
                view.add(current, sy if plane == "y" else sx)
                view.refresh()
                last_frame_id = frame_id
        plt.pause(max(0.05, update_s))


def main() -> None:
    parser = argparse.ArgumentParser(description="Live quad-scan viewer (decoupled).")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True, help="run directory")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    if not cfg.get("quad_scan_view", {}).get("enabled", False):
        print("quad_scan_view.enabled is false; nothing to show")
        return
    run_live(cfg, args.run)


if __name__ == "__main__":
    main()
