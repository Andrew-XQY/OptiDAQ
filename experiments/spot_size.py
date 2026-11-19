"""Live Gaussian and mask-based spot sizing on an OptiDAQ camera.

Use Cam2 for the spot on the input screen. Block the spot and click Capture
background, then unblock it. Enter the screen calibration in mm per acquired
pixel. Exposure and gain changes apply temporarily to the selected camera.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import signal
import sys
import time

import numpy as np

import camera_preview as preview


DEFAULT_CONFIG = preview.REPO_ROOT / "configs" / "experiment_lab_motor_raster.json"
CONTROL_KEYS = ("cameraExposureTimeUs", "cameraGain", "cameraExposureAuto", "cameraGainAuto")
AUTO_KEYS = {"cameraExposureTimeUs": "cameraExposureAuto", "cameraGain": "cameraGainAuto"}
FWHM_PER_SIGMA = math.sqrt(8 * math.log(2))
READOUT_FONTS = ("DejaVu Sans", "Liberation Sans", "Helvetica", "Liberation Mono")
# Samples of standard palettes, interpolated for display without extra dependencies.
PALETTE_SAMPLES = {
    "viridis": "440154 48186a 472d7b 424086 3b528b 33638d 2c728e 26828e 21918c 1f9f88 27ad81 3dbc74 5cc863 81d34d aadc32 d5e21a fde725",
    "magma": "000004 0a0822 1d1147 36106b 51127c 6a1c81 832681 9c2e7f b73779 cf4070 e55064 f4695c fb8761 fea571 fec287 fde0a1 fcfdbf",
    "cividis": "00224e 002e6a 1a386f 32436d 434e6c 535a6d 61656f 6f7073 7d7c78 8b8778 9a9376 aaa073 bbad6d ccba64 ddc858 efd748 fee838",
}


def display_palette(name: str) -> np.ndarray | None:
    if name == "gray":
        return None
    colors = np.frombuffer(bytes.fromhex(PALETTE_SAMPLES[name]), dtype=np.uint8).reshape(-1, 3)
    sample = np.rint(np.linspace(0, 255, len(colors)))
    return np.rint(np.column_stack([
        np.interp(np.arange(256), sample, channel) for channel in colors.T
    ])).astype(np.uint8)


def bold_fonts(root, tk_font):
    """Select a face the running Tk can actually render at both readout sizes."""
    families = {family.casefold(): family for family in tk_font.families(root)}
    for name in READOUT_FONTS:
        family = families.get(name.casefold())
        if family is None:
            continue
        panel = tk_font.Font(root, family=family, size=-16, weight="bold")
        readout = tk_font.Font(root, family=family, size=-22, weight="bold")
        if (panel.actual("weight") == readout.actual("weight") == "bold"
                and readout.metrics("linespace") > panel.metrics("linespace")):
            return panel, readout
    panel = tk_font.nametofont("TkDefaultFont", root=root).copy()
    readout = panel.copy()
    panel.configure(size=-16, weight="bold")
    readout.configure(size=-22, weight="bold")
    return panel, readout


def fit_display(image: np.ndarray, viewport: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center an aspect-preserving image in a fixed (height, width) canvas."""
    shape = np.asarray(image.shape)
    factor = min(1.0, min(np.asarray(viewport) / shape))
    height, width = np.maximum(1, np.rint(shape * factor).astype(int))
    top, left = ((np.asarray(viewport) - (height, width)) // 2)
    display = np.zeros(viewport, dtype=np.uint8)
    display[top:top + height, left:left + width] = preview.resize_nearest_to_shape(image, (height, width))
    return display, np.array([width, height]) / shape[::-1], np.array([left, top])


@dataclass(frozen=True)
class Gaussian:
    center: np.ndarray
    sigma: np.ndarray
    axes: np.ndarray
    residual: float

    @property
    def fwhm(self) -> np.ndarray:
        return self.sigma * FWHM_PER_SIGMA


@dataclass(frozen=True)
class Spot:
    center: np.ndarray
    box: tuple[int, int, int, int]
    pixels: int
    threshold: float
    mask_sigma: np.ndarray
    gaussian: Gaussian | None
    notes: tuple[str, ...]


def measurement_text(spot: Spot | None, scale: float | None) -> str:
    def widths(values):
        if values is None:
            return "— px  (— mm)"
        a, b = values
        mm = "—" if scale is None else f"{a * scale:.3g} × {b * scale:.3g}"
        return f"{a:.2f} × {b:.2f} px  ({mm} mm)"

    gaussian = spot.gaussian if spot is not None else None
    footprint = None
    count = "—"
    if spot is not None:
        x0, y0, x1, y1 = spot.box
        footprint = x1 - x0, y1 - y0
        count = str(spot.pixels)
    return "\n".join((
        f"Gaussian FWHM   {widths(gaussian.fwhm if gaussian is not None else None)}",
        f"Gaussian σ   {widths(gaussian.sigma if gaussian is not None else None)}",
        f"Mask footprint   {widths(footprint)}  ·  {count} pixels",
        f"Mask RMS σ   {widths(spot.mask_sigma if spot is not None else None)}",
    ))


def component_pixels(mask: np.ndarray, seed_y: int, seed_x: int) -> tuple[np.ndarray, np.ndarray]:
    """Find the spot's 8-connected component by following contiguous row spans."""
    rows: dict[int, np.ndarray] = {}
    seen = set()
    pending = deque([(seed_y, seed_x, seed_x)])
    xs, ys = [], []
    while pending:
        row, lo, hi = pending.popleft()
        if row < 0 or row >= mask.shape[0]:
            continue
        if row not in rows:
            changes = np.flatnonzero(np.diff(mask[row], prepend=False, append=False))
            rows[row] = changes.reshape(-1, 2)
        for left, stop in rows[row]:
            if stop <= lo or left > hi or (row, left) in seen:
                continue
            seen.add((row, left))
            xs.append(np.arange(left, stop))
            ys.append(np.full(stop - left, row))
            pending.extend(((row - 1, left - 1, stop), (row + 1, left - 1, stop)))
    return np.concatenate(xs), np.concatenate(ys)


def measure_spot(image: np.ndarray, full_scale: int, background: np.ndarray | None = None) -> Spot:
    """Measure the strongest neighboring-pixel spot and fit its unsaturated samples."""
    raw = np.asarray(image)
    if raw.ndim != 2 or min(raw.shape) < 5:
        raise ValueError("Expected a two-dimensional monochrome frame.")
    if background is not None and background.shape != raw.shape:
        raise ValueError("Frame dimensions changed. Recapture the background.")
    values = raw.astype(np.float32)
    if background is not None:
        values -= background

    border = np.concatenate((values[0], values[-1], values[1:-1, 0], values[1:-1, -1]))
    offset = np.median(border)
    noise = max(1.0, 1.4826 * np.median(np.abs(border - offset)))
    values -= offset
    pad = np.pad(values, 1, constant_values=-np.inf)
    height, width = raw.shape
    neighbor = np.full(raw.shape, -np.inf, dtype=values.dtype)
    for dy, dx in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)):
        np.maximum(neighbor, pad[dy:dy + height, dx:dx + width], out=neighbor)
    seed = np.unravel_index(np.argmax(np.where(neighbor >= 5 * noise, values, -np.inf)), raw.shape)
    peak = values[seed]
    if peak < 8 * noise or neighbor[seed] < 5 * noise:
        raise ValueError("No convincing spot. Need at least two neighboring bright pixels.")
    threshold = max(5 * noise, min(0.10 * peak, float(neighbor[seed])))
    x, y = component_pixels(values >= threshold, *seed)
    points = np.column_stack((x, y)).astype(np.float64)
    signal_values = values[y, x].astype(np.float64)
    box = (int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1)
    center = np.average(points, axis=0, weights=signal_values)
    centered = points - center
    covariance = (centered.T * signal_values) @ centered / signal_values.sum()
    # Include the variance of a uniformly illuminated pixel in the mask width.
    mask_sigma = np.sqrt(np.maximum(np.linalg.eigvalsh(covariance)[::-1], 0) + 1 / 12)
    clipped = raw[y, x] >= full_scale
    notes = []
    if clipped.any():
        notes.append(f"{clipped.sum()} saturated pixels. Reduce exposure or gain for a more reliable width.")
    gaussian = None
    try:
        gaussian = fit_gaussian(points[~clipped], signal_values[~clipped], box)
    except (ValueError, np.linalg.LinAlgError) as error:
        notes.append(str(error))
    if gaussian is not None and clipped.any():
        notes.append("Gaussian uses unsaturated wings. Mask widths use the clipped image.")
    if min(box[2] - box[0], box[3] - box[1]) <= 2 or (gaussian is not None and gaussian.sigma[1] < 0.75):
        notes.append("Undersampled spot. Treat subpixel widths as approximate.")
    if gaussian is not None:
        extent = 2 * np.sqrt((gaussian.axes**2) @ (gaussian.sigma**2))
        near_edge = np.any(gaussian.center - extent < 0) or np.any(
            gaussian.center + extent > np.array([width, height]) - 1
        )
    else:
        near_edge = False
    if near_edge or box[0] == 0 or box[1] == 0 or box[2] == width or box[3] == height:
        notes.append("Spot touches the image edge. Widths may be biased. Recenter or widen the ROI.")
    return Spot(center, box, len(points), threshold, mask_sigma, gaussian, tuple(notes))


def fit_gaussian(points: np.ndarray, signal_values: np.ndarray, box: tuple[int, int, int, int]) -> Gaussian:
    if len(points) < 8:
        raise ValueError("Too few unsaturated pixels for a Gaussian fit. Mask widths are shown.")
    origin = points.mean(axis=0)
    scale = max(float(np.ptp(points, axis=0).max()) / 2, 1.0)
    x_fit, y_fit = ((points - origin) / scale).T
    design = np.column_stack((np.ones_like(x_fit), x_fit, y_fit, x_fit**2, x_fit * y_fit, y_fit**2))
    weights = np.sqrt(signal_values / signal_values.max())
    coeff, _, rank, _ = np.linalg.lstsq(design * weights[:, None], np.log(signal_values) * weights, rcond=None)
    precision = -np.array([[2 * coeff[3], coeff[4]], [coeff[4], 2 * coeff[5]]])
    eigenvalues, axes = np.linalg.eigh(precision)
    if rank != 6 or np.any(eigenvalues <= 0):
        raise ValueError("Gaussian shape could not be resolved. Mask widths are shown.")
    center = origin + scale * np.linalg.solve(precision, coeff[1:3])
    sigma = scale / np.sqrt(eigenvalues)
    if sigma[0] > max(box[2] - box[0], box[3] - box[1]):
        raise ValueError("Gaussian width is unconstrained by the spot mask. Mask widths are shown.")
    if np.any(center < np.array(box[:2]) - 0.5) or np.any(center > np.array(box[2:]) - 0.5):
        raise ValueError("Gaussian centre falls outside the spot mask. Mask widths are shown.")
    with np.errstate(over="ignore"):
        fitted = np.exp(design @ coeff)
    residual = np.linalg.norm(fitted - signal_values) / np.linalg.norm(signal_values)
    if not np.isfinite(residual) or residual > 0.25:
        raise ValueError("Poor Gaussian fit. Mask widths are shown. Check the spot shape and background.")
    return Gaussian(center, sigma, axes, float(residual))


def read_controls(client, device: str) -> tuple[dict, dict]:
    requested = preview.get_value(client, f"{device}/CameraSetting")
    applied = preview.get_value(client, f"{device}/AcquiredParameters")
    return requested, applied


def controls_match(readback: dict, values: dict, *, applied: bool) -> bool:
    for key, value in values.items():
        # A camera in auto mode chooses its own applied exposure or gain.
        auto = AUTO_KEYS.get(key)
        if applied and auto and values[auto] != preview.CAMERA_AUTO_OFF_VALUE:
            continue
        name = "acq" + key[0].upper() + key[1:] if applied else key
        actual = readback.get(name)
        tolerance = preview.GAIN_READBACK_TOLERANCE if key == "cameraGain" else 0
        if actual is None or not math.isclose(float(actual), float(value), rel_tol=0, abs_tol=tolerance):
            return False
    return True


def set_controls(client, device: str, values: dict) -> None:
    preview.set_value(client, f"{device}/CameraSetting", preview.camera_setting_payload(values))
    deadline = time.monotonic() + preview.CAMERA_READBACK_TIMEOUT_S
    while True:
        requested, applied = read_controls(client, device)
        if controls_match(requested, values, applied=False) and controls_match(applied, values, applied=True):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("Camera settings did not reach the requested readback.")
        time.sleep(preview.CAMERA_READBACK_POLL_S)


class SpotPreview:
    def __init__(self, cfg: dict, camera: str | None) -> None:
        from tkinter import font, ttk

        self.cfg = cfg
        self.reader = preview.make_client(require_rbac=False)
        self.writer = None
        self.original: dict[str, dict] = {}
        self.taps = {}
        for tap in preview.build_camera_taps(cfg, self.reader, allow_image_number_reset=True):
            try:
                status = preview.get_value(self.reader, f"{tap.fesa_device}/AcquisitionStatus")
                if int(status["acqCameraStatus"]) != 0:
                    raise RuntimeError(f"camera status {status['acqCameraStatus']}")
            except Exception as error:
                print(f"{tap.device} unavailable: {error}", file=sys.stderr)
            else:
                self.taps[tap.device] = tap
        if not self.taps:
            raise RuntimeError("No configured cameras report ready.")
        if camera is not None and camera not in self.taps:
            raise RuntimeError(f"{camera} unavailable. Ready cameras: {', '.join(self.taps)}")

        self.tap = None
        self.latest = None
        self.last_frame = None
        self.background = None
        self.background_requested_at = None
        self.spot = None
        self.fit_message = ""
        self.viewer = preview.TkPreview("OptiDAQ spot size")
        root = self.viewer.root
        tk = preview.tk
        self.alias = tk.StringVar(value=camera or ("cam2" if "cam2" in self.taps else next(iter(self.taps))))
        self.exposure = tk.StringVar()
        self.gain = tk.StringVar()
        self.scale = tk.StringVar()
        self.cmap = tk.StringVar(value="viridis")
        self.palette = display_palette("viridis")
        self.display = None
        self.meta = tk.StringVar(value="Waiting for a camera frame…")
        self.result = tk.StringVar(value=measurement_text(None, None))
        self.fit_status = tk.StringVar()
        self.status = tk.StringVar()
        self.panel_font, self.result_font = bold_fonts(root, font)
        styles = ttk.Style(root)
        for widget in ("TLabel", "TEntry", "TButton", "TCombobox"):
            styles.configure(f"Spot.{widget}", font=self.panel_font)
        styles.configure("Spot.TLabel", foreground="#13181f")
        styles.configure("SpotResult.TLabel", font=self.result_font, foreground="#13181f")
        root.option_add("*TCombobox*Listbox.font", self.panel_font)
        panel = ttk.Frame(root, padding=8)
        panel.pack(before=self.viewer.canvas, fill="x")
        ttk.Label(panel, text="Camera", style="Spot.TLabel").grid(row=0, column=0, sticky="w")
        menu = ttk.Combobox(panel, textvariable=self.alias, state="readonly", width=7,
                            values=list(self.taps), style="Spot.TCombobox", font=self.panel_font)
        menu.grid(row=1, column=0, sticky="w", padx=(0, 8))
        menu.bind("<<ComboboxSelected>>", lambda _: self.command(self.select))
        for col, (text, value) in enumerate(
            (("Exposure (ms)", self.exposure), ("Gain (0–1)", self.gain), ("mm/pixel", self.scale)), 1
        ):
            ttk.Label(panel, text=text, style="Spot.TLabel").grid(row=0, column=col, sticky="w")
            ttk.Entry(panel, textvariable=value, width=10, style="Spot.TEntry",
                      font=self.panel_font).grid(row=1, column=col, padx=(0, 8), sticky="w")
        ttk.Button(panel, text="Apply", command=lambda: self.command(self.apply), style="Spot.TButton",
                   state="normal" if cfg["rbac"]["enabled"] else "disabled").grid(row=1, column=4, sticky="w")
        ttk.Label(panel, text="Colormap", style="Spot.TLabel").grid(row=0, column=5, sticky="w", padx=(8, 0))
        colormaps = ttk.Combobox(panel, textvariable=self.cmap, state="readonly", width=8,
                                values=(*PALETTE_SAMPLES, "gray"), style="Spot.TCombobox", font=self.panel_font)
        colormaps.grid(row=1, column=5, sticky="w", padx=(8, 0))
        colormaps.bind("<<ComboboxSelected>>", lambda _: self.set_palette())
        ttk.Button(panel, text="Capture background", command=self.request_background, style="Spot.TButton").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        ttk.Button(panel, text="Clear background", command=self.clear_background, style="Spot.TButton").grid(
            row=2, column=2, columnspan=2, sticky="w", pady=(6, 0)
        )
        ttk.Button(panel, text="Reconnect", command=lambda: self.command(self.select), style="Spot.TButton").grid(
            row=2, column=4, columnspan=2, sticky="w", pady=(6, 0)
        )
        root.update_idletasks()
        panel_width = max(panel.winfo_reqwidth() - 16, min(884, root.winfo_screenwidth() - 76))
        for row, variable, lines, style, text_font in (
            (3, self.meta, 2, "Spot.TLabel", self.panel_font),
            (4, self.result, 4, "SpotResult.TLabel", self.result_font),
            (5, self.fit_status, 4, "Spot.TLabel", self.panel_font),
            (6, self.status, 2, "Spot.TLabel", self.panel_font),
        ):
            slot = ttk.Frame(panel, width=panel_width, height=lines * text_font.metrics("linespace") + 4)
            slot.grid(row=row, column=0, columnspan=6, sticky="w")
            slot.pack_propagate(False)
            ttk.Label(slot, textvariable=variable, style=style, wraplength=panel_width,
                      anchor="nw", justify="left").pack(fill="both", expand=True)
        root.update_idletasks()
        view_height = min(760, max(120, root.winfo_screenheight() - panel.winfo_reqheight() - 100))
        view_width = panel.winfo_reqwidth()
        self.viewport = (view_height, view_width)
        self.viewer.canvas.configure(width=view_width, height=view_height, background="#161b22")
        self.scale.trace_add("write", lambda *_: self.show_result())

    def command(self, action) -> None:
        try:
            action()
        except Exception as error:
            self.status.set(str(error))
            print(f"Spot preview: {error}", file=sys.stderr)

    def stop_stream(self) -> None:
        if self.tap is None:
            return
        self.tap.stop()
        self.tap.join(timeout=3)
        if self.tap._thread is not None and self.tap._thread.is_alive():
            raise RuntimeError(f"{self.tap.device} camera stream did not stop.")
        self.tap = None

    def restore_controls(self, device: str) -> None:
        if device in self.original:
            set_controls(self.writer, device, self.original[device])
            del self.original[device]

    def select(self) -> None:
        name = self.alias.get()
        try:
            self.stop_stream()
        except Exception:
            self.alias.set(self.tap.device)
            raise
        for device in list(self.original):
            self.restore_controls(device)
        self.clear_background()
        self.scale.set("")
        self.connect(name)

    def connect(self, name: str) -> None:
        tap = self.taps[name]
        self.display = None
        self.viewer.canvas.delete("all")
        requested, applied = read_controls(self.reader, tap.fesa_device)
        self.exposure.set(f"{int(applied['acqCameraExposureTimeUs']) / 1000:g}")
        self.gain.set(f"{float(applied['acqCameraGain']):g}")
        settings = dict(requested)
        for key in ("CameraPixelFormat", "CameraBinningMode", "CameraBinningHor", "CameraBinningVer"):
            applied_key = "acq" + key
            if applied_key in applied:
                settings[key[0].lower() + key[1:]] = applied[applied_key]
        self.full_scale = preview.camera_full_scale(settings)
        self.applied_exposure_us = int(applied["acqCameraExposureTimeUs"])
        self.applied_gain = float(applied["acqCameraGain"])
        self.manual_controls = all(
            int(applied["acq" + key[0].upper() + key[1:]]) == preview.CAMERA_AUTO_OFF_VALUE
            for key in AUTO_KEYS.values()
        )
        self.last_frame = None
        self.spot = None
        self.fit_message = ""
        self.meta.set(f"{name} ({tap.fesa_device}) · waiting for a fresh frame")
        self.show_result()
        tap.probe()
        self.latest = preview.LatestFrames([tap])
        self.tap = tap
        tap.start(self.latest.emit)
        self.status.set("Background estimated from image border. Block the spot before capturing a background.")

    def apply(self) -> None:
        if not self.cfg["rbac"]["enabled"] or self.tap is None:
            return
        alias, device = self.tap.device, self.tap.fesa_device
        if alias != self.alias.get():
            raise RuntimeError("Reconnect the selected camera before applying settings.")
        exposure = round(float(self.exposure.get()) * 1000)
        gain = float(self.gain.get())
        lo, hi = preview.read_camera_exposure_limits(self.reader, device)
        if not lo <= exposure <= hi or not 0 <= gain <= 1:
            raise ValueError(f"Use exposure {lo / 1000:g}–{hi / 1000:g} ms and gain 0–1.")
        if self.writer is None:
            from optidaq.extensions.lab_camera import make_lab_client

            self.writer = make_lab_client(require_rbac=True)
        requested, _ = read_controls(self.writer, device)
        self.stop_stream()
        self.clear_background()
        self.original.setdefault(device, {key: requested[key] for key in CONTROL_KEYS})
        changes = dict(zip(CONTROL_KEYS, (exposure, gain, preview.CAMERA_AUTO_OFF_VALUE, preview.CAMERA_AUTO_OFF_VALUE)))
        try:
            set_controls(self.writer, device, changes)
            self.connect(alias)
        except Exception as error:
            try:
                self.stop_stream()
            finally:
                self.restore_controls(device)
            try:
                self.connect(alias)
            except Exception as reconnect_error:
                raise RuntimeError(f"{error} Reconnect failed: {reconnect_error}") from error
            raise

    def request_background(self) -> None:
        if self.tap is not None and not self.manual_controls:
            self.status.set("Apply fixed exposure and gain before capturing a background.")
        elif self.tap is not None:
            self.background_requested_at = time.monotonic()
            self.status.set("Waiting for the next frame with the spot blocked…")

    def clear_background(self) -> None:
        self.background = None
        self.background_requested_at = None
        self.spot = None
        self.fit_message = "Waiting for the next frame."
        self.viewer.canvas.delete("spot")
        self.show_result()
        self.status.set("Background estimated from image border.")

    def show_result(self) -> None:
        spot = self.spot
        scale = None
        notes = [self.fit_message] if spot is None else list(spot.notes)
        calibration = self.scale.get().strip()
        if calibration:
            try:
                scale = float(calibration)
                if not math.isfinite(scale) or scale <= 0:
                    raise ValueError
            except ValueError:
                scale = None
                notes.append("Enter a positive mm/pixel value.")

        self.result.set(measurement_text(spot, scale))
        if spot is not None:
            notes.append(f"Mask threshold {spot.threshold:.1f} counts above background. "
                         "Footprint is X × Y. Gaussian and RMS widths are major × minor.")
        self.fit_status.set(" ".join(notes))

    def draw_frame(self, frame, received_at: float) -> None:
        raw = preview.frame_to_image(frame, self.tap)
        if self.background_requested_at is not None and received_at >= self.background_requested_at:
            self.background = raw.copy()
            self.background_requested_at = None
            self.status.set("Background captured. Unblock the spot to measure.")
        if self.background is not None and self.background.shape != raw.shape:
            self.clear_background()
        self.spot = None
        try:
            self.spot = measure_spot(raw, self.full_scale, self.background)
        except (ValueError, np.linalg.LinAlgError) as error:
            self.fit_message = str(error)
        self.show_result()
        bg_label = "captured background" if self.background is not None else "border baseline"
        settings = (f"{self.applied_exposure_us / 1000:g} ms · gain {self.applied_gain:g}"
                    if self.manual_controls else "auto exposure or gain")
        self.meta.set(f"{self.tap.device} ({self.tap.fesa_device}) · frame {frame.source_image_number} · "
                      f"{raw.shape[1]} × {raw.shape[0]} px · {settings} · {bg_label}")
        self.display, self.display_scale, self.display_offset = fit_display(
            preview.image_to_u8(raw, self.full_scale), self.viewport
        )
        self.render()

    def set_palette(self) -> None:
        self.palette = display_palette(self.cmap.get())
        self.render()

    def render(self) -> None:
        if self.display is None:
            return
        display, palette = self.display, self.palette
        self.viewer.show(display, [], "OptiDAQ spot size", palette=palette)
        if self.viewer.closed or self.spot is None:
            return
        if self.display is not display or self.palette is not palette:
            return
        spot = self.spot
        canvas, scale, offset = self.viewer.canvas, self.display_scale, self.display_offset
        gaussian = spot.gaussian
        if gaussian is not None:
            angle = np.linspace(0, 2 * np.pi, 49)
            ring = np.column_stack((np.cos(angle), np.sin(angle))) * (gaussian.fwhm / 2)
            outline = ((ring @ gaussian.axes.T + gaussian.center + 0.5) * scale + offset).ravel()
            canvas.create_line(*outline, fill="black", width=4, tags="spot")
            canvas.create_line(*outline, fill="white", width=2, tags="spot")
        else:
            box = np.array(spot.box) * np.tile(scale, 2) + np.tile(offset, 2)
            canvas.create_rectangle(*box, outline="black", width=4, tags="spot")
            canvas.create_rectangle(*box, outline="white", width=2, tags="spot")
        cx, cy = ((gaussian.center if gaussian is not None else spot.center) + 0.5) * scale + offset
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            tick = (cx + 9 * dx, cy + 9 * dy, cx + 17 * dx, cy + 17 * dy)
            canvas.create_line(*tick, fill="black", width=5, tags="spot")
            canvas.create_line(*tick, fill="white", width=2, tags="spot")

    def run(self) -> None:
        try:
            self.command(self.select)
            while not self.viewer.closed and not preview.STOP_REQUESTED:
                if self.tap is not None:
                    if self.tap.error is not None:
                        self.status.set(f"{self.tap.device}: {self.tap.error}")
                        self.spot = None
                        self.fit_message = ""
                        self.viewer.canvas.delete("spot")
                        self.show_result()
                    else:
                        record = self.latest.snapshot()[self.tap.device]
                        frame = record["frame"]
                        if frame is not None and frame is not self.last_frame:
                            self.last_frame = frame
                            self.draw_frame(frame, record["received_at"])
                        elif frame is not None and time.monotonic() - record["received_at"] > max(
                            3, self.applied_exposure_us / 1e6 + 2
                        ):
                            self.spot = None
                            self.fit_message = "Waiting for a fresh frame. Check the camera trigger."
                            self.viewer.canvas.delete("spot")
                            self.show_result()
                if not self.viewer.closed:
                    self.viewer.root.update()
                    time.sleep(0.1)
        finally:
            errors = []
            try:
                self.stop_stream()
            except Exception as error:
                errors.append(str(error))
            for device in list(self.original):
                try:
                    self.restore_controls(device)
                except Exception as error:
                    errors.append(f"{device} restore failed: {error}")
            self.viewer.close()
            if errors:
                raise RuntimeError("; ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--camera", help="Initial camera alias (defaults to cam2 when ready)")
    args = parser.parse_args()
    signal.signal(signal.SIGINT, preview.request_stop)
    signal.signal(signal.SIGTERM, preview.request_stop)
    SpotPreview(preview.load_config(args.config), args.camera).run()


if __name__ == "__main__":
    main()
