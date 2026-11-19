"""View a live OptiDAQ camera and magnify a two-click selection.

Run in the CERN acc-py environment from the repository root:

    python experiments/camera_area_magnifier.py

The main pane shows the entire incoming frame. Exposure and gain changes apply
temporarily to the selected camera. Reset All, switching cameras, and normal
exit restore the camera settings changed by this viewer.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import signal
import sys
import time

import numpy as np

import camera_preview as preview

from optidaq.extensions.lab_camera import make_lab_client


DEFAULT_CONFIG = {
    "cameras": {
        "cam1": "BTVDC_865_BTVDIGITAL_1",
        "cam2": "BTVDC_865_BTVDIGITAL_2",
    },
    "camera_setting": {},
    "tap": {
        "mode": "subscribe",
        "get_selector": "SPS.USER.ALL",
        "subscribe_selector": "SPS.USER.ALL",
        "stamp_fields": ("acqStamp", "imageTimeStamp", "acqTimeStamp"),
        "min_ns_epoch": 100_000_000_000_000_000,
    },
    "sync": {},
}
EXPOSURE = "cameraExposureTimeUs"
GAIN = "cameraGain"
AUTO_FIELDS = {EXPOSURE: "cameraExposureAuto", GAIN: "cameraGainAuto"}


# Sampled RGB values from the standard colormaps, expanded to Tk's 256 colors.
PALETTE_SAMPLES = {
    "viridis": "440154 48186a 472d7b 424086 3b528b 33638d 2c728e 26828e 21918c 1f9f88 27ad81 3dbc74 5cc863 81d34d aadc32 d5e21a fde725",
    "magma": "000004 0a0822 1d1147 36106b 51127c 6a1c81 832681 9c2e7f b73779 cf4070 e55064 f4695c fb8761 fea571 fec287 fde0a1 fcfdbf",
    "cividis": "00224e 002e6a 1a386f 32436d 434e6c 535a6d 61656f 6f7073 7d7c78 8b8778 9a9376 aaa073 bbad6d ccba64 ddc858 efd748 fee838",
}
COLORMAPS = ("gray", *PALETTE_SAMPLES)
REFRESH_MS = 50


def display_palette(name: str):
    if name == "gray":
        return None
    colors = np.frombuffer(bytes.fromhex(PALETTE_SAMPLES[name]), dtype=np.uint8).reshape(-1, 3)
    positions = np.linspace(0, 255, len(colors))
    channels = [np.interp(np.arange(256), positions, channel) for channel in colors.T]
    return np.rint(np.column_stack(channels)).astype(np.uint8)


def fit_box(image_shape, viewport_shape):
    """Center the image in the viewport with a single scale for both axes."""
    height, width = image_shape
    view_height, view_width = viewport_shape
    scale = min(view_width / width, view_height / height)
    fitted_w = max(1, round(width * scale))
    fitted_h = max(1, round(height * scale))
    return (view_width - fitted_w) // 2, (view_height - fitted_h) // 2, fitted_w, fitted_h


def source_point(x, y, bounds, image_shape):
    """Map a click to the source pixel sampled by the displayed raster."""
    left, top, width, height = bounds
    x, y = x - left, y - top
    if not (0 <= x < width and 0 <= y < height):
        return None
    src_h, src_w = image_shape
    return x * (src_w - 1) // max(1, width - 1), y * (src_h - 1) // max(1, height - 1)


def parse_boost(text: str):
    text = text.strip()
    if not text or text.lower() == "none":
        return None
    factor = float(text)
    if not math.isfinite(factor) or factor < 1:
        raise ValueError("Intensity boost must be None or a finite number at least 1.")
    return factor


def display_u8(image, full_scale, boost):
    # Apply boost to sensor values before converting to 8-bit display pixels.
    if boost is None:
        return preview.image_to_u8(image, full_scale)
    normalized = np.clip(np.asarray(image, dtype=np.float64) / full_scale, 0, 1)
    return (np.minimum(normalized * boost, 1) * 255).astype(np.uint8)


def acquired_field(key):
    return "acq" + key[0].upper() + key[1:]


def controls_match(readback, values, *, applied):
    for key, expected in values.items():
        auto = AUTO_FIELDS.get(key)
        if applied and auto and values[auto] != preview.CAMERA_AUTO_OFF_VALUE:
            continue  # Automatic modes choose their own applied exposure or gain.
        field = acquired_field(key) if applied else key
        tolerance = preview.GAIN_READBACK_TOLERANCE if key == GAIN else 0
        if applied and key == EXPOSURE:
            tolerance = 1
        try:
            actual = float(readback[field])
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance):
            return False
    return True


class TemporaryCameraControls:
    """Save and restore only the fields this viewer changes."""

    def __init__(self):
        self.writer = None
        self.device = None
        self.original = {}

    def write(self, device, values):
        endpoint = f"{device}/CameraSetting"
        preview.set_value(self.writer, endpoint, preview.camera_setting_payload(values))
        deadline = time.monotonic() + preview.CAMERA_READBACK_TIMEOUT_S
        while True:
            requested = preview.get_value(self.writer, endpoint)
            applied = preview.get_value(self.writer, f"{device}/AcquiredParameters")
            if controls_match(requested, values, applied=False) and controls_match(applied, values, applied=True):
                return applied
            if time.monotonic() >= deadline:
                raise RuntimeError(f"{device} did not confirm the requested exposure or gain settings.")
            time.sleep(preview.CAMERA_READBACK_POLL_S)

    def apply(self, device, changes):
        if not changes or not changes.keys() <= AUTO_FIELDS.keys():
            raise ValueError("Supply exposure, gain, or both.")
        if self.original and self.device != device:
            raise RuntimeError(f"Restore {self.device} before changing {device}.")
        if self.writer is None:
            self.writer = make_lab_client(require_rbac=True)

        requested = preview.get_value(self.writer, f"{device}/CameraSetting")
        if EXPOSURE in changes:
            low, high = preview.read_camera_exposure_limits(self.writer, device)
            if not low <= changes[EXPOSURE] <= high:
                raise ValueError(f"Exposure must be between {low / 1000:g} and {high / 1000:g} ms.")
        if GAIN in changes:
            low = float(requested.get("cameraGain_min", preview.GAIN_MIN))
            high = float(requested.get("cameraGain_max", preview.GAIN_MAX))
            if not math.isfinite(low) or not math.isfinite(high) or low > high:
                raise RuntimeError(f"{device} returned invalid gain limits.")
            if not low <= changes[GAIN] <= high:
                raise ValueError(f"Gain must be between {low:g} and {high:g}.")

        values = dict(changes)
        values.update({AUTO_FIELDS[key]: preview.CAMERA_AUTO_OFF_VALUE for key in changes})
        try:
            before = {key: requested[key] for key in values}
            if not all(math.isfinite(float(value)) for value in before.values()):
                raise ValueError("non-finite camera setting")
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"Could not save the original settings for {device}.") from error

        saved = self.original.copy()
        self.device = device
        for key, value in before.items():
            self.original.setdefault(key, value)
        try:
            return self.write(device, values)
        except Exception as error:
            try:
                self.write(device, before)
            except Exception as rollback_error:
                raise RuntimeError(
                    f"{error} Restoring the previous settings also failed: {rollback_error}"
                ) from error
            self.original = saved
            self.device = device if saved else None
            raise

    def restore(self):
        if not self.original:
            return None
        applied = self.write(self.device, self.original)
        self.original.clear()
        self.device = None
        return applied


class ImagePane:
    def __init__(self, parent):
        self.canvas = preview.tk.Canvas(parent, background="black", highlightthickness=0)
        self.item = self.canvas.create_image(0, 0, anchor="nw")
        self.bounds = None
        self.photo = None
        self.use_put = False

    def clear(self):
        self.canvas.itemconfigure(self.item, image="")
        self.bounds = self.photo = None

    def show(self, image, palette, full_scale, boost):
        viewport = (self.canvas.winfo_height(), self.canvas.winfo_width())
        self.bounds = fit_box(image.shape, viewport)
        left, top, width, height = self.bounds
        sampled = preview.resize_nearest_to_shape(image, (height, width))
        pixels = display_u8(sampled, full_scale, boost)
        if not self.use_put:
            try:
                png = preview.gray_png_bytes(pixels, palette)
                self.photo = preview.tk.PhotoImage(
                    master=self.canvas, data=base64.b64encode(png), format="png"
                )
            except preview.tk.TclError:
                self.use_put = True
        if self.use_put:
            self.photo = preview.image_to_tk_put(pixels, palette)
        self.canvas.coords(self.item, left, top)
        self.canvas.itemconfigure(self.item, image=self.photo)


class CameraAreaMagnifier:
    def __init__(self, cfg):
        from tkinter import messagebox, ttk

        self.cfg = cfg
        self.client = preview.make_client(require_rbac=False)
        self.taps = {
            tap.device: tap
            for tap in preview.build_camera_taps(cfg, self.client, allow_image_number_reset=True)
        }
        self.active_tap = None
        self.connected_tap = None
        self.pending = None
        self.close_requested = False
        self.latest = None
        self.camera_controls = TemporaryCameraControls()
        self.control_text = {}
        self.frame = self.image = None
        self.first_corner = self.selection = None
        self.boost = None
        self.palette = None
        self.dirty = True
        self.messagebox = messagebox

        tk = preview.tk
        self.root = root = tk.Tk()
        root.title("OptiDAQ Camera Area Magnifier")
        root.geometry(f"{min(1320, root.winfo_screenwidth() - 48)}x"
                      f"{min(820, root.winfo_screenheight() - 100)}")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", self.close)

        self.alias = tk.StringVar(master=root, value=next(iter(self.taps)))
        self.cmap = tk.StringVar(master=root, value="gray")
        self.boost_text = tk.StringVar(master=root, value="None")
        self.status = tk.StringVar(master=root)
        self.exposure_ms = tk.StringVar(master=root)
        self.gain = tk.StringVar(master=root)
        self.control_inputs = {EXPOSURE: self.exposure_ms, GAIN: self.gain}
        controls = ttk.Frame(root, padding=8)
        controls.pack(fill="x")
        for label, variable, values, change in (
            ("Camera", self.alias, tuple(self.taps), self.select_camera),
            ("Colormap", self.cmap, COLORMAPS, self.set_colormap),
        ):
            ttk.Label(controls, text=label).pack(side="left", padx=(0, 4))
            menu = ttk.Combobox(controls, textvariable=variable, values=values,
                                state="readonly", width=9)
            menu.pack(side="left", padx=(0, 14))
            menu.bind("<<ComboboxSelected>>", change)
            if variable is self.alias:
                self.camera_menu = menu
        ttk.Label(controls, text="Intensity boost (×)").pack(side="left", padx=(0, 4))
        boost_input = ttk.Entry(controls, textvariable=self.boost_text, width=8)
        boost_input.pack(side="left", padx=(0, 4))
        boost_input.bind("<Return>", self.set_boost)
        ttk.Button(controls, text="Apply Boost", command=self.set_boost).pack(side="left")

        camera_settings = ttk.Frame(root, padding=(8, 0, 8, 8))
        camera_settings.pack(fill="x")
        self.camera_widgets = []
        for field, label in ((EXPOSURE, "Exposure (ms)"), (GAIN, "Gain")):
            ttk.Label(camera_settings, text=label).pack(side="left", padx=(0, 4))
            entry = ttk.Entry(camera_settings, textvariable=self.control_inputs[field], width=9)
            entry.pack(side="left", padx=(0, 10))
            entry.bind("<Return>", self.apply_camera_settings)
            self.camera_widgets.append(entry)
        apply_button = ttk.Button(camera_settings, text="Apply Camera", command=self.apply_camera_settings)
        apply_button.pack(side="left", padx=(0, 10))
        self.camera_widgets.append(apply_button)
        self.reset_button = ttk.Button(camera_settings, text="Reset All", command=self.reset_all)
        self.reset_button.pack(side="left")

        images = ttk.Frame(root, padding=(8, 0, 8, 0))
        images.pack(fill="both", expand=True)
        images.rowconfigure(1, weight=1)
        self.main_pane = ImagePane(images)
        self.roi_pane = ImagePane(images)
        for column, (label, pane) in enumerate(
            (("Camera image", self.main_pane), ("Selection preview", self.roi_pane))
        ):
            images.columnconfigure(column, weight=3 - column, uniform="panes")
            ttk.Label(images, text=label).grid(row=0, column=column, sticky="w", padx=3)
            pane.canvas.grid(row=1, column=column, sticky="nsew", padx=3)
            pane.canvas.bind("<Configure>", self.invalidate)
        self.main_pane.canvas.bind("<Button-1>", self.click)
        self.main_pane.canvas.configure(cursor="crosshair")
        self.roi_pane.canvas.bind("<Button-1>", self.clear_preview_on_click)

        ttk.Label(root, text="Click two opposite corners on the camera image. Click again to clear.",
                  padding=(8, 5, 8, 0)).pack(fill="x")
        ttk.Label(root, textvariable=self.status, padding=(8, 0, 8, 5)).pack(fill="x")
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.update_control_widgets()

    def invalidate(self, _event=None):
        self.dirty = True

    def set_colormap(self, _event=None):
        self.palette = display_palette(self.cmap.get())
        self.invalidate()

    def set_boost(self, _event=None):
        try:
            self.boost = parse_boost(self.boost_text.get())
        except ValueError:
            self.messagebox.showerror(
                "Intensity boost", "Enter None or a finite multiplier of at least 1.", parent=self.root
            )
            return
        if self.boost is None:
            self.boost_text.set("None")
        self.invalidate()

    def update_control_widgets(self):
        busy = self.pending is not None
        self.camera_menu.configure(state="disabled" if busy else "readonly")
        self.reset_button.configure(state="disabled" if busy else "normal")
        for widget in self.camera_widgets:
            widget.configure(state="normal" if not busy and self.connected_tap is not None else "disabled")

    def request(self, operation, finish, status):
        self.pending = self.executor.submit(operation), finish
        self.status.set(status)
        self.update_control_widgets()

    def show_camera_controls(self, acquired):
        for key, variable in self.control_inputs.items():
            try:
                value = float(acquired[acquired_field(key)])
            except (KeyError, TypeError, ValueError):
                value = math.nan
            if key == EXPOSURE:
                value /= 1000
            text = f"{value:g}" if math.isfinite(value) else ""
            self.control_text[key] = text
            variable.set(text)

    def apply_camera_settings(self, _event=None):
        if self.pending is not None or self.connected_tap is None:
            return
        changes = {}
        try:
            for key, variable in self.control_inputs.items():
                text = variable.get().strip()
                if text == self.control_text[key]:
                    continue
                value = float(text)
                if key == EXPOSURE:
                    microseconds = value * 1000
                    if not math.isfinite(microseconds) or microseconds < 1:
                        raise ValueError
                    value = round(microseconds)
                elif not math.isfinite(value):
                    raise ValueError
                changes[key] = value
        except ValueError:
            self.status.set("Enter a positive exposure in ms and a numeric gain.")
            return
        if changes:
            tap = self.connected_tap
            self.request(lambda: self.camera_controls.apply(tap.fesa_device, changes),
                         self.camera_settings_applied, f"Applying {tap.device} settings…")

    def camera_settings_applied(self, future):
        self.show_camera_controls(future.result())
        self.status.set(f"{self.connected_tap.device} camera settings applied.")

    def reset_all(self):
        if self.pending is not None:
            return
        self.boost = None
        self.boost_text.set("None")
        self.cmap.set("gray")
        self.set_colormap()
        self.clear_selection()
        self.request(self.camera_controls.restore, self.reset_finished, "Resetting…")

    def reset_finished(self, future):
        applied = future.result()
        if applied is not None:
            self.show_camera_controls(applied)
        else:
            for key, variable in self.control_inputs.items():
                variable.set(self.control_text.get(key, ""))
        self.status.set("Reset complete.")

    def stop_stream(self):
        tap = self.active_tap
        if tap is not None:
            tap.stop()
            tap.join(timeout=2.0)
            if tap._thread is not None and tap._thread.is_alive():
                raise RuntimeError(f"{tap.device} camera stream did not stop")
            self.active_tap = None

    def connect(self, tap):
        self.camera_controls.restore()
        self.stop_stream()
        latest = preview.LatestFrames([tap])
        # The preview's dynamic-shape tap initializes its shape and dtype from the seed.
        preview.seed_current_last_images([tap], latest.emit)

        setting = preview.camera_setting_for_alias(tap.device, self.cfg["camera_setting"], self.taps)
        try:
            acquired = preview.get_value(self.client, f"{tap.fesa_device}/AcquiredParameters")
        except Exception:
            acquired = {}
        hardware_roi = bool(acquired.get("acqImageROIEnable"))
        for field in ("CameraPixelFormat", "CameraBinningMode", "CameraBinningHor", "CameraBinningVer"):
            if "acq" + field in acquired:
                setting[field[0].lower() + field[1:]] = acquired["acq" + field]
        full_scale = preview.camera_full_scale(setting)

        tap.start(latest.emit)
        self.active_tap = tap
        return tap, latest, full_scale, hardware_roi, acquired

    def select_camera(self, _event=None):
        if self.pending is not None:
            return
        name = self.alias.get()
        self.request(lambda: self.connect(self.taps[name]), self.camera_connected, f"Connecting to {name}…")

    def camera_connected(self, future):
        try:
            tap, self.latest, self.full_scale, self.hardware_roi, acquired = future.result()
        except Exception:
            if self.active_tap is not None:
                self.alias.set(self.active_tap.device)
            else:
                self.connected_tap = self.latest = self.frame = self.image = None
                self.clear_selection()
                self.show_camera_controls({})
            raise
        self.connected_tap = tap
        self.frame = self.image = None
        self.clear_selection()
        self.show_camera_controls(acquired)

    def clear_selection(self):
        self.first_corner = self.selection = None
        self.invalidate()

    def clear_preview_on_click(self, _event):
        if self.selection is not None:
            self.clear_selection()

    def click(self, event):
        if self.connected_tap is not None and self.connected_tap.device != self.alias.get():
            return
        if self.selection is not None:
            self.clear_selection()
            return
        if self.image is None or self.main_pane.bounds is None:
            return
        point = source_point(event.x, event.y, self.main_pane.bounds, self.image.shape)
        if point is None:
            return
        if self.first_corner is None:
            self.first_corner = point
        else:
            x1, y1 = self.first_corner
            x2, y2 = point
            self.selection = (min(x1, x2), min(y1, y2), max(x1, x2) + 1, max(y1, y2) + 1)
        self.invalidate()

    def draw(self):
        canvas = self.main_pane.canvas
        canvas.delete("selection")
        if self.selection is None:
            self.roi_pane.clear()
        if self.image is None:
            self.main_pane.clear()
            return

        full_scale = min(self.full_scale, 255) if self.image.dtype == np.uint8 else self.full_scale
        self.main_pane.show(self.image, self.palette, full_scale, self.boost)
        if self.first_corner is None:
            return

        left, top, width, height = self.main_pane.bounds
        src_h, src_w = self.image.shape
        if self.selection is None:
            x, y = self.first_corner
            cx = left + (x + 0.5) * width / src_w
            cy = top + (y + 0.5) * height / src_h
            canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, outline="cyan", tags="selection")
            return

        x1, y1, x2, y2 = self.selection
        canvas.create_rectangle(
            left + x1 * width / src_w, top + y1 * height / src_h,
            left + x2 * width / src_w, top + y2 * height / src_h,
            outline="cyan", tags="selection"
        )
        self.roi_pane.show(self.image[y1:y2, x1:x2], self.palette, full_scale, self.boost)

    def tick(self):
        if self.pending is not None and self.pending[0].done():
            future, finish = self.pending
            self.pending = None
            try:
                finish(future)
            except Exception as error:
                self.status.set(str(error))
            self.update_control_widgets()
            if self.close_requested:
                self.close()

        if self.latest is not None:
            tap = self.connected_tap
            name = tap.device
            if tap.error is not None:
                if self.pending is None:
                    self.status.set(f"{name}: {tap.error}")
            else:
                frame = self.latest.snapshot()[name]["frame"]
                if frame is not None and frame is not self.frame:
                    image = preview.grayscale_image(preview.frame_to_image(frame, tap))
                    if self.image is None or self.image.shape != image.shape:
                        self.clear_selection()
                        height, width = image.shape
                        roi_note = "  ·  Hardware ROI active" if self.hardware_roi else ""
                        if self.pending is None:
                            self.status.set(f"{name}  ·  {width} × {height} px{roi_note}")
                    self.frame, self.image = frame, image
                    self.invalidate()
        if self.dirty:
            self.draw()
            self.dirty = False
        self.root.after(REFRESH_MS, self.tick)

    def close(self, _event=None):
        self.close_requested = self.pending is not None
        if not self.close_requested:
            self.request(self.camera_controls.restore, self.finish_close, "Restoring camera settings…")

    def finish_close(self, future):
        self.close_requested = False
        future.result()
        self.root.quit()

    def run(self):
        signal.signal(signal.SIGTERM, lambda *_: self.root.quit())
        signal.signal(signal.SIGINT, lambda *_: self.root.quit())
        try:
            self.select_camera()
            self.tick()
            self.root.mainloop()
        finally:
            self.root.destroy()
            if self.pending is not None:
                self.pending[0].cancel()
            self.executor.shutdown()
            try:
                self.camera_controls.restore()
            except Exception as error:
                print(f"Could not restore {self.camera_controls.device}: {error}", file=sys.stderr)
            finally:
                try:
                    self.stop_stream()
                except RuntimeError as error:
                    print(error, file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, help="Override the built-in CLEAR camera configuration.")
    args = parser.parse_args()
    if preview.tk is None:
        raise RuntimeError("camera_area_magnifier.py requires tkinter for display")
    cfg = DEFAULT_CONFIG if args.config is None else preview.load_config(args.config)
    CameraAreaMagnifier(cfg).run()


if __name__ == "__main__":
    main()
