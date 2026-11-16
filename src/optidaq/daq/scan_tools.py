"""Small generic scan helpers for OptiDAQ scripts.

No CERN, PyDA, camera, motor, or file-format assumptions live here.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Tuple


ScanStep = Tuple[Optional[int], int]


@dataclass(frozen=True)
class ScanPoint:
    """One planned target as offsets from the scan reference point."""

    index: int
    x_offset: float
    y_offset: float
    row: int
    col: int

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "x_offset": self.x_offset,
            "y_offset": self.y_offset,
            "row": self.row,
            "col": self.col,
        }


def boustrophedon(
    nx: int,
    ny: int,
    x_step: int,
    y_step: int,
    y_enabled: bool,
) -> Iterable[ScanStep]:
    """Yield ``(x_delta, y_delta)`` steps for an ``nx * ny`` line scan.

    The first yielded item is dwell-only: ``(None, 0)``. ``x_delta`` is ``None``
    for dwell-only row changes. If y is disabled, row changes still yield
    dwell-only points so total point count remains ``nx * ny``.
    """

    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive")

    yield (None, 0)
    direction = 1
    for row in range(ny):
        for _col in range(nx - 1):
            yield (direction * int(x_step), 0)
        if row != ny - 1:
            yield (None, int(y_step) if y_enabled else 0)
            direction *= -1


def grid_scan_points(
    nx: int,
    ny: int,
    width: float,
    height: float,
    *,
    order: str = "square_spiral",
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> Iterable[ScanPoint]:
    """Yield coordinate-offset points over a rectangular grid.

    ``square_spiral`` starts near the center and walks outward over grid
    neighbors. For even ``nx`` or ``ny`` there is no exact center grid point, so
    the first point is one of the central cells. Units are intentionally not
    assumed here; the caller decides what the coordinate unit means.
    """

    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive")
    if width < 0 or height < 0:
        raise ValueError("width and height must be non-negative")

    xs = _axis_values(nx, float(width), float(center_x))
    ys = _axis_values(ny, float(height), float(center_y))

    if order == "square_spiral":
        indices = _spiral_indices(nx, ny)
    elif order == "boustrophedon":
        indices = _boustrophedon_indices(nx, ny)
    elif order == "top_right_raster":
        indices = _top_right_raster_indices(nx, ny)
    else:
        raise ValueError("unknown scan order %r" % order)

    for index, (row, col) in enumerate(indices):
        yield ScanPoint(
            index=index,
            x_offset=xs[col],
            y_offset=ys[row],
            row=row,
            col=col,
        )


def make_scan_plan(scan_cfg: Mapping[str, Any]) -> List[ScanPoint]:
    """Build a small physical-coordinate scan plan from JSON config."""

    pattern = str(scan_cfg.get("pattern", "square_spiral"))
    if pattern in ("square_spiral", "grid_spiral", "center_out_square"):
        order = "square_spiral"
    elif pattern in ("boustrophedon", "raster", "snake"):
        order = "boustrophedon"
    elif pattern in ("top_right_raster", "top_right_snake"):
        order = "top_right_raster"
    else:
        raise ValueError("unknown scan pattern %r" % pattern)

    return list(
        grid_scan_points(
            int(scan_cfg["nx"]),
            int(scan_cfg["ny"]),
            float(scan_cfg["width"]),
            float(scan_cfg["height"]),
            order=order,
            center_x=float(scan_cfg.get("center_x", 0.0)),
            center_y=float(scan_cfg.get("center_y", 0.0)),
        )
    )


def make_counter_dwell(
    *,
    get_stats: Callable[[], Any],
    raise_if_failed: Callable[[], None],
    devices: Sequence[str],
    frames_per_point: int,
    timeout_s: float,
    point_log: list,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Callable[[], None]:
    """Return a dwell callable based only on per-device saved counts."""

    if frames_per_point <= 0:
        raise ValueError("frames_per_point must be positive")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")

    devices = tuple(devices)

    def dwell() -> None:
        stats0 = get_stats()
        markers = {
            device: stats0.per_device_saved.get(device, 0)
            for device in devices
        }
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            if should_stop is not None and should_stop():
                raise KeyboardInterrupt("stop requested")
            raise_if_failed()
            stats = get_stats()
            saved = stats.per_device_saved
            if all(
                saved.get(device, 0) >= markers[device] + frames_per_point
                for device in devices
            ):
                point_index = len(point_log) + 1
                point_log.append(
                    {
                        "point_index": point_index,
                        "per_device_saved_start": dict(markers),
                        "per_device_saved_end": dict(saved),
                        "latest_source_image_number": dict(
                            stats.latest_source_image_number
                        ),
                        "latest_acq_stamp_ns": dict(stats.latest_acq_stamp_ns),
                    }
                )
                if point_index == 1 or point_index % 10 == 0:
                    print("point %d saved=%s" % (point_index, dict(saved)))
                return
            time.sleep(0.05)

        raise RuntimeError(
            "point timeout: no %d saved frame(s) on all devices within %.1f s "
            "(markers=%s)" % (frames_per_point, timeout_s, markers)
        )

    return dwell


def _axis_values(count: int, span: float, center: float) -> List[float]:
    if count == 1:
        return [center]
    step = span / float(count - 1)
    start = center - span / 2.0
    return [start + step * index for index in range(count)]


def _boustrophedon_indices(nx: int, ny: int) -> Iterable[Tuple[int, int]]:
    for row in range(ny):
        cols = range(nx) if row % 2 == 0 else range(nx - 1, -1, -1)
        for col in cols:
            yield row, col


def _top_right_raster_indices(nx: int, ny: int) -> Iterable[Tuple[int, int]]:
    for row in range(ny):
        cols = range(nx - 1, -1, -1) if row % 2 == 0 else range(nx)
        for col in cols:
            yield row, col


def _spiral_indices(nx: int, ny: int) -> Iterable[Tuple[int, int]]:
    total = nx * ny
    yielded = 0
    row = (ny - 1) // 2
    col = (nx - 1) // 2

    if 0 <= row < ny and 0 <= col < nx:
        yield row, col
        yielded += 1

    step_len = 1
    directions = ((0, 1), (1, 0), (0, -1), (-1, 0))
    while yielded < total:
        for direction_index, (dr, dc) in enumerate(directions):
            for _ in range(step_len):
                row += dr
                col += dc
                if 0 <= row < ny and 0 <= col < nx:
                    yield row, col
                    yielded += 1
                    if yielded >= total:
                        return
            if direction_index % 2 == 1:
                step_len += 1
