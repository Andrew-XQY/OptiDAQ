"""Config-driven pattern streams for the DMD actuator.

A stream spec is ``{"name": <source>, "weight": w, "params": {...},
"transforms": [...]}``. ``name`` selects a source from ``SOURCES``; ``params``
go to that source; ``transforms`` use the shared registry in
``optidaq.data.transforms``. A source yields 2D images or ``(id, image)``
pairs, and after its transforms every stream must yield 2D uint8 patterns.
Several streams mix by drawing one stream per pattern with probability
proportional to ``weight``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
import itertools
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from optidaq.data.transforms import build_pipeline
from optidaq.extensions.sgm import dynamic_patterns


PatternItem = tuple[int, Optional[str], np.ndarray]


def png_dir(
    path: str,
    glob: str = "*.png",
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(file name, image)`` from a PNG folder, cycling forever.

    Color PNGs are converted to 8-bit gray; 8/16-bit gray PNGs keep their values.
    """

    from PIL import Image

    files = sorted(Path(path).expanduser().glob(glob))
    if not files:
        raise RuntimeError("png_dir: no %r files in %s" % (glob, path))
    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(len(files)) if shuffle else range(len(files))
        for index in order:
            with Image.open(files[index]) as img:
                if img.mode not in ("L", "I;16"):
                    img = img.convert("L")
                yield files[index].name, np.asarray(img)


SOURCES: dict[str, Callable[..., Iterable[Any]]] = {
    "png_dir": png_dir,
    "dynamic_patterns": dynamic_patterns,
}


def _stream(
    index: int,
    name: str,
    items: Iterator[Any],
    transform: Callable[[np.ndarray], np.ndarray],
) -> Iterator[PatternItem]:
    for item in items:
        if isinstance(item, tuple) and len(item) == 2:
            source_id, image = str(item[0]), item[1]
        else:
            source_id, image = None, item
        # Copy first: generators may hand back the same buffer every time.
        pattern = np.asarray(transform(np.array(image, copy=True)))
        if pattern.ndim != 2 or pattern.dtype != np.uint8:
            raise ValueError(
                "stream %d (%s) must yield 2D uint8 patterns, got %s %s; "
                "end its transforms with to_uint8"
                % (index, name, pattern.dtype, pattern.shape)
            )
        yield index, source_id, pattern
    raise RuntimeError("stream %d (%s) ran out of patterns" % (index, name))


def build_pattern_stream(
    specs: Sequence[Mapping[str, Any]],
    *,
    seed: Optional[int] = None,
) -> Iterator[PatternItem]:
    """Return an endless iterator of ``(stream index, source id, pattern)``.

    The first pattern of every stream is produced here, so a bad path, missing
    package or wrong output type fails before acquisition starts.
    """

    if not specs:
        raise ValueError("at least one pattern stream is required")
    streams = []
    weights = []
    for index, spec in enumerate(specs):
        name = str(spec.get("name", ""))
        if name not in SOURCES:
            raise ValueError(
                "unknown pattern source %r (known: %s)" % (name, ", ".join(SOURCES))
            )
        items = iter(SOURCES[name](**dict(spec.get("params", {}) or {})))
        stream = _stream(index, name, items, build_pipeline(spec.get("transforms")))
        streams.append(itertools.chain([next(stream)], stream))
        weights.append(float(spec.get("weight", 1.0)))

    probabilities = np.asarray(weights, dtype=np.float64)
    if np.any(probabilities < 0) or probabilities.sum() <= 0:
        raise ValueError("stream weights must be >= 0 with a positive sum")
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(seed)

    def mix() -> Iterator[PatternItem]:
        while True:
            yield next(streams[int(rng.choice(len(streams), p=probabilities))])

    return mix()
