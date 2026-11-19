"""Analysis pipeline: one config -> datasets -> pipeline steps -> figures.

    python -m optidaq.analysis.pipeline --config configs/analysis_v1.json
        --steps [names]     run only the pipeline steps (all, or the named ones)
        --figures [names]   draw only the figures, from the artefacts on disk

Without options every step runs, then every figure is drawn. The three parts
only meet in the files: steps read datasets and write artefacts, figures read
artefacts and never a dataset.

Config (JSON; keys that start with "_" are comments and are ignored):

    output_dir  results folder
    cameras     {"beam": camera of the screen image,
                 "fibre": {label: camera, ...}}; one fibre or several. A list of
                 cameras or one camera name also works; the labels are then the
                 camera names.
    datasets    {name: dataset block}; see datasets.py
    pipeline    {name: step block}; "module" selects the code (MODULES), the
                other keys are its parameters. A step writes its artefacts to
                <output_dir>/<name>/. The same module can appear under several
                names with different parameters.
    style       look shared by all figures; see figures/style.py
    figures     {name: figure block}; "figure" selects the code (FIGURES),
                "source" names the step whose artefacts are drawn, "style"
                overrides the shared style, the other keys are the figure's own
                options. A figure is written to <output_dir>/figures/<name>/.

The config as run is copied to <output_dir>/config.json.

To add a module: a function run(params, *, cameras, dataset, out_dir, log) that
writes artefacts, plus one entry in MODULES. To add a figure: a function
draw(params, *, root, out_dir, style, log) that reads artefacts under ``root``,
plus one entry in FIGURES.
"""

from __future__ import annotations

import argparse
from functools import partial
from importlib import import_module
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

from optidaq.analysis.config import fill, pick, strip

VERSION = "1.1"
MODULES = {"forward_fit": "optidaq.analysis.forward_fit.run:run"}
FIGURES = {"fit_samples": "optidaq.analysis.figures.fit_samples:draw"}
DEFAULTS = {
    "output_dir": "outputs/analysis",
    "cameras": None,
    "datasets": {},
    "pipeline": {},
    "style": {},
    "figures": {},
}


def run(config: Mapping, steps: Optional[Sequence[str]] = None, figures: Optional[Sequence[str]] = None,
        log=print) -> dict:
    """Run the steps and draw the figures of ``config``.

    ``steps`` and ``figures`` name the blocks to run; None means all, an empty
    list none. Returns {step name: what its module returned}.
    """

    cfg = fill(strip(config), DEFAULTS, "analysis config", required=("cameras",))
    cameras = _cameras(cfg["cameras"])
    root = Path(cfg["output_dir"])
    step_names = _selected(cfg["pipeline"], steps, "pipeline step")
    figure_names = _selected(cfg["figures"], figures, "figure")
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"version": VERSION, **cfg}, indent=2))

    results = {}
    if step_names:
        from optidaq.analysis.datasets import opener

        dataset = opener(cfg["datasets"])
        for name in step_names:
            block = dict(cfg["pipeline"][name])
            module = _load(pick(MODULES, block.get("module"), "module"))
            log("== step %s (%s)" % (name, block["module"]))
            results[name] = module(block, cameras=cameras, dataset=dataset, out_dir=root / name, log=log)
    for name in figure_names:
        block = dict(cfg["figures"][name])
        figure = _load(pick(FIGURES, block.get("figure"), "figure"))
        log("== figure %s (%s)" % (name, block["figure"]))
        figure(block, root=root, out_dir=root / "figures" / name, style=cfg["style"], log=log)
    return results


def _cameras(cameras) -> dict:
    fibre = cameras.get("fibre")
    if isinstance(fibre, str):
        fibre = [fibre]
    if not isinstance(fibre, Mapping):
        fibre = {camera: camera for camera in fibre or []}
    if not cameras.get("beam") or not fibre:
        raise ValueError('cameras: expected {"beam": camera, "fibre": {label: camera}}, got %r' % (cameras,))
    return {"beam": cameras["beam"], "fibre": dict(fibre)}


def _selected(blocks: Mapping, names, what: str) -> list:
    if names is None:
        return list(blocks)
    for name in names:
        pick(blocks, name, what)
    return list(names)


def _load(spec: str):
    module, function = spec.split(":")
    return getattr(import_module(module), function)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="analysis config, e.g. configs/analysis_v1.json")
    parser.add_argument("--steps", nargs="*", default=None, help="run only these pipeline steps (no name: all)")
    parser.add_argument("--figures", nargs="*", default=None, help="draw only these figures (no name: all)")
    args = parser.parse_args(argv)
    steps, figures = args.steps, args.figures
    if steps is not None or figures is not None:      # an option was given: run only what it asks for
        steps = [] if steps is None else (steps or None)
        figures = [] if figures is None else (figures or None)
    run(json.loads(args.config.read_text()), steps, figures, log=partial(print, flush=True))


if __name__ == "__main__":
    main()
