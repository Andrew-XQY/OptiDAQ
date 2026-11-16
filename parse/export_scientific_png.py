#!/usr/bin/env python3
"""Export a raw run using the selected PNG and SQLite export configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from optidaq.data.scientific_png_export import (  # noqa: E402
    config_from_mapping,
    export_run_to_png_dataset,
    load_export_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export an OptiDAQ raw run as source-truth grayscale PNGs plus dataset.db."
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "parse" / "export_config_full.json"),
        help="JSON export config path. Set crop_enabled=false for the full lossless image.",
    )
    parser.add_argument("--run-dir", default=None, help="Override config run_dir.")
    parser.add_argument("--out-dir", default=None, help="Override config out_dir.")
    parser.add_argument("--run-id", default=None, help="Override config run_id.")
    parser.add_argument(
        "--sync-source",
        default=None,
        help="Override sync alignment source. Default/config value is image_ready_ns.",
    )
    parser.add_argument(
        "--sync-tolerance-ns",
        type=int,
        default=None,
        help="Override max per-sample sync span in ns. Default/config value is 50000000.",
    )
    parser.add_argument(
        "--camera",
        dest="cameras",
        action="append",
        default=None,
        help="Camera to export. Repeat for multiple cameras. Overrides config cameras.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory and replacing dataset.db.",
    )
    args = parser.parse_args()

    overrides = {
        "run_dir": args.run_dir,
        "out_dir": args.out_dir,
        "run_id": args.run_id,
        "cameras": args.cameras,
        "sync_alignment_source": args.sync_source,
        "sync_alignment_tolerance_ns": args.sync_tolerance_ns,
    }

    config_path = Path(args.config).expanduser()
    if config_path.exists():
        config = load_export_config(config_path, **overrides)
    else:
        data = {key: value for key, value in overrides.items() if value is not None}
        data.setdefault("png", {"compress_level": 6})
        data.setdefault("hooks", {"image": [], "metadata": ["stats"]})
        data.setdefault("extra_info", {})
        config = config_from_mapping(data)

    if args.overwrite:
        config.overwrite = True

    result = export_run_to_png_dataset(config)
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "out_dir": str(result.out_dir),
                "db_path": str(result.db_path),
                "image_count": result.image_count,
                "counts_by_camera": dict(result.counts_by_camera),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
