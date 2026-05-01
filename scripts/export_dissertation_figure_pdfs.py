#!/usr/bin/env python3
"""Export dissertation SVG figures to PDF fallbacks using ImageMagick."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

FIGURES = [
    (
        "figure_3_1_dataset_inventory",
        "apps/admin/public/figures/dataset_inventory.svg",
    ),
    (
        "figure_4_1_har_domain_adaptation_delta",
        "results/validation/har_domain_adaptation_delta.svg",
    ),
    (
        "figure_6_1_phone1_runtime_timeline",
        "apps/admin/public/figures/phone1_runtime_timeline.svg",
    ),
    (
        "figure_6_2_phone1_annotation_alignment",
        "apps/admin/public/figures/phone1_annotation_alignment.svg",
    ),
    (
        "figure_6_3_phone1_grouped_fall_events",
        "apps/admin/public/figures/phone1_grouped_fall_events.svg",
    ),
    (
        "figure_8_1_primary_model_comparison",
        "apps/admin/public/figures/primary_model_comparison.svg",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="docs/figure_exports",
        help="Directory for PDF fallbacks used by final dissertation export",
    )
    parser.add_argument(
        "--density",
        type=int,
        default=192,
        help="Rasterisation density passed to ImageMagick",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    magick = shutil.which("magick")
    if magick is None:
        raise SystemExit("ImageMagick 'magick' command is required for PDF export")

    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for stem, source in FIGURES:
        source_path = (REPO_ROOT / source).resolve()
        if not source_path.exists():
            raise SystemExit(f"Missing source figure: {source_path}")
        out_path = out_dir / f"{stem}.pdf"
        subprocess.run(
            [
                magick,
                "-density",
                str(args.density),
                str(source_path),
                str(out_path),
            ],
            check=True,
        )
        print(f"Wrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
