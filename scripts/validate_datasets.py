#!/usr/bin/env python3
"""Chapter 3 dataset ingestion validation CLI.

This script validates dataset extracts (or test fixtures) against the unified schema and prints
human-readable summaries. It intentionally avoids model training and resampling/windowing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable, Dict, Iterable

import pandas as pd

# Allow running the script directly via `python scripts/validate_datasets.py`
# without installing the project as a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.ingest import load_mobifall, load_pamap2, load_sisfall, load_uci_har, load_wisdm
from pipeline.validation import validate_ingestion_dataframe

LoaderFn = Callable[[str | Path], pd.DataFrame]


def _default_fixture_path(name: str) -> Path:
    return Path("tests/fixtures") / f"{name}_sample.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dataset ingestion outputs for Chapter 3")
    parser.add_argument("--uci-har", dest="uci_har", default=None, help="Path to UCI HAR simplified CSV extract")
    parser.add_argument("--pamap2", dest="pamap2", default=None, help="Path to PAMAP2 simplified CSV extract")
    parser.add_argument("--wisdm", dest="wisdm", default=None, help="Path to WISDM CSV extract or folder")
    parser.add_argument("--mobifall", dest="mobifall", default=None, help="Path to MobiFall simplified CSV extract")
    parser.add_argument("--sisfall", dest="sisfall", default=None, help="Path to SisFall simplified CSV extract")
    parser.add_argument(
        "--use-fixtures",
        action="store_true",
        help="Use tests/fixtures/*_sample.csv defaults for a quick smoke test",
    )
    parser.add_argument(
        "--limit-labels",
        type=int,
        default=10,
        help="Max label counts to print per dataset summary",
    )
    return parser.parse_args()


def summarize_dataframe(df: pd.DataFrame, *, limit_labels: int) -> str:
    lines: list[str] = []
    lines.append(f"rows_loaded: {len(df)}")
    lines.append(f"columns_present: {', '.join(df.columns.tolist())}")

    subject_count = int(df["subject_id"].nunique(dropna=True)) if "subject_id" in df.columns else 0
    lines.append(f"subjects_count: {subject_count}")

    label_raw_counts = df["label_raw"].astype(str).value_counts(dropna=False).head(limit_labels)
    lines.append("label_raw_counts:")
    for k, v in label_raw_counts.items():
        lines.append(f"  - {k}: {int(v)}")

    label_mapped_counts = df["label_mapped"].astype(str).value_counts(dropna=False).head(limit_labels)
    lines.append("label_mapped_counts:")
    for k, v in label_mapped_counts.items():
        lines.append(f"  - {k}: {int(v)}")

    key_cols = ["timestamp", "ax", "ay", "az", "gx", "gy", "gz", "label_raw", "label_mapped"]
    lines.append("null_percentages:")
    for col in key_cols:
        if col in df.columns:
            null_pct = float(df[col].isna().mean() * 100) if len(df) else 0.0
            lines.append(f"  - {col}: {null_pct:.1f}%")

    return "\n".join(lines)


def validate_one(name: str, path: Path, loader: LoaderFn, *, limit_labels: int) -> bool:
    print(f"\n=== {name} ===")
    if not path.exists():
        print(f"WARN: path not found, skipping: {path}")
        return True

    try:
        df = loader(path)
        result = validate_ingestion_dataframe(df)
        print(summarize_dataframe(df, limit_labels=limit_labels))
        if result.warnings:
            print("warnings:")
            for warning in result.warnings:
                print(f"  - {warning}")
        if not result.is_valid:
            print("errors:")
            for error in result.errors:
                print(f"  - {error}")
            return False
        print("validation_status: PASS")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: failed to load/validate {name}: {exc}")
        return False


def main() -> int:
    args = parse_args()

    requested_paths: Dict[str, Path] = {}
    if args.use_fixtures:
        requested_paths = {
            "uci_har": _default_fixture_path("uci_har"),
            "pamap2": _default_fixture_path("pamap2"),
            "wisdm": _default_fixture_path("wisdm"),
            "mobifall": _default_fixture_path("mobifall"),
            "sisfall": _default_fixture_path("sisfall"),
        }
    else:
        # Default to common lesson folders only if specific paths are not provided.
        requested_paths = {
            "uci_har": Path(args.uci_har) if args.uci_har else Path("data/raw/UCIHAR_Dataset/UCI-HAR Dataset/uci_har_sample.csv"),
            "pamap2": Path(args.pamap2) if args.pamap2 else Path("data/raw/PAMAP2_Dataset/pamap2_sample.csv"),
            "wisdm": Path(args.wisdm) if args.wisdm else Path("data/raw/WISDM"),
            "mobifall": Path(args.mobifall) if args.mobifall else Path("data/raw/MOBIACT_Dataset/mobifall_sample.csv"),
            "sisfall": Path(args.sisfall) if args.sisfall else Path("data/raw/SISFALL_Dataset/sisfall_sample.csv"),
        }

    loaders: Dict[str, LoaderFn] = {
        "uci_har": load_uci_har,
        "pamap2": load_pamap2,
        "wisdm": load_wisdm,
        "mobifall": load_mobifall,
        "sisfall": load_sisfall,
    }

    all_ok = True
    for name in ["uci_har", "pamap2", "wisdm", "mobifall", "sisfall"]:
        ok = validate_one(name, requested_paths[name], loaders[name], limit_labels=args.limit_labels)
        all_ok = all_ok and ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
