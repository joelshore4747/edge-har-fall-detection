#!/usr/bin/env python3
"""Inspect a generated HAR feature table CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


METADATA_HINT_COLS = [
    "window_id",
    "dataset_name",
    "subject_id",
    "session_id",
    "source_file",
    "task_type",
    "label_mapped_majority",
    "is_acceptable",
    "n_samples",
    "missing_ratio",
    "has_large_gap",
    "n_gaps",
    "start_ts",
    "end_ts",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a HAR feature table CSV")
    parser.add_argument("path", help="Path to feature table CSV")
    parser.add_argument("--head", type=int, default=10)
    parser.add_argument("--top-null", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.path)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        return 1

    df = pd.read_csv(path)
    metadata_cols = [c for c in METADATA_HINT_COLS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    print(f"path={path}")
    print(f"rows={len(df)} cols={len(df.columns)}")
    print(f"metadata_cols={len(metadata_cols)} feature_cols={len(feature_cols)}")
    print("metadata_columns:", metadata_cols)
    print("feature_columns_preview:", feature_cols[:20])

    if "label_mapped_majority" in df.columns:
        print("label_counts:", df["label_mapped_majority"].astype(str).value_counts(dropna=False).to_dict())
    if "subject_id" in df.columns:
        print("subjects_count:", int(df["subject_id"].nunique(dropna=True)))
    if "dataset_name" in df.columns:
        print("datasets:", sorted(df["dataset_name"].astype(str).dropna().unique().tolist()))

    print("\nnull_counts_top:")
    print(df.isna().sum().sort_values(ascending=False).head(args.top_null))

    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    if numeric_cols:
        print("\nnumeric_feature_summary_preview:")
        summary_cols = [c for c in numeric_cols if c in feature_cols][:10]
        if summary_cols:
            print(df[summary_cols].describe().T)

    print("\nhead:")
    print(df.head(args.head))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
