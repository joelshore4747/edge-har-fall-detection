#!/usr/bin/env python3
"""Prepare the checked-in WISDM export into the common schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.ingest import load_wisdm
from pipeline.validation import validate_ingestion_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare WISDM into the dissertation common schema")
    parser.add_argument("--input", default="data/raw/WISDM", help="WISDM directory or split CSV")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--sample-limit", type=int, default=0, help="Optional max sessions per split")
    parser.add_argument("--out-csv", default="data/processed/wisdm/wisdm_common_schema.csv")
    parser.add_argument("--summary-json", default="data/processed/wisdm/wisdm_common_schema.summary.json")
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def main() -> int:
    args = parse_args()
    input_path = _resolve(args.input)
    out_csv = _resolve(args.out_csv)
    summary_json = _resolve(args.summary_json)

    if not input_path.exists():
        print(f"ERROR: WISDM input path not found: {input_path}")
        return 1

    max_sessions = None if args.sample_limit <= 0 else int(args.sample_limit)
    split = None if args.split == "all" else args.split
    df = load_wisdm(input_path, split=split, max_sessions=max_sessions)

    validation = validate_ingestion_dataframe(df)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_csv, index=False)

    summary = {
        "dataset": "WISDM",
        "input_path": str(input_path),
        "split": args.split,
        "sample_limit": args.sample_limit,
        "rows": int(len(df)),
        "subjects_count": int(df["subject_id"].nunique(dropna=True)) if "subject_id" in df.columns else 0,
        "sessions_count": int(df["session_id"].nunique(dropna=True)) if "session_id" in df.columns else 0,
        "label_raw_counts": df["label_raw"].astype(str).value_counts(dropna=False).to_dict(),
        "label_mapped_counts": df["label_mapped"].astype(str).value_counts(dropna=False).to_dict(),
        "sampling_rate_hz_summary": {
            "median": _json_safe(pd.to_numeric(df["sampling_rate_hz"], errors="coerce").median()),
            "min": _json_safe(pd.to_numeric(df["sampling_rate_hz"], errors="coerce").min()),
            "max": _json_safe(pd.to_numeric(df["sampling_rate_hz"], errors="coerce").max()),
        },
        "validation": {
            "is_valid": bool(validation.is_valid),
            "errors": list(validation.errors),
            "warnings": list(validation.warnings),
        },
        "notes": [
            "This WISDM export is accelerometer-only.",
            "The checked-in WISDM files do not include subject IDs; subject_id preserves the provided split name only.",
            "Session IDs are derived from timestamp resets, large gaps, and label changes.",
        ],
        "output_csv": str(out_csv),
    }
    summary_json.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")

    print(f"Prepared WISDM common-schema CSV: {out_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Rows={len(df)} sessions={summary['sessions_count']} mapped_labels={summary['label_mapped_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
