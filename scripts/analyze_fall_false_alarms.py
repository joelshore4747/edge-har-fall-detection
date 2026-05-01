#!/usr/bin/env python3
"""Analyze false alarms from a Chapter 5 threshold-baseline run directory."""

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


KEY_NUMERIC_FIELDS = [
    "peak_acc",
    "post_impact_motion",
    "post_impact_dyn_mean",
    "post_impact_dyn_rms",
    "post_impact_dyn_ratio_mean",
    "post_impact_dyn_ratio_rms",
    "jerk_peak",
    "peak_minus_mean",
    "post_impact_motion_to_peak_ratio",
    "confirm_post_impact_motion_max",
    "confirm_post_impact_variance_max",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze false alarms from threshold-fall baseline artifacts")
    parser.add_argument("--run-dir", required=True, help="Run directory containing false_alarms.csv")
    parser.add_argument("--false-alarms-csv", default=None, help="Optional explicit false_alarms CSV path")
    parser.add_argument("--test-predictions-csv", default=None, help="Optional explicit test_predictions_windows.csv path")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", default=None, help="Optional output JSON path (default: <run_dir>/false_alarm_analysis.json)")
    return parser.parse_args()


def _resolve(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    p = Path(path_str)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p


def _json_safe(value: Any):
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


def _top_counts(df: pd.DataFrame, col: str, top_k: int) -> dict[str, int]:
    if col not in df.columns:
        return {}
    counts = df[col].astype(str).value_counts(dropna=False).head(top_k)
    return {str(k): int(v) for k, v in counts.items()}


def _numeric_stats(df: pd.DataFrame, cols: list[str]) -> dict[str, dict[str, float | int | None]]:
    out: dict[str, dict[str, float | int | None]] = {}
    for col in cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").astype(float).dropna()
        if s.empty:
            out[col] = {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "p50": None,
                "p75": None,
                "p90": None,
                "p95": None,
                "p99": None,
                "max": None,
            }
            continue
        out[col] = {
            "count": int(s.count()),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if s.count() > 1 else 0.0,
            "min": float(s.min()),
            "p50": float(s.quantile(0.50)),
            "p75": float(s.quantile(0.75)),
            "p90": float(s.quantile(0.90)),
            "p95": float(s.quantile(0.95)),
            "p99": float(s.quantile(0.99)),
            "max": float(s.max()),
        }
    return out


def _quantile_compare_by_label(df: pd.DataFrame, cols: list[str]) -> dict[str, dict[str, dict[str, float | int | None]]]:
    out: dict[str, dict[str, dict[str, float | int | None]]] = {}
    if "true_label" not in df.columns:
        return out

    for label_value in ["fall", "non_fall"]:
        subset = df[df["true_label"].astype(str) == label_value]
        out[label_value] = _numeric_stats(subset, cols)
    return out


def main() -> int:
    args = parse_args()
    run_dir = _resolve(args.run_dir)
    if run_dir is None or not run_dir.exists():
        print(f"ERROR: run directory not found: {run_dir}")
        return 1

    false_alarm_csv = _resolve(args.false_alarms_csv) if args.false_alarms_csv else (run_dir / "false_alarms.csv")
    test_pred_csv = _resolve(args.test_predictions_csv) if args.test_predictions_csv else (run_dir / "test_predictions_windows.csv")

    if false_alarm_csv is None or not false_alarm_csv.exists():
        print(f"ERROR: false alarms CSV not found: {false_alarm_csv}")
        return 1

    false_alarm_df = pd.read_csv(false_alarm_csv)
    test_pred_df = pd.read_csv(test_pred_csv) if test_pred_csv is not None and test_pred_csv.exists() else None

    report = {
        "analysis_name": "lesson7_false_alarm_analysis",
        "run_dir": str(run_dir),
        "false_alarms_csv": str(false_alarm_csv),
        "test_predictions_csv": str(test_pred_csv) if test_pred_df is not None else None,
        "rows": {
            "false_alarms": int(len(false_alarm_df)),
            "test_predictions": int(len(test_pred_df)) if test_pred_df is not None else None,
        },
        "top_sources": {
            "subject_id": _top_counts(false_alarm_df, "subject_id", args.top_k),
            "session_id": _top_counts(false_alarm_df, "session_id", args.top_k),
            "source_file": _top_counts(false_alarm_df, "source_file", args.top_k),
        },
        "false_alarm_numeric_summary": _numeric_stats(false_alarm_df, KEY_NUMERIC_FIELDS),
        "label_quantile_comparison": _quantile_compare_by_label(test_pred_df, KEY_NUMERIC_FIELDS) if test_pred_df is not None else {},
        "notes": [],
    }

    if test_pred_df is None:
        report["notes"].append("test_predictions_windows.csv not found; quantile comparison by true_label was skipped")

    output_path = _resolve(args.output) if args.output else (run_dir / "false_alarm_analysis.json")
    assert output_path is not None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_safe(report), indent=2, default=str) + "\n", encoding="utf-8")

    print(f"false_alarms_rows={len(false_alarm_df)}")
    print("top_subjects:", report["top_sources"]["subject_id"])
    print("top_sessions:", report["top_sources"]["session_id"])
    print(f"saved_report={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
