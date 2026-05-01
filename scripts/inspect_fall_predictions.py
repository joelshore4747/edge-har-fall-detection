#!/usr/bin/env python3
"""Inspect saved fall-threshold prediction artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect fall-threshold prediction CSVs and metrics")
    parser.add_argument("--predictions-csv", required=True, help="Path to predictions_windows.csv or test_predictions_windows.csv")
    parser.add_argument("--metrics-json", default=None, help="Optional metrics.json path (defaults to sibling metrics.json)")
    parser.add_argument("--top", type=int, default=10)
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p


def main() -> int:
    args = parse_args()
    pred_path = _resolve(args.predictions_csv)
    if not pred_path.exists():
        print(f"ERROR: predictions CSV not found: {pred_path}")
        return 1

    metrics_path = _resolve(args.metrics_json) if args.metrics_json else pred_path.parent / "metrics.json"
    df = pd.read_csv(pred_path)
    print(f"predictions_csv={pred_path}")
    print(f"rows={len(df)} cols={len(df.columns)}")
    print("columns:", list(df.columns))

    if "true_label" in df.columns:
        print("true_label_counts:", df["true_label"].astype(str).value_counts(dropna=False).to_dict())
    if "predicted_label" in df.columns:
        print("predicted_label_counts:", df["predicted_label"].astype(str).value_counts(dropna=False).to_dict())

    if {"true_label", "predicted_label"}.issubset(df.columns):
        false_alarms = df[(df["predicted_label"].astype(str) == "fall") & (df["true_label"].astype(str) == "non_fall")]
        false_negatives = df[(df["predicted_label"].astype(str) == "non_fall") & (df["true_label"].astype(str) == "fall")]
        print(f"false_alarms={len(false_alarms)} false_negatives={len(false_negatives)}")
        if not false_alarms.empty:
            print("false_alarm_examples:")
            cols = [c for c in ["window_id", "subject_id", "session_id", "peak_acc", "post_impact_motion", "detector_reason"] if c in false_alarms.columns]
            print(false_alarms[cols].head(args.top))

    if metrics_path.exists():
        try:
            metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            print(f"\nmetrics_json={metrics_path}")
            metrics = metrics_payload.get("metrics", {})
            if metrics:
                print(
                    "metrics_summary:",
                    {
                        "accuracy": metrics.get("accuracy"),
                        "sensitivity": metrics.get("sensitivity"),
                        "specificity": metrics.get("specificity"),
                        "precision": metrics.get("precision"),
                        "f1": metrics.get("f1"),
                        "support_total": metrics.get("support_total"),
                    },
                )
            split = metrics_payload.get("split", {})
            if split:
                print("split_summary:", {k: split.get(k) for k in ["strategy", "train_rows", "test_rows", "train_subjects_count", "test_subjects_count"]})
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: failed to read metrics JSON {metrics_path}: {exc}")

    print("\nhead:")
    print(df.head(args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
