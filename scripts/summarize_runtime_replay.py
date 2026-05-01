#!/usr/bin/env python3
"""Summarize runtime replay outputs for quick inspection.

Reads:
- HAR replay CSV
- fall replay CSV
- combined timeline CSV

Prints:
- row counts
- HAR predicted label counts
- HAR confidence summary
- fall predicted label counts
- fall probability summary
- top fall-risk windows
- optional timeline preview

This is meant for unlabeled runtime/phone sessions where metrics like F1 are not meaningful.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize runtime replay outputs")
    parser.add_argument(
        "--har-csv",
        default="results/validation/phone1_har.csv",
        help="Path to HAR replay CSV",
    )
    parser.add_argument(
        "--fall-csv",
        default="results/validation/phone1_fall.csv",
        help="Path to fall replay CSV",
    )
    parser.add_argument(
        "--timeline-csv",
        default="results/validation/phone1_timeline.csv",
        help="Path to combined timeline CSV",
    )
    parser.add_argument(
        "--top-k-fall",
        type=int,
        default=10,
        help="How many top fall-probability rows to show",
    )
    parser.add_argument(
        "--timeline-preview",
        type=int,
        default=10,
        help="How many timeline rows to preview",
    )
    return parser.parse_args()


def _load_csv(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def _print_header(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _safe_counts(df: pd.DataFrame, col: str) -> pd.Series | None:
    if col not in df.columns:
        return None
    return df[col].astype(str).value_counts(dropna=False)


def _safe_numeric(df: pd.DataFrame, col: str) -> pd.Series | None:
    if col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def _print_har_summary(har_df: pd.DataFrame) -> None:
    _print_header("HAR summary")
    print(f"Rows: {len(har_df)}")

    counts = _safe_counts(har_df, "predicted_label")
    if counts is not None:
        print("Predicted label counts:")
        for label, count in counts.items():
            print(f"  {label}: {int(count)}")

    conf = _safe_numeric(har_df, "predicted_confidence")
    if conf is not None and conf.dropna().size > 0:
        print(
            "Confidence: "
            f"mean={conf.mean():.4f} "
            f"median={conf.median():.4f} "
            f"max={conf.max():.4f}"
        )

    if "session_id" in har_df.columns:
        session_counts = har_df["session_id"].astype(str).value_counts(dropna=False)
        print("Session counts:")
        for sid, count in session_counts.items():
            print(f"  {sid}: {int(count)}")


def _print_fall_summary(fall_df: pd.DataFrame, top_k: int) -> None:
    _print_header("Fall summary")
    print(f"Rows: {len(fall_df)}")

    counts = _safe_counts(fall_df, "predicted_label")
    if counts is not None:
        print("Predicted label counts:")
        for label, count in counts.items():
            print(f"  {label}: {int(count)}")

    prob = _safe_numeric(fall_df, "predicted_probability")
    if prob is not None and prob.dropna().size > 0:
        print(
            "Probability: "
            f"mean={prob.mean():.4f} "
            f"median={prob.median():.4f} "
            f"max={prob.max():.4f}"
        )

    if "predicted_is_fall" in fall_df.columns:
        pred_is_fall = fall_df["predicted_is_fall"]
        try:
            n_falls = int(pred_is_fall.astype(bool).sum())
        except Exception:
            n_falls = int(pred_is_fall.astype(str).str.lower().eq("true").sum())
        print(f"Predicted fall-positive windows: {n_falls}")

    if prob is not None and prob.dropna().size > 0:
        cols = [
            c for c in [
                "session_id",
                "midpoint_ts",
                "start_ts",
                "end_ts",
                "predicted_probability",
                "predicted_label",
                "predicted_is_fall",
            ]
            if c in fall_df.columns
        ]
        top_df = (
            fall_df.loc[:, cols]
            .assign(predicted_probability=pd.to_numeric(fall_df["predicted_probability"], errors="coerce"))
            .sort_values("predicted_probability", ascending=False, kind="stable")
            .head(top_k)
            .reset_index(drop=True)
        )
        print()
        print(f"Top {min(top_k, len(top_df))} fall-probability rows:")
        if top_df.empty:
            print("  none")
        else:
            print(top_df.to_string(index=False))


def _print_timeline_summary(timeline_df: pd.DataFrame, preview_rows: int) -> None:
    _print_header("Combined timeline summary")
    print(f"Rows: {len(timeline_df)}")

    if "har_predicted_label" in timeline_df.columns:
        counts = timeline_df["har_predicted_label"].astype(str).value_counts(dropna=False)
        print("HAR labels in timeline:")
        for label, count in counts.items():
            print(f"  {label}: {int(count)}")

    if "fall_predicted_label" in timeline_df.columns:
        counts = timeline_df["fall_predicted_label"].astype(str).value_counts(dropna=False)
        print("Fall labels in timeline:")
        for label, count in counts.items():
            print(f"  {label}: {int(count)}")

    preview_cols = [
        c for c in [
            "session_id",
            "midpoint_ts",
            "har_predicted_label",
            "har_predicted_confidence",
            "fall_predicted_label",
            "fall_predicted_probability",
            "fall_predicted_is_fall",
        ]
        if c in timeline_df.columns
    ]
    if preview_cols:
        print()
        print(f"Timeline preview (first {min(preview_rows, len(timeline_df))} rows):")
        print(timeline_df.loc[:, preview_cols].head(preview_rows).to_string(index=False))


def main() -> int:
    args = parse_args()

    har_df = _load_csv(args.har_csv)
    fall_df = _load_csv(args.fall_csv)
    timeline_df = _load_csv(args.timeline_csv)

    print("Runtime replay summary")
    print("======================")
    _print_har_summary(har_df)
    _print_fall_summary(fall_df, top_k=args.top_k_fall)
    _print_timeline_summary(timeline_df, preview_rows=args.timeline_preview)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())