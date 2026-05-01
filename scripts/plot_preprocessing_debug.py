#!/usr/bin/env python3
"""Visual debugging helper for Chapter 3 preprocessing.

Loads the irregular fixture CSV, prints intermediate inspection summaries, runs:
- resampling to 50 Hz
- acceleration magnitude derivation
- fixture-friendly sliding windows
- basic plots for raw vs resampled signals

This script is designed for local debugging and dissertation screenshots/workflow evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

# Make local package imports work regardless of the current working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.preprocess import (
    PreprocessConfig,
    append_acc_magnitude,
    estimate_sampling_rate,
    resample_dataframe,
    window_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot and inspect preprocessing steps on fixture data")
    parser.add_argument(
        "--input",
        type=str,
        default=str(REPO_ROOT / "tests/fixtures/timeseries_irregular.csv"),
        help="Path to CSV in common-schema style (defaults to irregular fixture)",
    )
    parser.add_argument("--target-rate", type=float, default=50.0, help="Resampling target rate in Hz")
    parser.add_argument("--window-size", type=int, default=4, help="Fixture-friendly window size in samples")
    parser.add_argument("--step-size", type=int, default=2, help="Fixture-friendly step size in samples")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive plot windows (useful in headless environments)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Optional directory to save plots (e.g., for dissertation figures/screenshots)",
    )
    return parser.parse_args()


def print_df_debug(name: str, df: pd.DataFrame, head_rows: int = 5) -> None:
    print(f"\n=== {name} ===")
    print(f"shape: {df.shape}")
    print("columns:")
    print(list(df.columns))
    print("null_counts:")
    print(df.isna().sum().sort_values(ascending=False))
    print(f"head({head_rows}):")
    print(df.head(head_rows))


def build_window_summary_frame(windows: list[dict]) -> pd.DataFrame:
    if not windows:
        return pd.DataFrame()

    summary_rows = []
    for w in windows:
        q = w.get("quality_summary", {}) or {}
        summary_rows.append(
            {
                "window_id": w.get("window_id"),
                "dataset_name": w.get("dataset_name"),
                "subject_id": w.get("subject_id"),
                "session_id": w.get("session_id"),
                "task_type": w.get("task_type"),
                "start_ts": w.get("start_ts"),
                "end_ts": w.get("end_ts"),
                "label_mapped_majority": w.get("label_mapped_majority"),
                "n_samples": w.get("n_samples"),
                "missing_ratio": w.get("missing_ratio"),
                "is_acceptable": w.get("is_acceptable"),
                "has_large_gap": q.get("has_large_gap"),
                "n_gaps": q.get("n_gaps"),
            }
        )
    return pd.DataFrame(summary_rows)


def plot_debug_views(raw_with_mag: pd.DataFrame, resampled_with_mag: pd.DataFrame) -> tuple[plt.Figure, plt.Figure]:
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(raw_with_mag["timestamp"], raw_with_mag["ax"], marker="o", linestyle="-", label="raw ax")
    ax1.plot(
        resampled_with_mag["timestamp"],
        resampled_with_mag["ax"],
        marker="x",
        linestyle="--",
        label="resampled ax (50 Hz)",
    )
    ax1.set_title("Raw vs Resampled ax")
    ax1.set_xlabel("timestamp (s)")
    ax1.set_ylabel("ax")
    ax1.legend()
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(
        resampled_with_mag["timestamp"],
        resampled_with_mag["acc_magnitude"],
        marker="o",
        linestyle="-",
        label="resampled acc_magnitude",
    )
    ax2.set_title("Resampled Acceleration Magnitude")
    ax2.set_xlabel("timestamp (s)")
    ax2.set_ylabel("acc_magnitude")
    ax2.legend()
    fig2.tight_layout()

    return fig1, fig2


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input CSV not found: {input_path}")
        return 1

    raw = pd.read_csv(input_path)
    print(f"repo_root: {REPO_ROOT}")
    print(f"input_path: {input_path.resolve()}")

    print_df_debug("Raw Data", raw)

    est_rate = estimate_sampling_rate(raw)
    if est_rate is None:
        print("estimated_sampling_rate_hz: unavailable")
    else:
        print(f"estimated_sampling_rate_hz: {est_rate:.3f}")

    resampled = resample_dataframe(raw, target_rate_hz=args.target_rate)
    raw_with_mag = append_acc_magnitude(raw)
    resampled_with_mag = append_acc_magnitude(resampled)

    print_df_debug("Resampled + Acc Magnitude", resampled_with_mag)

    cfg = PreprocessConfig(target_sampling_rate_hz=args.target_rate)
    windows = window_dataframe(
        resampled_with_mag,
        window_size=args.window_size,
        step_size=args.step_size,
        config=cfg,
    )
    window_summary_df = build_window_summary_frame(windows)

    print("\n=== Window Summary ===")
    if window_summary_df.empty:
        print("No windows generated (check window_size/step_size vs rows).")
    else:
        print(window_summary_df)
        print("\nwindow_label_counts:")
        print(window_summary_df["label_mapped_majority"].value_counts(dropna=False))
        print("\nwindow_acceptance_counts:")
        print(window_summary_df["is_acceptable"].value_counts(dropna=False))

    fig1, fig2 = plot_debug_views(raw_with_mag, resampled_with_mag)

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig1_path = save_dir / "raw_vs_resampled_ax.png"
        fig2_path = save_dir / "resampled_acc_magnitude.png"
        fig1.savefig(fig1_path, dpi=150)
        fig2.savefig(fig2_path, dpi=150)
        print(f"Saved plots to: {fig1_path} and {fig2_path}")

    if args.no_show:
        plt.close(fig1)
        plt.close(fig2)
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
