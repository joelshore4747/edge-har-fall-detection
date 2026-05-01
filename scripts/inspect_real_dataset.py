#!/usr/bin/env python3
"""Deep inspection utility for one dataset/path at a time.

This script is intended for manual debugging and dissertation evidence capture.
It reuses the existing loaders and Chapter 3 preprocessing modules to answer:
- did the loader interpret the path correctly?
- do subject/session/source labels look sensible?
- does grouped resampling/windowing behave sensibly on this sample?
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.ingest import load_mobifall, load_pamap2, load_sisfall, load_uci_har  # noqa: E402
from pipeline.preprocess import (  # noqa: E402
    PreprocessConfig,
    append_acc_magnitude,
    append_derived_channels,
    estimate_sampling_rate,
    resample_dataframe,
    window_dataframe,
)
from pipeline.validation import validate_ingestion_dataframe  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one real dataset path with loader + preprocessing summaries")
    parser.add_argument("--dataset", required=True, choices=["uci_har", "pamap2", "mobifall", "sisfall", "common"])
    parser.add_argument("--path", required=True, help="Dataset file or root path")
    parser.add_argument("--sample-limit", type=int, default=3, help="Limit files/windows loaded for real dataset loaders")
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="For PAMAP2 dataset roots, also include Optional/ subject*.dat files (Protocol is default).",
    )
    parser.add_argument("--target-rate", type=float, default=50.0)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument("--head", type=int, default=10)
    parser.add_argument("--plot", action="store_true", help="Plot raw vs resampled ax for one sequence")
    parser.add_argument("--no-show", action="store_true", help="Do not show plot window (useful with --plot in headless runs)")
    parser.add_argument("--save-plot", default=None, help="Optional output image path for plot")
    return parser.parse_args()


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p


def _load_dataset(args: argparse.Namespace, path: Path) -> pd.DataFrame:
    if args.dataset == "common":
        return pd.read_csv(path)
    if args.dataset == "uci_har":
        return load_uci_har(path, max_windows_per_split=args.sample_limit)
    if args.dataset == "pamap2":
        max_files = None if args.sample_limit <= 0 else args.sample_limit
        return load_pamap2(path, max_files=max_files, include_optional=args.include_optional)
    if args.dataset == "mobifall":
        return load_mobifall(path, max_files=args.sample_limit)
    if args.dataset == "sisfall":
        return load_sisfall(path, max_files=args.sample_limit)
    raise ValueError(args.dataset)


def _window_sizes(df: pd.DataFrame, cfg: PreprocessConfig, window_size: int | None, step_size: int | None) -> tuple[int, int]:
    if window_size is not None:
        return int(window_size), int(step_size or max(1, window_size // 2))
    group_cols = [c for c in ["dataset_name", "subject_id", "session_id", "source_file"] if c in df.columns]
    min_group_size = None
    if group_cols and len(df) > 0:
        try:
            min_group_size = int(df.groupby(group_cols, dropna=False, sort=False).size().min())
        except Exception:
            min_group_size = None

    if min_group_size is not None:
        w = max(2, min(32, min_group_size))
    else:
        w = max(2, min(32, len(df)))
    s = int(step_size or max(1, w // 2))
    return w, s


def _window_summary_table(windows: list[dict]) -> pd.DataFrame:
    rows = []
    for w in windows:
        rows.append(
            {
                "window_id": w.get("window_id"),
                "dataset_name": w.get("dataset_name"),
                "subject_id": w.get("subject_id"),
                "session_id": w.get("session_id"),
                "source_file": w.get("source_file"),
                "task_type": w.get("task_type"),
                "start_ts": w.get("start_ts"),
                "end_ts": w.get("end_ts"),
                "label_mapped_majority": w.get("label_mapped_majority"),
                "n_samples": w.get("n_samples"),
                "missing_ratio": w.get("missing_ratio"),
                "is_acceptable": w.get("is_acceptable"),
                "has_large_gap": w.get("has_large_gap"),
                "n_gaps": w.get("n_gaps"),
            }
        )
    return pd.DataFrame(rows)


def _maybe_plot_sequence(
    raw_df: pd.DataFrame,
    resampled_df: pd.DataFrame,
    *,
    save_plot: str | None,
    no_show: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: matplotlib unavailable, skipping plot: {exc}")
        return

    group_cols = [c for c in ["dataset_name", "subject_id", "session_id", "source_file"] if c in raw_df.columns]
    if group_cols:
        first_key, raw_seq = next(iter(raw_df.groupby(group_cols, dropna=False, sort=False)))
        if not isinstance(first_key, tuple):
            first_key = (first_key,)
        key_map = dict(zip(group_cols, first_key))
        mask = pd.Series(True, index=resampled_df.index)
        for col, val in key_map.items():
            mask &= (resampled_df[col] == val)
        res_seq = resampled_df[mask].copy()
    else:
        raw_seq = raw_df.copy()
        res_seq = resampled_df.copy()

    raw_seq = raw_seq.sort_values([c for c in ["timestamp", "row_index"] if c in raw_seq.columns], kind="stable")
    res_seq = res_seq.sort_values([c for c in ["timestamp", "row_index"] if c in res_seq.columns], kind="stable")

    raw_seq = append_acc_magnitude(raw_seq)
    res_seq = append_acc_magnitude(res_seq)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    axes[0].plot(raw_seq["timestamp"], raw_seq["ax"], marker="o", linestyle="-", label="raw ax")
    axes[0].plot(res_seq["timestamp"], res_seq["ax"], marker="x", linestyle="--", label="resampled ax")
    axes[0].set_title("Raw vs Resampled ax (first sequence)")
    axes[0].set_xlabel("timestamp (s)")
    axes[0].set_ylabel("ax")
    axes[0].legend()

    axes[1].plot(res_seq["timestamp"], res_seq["acc_magnitude"], marker="o", linestyle="-", label="resampled acc_magnitude")
    axes[1].set_title("Resampled Acceleration Magnitude (first sequence)")
    axes[1].set_xlabel("timestamp (s)")
    axes[1].set_ylabel("acc_magnitude")
    axes[1].legend()

    fig.tight_layout()
    if save_plot:
        out_path = Path(save_plot)
        if not out_path.is_absolute():
            out_path = (REPO_ROOT / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot: {out_path}")
    if no_show:
        plt.close(fig)
    else:
        plt.show()


def main() -> int:
    args = parse_args()
    path = _resolve_path(args.path)
    if not path.exists():
        print(f"ERROR: path not found: {path}")
        return 1

    try:
        df = _load_dataset(args, path)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: loader failed ({args.dataset}): {type(exc).__name__}: {exc}")
        return 1

    validation = validate_ingestion_dataframe(df)
    cfg = PreprocessConfig(target_sampling_rate_hz=args.target_rate)
    resampled = resample_dataframe(df, target_rate_hz=args.target_rate)
    resampled = append_derived_channels(resampled)
    w_size, s_size = _window_sizes(resampled, cfg, args.window_size, args.step_size)
    windows = window_dataframe(resampled, window_size=w_size, step_size=s_size, config=cfg)
    window_summary_df = _window_summary_table(windows)

    print(f"dataset={args.dataset}")
    print(f"path={path}")
    print(f"rows={len(df)} cols={len(df.columns)}")
    print(f"validation={'PASS' if validation.is_valid else 'FAIL'}")
    for warning in validation.warnings:
        print(f"warning: {warning}")
    for error in validation.errors:
        print(f"error: {error}")
    if bool(getattr(df, 'attrs', {}).get('is_prewindowed', False)):
        print(f"note: prewindowed_source={df.attrs.get('prewindowed_source')}")

    print("\ncolumns:")
    print(list(df.columns))
    print("\nnull_counts:")
    print(df.isna().sum().sort_values(ascending=False))

    if "subject_id" in df.columns:
        print(f"\nsubjects_count={df['subject_id'].nunique(dropna=True)}")
        print("subjects_sample:", df["subject_id"].astype(str).drop_duplicates().head(args.head).tolist())
    if "session_id" in df.columns:
        print(f"sessions_count={df['session_id'].nunique(dropna=True)}")
        print("sessions_sample:", df["session_id"].astype(str).drop_duplicates().head(args.head).tolist())
    if "source_file" in df.columns:
        print(f"source_files_count={df['source_file'].nunique(dropna=True)}")
        print("source_files_sample:", df["source_file"].astype(str).drop_duplicates().head(args.head).tolist())
    if "label_raw" in df.columns:
        print("label_raw_counts:", df["label_raw"].astype(str).value_counts(dropna=False).head(20).to_dict())
    if "label_mapped" in df.columns:
        print("label_mapped_counts:", df["label_mapped"].astype(str).value_counts(dropna=False).head(20).to_dict())

    est_rate = estimate_sampling_rate(df)
    print(f"\nestimated_sampling_rate_hz={est_rate}" if est_rate is not None else "\nestimated_sampling_rate_hz=unknown")
    print(f"rows_before_resampling={len(df)}")
    print(f"rows_after_resampling={len(resampled)}")
    print(f"window_size_samples={w_size}")
    print(f"step_size_samples={s_size}")
    print(f"windows_total={len(windows)}")
    print(f"windows_accepted={sum(1 for w in windows if bool(w.get('is_acceptable')))}")
    print(f"windows_rejected={sum(1 for w in windows if not bool(w.get('is_acceptable')))}")

    print("\nhead:")
    print(df.head(args.head))
    print("\nwindow_summary_head:")
    if window_summary_df.empty:
        print("(no windows)")
    else:
        print(window_summary_df.head(args.head))

    if args.plot:
        _maybe_plot_sequence(df, resampled, save_plot=args.save_plot, no_show=args.no_show)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
