#!/usr/bin/env python3
"""Print and optionally export preprocessing/window summaries for a dataset CSV.

This utility is intended for debugging and dissertation evidence capture.
It supports:
- common-schema CSV files (already normalized)
- Chapter 3 fixture/extract CSVs via dataset-specific loaders

Outputs:
- concise console summary
- optional JSON export with preprocessing and window metadata summaries
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.ingest import load_mobifall, load_pamap2, load_sisfall, load_uci_har
from pipeline.preprocess import (
    PreprocessConfig,
    append_derived_channels,
    estimate_sampling_rate,
    resample_dataframe,
    window_dataframe,
)
from pipeline.validation import validate_ingestion_dataframe

LoaderFn = Callable[[str | Path], pd.DataFrame]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print preprocessing/window summary for a CSV")
    parser.add_argument("path", help="Path to CSV file")
    parser.add_argument(
        "--loader",
        default="common",
        choices=["common", "uci_har", "pamap2", "mobifall", "sisfall"],
        help="Use a Chapter 3 dataset loader or treat the file as a common-schema CSV",
    )
    parser.add_argument("--target-rate", type=float, default=50.0, help="Resampling target rate (Hz)")
    parser.add_argument("--window-size", type=int, default=None, help="Window size in samples (defaults to fixture-friendly fallback)")
    parser.add_argument("--step-size", type=int, default=None, help="Step size in samples (defaults to ~50%% overlap)")
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to write JSON summary")
    parser.add_argument("--limit-window-preview", type=int, default=5, help="Number of window metadata rows to keep in JSON preview")
    return parser.parse_args()


def _loaders() -> dict[str, LoaderFn]:
    return {
        "uci_har": load_uci_har,
        "pamap2": load_pamap2,
        "mobifall": load_mobifall,
        "sisfall": load_sisfall,
    }


def _resolve_input_path(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p


def _fixture_friendly_window_sizes(
    df: pd.DataFrame,
    cfg: PreprocessConfig,
    window_size: int | None,
    step_size: int | None,
) -> tuple[int, int]:
    n_rows = len(df)
    group_cols = [c for c in ["dataset_name", "subject_id", "session_id", "source_file"] if c in df.columns]
    min_group_size: int | None = None
    if group_cols and n_rows > 0:
        try:
            min_group_size = int(df.groupby(group_cols, dropna=False, sort=False).size().min())
        except Exception:
            min_group_size = None

    if window_size is not None:
        return window_size, step_size or max(1, window_size // 2)
    if n_rows >= cfg.window_size_samples:
        # If the dataframe contains multiple short sequences, use the smallest group size
        # so grouped windowing still produces at least one debug window per sequence.
        if min_group_size is not None and min_group_size < cfg.window_size_samples:
            w = max(2, min(min_group_size, 8))
            s = step_size or max(1, w // 2)
            return w, s
        return cfg.window_size_samples, step_size or cfg.step_size_samples
    # Debug-friendly fallback for short fixture/extract files.
    if min_group_size is not None:
        w = max(2, min(8, min_group_size))
    else:
        w = max(4, min(8, n_rows))
    s = step_size or max(1, w // 2)
    return w, s


def _window_summary_df(windows: list[dict]) -> pd.DataFrame:
    rows = []
    for w in windows:
        q = w.get("quality_summary", {}) or {}
        rows.append(
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
    return pd.DataFrame(rows)


def _load_dataframe(path: Path, loader_name: str) -> pd.DataFrame:
    if loader_name == "common":
        return pd.read_csv(path)
    loader = _loaders()[loader_name]
    return loader(path)


def _json_safe_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _series_counts(series: pd.Series) -> dict[str, int]:
    counts = series.astype(str).value_counts(dropna=False)
    return {str(k): int(v) for k, v in counts.items()}


def build_summary(
    df_raw: pd.DataFrame,
    df_resampled: pd.DataFrame,
    windows: list[dict],
    *,
    source_path: Path,
    loader_name: str,
    target_rate: float,
    validation,
    window_size: int,
    step_size: int,
    limit_window_preview: int,
) -> dict:
    ws = _window_summary_df(windows)
    window_preview = ws.head(limit_window_preview).to_dict(orient="records") if not ws.empty else []

    return {
        "source": {
            "path": str(source_path),
            "loader": loader_name,
        },
        "preprocessing": {
            "target_sampling_rate_hz": float(target_rate),
            "estimated_sampling_rate_hz_raw": estimate_sampling_rate(df_raw),
            "rows_before": int(len(df_raw)),
            "rows_after_resample": int(len(df_resampled)),
            "window_size_samples": int(window_size),
            "step_size_samples": int(step_size),
        },
        "validation": {
            "is_valid": bool(validation.is_valid),
            "errors": list(validation.errors),
            "warnings": list(validation.warnings),
        },
        "raw_summary": {
            "columns": list(df_raw.columns),
            "null_counts": {str(k): int(v) for k, v in df_raw.isna().sum().items()},
            "dataset_names": [str(v) for v in sorted(df_raw["dataset_name"].dropna().astype(str).unique().tolist())] if "dataset_name" in df_raw.columns else [],
            "task_types": [str(v) for v in sorted(df_raw["task_type"].dropna().astype(str).unique().tolist())] if "task_type" in df_raw.columns else [],
            "subjects_count": int(df_raw["subject_id"].nunique(dropna=True)) if "subject_id" in df_raw.columns else 0,
            "label_raw_counts": _series_counts(df_raw["label_raw"]) if "label_raw" in df_raw.columns else {},
            "label_mapped_counts": _series_counts(df_raw["label_mapped"]) if "label_mapped" in df_raw.columns else {},
        },
        "resampled_summary": {
            "columns": list(df_resampled.columns),
            "null_counts": {str(k): int(v) for k, v in df_resampled.isna().sum().items()},
            "subjects_count": int(df_resampled["subject_id"].nunique(dropna=True)) if "subject_id" in df_resampled.columns else 0,
            "label_raw_counts": _series_counts(df_resampled["label_raw"]) if "label_raw" in df_resampled.columns else {},
            "label_mapped_counts": _series_counts(df_resampled["label_mapped"]) if "label_mapped" in df_resampled.columns else {},
        },
        "window_summary": {
            "window_count": int(len(windows)),
            "accepted_count": int(sum(1 for w in windows if bool(w.get("is_acceptable")))),
            "rejected_count": int(sum(1 for w in windows if not bool(w.get("is_acceptable")))),
            "majority_label_counts": _series_counts(ws["label_mapped_majority"]) if not ws.empty else {},
            "acceptability_counts": _series_counts(ws["is_acceptable"].astype(str)) if not ws.empty else {},
            "preview": [
                {str(k): _json_safe_value(v) for k, v in row.items()}
                for row in window_preview
            ],
        },
    }


def main() -> int:
    args = parse_args()
    input_path = _resolve_input_path(args.path)
    if not input_path.exists():
        print(f"ERROR: file not found: {input_path}")
        return 1

    df_raw = _load_dataframe(input_path, args.loader)
    validation = validate_ingestion_dataframe(df_raw)

    df_resampled = resample_dataframe(df_raw, target_rate_hz=args.target_rate)
    df_resampled = append_derived_channels(df_resampled, include_acc=True, include_gyro=True)

    cfg = PreprocessConfig(target_sampling_rate_hz=args.target_rate)
    window_size, step_size = _fixture_friendly_window_sizes(df_resampled, cfg, args.window_size, args.step_size)
    windows = window_dataframe(df_resampled, window_size=window_size, step_size=step_size, config=cfg)

    summary = build_summary(
        df_raw=df_raw,
        df_resampled=df_resampled,
        windows=windows,
        source_path=input_path,
        loader_name=args.loader,
        target_rate=args.target_rate,
        validation=validation,
        window_size=window_size,
        step_size=step_size,
        limit_window_preview=args.limit_window_preview,
    )

    print(f"source: {summary['source']['path']}")
    print(f"loader: {summary['source']['loader']}")
    print(f"validation: {'PASS' if summary['validation']['is_valid'] else 'FAIL'}")
    if summary["validation"]["warnings"]:
        for warning in summary["validation"]["warnings"]:
            print(f"warning: {warning}")
    print(
        "rows:",
        f"{summary['preprocessing']['rows_before']} -> {summary['preprocessing']['rows_after_resample']}",
        f"(resampled @ {summary['preprocessing']['target_sampling_rate_hz']} Hz)",
    )
    print(
        "windows:",
        f"{summary['window_summary']['window_count']} total,",
        f"{summary['window_summary']['accepted_count']} accepted,",
        f"{summary['window_summary']['rejected_count']} rejected",
    )
    print("label_mapped_counts (resampled):", summary["resampled_summary"]["label_mapped_counts"])
    print("window_majority_label_counts:", summary["window_summary"]["majority_label_counts"])

    if args.json_out:
        out_path = Path(args.json_out)
        if not out_path.is_absolute():
            out_path = (REPO_ROOT / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"json_out: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
