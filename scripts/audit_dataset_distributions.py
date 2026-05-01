#!/usr/bin/env python3
"""Audit cross-dataset sensor distributions before model retraining.

Purpose:
- load existing ingested datasets using the current loaders
- summarize label balance, subjects, sessions, placements, sampling rates
- profile accel/gyro magnitude scales
- flag likely cross-dataset scale mismatch before we add normalization
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.ingest import load_mobifall, load_pamap2, load_sisfall, load_uci_har

LoaderFn = Callable[[str | Path], pd.DataFrame]

QUANTILES = (0.01, 0.05, 0.50, 0.75, 0.90, 0.95, 0.99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit dataset distributions for cross-dataset generalization work")
    parser.add_argument("--mobifall", default=None, help="Path to MobiFall dataset root/file")
    parser.add_argument("--sisfall", default=None, help="Path to SisFall dataset root/file")
    parser.add_argument("--pamap2", default=None, help="Path to PAMAP2 dataset root/file")
    parser.add_argument("--uci-har", dest="uci_har", default=None, help="Optional UCI HAR path")
    parser.add_argument(
        "--out-json",
        default="results/validation/dataset_distribution_audit.json",
        help="Where to save the audit JSON",
    )
    return parser.parse_args()


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _available_axes(df: pd.DataFrame, axes: tuple[str, str, str]) -> list[str]:
    cols: list[str] = []
    for col in axes:
        if col in df.columns:
            numeric = _safe_numeric(df[col])
            if numeric.notna().any():
                cols.append(col)
    return cols


def _vector_magnitude(df: pd.DataFrame, axes: tuple[str, str, str]) -> pd.Series:
    cols = _available_axes(df, axes)
    if not cols:
        return pd.Series(dtype=float)

    squared = pd.DataFrame({col: _safe_numeric(df[col]).pow(2) for col in cols})
    return np.sqrt(squared.sum(axis=1))


def _quantile_dict(series: pd.Series) -> dict[str, float | None]:
    numeric = _safe_numeric(series).replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return {f"q{int(q * 100):02d}": None for q in QUANTILES}
    return {f"q{int(q * 100):02d}": float(numeric.quantile(q)) for q in QUANTILES}


def _value_counts_dict(series: pd.Series, *, limit: int = 20) -> dict[str, int]:
    counts = series.astype("string").fillna("<NA>").value_counts(dropna=False).head(limit)
    return {str(k): int(v) for k, v in counts.items()}


def _sampling_rate_summary(df: pd.DataFrame) -> dict[str, float | None]:
    if "sampling_rate_hz" not in df.columns:
        return {"median": None, "min": None, "max": None}

    numeric = _safe_numeric(df["sampling_rate_hz"]).dropna()
    if numeric.empty:
        return {"median": None, "min": None, "max": None}

    return {
        "median": float(numeric.median()),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
    }


def _timestamp_span_summary(df: pd.DataFrame) -> dict[str, float | None]:
    if "timestamp" not in df.columns:
        return {"min": None, "max": None, "span": None}

    numeric = _safe_numeric(df["timestamp"]).dropna()
    if numeric.empty:
        return {"min": None, "max": None, "span": None}

    tmin = float(numeric.min())
    tmax = float(numeric.max())
    return {
        "min": tmin,
        "max": tmax,
        "span": float(tmax - tmin),
    }


def _infer_accel_scale_hint(acc_mag_quantiles: dict[str, float | None]) -> str:
    q50 = acc_mag_quantiles.get("q50")
    q99 = acc_mag_quantiles.get("q99")

    if q50 is None or q99 is None:
        return "unknown"

    q50f = float(q50)
    q99f = float(q99)

    if q50f < 5.0 and q99f < 30.0:
        return "g_like_or_small_scale"
    if 5.0 <= q50f <= 30.0 and q99f < 120.0:
        return "m_s2_like"
    return "large_or_raw_scale"


def _infer_gyro_scale_hint(gyro_mag_quantiles: dict[str, float | None]) -> str:
    q50 = gyro_mag_quantiles.get("q50")
    q99 = gyro_mag_quantiles.get("q99")

    if q50 is None or q99 is None:
        return "unknown"

    q50f = float(q50)
    q99f = float(q99)

    if q99f < 25.0:
        return "small_scale_or_rad_s_like"
    if q99f < 1000.0:
        return "deg_s_like_or_moderate_scale"
    return "large_or_raw_scale"


def summarize_dataset(name: str, df: pd.DataFrame) -> dict[str, Any]:
    acc_mag = _vector_magnitude(df, ("ax", "ay", "az"))
    gyro_mag = _vector_magnitude(df, ("gx", "gy", "gz"))

    summary = {
        "dataset_name": name,
        "rows": int(len(df)),
        "subjects": int(df["subject_id"].nunique(dropna=True)) if "subject_id" in df.columns else 0,
        "sessions": int(df["session_id"].nunique(dropna=True)) if "session_id" in df.columns else 0,
        "task_types": _value_counts_dict(df["task_type"]) if "task_type" in df.columns else {},
        "label_raw_counts": _value_counts_dict(df["label_raw"]) if "label_raw" in df.columns else {},
        "label_mapped_counts": _value_counts_dict(df["label_mapped"]) if "label_mapped" in df.columns else {},
        "placement_counts": _value_counts_dict(df["placement"]) if "placement" in df.columns else {},
        "sampling_rate_hz": _sampling_rate_summary(df),
        "timestamp_span": _timestamp_span_summary(df),
        "missing_ratio": {
            col: float(df[col].isna().mean())
            for col in ["timestamp", "ax", "ay", "az", "gx", "gy", "gz", "sampling_rate_hz"]
            if col in df.columns
        },
        "acc_magnitude_quantiles": _quantile_dict(acc_mag),
        "gyro_magnitude_quantiles": _quantile_dict(gyro_mag),
    }

    summary["accel_scale_hint"] = _infer_accel_scale_hint(summary["acc_magnitude_quantiles"])
    summary["gyro_scale_hint"] = _infer_gyro_scale_hint(summary["gyro_magnitude_quantiles"])

    return summary


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _loader_specs(args: argparse.Namespace) -> list[tuple[str, str | None, LoaderFn]]:
    return [
        ("MOBIFALL", args.mobifall, load_mobifall),
        ("SISFALL", args.sisfall, load_sisfall),
        ("PAMAP2", args.pamap2, load_pamap2),
        ("UCI_HAR", args.uci_har, load_uci_har),
    ]


def _comparison_block(dataset_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    medians: dict[str, float] = {}
    for name, summary in dataset_summaries.items():
        q50 = summary.get("acc_magnitude_quantiles", {}).get("q50")
        if q50 is not None:
            medians[name] = float(q50)

    ratio = None
    if len(medians) >= 2:
        min_v = min(medians.values())
        max_v = max(medians.values())
        if min_v > 0:
            ratio = float(max_v / min_v)

    return {
        "acc_magnitude_q50_by_dataset": medians,
        "max_over_min_acc_q50_ratio": ratio,
        "possible_cross_dataset_scale_issue": bool(ratio is not None and ratio >= 3.0),
    }


def main() -> int:
    args = parse_args()

    dataset_summaries: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}

    for name, raw_path, loader in _loader_specs(args):
        if not raw_path:
            continue

        path = Path(raw_path)
        if not path.exists():
            errors[name] = f"path not found: {path}"
            continue

        try:
            df = loader(path)
            dataset_summaries[name] = summarize_dataset(name, df)
        except Exception as exc:  # noqa: BLE001
            errors[name] = str(exc)

    out = {
        "datasets": dataset_summaries,
        "comparison": _comparison_block(dataset_summaries),
        "errors": errors,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_safe(out), indent=2), encoding="utf-8")

    print(f"Saved audit JSON to: {out_path}")
    print()

    for name, summary in dataset_summaries.items():
        acc_q = summary["acc_magnitude_quantiles"]
        gyro_q = summary["gyro_magnitude_quantiles"]
        print(f"=== {name} ===")
        print(f"rows: {summary['rows']}")
        print(f"subjects: {summary['subjects']}")
        print(f"sessions: {summary['sessions']}")
        print(f"sampling_rate_hz: {summary['sampling_rate_hz']}")
        print(f"acc_q50/q95/q99: {acc_q['q50']} / {acc_q['q95']} / {acc_q['q99']}")
        print(f"gyro_q50/q95/q99: {gyro_q['q50']} / {gyro_q['q95']} / {gyro_q['q99']}")
        print(f"accel_scale_hint: {summary['accel_scale_hint']}")
        print(f"gyro_scale_hint: {summary['gyro_scale_hint']}")
        print(f"label_mapped_counts: {summary['label_mapped_counts']}")
        print()

    if out["comparison"]["possible_cross_dataset_scale_issue"]:
        print("WARNING: cross-dataset accel median ratio suggests a likely scale mismatch.")

    if errors:
        print("Errors:")
        for name, err in errors.items():
            print(f"  - {name}: {err}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())