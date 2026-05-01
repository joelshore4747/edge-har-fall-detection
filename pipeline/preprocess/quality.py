from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from pipeline.preprocess.config import PreprocessConfig

DEFAULT_ACCEL_COLS = ["ax", "ay", "az"]
DEFAULT_GYRO_COLS = ["gx", "gy", "gz"]
DEFAULT_SENSOR_COLS = DEFAULT_ACCEL_COLS + DEFAULT_GYRO_COLS


def _existing_sensor_cols(df: pd.DataFrame, sensor_cols: Iterable[str] | None = None) -> list[str]:
    requested = list(sensor_cols) if sensor_cols is not None else DEFAULT_SENSOR_COLS
    return [c for c in requested if c in df.columns]


def infer_active_sensor_columns(
    df: pd.DataFrame,
    *,
    accel_cols: Sequence[str] = tuple(DEFAULT_ACCEL_COLS),
    gyro_cols: Sequence[str] = tuple(DEFAULT_GYRO_COLS),
) -> list[str]:
    active: list[str] = []

    for col in accel_cols:
        if col in df.columns:
            active.append(col)

    present_gyro = [col for col in gyro_cols if col in df.columns]
    if present_gyro:
        has_usable_gyro = bool(df[present_gyro].notna().any().any())
        if has_usable_gyro:
            active.extend(present_gyro)

    return active


def compute_missing_ratio(df: pd.DataFrame, sensor_cols: Iterable[str] | None = None) -> float:
    cols = _existing_sensor_cols(df, sensor_cols)
    if len(df) == 0 or len(cols) == 0:
        return 1.0
    total_cells = len(df) * len(cols)
    missing_cells = int(df[cols].isna().sum().sum())
    return float(missing_cells / total_cells)


def detect_large_time_gaps(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    *,
    threshold_multiplier: float = 2.5,
) -> dict:
    empty_result = {
        "n_gaps": 0,
        "has_large_gap": False,
        "max_gap": None,
        "median_gap": None,
        "gap_indices": [],
        "threshold": None,
    }

    if timestamp_col not in df.columns or len(df) < 2:
        return empty_result

    ts = pd.to_numeric(df[timestamp_col], errors="coerce").dropna().sort_values(kind="stable").reset_index(drop=True)
    if len(ts) < 2:
        return empty_result

    diffs = ts.diff().iloc[1:]
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return empty_result

    median_gap = float(diffs.median())
    threshold = float(median_gap * threshold_multiplier)
    large_mask = diffs > threshold
    gap_indices = [int(i) for i in diffs.index[large_mask].tolist()]
    max_gap = float(diffs.max())

    return {
        "n_gaps": int(large_mask.sum()),
        "has_large_gap": bool(large_mask.any()),
        "max_gap": max_gap,
        "median_gap": median_gap,
        "gap_indices": gap_indices,
        "threshold": threshold,
    }


def window_quality_summary(
    window_df: pd.DataFrame,
    config: PreprocessConfig,
    *,
    active_sensor_cols: Iterable[str] | None = None,
) -> dict:
    used_sensor_cols = list(active_sensor_cols) if active_sensor_cols is not None else infer_active_sensor_columns(window_df)
    missing_ratio = compute_missing_ratio(window_df, sensor_cols=used_sensor_cols)
    gap_summary = detect_large_time_gaps(window_df)

    has_sensor_data = len(used_sensor_cols) > 0

    summary = {
        "n_samples": int(len(window_df)),
        "active_sensor_columns": used_sensor_cols,
        "has_sensor_data": bool(has_sensor_data),
        "missing_ratio": float(missing_ratio),
        "max_missing_ratio_allowed": float(config.max_missing_ratio_per_window),
        "has_large_gap": bool(gap_summary["has_large_gap"]),
        "n_gaps": int(gap_summary["n_gaps"]),
        "max_gap": gap_summary["max_gap"],
        "median_gap": gap_summary["median_gap"],
        "gap_threshold": gap_summary["threshold"],
    }
    summary["is_acceptable"] = bool(
        has_sensor_data
        and summary["missing_ratio"] <= config.max_missing_ratio_per_window
        and not summary["has_large_gap"]
    )
    return summary


def is_window_acceptable(
    window_df: pd.DataFrame,
    config: PreprocessConfig,
    *,
    active_sensor_cols: Iterable[str] | None = None,
) -> bool:
    return bool(window_quality_summary(window_df, config, active_sensor_cols=active_sensor_cols)["is_acceptable"])