from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from pipeline.schema import COMMON_SCHEMA_COLUMNS

SENSOR_NUMERIC_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]
LABEL_COLS = ["label_raw", "label_mapped"]
METADATA_CONSTANT_COLS = [
    "dataset_name",
    "task_type",
    "subject_id",
    "session_id",
    "placement",
    "source_file",
]

SUPPORTED_INTERPOLATION_METHODS = {"linear", "nearest"}


def default_resample_group_cols(df: pd.DataFrame) -> list[str]:
    candidate = ["dataset_name", "task_type", "subject_id", "session_id", "source_file"]
    return [c for c in candidate if c in df.columns]


def default_rate_summary_group_cols(df: pd.DataFrame) -> list[str]:
    """Preferred keys for grouped sampling-rate estimation/reporting."""
    candidate = ["dataset_name", "subject_id", "session_id", "source_file"]
    return [c for c in candidate if c in df.columns]


def estimate_sampling_rate(df: pd.DataFrame, timestamp_col: str = "timestamp") -> float | None:
    if timestamp_col not in df.columns:
        return None

    ts = pd.to_numeric(df[timestamp_col], errors="coerce").dropna().sort_values(kind="stable")
    if len(ts) < 2:
        return None

    diffs = ts.diff().iloc[1:]
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return None

    median_dt = float(diffs.median())
    if median_dt <= 0:
        return None

    return float(1.0 / median_dt)


def summarize_sampling_rate_by_group(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    group_cols: Sequence[str] | None = None,
    preview_limit: int = 10,
) -> dict:
    base = {
        "groups_checked": 0,
        "median_hz": None,
        "min_hz": None,
        "max_hz": None,
        "estimated_rates_hz_preview": [],
    }
    if df.empty or timestamp_col not in df.columns:
        return base

    if group_cols is None:
        group_cols = default_rate_summary_group_cols(df)
    active_group_cols = [c for c in group_cols if c in df.columns]

    rates: list[float] = []
    if active_group_cols:
        for _, group_df in df.groupby(active_group_cols, dropna=False, sort=False):
            rate = estimate_sampling_rate(group_df, timestamp_col=timestamp_col)
            if rate is not None:
                rates.append(float(rate))
    else:
        rate = estimate_sampling_rate(df, timestamp_col=timestamp_col)
        if rate is not None:
            rates.append(float(rate))

    if not rates:
        return base

    series = pd.Series(rates, dtype=float)
    return {
        "groups_checked": int(len(rates)),
        "median_hz": float(series.median()),
        "min_hz": float(series.min()),
        "max_hz": float(series.max()),
        "estimated_rates_hz_preview": [round(float(v), 6) for v in rates[: max(0, int(preview_limit))]],
    }


def build_uniform_timeline(start_ts: float, end_ts: float, target_rate_hz: float) -> np.ndarray:
    if target_rate_hz <= 0:
        raise ValueError("target_rate_hz must be > 0")
    if end_ts < start_ts:
        raise ValueError("end_ts must be >= start_ts")

    step = 1.0 / float(target_rate_hz)
    timeline = np.arange(start_ts, end_ts + (step * 0.5), step, dtype=float)
    return timeline


def _validate_interpolation_method(method: str) -> str:
    clean = str(method).strip().lower()
    if clean not in SUPPORTED_INTERPOLATION_METHODS:
        supported = ", ".join(sorted(SUPPORTED_INTERPOLATION_METHODS))
        raise ValueError(f"Unsupported interpolation_method: {method!r}. Supported values: {supported}")
    return clean


def _nearest_indices(source_ts: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    """Return indices of nearest source timestamps for each target timestamp."""
    if source_ts.size == 0:
        raise ValueError("Cannot assign nearest labels from an empty source timestamp array")

    insert_pos = np.searchsorted(source_ts, target_ts, side="left")
    left_idx = np.clip(insert_pos - 1, 0, source_ts.size - 1)
    right_idx = np.clip(insert_pos, 0, source_ts.size - 1)

    left_dist = np.abs(target_ts - source_ts[left_idx])
    right_dist = np.abs(source_ts[right_idx] - target_ts)
    choose_right = right_dist < left_dist
    nearest = np.where(choose_right, right_idx, left_idx)
    return nearest.astype(int)


def _interpolate_numeric_series(
    source_ts: np.ndarray,
    source_values: np.ndarray,
    target_ts: np.ndarray,
    *,
    interpolation_method: str = "linear",
) -> np.ndarray:
    method = _validate_interpolation_method(interpolation_method)

    valid_mask = np.isfinite(source_ts) & np.isfinite(source_values)
    x = source_ts[valid_mask]
    y = source_values[valid_mask]

    if x.size == 0:
        return np.full_like(target_ts, np.nan, dtype=float)
    if x.size == 1:
        return np.full_like(target_ts, float(y[0]), dtype=float)

    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]

    if method == "linear":
        return np.interp(target_ts, x, y).astype(float)

    if method == "nearest":
        nearest_idx = _nearest_indices(x, target_ts)
        return y[nearest_idx].astype(float)

    raise ValueError(f"Unsupported interpolation_method: {interpolation_method!r}")


def _first_non_null(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return pd.NA
    return non_null.iloc[0]


def resample_group_to_rate(
    df: pd.DataFrame,
    target_rate_hz: float,
    *,
    timestamp_col: str = "timestamp",
    interpolation_method: str = "linear",
) -> pd.DataFrame:
    if timestamp_col not in df.columns:
        raise ValueError(f"Column '{timestamp_col}' is required for resampling")

    method = _validate_interpolation_method(interpolation_method)

    working = df.copy()
    working[timestamp_col] = pd.to_numeric(working[timestamp_col], errors="coerce")
    working = working.dropna(subset=[timestamp_col]).sort_values(timestamp_col, kind="stable").reset_index(drop=True)

    if working.empty:
        raise ValueError(
            "Resampling requires at least one non-null timestamp. "
            "Timestamp-free fallback is not implemented in Chapter 3."
        )

    working = working.drop_duplicates(subset=[timestamp_col], keep="first").reset_index(drop=True)

    from pipeline.preprocess.dejitter import drop_phantom_leading_samples
    working = drop_phantom_leading_samples(working, timestamp_col=timestamp_col)
    if working.empty:
        raise ValueError("Resampling requires at least one non-phantom sample")

    start_ts = float(working[timestamp_col].iloc[0])
    end_ts = float(working[timestamp_col].iloc[-1])
    uniform_ts = build_uniform_timeline(start_ts, end_ts, target_rate_hz)

    out = pd.DataFrame({timestamp_col: uniform_ts})

    src_ts = working[timestamp_col].to_numpy(dtype=float)

    for col in SENSOR_NUMERIC_COLS:
        if col in working.columns:
            values = pd.to_numeric(working[col], errors="coerce").to_numpy(dtype=float)
            out[col] = _interpolate_numeric_series(
                src_ts,
                values,
                uniform_ts,
                interpolation_method=method,
            )
        else:
            out[col] = np.nan

    nearest_idx = _nearest_indices(src_ts, uniform_ts)
    for col in LABEL_COLS:
        if col in working.columns:
            sampled = working.iloc[nearest_idx][col].reset_index(drop=True)
            out[col] = sampled.astype("string")
        else:
            out[col] = pd.Series([pd.NA] * len(out), dtype="string")

    for col in METADATA_CONSTANT_COLS:
        if col in working.columns:
            out[col] = _first_non_null(working[col])
        else:
            out[col] = pd.NA

    out["sampling_rate_hz"] = float(target_rate_hz)

    out["row_index"] = np.arange(len(out), dtype=int)

    ordered = [c for c in COMMON_SCHEMA_COLUMNS if c in out.columns]
    ordered.extend([c for c in out.columns if c not in ordered])
    out = out[ordered]
    return out


def resample_dataframe(
    df: pd.DataFrame,
    target_rate_hz: float,
    *,
    group_cols: list[str] | None = None,
    timestamp_col: str = "timestamp",
    interpolation_method: str = "linear",
) -> pd.DataFrame:
    method = _validate_interpolation_method(interpolation_method)

    if group_cols is None:
        group_cols = default_resample_group_cols(df)

    if not group_cols:
        return resample_group_to_rate(
            df,
            target_rate_hz=target_rate_hz,
            timestamp_col=timestamp_col,
            interpolation_method=method,
        )

    parts: list[pd.DataFrame] = []
    grouped = df.groupby(group_cols, dropna=False, sort=False)
    for _, group_df in grouped:
        parts.append(
            resample_group_to_rate(
                group_df,
                target_rate_hz=target_rate_hz,
                timestamp_col=timestamp_col,
                interpolation_method=method,
            )
        )

    if not parts:
        return df.iloc[0:0].copy()

    return pd.concat(parts, ignore_index=True)