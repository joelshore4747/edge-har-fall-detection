"""Build window-level HAR feature tables from Chapter 3 window dictionaries."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from pipeline.features.frequency_domain import extract_frequency_features_for_window
from pipeline.features.magnitude_features import extract_magnitude_features_for_window
from pipeline.features.time_domain import extract_time_domain_features_for_window


FEATURE_TABLE_METADATA_COLUMNS = [
    "window_id",
    "dataset_name",
    "subject_id",
    "session_id",
    "source_file",
    "task_type",
    "label_mapped_majority",
    "is_acceptable",
]


def infer_window_sampling_rate_hz(
    window: dict[str, Any],
    *,
    default_sampling_rate_hz: float | None = None,
) -> float | None:
    if default_sampling_rate_hz is not None:
        try:
            default_value = float(default_sampling_rate_hz)
            if np.isfinite(default_value):
                return default_value
        except Exception:
            pass

    n_samples = int(window.get("n_samples", 0) or 0)
    start_ts = window.get("start_ts")
    end_ts = window.get("end_ts")
    if n_samples >= 2 and start_ts is not None and end_ts is not None:
        try:
            duration = float(end_ts) - float(start_ts)
            if duration > 0:
                # n samples over duration between first and last sample => n-1 intervals
                return float((n_samples - 1) / duration)
        except Exception:
            return None
    return None


def _window_metadata_row(window: dict[str, Any]) -> dict[str, Any]:
    row = {
        "window_id": window.get("window_id"),
        "dataset_name": window.get("dataset_name"),
        "subject_id": window.get("subject_id"),
        "session_id": window.get("session_id"),
        "source_file": window.get("source_file"),
        "task_type": window.get("task_type"),
        "label_mapped_majority": window.get("label_mapped_majority"),
        "is_acceptable": bool(window.get("is_acceptable", False)),
    }
    # Additional debugging/quality metadata is useful for filtering and audits.
    row["n_samples"] = int(window.get("n_samples", 0) or 0)
    row["missing_ratio"] = float(window.get("missing_ratio", np.nan))
    row["has_large_gap"] = bool(window.get("has_large_gap", False))
    row["n_gaps"] = int(window.get("n_gaps", 0) or 0)
    row["start_ts"] = window.get("start_ts")
    row["end_ts"] = window.get("end_ts")
    row["midpoint_ts"] = window.get("midpoint_ts")
    return row


def _extract_features_for_one_window(
    window: dict[str, Any],
    *,
    default_sampling_rate_hz: float | None = None,
) -> dict[str, float]:
    payload = window.get("sensor_payload", {}) or {}
    if not isinstance(payload, dict):
        raise ValueError("window['sensor_payload'] must be a dict of channel -> numeric array")

    sampling_rate_hz = infer_window_sampling_rate_hz(window, default_sampling_rate_hz=default_sampling_rate_hz)
    if sampling_rate_hz is None:
        # Frequency/jerk features depend on sample rate. Keep explicit NaNs if unavailable.
        sampling_rate_hz = np.nan

    features: dict[str, float] = {}
    features.update(extract_time_domain_features_for_window(payload))
    features.update(extract_magnitude_features_for_window(payload, sampling_rate_hz=float(sampling_rate_hz)))
    features.update(extract_frequency_features_for_window(payload, sampling_rate_hz=float(sampling_rate_hz)))
    features["window_sampling_rate_hz"] = float(sampling_rate_hz) if np.isfinite(sampling_rate_hz) else np.nan
    return features


def build_feature_table(
    windows: Iterable[dict[str, Any]],
    *,
    filter_unacceptable: bool = True,
    default_sampling_rate_hz: float | None = None,
) -> pd.DataFrame:
    """Build a feature table (one row per window) from Chapter 3 window dictionaries."""
    rows: list[dict[str, Any]] = []
    for window in windows:
        if filter_unacceptable and not bool(window.get("is_acceptable", False)):
            continue
        row = _window_metadata_row(window)
        row.update(_extract_features_for_one_window(window, default_sampling_rate_hz=default_sampling_rate_hz))
        rows.append(row)

    if not rows:
        base_cols = FEATURE_TABLE_METADATA_COLUMNS + [
            "n_samples",
            "missing_ratio",
            "has_large_gap",
            "n_gaps",
            "start_ts",
            "end_ts",
            "midpoint_ts",
        ]
        return pd.DataFrame(columns=base_cols)

    df = pd.DataFrame(rows)

    metadata_cols = [c for c in FEATURE_TABLE_METADATA_COLUMNS if c in df.columns]
    extra_meta = [c for c in ["n_samples", "missing_ratio", "has_large_gap", "n_gaps", "start_ts", "end_ts", "midpoint_ts"] if c in df.columns]
    feature_cols = sorted([c for c in df.columns if c not in metadata_cols + extra_meta])
    ordered_cols = metadata_cols + extra_meta + feature_cols
    return df[ordered_cols].copy()


def feature_table_schema_summary(feature_df: pd.DataFrame) -> dict[str, Any]:
    metadata_cols = [c for c in FEATURE_TABLE_METADATA_COLUMNS if c in feature_df.columns]
    extra_meta_cols = [c for c in ["n_samples", "missing_ratio", "has_large_gap", "n_gaps", "start_ts", "end_ts", "midpoint_ts"] if c in feature_df.columns]
    feature_cols = [c for c in feature_df.columns if c not in metadata_cols + extra_meta_cols]
    return {
        "rows": int(len(feature_df)),
        "columns_total": int(len(feature_df.columns)),
        "metadata_columns": metadata_cols,
        "extra_metadata_columns": extra_meta_cols,
        "feature_columns_count": int(len(feature_cols)),
        "feature_columns_preview": feature_cols[:20],
        "label_counts": (
            feature_df["label_mapped_majority"].astype(str).value_counts(dropna=False).to_dict()
            if "label_mapped_majority" in feature_df.columns
            else {}
        ),
        "subjects_count": (
            int(feature_df["subject_id"].nunique(dropna=True))
            if "subject_id" in feature_df.columns
            else 0
        ),
        "sessions_count": (
            int(feature_df["session_id"].nunique(dropna=True))
            if "session_id" in feature_df.columns
            else 0
        ),
        "datasets_present": (
            sorted(feature_df["dataset_name"].astype(str).dropna().unique().tolist())
            if "dataset_name" in feature_df.columns
            else []
        ),
    }