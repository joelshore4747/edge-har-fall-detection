from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from pipeline.preprocess.config import PreprocessConfig, default_preprocess_config
from pipeline.preprocess.quality import infer_active_sensor_columns, window_quality_summary

HAR_LABEL_PURITY_THRESHOLD = 0.90
FALL_LABEL_PURITY_THRESHOLD = 0.80
DEFAULT_LABEL_PURITY_THRESHOLD = 0.85


def _clean_labels(series: pd.Series) -> list[str]:
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .replace({"": pd.NA, "nan": pd.NA, "<na>": pd.NA, "none": pd.NA})
        .dropna()
    )
    return cleaned.astype(str).tolist()


def assign_majority_label(window_df: pd.DataFrame, label_col: str = "label_mapped") -> str:
    if label_col not in window_df.columns:
        raise ValueError(f"Label column '{label_col}' not found")

    labels = _clean_labels(window_df[label_col])
    if not labels:
        return "unknown"

    counts = Counter(labels)
    max_count = max(counts.values())
    tied = {label for label, count in counts.items() if count == max_count}

    # Deterministic tie-breaker: earliest occurrence in the window.
    for label in labels:
        if label in tied:
            return label
    return labels[0]


def _majority_fraction(window_df: pd.DataFrame, label_col: str = "label_mapped") -> float:
    if label_col not in window_df.columns:
        return 0.0

    labels = _clean_labels(window_df[label_col])
    if not labels:
        return 0.0

    counts = Counter(labels)
    max_count = max(counts.values())
    return float(max_count / len(labels))


def sliding_window_indices(n_samples: int, window_size: int, step_size: int) -> list[tuple[int, int]]:
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if step_size <= 0:
        raise ValueError("step_size must be > 0")
    if n_samples < window_size:
        return []

    indices: list[tuple[int, int]] = []
    start = 0
    while start + window_size <= n_samples:
        end = start + window_size
        indices.append((start, end))
        start += step_size
    return indices


def _first_non_null(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return None
    return non_null.iloc[0]


def _window_sensor_payload(window_df: pd.DataFrame) -> dict[str, np.ndarray]:
    payload_cols = [
        c
        for c in ["ax", "ay", "az", "gx", "gy", "gz", "acc_magnitude", "gyro_magnitude"]
        if c in window_df.columns
    ]
    payload: dict[str, np.ndarray] = {}
    for col in payload_cols:
        payload[col] = pd.to_numeric(window_df[col], errors="coerce").to_numpy(dtype=float)
    return payload


def _default_window_group_cols(df: pd.DataFrame) -> list[str]:
    """Return logical sequence keys used to prevent cross-sequence windows."""
    candidates = ["dataset_name", "subject_id", "session_id", "source_file"]
    return [c for c in candidates if c in df.columns]


def _sort_for_windowing(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    if "timestamp" in working.columns:
        working["timestamp"] = pd.to_numeric(working["timestamp"], errors="coerce")
    sort_cols = [c for c in ["timestamp", "row_index"] if c in working.columns]
    if sort_cols:
        working = working.sort_values(sort_cols, kind="stable")
    return working.reset_index(drop=True)


def _task_label_purity_threshold(task_type: str | None) -> float:
    task = str(task_type or "").lower()
    if task == "har":
        return HAR_LABEL_PURITY_THRESHOLD
    if task == "fall":
        return FALL_LABEL_PURITY_THRESHOLD
    return DEFAULT_LABEL_PURITY_THRESHOLD


def window_dataframe(
    df: pd.DataFrame,
    window_size: int,
    step_size: int,
    label_col: str = "label_mapped",
    *,
    config: PreprocessConfig | None = None,
    group_cols: list[str] | None = None,
) -> list[dict[str, Any]]:
    if label_col not in df.columns:
        raise ValueError(f"window_dataframe requires label column '{label_col}'")

    cfg = config or default_preprocess_config()
    windows: list[dict[str, Any]] = []

    effective_group_cols = (
        _default_window_group_cols(df)
        if group_cols is None
        else [c for c in group_cols if c in df.columns]
    )

    if effective_group_cols:
        group_iter = df.groupby(effective_group_cols, dropna=False, sort=False)
    else:
        group_iter = [(None, df)]

    next_window_id = 0
    for _, group_df in group_iter:
        working = _sort_for_windowing(group_df)
        if working.empty:
            continue

        group_active_sensor_cols = infer_active_sensor_columns(working)
        indices = sliding_window_indices(
            len(working),
            window_size=window_size,
            step_size=step_size,
        )

        for start_idx, end_idx in indices:
            window_df = working.iloc[start_idx:end_idx].copy()
            q_summary = window_quality_summary(
                window_df,
                cfg,
                active_sensor_cols=group_active_sensor_cols,
            )

            start_ts = None
            end_ts = None
            midpoint_ts = None

            if "timestamp" in window_df.columns:
                ts = pd.to_numeric(window_df["timestamp"], errors="coerce")
                if ts.notna().any():
                    start_ts = float(ts.dropna().iloc[0])
                    end_ts = float(ts.dropna().iloc[-1])
                    midpoint_ts = (start_ts + end_ts) / 2.0

            task_type = _first_non_null(window_df["task_type"]) if "task_type" in window_df.columns else None
            mapped_majority = assign_majority_label(window_df, label_col=label_col)
            mapped_majority_fraction = _majority_fraction(window_df, label_col=label_col)

            raw_majority = None
            if "label_raw" in window_df.columns:
                raw_majority = assign_majority_label(window_df, label_col="label_raw")

            purity_threshold = _task_label_purity_threshold(task_type)
            label_pure_enough = mapped_majority_fraction >= purity_threshold

            # Preserve old field name while making it stricter.
            is_acceptable = bool(q_summary["is_acceptable"] and label_pure_enough)

            window_record = {
                "window_id": int(next_window_id),
                "dataset_name": _first_non_null(window_df["dataset_name"]) if "dataset_name" in window_df.columns else None,
                "subject_id": _first_non_null(window_df["subject_id"]) if "subject_id" in window_df.columns else None,
                "session_id": _first_non_null(window_df["session_id"]) if "session_id" in window_df.columns else None,
                "source_file": _first_non_null(window_df["source_file"]) if "source_file" in window_df.columns else None,
                "task_type": task_type,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "midpoint_ts": midpoint_ts,
                "label_raw_majority": raw_majority,
                "label_mapped_majority": mapped_majority,
                "label_majority_fraction": float(mapped_majority_fraction),
                "acceptable_window": bool(is_acceptable),
                "n_samples": int(len(window_df)),
                "missing_ratio": float(q_summary["missing_ratio"]),
                "is_acceptable": bool(is_acceptable),
                "has_large_gap": bool(q_summary["has_large_gap"]),
                "n_gaps": int(q_summary["n_gaps"]),
                "quality_summary": {
                    **q_summary,
                    "label_majority_fraction": float(mapped_majority_fraction),
                    "label_purity_threshold": float(purity_threshold),
                    "label_pure_enough": bool(label_pure_enough),
                },
                "sensor_payload": _window_sensor_payload(window_df),
            }
            windows.append(window_record)
            next_window_id += 1

    return windows