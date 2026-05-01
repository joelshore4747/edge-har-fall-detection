"""Evaluation helpers for the Chapter 5 threshold-based fall detector."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from metrics.fall_false_alarms import build_false_alarm_table
from metrics.fall_metrics import compute_fall_detection_metrics
from pipeline.fall.features import extract_fall_window_features
from pipeline.fall.threshold_detector import FallThresholdConfig, detect_fall_from_features


PREDICTION_METADATA_COLUMNS = [
    "window_id",
    "dataset_name",
    "subject_id",
    "session_id",
    "source_file",
    "task_type",
    "start_ts",
    "end_ts",
    "n_samples",
    "is_acceptable",
    "true_label",
    "predicted_label",
]


def _prediction_row_from_window(
    window: dict[str, Any],
    *,
    detector_config: FallThresholdConfig,
    default_sampling_rate_hz: float | None,
) -> dict[str, Any]:
    features = extract_fall_window_features(
        window,
        default_sampling_rate_hz=default_sampling_rate_hz,
        post_impact_skip_samples=detector_config.post_impact_skip_samples,
    )
    decision = detect_fall_from_features(features, detector_config)

    row = {
        "window_id": window.get("window_id"),
        "dataset_name": window.get("dataset_name"),
        "subject_id": window.get("subject_id"),
        "session_id": window.get("session_id"),
        "source_file": window.get("source_file"),
        "task_type": window.get("task_type"),
        "start_ts": window.get("start_ts"),
        "end_ts": window.get("end_ts"),
        "n_samples": int(window.get("n_samples", 0) or 0),
        "missing_ratio": float(window.get("missing_ratio", 0.0) or 0.0),
        "is_acceptable": bool(window.get("is_acceptable", False)),
        "has_large_gap": bool(window.get("has_large_gap", False)),
        "n_gaps": int(window.get("n_gaps", 0) or 0),
        "true_label": str(window.get("label_mapped_majority")),
        "predicted_label": decision["predicted_label"],
        "detector_reason": decision["detector_reason"],
        "stage_impact_pass": bool(decision["stage_impact_pass"]),
        "stage_support_pass": bool(decision["stage_support_pass"]),
        "stage_confirm_pass": bool(decision["stage_confirm_pass"]),
    }
    # Flatten feature values and the key support/confirm checks for false-alarm inspection.
    row.update(features)
    for k, v in (decision.get("support_checks") or {}).items():
        row[f"support_{k}"] = bool(v)
    for k, v in (decision.get("confirm_checks") or {}).items():
        row[f"confirm_{k}"] = bool(v)
    return row


def build_threshold_prediction_table(
    windows: list[dict[str, Any]],
    *,
    detector_config: FallThresholdConfig,
    filter_unacceptable: bool = True,
    default_sampling_rate_hz: float | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window in windows:
        if filter_unacceptable and not bool(window.get("is_acceptable", False)):
            continue
        rows.append(
            _prediction_row_from_window(
                window,
                detector_config=detector_config,
                default_sampling_rate_hz=default_sampling_rate_hz,
            )
        )

    if not rows:
        return pd.DataFrame(columns=PREDICTION_METADATA_COLUMNS)

    df = pd.DataFrame(rows)
    meta_cols = [c for c in PREDICTION_METADATA_COLUMNS if c in df.columns]
    extra_cols = [c for c in df.columns if c not in meta_cols]
    return df[meta_cols + sorted(extra_cols)].copy()


def _group_labels(df: pd.DataFrame) -> pd.Series:
    dataset = df["dataset_name"].astype(str) if "dataset_name" in df.columns else pd.Series(["UNKNOWN"] * len(df), index=df.index)
    subject = df["subject_id"].astype(str) if "subject_id" in df.columns else pd.Series(["UNKNOWN_SUBJECT"] * len(df), index=df.index)
    return (dataset.astype(str) + "::" + subject.astype(str)).astype("string")


def split_fall_predictions_by_subject(
    pred_df: pd.DataFrame,
    *,
    test_size: float = 0.30,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if pred_df.empty:
        raise ValueError("Prediction table is empty")

    if "subject_id" not in pred_df.columns or pred_df["subject_id"].dropna().nunique() < 2:
        test_df = pred_df.reset_index(drop=True).copy()
        train_df = pred_df.iloc[0:0].copy()
        split_summary = {
            "strategy": "all_data_fallback_insufficient_subject_groups",
            "train_rows": 0,
            "test_rows": int(len(test_df)),
            "train_subject_groups": [],
            "test_subject_groups": sorted(_group_labels(test_df).astype(str).unique().tolist()),
            "train_subjects_count": 0,
            "test_subjects_count": int(test_df["subject_id"].nunique(dropna=True)) if "subject_id" in test_df.columns else 0,
            "test_size": float(test_size),
            "random_state": int(random_state),
            "note": "Subject-aware split unavailable (insufficient subject groups); evaluated on all available windows.",
        }
        return train_df, test_df, split_summary

    groups = _group_labels(pred_df)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(pred_df, pred_df.get("true_label"), groups=groups))
    train_df = pred_df.iloc[sorted(train_idx)].reset_index(drop=True)
    test_df = pred_df.iloc[sorted(test_idx)].reset_index(drop=True)

    split_summary = {
        "strategy": "group_shuffle_split_by_subject",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_subject_groups": sorted(_group_labels(train_df).astype(str).unique().tolist()),
        "test_subject_groups": sorted(_group_labels(test_df).astype(str).unique().tolist()),
        "train_subjects_count": int(train_df["subject_id"].nunique(dropna=True)) if "subject_id" in train_df.columns else 0,
        "test_subjects_count": int(test_df["subject_id"].nunique(dropna=True)) if "subject_id" in test_df.columns else 0,
        "test_size": float(test_size),
        "random_state": int(random_state),
    }
    return train_df, test_df, split_summary


def evaluate_threshold_fall_predictions(
    pred_df: pd.DataFrame,
    *,
    test_size: float = 0.30,
    random_state: int = 42,
    positive_label: str = "fall",
    negative_label: str = "non_fall",
) -> dict[str, Any]:
    if pred_df.empty:
        raise ValueError("Prediction table is empty")
    if "true_label" not in pred_df.columns or "predicted_label" not in pred_df.columns:
        raise ValueError("Prediction table must include true_label and predicted_label")

    train_df, test_df, split_summary = split_fall_predictions_by_subject(
        pred_df,
        test_size=test_size,
        random_state=random_state,
    )

    y_true = test_df["true_label"].astype(str).tolist()
    y_pred = test_df["predicted_label"].astype(str).tolist()
    metrics = compute_fall_detection_metrics(
        y_true,
        y_pred,
        positive_label=positive_label,
        negative_label=negative_label,
    )
    false_alarm_df = build_false_alarm_table(test_df)

    split_summary["train_label_counts"] = train_df["true_label"].astype(str).value_counts(dropna=False).to_dict() if not train_df.empty else {}
    split_summary["test_label_counts"] = test_df["true_label"].astype(str).value_counts(dropna=False).to_dict()

    return {
        "split": split_summary,
        "metrics": metrics,
        "train_predictions": train_df,
        "test_predictions": test_df,
        "false_alarms": false_alarm_df,
    }
