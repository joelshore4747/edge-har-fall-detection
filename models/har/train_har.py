from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from models.har.baselines import train_random_forest_classifier

SUBJECT_AWARE_SPLIT_STRATEGY = "group_shuffle_split_by_subject"
DEFAULT_HAR_ALLOWED_LABELS = ["static", "locomotion", "stairs", "other"]


FEATURE_TABLE_METADATA_EXCLUDE = {
    "window_id",
    "dataset_name",
    "subject_id",
    "session_id",
    "source_file",
    "task_type",
    "label_raw_majority",
    "label_mapped_majority",
    "label_majority_fraction",
    "is_acceptable",
    "acceptable_window",
    "n_samples",
    "missing_ratio",
    "has_large_gap",
    "n_gaps",
    "start_ts",
    "end_ts",
    "midpoint_ts",
    "sampling_rate_hz",
    "window_sampling_rate_hz",
}


def build_group_labels(feature_df: pd.DataFrame) -> pd.Series:
    dataset = (
        feature_df["dataset_name"].astype(str)
        if "dataset_name" in feature_df.columns
        else pd.Series(["UNKNOWN"] * len(feature_df), index=feature_df.index, dtype="string")
    )
    subject = (
        feature_df["subject_id"].astype(str)
        if "subject_id" in feature_df.columns
        else pd.Series(["UNKNOWN_SUBJECT"] * len(feature_df), index=feature_df.index, dtype="string")
    )
    return (dataset + "::" + subject).astype("string")

def filter_har_training_rows(
    feature_df: pd.DataFrame,
    *,
    label_col: str = "label_mapped_majority",
    allowed_labels: Iterable[str] | None = None,
    require_acceptable: bool = True,
) -> pd.DataFrame:
    if label_col not in feature_df.columns:
        raise ValueError(f"Missing label column '{label_col}'")

    working = feature_df.copy()

    labels = (
        working[label_col]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace({"": pd.NA, "nan": pd.NA, "<na>": pd.NA, "none": pd.NA, "unknown": pd.NA})
    )
    working[label_col] = labels
    working = working[working[label_col].notna()].copy()

    if require_acceptable and "is_acceptable" in working.columns:
        working = working[working["is_acceptable"].fillna(False).astype(bool)].copy()

    if allowed_labels is not None:
        allowed = {str(v).strip().lower() for v in allowed_labels}
        working = working[working[label_col].isin(allowed)].copy()

    return working.reset_index(drop=True)

def select_feature_columns(
    feature_df: pd.DataFrame,
    *,
    explicit_feature_cols: Iterable[str] | None = None,
) -> list[str]:
    if explicit_feature_cols is not None:
        cols = [c for c in explicit_feature_cols if c in feature_df.columns]
        if not cols:
            raise ValueError("No explicit feature columns found in feature table")
        return cols

    numeric_cols = feature_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in FEATURE_TABLE_METADATA_EXCLUDE]
    if not feature_cols:
        raise ValueError("No numeric feature columns found in feature table")
    return sorted(feature_cols)


def subject_aware_group_split(
    feature_df: pd.DataFrame,
    *,
    label_col: str = "label_mapped_majority",
    test_size: float = 0.30,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if label_col not in feature_df.columns:
        raise ValueError(f"Missing label column '{label_col}'")
    if "subject_id" not in feature_df.columns:
        raise ValueError("Subject-aware split requires 'subject_id' in feature table")
    if len(feature_df) < 2:
        raise ValueError("Need at least 2 rows for train/test split")

    groups = build_group_labels(feature_df)
    unique_groups = groups.nunique(dropna=True)
    if unique_groups < 2:
        raise ValueError("Need at least 2 distinct subject groups for subject-aware split")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(feature_df, feature_df[label_col], groups=groups))

    train_df = feature_df.iloc[np.sort(train_idx)].reset_index(drop=True)
    test_df = feature_df.iloc[np.sort(test_idx)].reset_index(drop=True)
    return train_df, test_df


def prepare_feature_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    label_col: str = "label_mapped_majority",
    feature_cols: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], dict[str, float]]:
    cols = select_feature_columns(train_df, explicit_feature_cols=feature_cols)
    cols = [c for c in cols if c in test_df.columns]
    if not cols:
        raise ValueError("No overlapping feature columns between train and test tables")

    X_train = train_df[cols].copy()
    X_test = test_df[cols].copy()
    y_train = train_df[label_col].astype(str).copy()
    y_test = test_df[label_col].astype(str).copy()

    fill_values = X_train.median(numeric_only=True).to_dict()
    fill_values = {str(k): float(v) for k, v in fill_values.items()}

    X_train = X_train.fillna(fill_values).fillna(0.0)
    X_test = X_test.fillna(fill_values).fillna(0.0)

    return X_train, X_test, y_train, y_test, cols, fill_values


def train_har_random_forest(
    feature_df: pd.DataFrame,
    *,
    label_col: str = "label_mapped_majority",
    feature_cols: Iterable[str] | None = None,
    random_state: int = 42,
    allowed_labels: Iterable[str] | None = None,
    require_acceptable: bool = True,
    rf_params: dict[str, Any] | None = None,
) -> tuple[Any, list[str], dict[str, float], pd.Series]:
    """Train a HAR Random Forest on the provided feature table."""
    if feature_df.empty:
        raise ValueError("feature_df is empty")
    if label_col not in feature_df.columns:
        raise ValueError(f"Missing label column '{label_col}'")
    if allowed_labels is None:
        allowed_labels = DEFAULT_HAR_ALLOWED_LABELS

    working = filter_har_training_rows(
        feature_df,
        label_col=label_col,
        allowed_labels=allowed_labels,
        require_acceptable=require_acceptable,
    )

    if working.empty:
        raise ValueError("No rows available for HAR training after filtering")

    cols = select_feature_columns(working, explicit_feature_cols=feature_cols)
    X = working[cols].copy()
    y = working[label_col].astype(str).copy()

    if y.nunique(dropna=True) < 2:
        raise ValueError("HAR training requires at least 2 distinct classes after filtering")

    fill_values = X.median(numeric_only=True).to_dict()
    fill_values = {str(k): float(v) for k, v in fill_values.items()}
    X = X.fillna(fill_values).fillna(0.0)

    model = train_random_forest_classifier(
        X,
        y,
        random_state=random_state,
        **(rf_params or {}),
    )
    return model, cols, fill_values, y

def _artifact_dataset_names(feature_df: pd.DataFrame) -> list[str]:
    if "dataset_name" not in feature_df.columns:
        return []
    vals = feature_df["dataset_name"].dropna().astype(str).unique().tolist()
    return sorted(vals)


def _artifact_label_order(y: pd.Series) -> list[str]:
    return sorted(pd.Series(y).dropna().astype(str).unique().tolist())


def train_and_export_har_model(
    feature_df: pd.DataFrame,
    *,
    output_path: str | Path,
    label_col: str = "label_mapped_majority",
    feature_cols: Iterable[str] | None = None,
    random_state: int = 42,
    metadata: dict[str, Any] | None = None,
    allowed_labels: Iterable[str] | None = None,
    require_acceptable: bool = True,
    rf_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model, cols, fill_values, y = train_har_random_forest(
        feature_df,
        label_col=label_col,
        feature_cols=feature_cols,
        random_state=random_state,
        allowed_labels=allowed_labels,
        require_acceptable=require_acceptable,
        rf_params=rf_params,
    )

    artifact_metadata: dict[str, Any] = {
        "task_type": "har",
        "label_col": label_col,
        "dataset_names": _artifact_dataset_names(feature_df),
        "n_rows": int(len(feature_df)),
        "n_features": int(len(cols)),
        "random_state": int(random_state),
    }
    if metadata:
        artifact_metadata.update(metadata)

    artifact = {
        "model_type": "random_forest",
        "task_type": "har",
        "model": model,
        "feature_columns": cols,
        "fill_values": fill_values,
        "label_order": _artifact_label_order(y),
        "metadata": artifact_metadata,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)

    return artifact


def load_har_model_artifact(path: str | Path) -> dict[str, Any]:
    artifact = joblib.load(Path(path))
    if not isinstance(artifact, dict):
        raise ValueError("Loaded HAR artifact is not a dictionary")
    if artifact.get("schema_version") == "v1":
        # unifallmonitor deployable bundle: route through the schema-aware
        # loader for strict validation (catches a missing metadata block,
        # which silently caused window_size=128 fallback in the 2026-04-30
        # runtime_v1 incident). The bundle keeps its native shape; we only
        # add the legacy compat keys downstream callers expect.
        from services.runtime_model_loader import load_bundle as _load_v1_bundle

        _load_v1_bundle(Path(path))  # raises with a clear error on malformed v1
        if "feature_columns" not in artifact and "feature_cols" in artifact:
            artifact["feature_columns"] = list(artifact["feature_cols"])
        if "label_order" not in artifact and "labels" in artifact:
            artifact["label_order"] = [str(x) for x in artifact["labels"]]
        if "fill_values" not in artifact:
            artifact["fill_values"] = {}
    required = {"model", "feature_columns", "fill_values", "label_order"}
    missing = required - set(artifact.keys())
    if missing:
        raise ValueError(f"HAR artifact missing required keys: {sorted(missing)}")
    return artifact
