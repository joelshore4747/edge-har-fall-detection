"""Inference utilities for exported HAR models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.har.train_har import load_har_model_artifact


def _coerce_feature_frame(feature_df: pd.DataFrame, feature_columns: list[str], fill_values: dict[str, float]) -> pd.DataFrame:
    """Align a feature table to the trained HAR artifact schema."""
    if feature_df.empty:
        raise ValueError("feature_df is empty")

    X = feature_df.copy()

    for col in feature_columns:
        if col not in X.columns:
            X[col] = np.nan

    X = X[feature_columns].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.fillna(fill_values).fillna(0.0)
    return X


def predict_har_with_artifact(
    feature_df: pd.DataFrame,
    *,
    artifact: dict[str, Any],
) -> pd.DataFrame:
    """Run HAR inference using a loaded artifact.

    Returns a DataFrame containing:
    - predicted_label
    - optional probability columns if the model supports predict_proba
    """
    if artifact.get("schema_version") == "v1":
        artifact = dict(artifact)
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

    model = artifact["model"]
    feature_columns = list(artifact["feature_columns"])
    fill_values = dict(artifact["fill_values"])
    label_order = [str(x) for x in artifact["label_order"]]

    X = _coerce_feature_frame(feature_df, feature_columns, fill_values)
    preds = pd.Series(model.predict(X), index=feature_df.index, dtype="string")

    out = pd.DataFrame(index=feature_df.index)
    out["predicted_label"] = preds

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        model_classes = [str(c) for c in getattr(model, "classes_", label_order)]

        for idx, class_name in enumerate(model_classes):
            out[f"proba_{class_name}"] = proba[:, idx]

        out["predicted_confidence"] = out[[f"proba_{c}" for c in model_classes]].max(axis=1)
    else:
        out["predicted_confidence"] = np.nan

    return out


def predict_har_from_artifact_path(
    feature_df: pd.DataFrame,
    *,
    artifact_path: str | Path,
) -> pd.DataFrame:
    """Load a HAR artifact from disk and run inference."""
    artifact = load_har_model_artifact(artifact_path)
    return predict_har_with_artifact(feature_df, artifact=artifact)