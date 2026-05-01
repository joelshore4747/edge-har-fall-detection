"""Inference utilities for exported fall meta-model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def load_fall_model_artifact(path: str | Path) -> dict[str, Any]:
    """Load a previously exported fall model artifact."""
    artifact = joblib.load(Path(path))
    if not isinstance(artifact, dict):
        raise ValueError("Loaded fall artifact is not a dictionary")

    required = {
        "model",
        "used_features",
        "positive_label",
        "negative_label",
        "probability_threshold",
    }
    missing = required - set(artifact.keys())
    if missing:
        raise ValueError(f"Fall artifact missing required keys: {sorted(missing)}")

    return artifact


def _coerce_boolean_like_to_float(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(float)

    lowered = series.astype(str).str.lower()
    if lowered.isin({"true", "false"}).all():
        return lowered.map({"true": 1.0, "false": 0.0}).astype(float)

    return pd.to_numeric(series, errors="coerce")


def _coerce_feature_frame(
    feature_df: pd.DataFrame,
    used_features: list[str],
) -> pd.DataFrame:
    """Align incoming data to the trained fall artifact schema."""
    if feature_df.empty:
        raise ValueError("feature_df is empty")

    X = pd.DataFrame(index=feature_df.index)

    for col in used_features:
        if col in feature_df.columns:
            X[col] = _coerce_boolean_like_to_float(feature_df[col])
        else:
            X[col] = np.nan

    return X


def predict_fall_with_artifact(
    feature_df: pd.DataFrame,
    *,
    artifact: dict[str, Any],
) -> pd.DataFrame:
    """Run fall inference using a loaded artifact.

    Returns a DataFrame containing:
    - predicted_probability
    - predicted_label
    - predicted_is_fall
    - probability_threshold_used
    """
    required = {
        "model",
        "used_features",
        "positive_label",
        "negative_label",
        "probability_threshold",
    }
    missing = required - set(artifact.keys())
    if missing:
        raise ValueError(f"Fall artifact missing required keys: {sorted(missing)}")

    model = artifact["model"]
    used_features = [str(x) for x in artifact["used_features"]]
    positive_label = str(artifact["positive_label"])
    negative_label = str(artifact["negative_label"])
    probability_threshold = float(artifact["probability_threshold"])

    X = _coerce_feature_frame(feature_df, used_features)

    if not hasattr(model, "predict_proba"):
        raise ValueError("Fall model does not support predict_proba")

    proba = model.predict_proba(X)
    model_classes = [int(c) for c in getattr(model, "classes_", [0, 1])]

    if 1 not in model_classes:
        raise ValueError("Fall model classes do not contain positive class '1'")

    positive_index = model_classes.index(1)
    positive_prob = proba[:, positive_index]

    predicted_is_fall = positive_prob >= probability_threshold
    predicted_label = np.where(predicted_is_fall, positive_label, negative_label)

    out = pd.DataFrame(index=feature_df.index)
    out["predicted_probability"] = positive_prob
    out["predicted_label"] = pd.Series(predicted_label, index=feature_df.index, dtype="string")
    out["predicted_is_fall"] = pd.Series(predicted_is_fall, index=feature_df.index, dtype=bool)
    out["probability_threshold_used"] = probability_threshold

    return out


def predict_fall_from_artifact_path(
    feature_df: pd.DataFrame,
    *,
    artifact_path: str | Path,
) -> pd.DataFrame:
    """Load a fall artifact from disk and run inference."""
    artifact = load_fall_model_artifact(artifact_path)
    return predict_fall_with_artifact(feature_df, artifact=artifact)