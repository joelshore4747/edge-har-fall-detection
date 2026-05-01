"""HAR baseline models for Chapter 4.

Includes:
- a simple transparent heuristic baseline
- a RandomForestClassifier baseline (primary classical ML baseline)
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DEFAULT_HAR_LABEL_ORDER = ["static", "locomotion", "stairs", "other"]


def _series_or_nan(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([np.nan] * len(df), index=df.index, dtype=float)


def heuristic_har_predict(feature_df: pd.DataFrame) -> pd.Series:
    """A simple, transparent rule-based HAR baseline.

    The goal is explainability rather than high accuracy. Thresholds are deliberately
    coarse and operate on a small set of robust movement features.
    """
    acc_std = _series_or_nan(feature_df, ["acc_magnitude_std", "acc_magnitude_iqr", "ax_std"])
    acc_range = _series_or_nan(feature_df, ["acc_magnitude_range"])
    jerk = _series_or_nan(feature_df, ["acc_magnitude_jerk_mean_abs", "acc_magnitude_mean_abs_diff"])
    dom_f = _series_or_nan(feature_df, ["acc_magnitude_dominant_freq_hz"])
    az_std = _series_or_nan(feature_df, ["az_std"])

    preds: list[str] = []
    for idx in feature_df.index:
        s_std = float(acc_std.loc[idx]) if np.isfinite(acc_std.loc[idx]) else np.nan
        s_range = float(acc_range.loc[idx]) if np.isfinite(acc_range.loc[idx]) else np.nan
        s_jerk = float(jerk.loc[idx]) if np.isfinite(jerk.loc[idx]) else np.nan
        s_dom = float(dom_f.loc[idx]) if np.isfinite(dom_f.loc[idx]) else np.nan
        s_az_std = float(az_std.loc[idx]) if np.isfinite(az_std.loc[idx]) else np.nan

        # Static: low variability and low jerk.
        if np.isfinite(s_std) and s_std < 0.18 and (not np.isfinite(s_jerk) or s_jerk < 4.0):
            preds.append("static")
            continue

        locomotion_like = False
        if np.isfinite(s_std) and s_std >= 0.18:
            locomotion_like = True
        if np.isfinite(s_range) and s_range >= 0.35:
            locomotion_like = True
        if np.isfinite(s_dom) and 0.8 <= s_dom <= 4.5:
            locomotion_like = True

        if locomotion_like:
            # Crude stairs heuristic: stronger vertical variability or jerk than flat locomotion.
            if (np.isfinite(s_az_std) and s_az_std > 0.45) or (np.isfinite(s_jerk) and s_jerk > 10.0):
                preds.append("stairs")
            else:
                preds.append("locomotion")
            continue

        preds.append("other")

    return pd.Series(preds, index=feature_df.index, dtype="string")


def train_random_forest_classifier(
    X_train,
    y_train,
    *,
    random_state: int = 42,
    class_weight: str | None = "balanced_subsample",
    max_depth: int | None = None,
    min_samples_leaf: int = 2,
    sample_weight=None,
):
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=max_depth,
        min_samples_split=4,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        class_weight=class_weight,
        bootstrap=True,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def get_feature_importances_dataframe(
    model: RandomForestClassifier,
    feature_names: Iterable[str],
) -> pd.DataFrame:
    importances = getattr(model, "feature_importances_", None)
    feature_names = list(feature_names)
    if importances is None:
        return pd.DataFrame(columns=["feature", "importance"])

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": np.asarray(importances, dtype=float),
        }
    )
    return df.sort_values("importance", ascending=False, kind="stable").reset_index(drop=True)
