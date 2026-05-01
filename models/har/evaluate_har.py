from __future__ import annotations

from typing import Iterable

import pandas as pd

from metrics.classification import compute_classification_metrics
from models.har.baselines import (
    DEFAULT_HAR_LABEL_ORDER,
    get_feature_importances_dataframe,
    heuristic_har_predict,
    train_random_forest_classifier,
)
from models.har.train_har import (
    SUBJECT_AWARE_SPLIT_STRATEGY,
    build_group_labels,
    prepare_feature_matrices,
    subject_aware_group_split,
)

EXPLICIT_SPLIT_STRATEGY = "explicit_train_test_split"


def _labels_for_eval(feature_df: pd.DataFrame, labels: Iterable[str] | None = None) -> list[str]:
    if labels is not None:
        return [str(v) for v in labels]
    seen = set(feature_df["label_mapped_majority"].astype(str).tolist()) if "label_mapped_majority" in feature_df.columns else set()
    ordered = [label for label in DEFAULT_HAR_LABEL_ORDER if label in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def _run_har_baselines_for_split(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    labels: Iterable[str] | None,
    random_state: int,
    split_strategy: str,
    split_extra: dict | None = None,
) -> dict:
    if train_df[label_col].astype(str).nunique(dropna=True) < 2:
        raise ValueError("Training split has fewer than 2 classes")
    if test_df[label_col].astype(str).nunique(dropna=True) < 1:
        raise ValueError("Test split has no classes")

    label_source = pd.concat([train_df[[label_col]], test_df[[label_col]]], ignore_index=True)
    label_order = _labels_for_eval(label_source.rename(columns={label_col: "label_mapped_majority"}), labels=labels)

    y_test = test_df[label_col].astype(str)
    heuristic_pred = heuristic_har_predict(test_df).astype(str)
    heuristic_metrics = compute_classification_metrics(y_test.tolist(), heuristic_pred.tolist(), labels=label_order)

    X_train, X_test, y_train, y_test_ml, feature_cols, imputer_fill_values = prepare_feature_matrices(
        train_df,
        test_df,
        label_col=label_col,
    )

    rf_model = train_random_forest_classifier(X_train, y_train, random_state=random_state)
    rf_pred = pd.Series(rf_model.predict(X_test), index=test_df.index, dtype="string")
    rf_metrics = compute_classification_metrics(y_test_ml.tolist(), rf_pred.astype(str).tolist(), labels=label_order)
    rf_feature_importances = get_feature_importances_dataframe(rf_model, feature_cols)

    train_groups = build_group_labels(train_df)
    test_groups = build_group_labels(test_df)
    train_label_counts = train_df[label_col].astype(str).value_counts(dropna=False).to_dict()
    test_label_counts = test_df[label_col].astype(str).value_counts(dropna=False).to_dict()

    split_payload = {
        "strategy": split_strategy,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_subject_groups": sorted(train_groups.astype(str).unique().tolist()),
        "test_subject_groups": sorted(test_groups.astype(str).unique().tolist()),
        "train_subjects_count": int(train_df["subject_id"].nunique(dropna=True)) if "subject_id" in train_df.columns else 0,
        "test_subjects_count": int(test_df["subject_id"].nunique(dropna=True)) if "subject_id" in test_df.columns else 0,
        "train_label_counts": {str(k): int(v) for k, v in train_label_counts.items()},
        "test_label_counts": {str(k): int(v) for k, v in test_label_counts.items()},
    }
    if split_extra:
        split_payload.update(split_extra)

    return {
        "split": split_payload,
        "label_order": label_order,
        "feature_columns": feature_cols,
        "imputer_fill_values": imputer_fill_values,
        "heuristic": {
            "metrics": heuristic_metrics,
            "predictions_preview": heuristic_pred.head(10).astype(str).tolist(),
        },
        "random_forest": {
            "metrics": rf_metrics,
            "predictions_preview": rf_pred.head(10).astype(str).tolist(),
            "feature_importances": rf_feature_importances,
        },
        "train_feature_table": train_df,
        "test_feature_table": test_df,
        "rf_model": rf_model,
    }


def run_har_baselines_on_train_test_feature_tables(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    label_col: str = "label_mapped_majority",
    labels: Iterable[str] | None = None,
    random_state: int = 42,
    split_extra: dict | None = None,
) -> dict:
    if train_df.empty:
        raise ValueError("Training feature table is empty")
    if test_df.empty:
        raise ValueError("Test feature table is empty")
    if label_col not in train_df.columns or label_col not in test_df.columns:
        raise ValueError(f"Feature tables must contain label column '{label_col}'")

    return _run_har_baselines_for_split(
        train_df=train_df,
        test_df=test_df,
        label_col=label_col,
        labels=labels,
        random_state=random_state,
        split_strategy=EXPLICIT_SPLIT_STRATEGY,
        split_extra=split_extra,
    )


def run_har_baselines_on_feature_table(
    feature_df: pd.DataFrame,
    *,
    label_col: str = "label_mapped_majority",
    test_size: float = 0.30,
    random_state: int = 42,
    labels: Iterable[str] | None = None,
) -> dict:
    if feature_df.empty:
        raise ValueError("Feature table is empty")
    if label_col not in feature_df.columns:
        raise ValueError(f"Feature table missing label column '{label_col}'")

    train_df, test_df = subject_aware_group_split(
        feature_df,
        label_col=label_col,
        test_size=test_size,
        random_state=random_state,
    )

    return _run_har_baselines_for_split(
        train_df=train_df,
        test_df=test_df,
        label_col=label_col,
        labels=labels,
        random_state=random_state,
        split_strategy=SUBJECT_AWARE_SPLIT_STRATEGY,
        split_extra={
            "test_size": float(test_size),
            "random_state": int(random_state),
        },
    )
