from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_META_FEATURE_COLUMNS = [
    # Core impact features
    "peak_acc",
    "peak_over_mean_ratio",
    "peak_minus_mean",
    "acc_variance",
    "mean_acc",
    "acc_baseline",
    # Dynamics / motion
    "jerk_peak",
    "jerk_mean",
    "jerk_rms",
    "gyro_peak",
    "gyro_mean",
    # Post-impact behaviour
    "post_impact_motion",
    "post_impact_variance",
    "post_impact_dyn_mean",
    "post_impact_dyn_rms",
    "post_impact_dyn_ratio_mean",
    "post_impact_dyn_ratio_rms",
    "post_impact_motion_to_peak_ratio",
    # Stage flags from threshold detector
    "stage_impact_pass",
    "stage_support_pass",
    "stage_confirm_pass",
]


@dataclass(slots=True)
class FallMetaModelConfig:
    """
    Configuration for the dissertation-grade second-stage fall meta-model.

    This model is designed to sit on top of the threshold detector output and
    learn a probabilistic mapping from engineered fall features to fall risk.
    """

    positive_label: str = "fall"
    negative_label: str = "non_fall"

    feature_columns: list[str] = field(default_factory=lambda: list(DEFAULT_META_FEATURE_COLUMNS))

    test_size: float = 0.30
    random_state: int = 42

    logistic_c: float = 1.0
    logistic_max_iter: int = 2000
    class_weight: str | dict[str, float] | None = "balanced"

    probability_threshold: float = 0.50

    # New: threshold tuning on a validation split of the outer training fold
    tune_probability_threshold: bool = True
    threshold_tuning_metric: str = "f1"
    threshold_grid: list[float] = field(
        default_factory=lambda: [round(x, 2) for x in np.arange(0.05, 0.96, 0.05).tolist()]
    )
    validation_size_within_train: float = 0.25


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _coerce_boolean_like_to_float(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(float)

    lowered = series.astype(str).str.lower()
    if lowered.isin({"true", "false"}).all():
        return lowered.map({"true": 1.0, "false": 0.0}).astype(float)

    return pd.to_numeric(series, errors="coerce")


def _prepare_feature_frame(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
) -> tuple[pd.DataFrame, list[str]]:
    available = [c for c in feature_columns if c in df.columns]
    if not available:
        raise ValueError("None of the requested meta-model feature columns are present in the dataframe")

    X = pd.DataFrame(index=df.index)
    for col in available:
        X[col] = _coerce_boolean_like_to_float(df[col])

    return X, available


def _get_subject_groups(df: pd.DataFrame) -> pd.Series:
    if "subject_id" not in df.columns:
        raise ValueError("Prediction dataframe must include subject_id for subject-aware splitting")

    dataset = (
        df["dataset_name"].astype(str)
        if "dataset_name" in df.columns
        else pd.Series(["UNKNOWN"] * len(df), index=df.index)
    )
    subject = df["subject_id"].astype(str)
    return (dataset + "::" + subject).astype("string")


def _split_subject_aware(
    df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    groups = _get_subject_groups(df)
    unique_groups = groups.dropna().unique().tolist()
    if len(unique_groups) < 2:
        raise ValueError("Need at least 2 subject groups for a subject-aware split")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=groups))

    train_df = df.iloc[sorted(train_idx)].reset_index(drop=True)
    test_df = df.iloc[sorted(test_idx)].reset_index(drop=True)

    split_summary = {
        "strategy": "group_shuffle_split_by_subject",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_subject_groups": sorted(_get_subject_groups(train_df).astype(str).unique().tolist()),
        "test_subject_groups": sorted(_get_subject_groups(test_df).astype(str).unique().tolist()),
        "train_subjects_count": int(train_df["subject_id"].nunique(dropna=True)),
        "test_subjects_count": int(test_df["subject_id"].nunique(dropna=True)),
        "test_size": float(test_size),
        "random_state": int(random_state),
    }
    return train_df, test_df, split_summary


def _binarise_labels(
    labels: pd.Series,
    *,
    positive_label: str,
    negative_label: str,
) -> np.ndarray:
    mapped = labels.astype(str).map(
        {
            positive_label: 1,
            negative_label: 0,
        }
    )
    if mapped.isna().any():
        bad = sorted(labels[mapped.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unexpected labels in training data: {bad}")
    return mapped.astype(int).to_numpy()


def _compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics: dict[str, Any] = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "support_total": int(len(y_true)),
        "support_positive": int(np.sum(y_true == 1)),
        "support_negative": int(np.sum(y_true == 0)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")

    return metrics


def _extract_logistic_coefficients(
    model: Pipeline,
    feature_names: list[str],
) -> pd.DataFrame:
    logistic = model.named_steps["logistic_regression"]
    coeffs = logistic.coef_.reshape(-1)

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coeffs,
            "abs_coefficient": np.abs(coeffs),
        }
    ).sort_values("abs_coefficient", ascending=False)

    return df.reset_index(drop=True)


def _build_model_pipeline(config: FallMetaModelConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    C=config.logistic_c,
                    max_iter=config.logistic_max_iter,
                    class_weight=config.class_weight,
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def _score_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    metric_name: str,
) -> float:
    y_pred = (y_prob >= threshold).astype(int)

    if metric_name == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if metric_name == "precision":
        return float(precision_score(y_true, y_pred, zero_division=0))
    if metric_name == "recall":
        return float(recall_score(y_true, y_pred, zero_division=0))
    if metric_name == "balanced_score":
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        return (precision + recall) / 2.0

    raise ValueError(f"Unsupported threshold_tuning_metric: {metric_name}")


def _select_probability_threshold(
    train_df: pd.DataFrame,
    *,
    config: FallMetaModelConfig,
) -> tuple[float, dict[str, Any]]:
    """
    Tune probability threshold on an inner subject-aware validation split.

    This avoids selecting the threshold directly on the outer test fold.
    """
    inner_train_df, val_df, val_split_summary = _split_subject_aware(
        train_df,
        test_size=config.validation_size_within_train,
        random_state=config.random_state,
    )

    X_inner_train, used_features = _prepare_feature_frame(inner_train_df, config.feature_columns)
    X_val, _ = _prepare_feature_frame(val_df, used_features)

    y_inner_train = _binarise_labels(
        inner_train_df["true_label"],
        positive_label=config.positive_label,
        negative_label=config.negative_label,
    )
    y_val = _binarise_labels(
        val_df["true_label"],
        positive_label=config.positive_label,
        negative_label=config.negative_label,
    )

    model = _build_model_pipeline(config)
    model.fit(X_inner_train, y_inner_train)

    y_val_prob = model.predict_proba(X_val)[:, 1]

    best_threshold = float(config.probability_threshold)
    best_score = float("-inf")
    threshold_rows: list[dict[str, Any]] = []

    for threshold in config.threshold_grid:
        score = _score_threshold(
            y_val,
            y_val_prob,
            threshold=float(threshold),
            metric_name=config.threshold_tuning_metric,
        )
        threshold_rows.append(
            {
                "threshold": float(threshold),
                "score": float(score),
            }
        )
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)

    threshold_table = pd.DataFrame(threshold_rows).sort_values(
        ["score", "threshold"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)

    tuning_summary = {
        "selected_threshold": float(best_threshold),
        "selected_score": float(best_score),
        "metric": config.threshold_tuning_metric,
        "validation_split": val_split_summary,
        "threshold_table": threshold_table,
        "used_features": used_features,
    }
    return best_threshold, tuning_summary


def train_fall_meta_model(
    prediction_df: pd.DataFrame,
    *,
    config: FallMetaModelConfig | None = None,
) -> dict[str, Any]:
    """
    Train a probabilistic second-stage fall model from threshold-pipeline output.

    Expected input:
    - one row per prediction window
    - includes subject_id, true_label, and engineered feature columns

    Returns:
    - fitted sklearn pipeline
    - split summary
    - metrics
    - prediction dataframe with probabilities
    - coefficient table
    """
    config = config or FallMetaModelConfig()

    if prediction_df.empty:
        raise ValueError("Prediction dataframe is empty")
    if "true_label" not in prediction_df.columns:
        raise ValueError("Prediction dataframe must contain true_label")

    train_df, test_df, split_summary = _split_subject_aware(
        prediction_df,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    selected_threshold = float(config.probability_threshold)
    threshold_tuning_summary: dict[str, Any] | None = None

    if config.tune_probability_threshold:
        train_groups = _get_subject_groups(train_df).nunique(dropna=True)
        if train_groups >= 2:
            selected_threshold, threshold_tuning_summary = _select_probability_threshold(
                train_df,
                config=config,
            )

    X_train_df, used_features = _prepare_feature_frame(train_df, config.feature_columns)
    X_test_df, _ = _prepare_feature_frame(test_df, used_features)

    y_train = _binarise_labels(
        train_df["true_label"],
        positive_label=config.positive_label,
        negative_label=config.negative_label,
    )
    y_test = _binarise_labels(
        test_df["true_label"],
        positive_label=config.positive_label,
        negative_label=config.negative_label,
    )

    model = _build_model_pipeline(config)
    model.fit(X_train_df, y_train)

    y_prob = model.predict_proba(X_test_df)[:, 1]
    y_pred = (y_prob >= selected_threshold).astype(int)

    metrics = _compute_binary_metrics(y_test, y_pred, y_prob)
    metrics["probability_threshold_used"] = float(selected_threshold)

    coefficient_df = _extract_logistic_coefficients(model, used_features)

    prediction_columns = [
        c
        for c in [
            "window_id",
            "dataset_name",
            "subject_id",
            "session_id",
            "source_file",
            "task_type",
            "true_label",
            "predicted_label",
            "detector_reason",
        ]
        if c in test_df.columns
    ]

    test_prediction_df = test_df[prediction_columns].copy()
    test_prediction_df["meta_probability"] = y_prob
    test_prediction_df["meta_probability_threshold_used"] = float(selected_threshold)
    test_prediction_df["meta_predicted_label"] = np.where(
        y_pred == 1,
        config.positive_label,
        config.negative_label,
    )
    test_prediction_df["meta_predicted_is_fall"] = y_pred.astype(bool)

    for col in used_features:
        if col in test_df.columns:
            test_prediction_df[col] = test_df[col]

    result = {
        "config": asdict(config),
        "used_features": used_features,
        "split": split_summary,
        "metrics": metrics,
        "coefficient_table": coefficient_df,
        "test_predictions": test_prediction_df,
        "model": model,
        "selected_probability_threshold": float(selected_threshold),
        "threshold_tuning": threshold_tuning_summary,
    }
    return result


def save_fall_meta_model_artifacts(
    result: dict[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    """
    Persist a trained fall meta-model and its artifacts to disk.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.json"
    predictions_path = out_dir / "test_predictions.csv"
    coefficients_path = out_dir / "coefficients.csv"
    model_path = out_dir / "model.joblib"
    run_summary_path = out_dir / "run_summary.json"
    threshold_tuning_path = out_dir / "threshold_tuning.csv"

    metrics_payload = {
        "config": result["config"],
        "used_features": result["used_features"],
        "split": result["split"],
        "metrics": result["metrics"],
        "selected_probability_threshold": result.get("selected_probability_threshold"),
    }

    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    result["test_predictions"].to_csv(predictions_path, index=False)
    result["coefficient_table"].to_csv(coefficients_path, index=False)

    model_artifact = {
        "model_type": "logistic_regression",
        "task_type": "fall",
        "model": result["model"],
        "used_features": result["used_features"],
        "positive_label": result["config"]["positive_label"],
        "negative_label": result["config"]["negative_label"],
        "probability_threshold": result.get("selected_probability_threshold", result["config"]["probability_threshold"]),
        "config": result["config"],
    }
    joblib.dump(model_artifact, model_path)

    threshold_tuning = result.get("threshold_tuning")
    if threshold_tuning and isinstance(threshold_tuning.get("threshold_table"), pd.DataFrame):
        threshold_tuning["threshold_table"].to_csv(threshold_tuning_path, index=False)

    run_summary = {
        "artifacts": {
            "metrics_json": str(metrics_path),
            "test_predictions_csv": str(predictions_path),
            "coefficients_csv": str(coefficients_path),
            "model_joblib": str(model_path),
            "threshold_tuning_csv": str(threshold_tuning_path) if threshold_tuning else None,
        },
        "used_features": result["used_features"],
        "split": result["split"],
        "metrics": result["metrics"],
        "selected_probability_threshold": result.get("selected_probability_threshold"),
    }
    run_summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    return {
        "metrics_json": str(metrics_path),
        "test_predictions_csv": str(predictions_path),
        "coefficients_csv": str(coefficients_path),
        "model_joblib": str(model_path),
        "run_summary_json": str(run_summary_path),
        "threshold_tuning_csv": str(threshold_tuning_path) if threshold_tuning else "",
    }


def train_fall_meta_model_from_csv(
    predictions_csv: str | Path,
    *,
    output_dir: str | Path | None = None,
    config: FallMetaModelConfig | None = None,
) -> dict[str, Any]:
    """
    Convenience wrapper for training from a saved threshold prediction table CSV.
    """
    pred_path = Path(predictions_csv)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {pred_path}")

    pred_df = pd.read_csv(pred_path)
    result = train_fall_meta_model(pred_df, config=config)

    if output_dir is not None:
        artifacts = save_fall_meta_model_artifacts(result, output_dir=output_dir)
        result["artifacts"] = artifacts

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a dissertation-grade logistic-regression fall meta-model from threshold prediction features."
    )
    parser.add_argument(
        "--predictions-csv",
        required=True,
        help="Path to a prediction table CSV, e.g. test_predictions_windows.csv",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where metrics, predictions, coefficients, and model artifacts will be saved",
    )
    parser.add_argument(
        "--probability-threshold",
        type=float,
        default=0.50,
        help="Fallback decision threshold if threshold tuning is disabled",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.30,
        help="Held-out subject-group fraction for evaluation",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for subject-aware split",
    )
    parser.add_argument(
        "--disable-threshold-tuning",
        action="store_true",
        help="Disable validation-based threshold tuning and use the fallback threshold directly",
    )
    parser.add_argument(
        "--validation-size-within-train",
        type=float,
        default=0.25,
        help="Inner validation split size for threshold tuning",
    )
    args = parser.parse_args()

    cfg = FallMetaModelConfig(
        probability_threshold=float(args.probability_threshold),
        test_size=float(args.test_size),
        random_state=int(args.random_state),
        tune_probability_threshold=not bool(args.disable_threshold_tuning),
        validation_size_within_train=float(args.validation_size_within_train),
    )

    result = train_fall_meta_model_from_csv(
        predictions_csv=args.predictions_csv,
        output_dir=args.output_dir,
        config=cfg,
    )

    metrics = result["metrics"]
    print("Fall meta-model summary")
    print(
        "accuracy={:.4f} sensitivity={:.4f} specificity={:.4f} precision={:.4f} f1={:.4f} roc_auc={:.4f} average_precision={:.4f} brier={:.4f} threshold={:.2f}".format(
            metrics["accuracy"],
            metrics["sensitivity"],
            metrics["specificity"],
            metrics["precision"],
            metrics["f1"],
            metrics["roc_auc"],
            metrics["average_precision"],
            metrics["brier_score"],
            metrics["probability_threshold_used"],
        )
    )
    if "artifacts" in result:
        print(f"Saved artifacts to: {result['artifacts']['run_summary_json']}")