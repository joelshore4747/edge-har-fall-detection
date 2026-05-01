from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.fall.train_fall_meta_model import DEFAULT_META_FEATURE_COLUMNS


@dataclass(slots=True)
class TuneConfig:
    positive_label: str = "fall"
    negative_label: str = "non_fall"
    feature_columns: list[str] = None  # type: ignore[assignment]

    outer_test_size: float = 0.30
    inner_val_size: float = 0.25
    random_state: int = 42

    c_grid: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0)
    threshold_grid: tuple[float, ...] = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
    class_weight_grid: tuple[str | None, ...] = ("balanced", None)

    selection_metric: str = "f1"

    def __post_init__(self) -> None:
        if self.feature_columns is None:
            self.feature_columns = list(DEFAULT_META_FEATURE_COLUMNS)


def _safe_float(value: Any) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    if not np.isfinite(x):
        return None
    return x


def _coerce_boolean_like_to_float(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(float)

    lowered = series.astype(str).str.lower()
    if lowered.isin({"true", "false"}).all():
        return lowered.map({"true": 1.0, "false": 0.0}).astype(float)

    return pd.to_numeric(series, errors="coerce")


def _prepare_feature_frame(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    available = [c for c in feature_columns if c in df.columns]
    if not available:
        raise ValueError("None of the requested feature columns are present")

    X = pd.DataFrame(index=df.index)
    for col in available:
        X[col] = _coerce_boolean_like_to_float(df[col])

    return X, available


def _get_subject_groups(df: pd.DataFrame) -> pd.Series:
    if "subject_id" not in df.columns:
        raise ValueError("Prediction dataframe must include subject_id")

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
        raise ValueError("Need at least 2 subject groups for subject-aware split")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=groups))

    train_df = df.iloc[sorted(train_idx)].reset_index(drop=True)
    test_df = df.iloc[sorted(test_idx)].reset_index(drop=True)

    summary = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_subjects_count": int(train_df["subject_id"].nunique(dropna=True)),
        "test_subjects_count": int(test_df["subject_id"].nunique(dropna=True)),
        "train_subject_groups": sorted(_get_subject_groups(train_df).astype(str).unique().tolist()),
        "test_subject_groups": sorted(_get_subject_groups(test_df).astype(str).unique().tolist()),
        "test_size": float(test_size),
        "random_state": int(random_state),
    }
    return train_df, test_df, summary


def _binarise_labels(labels: pd.Series, *, positive_label: str, negative_label: str) -> np.ndarray:
    mapped = labels.astype(str).map({positive_label: 1, negative_label: 0})
    if mapped.isna().any():
        bad = sorted(labels[mapped.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unexpected labels: {bad}")
    return mapped.astype(int).to_numpy()


def _compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    accuracy = float((tp + tn) / len(y_true)) if len(y_true) else 0.0
    f1 = float((2 * precision * sensitivity) / (precision + sensitivity)) if (precision + sensitivity) else 0.0

    metrics = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "support_total": int(len(y_true)),
        "support_positive": int(np.sum(y_true == 1)),
        "support_negative": int(np.sum(y_true == 0)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None

    return metrics


def _build_pipeline(c_value: float, class_weight: str | None, random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    C=float(c_value),
                    max_iter=2000,
                    class_weight=class_weight,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _select_metric(metrics: dict[str, Any], selection_metric: str) -> float:
    value = metrics.get(selection_metric)
    if value is None:
        raise ValueError(f"Selection metric not found: {selection_metric}")
    return float(value)


def tune_fall_meta_model(
    prediction_df: pd.DataFrame,
    *,
    config: TuneConfig | None = None,
) -> dict[str, Any]:
    config = config or TuneConfig()

    if prediction_df.empty:
        raise ValueError("Prediction dataframe is empty")
    if "true_label" not in prediction_df.columns:
        raise ValueError("Prediction dataframe must contain true_label")

    outer_train_df, outer_test_df, outer_split = _split_subject_aware(
        prediction_df,
        test_size=config.outer_test_size,
        random_state=config.random_state,
    )

    inner_train_df, inner_val_df, inner_split = _split_subject_aware(
        outer_train_df,
        test_size=config.inner_val_size,
        random_state=config.random_state,
    )

    X_inner_train, used_features = _prepare_feature_frame(inner_train_df, config.feature_columns)
    X_inner_val, _ = _prepare_feature_frame(inner_val_df, used_features)

    y_inner_train = _binarise_labels(
        inner_train_df["true_label"],
        positive_label=config.positive_label,
        negative_label=config.negative_label,
    )
    y_inner_val = _binarise_labels(
        inner_val_df["true_label"],
        positive_label=config.positive_label,
        negative_label=config.negative_label,
    )

    sweep_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for c_value in config.c_grid:
        for class_weight in config.class_weight_grid:
            model = _build_pipeline(c_value, class_weight, config.random_state)
            model.fit(X_inner_train, y_inner_train)

            val_prob = model.predict_proba(X_inner_val)[:, 1]

            for threshold in config.threshold_grid:
                val_pred = (val_prob >= float(threshold)).astype(int)
                metrics = _compute_binary_metrics(y_inner_val, val_pred, val_prob)

                row = {
                    "C": float(c_value),
                    "class_weight": class_weight,
                    "threshold": float(threshold),
                    **metrics,
                }
                sweep_rows.append(row)

                score = _select_metric(metrics, config.selection_metric)
                if best is None or score > best["selection_score"]:
                    best = {
                        "C": float(c_value),
                        "class_weight": class_weight,
                        "threshold": float(threshold),
                        "selection_score": score,
                        "validation_metrics": metrics,
                    }

    if best is None:
        raise RuntimeError("No tuning candidates were evaluated")

    # Refit on the full outer-train split using the selected config.
    X_outer_train, _ = _prepare_feature_frame(outer_train_df, used_features)
    X_outer_test, _ = _prepare_feature_frame(outer_test_df, used_features)

    y_outer_train = _binarise_labels(
        outer_train_df["true_label"],
        positive_label=config.positive_label,
        negative_label=config.negative_label,
    )
    y_outer_test = _binarise_labels(
        outer_test_df["true_label"],
        positive_label=config.positive_label,
        negative_label=config.negative_label,
    )

    final_model = _build_pipeline(best["C"], best["class_weight"], config.random_state)
    final_model.fit(X_outer_train, y_outer_train)

    test_prob = final_model.predict_proba(X_outer_test)[:, 1]
    test_pred = (test_prob >= float(best["threshold"])).astype(int)
    test_metrics = _compute_binary_metrics(y_outer_test, test_pred, test_prob)

    output_predictions = outer_test_df.copy()
    output_predictions["meta_probability_tuned"] = test_prob
    output_predictions["meta_predicted_is_fall_tuned"] = test_pred.astype(bool)
    output_predictions["meta_predicted_label_tuned"] = np.where(
        test_pred == 1,
        config.positive_label,
        config.negative_label,
    )

    sweep_df = pd.DataFrame(sweep_rows).sort_values(
        [config.selection_metric, "average_precision", "roc_auc"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    result = {
        "config": {
            "positive_label": config.positive_label,
            "negative_label": config.negative_label,
            "feature_columns": used_features,
            "outer_test_size": config.outer_test_size,
            "inner_val_size": config.inner_val_size,
            "random_state": config.random_state,
            "selection_metric": config.selection_metric,
            "c_grid": list(config.c_grid),
            "threshold_grid": list(config.threshold_grid),
            "class_weight_grid": list(config.class_weight_grid),
        },
        "outer_split": outer_split,
        "inner_split": inner_split,
        "best_config": {
            "C": best["C"],
            "class_weight": best["class_weight"],
            "threshold": best["threshold"],
        },
        "validation_metrics": best["validation_metrics"],
        "test_metrics": test_metrics,
        "sweep_results": sweep_df,
        "test_predictions": output_predictions,
        "model": final_model,
    }
    return result


def save_tuning_artifacts(result: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "run_summary.json"
    sweep_path = out_dir / "sweep_results.csv"
    preds_path = out_dir / "test_predictions_tuned.csv"
    model_path = out_dir / "model.joblib"

    result["sweep_results"].to_csv(sweep_path, index=False)
    result["test_predictions"].to_csv(preds_path, index=False)
    joblib.dump(result["model"], model_path)

    summary = {
        "config": result["config"],
        "outer_split": result["outer_split"],
        "inner_split": result["inner_split"],
        "best_config": result["best_config"],
        "validation_metrics": result["validation_metrics"],
        "test_metrics": result["test_metrics"],
        "artifacts": {
            "sweep_results_csv": str(sweep_path),
            "test_predictions_csv": str(preds_path),
            "model_joblib": str(model_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "run_summary_json": str(summary_path),
        "sweep_results_csv": str(sweep_path),
        "test_predictions_csv": str(preds_path),
        "model_joblib": str(model_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tune the fall meta-model with nested subject-aware validation."
    )
    parser.add_argument("--predictions-csv", required=True, help="Path to predictions_windows.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for tuning artifacts")
    parser.add_argument("--selection-metric", default="f1", choices=["f1", "precision", "sensitivity", "specificity", "accuracy"])
    parser.add_argument("--outer-test-size", type=float, default=0.30)
    parser.add_argument("--inner-val-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.predictions_csv)

    cfg = TuneConfig(
        outer_test_size=float(args.outer_test_size),
        inner_val_size=float(args.inner_val_size),
        random_state=int(args.random_state),
        selection_metric=str(args.selection_metric),
    )

    result = tune_fall_meta_model(df, config=cfg)
    artifacts = save_tuning_artifacts(result, args.output_dir)

    m = result["test_metrics"]
    print("Tuned fall meta-model summary")
    print(
        "accuracy={:.4f} sensitivity={:.4f} specificity={:.4f} precision={:.4f} f1={:.4f} roc_auc={} average_precision={} brier={:.4f}".format(
            m["accuracy"],
            m["sensitivity"],
            m["specificity"],
            m["precision"],
            m["f1"],
            f"{m['roc_auc']:.4f}" if m.get("roc_auc") is not None else "nan",
            f"{m['average_precision']:.4f}" if m.get("average_precision") is not None else "nan",
            m["brier_score"],
        )
    )
    print(f"Best config: {result['best_config']}")
    print(f"Saved summary to: {artifacts['run_summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())