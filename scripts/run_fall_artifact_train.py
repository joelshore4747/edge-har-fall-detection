#!/usr/bin/env python3
"""Train fall detector artifacts across multiple tree-ensemble build methods.

Replaces the legacy two-stage threshold + logistic-regression pipeline with a
set of single-stage classifiers trained directly on the 21 engineered fall
window features (same feature set as models/fall/train_fall_meta_model.py).

Models trained (one artifact + report per model):
- HistGradientBoostingClassifier (sklearn)
- XGBoost (if installed)
- RandomForestClassifier (sklearn)

Outputs per model <kind>:
- artifacts/fall_detector_<kind>.joblib          runtime-compatible artifact
- results/validation/fall_artifact_eval_<kind>.json    evaluation report
- results/validation/fall_artifact_eval_<kind>_predictions.csv

Also produced:
- artifacts/fall_detector.joblib       copy of best-F1 model (runtime default)
- results/validation/fall_artifact_eval_comparison.json

Evaluation (identical splits / seed across all models):
- outer subject-aware group split (dataset::subject)
- inner subject-aware split for threshold tuning
- per-model threshold tuning on inner val
- within-dataset + combined held-out metrics
- leave-one-subject-out cross-validation (per-model)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.fall_metrics import compute_fall_detection_metrics
from models.fall.evaluate_threshold_fall import build_threshold_prediction_table
from models.fall.train_fall_meta_model import DEFAULT_META_FEATURE_COLUMNS
from pipeline.fall.threshold_detector import default_fall_threshold_config
from pipeline.ingest import load_mobifall, load_sisfall
from pipeline.preprocess import (
    PreprocessConfig,
    append_derived_channels,
    resample_dataframe,
    window_dataframe,
)


FEATURE_COLUMNS: list[str] = list(DEFAULT_META_FEATURE_COLUMNS)
POSITIVE_LABEL = "fall"
NEGATIVE_LABEL = "non_fall"

DEFAULT_TARGET_RATE_HZ = 100.0
DEFAULT_WINDOW_SIZE = 128
DEFAULT_STEP_SIZE = 64
DEFAULT_RANDOM_STATE = 42
DEFAULT_OUTER_TEST_SIZE = 0.25
DEFAULT_INNER_VAL_SIZE = 0.20
DEFAULT_THRESHOLD_GRID = [round(x, 2) for x in np.arange(0.10, 0.91, 0.05)]

DEFAULT_ARTIFACT_DIR = "artifacts"
DEFAULT_REPORT_DIR = "results/validation"
DEFAULT_RUNTIME_ARTIFACT = "artifacts/fall_detector.joblib"
DEFAULT_N_JOBS = 2

MODEL_KINDS_DEFAULT = ("hgb", "xgb", "rf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mobifall-path", default="data/raw/MOBIACT_Dataset/MobiFall_Dataset_v2.0")
    parser.add_argument("--sisfall-path", default="data/raw/SISFALL_Dataset/SisFall_dataset")

    parser.add_argument("--target-rate", type=float, default=DEFAULT_TARGET_RATE_HZ)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--step-size", type=int, default=DEFAULT_STEP_SIZE)

    parser.add_argument("--outer-test-size", type=float, default=DEFAULT_OUTER_TEST_SIZE)
    parser.add_argument("--inner-val-size", type=float, default=DEFAULT_INNER_VAL_SIZE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)

    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_KINDS_DEFAULT),
        default=list(MODEL_KINDS_DEFAULT),
        help="Which model kinds to train (default: all three).",
    )
    parser.add_argument(
        "--skip-loso",
        action="store_true",
        help="Skip the leave-one-subject-out diagnostic (faster)",
    )
    parser.add_argument(
        "--skip-feature-importance",
        action="store_true",
        help="Skip permutation feature importance so artifact saving is not blocked by diagnostics.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help="Parallel workers for XGB/RF/permutation importance. Use a small value to avoid freezing the machine.",
    )
    parser.add_argument(
        "--feature-table",
        default=None,
        help="Optional cached combined fall feature table (.parquet or .csv). Skips raw dataset loading.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["validation_f1", "validation_f2", "heldout_f1"],
        default="validation_f2",
        help="Metric used to select the runtime artifact. Avoid heldout_f1 for final dissertation reporting.",
    )
    parser.add_argument(
        "--min-validation-specificity",
        type=float,
        default=0.0,
        help="Optional guardrail when selecting by validation metrics.",
    )
    parser.add_argument(
        "--output-run-id",
        default=None,
        help="Optional identifier written into artifact/report metadata.",
    )

    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--runtime-artifact-out",
        default=DEFAULT_RUNTIME_ARTIFACT,
        help="Where to copy the best-F1 model for runtime use.",
    )

    return parser.parse_args()


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return _json_safe(value.to_dict())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        f = float(value)
        return f if np.isfinite(f) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _coerce_boolean_like_to_float(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(float)
    lowered = series.astype(str).str.lower()
    if lowered.isin({"true", "false"}).all():
        return lowered.map({"true": 1.0, "false": 0.0}).astype(float)
    return pd.to_numeric(series, errors="coerce")


def _prepare_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for col in feature_cols:
        if col in df.columns:
            X[col] = _coerce_boolean_like_to_float(df[col])
        else:
            X[col] = np.nan
    return X


def _subject_groups(df: pd.DataFrame) -> pd.Series:
    dataset = df["dataset_name"].astype(str) if "dataset_name" in df.columns else pd.Series(["UNKNOWN"] * len(df), index=df.index)
    subject = df["subject_id"].astype(str) if "subject_id" in df.columns else pd.Series(["UNKNOWN_SUBJECT"] * len(df), index=df.index)
    return (dataset + "::" + subject).astype("string")


def _binarise(labels: pd.Series) -> np.ndarray:
    mapped = labels.astype(str).map({POSITIVE_LABEL: 1, NEGATIVE_LABEL: 0})
    if mapped.isna().any():
        bad = sorted(labels[mapped.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unexpected labels present: {bad}")
    return mapped.astype(int).to_numpy()


def _split_subject_aware(
    df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    groups = _subject_groups(df)
    if groups.nunique(dropna=True) < 2:
        raise ValueError("Need at least 2 subject groups for a subject-aware split")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    train_df = df.iloc[sorted(train_idx)].reset_index(drop=True)
    test_df = df.iloc[sorted(test_idx)].reset_index(drop=True)

    summary = {
        "strategy": "group_shuffle_split_by_subject",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_subject_groups": sorted(_subject_groups(train_df).astype(str).unique().tolist()),
        "test_subject_groups": sorted(_subject_groups(test_df).astype(str).unique().tolist()),
        "train_subjects_count": int(train_df["subject_id"].nunique(dropna=True)),
        "test_subjects_count": int(test_df["subject_id"].nunique(dropna=True)),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "train_label_counts": train_df["true_label"].astype(str).value_counts(dropna=False).to_dict(),
        "test_label_counts": test_df["true_label"].astype(str).value_counts(dropna=False).to_dict(),
    }
    return train_df, test_df, summary


def _load_and_feature_extract(
    dataset_key: str,
    path: Path,
    *,
    target_rate: float,
    window_size: int,
    step_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if dataset_key == "mobifall":
        raw = load_mobifall(path)
    elif dataset_key == "sisfall":
        raw = load_sisfall(path)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_key}")

    dataset_name = (
        str(raw["dataset_name"].dropna().astype(str).iloc[0])
        if "dataset_name" in raw.columns and not raw.empty
        else dataset_key.upper()
    )

    resampled = resample_dataframe(raw, target_rate_hz=target_rate)
    resampled = append_derived_channels(resampled)

    window_seconds = window_size / float(target_rate)
    preprocess_cfg = PreprocessConfig(
        target_sampling_rate_hz=target_rate,
        window_size_seconds=window_seconds,
        overlap_ratio=max(0.0, 1.0 - (step_size / float(window_size))),
    )
    windows = window_dataframe(
        resampled,
        window_size=window_size,
        step_size=step_size,
        config=preprocess_cfg,
    )

    detector_cfg = default_fall_threshold_config(None)
    feature_df = build_threshold_prediction_table(
        windows,
        detector_config=detector_cfg,
        filter_unacceptable=True,
        default_sampling_rate_hz=target_rate,
    )
    feature_df = feature_df[feature_df["true_label"].astype(str).isin({POSITIVE_LABEL, NEGATIVE_LABEL})].reset_index(drop=True)
    if feature_df.empty:
        raise ValueError(f"Feature table is empty for {dataset_name}")

    summary = {
        "dataset_key": dataset_key,
        "dataset_name": dataset_name,
        "rows_loaded": int(len(raw)),
        "rows_after_resample": int(len(resampled)),
        "windows_total": int(len(feature_df)),
        "label_counts": feature_df["true_label"].astype(str).value_counts(dropna=False).to_dict(),
        "subject_count": int(feature_df["subject_id"].nunique(dropna=True)),
    }
    return feature_df, summary


def _read_feature_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported feature table format: {path}. Use .parquet or .csv")


def _dataset_summary_from_features(df: pd.DataFrame, dataset_name: str) -> dict[str, Any]:
    sub = df[df["dataset_name"].astype(str).str.upper() == dataset_name.upper()].copy()
    return {
        "dataset_key": dataset_name.lower(),
        "dataset_name": dataset_name.upper(),
        "mode": "cached_feature_table",
        "rows_loaded": None,
        "rows_after_resample": None,
        "windows_total": int(len(sub)),
        "label_counts": sub["true_label"].astype(str).value_counts(dropna=False).to_dict(),
        "subject_count": int(sub["subject_id"].nunique(dropna=True)) if "subject_id" in sub.columns else None,
    }


def _build_hgb(random_state: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=600,
        max_leaf_nodes=63,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        class_weight="balanced",
        random_state=random_state,
    )


def _build_xgb(y_train: np.ndarray, random_state: int, *, n_jobs: int):
    if not XGBOOST_AVAILABLE:
        return None
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
    return XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="logloss",
        scale_pos_weight=float(scale_pos_weight),
        random_state=random_state,
        n_jobs=int(n_jobs),
    )


def _build_rf(random_state: int, *, n_jobs: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_leaf=2,
                    min_samples_split=4,
                    max_features="sqrt",
                    class_weight="balanced",
                    n_jobs=int(n_jobs),
                    random_state=random_state,
                ),
            ),
        ]
    )


def _build_model(kind: str, *, y_train: np.ndarray, random_state: int, n_jobs: int):
    if kind == "hgb":
        return _build_hgb(random_state)
    if kind == "xgb":
        model = _build_xgb(y_train, random_state, n_jobs=n_jobs)
        if model is None:
            raise RuntimeError("XGBoost requested but not installed. `pip install xgboost`.")
        return model
    if kind == "rf":
        return _build_rf(random_state, n_jobs=n_jobs)
    raise ValueError(f"Unknown model kind: {kind}")


def _tune_threshold(y_true: np.ndarray, y_prob: np.ndarray, grid: list[float]) -> tuple[float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    best_threshold = 0.5
    best_score = -1.0
    for threshold in grid:
        y_pred = (y_prob >= threshold).astype(int)
        score = float(f1_score(y_true, y_pred, zero_division=0))
        rows.append({"threshold": float(threshold), "f1": float(score)})
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, rows


def _binary_metrics_with_probs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    y_true_labels = np.where(y_true == 1, POSITIVE_LABEL, NEGATIVE_LABEL)
    y_pred_labels = np.where(y_pred == 1, POSITIVE_LABEL, NEGATIVE_LABEL)
    metrics = compute_fall_detection_metrics(
        y_true_labels.tolist(),
        y_pred_labels.tolist(),
        positive_label=POSITIVE_LABEL,
        negative_label=NEGATIVE_LABEL,
    )
    metrics["f2"] = float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")
    metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
    metrics["probability_threshold"] = float(threshold)
    return metrics


def _positive_proba(model, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = [int(c) for c in getattr(model, "classes_", [0, 1])]
    if 1 not in classes:
        raise ValueError("Model classes do not contain positive class '1'")
    return proba[:, classes.index(1)]


def _fit_and_tune_candidate(
    kind: str,
    *,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    random_state: int,
    threshold_grid: list[float],
    n_jobs: int,
) -> dict[str, Any]:
    model = _build_model(kind, y_train=y_train, random_state=random_state, n_jobs=n_jobs)
    model.fit(X_train, y_train)
    y_val_prob = _positive_proba(model, X_val)
    best_threshold, tuning_rows = _tune_threshold(y_val, y_val_prob, threshold_grid)
    val_metrics = _binary_metrics_with_probs(y_val, y_val_prob, threshold=best_threshold)

    return {
        "kind": kind,
        "model": model,
        "selected_threshold": float(best_threshold),
        "val_metrics": val_metrics,
        "threshold_tuning_table": tuning_rows,
    }


def _evaluate_holdout(
    model,
    *,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    X_test = _prepare_feature_matrix(test_df, feature_cols)
    y_test = _binarise(test_df["true_label"])
    y_prob = _positive_proba(model, X_test)
    metrics = _binary_metrics_with_probs(y_test, y_prob, threshold=threshold)

    per_dataset: dict[str, dict[str, Any]] = {}
    for dataset_name, sub_df in test_df.groupby("dataset_name", dropna=False):
        sub_y = _binarise(sub_df["true_label"])
        sub_prob = y_prob[sub_df.index.to_numpy()]
        per_dataset[str(dataset_name)] = _binary_metrics_with_probs(sub_y, sub_prob, threshold=threshold)

    metrics_block = {
        "combined": metrics,
        "per_dataset": per_dataset,
    }

    pred_df = test_df[[c for c in ["dataset_name", "subject_id", "session_id", "window_id", "start_ts", "end_ts", "true_label"] if c in test_df.columns]].copy()
    pred_df["predicted_probability"] = y_prob
    pred_df["predicted_is_fall"] = (y_prob >= threshold)
    pred_df["predicted_label"] = np.where(pred_df["predicted_is_fall"], POSITIVE_LABEL, NEGATIVE_LABEL)
    pred_df["probability_threshold_used"] = float(threshold)

    return metrics_block, pred_df


def _run_loso(
    full_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    kind: str,
    threshold: float,
    random_state: int,
    n_jobs: int,
) -> dict[str, Any]:
    groups = _subject_groups(full_df).astype(str).to_numpy()
    logo = LeaveOneGroupOut()
    y_all = _binarise(full_df["true_label"])
    X_all = _prepare_feature_matrix(full_df, feature_cols)

    per_subject_rows: list[dict[str, Any]] = []
    f1_values: list[float] = []
    auc_values: list[float] = []
    sens_values: list[float] = []
    spec_values: list[float] = []
    prec_values: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X_all, y_all, groups=groups)):
        test_group = groups[test_idx[0]]
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            # Can happen when a subject has only falls or only non-falls; skip from aggregate but record.
            per_subject_rows.append({
                "subject_group": str(test_group),
                "n_test": int(len(y_test)),
                "n_test_positive": int(np.sum(y_test == 1)),
                "n_test_negative": int(np.sum(y_test == 0)),
                "skipped": True,
                "reason": "degenerate_class_distribution",
            })
            continue

        X_train = X_all.iloc[train_idx]
        X_test = X_all.iloc[test_idx]

        model = _build_model(kind, y_train=y_train, random_state=random_state, n_jobs=n_jobs)
        model.fit(X_train, y_train)
        y_prob = _positive_proba(model, X_test)
        metrics = _binary_metrics_with_probs(y_prob=y_prob, y_true=y_test, threshold=threshold)

        row = {
            "subject_group": str(test_group),
            "dataset_name": str(test_group.split("::", 1)[0]) if "::" in test_group else str(test_group),
            "subject_id": str(test_group.split("::", 1)[1]) if "::" in test_group else "",
            "n_test": int(len(y_test)),
            "n_test_positive": int(np.sum(y_test == 1)),
            "n_test_negative": int(np.sum(y_test == 0)),
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "precision": metrics["precision"],
        }
        per_subject_rows.append(row)
        f1_values.append(metrics["f1"])
        if np.isfinite(metrics["roc_auc"]):
            auc_values.append(metrics["roc_auc"])
        sens_values.append(metrics["sensitivity"])
        spec_values.append(metrics["specificity"])
        prec_values.append(metrics["precision"])

        if fold_idx % 10 == 0 or fold_idx == logo.get_n_splits(groups=groups) - 1:
            print(f"  LOSO fold {fold_idx+1}/{logo.get_n_splits(groups=groups)} [{test_group}] f1={metrics['f1']:.3f}")

    def _agg(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
        arr = np.asarray(values, dtype=float)
        return {
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "n": int(len(arr)),
        }

    # Per-dataset aggregation
    per_dataset: dict[str, dict[str, Any]] = {}
    df_rows = pd.DataFrame([r for r in per_subject_rows if not r.get("skipped")])
    if not df_rows.empty:
        for dataset_name, sub in df_rows.groupby("dataset_name"):
            per_dataset[str(dataset_name)] = {
                "f1": _agg(sub["f1"].tolist()),
                "roc_auc": _agg([v for v in sub["roc_auc"].tolist() if np.isfinite(v)]),
                "sensitivity": _agg(sub["sensitivity"].tolist()),
                "specificity": _agg(sub["specificity"].tolist()),
                "precision": _agg(sub["precision"].tolist()),
                "n_subjects": int(len(sub)),
            }

    return {
        "n_folds": int(len(per_subject_rows)),
        "n_folds_used": int(len(f1_values)),
        "threshold_used": float(threshold),
        "aggregate": {
            "f1": _agg(f1_values),
            "roc_auc": _agg(auc_values),
            "sensitivity": _agg(sens_values),
            "specificity": _agg(spec_values),
            "precision": _agg(prec_values),
        },
        "per_dataset": per_dataset,
        "per_subject": per_subject_rows,
    }


def _feature_importances(
    model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    *,
    random_state: int,
    n_repeats: int = 10,
    n_jobs: int = DEFAULT_N_JOBS,
) -> list[dict[str, float]]:
    try:
        result = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="roc_auc",
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=int(n_jobs),
        )
    except Exception as exc:
        warnings.warn(f"permutation_importance failed: {exc}")
        return []

    rows = [
        {
            "feature": str(col),
            "importance_mean": float(result.importances_mean[i]),
            "importance_std": float(result.importances_std[i]),
        }
        for i, col in enumerate(X_test.columns)
    ]
    rows.sort(key=lambda r: r["importance_mean"], reverse=True)
    return rows


def _save_artifact(
    *,
    model,
    used_features: list[str],
    probability_threshold: float,
    out_path: Path,
    metadata: dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_type": type(model).__name__,
        "task_type": "fall",
        "model": model,
        "used_features": list(used_features),
        "positive_label": POSITIVE_LABEL,
        "negative_label": NEGATIVE_LABEL,
        "probability_threshold": float(probability_threshold),
        "metadata": metadata,
    }
    joblib.dump(artifact, out_path)


def _artifact_path_for(kind: str, artifact_dir: Path) -> Path:
    return artifact_dir / f"fall_detector_{kind}.joblib"


def _report_paths_for(kind: str, report_dir: Path) -> tuple[Path, Path]:
    return (
        report_dir / f"fall_artifact_eval_{kind}.json",
        report_dir / f"fall_artifact_eval_{kind}_predictions.csv",
    )


def _train_one_model(
    kind: str,
    *,
    train_inner_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_outer_df: pd.DataFrame,
    test_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    feature_cols: list[str],
    threshold_grid: list[float],
    random_state: int,
    skip_loso: bool,
    skip_feature_importance: bool,
    n_jobs: int,
    artifact_dir: Path,
    report_dir: Path,
    mobi_summary: dict[str, Any],
    sis_summary: dict[str, Any],
    outer_split: dict[str, Any],
    inner_split: dict[str, Any],
    common_config: dict[str, Any],
    created_utc: str,
) -> dict[str, Any]:
    print(f"\n=== Training [{kind.upper()}] ===")

    X_inner = _prepare_feature_matrix(train_inner_df, feature_cols)
    y_inner = _binarise(train_inner_df["true_label"])
    X_val = _prepare_feature_matrix(val_df, feature_cols)
    y_val = _binarise(val_df["true_label"])

    candidate = _fit_and_tune_candidate(
        kind,
        X_train=X_inner,
        y_train=y_inner,
        X_val=X_val,
        y_val=y_val,
        random_state=random_state,
        threshold_grid=threshold_grid,
        n_jobs=n_jobs,
    )
    selected_threshold = float(candidate["selected_threshold"])
    print(
        f"  [{kind}] val_f1={candidate['val_metrics']['f1']:.4f} "
        f"threshold={selected_threshold:.2f} "
        f"roc_auc={candidate['val_metrics']['roc_auc']:.4f}"
    )

    print(f"  [{kind}] Refitting on full outer-train set...")
    X_outer_train = _prepare_feature_matrix(train_outer_df, feature_cols)
    y_outer_train = _binarise(train_outer_df["true_label"])
    final_model = _build_model(
        kind,
        y_train=y_outer_train,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    final_model.fit(X_outer_train, y_outer_train)

    print(f"  [{kind}] Held-out evaluation...")
    holdout_metrics, holdout_predictions = _evaluate_holdout(
        final_model,
        test_df=test_df,
        feature_cols=feature_cols,
        threshold=selected_threshold,
    )
    print(
        f"  [{kind}] held-out combined: f1={holdout_metrics['combined']['f1']:.4f} "
        f"roc_auc={holdout_metrics['combined']['roc_auc']:.4f} "
        f"sens={holdout_metrics['combined']['sensitivity']:.4f} "
        f"spec={holdout_metrics['combined']['specificity']:.4f}"
    )
    for ds, m in holdout_metrics["per_dataset"].items():
        print(f"  [{kind}] held-out {ds}: f1={m['f1']:.4f} roc_auc={m['roc_auc']:.4f}")

    artifact_path = _artifact_path_for(kind, artifact_dir)
    report_path, predictions_path = _report_paths_for(kind, report_dir)

    artifact_metadata = {
        "artifact_version": f"fall_candidate_{kind}",
        "created_utc": created_utc,
        "model_type": type(final_model).__name__,
        "model_kind": kind,
        "artifact_id": common_config.get("output_run_id") or f"fall_{kind}_{created_utc}",
        "status": "candidate",
        "train_source_composition": {
            "mobifall": mobi_summary,
            "sisfall": sis_summary,
            "combined_rows": int(len(combined_df)),
        },
        "training_config": {
            **common_config,
            "model_params": final_model.get_params(),
        },
        "feature_columns": list(feature_cols),
        "selection_metric": common_config.get("selection_metric", "validation_f2"),
        "library_versions": _library_versions(),
    }

    print(f"  [{kind}] Saving artifact -> {artifact_path}")
    _save_artifact(
        model=final_model,
        used_features=feature_cols,
        probability_threshold=selected_threshold,
        out_path=artifact_path,
        metadata=artifact_metadata,
    )

    feature_importances: list[dict[str, float]] = []
    loso_block: dict[str, Any] | None = None
    report = {
        "evaluation_name": f"fall_artifact_train_{kind}",
        "created_utc": created_utc,
        "model_kind": kind,
        "config": {
            **common_config,
            "skip_loso": bool(skip_loso),
            "skip_feature_importance": bool(skip_feature_importance),
            "n_jobs": int(n_jobs),
        },
        "preprocessing": {
            "target_rate_hz": common_config["target_rate_hz"],
            "window_size": common_config["window_size"],
            "step_size": common_config["step_size"],
        },
        "data": {
            "datasets": {"MOBIFALL": mobi_summary, "SISFALL": sis_summary},
            "combined_rows": int(len(combined_df)),
            "feature_columns": list(feature_cols),
        },
        "outer_split": outer_split,
        "inner_split": inner_split,
        "threshold_tuning": {
            "selected_threshold": selected_threshold,
            "val_metrics": candidate["val_metrics"],
            "threshold_tuning_table_top": sorted(
                candidate["threshold_tuning_table"], key=lambda r: r["f1"], reverse=True
            )[:10],
        },
        "held_out_metrics": holdout_metrics,
        "loso": loso_block,
        "feature_importances": feature_importances,
        "artifact": {
            "path": str(artifact_path),
            "model_type": type(final_model).__name__,
            "model_kind": kind,
            "probability_threshold": selected_threshold,
            "used_features": list(feature_cols),
            "library_versions": artifact_metadata["library_versions"],
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    holdout_predictions.to_csv(predictions_path, index=False)
    print(f"  [{kind}] Saved report -> {report_path}")
    print(f"  [{kind}] Saved predictions -> {predictions_path}")

    diagnostics_updated = False
    if not skip_feature_importance:
        print(f"  [{kind}] Permutation feature importances...")
        X_test = _prepare_feature_matrix(test_df, feature_cols)
        y_test = _binarise(test_df["true_label"])
        feature_importances = _feature_importances(
            final_model,
            X_test,
            y_test,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        report["feature_importances"] = feature_importances
        diagnostics_updated = True

    if not skip_loso:
        print(f"  [{kind}] LOSO cross-validation (one fit per subject group)...")
        loso_block = _run_loso(
            combined_df,
            feature_cols=feature_cols,
            kind=kind,
            threshold=selected_threshold,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        print(
            f"  [{kind}] LOSO aggregate: "
            f"f1={loso_block['aggregate']['f1']['mean']:.4f}+-{loso_block['aggregate']['f1']['std']:.4f}  "
            f"roc_auc={loso_block['aggregate']['roc_auc']['mean']:.4f}+-{loso_block['aggregate']['roc_auc']['std']:.4f}"
        )
        report["loso"] = loso_block
        diagnostics_updated = True

    if diagnostics_updated:
        report_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
        print(f"  [{kind}] Updated report with optional diagnostics -> {report_path}")

    return {
        "kind": kind,
        "model": final_model,
        "selected_threshold": selected_threshold,
        "val_metrics": candidate["val_metrics"],
        "holdout_metrics": holdout_metrics,
        "loso": loso_block,
        "artifact_path": artifact_path,
        "report_path": report_path,
        "predictions_path": predictions_path,
    }


def _library_versions() -> dict[str, str]:
    versions = {
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    if XGBOOST_AVAILABLE:
        import xgboost

        versions["xgboost"] = xgboost.__version__
    return versions


def main() -> int:
    args = parse_args()

    requested_models = list(args.models)
    if "xgb" in requested_models and not XGBOOST_AVAILABLE:
        raise RuntimeError("xgboost requested but not installed. `pip install xgboost`.")

    mobifall_path = _resolve_path(args.mobifall_path)
    sisfall_path = _resolve_path(args.sisfall_path)
    feature_table_path = _resolve_path(args.feature_table) if args.feature_table else None
    artifact_dir = _resolve_path(args.artifact_dir)
    report_dir = _resolve_path(args.report_dir)
    runtime_artifact_out = _resolve_path(args.runtime_artifact_out)

    if feature_table_path is None and not mobifall_path.exists():
        raise FileNotFoundError(f"MobiFall path not found: {mobifall_path}")
    if feature_table_path is None and not sisfall_path.exists():
        raise FileNotFoundError(f"SisFall path not found: {sisfall_path}")
    if feature_table_path is not None and not feature_table_path.exists():
        raise FileNotFoundError(f"Feature table not found: {feature_table_path}")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Training models: {requested_models}  "
        f"(xgboost available={XGBOOST_AVAILABLE}, n_jobs={args.n_jobs})"
    )

    if feature_table_path is not None:
        print(f"Loading cached fall feature table -> {feature_table_path}")
        combined_df = _read_feature_table(feature_table_path)
        combined_df = combined_df[combined_df["true_label"].astype(str).isin({POSITIVE_LABEL, NEGATIVE_LABEL})].reset_index(drop=True)
        mobi_summary = _dataset_summary_from_features(combined_df, "MOBIFALL")
        sis_summary = _dataset_summary_from_features(combined_df, "SISFALL")
    else:
        print("Building MobiFall feature table...")
        mobi_df, mobi_summary = _load_and_feature_extract(
            "mobifall",
            mobifall_path,
            target_rate=args.target_rate,
            window_size=args.window_size,
            step_size=args.step_size,
        )
        print(f"  MobiFall windows={len(mobi_df)} subjects={mobi_summary['subject_count']}")

        print("Building SisFall feature table...")
        sis_df, sis_summary = _load_and_feature_extract(
            "sisfall",
            sisfall_path,
            target_rate=args.target_rate,
            window_size=args.window_size,
            step_size=args.step_size,
        )
        print(f"  SisFall windows={len(sis_df)} subjects={sis_summary['subject_count']}")

        combined_df = pd.concat([mobi_df, sis_df], ignore_index=True)
    print(f"Combined windows={len(combined_df)} label_counts={combined_df['true_label'].value_counts().to_dict()}")

    train_outer_df, test_df, outer_split = _split_subject_aware(
        combined_df,
        test_size=args.outer_test_size,
        random_state=args.random_state,
    )
    print(f"Outer split: train_rows={len(train_outer_df)} test_rows={len(test_df)}")

    train_inner_df, val_df, inner_split = _split_subject_aware(
        train_outer_df,
        test_size=args.inner_val_size,
        random_state=args.random_state,
    )
    print(f"Inner split: train_rows={len(train_inner_df)} val_rows={len(val_df)}")

    created_utc = datetime.now(timezone.utc).isoformat()
    common_config = {
        "target_rate_hz": float(args.target_rate),
        "window_size": int(args.window_size),
        "step_size": int(args.step_size),
        "outer_test_size": float(args.outer_test_size),
        "inner_val_size": float(args.inner_val_size),
        "random_state": int(args.random_state),
        "feature_table": str(feature_table_path) if feature_table_path is not None else None,
        "selection_metric": str(args.selection_metric),
        "min_validation_specificity": float(args.min_validation_specificity),
        "output_run_id": args.output_run_id,
    }

    results_by_kind: dict[str, dict[str, Any]] = {}
    for kind in requested_models:
        try:
            results_by_kind[kind] = _train_one_model(
                kind,
                train_inner_df=train_inner_df,
                val_df=val_df,
                train_outer_df=train_outer_df,
                test_df=test_df,
                combined_df=combined_df,
                feature_cols=FEATURE_COLUMNS,
                threshold_grid=DEFAULT_THRESHOLD_GRID,
                random_state=args.random_state,
                skip_loso=args.skip_loso,
                skip_feature_importance=args.skip_feature_importance,
                n_jobs=args.n_jobs,
                artifact_dir=artifact_dir,
                report_dir=report_dir,
                mobi_summary=mobi_summary,
                sis_summary=sis_summary,
                outer_split=outer_split,
                inner_split=inner_split,
                common_config=common_config,
                created_utc=created_utc,
            )
        except Exception as exc:
            print(f"!! Model [{kind}] failed: {exc}")
            raise

    # Pick the runtime winner without using outer-test metrics unless explicitly requested.
    eligible_kinds = [
        k
        for k, r in results_by_kind.items()
        if float(r["val_metrics"]["specificity"]) >= float(args.min_validation_specificity)
    ]
    if not eligible_kinds:
        warnings.warn(
            "No model met --min-validation-specificity; falling back to all trained models."
        )
        eligible_kinds = list(results_by_kind.keys())

    def _sort_key(k: str) -> tuple[float, float, float]:
        r = results_by_kind[k]
        if args.selection_metric == "heldout_f1":
            warnings.warn(
                "Selecting by heldout_f1 leaks test-set information; use only for diagnostics."
            )
            primary = float(r["holdout_metrics"]["combined"]["f1"])
        elif args.selection_metric == "validation_f1":
            primary = float(r["val_metrics"]["f1"])
        else:
            primary = float(r["val_metrics"].get("f2", r["val_metrics"]["f1"]))
        return primary, float(r["val_metrics"]["specificity"]), float(r["val_metrics"]["f1"])

    best_kind = max(eligible_kinds, key=_sort_key)
    best_artifact = results_by_kind[best_kind]["artifact_path"]
    runtime_artifact_out.parent.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(best_artifact, runtime_artifact_out)
    print(
        f"\nBest model by {args.selection_metric}: [{best_kind}] "
        f"-> copied to {runtime_artifact_out}"
    )

    comparison = {
        "evaluation_name": "fall_artifact_train_comparison",
        "created_utc": created_utc,
        "config": {
            **common_config,
            "skip_loso": bool(args.skip_loso),
            "models_trained": list(results_by_kind.keys()),
            "xgboost_available": bool(XGBOOST_AVAILABLE),
            "selection_metric": str(args.selection_metric),
            "min_validation_specificity": float(args.min_validation_specificity),
        },
        "data": {
            "datasets": {"MOBIFALL": mobi_summary, "SISFALL": sis_summary},
            "combined_rows": int(len(combined_df)),
        },
        "outer_split": outer_split,
        "inner_split": inner_split,
        "best_by_selection_metric": best_kind,
        "runtime_artifact": str(runtime_artifact_out),
        "per_model": {
            k: {
                "kind": k,
                "artifact_path": str(r["artifact_path"]),
                "report_path": str(r["report_path"]),
                "predictions_path": str(r["predictions_path"]),
                "selected_threshold": r["selected_threshold"],
                "val_metrics": {
                    "f1": r["val_metrics"]["f1"],
                    "f2": r["val_metrics"].get("f2"),
                    "roc_auc": r["val_metrics"]["roc_auc"],
                    "sensitivity": r["val_metrics"]["sensitivity"],
                    "specificity": r["val_metrics"]["specificity"],
                    "precision": r["val_metrics"]["precision"],
                },
                "held_out": {
                    "combined": {
                        "f1": r["holdout_metrics"]["combined"]["f1"],
                        "f2": r["holdout_metrics"]["combined"].get("f2"),
                        "roc_auc": r["holdout_metrics"]["combined"]["roc_auc"],
                        "sensitivity": r["holdout_metrics"]["combined"]["sensitivity"],
                        "specificity": r["holdout_metrics"]["combined"]["specificity"],
                        "precision": r["holdout_metrics"]["combined"]["precision"],
                        "average_precision": r["holdout_metrics"]["combined"]["average_precision"],
                        "brier_score": r["holdout_metrics"]["combined"]["brier_score"],
                    },
                    "per_dataset": {
                        ds: {
                            "f1": m["f1"],
                            "f2": m.get("f2"),
                            "roc_auc": m["roc_auc"],
                            "sensitivity": m["sensitivity"],
                            "specificity": m["specificity"],
                            "precision": m["precision"],
                        }
                        for ds, m in r["holdout_metrics"]["per_dataset"].items()
                    },
                },
                "loso_aggregate": (
                    r["loso"]["aggregate"] if r["loso"] is not None else None
                ),
            }
            for k, r in results_by_kind.items()
        },
    }

    comparison_path = report_dir / "fall_artifact_eval_comparison.json"
    comparison_path.write_text(json.dumps(_json_safe(comparison), indent=2), encoding="utf-8")
    print(f"Saved comparison -> {comparison_path}")

    print("\n=== Summary ===")
    print(f"{'kind':<6} {'val_f2':>8} {'val_f1':>8} {'ho_f1':>8} {'ho_auc':>8} {'ho_sens':>8} {'ho_spec':>8}  loso_f1_mean")
    for k, r in results_by_kind.items():
        loso_mean = r["loso"]["aggregate"]["f1"]["mean"] if r["loso"] is not None else float("nan")
        print(
            f"{k:<6} "
            f"{r['val_metrics'].get('f2', float('nan')):>8.4f} "
            f"{r['val_metrics']['f1']:>8.4f} "
            f"{r['holdout_metrics']['combined']['f1']:>8.4f} "
            f"{r['holdout_metrics']['combined']['roc_auc']:>8.4f} "
            f"{r['holdout_metrics']['combined']['sensitivity']:>8.4f} "
            f"{r['holdout_metrics']['combined']['specificity']:>8.4f}  "
            f"{loso_mean:.4f}"
        )
    print(f"best-by-{args.selection_metric}: {best_kind}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
