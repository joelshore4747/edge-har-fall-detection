#!/usr/bin/env python3
"""Train a phone-aware fall meta-model using phone hard negatives.

Goal:
- compare the current public-data-only fall model against an adapted model
  that also sees labeled phone hard negatives (stairs, walking, sit-down, etc.)
- measure whether phone false positives go down without destroying fall detection

Main comparison:
1) baseline_model = trained on public source only
2) adapted_model = trained on public source + selected phone rows

Evaluates on:
- optional public evaluation source (MobiFall or SisFall)
- phone labeled windows from one annotated runtime session
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.fall_metrics import compute_fall_detection_metrics
from models.fall.train_fall_meta_model import FallMetaModelConfig
from models.fall.evaluate_threshold_fall import build_threshold_prediction_table
from pipeline.fall.threshold_detector import default_fall_threshold_config
from pipeline.ingest import load_mobifall, load_sisfall
from pipeline.preprocess import (
    PreprocessConfig,
    append_derived_channels,
    resample_dataframe,
    window_dataframe,
)


DEFAULT_PHONE_HARD_NEGATIVE_TYPES = [
    "walking",
    "stairs",
    "sit_down_transition",
    "phone_handling",
    "sitting",
    "other",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fall model with phone hard negatives")

    parser.add_argument("--mobifall-path", default="data/raw/MOBIACT_Dataset/MobiFall_Dataset_v2.0")
    parser.add_argument("--sisfall-path", default="data/raw/SISFALL_Dataset/SisFall_dataset")
    parser.add_argument(
        "--phone-fall-labeled-csv",
        default="results/validation/phone1_fall_labeled_windows.csv",
        help="Phone-labeled fall window CSV from build_phone_runtime_training_set.py",
    )

    parser.add_argument(
        "--public-train-source",
        choices=["mobifall", "sisfall", "combined"],
        default="combined",
        help="Public source used for the baseline/adapted source model",
    )
    parser.add_argument(
        "--public-eval-source",
        choices=["mobifall", "sisfall", "none"],
        default="sisfall",
        help="Optional public evaluation source used for side-by-side comparison",
    )

    parser.add_argument(
        "--threshold-mode",
        choices=["shared", "dataset_presets"],
        default="shared",
        help="shared is recommended after unit harmonization",
    )
    parser.add_argument("--target-rate", type=float, default=100.0)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--step-size", type=int, default=64)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument(
        "--phone-negatives-only",
        action="store_true",
        help="Use only phone non_fall rows for adaptation (recommended first pass)",
    )
    parser.add_argument(
        "--include-phone-positives",
        action="store_true",
        help="Allow annotated phone fall rows to be added to the adaptation set",
    )
    parser.add_argument(
        "--phone-hard-negative-types",
        default=",".join(DEFAULT_PHONE_HARD_NEGATIVE_TYPES),
        help="Comma-separated phone hard-negative types to include",
    )

    parser.add_argument(
        "--phone-max-rows",
        type=int,
        default=0,
        help="Optional cap on number of selected phone rows (0 = no cap)",
    )

    parser.add_argument(
        "--out-json",
        default="results/validation/fall_phone_adaptation_comparison.json",
        help="Summary comparison JSON",
    )
    parser.add_argument(
        "--phone-predictions-out",
        default="results/validation/fall_phone_adaptation_phone_predictions.csv",
        help="Side-by-side phone prediction CSV",
    )
    parser.add_argument(
        "--public-predictions-out",
        default="results/validation/fall_phone_adaptation_public_predictions.csv",
        help="Side-by-side public evaluation prediction CSV",
    )
    parser.add_argument(
        "--save-adapted-model",
        default=None,
        help="If set, save the adapted model bundle (joblib) to this path in the prod format",
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
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
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


def _prepare_feature_frame(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    available = [c for c in feature_columns if c in df.columns]
    if not available:
        raise ValueError("None of the requested fall meta-model feature columns are present")
    X = pd.DataFrame(index=df.index)
    for col in available:
        X[col] = _coerce_boolean_like_to_float(df[col])
    return X, available


def _get_subject_groups(df: pd.DataFrame) -> pd.Series:
    dataset = (
        df["dataset_name"].astype(str)
        if "dataset_name" in df.columns
        else pd.Series(["UNKNOWN"] * len(df), index=df.index, dtype="string")
    )
    subject = (
        df["subject_id"].astype(str)
        if "subject_id" in df.columns
        else pd.Series(["UNKNOWN_SUBJECT"] * len(df), index=df.index, dtype="string")
    )
    return (dataset + "::" + subject).astype("string")


def _split_subject_aware(
    df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = _get_subject_groups(df)
    if groups.nunique(dropna=True) < 2:
        raise ValueError("Need at least 2 subject groups for subject-aware split")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(df, groups=groups))
    train_df = df.iloc[np.sort(train_idx)].reset_index(drop=True)
    val_df = df.iloc[np.sort(val_idx)].reset_index(drop=True)
    return train_df, val_df


def _binarise_labels(
    labels: pd.Series,
    *,
    positive_label: str,
    negative_label: str,
) -> np.ndarray:
    mapped = labels.astype(str).map({positive_label: 1, negative_label: 0})
    if mapped.isna().any():
        bad = sorted(labels[mapped.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unexpected labels present: {bad}")
    return mapped.astype(int).to_numpy()


def _compute_binary_metrics_with_probs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    probability_threshold: float,
    positive_label: str,
    negative_label: str,
) -> dict[str, Any]:
    y_pred_bin = (y_prob >= probability_threshold).astype(int)
    y_true_labels = np.where(y_true == 1, positive_label, negative_label)
    y_pred_labels = np.where(y_pred_bin == 1, positive_label, negative_label)

    metrics = compute_fall_detection_metrics(
        y_true_labels.tolist(),
        y_pred_labels.tolist(),
        positive_label=positive_label,
        negative_label=negative_label,
    )
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")
    metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
    metrics["probability_threshold"] = float(probability_threshold)
    return metrics


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

    from sklearn.metrics import f1_score, precision_score, recall_score

    if metric_name == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if metric_name == "precision":
        return float(precision_score(y_true, y_pred, zero_division=0))
    if metric_name == "recall":
        return float(recall_score(y_true, y_pred, zero_division=0))
    if metric_name == "balanced_score":
        p = float(precision_score(y_true, y_pred, zero_division=0))
        r = float(recall_score(y_true, y_pred, zero_division=0))
        return (p + r) / 2.0

    raise ValueError(f"Unsupported threshold_tuning_metric: {metric_name}")


def _select_probability_threshold(
    train_df: pd.DataFrame,
    *,
    config: FallMetaModelConfig,
) -> tuple[float, pd.DataFrame]:
    if _get_subject_groups(train_df).nunique(dropna=True) < 2:
        return float(config.probability_threshold), pd.DataFrame(
            [{"threshold": float(config.probability_threshold), "score": float("nan")}]
        )

    inner_train_df, val_df = _split_subject_aware(
        train_df,
        test_size=config.validation_size_within_train,
        random_state=config.random_state,
    )

    X_inner, used_features = _prepare_feature_frame(inner_train_df, config.feature_columns)
    X_val, _ = _prepare_feature_frame(val_df, used_features)

    y_inner = _binarise_labels(
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
    model.fit(X_inner, y_inner)
    y_val_prob = model.predict_proba(X_val)[:, 1]

    rows: list[dict[str, float]] = []
    best_threshold = float(config.probability_threshold)
    best_score = float("-inf")

    for threshold in config.threshold_grid:
        score = _score_threshold(
            y_val,
            y_val_prob,
            threshold=float(threshold),
            metric_name=config.threshold_tuning_metric,
        )
        rows.append({"threshold": float(threshold), "score": float(score)})
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)

    table = pd.DataFrame(rows).sort_values(["score", "threshold"], ascending=[False, True], kind="stable")
    return best_threshold, table.reset_index(drop=True)


def _fit_source_model(
    train_df: pd.DataFrame,
    *,
    config: FallMetaModelConfig,
) -> tuple[Pipeline, list[str], float, pd.DataFrame]:
    selected_threshold = float(config.probability_threshold)
    tuning_table = pd.DataFrame()

    if config.tune_probability_threshold:
        selected_threshold, tuning_table = _select_probability_threshold(train_df, config=config)

    X_train, used_features = _prepare_feature_frame(train_df, config.feature_columns)
    y_train = _binarise_labels(
        train_df["true_label"],
        positive_label=config.positive_label,
        negative_label=config.negative_label,
    )

    model = _build_model_pipeline(config)
    model.fit(X_train, y_train)
    return model, used_features, selected_threshold, tuning_table


def _evaluate_model_on_df(
    eval_df: pd.DataFrame,
    *,
    model: Pipeline,
    used_features: list[str],
    probability_threshold: float,
    config: FallMetaModelConfig,
    include_prediction_columns: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    X_eval, _ = _prepare_feature_frame(eval_df, used_features)
    y_eval = _binarise_labels(
        eval_df["true_label"],
        positive_label=config.positive_label,
        negative_label=config.negative_label,
    )
    y_prob = model.predict_proba(X_eval)[:, 1]
    metrics = _compute_binary_metrics_with_probs(
        y_eval,
        y_prob,
        probability_threshold=probability_threshold,
        positive_label=config.positive_label,
        negative_label=config.negative_label,
    )

    out = eval_df.copy()
    if include_prediction_columns:
        pred_labels = np.where(
            y_prob >= probability_threshold,
            config.positive_label,
            config.negative_label,
        )
        out["predicted_probability"] = y_prob
        out["predicted_label"] = pd.Series(pred_labels, index=out.index, dtype="string")
        out["predicted_is_fall"] = out["predicted_label"].astype(str).eq(config.positive_label)

    return metrics, out


def _false_positive_breakdown(eval_pred_df: pd.DataFrame) -> dict[str, int]:
    required = {"true_label", "predicted_label"}
    if not required <= set(eval_pred_df.columns):
        return {}

    fp_df = eval_pred_df[
        eval_pred_df["true_label"].astype(str).eq("non_fall")
        & eval_pred_df["predicted_label"].astype(str).eq("fall")
    ].copy()

    if fp_df.empty:
        return {}

    if "fall_hard_negative_type" in fp_df.columns:
        series = fp_df["fall_hard_negative_type"].dropna().astype(str)
        if not series.empty:
            return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}

    if "annotation_label" in fp_df.columns:
        series = fp_df["annotation_label"].dropna().astype(str)
        if not series.empty:
            return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}

    return {"all_false_positives": int(len(fp_df))}


def _load_fall_dataframe(dataset_key: str, path: Path) -> pd.DataFrame:
    if dataset_key == "mobifall":
        return load_mobifall(path)
    if dataset_key == "sisfall":
        return load_sisfall(path)
    raise ValueError(f"Unsupported dataset: {dataset_key}")


def _make_detector_config(dataset_name: str, threshold_mode: str):
    if threshold_mode == "dataset_presets":
        return default_fall_threshold_config(dataset_name)
    return default_fall_threshold_config(None)


def _prepare_public_prediction_table(
    *,
    dataset_key: str,
    path: Path,
    threshold_mode: str,
    target_rate: float,
    window_size: int,
    step_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = _load_fall_dataframe(dataset_key, path)
    dataset_name = (
        str(df["dataset_name"].dropna().astype(str).iloc[0])
        if "dataset_name" in df.columns and not df.empty
        else dataset_key.upper()
    )

    resampled = resample_dataframe(df, target_rate_hz=target_rate)
    resampled = append_derived_channels(resampled)

    preprocess_cfg = PreprocessConfig(target_sampling_rate_hz=target_rate)
    windows = window_dataframe(
        resampled,
        window_size=window_size,
        step_size=step_size,
        config=preprocess_cfg,
    )

    detector_config = _make_detector_config(dataset_name, threshold_mode)
    pred_df = build_threshold_prediction_table(
        windows,
        detector_config=detector_config,
        filter_unacceptable=True,
        default_sampling_rate_hz=target_rate,
    )
    if pred_df.empty:
        raise ValueError(f"Public threshold prediction table is empty for {dataset_name}")

    summary = {
        "dataset_name": dataset_name,
        "rows": int(len(pred_df)),
        "true_label_counts": pred_df["true_label"].astype(str).value_counts(dropna=False).to_dict(),
    }
    return pred_df, summary


def _select_public_train_table(
    source: str,
    *,
    mobi_df: pd.DataFrame,
    sis_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if source == "mobifall":
        return mobi_df.copy(), {"public_train_source": "mobifall"}
    if source == "sisfall":
        return sis_df.copy(), {"public_train_source": "sisfall"}
    if source == "combined":
        return pd.concat([mobi_df, sis_df], ignore_index=True), {"public_train_source": "combined"}
    raise ValueError(f"Unsupported public_train_source: {source}")


def _select_public_eval_table(
    source: str,
    *,
    mobi_df: pd.DataFrame,
    sis_df: pd.DataFrame,
) -> tuple[pd.DataFrame | None, str]:
    if source == "none":
        return None, "none"
    if source == "mobifall":
        return mobi_df.copy(), "mobifall"
    if source == "sisfall":
        return sis_df.copy(), "sisfall"
    raise ValueError(f"Unsupported public_eval_source: {source}")


def _prepare_phone_training_and_eval_tables(
    phone_df: pd.DataFrame,
    *,
    phone_negatives_only: bool,
    include_phone_positives: bool,
    hard_negative_types: list[str],
    phone_max_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if "use_for_fall_supervision" not in phone_df.columns or "fall_target_label" not in phone_df.columns:
        raise ValueError("Phone labeled CSV is missing use_for_fall_supervision / fall_target_label")

    eval_df = phone_df[phone_df["use_for_fall_supervision"].astype(bool)].copy()
    eval_df = eval_df[eval_df["fall_target_label"].notna()].copy()
    eval_df["true_label"] = eval_df["fall_target_label"].astype(str)
    eval_df = eval_df.reset_index(drop=True)

    train_df = eval_df.copy()

    if phone_negatives_only:
        train_df = train_df[train_df["true_label"].astype(str).eq("non_fall")].copy()

    if "fall_hard_negative_type" in train_df.columns and hard_negative_types:
        train_df = train_df[
            train_df["fall_hard_negative_type"].fillna("").astype(str).isin(hard_negative_types)
            | train_df["true_label"].astype(str).eq("fall")
        ].copy()

    if not include_phone_positives:
        train_df = train_df[train_df["true_label"].astype(str).ne("fall")].copy()

    if phone_max_rows > 0 and len(train_df) > phone_max_rows:
        train_df = train_df.sample(n=int(phone_max_rows), random_state=42).reset_index(drop=True)
    else:
        train_df = train_df.reset_index(drop=True)

    summary = {
        "phone_eval_rows": int(len(eval_df)),
        "phone_eval_label_counts": eval_df["true_label"].astype(str).value_counts(dropna=False).to_dict(),
        "phone_train_rows_selected": int(len(train_df)),
        "phone_train_label_counts": train_df["true_label"].astype(str).value_counts(dropna=False).to_dict()
        if not train_df.empty
        else {},
        "phone_train_hard_negative_type_counts": train_df["fall_hard_negative_type"].dropna().astype(str).value_counts(dropna=False).to_dict()
        if "fall_hard_negative_type" in train_df.columns and not train_df.empty
        else {},
    }
    return train_df, eval_df, summary


def _side_by_side_prediction_export(
    baseline_pred_df: pd.DataFrame,
    adapted_pred_df: pd.DataFrame,
    *,
    include_phone_fields: bool,
) -> pd.DataFrame:
    baseline = baseline_pred_df.copy()
    adapted = adapted_pred_df.copy()

    baseline = baseline.reset_index(drop=True)
    adapted = adapted.reset_index(drop=True)

    keep = [
        "dataset_name",
        "subject_id",
        "session_id",
        "window_id",
        "start_ts",
        "end_ts",
        "midpoint_ts",
        "true_label",
        "annotation_label",
        "fall_hard_negative_type",
    ]
    left = baseline[[c for c in keep if c in baseline.columns]].copy()
    left["baseline_predicted_label"] = baseline["predicted_label"].astype("string")
    left["baseline_predicted_probability"] = baseline["predicted_probability"]

    left["adapted_predicted_label"] = adapted["predicted_label"].astype("string")
    left["adapted_predicted_probability"] = adapted["predicted_probability"]

    if include_phone_fields and "annotation_notes" in baseline.columns:
        left["annotation_notes"] = baseline["annotation_notes"]

    return left


def main() -> int:
    args = parse_args()

    mobifall_path = _resolve_path(args.mobifall_path)
    sisfall_path = _resolve_path(args.sisfall_path)
    phone_csv = _resolve_path(args.phone_fall_labeled_csv)
    out_json = _resolve_path(args.out_json)
    phone_predictions_out = _resolve_path(args.phone_predictions_out)
    public_predictions_out = _resolve_path(args.public_predictions_out)

    if not mobifall_path.exists():
        raise FileNotFoundError(f"MobiFall path not found: {mobifall_path}")
    if not sisfall_path.exists():
        raise FileNotFoundError(f"SisFall path not found: {sisfall_path}")
    if not phone_csv.exists():
        raise FileNotFoundError(f"Phone labeled CSV not found: {phone_csv}")

    print("Preparing public threshold prediction tables...")
    mobi_pred_df, mobi_summary = _prepare_public_prediction_table(
        dataset_key="mobifall",
        path=mobifall_path,
        threshold_mode=args.threshold_mode,
        target_rate=args.target_rate,
        window_size=args.window_size,
        step_size=args.step_size,
    )
    sis_pred_df, sis_summary = _prepare_public_prediction_table(
        dataset_key="sisfall",
        path=sisfall_path,
        threshold_mode=args.threshold_mode,
        target_rate=args.target_rate,
        window_size=args.window_size,
        step_size=args.step_size,
    )

    public_train_df, public_train_meta = _select_public_train_table(
        args.public_train_source,
        mobi_df=mobi_pred_df,
        sis_df=sis_pred_df,
    )
    public_eval_df, public_eval_name = _select_public_eval_table(
        args.public_eval_source,
        mobi_df=mobi_pred_df,
        sis_df=sis_pred_df,
    )

    print("Loading phone labeled fall windows...")
    phone_labeled_df = pd.read_csv(phone_csv)
    hard_negative_types = [s.strip() for s in args.phone_hard_negative_types.split(",") if s.strip()]
    phone_train_df, phone_eval_df, phone_summary = _prepare_phone_training_and_eval_tables(
        phone_labeled_df,
        phone_negatives_only=bool(args.phone_negatives_only),
        include_phone_positives=bool(args.include_phone_positives),
        hard_negative_types=hard_negative_types,
        phone_max_rows=int(args.phone_max_rows),
    )

    config = FallMetaModelConfig(
        test_size=0.30,
        random_state=args.random_state,
    )

    print("Training baseline public-only fall model...")
    baseline_model, baseline_features, baseline_threshold, baseline_tuning = _fit_source_model(
        public_train_df,
        config=config,
    )

    print("Training adapted fall model with phone hard negatives...")
    adapted_train_df = public_train_df.copy()
    if not phone_train_df.empty:
        adapted_train_df = pd.concat([adapted_train_df, phone_train_df], ignore_index=True)

    adapted_model, adapted_features, adapted_threshold, adapted_tuning = _fit_source_model(
        adapted_train_df,
        config=config,
    )

    phone_predictions_out.parent.mkdir(parents=True, exist_ok=True)
    public_predictions_out.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    print("Evaluating on phone labeled windows...")
    baseline_phone_metrics, baseline_phone_pred = _evaluate_model_on_df(
        phone_eval_df,
        model=baseline_model,
        used_features=baseline_features,
        probability_threshold=baseline_threshold,
        config=config,
    )
    adapted_phone_metrics, adapted_phone_pred = _evaluate_model_on_df(
        phone_eval_df,
        model=adapted_model,
        used_features=adapted_features,
        probability_threshold=adapted_threshold,
        config=config,
    )

    phone_side_by_side = _side_by_side_prediction_export(
        baseline_phone_pred,
        adapted_phone_pred,
        include_phone_fields=True,
    )
    phone_side_by_side.to_csv(phone_predictions_out, index=False)

    public_metrics_block: dict[str, Any] = {}
    if public_eval_df is not None:
        print(f"Evaluating on public source: {public_eval_name}")
        baseline_public_metrics, baseline_public_pred = _evaluate_model_on_df(
            public_eval_df,
            model=baseline_model,
            used_features=baseline_features,
            probability_threshold=baseline_threshold,
            config=config,
        )
        adapted_public_metrics, adapted_public_pred = _evaluate_model_on_df(
            public_eval_df,
            model=adapted_model,
            used_features=adapted_features,
            probability_threshold=adapted_threshold,
            config=config,
        )

        public_side_by_side = _side_by_side_prediction_export(
            baseline_public_pred,
            adapted_public_pred,
            include_phone_fields=False,
        )
        public_side_by_side.to_csv(public_predictions_out, index=False)

        public_metrics_block = {
            "eval_source": public_eval_name,
            "baseline": baseline_public_metrics,
            "adapted": adapted_public_metrics,
        }
    else:
        public_metrics_block = {
            "eval_source": "none",
            "baseline": None,
            "adapted": None,
        }

    summary = {
        "evaluation_name": "fall_phone_hard_negative_adaptation",
        "config": {
            "public_train_source": args.public_train_source,
            "public_eval_source": args.public_eval_source,
            "threshold_mode": args.threshold_mode,
            "target_rate_hz": float(args.target_rate),
            "window_size": int(args.window_size),
            "step_size": int(args.step_size),
            "random_state": int(args.random_state),
            "phone_negatives_only": bool(args.phone_negatives_only),
            "include_phone_positives": bool(args.include_phone_positives),
            "phone_hard_negative_types": hard_negative_types,
            "phone_max_rows": int(args.phone_max_rows),
        },
        "public_data": {
            "mobifall": mobi_summary,
            "sisfall": sis_summary,
            "public_train_rows": int(len(public_train_df)),
            "public_train_label_counts": public_train_df["true_label"].astype(str).value_counts(dropna=False).to_dict(),
        },
        "phone_data": phone_summary,
        "baseline_model": {
            "train_rows": int(len(public_train_df)),
            "used_features": baseline_features,
            "selected_probability_threshold": float(baseline_threshold),
            "threshold_tuning_table_top": baseline_tuning.head(10) if not baseline_tuning.empty else [],
        },
        "adapted_model": {
            "train_rows": int(len(adapted_train_df)),
            "used_features": adapted_features,
            "selected_probability_threshold": float(adapted_threshold),
            "threshold_tuning_table_top": adapted_tuning.head(10) if not adapted_tuning.empty else [],
        },
        "phone_eval": {
            "baseline": baseline_phone_metrics,
            "adapted": adapted_phone_metrics,
            "baseline_false_positive_breakdown": _false_positive_breakdown(baseline_phone_pred),
            "adapted_false_positive_breakdown": _false_positive_breakdown(adapted_phone_pred),
        },
        "public_eval": public_metrics_block,
        "outputs": {
            "out_json": str(out_json),
            "phone_predictions_out": str(phone_predictions_out),
            "public_predictions_out": str(public_predictions_out),
        },
    }

    out_json.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")

    if args.save_adapted_model:
        import joblib
        save_path = _resolve_path(args.save_adapted_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model": adapted_model,
            "used_features": adapted_features,
            "probability_threshold": float(adapted_threshold),
            "positive_label": "fall",
            "negative_label": "non_fall",
            "metadata": {
                "artifact_role": "fall_meta_phone_negatives",
                "artifact_name": save_path.stem,
                "public_train_source": args.public_train_source,
                "phone_negatives_only": bool(args.phone_negatives_only),
                "include_phone_positives": bool(args.include_phone_positives),
                "phone_hard_negative_types": [
                    s.strip() for s in args.phone_hard_negative_types.split(",") if s.strip()
                ],
                "phone_train_rows": int(len(adapted_train_df)),
                "trained_at_iso": pd.Timestamp.utcnow().isoformat(),
            },
        }
        joblib.dump(bundle, save_path)
        print(f"Saved adapted fall meta-model bundle to: {save_path}")

    print()
    print(f"Saved summary JSON to: {out_json}")
    print(f"Saved phone prediction comparison CSV to: {phone_predictions_out}")
    if public_eval_df is not None:
        print(f"Saved public prediction comparison CSV to: {public_predictions_out}")
    print()
    print("Phone evaluation F1:")
    print(f"  baseline={baseline_phone_metrics['f1']:.4f}")
    print(f"  adapted ={adapted_phone_metrics['f1']:.4f}")
    print("Phone false-positive breakdown:")
    print(f"  baseline={summary['phone_eval']['baseline_false_positive_breakdown']}")
    print(f"  adapted ={summary['phone_eval']['adapted_false_positive_breakdown']}")
    if public_eval_df is not None:
        print("Public evaluation F1:")
        print(f"  baseline={public_metrics_block['baseline']['f1']:.4f}")
        print(f"  adapted ={public_metrics_block['adapted']['f1']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())