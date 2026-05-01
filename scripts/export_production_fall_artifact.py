#!/usr/bin/env python3
"""Export a production-ready fall artifact bundle.

Default target:
- public source: mobifall
- phone adaptation: hard negatives only
- runtime threshold: 0.75

This script trains the chosen deployment-oriented model and saves a bundle that
matches the existing fall artifact layout used by the runtime pipeline:
- model.joblib
- metrics.json
- coefficients.csv
- run_summary.json

The output can then be pointed to by apps/api/main.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.fall.evaluate_threshold_fall import build_threshold_prediction_table
from pipeline.fall.threshold_detector import default_fall_threshold_config
from pipeline.ingest import load_mobifall, load_sisfall
from pipeline.preprocess import (
    PreprocessConfig,
    append_derived_channels,
    resample_dataframe,
    window_dataframe,
)


DEFAULT_META_FEATURE_COLUMNS = [
    "peak_acc",
    "peak_over_mean_ratio",
    "peak_minus_mean",
    "acc_variance",
    "mean_acc",
    "acc_baseline",
    "jerk_peak",
    "jerk_mean",
    "jerk_rms",
    "gyro_peak",
    "gyro_mean",
    "post_impact_motion",
    "post_impact_variance",
    "post_impact_dyn_mean",
    "post_impact_dyn_rms",
    "post_impact_dyn_ratio_mean",
    "post_impact_dyn_ratio_rms",
    "post_impact_motion_to_peak_ratio",
    "stage_impact_pass",
    "stage_support_pass",
    "stage_confirm_pass",
]

DEFAULT_PHONE_HARD_NEGATIVE_TYPES = [
    "walking",
    "stairs",
    "sit_down_transition",
    "phone_handling",
    "sitting",
    "other",
]


@dataclass(slots=True)
class ProductionFallArtifactConfig:
    positive_label: str = "fall"
    negative_label: str = "non_fall"
    feature_columns: list[str] = field(default_factory=lambda: list(DEFAULT_META_FEATURE_COLUMNS))
    probability_threshold: float = 0.75
    logistic_c: float = 1.0
    logistic_max_iter: int = 2000
    class_weight: str | dict[str, float] | None = "balanced"
    random_state: int = 42
    public_train_source: str = "mobifall"
    phone_negatives_only: bool = True
    include_phone_positives: bool = False
    phone_hard_negative_types: list[str] = field(default_factory=lambda: list(DEFAULT_PHONE_HARD_NEGATIVE_TYPES))
    threshold_mode: str = "shared"
    target_rate_hz: float = 100.0
    window_size: int = 128
    step_size: int = 64
    # v2 additions — calibration + per-source threshold tuning. Defaults
    # preserve v1 behaviour when the relevant CLI flags are not passed.
    calibrate: str = "none"  # one of {"none", "isotonic", "sigmoid"}
    threshold_mode_runtime: str = "shared"  # one of {"shared", "per_source"}
    validation_fraction: float = 0.2
    threshold_grid: tuple[float, ...] = (
        0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
    )
    threshold_metric: str = "f1"


def parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Export a production fall artifact bundle")
    parser.add_argument("--mobifall-path", default="data/raw/MOBIACT_Dataset/MobiFall_Dataset_v2.0")
    parser.add_argument("--sisfall-path", default="data/raw/SISFALL_Dataset/SisFall_dataset")
    parser.add_argument(
        "--phone-fall-labeled-csv",
        default="results/validation/phone1_fall_labeled_windows.csv",
        help="Phone labeled fall windows from build_phone_runtime_training_set.py",
    )
    parser.add_argument(
        "--public-source",
        choices=["mobifall", "sisfall", "combined"],
        default="mobifall",
        help="Public source used as the base training data",
    )
    parser.add_argument(
        "--public-predictions-csv",
        default="",
        help="Optional cached public threshold prediction CSV; skips raw rebuild when provided",
    )
    parser.add_argument(
        "--include-phone-positives",
        action="store_true",
        help="Include phone fall-labeled rows in addition to hard negatives",
    )
    parser.add_argument(
        "--phone-negatives-only",
        action="store_true",
        help="Explicitly force negatives-only phone adaptation (recommended deployment default)",
    )
    parser.add_argument(
        "--phone-hard-negative-types",
        default=",".join(DEFAULT_PHONE_HARD_NEGATIVE_TYPES),
        help="Comma-separated phone hard-negative types to include",
    )
    parser.add_argument(
        "--probability-threshold",
        type=float,
        default=0.75,
        help="Fallback runtime threshold (used when --threshold-mode-runtime=shared and no tuning is available)",
    )
    parser.add_argument(
        "--calibrate",
        choices=["none", "isotonic", "sigmoid"],
        default="none",
        help="Optional probability calibration on a held-out subject-aware fold",
    )
    parser.add_argument(
        "--threshold-mode-runtime",
        choices=["shared", "per_source"],
        default="shared",
        help="If per_source, tune separate thresholds for MobiFall and PHONE on the validation fold and "
        "select the phone threshold as the runtime default",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of training data held out (subject-aware) for calibration + threshold tuning",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/fall/fall_meta_phone_negatives_v1",
        help="Artifact bundle output directory",
    )
    return parser.parse_args()


# -- v2 helpers: subject-aware split + threshold sweep + per-source metrics --


def _get_subject_groups(df: pd.DataFrame) -> pd.Series:
    """Identity tag per row used for subject-aware splitting.

    Combines dataset_name + subject_id so MobiFall sub1 and a phone subject
    happening to share the id never end up in the same group.
    """
    dataset = df["dataset_name"].astype(str) if "dataset_name" in df.columns else pd.Series([""] * len(df))
    subject = df["subject_id"].astype(str) if "subject_id" in df.columns else pd.Series([""] * len(df))
    return (dataset.fillna("") + "::" + subject.fillna("")).reset_index(drop=True)


def _split_subject_aware(
    df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = _get_subject_groups(df)
    if groups.nunique(dropna=True) < 2:
        # Not enough distinct groups to split — return train, empty val.
        return df.reset_index(drop=True), df.iloc[0:0].copy().reset_index(drop=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(df, groups=groups))
    train_df = df.iloc[np.sort(train_idx)].reset_index(drop=True)
    val_df = df.iloc[np.sort(val_idx)].reset_index(drop=True)
    return train_df, val_df


def _score_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, metric: str) -> float:
    y_pred = (y_prob >= threshold).astype(int)
    if metric == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if metric == "precision":
        return float(precision_score(y_true, y_pred, zero_division=0))
    if metric == "recall":
        return float(recall_score(y_true, y_pred, zero_division=0))
    if metric == "balanced_score":
        return 0.5 * (
            float(precision_score(y_true, y_pred, zero_division=0))
            + float(recall_score(y_true, y_pred, zero_division=0))
        )
    raise ValueError(f"Unsupported threshold metric: {metric}")


def _sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    grid: tuple[float, ...],
    metric: str,
    fallback: float,
) -> tuple[float, list[dict[str, float]]]:
    """Pick the threshold that maximises ``metric`` over ``grid``."""
    rows: list[dict[str, float]] = []
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return fallback, [{"threshold": fallback, "score": float("nan")}]
    best_t, best_score = fallback, float("-inf")
    for t in grid:
        s = _score_at_threshold(y_true, y_prob, float(t), metric)
        rows.append({"threshold": float(t), "score": float(s)})
        if s > best_score:
            best_score, best_t = float(s), float(t)
    return best_t, rows


def _per_source_metrics(
    val_df: pd.DataFrame,
    y_true_val: np.ndarray,
    y_prob_val: np.ndarray,
    *,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    """Confusion-matrix-style metrics computed per dataset_name on the val fold."""
    out: dict[str, dict[str, Any]] = {}
    if val_df.empty or "dataset_name" not in val_df.columns:
        return out
    dataset_names = val_df["dataset_name"].astype(str).fillna("").to_numpy()
    for source in sorted(set(dataset_names)):
        if not source:
            continue
        mask = dataset_names == source
        if not mask.any():
            continue
        yt = y_true_val[mask]
        yp = y_prob_val[mask]
        y_pred = (yp >= threshold).astype(int)
        tp = int(((y_pred == 1) & (yt == 1)).sum())
        tn = int(((y_pred == 0) & (yt == 0)).sum())
        fp = int(((y_pred == 1) & (yt == 0)).sum())
        fn = int(((y_pred == 0) & (yt == 1)).sum())
        total = tp + tn + fp + fn
        positives = int((yt == 1).sum())
        out[source] = {
            "support_total": total,
            "support_positive": positives,
            "support_negative": total - positives,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            "f1": float(f1_score(yt, y_pred, zero_division=0)) if total > 0 else 0.0,
            "threshold_used": float(threshold),
        }
    return out


def _resolve_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().resolve()


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


def _prepare_feature_frame(df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    available = [c for c in feature_columns if c in df.columns]
    if not available:
        raise ValueError("None of the requested fall feature columns are present in the training dataframe")

    X = pd.DataFrame(index=df.index)
    for col in available:
        X[col] = _coerce_boolean_like_to_float(df[col])

    return X, available


def _binarise_labels(labels: pd.Series, *, positive_label: str, negative_label: str) -> np.ndarray:
    mapped = labels.astype(str).map({positive_label: 1, negative_label: 0})
    if mapped.isna().any():
        bad = sorted(labels[mapped.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unexpected labels present: {bad}")
    return mapped.astype(int).to_numpy()


def _load_fall_dataframe(dataset_key: str, path: Path) -> pd.DataFrame:
    if dataset_key == "mobifall":
        return load_mobifall(path)
    if dataset_key == "sisfall":
        return load_sisfall(path)
    raise ValueError(f"Unsupported dataset: {dataset_key}")


def _prepare_public_prediction_table(
    *,
    dataset_key: str,
    path: Path,
    threshold_mode: str,
    target_rate_hz: float,
    window_size: int,
    step_size: int,
) -> pd.DataFrame:
    df = _load_fall_dataframe(dataset_key, path)
    dataset_name = (
        str(df["dataset_name"].dropna().astype(str).iloc[0])
        if "dataset_name" in df.columns and not df.empty
        else dataset_key.upper()
    )

    resampled = resample_dataframe(df, target_rate_hz=target_rate_hz)
    resampled = append_derived_channels(resampled)

    preprocess_cfg = PreprocessConfig(target_sampling_rate_hz=target_rate_hz)
    windows = window_dataframe(
        resampled,
        window_size=window_size,
        step_size=step_size,
        config=preprocess_cfg,
    )

    detector_config = default_fall_threshold_config(dataset_name if threshold_mode == "dataset_presets" else None)
    pred_df = build_threshold_prediction_table(
        windows,
        detector_config=detector_config,
        filter_unacceptable=True,
        default_sampling_rate_hz=target_rate_hz,
    )
    if pred_df.empty:
        raise ValueError(f"Public threshold prediction table is empty for {dataset_key}")
    return pred_df.reset_index(drop=True)


def _load_public_train_df(
    *,
    public_source: str,
    public_predictions_csv: str,
    mobifall_path: Path,
    sisfall_path: Path,
    config: ProductionFallArtifactConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if public_predictions_csv:
        csv_path = _resolve_path(public_predictions_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"Public predictions CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        if "true_label" not in df.columns:
            raise ValueError("Cached public predictions CSV must contain true_label")
        return df.reset_index(drop=True), {
            "source": public_source,
            "mode": "cached_csv",
            "path": str(csv_path),
            "rows": int(len(df)),
            "label_counts": df["true_label"].astype(str).value_counts(dropna=False).to_dict(),
        }

    if public_source == "mobifall":
        df = _prepare_public_prediction_table(
            dataset_key="mobifall",
            path=mobifall_path,
            threshold_mode=config.threshold_mode,
            target_rate_hz=config.target_rate_hz,
            window_size=config.window_size,
            step_size=config.step_size,
        )
    elif public_source == "sisfall":
        df = _prepare_public_prediction_table(
            dataset_key="sisfall",
            path=sisfall_path,
            threshold_mode=config.threshold_mode,
            target_rate_hz=config.target_rate_hz,
            window_size=config.window_size,
            step_size=config.step_size,
        )
    elif public_source == "combined":
        mobi = _prepare_public_prediction_table(
            dataset_key="mobifall",
            path=mobifall_path,
            threshold_mode=config.threshold_mode,
            target_rate_hz=config.target_rate_hz,
            window_size=config.window_size,
            step_size=config.step_size,
        )
        sis = _prepare_public_prediction_table(
            dataset_key="sisfall",
            path=sisfall_path,
            threshold_mode=config.threshold_mode,
            target_rate_hz=config.target_rate_hz,
            window_size=config.window_size,
            step_size=config.step_size,
        )
        df = pd.concat([mobi, sis], ignore_index=True)
    else:
        raise ValueError(f"Unsupported public source: {public_source}")

    return df.reset_index(drop=True), {
        "source": public_source,
        "mode": "rebuilt_from_raw",
        "rows": int(len(df)),
        "label_counts": df["true_label"].astype(str).value_counts(dropna=False).to_dict(),
    }


def _prepare_phone_train_df(
    phone_df: pd.DataFrame,
    *,
    config: ProductionFallArtifactConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"use_for_fall_supervision", "fall_target_label"}
    missing = required - set(phone_df.columns)
    if missing:
        raise ValueError(f"Phone labeled CSV is missing required columns: {sorted(missing)}")

    eval_df = phone_df[phone_df["use_for_fall_supervision"].astype(bool)].copy()
    eval_df = eval_df[eval_df["fall_target_label"].notna()].copy()
    eval_df["true_label"] = eval_df["fall_target_label"].astype(str)
    eval_df = eval_df.reset_index(drop=True)

    train_df = eval_df.copy()

    if config.phone_negatives_only:
        train_df = train_df[train_df["true_label"].astype(str).eq("non_fall")].copy()

    if "fall_hard_negative_type" in train_df.columns and config.phone_hard_negative_types:
        train_df = train_df[
            train_df["fall_hard_negative_type"].fillna("").astype(str).isin(config.phone_hard_negative_types)
            | train_df["true_label"].astype(str).eq("fall")
        ].copy()

    if not config.include_phone_positives:
        train_df = train_df[train_df["true_label"].astype(str).ne("fall")].copy()

    train_df = train_df.reset_index(drop=True)

    return train_df, {
        "rows_selected": int(len(train_df)),
        "label_counts": train_df["true_label"].astype(str).value_counts(dropna=False).to_dict()
        if not train_df.empty
        else {},
        "hard_negative_type_counts": train_df["fall_hard_negative_type"].dropna().astype(str).value_counts(dropna=False).to_dict()
        if "fall_hard_negative_type" in train_df.columns and not train_df.empty
        else {},
    }


def _build_model_pipeline(config: ProductionFallArtifactConfig) -> Pipeline:
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


def _extract_logistic_coefficients(model: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    logistic = model.named_steps["logistic_regression"]
    coeffs = np.asarray(logistic.coef_).reshape(-1)
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coeffs,
            "abs_coefficient": np.abs(coeffs),
        }
    )
    return df.sort_values("abs_coefficient", ascending=False, kind="stable").reset_index(drop=True)


def main() -> int:
    args = parse_args()

    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mobifall_path = _resolve_path(args.mobifall_path)
    sisfall_path = _resolve_path(args.sisfall_path)
    phone_csv = _resolve_path(args.phone_fall_labeled_csv)

    if not phone_csv.exists():
        raise FileNotFoundError(f"Phone labeled CSV not found: {phone_csv}")
    if args.public_source in {"mobifall", "combined"} and not mobifall_path.exists():
        raise FileNotFoundError(f"MobiFall path not found: {mobifall_path}")
    if args.public_source in {"sisfall", "combined"} and not sisfall_path.exists():
        raise FileNotFoundError(f"SisFall path not found: {sisfall_path}")

    cfg = ProductionFallArtifactConfig(
        public_train_source=args.public_source,
        phone_negatives_only=bool(args.phone_negatives_only or not args.include_phone_positives),
        include_phone_positives=bool(args.include_phone_positives),
        phone_hard_negative_types=[s.strip() for s in args.phone_hard_negative_types.split(",") if s.strip()],
        probability_threshold=float(args.probability_threshold),
        calibrate=str(args.calibrate),
        threshold_mode_runtime=str(args.threshold_mode_runtime),
        validation_fraction=float(args.validation_fraction),
    )

    print("Loading public training data...")
    public_train_df, public_summary = _load_public_train_df(
        public_source=args.public_source,
        public_predictions_csv=args.public_predictions_csv,
        mobifall_path=mobifall_path,
        sisfall_path=sisfall_path,
        config=cfg,
    )

    print("Loading phone labeled windows...")
    phone_labeled_df = pd.read_csv(phone_csv)
    phone_train_df, phone_summary = _prepare_phone_train_df(phone_labeled_df, config=cfg)

    train_df = public_train_df.copy()
    if not phone_train_df.empty:
        train_df = pd.concat([train_df, phone_train_df], ignore_index=True)

    print(f"Training rows: {len(train_df)}")

    # Subject-aware split for calibration + threshold tuning. Both layers
    # are no-ops when validation_fraction == 0 (preserves v1 behaviour).
    needs_validation = (
        cfg.calibrate != "none" or cfg.threshold_mode_runtime == "per_source"
    )
    if needs_validation and cfg.validation_fraction > 0:
        fit_df, val_df = _split_subject_aware(
            train_df,
            test_size=cfg.validation_fraction,
            random_state=cfg.random_state,
        )
    else:
        fit_df = train_df.reset_index(drop=True)
        val_df = train_df.iloc[0:0].copy().reset_index(drop=True)

    print(
        f"Fit rows: {len(fit_df)}, validation rows: {len(val_df)} "
        f"(calibrate={cfg.calibrate}, threshold_mode_runtime={cfg.threshold_mode_runtime})"
    )

    X_fit, used_features = _prepare_feature_frame(fit_df, cfg.feature_columns)
    y_fit = _binarise_labels(
        fit_df["true_label"],
        positive_label=cfg.positive_label,
        negative_label=cfg.negative_label,
    )

    base_model = _build_model_pipeline(cfg)
    base_model.fit(X_fit, y_fit)

    # Coefficients come from the base LR even after calibration —
    # CalibratedClassifierCV wraps it but the underlying LR is unchanged.
    coeff_df = _extract_logistic_coefficients(base_model, used_features)

    # Optional probability calibration on the held-out validation fold.
    model = base_model
    calibration_summary: dict[str, Any] = {"method": cfg.calibrate}
    if cfg.calibrate != "none" and not val_df.empty:
        X_val, _ = _prepare_feature_frame(val_df, used_features)
        y_val = _binarise_labels(
            val_df["true_label"],
            positive_label=cfg.positive_label,
            negative_label=cfg.negative_label,
        )
        if len(np.unique(y_val)) >= 2:
            calibrator = CalibratedClassifierCV(base_model, method=cfg.calibrate, cv="prefit")
            calibrator.fit(X_val, y_val)
            model = calibrator
            calibration_summary.update({
                "applied": True,
                "calibration_rows": int(len(val_df)),
                "calibration_positive_rows": int((y_val == 1).sum()),
            })
        else:
            calibration_summary.update({
                "applied": False,
                "skipped_reason": "validation fold has only one class",
            })
    elif cfg.calibrate != "none":
        calibration_summary.update({
            "applied": False,
            "skipped_reason": "validation fold is empty (validation_fraction=0?)",
        })
    else:
        calibration_summary.update({"applied": False})

    # Per-source threshold tuning on the validation fold (after calibration
    # if calibration was applied — we tune in the deployed probability space).
    threshold_summary: dict[str, Any] = {
        "mode": cfg.threshold_mode_runtime,
        "fallback_threshold": float(cfg.probability_threshold),
    }
    selected_threshold = float(cfg.probability_threshold)
    per_source_metrics: dict[str, dict[str, Any]] = {}
    threshold_rows: list[dict[str, Any]] = []

    if cfg.threshold_mode_runtime == "per_source" and not val_df.empty:
        X_val, _ = _prepare_feature_frame(val_df, used_features)
        y_val = _binarise_labels(
            val_df["true_label"],
            positive_label=cfg.positive_label,
            negative_label=cfg.negative_label,
        )
        y_prob_val = model.predict_proba(X_val)[:, 1]

        # Sweep per dataset_name.
        ds_names = val_df["dataset_name"].astype(str).fillna("").to_numpy() if "dataset_name" in val_df.columns else np.array([""] * len(val_df))
        per_source_thresholds: dict[str, float] = {}
        for source in sorted(set(ds_names)):
            if not source:
                continue
            mask = ds_names == source
            t, sweep_rows = _sweep_thresholds(
                y_val[mask],
                y_prob_val[mask],
                grid=cfg.threshold_grid,
                metric=cfg.threshold_metric,
                fallback=cfg.probability_threshold,
            )
            per_source_thresholds[source] = t
            for row in sweep_rows:
                threshold_rows.append({"source": source, **row})

        threshold_summary["per_source_thresholds"] = per_source_thresholds

        # Production traffic is phone-only; prefer the phone threshold if
        # we found one, else fall back to MOBIFALL, else to the CLI default.
        if "PHONE1" in per_source_thresholds:
            selected_threshold = per_source_thresholds["PHONE1"]
            threshold_summary["selected_source"] = "PHONE1"
        elif "MOBIFALL" in per_source_thresholds:
            selected_threshold = per_source_thresholds["MOBIFALL"]
            threshold_summary["selected_source"] = "MOBIFALL"
        else:
            threshold_summary["selected_source"] = "fallback"

        per_source_metrics = _per_source_metrics(
            val_df, y_val, y_prob_val, threshold=selected_threshold
        )

    # Update the runtime-effective threshold used in the bundle so callers
    # observe the tuned value, not the CLI default.
    cfg.probability_threshold = float(selected_threshold)

    model_path = output_dir / "model.joblib"
    metrics_path = output_dir / "metrics.json"
    coeffs_path = output_dir / "coefficients.csv"
    run_summary_path = output_dir / "run_summary.json"

    artifact_bundle = {
        "model": model,
        "used_features": used_features,
        "probability_threshold": float(cfg.probability_threshold),
        "positive_label": str(cfg.positive_label),
        "negative_label": str(cfg.negative_label),
        "metadata": {
            "artifact_role": "production_fall_model",
            "artifact_name": output_dir.name,
            "public_train_source": str(cfg.public_train_source),
            "phone_negatives_only": bool(cfg.phone_negatives_only),
            "include_phone_positives": bool(cfg.include_phone_positives),
            "phone_hard_negative_types": list(cfg.phone_hard_negative_types),
            "target_rate_hz": float(cfg.target_rate_hz),
            "window_size": int(cfg.window_size),
            "step_size": int(cfg.step_size),
            "random_state": int(cfg.random_state),
            "feature_count": int(len(used_features)),
            "calibration": calibration_summary,
            "threshold_summary": threshold_summary,
        },
    }
    joblib.dump(artifact_bundle, model_path)
    coeff_df.to_csv(coeffs_path, index=False)

    threshold_tuning_path = output_dir / "threshold_tuning.csv"
    if threshold_rows:
        pd.DataFrame(threshold_rows).to_csv(threshold_tuning_path, index=False)

    metrics_payload = {
        "config": asdict(cfg),
        "used_features": used_features,
        "split": {
            "validation_fraction": float(cfg.validation_fraction),
            "fit_rows": int(len(fit_df)),
            "validation_rows": int(len(val_df)),
        },
        "metrics": {
            "train_rows": int(len(train_df)),
            "train_label_counts": train_df["true_label"].astype(str).value_counts(dropna=False).to_dict(),
            "public_summary": public_summary,
            "phone_summary": phone_summary,
            "calibration": calibration_summary,
            "threshold_summary": threshold_summary,
            "per_source": per_source_metrics,
        },
    }
    metrics_path.write_text(json.dumps(_json_safe(metrics_payload), indent=2), encoding="utf-8")

    run_summary = {
        "artifact_name": output_dir.name,
        "artifact_role": "production_fall_model",
        "selected_runtime_threshold": float(cfg.probability_threshold),
        "artifacts": {
            "model_joblib": str(model_path),
            "metrics_json": str(metrics_path),
            "coefficients_csv": str(coeffs_path),
            **(
                {"threshold_tuning_csv": str(threshold_tuning_path)}
                if threshold_rows else {}
            ),
        },
        "config": asdict(cfg),
        "used_features": used_features,
        "public_summary": public_summary,
        "phone_summary": phone_summary,
        "calibration": calibration_summary,
        "threshold_summary": threshold_summary,
        "per_source": per_source_metrics,
        "top_coefficients": _json_safe(coeff_df.head(15)),
    }
    run_summary_path.write_text(json.dumps(_json_safe(run_summary), indent=2), encoding="utf-8")

    print()
    print(f"Saved production artifact bundle to: {output_dir}")
    print(f"Model: {model_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Coefficients: {coeffs_path}")
    print(f"Run summary: {run_summary_path}")
    print()
    print(f"Selected runtime threshold: {cfg.probability_threshold:.3f} ({threshold_summary.get('selected_source', 'static')})")
    print(f"Calibration: {calibration_summary}")
    if per_source_metrics:
        print(f"Per-source eval (validation fold, threshold={cfg.probability_threshold:.3f}):")
        for source, m in per_source_metrics.items():
            print(
                f"  {source}: F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} "
                f"FP={m['fp']} FN={m['fn']} (n={m['support_total']}, +{m['support_positive']})"
            )
    print(f"Used features: {len(used_features)}")
    print(f"Phone adaptation rows: {phone_summary['rows_selected']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())