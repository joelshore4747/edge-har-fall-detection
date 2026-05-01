#!/usr/bin/env python3
"""Run cross-dataset fall evaluation between MobiFall and SisFall.

Purpose:
- build fall windows from normalized loaders
- evaluate the transparent threshold baseline within each dataset
- evaluate the logistic fall meta-model within each dataset
- evaluate cross-dataset transfer of the fall meta-model
- keep threshold transfer source-independent by reporting target-only threshold reference

Important default:
- uses a SHARED threshold config by default, not dataset-specific presets
- this is recommended after unit harmonization, especially because the old
  SisFall preset was tuned for a different scale regime
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.fall_metrics import compute_fall_detection_metrics
from models.fall.evaluate_threshold_fall import (
    build_threshold_prediction_table,
    evaluate_threshold_fall_predictions,
)
from models.fall.train_fall_meta_model import FallMetaModelConfig, train_fall_meta_model
from pipeline.fall.threshold_detector import default_fall_threshold_config
from pipeline.ingest import load_mobiact_v2, load_mobifall, load_sisfall
from pipeline.preprocess import (
    PreprocessConfig,
    append_derived_channels,
    resample_dataframe,
    window_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cross-dataset fall evaluation")
    parser.add_argument("--mobifall-path", default="data/raw/MOBIACT_Dataset/MobiFall_Dataset_v2.0")
    parser.add_argument("--sisfall-path", default="data/raw/SISFALL_Dataset/SisFall_dataset")
    parser.add_argument(
        "--mobiact-v2-path",
        default="data/raw/MobiAct_Dataset_v2.0/Annotated Data",
        help="Path to MobiAct v2 Annotated Data (used only when --eval-source=mobiact_v2)",
    )

    parser.add_argument(
        "--train-source",
        choices=["mobifall", "sisfall", "combined"],
        default="mobifall",
        help="Dataset used to train the cross-dataset fall meta-model",
    )
    parser.add_argument(
        "--eval-source",
        choices=["mobifall", "sisfall", "mobiact_v2"],
        default="sisfall",
        help=(
            "Dataset used for cross-dataset evaluation. "
            "mobiact_v2 is treated as an external held-out corpus: it is never "
            "available as a train_source by design."
        ),
    )
    parser.add_argument(
        "--mobiact-v2-max-files",
        type=int,
        default=None,
        help="Optional cap on MobiAct v2 trial CSVs loaded (smoke testing).",
    )

    parser.add_argument(
        "--threshold-mode",
        choices=["shared", "dataset_presets"],
        default="shared",
        help=(
            "shared = use one normalized threshold config for all datasets "
            "(recommended after unit harmonization); "
            "dataset_presets = use default_fall_threshold_config(dataset_name)"
        ),
    )

    parser.add_argument("--target-rate", type=float, default=100.0)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--step-size", type=int, default=64)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument(
        "--out-json",
        default="results/validation/fall_cross_dataset_eval.json",
        help="Output JSON path",
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


def _prepare_meta_feature_frame(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    available = [c for c in feature_columns if c in df.columns]
    if not available:
        raise ValueError("No requested meta-model feature columns are present in dataframe")

    X = pd.DataFrame(index=df.index)
    for col in available:
        X[col] = _coerce_boolean_like_to_float(df[col])

    return X, available


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
        raise ValueError(f"Unexpected labels in evaluation data: {bad}")
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


def _make_detector_config(dataset_name: str, threshold_mode: str):
    if threshold_mode == "dataset_presets":
        return default_fall_threshold_config(dataset_name)
    return default_fall_threshold_config(None)


def _load_fall_dataframe(
    dataset_key: str,
    path: Path,
    *,
    max_files: int | None = None,
) -> pd.DataFrame:
    if dataset_key == "mobifall":
        return load_mobifall(path)
    if dataset_key == "sisfall":
        return load_sisfall(path)
    if dataset_key == "mobiact_v2":
        return load_mobiact_v2(path, max_files=max_files)
    raise ValueError(f"Unsupported dataset: {dataset_key}")


def _prepare_window_set(
    *,
    dataset_key: str,
    path: Path,
    target_rate: float,
    window_size: int,
    step_size: int,
    max_files: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    df = _load_fall_dataframe(dataset_key, path, max_files=max_files)
    dataset_name = str(df["dataset_name"].dropna().astype(str).iloc[0]) if "dataset_name" in df.columns and not df.empty else dataset_key.upper()

    resampled = resample_dataframe(df, target_rate_hz=target_rate)
    resampled = append_derived_channels(resampled)

    cfg = PreprocessConfig(target_sampling_rate_hz=target_rate)
    windows = window_dataframe(
        resampled,
        window_size=window_size,
        step_size=step_size,
        config=cfg,
    )

    summary = {
        "dataset_key": dataset_key,
        "dataset_name": dataset_name,
        "rows_loaded": int(len(df)),
        "rows_after_resample": int(len(resampled)),
        "windows_total": int(len(windows)),
        "label_counts_rows": df["label_mapped"].astype(str).value_counts(dropna=False).to_dict() if "label_mapped" in df.columns else {},
    }
    return windows, summary


def _build_threshold_table_for_dataset(
    *,
    dataset_name: str,
    windows: list[dict[str, Any]],
    threshold_mode: str,
    target_rate: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    detector_config = _make_detector_config(dataset_name, threshold_mode)
    pred_df = build_threshold_prediction_table(
        windows,
        detector_config=detector_config,
        filter_unacceptable=True,
        default_sampling_rate_hz=target_rate,
    )
    if pred_df.empty:
        raise ValueError(f"Threshold prediction table is empty for {dataset_name}")

    summary = {
        "dataset_name": dataset_name,
        "rows": int(len(pred_df)),
        "true_label_counts": pred_df["true_label"].astype(str).value_counts(dropna=False).to_dict(),
        "predicted_label_counts": pred_df["predicted_label"].astype(str).value_counts(dropna=False).to_dict(),
        "detector_config": detector_config.__dict__,
    }
    return pred_df, summary


def _within_dataset_threshold_summary(
    pred_df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    result = evaluate_threshold_fall_predictions(
        pred_df,
        test_size=test_size,
        random_state=random_state,
        positive_label="fall",
        negative_label="non_fall",
    )
    return {
        "split": result["split"],
        "metrics": result["metrics"],
        "false_alarm_count": int(len(result["false_alarms"])),
    }


def _within_dataset_meta_summary(
    pred_df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    cfg = FallMetaModelConfig(
        test_size=test_size,
        random_state=random_state,
    )
    result = train_fall_meta_model(pred_df, config=cfg)
    coef_df = result["coefficient_table"]

    return {
        "split": result["split"],
        "metrics": result["metrics"],
        "used_features": result["used_features"],
        "top_coefficients": coef_df.head(15),
    }


def _select_meta_train_table(
    train_source: str,
    *,
    mobifall_pred_df: pd.DataFrame,
    sisfall_pred_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if train_source == "mobifall":
        return mobifall_pred_df.copy(), {
            "train_source": "mobifall",
            "train_dataset_names": ["MOBIFALL"],
        }
    if train_source == "sisfall":
        return sisfall_pred_df.copy(), {
            "train_source": "sisfall",
            "train_dataset_names": ["SISFALL"],
        }
    if train_source == "combined":
        combined = pd.concat([mobifall_pred_df, sisfall_pred_df], ignore_index=True)
        return combined, {
            "train_source": "combined",
            "train_dataset_names": sorted(combined["dataset_name"].dropna().astype(str).unique().tolist()),
        }
    raise ValueError(f"Unsupported train_source: {train_source}")


def _select_eval_table(
    eval_source: str,
    *,
    mobifall_pred_df: pd.DataFrame,
    sisfall_pred_df: pd.DataFrame,
    mobiact_v2_pred_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, str]:
    if eval_source == "mobifall":
        return mobifall_pred_df.copy(), "MOBIFALL"
    if eval_source == "sisfall":
        return sisfall_pred_df.copy(), "SISFALL"
    if eval_source == "mobiact_v2":
        if mobiact_v2_pred_df is None:
            raise ValueError(
                "eval_source=mobiact_v2 selected but MobiAct v2 prediction table "
                "was not built — check --mobiact-v2-path"
            )
        return mobiact_v2_pred_df.copy(), "MOBIACT_V2"
    raise ValueError(f"Unsupported eval_source: {eval_source}")


def _fit_meta_pipeline_on_source(
    train_df: pd.DataFrame,
    *,
    config: FallMetaModelConfig,
) -> tuple[Pipeline, list[str]]:
    X_train, used_features = _prepare_meta_feature_frame(train_df, config.feature_columns)
    y_train = _binarise_labels(
        train_df["true_label"],
        positive_label=config.positive_label,
        negative_label=config.negative_label,
    )

    model = Pipeline(
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
    model.fit(X_train, y_train)
    return model, used_features


def _evaluate_meta_transfer(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    config: FallMetaModelConfig,
) -> dict[str, Any]:
    # Use feature intersection so cross-dataset evaluation never silently breaks on missing columns.
    source_feats = [c for c in config.feature_columns if c in train_df.columns]
    target_feats = [c for c in config.feature_columns if c in eval_df.columns]
    used_features = [c for c in source_feats if c in target_feats]
    if not used_features:
        raise ValueError("No overlapping meta-model feature columns between source and target")

    fit_cfg = FallMetaModelConfig(
        positive_label=config.positive_label,
        negative_label=config.negative_label,
        feature_columns=used_features,
        test_size=config.test_size,
        random_state=config.random_state,
        logistic_c=config.logistic_c,
        logistic_max_iter=config.logistic_max_iter,
        class_weight=config.class_weight,
        probability_threshold=config.probability_threshold,
    )

    model, used_features = _fit_meta_pipeline_on_source(train_df, config=fit_cfg)

    X_eval, _ = _prepare_meta_feature_frame(eval_df, used_features)
    y_eval = _binarise_labels(
        eval_df["true_label"],
        positive_label=fit_cfg.positive_label,
        negative_label=fit_cfg.negative_label,
    )
    y_prob = model.predict_proba(X_eval)[:, 1]
    metrics = _compute_binary_metrics_with_probs(
        y_eval,
        y_prob,
        probability_threshold=fit_cfg.probability_threshold,
        positive_label=fit_cfg.positive_label,
        negative_label=fit_cfg.negative_label,
    )

    coef = model.named_steps["logistic_regression"].coef_.reshape(-1)
    coef_df = pd.DataFrame(
        {
            "feature": used_features,
            "coefficient": coef,
            "abs_coefficient": np.abs(coef),
        }
    ).sort_values("abs_coefficient", ascending=False, kind="stable").reset_index(drop=True)

    pred_labels = np.where(
        y_prob >= fit_cfg.probability_threshold,
        fit_cfg.positive_label,
        fit_cfg.negative_label,
    )

    return {
        "metrics": metrics,
        "used_features": used_features,
        "train_rows": int(len(train_df)),
        "eval_rows": int(len(eval_df)),
        "train_label_counts": train_df["true_label"].astype(str).value_counts(dropna=False).to_dict(),
        "eval_label_counts": eval_df["true_label"].astype(str).value_counts(dropna=False).to_dict(),
        "top_coefficients": coef_df.head(15),
        "predicted_label_counts": pd.Series(pred_labels, dtype="string").value_counts(dropna=False).to_dict(),
    }


def _threshold_target_reference(eval_df: pd.DataFrame) -> dict[str, Any]:
    y_true = eval_df["true_label"].astype(str).tolist()
    y_pred = eval_df["predicted_label"].astype(str).tolist()
    metrics = compute_fall_detection_metrics(
        y_true,
        y_pred,
        positive_label="fall",
        negative_label="non_fall",
    )
    return {
        "metrics": metrics,
        "rows": int(len(eval_df)),
        "label_counts": eval_df["true_label"].astype(str).value_counts(dropna=False).to_dict(),
    }


def main() -> int:
    args = parse_args()

    mobifall_path = _resolve_path(args.mobifall_path)
    sisfall_path = _resolve_path(args.sisfall_path)
    mobiact_v2_path = _resolve_path(args.mobiact_v2_path)
    out_json = _resolve_path(args.out_json)

    if not mobifall_path.exists():
        print(f"ERROR: MobiFall path not found: {mobifall_path}")
        return 1
    if not sisfall_path.exists():
        print(f"ERROR: SisFall path not found: {sisfall_path}")
        return 1
    if args.eval_source == "mobiact_v2" and not mobiact_v2_path.exists():
        print(f"ERROR: MobiAct v2 path not found: {mobiact_v2_path}")
        return 1
    if args.train_source == "mobiact_v2":  # defensive: argparse already excludes this
        print("ERROR: mobiact_v2 is eval-only and cannot be used as --train-source")
        return 1

    print("Preparing MobiFall windows...")
    mobi_windows, mobi_window_summary = _prepare_window_set(
        dataset_key="mobifall",
        path=mobifall_path,
        target_rate=args.target_rate,
        window_size=args.window_size,
        step_size=args.step_size,
    )

    print("Preparing SisFall windows...")
    sis_windows, sis_window_summary = _prepare_window_set(
        dataset_key="sisfall",
        path=sisfall_path,
        target_rate=args.target_rate,
        window_size=args.window_size,
        step_size=args.step_size,
    )

    print("Building threshold prediction tables...")
    mobi_pred_df, mobi_threshold_summary = _build_threshold_table_for_dataset(
        dataset_name="MOBIFALL",
        windows=mobi_windows,
        threshold_mode=args.threshold_mode,
        target_rate=args.target_rate,
    )
    sis_pred_df, sis_threshold_summary = _build_threshold_table_for_dataset(
        dataset_name="SISFALL",
        windows=sis_windows,
        threshold_mode=args.threshold_mode,
        target_rate=args.target_rate,
    )

    # MobiAct v2 is loaded and windowed only when actually needed as an
    # external eval target. This keeps the locked MobiFall/SisFall numbers
    # bit-identical when the script is run with default args.
    mobiact_v2_pred_df: pd.DataFrame | None = None
    mobiact_v2_window_summary: dict[str, Any] | None = None
    mobiact_v2_threshold_summary: dict[str, Any] | None = None
    if args.eval_source == "mobiact_v2":
        print("Preparing MobiAct v2 windows (external eval corpus)...")
        mobiact_v2_windows, mobiact_v2_window_summary = _prepare_window_set(
            dataset_key="mobiact_v2",
            path=mobiact_v2_path,
            target_rate=args.target_rate,
            window_size=args.window_size,
            step_size=args.step_size,
            max_files=args.mobiact_v2_max_files,
        )
        mobiact_v2_pred_df, mobiact_v2_threshold_summary = _build_threshold_table_for_dataset(
            dataset_name="MOBIACT_V2",
            windows=mobiact_v2_windows,
            threshold_mode=args.threshold_mode,
            target_rate=args.target_rate,
        )

    print("Running within-dataset fall summaries...")
    within_mobi_threshold = _within_dataset_threshold_summary(
        mobi_pred_df,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    within_sis_threshold = _within_dataset_threshold_summary(
        sis_pred_df,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    within_mobi_meta = _within_dataset_meta_summary(
        mobi_pred_df,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    within_sis_meta = _within_dataset_meta_summary(
        sis_pred_df,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    train_df, train_meta = _select_meta_train_table(
        args.train_source,
        mobifall_pred_df=mobi_pred_df,
        sisfall_pred_df=sis_pred_df,
    )
    eval_df, eval_dataset_name = _select_eval_table(
        args.eval_source,
        mobifall_pred_df=mobi_pred_df,
        sisfall_pred_df=sis_pred_df,
        mobiact_v2_pred_df=mobiact_v2_pred_df,
    )

    print(f"Running cross-dataset meta-model transfer: {args.train_source} -> {args.eval_source}")
    meta_cfg = FallMetaModelConfig(
        test_size=args.test_size,
        random_state=args.random_state,
    )
    cross_meta = _evaluate_meta_transfer(
        train_df,
        eval_df,
        config=meta_cfg,
    )

    threshold_reference = _threshold_target_reference(eval_df)

    payload = {
        "evaluation_name": "fall_cross_dataset_eval",
        "preprocessing": {
            "target_rate_hz": float(args.target_rate),
            "window_size": int(args.window_size),
            "step_size": int(args.step_size),
            "threshold_mode": args.threshold_mode,
            "test_size_within_dataset": float(args.test_size),
            "random_state": int(args.random_state),
        },
        "datasets": {
            "MOBIFALL": {
                "window_summary": mobi_window_summary,
                "threshold_summary": mobi_threshold_summary,
            },
            "SISFALL": {
                "window_summary": sis_window_summary,
                "threshold_summary": sis_threshold_summary,
            },
            **(
                {
                    "MOBIACT_V2": {
                        "window_summary": mobiact_v2_window_summary,
                        "threshold_summary": mobiact_v2_threshold_summary,
                        "role": "external_eval_only",
                    }
                }
                if mobiact_v2_pred_df is not None
                else {}
            ),
        },
        "within_dataset": {
            "MOBIFALL": {
                "threshold": within_mobi_threshold,
                "meta_model": within_mobi_meta,
            },
            "SISFALL": {
                "threshold": within_sis_threshold,
                "meta_model": within_sis_meta,
            },
        },
        "cross_dataset": {
            "train": {
                **train_meta,
                "label_counts": train_df["true_label"].astype(str).value_counts(dropna=False).to_dict(),
                "rows": int(len(train_df)),
            },
            "eval": {
                "eval_source": args.eval_source,
                "eval_dataset_name": eval_dataset_name,
                "label_counts": eval_df["true_label"].astype(str).value_counts(dropna=False).to_dict(),
                "rows": int(len(eval_df)),
            },
            "threshold_target_reference": threshold_reference,
            "meta_model_transfer": cross_meta,
        },
        "notes": [
            "Threshold target reference is source-independent; it is reported for the evaluation dataset only.",
            "Meta-model transfer trains on the full source threshold-prediction table and evaluates on the full target table.",
            "Default threshold_mode='shared' is recommended after unit harmonization so both datasets are evaluated in the same physical regime.",
        ],
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")

    print()
    print(f"Saved fall cross-dataset evaluation to: {out_json}")
    print("Within-dataset F1:")
    print(
        f"  MOBIFALL threshold={within_mobi_threshold['metrics']['f1']:.4f} "
        f"meta={within_mobi_meta['metrics']['f1']:.4f}"
    )
    print(
        f"  SISFALL threshold={within_sis_threshold['metrics']['f1']:.4f} "
        f"meta={within_sis_meta['metrics']['f1']:.4f}"
    )
    print("Cross-dataset F1:")
    print(
        f"  threshold target reference ({args.eval_source})="
        f"{threshold_reference['metrics']['f1']:.4f}"
    )
    print(
        f"  meta-model transfer {args.train_source}->{args.eval_source}="
        f"{cross_meta['metrics']['f1']:.4f}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())