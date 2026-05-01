#!/usr/bin/env python3
"""Train/export/reload/replay the fall meta-model artifact.

This is the fall equivalent of the HAR artifact replay path.

Flow:
- build threshold prediction tables from one or both source datasets
- train the fall meta-model
- save artifact(s)
- reload the saved artifact
- replay inference on an evaluation dataset
- save predictions + report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.fall_metrics import compute_fall_detection_metrics
from models.fall.infer_fall import predict_fall_from_artifact_path
from models.fall.train_fall_meta_model import (
    FallMetaModelConfig,
    save_fall_meta_model_artifacts,
    train_fall_meta_model,
)
from pipeline.fall.threshold_detector import default_fall_threshold_config
from pipeline.ingest import load_mobifall, load_sisfall
from pipeline.preprocess import (
    PreprocessConfig,
    append_derived_channels,
    resample_dataframe,
    window_dataframe,
)
from models.fall.evaluate_threshold_fall import build_threshold_prediction_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/export/reload/replay fall artifact")
    parser.add_argument("--mobifall-path", default="data/raw/MOBIACT_Dataset/MobiFall_Dataset_v2.0")
    parser.add_argument("--sisfall-path", default="data/raw/SISFALL_Dataset/SisFall_dataset")

    parser.add_argument(
        "--train-source",
        choices=["mobifall", "sisfall", "combined"],
        default="combined",
        help="Dataset used to train/export the fall artifact",
    )
    parser.add_argument(
        "--eval-source",
        choices=["mobifall", "sisfall"],
        default="sisfall",
        help="Dataset used for replay/evaluation",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["shared", "dataset_presets"],
        default="shared",
        help="shared = one normalized threshold config; dataset_presets = use dataset-specific config",
    )

    parser.add_argument("--target-rate", type=float, default=100.0)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--step-size", type=int, default=64)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument(
        "--artifact-dir",
        default="artifacts/fall/fall_meta_artifact",
        help="Directory where the exported fall artifact bundle will be saved",
    )
    parser.add_argument(
        "--report-out",
        default="results/validation/fall_artifact_replay.json",
        help="Where to save replay report JSON",
    )
    parser.add_argument(
        "--predictions-out",
        default="results/validation/fall_artifact_replay_predictions.csv",
        help="Where to save per-window replay predictions",
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


def _load_fall_dataframe(dataset_key: str, path: Path) -> pd.DataFrame:
    if dataset_key == "mobifall":
        return load_mobifall(path)
    if dataset_key == "sisfall":
        return load_sisfall(path)
    raise ValueError(f"Unsupported dataset: {dataset_key}")


def _prepare_window_set(
    *,
    dataset_key: str,
    path: Path,
    target_rate: float,
    window_size: int,
    step_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    df = _load_fall_dataframe(dataset_key, path)
    dataset_name = (
        str(df["dataset_name"].dropna().astype(str).iloc[0])
        if "dataset_name" in df.columns and not df.empty
        else dataset_key.upper()
    )

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
        "label_counts_rows": df["label_mapped"].astype(str).value_counts(dropna=False).to_dict()
        if "label_mapped" in df.columns
        else {},
    }
    return windows, summary


def _make_detector_config(dataset_name: str, threshold_mode: str):
    if threshold_mode == "dataset_presets":
        return default_fall_threshold_config(dataset_name)
    return default_fall_threshold_config(None)


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


def _train_table(
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


def _eval_table(
    eval_source: str,
    *,
    mobifall_pred_df: pd.DataFrame,
    sisfall_pred_df: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    if eval_source == "mobifall":
        return mobifall_pred_df.copy(), "MOBIFALL"
    if eval_source == "sisfall":
        return sisfall_pred_df.copy(), "SISFALL"
    raise ValueError(f"Unsupported eval_source: {eval_source}")


def _select_prediction_export_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "window_id",
        "dataset_name",
        "subject_id",
        "session_id",
        "task_type",
        "true_label",
        "predicted_label",
        "predicted_probability",
        "predicted_is_fall",
        "probability_threshold_used",
    ]
    return [c for c in preferred if c in df.columns]


def main() -> int:
    args = parse_args()

    mobifall_path = _resolve_path(args.mobifall_path)
    sisfall_path = _resolve_path(args.sisfall_path)
    artifact_dir = _resolve_path(args.artifact_dir)
    report_out = _resolve_path(args.report_out)
    predictions_out = _resolve_path(args.predictions_out)

    if not mobifall_path.exists():
        print(f"ERROR: MobiFall path not found: {mobifall_path}")
        return 1
    if not sisfall_path.exists():
        print(f"ERROR: SisFall path not found: {sisfall_path}")
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

    train_df, train_meta = _train_table(
        args.train_source,
        mobifall_pred_df=mobi_pred_df,
        sisfall_pred_df=sis_pred_df,
    )
    eval_df, eval_dataset_name = _eval_table(
        args.eval_source,
        mobifall_pred_df=mobi_pred_df,
        sisfall_pred_df=sis_pred_df,
    )

    print(f"Training/exporting fall artifact from: {train_meta['train_source']}")
    cfg = FallMetaModelConfig(
        test_size=0.30,
        random_state=args.random_state,
    )
    train_result = train_fall_meta_model(train_df, config=cfg)
    artifact_paths = save_fall_meta_model_artifacts(train_result, output_dir=artifact_dir)
    model_artifact_path = Path(artifact_paths["model_joblib"])

    print(f"Reloading artifact and replaying on: {args.eval_source}")
    preds_df = predict_fall_from_artifact_path(eval_df, artifact_path=model_artifact_path)

    replay_df = eval_df.copy()
    replay_df["predicted_probability"] = preds_df["predicted_probability"]
    replay_df["predicted_label"] = preds_df["predicted_label"].astype("string")
    replay_df["predicted_is_fall"] = preds_df["predicted_is_fall"]
    replay_df["probability_threshold_used"] = preds_df["probability_threshold_used"]

    y_true = replay_df["true_label"].astype(str).tolist()
    y_pred = replay_df["predicted_label"].astype(str).tolist()

    metrics = compute_fall_detection_metrics(
        y_true,
        y_pred,
        positive_label="fall",
        negative_label="non_fall",
    )

    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    export_cols = _select_prediction_export_columns(replay_df)
    replay_df[export_cols].to_csv(predictions_out, index=False)

    payload = {
        "evaluation_name": "fall_artifact_replay",
        "artifact_dir": str(artifact_dir),
        "artifact_model_path": str(model_artifact_path),
        "predictions_path": str(predictions_out),
        "preprocessing": {
            "target_rate_hz": float(args.target_rate),
            "window_size": int(args.window_size),
            "step_size": int(args.step_size),
            "threshold_mode": args.threshold_mode,
            "random_state": int(args.random_state),
        },
        "train": {
            **train_meta,
            "rows": int(len(train_df)),
            "label_counts": train_df["true_label"].astype(str).value_counts(dropna=False).to_dict(),
            "selected_probability_threshold": train_result.get("selected_probability_threshold"),
            "within_train_metrics": train_result["metrics"],
        },
        "eval": {
            "eval_source": args.eval_source,
            "eval_dataset_name": eval_dataset_name,
            "rows": int(len(eval_df)),
            "label_counts": eval_df["true_label"].astype(str).value_counts(dropna=False).to_dict(),
        },
        "metrics": metrics,
        "datasets": {
            "MOBIFALL": {
                "window_summary": mobi_window_summary,
                "threshold_summary": mobi_threshold_summary,
            },
            "SISFALL": {
                "window_summary": sis_window_summary,
                "threshold_summary": sis_threshold_summary,
            },
        },
        "notes": [
            "This script proves the exported fall artifact can be reloaded and used for replay inference.",
            "Best current strategy is usually train-source=combined after cross-dataset evaluation.",
            "Replay predictions CSV is suitable for later runtime inspection and phone-data integration.",
        ],
    }

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")

    print()
    print(f"Saved fall artifact bundle to: {artifact_dir}")
    print(f"Saved replay report to: {report_out}")
    print(f"Saved replay predictions to: {predictions_out}")
    print("Replay F1:")
    print(f"  {args.train_source} -> {args.eval_source}: {metrics['f1']:.4f}")
    print(f"Probability threshold used: {train_result.get('selected_probability_threshold')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())