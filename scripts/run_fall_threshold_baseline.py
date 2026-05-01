#!/usr/bin/env python3
"""Run the Chapter 5 threshold-based fall detection baseline on MobiFall or SisFall."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.confusion_matrix_plots import plot_confusion_matrix, save_confusion_matrix_csv  # noqa: E402
from analysis.fall_false_alarms import save_false_alarm_csv  # noqa: E402
from models.fall.evaluate_threshold_fall import (  # noqa: E402
    build_threshold_prediction_table,
    evaluate_threshold_fall_predictions,
)
from pipeline.fall.threshold_detector import FallThresholdConfig, default_fall_threshold_config  # noqa: E402
from pipeline.ingest import load_mobifall, load_sisfall  # noqa: E402
from pipeline.preprocess import (  # noqa: E402
    PreprocessConfig,
    append_derived_channels,
    resample_dataframe,
    summarize_sampling_rate_by_group,
    window_dataframe,
)
from pipeline.validation import validate_ingestion_dataframe  # noqa: E402


DATASET_LABELS = {"mobifall": "MOBIFALL", "sisfall": "SISFALL"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Chapter 5 threshold fall baseline")
    parser.add_argument("--dataset", required=True, choices=["mobifall", "sisfall"])
    parser.add_argument("--path", required=True, help="Dataset root/file path")
    parser.add_argument("--sample-limit", type=int, default=0, help="0 = full load, >0 = loader-specific sample/file limit")
    parser.add_argument("--target-rate", type=float, default=50.0)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument("--keep-unacceptable", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--results-root", default="results/runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--skip-plots", action="store_true")

    # Threshold overrides (optional)
    parser.add_argument("--impact-threshold", type=float, default=None)
    parser.add_argument("--impact-ratio-threshold", type=float, default=None)
    parser.add_argument("--jerk-threshold", type=float, default=None)
    parser.add_argument("--gyro-threshold", type=float, default=None)
    parser.add_argument("--post-dyn-ratio-threshold", type=float, default=None)
    parser.add_argument("--post-motion-threshold", type=float, default=None)
    parser.add_argument("--post-var-threshold", type=float, default=None)
    parser.add_argument("--post-motion-ratio-threshold", type=float, default=None)
    parser.add_argument("--enable-support-stage", action="store_true")
    parser.add_argument("--disable-support-stage", action="store_true")
    parser.add_argument("--disable-confirm-stage", action="store_true")
    return parser.parse_args()


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p


def _json_safe(value: Any):
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


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, default=str), encoding="utf-8")


def _build_run_id(dataset: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"fall_threshold_{dataset}__{ts}"


def _load_dataset(args: argparse.Namespace, path: Path) -> pd.DataFrame:
    max_files = None if args.sample_limit <= 0 else int(args.sample_limit)
    if args.dataset == "mobifall":
        return load_mobifall(path, max_files=max_files)
    if args.dataset == "sisfall":
        return load_sisfall(path, max_files=max_files)
    raise ValueError(args.dataset)


def _effective_window_sizes(
    df: pd.DataFrame,
    cfg: PreprocessConfig,
    *,
    window_size: int | None,
    step_size: int | None,
) -> tuple[int, int, str | None]:
    if window_size is not None:
        return int(window_size), int(step_size or max(1, window_size // 2)), None

    group_cols = [c for c in ["dataset_name", "subject_id", "session_id", "source_file"] if c in df.columns]
    min_group_size = None
    if group_cols and len(df) > 0:
        try:
            min_group_size = int(df.groupby(group_cols, dropna=False, sort=False).size().min())
        except Exception:
            min_group_size = None

    if min_group_size is None or min_group_size >= cfg.window_size_samples:
        return cfg.window_size_samples, cfg.step_size_samples, None

    w = max(2, min(32, min_group_size))
    s = int(step_size or max(1, w // 2))
    note = (
        f"Using short-group fallback window_size={w}, step_size={s} because "
        f"minimum group length ({min_group_size}) is below the default {cfg.window_size_samples}."
    )
    return w, s, note


def _apply_threshold_overrides(config: FallThresholdConfig, args: argparse.Namespace) -> FallThresholdConfig:
    updates: dict[str, Any] = {}
    if args.impact_threshold is not None:
        updates["impact_peak_acc_threshold"] = float(args.impact_threshold)
    if args.impact_ratio_threshold is not None:
        updates["impact_peak_ratio_threshold"] = float(args.impact_ratio_threshold)
    if args.jerk_threshold is not None:
        updates["jerk_peak_threshold"] = float(args.jerk_threshold)
    if args.gyro_threshold is not None:
        updates["gyro_peak_threshold"] = float(args.gyro_threshold)
    if args.post_dyn_ratio_threshold is not None:
        updates["confirm_post_dyn_ratio_mean_max"] = float(args.post_dyn_ratio_threshold)
        updates["confirm_requires_post_impact"] = True
    if args.post_motion_threshold is not None:
        updates["post_impact_motion_max"] = float(args.post_motion_threshold)
    if args.post_var_threshold is not None:
        updates["post_impact_variance_max"] = float(args.post_var_threshold)
    if args.post_motion_ratio_threshold is not None:
        updates["post_impact_motion_ratio_max"] = float(args.post_motion_ratio_threshold)
    if args.enable_support_stage:
        updates["require_support_stage"] = True
    if args.disable_support_stage:
        updates["require_support_stage"] = False
    if args.disable_confirm_stage:
        updates["require_confirm_stage"] = False
    return replace(config, **updates) if updates else config


def _format_rate_summary(rate_summary: dict[str, Any]) -> str:
    groups = int(rate_summary.get("groups_checked") or 0)
    median_hz = rate_summary.get("median_hz")
    min_hz = rate_summary.get("min_hz")
    max_hz = rate_summary.get("max_hz")
    if groups <= 0 or median_hz is None:
        return "unavailable"
    return f"median={median_hz:.1f}Hz min={min_hz:.1f}Hz max={max_hz:.1f}Hz groups={groups}"


def main() -> int:
    args = parse_args()
    dataset_path = _resolve_path(args.path)
    if not dataset_path.exists():
        print(f"ERROR: dataset path not found: {dataset_path}")
        return 1

    run_id = args.run_id or _build_run_id(args.dataset)
    results_root = _resolve_path(args.results_root) if not Path(args.results_root).is_absolute() else Path(args.results_root)
    assert results_root is not None
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset} from {dataset_path}")
    df = _load_dataset(args, dataset_path)
    validation = validate_ingestion_dataframe(df)
    for err in validation.errors:
        print(f"validation error: {err}")
    for warn in validation.warnings:
        print(f"validation warning: {warn}")

    rate_summary = summarize_sampling_rate_by_group(df)
    print(f"rows_loaded={len(df)} estimated_sampling_rate_summary: {_format_rate_summary(rate_summary)}")

    resampled = resample_dataframe(df, target_rate_hz=args.target_rate)
    resampled = append_derived_channels(resampled)

    preprocess_cfg = PreprocessConfig(target_sampling_rate_hz=args.target_rate)
    window_size, step_size, window_note = _effective_window_sizes(
        resampled,
        preprocess_cfg,
        window_size=args.window_size,
        step_size=args.step_size,
    )
    if window_note:
        print(f"window_note: {window_note}")

    windows = window_dataframe(resampled, window_size=window_size, step_size=step_size, config=preprocess_cfg)
    detector_cfg = default_fall_threshold_config(DATASET_LABELS[args.dataset])
    detector_cfg = _apply_threshold_overrides(detector_cfg, args)

    pred_df = build_threshold_prediction_table(
        windows,
        detector_config=detector_cfg,
        filter_unacceptable=not args.keep_unacceptable,
        default_sampling_rate_hz=args.target_rate,
    )
    if pred_df.empty:
        print("ERROR: no prediction rows were generated (no windows or all windows filtered)")
        return 1

    eval_result = evaluate_threshold_fall_predictions(
        pred_df,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    metrics = eval_result["metrics"]
    false_alarm_df = eval_result["false_alarms"]
    test_pred_df = eval_result["test_predictions"]

    print("\nThreshold fall baseline summary")
    print(
        f"accuracy={metrics['accuracy']:.4f} sensitivity={metrics['sensitivity']:.4f} "
        f"specificity={metrics['specificity']:.4f} precision={metrics['precision']:.4f} f1={metrics['f1']:.4f}"
    )
    print(
        f"split_strategy={eval_result['split']['strategy']} train_rows={eval_result['split']['train_rows']} "
        f"test_rows={eval_result['split']['test_rows']} false_alarms={len(false_alarm_df)}"
    )

    # Save artifacts
    predictions_csv = run_dir / "predictions_windows.csv"
    test_predictions_csv = run_dir / "test_predictions_windows.csv"
    false_alarm_csv = run_dir / "false_alarms.csv"
    pred_df.to_csv(predictions_csv, index=False)
    test_pred_df.to_csv(test_predictions_csv, index=False)
    save_false_alarm_csv(false_alarm_df, false_alarm_csv)

    cm_csv = save_confusion_matrix_csv(metrics["confusion_matrix"], metrics["labels"], run_dir / "confusion_matrix_threshold_fall.csv")

    plot_warning = None
    cm_png_path = run_dir / "confusion_matrix_threshold_fall.png"
    if not args.skip_plots:
        try:
            plot_confusion_matrix(
                metrics["confusion_matrix"],
                metrics["labels"],
                title=f"Threshold Fall Confusion Matrix ({args.dataset})",
                out_path=cm_png_path,
            )
        except Exception as exc:  # noqa: BLE001
            plot_warning = f"failed to generate confusion matrix PNG: {type(exc).__name__}: {exc}"
            print(f"plot warning: {plot_warning}")

    metrics_payload = {
        "dataset": args.dataset,
        "dataset_path": str(dataset_path),
        "config": {
            "target_rate_hz": args.target_rate,
            "window_size_samples": window_size,
            "step_size_samples": step_size,
            "keep_unacceptable": bool(args.keep_unacceptable),
            "test_size": args.test_size,
            "random_state": args.random_state,
            "sample_limit": args.sample_limit,
        },
        "ingestion_validation": {
            "is_valid": bool(validation.is_valid),
            "errors": list(validation.errors),
            "warnings": list(validation.warnings),
        },
        "preprocessing_summary": {
            "rows_loaded": int(len(df)),
            "estimated_sampling_rate_hz": rate_summary.get("median_hz"),
            "grouped_sampling_rate_summary": rate_summary,
            "rows_after_resampling": int(len(resampled)),
            "windows_total": int(len(windows)),
            "prediction_rows": int(len(pred_df)),
            "test_prediction_rows": int(len(test_pred_df)),
        },
        "threshold_detector": {
            "config": asdict(detector_cfg),
        },
        "split": eval_result["split"],
        "metrics": metrics,
        "false_alarm_summary": {
            "count": int(len(false_alarm_df)),
            "by_subject": false_alarm_df["subject_id"].astype(str).value_counts().to_dict() if not false_alarm_df.empty and "subject_id" in false_alarm_df.columns else {},
            "by_session": false_alarm_df["session_id"].astype(str).value_counts().to_dict() if not false_alarm_df.empty and "session_id" in false_alarm_df.columns else {},
        },
        "artifact_status": {
            "predictions_windows_csv": str(predictions_csv),
            "test_predictions_windows_csv": str(test_predictions_csv),
            "false_alarms_csv": str(false_alarm_csv),
            "confusion_matrix_csv": str(cm_csv),
            "confusion_matrix_png": str(cm_png_path) if cm_png_path.exists() else None,
            "plot_warning": plot_warning,
        },
    }
    _save_json(run_dir / "metrics.json", metrics_payload)

    _save_json(
        run_dir / "run_summary.json",
        {
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": args.dataset,
            "dataset_path": str(dataset_path),
            "results_dir": str(run_dir),
            "metrics_file": "metrics.json",
        },
    )

    print(f"\nSaved fall threshold baseline artifacts to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
