#!/usr/bin/env python3
"""Run controlled threshold sweeps for Chapter 5 tuning fall-detector tuning."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.fall_threshold_tradeoffs import (  # noqa: E402
    best_config_by_f1,
    best_config_by_false_alarms_with_sensitivity_floor,
    best_config_by_sensitivity_with_specificity_floor,
    save_tradeoff_plot,
)
from metrics.fall_event_metrics import compute_event_level_metrics  # noqa: E402
from models.fall.evaluate_threshold_fall import split_fall_predictions_by_subject  # noqa: E402
from pipeline.fall.features import extract_fall_window_features  # noqa: E402
from pipeline.fall.threshold_detector import (  # noqa: E402
    FallThresholdConfig,
    default_fall_threshold_config,
)
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

WINDOW_META_COLUMNS = [
    "window_id",
    "dataset_name",
    "subject_id",
    "session_id",
    "source_file",
    "task_type",
    "start_ts",
    "end_ts",
    "n_samples",
    "is_acceptable",
    "missing_ratio",
    "has_large_gap",
    "n_gaps",
    "true_label",
]

FEATURE_COLUMNS = [
    "window_sampling_rate_hz",
    "g_reference",
    "peak_acc",
    "mean_acc",
    "acc_variance",
    "peak_minus_mean",
    "peak_over_mean_ratio",
    "impact_index",
    "impact_time_offset_s",
    "post_impact_motion",
    "post_impact_variance",
    "post_impact_samples",
    "post_impact_dyn_mean",
    "post_impact_dyn_rms",
    "post_impact_dyn_ratio_mean",
    "post_impact_dyn_ratio_rms",
    "post_impact_available",
    "jerk_peak",
    "jerk_mean",
    "jerk_rms",
    "gyro_peak",
    "gyro_mean",
]


@dataclass(frozen=True)
class VectorizedSweepData:
    y_true_fall: np.ndarray
    y_true_non_fall: np.ndarray
    y_true_known: np.ndarray
    peak_acc: np.ndarray
    peak_ratio: np.ndarray
    jerk_peak: np.ndarray
    gyro_peak: np.ndarray
    post_motion: np.ndarray
    post_dyn_ratio_mean: np.ndarray
    post_var: np.ndarray
    post_impact_available: np.ndarray
    jerk_rms: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep threshold configs for Chapter 5 fall detector")
    parser.add_argument("--dataset", required=True, choices=["mobifall", "sisfall"])
    parser.add_argument("--path", required=True, help="Dataset root path")
    parser.add_argument("--sample-limit", type=int, default=0, help="0 = full load; >0 = file/sample limit")
    parser.add_argument("--target-rate", type=float, default=50.0)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument("--keep-unacceptable", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--results-root", default="results/runs")
    parser.add_argument("--run-id", default=None, help="If omitted, timestamped ID is used; '__AUTO' suffix is replaced")
    parser.add_argument("--impact-thresholds", default="10,12,14,16,18")
    parser.add_argument("--confirm-post-dyn-ratio-mean-max", default="0.02,0.05,0.08,0.10,0.15,0.20,0.30")
    # Deprecated backward-compatible alias for older prompts/scripts.
    parser.add_argument("--confirm-post-dyn-mean-max", default=None, help=argparse.SUPPRESS)
    # Backward-compatible alias for older scripts/prompts.
    parser.add_argument("--confirm-post-motion-max", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--jerk-thresholds", default="0,30,60")
    parser.add_argument("--confirm-post-var-max", default="0.05,0.1,0.2,0.5")
    parser.add_argument("--specificity-floor", type=float, default=0.70)
    parser.add_argument("--sensitivity-floor", type=float, default=0.50)
    parser.add_argument("--event-metrics", choices=["true", "false"], default="false")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--export-best-predictions", choices=["true", "false"], default="false")
    parser.add_argument("--checkpoint-every", type=int, default=10)
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


def _parse_float_grid(value: str) -> list[float]:
    out: list[float] = []
    for part in str(value).split(","):
        text = part.strip()
        if not text:
            continue
        out.append(float(text))
    if not out:
        raise ValueError("Threshold grid must contain at least one numeric value")
    return out


def _build_run_id(dataset: str, run_id_arg: str | None) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not run_id_arg:
        return f"fall_threshold_sweep_{dataset}__{ts}"
    if run_id_arg.endswith("__AUTO"):
        return run_id_arg.replace("__AUTO", f"__{ts}")
    return run_id_arg


def _load_dataset(dataset: str, path: Path, sample_limit: int) -> pd.DataFrame:
    max_files = None if sample_limit <= 0 else int(sample_limit)
    if dataset == "mobifall":
        return load_mobifall(path, max_files=max_files)
    if dataset == "sisfall":
        return load_sisfall(path, max_files=max_files)
    raise ValueError(dataset)


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


def _build_feature_table(
    windows: list[dict[str, Any]],
    *,
    default_sampling_rate_hz: float,
    post_impact_skip_samples: int,
    keep_unacceptable: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window in windows:
        if (not keep_unacceptable) and (not bool(window.get("is_acceptable", False))):
            continue
        feats = extract_fall_window_features(
            window,
            default_sampling_rate_hz=default_sampling_rate_hz,
            post_impact_skip_samples=post_impact_skip_samples,
        )
        row = {
            "window_id": window.get("window_id"),
            "dataset_name": window.get("dataset_name"),
            "subject_id": window.get("subject_id"),
            "session_id": window.get("session_id"),
            "source_file": window.get("source_file"),
            "task_type": window.get("task_type"),
            "start_ts": window.get("start_ts"),
            "end_ts": window.get("end_ts"),
            "n_samples": int(window.get("n_samples", 0) or 0),
            "is_acceptable": bool(window.get("is_acceptable", False)),
            "missing_ratio": float(window.get("missing_ratio", 0.0) or 0.0),
            "has_large_gap": bool(window.get("has_large_gap", False)),
            "n_gaps": int(window.get("n_gaps", 0) or 0),
            "true_label": str(window.get("label_mapped_majority")),
        }
        row.update(feats)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=WINDOW_META_COLUMNS + FEATURE_COLUMNS)

    out = pd.DataFrame(rows)
    order = [c for c in (WINDOW_META_COLUMNS + FEATURE_COLUMNS) if c in out.columns]
    order.extend([c for c in out.columns if c not in order])
    return out[order]


def _normalize_label_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace(
            {
                "non-fall": "non_fall",
                "nonfall": "non_fall",
                "adl": "non_fall",
            }
        )
    )


def _numeric_array(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        return np.full(len(df), np.nan, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)


def prepare_vectorized_sweep_data(feature_df: pd.DataFrame) -> VectorizedSweepData:
    labels = _normalize_label_series(feature_df["true_label"]) if "true_label" in feature_df.columns else pd.Series([""] * len(feature_df))
    y_true_fall = labels.eq("fall").to_numpy(dtype=bool)
    y_true_non_fall = labels.eq("non_fall").to_numpy(dtype=bool)
    y_true_known = y_true_fall | y_true_non_fall
    return VectorizedSweepData(
        y_true_fall=y_true_fall,
        y_true_non_fall=y_true_non_fall,
        y_true_known=y_true_known,
        peak_acc=_numeric_array(feature_df, "peak_acc"),
        peak_ratio=_numeric_array(feature_df, "peak_over_mean_ratio"),
        jerk_peak=_numeric_array(feature_df, "jerk_peak"),
        gyro_peak=_numeric_array(feature_df, "gyro_peak"),
        post_motion=_numeric_array(feature_df, "post_impact_motion"),
        post_dyn_ratio_mean=_numeric_array(feature_df, "post_impact_dyn_ratio_mean"),
        post_var=_numeric_array(feature_df, "post_impact_variance"),
        post_impact_available=pd.Series(feature_df["post_impact_available"]).fillna(False).astype(bool).to_numpy(dtype=bool)
        if "post_impact_available" in feature_df.columns
        else np.isfinite(_numeric_array(feature_df, "post_impact_samples")) & (_numeric_array(feature_df, "post_impact_samples") > 0.0),
        jerk_rms=_numeric_array(feature_df, "jerk_rms"),
    )


def _reduce_checks(checks: list[np.ndarray], *, logic: str, default: bool, n_rows: int) -> np.ndarray:
    if not checks:
        return np.full(n_rows, default, dtype=bool)
    if str(logic).lower() == "all":
        return np.logical_and.reduce(checks)
    return np.logical_or.reduce(checks)


def _compute_metrics_from_counts(tn: int, fp: int, fn: int, tp: int) -> dict[str, float | int]:
    total = int(tn + fp + fn + tp)
    accuracy = float((tn + tp) / total) if total else 0.0
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    f1 = float((2.0 * precision * sensitivity) / (precision + sensitivity)) if (precision + sensitivity) else 0.0
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "support_total": total,
    }


def evaluate_config_vectorized(
    data: VectorizedSweepData,
    cfg: FallThresholdConfig,
) -> dict[str, Any]:
    n_rows = len(data.peak_acc)

    impact_checks = [
        np.isfinite(data.peak_acc) & (data.peak_acc >= float(cfg.impact_peak_acc_threshold)),
        np.isfinite(data.peak_ratio) & (data.peak_ratio >= float(cfg.impact_peak_ratio_threshold)),
    ]
    stage_impact_pass = np.logical_and.reduce(impact_checks)

    support_checks: list[np.ndarray] = []
    if cfg.jerk_peak_threshold is not None:
        support_checks.append(np.isfinite(data.jerk_peak) & (data.jerk_peak >= float(cfg.jerk_peak_threshold)))
    if cfg.gyro_peak_threshold is not None:
        support_checks.append(np.isfinite(data.gyro_peak) & (data.gyro_peak >= float(cfg.gyro_peak_threshold)))
    if cfg.require_support_stage:
        stage_support_pass = _reduce_checks(
            support_checks,
            logic=cfg.support_logic,
            default=True,
            n_rows=n_rows,
        )
    else:
        stage_support_pass = np.ones(n_rows, dtype=bool)

    confirm_checks: list[np.ndarray] = []
    ratio_threshold = cfg.confirm_post_dyn_ratio_mean_max
    if ratio_threshold is not None:
        has_ratio = np.isfinite(data.post_dyn_ratio_mean)
        ratio_pass = has_ratio & (data.post_dyn_ratio_mean <= float(ratio_threshold))
        if bool(cfg.confirm_requires_post_impact):
            # Require measured post-impact segment when ratio confirmation is enabled.
            ratio_check = ratio_pass & data.post_impact_available
        else:
            ratio_check = np.where(has_ratio, ratio_pass, True)
        confirm_checks.append(ratio_check.astype(bool))

    var_threshold = cfg.confirm_post_var_max
    if var_threshold is None and cfg.post_impact_variance_max is not None:
        var_threshold = float(cfg.post_impact_variance_max)
    if var_threshold is not None:
        confirm_checks.append(np.isfinite(data.post_var) & (data.post_var <= float(var_threshold)))

    if cfg.confirm_post_jerk_rms_max is not None:
        confirm_checks.append(np.isfinite(data.jerk_rms) & (data.jerk_rms <= float(cfg.confirm_post_jerk_rms_max)))

    if cfg.post_impact_motion_ratio_max is not None:
        ratio = np.full(n_rows, np.nan, dtype=float)
        valid_ratio = np.isfinite(data.post_motion) & np.isfinite(data.peak_acc) & (data.peak_acc != 0.0)
        ratio[valid_ratio] = data.post_motion[valid_ratio] / data.peak_acc[valid_ratio]
        confirm_checks.append(np.isfinite(ratio) & (ratio <= float(cfg.post_impact_motion_ratio_max)))

    if cfg.require_confirm_stage:
        if not confirm_checks:
            stage_confirm_pass = np.zeros(n_rows, dtype=bool)
        else:
            stage_confirm_pass = _reduce_checks(
                confirm_checks,
                logic=cfg.confirm_logic,
                default=False,
                n_rows=n_rows,
            )
    else:
        stage_confirm_pass = np.ones(n_rows, dtype=bool)

    y_pred_fall = stage_impact_pass & stage_support_pass & stage_confirm_pass

    known = data.y_true_known
    y_true_fall = data.y_true_fall & known
    y_true_non_fall = data.y_true_non_fall & known

    tp = int(np.sum(y_pred_fall & y_true_fall))
    fp = int(np.sum(y_pred_fall & y_true_non_fall))
    fn = int(np.sum((~y_pred_fall) & y_true_fall))
    tn = int(np.sum((~y_pred_fall) & y_true_non_fall))
    metrics = _compute_metrics_from_counts(tn, fp, fn, tp)

    return {
        "y_pred_fall": y_pred_fall,
        "stage_impact_pass": stage_impact_pass,
        "stage_support_pass": stage_support_pass,
        "stage_confirm_pass": stage_confirm_pass,
        "metrics": metrics,
        "false_alarms_count": fp,
    }


def _detector_reason_from_stages(
    stage_impact_pass: np.ndarray,
    stage_support_pass: np.ndarray,
    stage_confirm_pass: np.ndarray,
) -> np.ndarray:
    return np.where(
        ~stage_impact_pass,
        "failed_impact_stage",
        np.where(
            ~stage_support_pass,
            "failed_support_stage",
            np.where(~stage_confirm_pass, "failed_confirm_stage", "fall_detected"),
        ),
    )


def _fp_per_hour(false_alarm_count: int, n_windows: int, *, step_size_samples: int, sampling_rate_hz: float) -> float | None:
    if n_windows <= 0 or step_size_samples <= 0 or sampling_rate_hz <= 0:
        return None
    hours = (float(n_windows) * float(step_size_samples) / float(sampling_rate_hz)) / 3600.0
    if hours <= 0:
        return None
    return float(false_alarm_count / hours)


def _optional_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if value is None:
        return None
    return float(value)


def _build_cfg_from_sweep_row(base_cfg: FallThresholdConfig, row: dict[str, Any]) -> FallThresholdConfig:
    jerk_threshold = _optional_float(row.get("jerk_threshold"))
    support_stage = bool(jerk_threshold is not None and float(jerk_threshold) > 0.0)
    return replace(
        base_cfg,
        impact_peak_acc_threshold=float(row["impact_threshold"]),
        confirm_post_dyn_ratio_mean_max=_optional_float(
            row.get("confirm_post_dyn_ratio_mean_max", row.get("confirm_post_dyn_mean_max"))
        ),
        confirm_requires_post_impact=True,
        confirm_post_dyn_mean_max=None,
        confirm_post_var_max=_optional_float(row.get("confirm_post_var_max")),
        post_impact_motion_max=None,
        post_impact_variance_max=_optional_float(row.get("confirm_post_var_max")),
        require_support_stage=support_stage,
        jerk_peak_threshold=(float(jerk_threshold) if support_stage and jerk_threshold is not None else None),
    )


def main() -> int:
    args = parse_args()
    dataset_path = _resolve_path(args.path)
    if not dataset_path.exists():
        print(f"ERROR: dataset path not found: {dataset_path}")
        return 1

    impact_grid = _parse_float_grid(args.impact_thresholds)
    confirm_ratio_arg = args.confirm_post_dyn_ratio_mean_max
    if args.confirm_post_dyn_mean_max:
        print("warning: --confirm-post-dyn-mean-max is deprecated; mapping values to --confirm-post-dyn-ratio-mean-max.")
        confirm_ratio_arg = args.confirm_post_dyn_mean_max
    if args.confirm_post_motion_max:
        print("warning: --confirm-post-motion-max is deprecated for mixed-scale datasets; mapping values to dyn-ratio thresholds.")
        confirm_ratio_arg = args.confirm_post_motion_max
    post_dyn_ratio_mean_grid = _parse_float_grid(confirm_ratio_arg)
    jerk_grid = _parse_float_grid(args.jerk_thresholds)
    post_var_grid = _parse_float_grid(args.confirm_post_var_max)
    use_event_metrics = str(args.event_metrics).lower() == "true"
    export_best_predictions = str(args.export_best_predictions).lower() == "true"
    checkpoint_every = max(1, int(args.checkpoint_every))

    run_id = _build_run_id(args.dataset, args.run_id)
    results_root = _resolve_path(args.results_root)
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset} from {dataset_path}")
    df = _load_dataset(args.dataset, dataset_path, args.sample_limit)
    validation = validate_ingestion_dataframe(df)
    for err in validation.errors:
        print(f"validation error: {err}")
    for warn in validation.warnings:
        print(f"validation warning: {warn}")

    rate_summary = summarize_sampling_rate_by_group(df)
    print(
        "sampling_rate_summary:",
        f"median={rate_summary.get('median_hz')} min={rate_summary.get('min_hz')} max={rate_summary.get('max_hz')} groups={rate_summary.get('groups_checked')}",
    )

    preprocess_cfg = PreprocessConfig(target_sampling_rate_hz=args.target_rate)
    resampled = resample_dataframe(df, target_rate_hz=args.target_rate)
    resampled = append_derived_channels(resampled)

    window_size, step_size, window_note = _effective_window_sizes(
        resampled,
        preprocess_cfg,
        window_size=args.window_size,
        step_size=args.step_size,
    )
    if window_note:
        print(f"window_note: {window_note}")

    windows = window_dataframe(resampled, window_size=window_size, step_size=step_size, config=preprocess_cfg)
    base_cfg = default_fall_threshold_config(DATASET_LABELS[args.dataset])

    feature_df = _build_feature_table(
        windows,
        default_sampling_rate_hz=args.target_rate,
        post_impact_skip_samples=base_cfg.post_impact_skip_samples,
        keep_unacceptable=bool(args.keep_unacceptable),
    )
    if feature_df.empty:
        print("ERROR: feature table is empty (no windows available after filtering)")
        return 1

    train_df, test_df, split_summary = split_fall_predictions_by_subject(
        feature_df,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(
        f"fixed_split: strategy={split_summary.get('strategy')} train_rows={len(train_df)} test_rows={len(test_df)}"
    )

    vector_data = prepare_vectorized_sweep_data(test_df)
    event_df = None
    if use_event_metrics:
        event_cols = [
            c
            for c in [
                "dataset_name",
                "subject_id",
                "session_id",
                "source_file",
                "start_ts",
                "end_ts",
                "window_id",
                "true_label",
            ]
            if c in test_df.columns
        ]
        event_df = test_df[event_cols].copy()

    csv_path = run_dir / "threshold_sweep_results.csv"
    json_path = run_dir / "threshold_sweep_results.json"
    plot_path = run_dir / "threshold_sweep_tradeoff.png"
    best_predictions_csv: Path | None = None

    total_configs = len(impact_grid) * len(post_dyn_ratio_mean_grid) * len(jerk_grid) * len(post_var_grid)
    sweep_rows: list[dict[str, Any]] = []
    plot_warning = None

    def _write_payload(
        results_df: pd.DataFrame,
        *,
        status: str,
        interrupted: bool = False,
        include_plot: bool = False,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
        results_df.to_csv(csv_path, index=False)
        best_f1 = best_config_by_f1(results_df)
        best_sens_spec = best_config_by_sensitivity_with_specificity_floor(
            results_df,
            specificity_floor=args.specificity_floor,
        )
        best_low_fp = best_config_by_false_alarms_with_sensitivity_floor(
            results_df,
            sensitivity_floor=args.sensitivity_floor,
        )

        payload = {
            "sweep_name": "lesson7_threshold_sweep",
            "dataset": args.dataset,
            "dataset_path": str(dataset_path),
            "run_id": run_id,
            "results_dir": str(run_dir),
            "status": status,
            "interrupted": bool(interrupted),
            "config": {
                "sample_limit": int(args.sample_limit),
                "target_rate_hz": float(args.target_rate),
                "window_size_samples": int(window_size),
                "step_size_samples": int(step_size),
                "test_size": float(args.test_size),
                "random_state": int(args.random_state),
                "keep_unacceptable": bool(args.keep_unacceptable),
                "event_metrics": bool(use_event_metrics),
                "specificity_floor": float(args.specificity_floor),
                "sensitivity_floor": float(args.sensitivity_floor),
                "impact_thresholds": impact_grid,
                "confirm_post_dyn_ratio_mean_max": post_dyn_ratio_mean_grid,
                "confirm_post_dyn_mean_max": post_dyn_ratio_mean_grid,  # legacy alias for older notebooks/scripts
                "jerk_thresholds": jerk_grid,
                "confirm_post_var_max": post_var_grid,
                "export_best_predictions": bool(export_best_predictions),
                "checkpoint_every": int(checkpoint_every),
                "base_detector_config": asdict(base_cfg),
            },
            "ingestion_validation": {
                "is_valid": bool(validation.is_valid),
                "errors": list(validation.errors),
                "warnings": list(validation.warnings),
            },
            "preprocessing_summary": {
                "rows_loaded": int(len(df)),
                "rows_after_resampling": int(len(resampled)),
                "windows_total": int(len(windows)),
                "feature_rows": int(len(feature_df)),
                "grouped_sampling_rate_summary": rate_summary,
            },
            "split_summary": split_summary,
            "sweep_summary": {
                "configs_evaluated": int(len(results_df)),
                "best_by_f1": best_f1,
                "best_by_sensitivity_with_specificity_floor": best_sens_spec,
                "best_by_false_alarms_with_sensitivity_floor": best_low_fp,
            },
            "artifacts": {
                "results_csv": str(csv_path),
                "results_json": str(json_path),
                "tradeoff_plot_png": str(plot_path) if include_plot and plot_path.exists() else None,
                "plot_warning": plot_warning,
                "best_predictions_csv": str(best_predictions_csv) if best_predictions_csv and best_predictions_csv.exists() else None,
            },
        }
        json_path.write_text(json.dumps(_json_safe(payload), indent=2, default=str) + "\n", encoding="utf-8")
        return best_f1, best_sens_spec, best_low_fp

    interrupted = False
    best_f1 = None
    best_sens_spec = None
    best_low_fp = None

    try:
        for config_index, (impact_threshold, post_dyn_ratio_mean_max, jerk_threshold, post_var_max) in enumerate(
            itertools.product(impact_grid, post_dyn_ratio_mean_grid, jerk_grid, post_var_grid)
        ):
            cfg = replace(
                base_cfg,
                impact_peak_acc_threshold=float(impact_threshold),
                confirm_post_dyn_ratio_mean_max=float(post_dyn_ratio_mean_max),
                confirm_requires_post_impact=True,
                confirm_post_dyn_mean_max=None,
                confirm_post_var_max=float(post_var_max),
                post_impact_motion_max=None,
                post_impact_variance_max=float(post_var_max),  # keep legacy mirror for artifact compatibility
                require_support_stage=bool(float(jerk_threshold) > 0),
                jerk_peak_threshold=(float(jerk_threshold) if float(jerk_threshold) > 0 else None),
            )

            eval_out = evaluate_config_vectorized(vector_data, cfg)
            metrics = eval_out["metrics"]
            false_alarm_count = int(eval_out["false_alarms_count"])
            fp_hour = _fp_per_hour(
                false_alarm_count,
                len(test_df),
                step_size_samples=step_size,
                sampling_rate_hz=args.target_rate,
            )

            row = {
                "config_id": int(config_index),
                "impact_threshold": float(impact_threshold),
                "confirm_post_dyn_ratio_mean_max": float(post_dyn_ratio_mean_max),
                "confirm_post_dyn_mean_max": float(post_dyn_ratio_mean_max),  # legacy alias
                "confirm_post_var_max": float(post_var_max),
                "jerk_threshold": float(jerk_threshold),
                "support_stage_enabled": bool(float(jerk_threshold) > 0),
                "confirm_logic": cfg.confirm_logic,
                "accuracy": float(metrics["accuracy"]),
                "sensitivity": float(metrics["sensitivity"]),
                "specificity": float(metrics["specificity"]),
                "precision": float(metrics["precision"]),
                "f1": float(metrics["f1"]),
                "tn": int(metrics["tn"]),
                "fp": int(metrics["fp"]),
                "fn": int(metrics["fn"]),
                "tp": int(metrics["tp"]),
                "false_alarms_count": int(false_alarm_count),
                "fp_per_hour_equivalent": float(fp_hour) if fp_hour is not None else None,
            }

            if use_event_metrics and event_df is not None:
                event_df["predicted_label"] = np.where(eval_out["y_pred_fall"], "fall", "non_fall")
                event_metrics = compute_event_level_metrics(event_df)
                row.update(
                    {
                        "event_precision": float(event_metrics["event_precision"]),
                        "event_recall": float(event_metrics["event_recall"]),
                        "predicted_fall_events_count": int(event_metrics["predicted_fall_events_count"]),
                        "true_fall_events_count": int(event_metrics["true_fall_events_count"]),
                        "true_positive_events_count": int(event_metrics["true_positive_events_count"]),
                        "false_positive_events_count": int(event_metrics["false_positive_events_count"]),
                        "false_negative_events_count": int(event_metrics["false_negative_events_count"]),
                    }
                )

            sweep_rows.append(row)
            print(f"config {config_index + 1}/{total_configs} evaluated")

            if (config_index + 1) % checkpoint_every == 0:
                checkpoint_df = pd.DataFrame(sweep_rows)
                best_f1, best_sens_spec, best_low_fp = _write_payload(
                    checkpoint_df,
                    status="in_progress",
                    interrupted=False,
                    include_plot=False,
                )
                print(f"checkpoint_saved={len(checkpoint_df)}")
    except KeyboardInterrupt:
        interrupted = True
        print("interrupt_received: saving partial sweep artifacts...")

    if not sweep_rows:
        print("ERROR: no sweep results were generated")
        return 1

    results_df = pd.DataFrame(sweep_rows)

    if (best_f1 is None) or (len(results_df) % checkpoint_every != 0):
        best_f1, best_sens_spec, best_low_fp = _write_payload(
            results_df,
            status="interrupted" if interrupted else "in_progress",
            interrupted=interrupted,
            include_plot=False,
        )

    if not interrupted and export_best_predictions and best_f1:
        best_cfg = _build_cfg_from_sweep_row(base_cfg, best_f1)
        best_eval = evaluate_config_vectorized(vector_data, best_cfg)
        export_df = test_df.copy()
        export_df["predicted_label"] = np.where(best_eval["y_pred_fall"], "fall", "non_fall")
        export_df["stage_impact_pass"] = best_eval["stage_impact_pass"]
        export_df["stage_support_pass"] = best_eval["stage_support_pass"]
        export_df["stage_confirm_pass"] = best_eval["stage_confirm_pass"]
        export_df["detector_reason"] = _detector_reason_from_stages(
            best_eval["stage_impact_pass"],
            best_eval["stage_support_pass"],
            best_eval["stage_confirm_pass"],
        )
        best_predictions_csv = run_dir / "best_config_test_predictions_windows.csv"
        export_df.to_csv(best_predictions_csv, index=False)
        print(f"saved_best_predictions={best_predictions_csv}")

    if not interrupted and not args.skip_plots:
        try:
            save_tradeoff_plot(
                results_df,
                plot_path,
                x_metric="false_alarms_count",
                y_metric="sensitivity",
                title=f"Threshold Trade-off ({args.dataset})",
            )
        except Exception as exc:  # noqa: BLE001
            plot_warning = f"failed_to_plot: {type(exc).__name__}: {exc}"
            print(f"plot warning: {plot_warning}")

    best_f1, best_sens_spec, best_low_fp = _write_payload(
        results_df,
        status="interrupted" if interrupted else "completed",
        interrupted=interrupted,
        include_plot=not interrupted,
    )

    print(f"configs_evaluated={len(results_df)}")
    if best_f1:
        print(
            "best_by_f1:",
            {
                "config_id": best_f1.get("config_id"),
                "f1": best_f1.get("f1"),
                "sensitivity": best_f1.get("sensitivity"),
                "specificity": best_f1.get("specificity"),
                "false_alarms_count": best_f1.get("false_alarms_count"),
            },
        )
    if best_sens_spec:
        print(
            "best_by_sensitivity_with_specificity_floor:",
            {
                "config_id": best_sens_spec.get("config_id"),
                "sensitivity": best_sens_spec.get("sensitivity"),
                "specificity": best_sens_spec.get("specificity"),
            },
        )
    if best_low_fp:
        print(
            "best_by_false_alarms_with_sensitivity_floor:",
            {
                "config_id": best_low_fp.get("config_id"),
                "false_alarms_count": best_low_fp.get("false_alarms_count"),
                "sensitivity": best_low_fp.get("sensitivity"),
            },
        )

    print(f"saved_csv={csv_path}")
    print(f"saved_json={json_path}")
    if interrupted:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
