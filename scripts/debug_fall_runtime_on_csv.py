#!/usr/bin/env python3
"""Export per-window fall-detector decisions for a runtime CSV."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.fall.threshold_detector import default_fall_threshold_config, detect_fall_window  # noqa: E402
from pipeline.preprocess import PreprocessConfig, append_derived_channels, resample_dataframe, window_dataframe  # noqa: E402
from pipeline.schema import COMMON_SCHEMA_COLUMNS  # noqa: E402


REQUIRED_BASE_COLUMNS = {"timestamp", "ax", "ay", "az"}
OPTIONAL_GYRO_COLUMNS = {"gx", "gy", "gz"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug fall-detector decisions per window")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--target-rate", type=float, required=True)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


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


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _load_csv_to_common_schema(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = set(df.columns)

    if set(COMMON_SCHEMA_COLUMNS).issubset(cols):
        return df.copy()

    missing = REQUIRED_BASE_COLUMNS - cols
    if missing:
        raise ValueError(
            "Input CSV must include at least timestamp, ax, ay, az columns. "
            f"Missing: {', '.join(sorted(missing))}. Columns found: {', '.join(sorted(cols))}"
        )

    defaults = {
        "dataset_name": "runtime",
        "task_type": "har",
        "subject_id": "subject_0",
        "session_id": "session_0",
        "label_raw": "unknown",
        "label_mapped": "unknown",
        "placement": None,
        "sampling_rate_hz": np.nan,
        "source_file": str(path),
    }

    out = pd.DataFrame()
    for col in COMMON_SCHEMA_COLUMNS:
        if col in df.columns:
            out[col] = df[col]
        elif col in defaults:
            out[col] = defaults[col]
        elif col in OPTIONAL_GYRO_COLUMNS:
            out[col] = np.nan
        elif col == "row_index":
            out[col] = np.arange(len(df), dtype=int)
        else:
            out[col] = np.nan

    _coerce_numeric(out, ["timestamp", "ax", "ay", "az", "gx", "gy", "gz", "sampling_rate_hz", "row_index"])
    out["label_raw"] = out["label_raw"].astype("string")
    out["label_mapped"] = out["label_mapped"].astype("string")
    return out


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


def _as_reason_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return ",".join([str(v) for v in value])
    return str(value)


def main() -> int:
    args = parse_args()
    input_path = _resolve_path(args.input)
    if not input_path.exists():
        print(f"ERROR: input CSV not found: {input_path}")
        return 1

    out_csv = _resolve_path(args.out_csv)
    out_json = _resolve_path(args.out_json)

    try:
        df = _load_csv_to_common_schema(input_path)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

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
    if not windows:
        print("ERROR: no windows generated from input")
        return 1

    detector_cfg = default_fall_threshold_config()

    records: list[dict[str, Any]] = []
    feature_keys = [
        "peak_acc",
        "mean_acc",
        "peak_minus_mean",
        "peak_over_mean_ratio",
        "post_impact_motion",
        "post_impact_variance",
        "post_impact_dyn_mean",
        "post_impact_dyn_rms",
        "post_impact_dyn_ratio_mean",
        "post_impact_dyn_ratio_rms",
        "post_impact_motion_to_peak_ratio",
        "jerk_peak",
        "jerk_rms",
        "gyro_peak",
        "impact_time_offset_s",
        "window_sampling_rate_hz",
        "g_reference",
        "acc_baseline",
    ]

    for window in windows:
        fall_result = detect_fall_window(
            window,
            config=detector_cfg,
            default_sampling_rate_hz=args.target_rate,
        )
        features = fall_result.get("features", {}) or {}
        decision = fall_result.get("decision", {}) or {}

        impact_checks = decision.get("impact_checks", {}) or {}
        support_checks = decision.get("support_checks", {}) or {}
        confirm_checks = decision.get("confirm_checks", {}) or {}

        label_majority = window.get("label_mapped_majority")
        if label_majority is None or str(label_majority).strip() == "":
            label_majority = "unknown"

        record: dict[str, Any] = {
            "window_id": int(window.get("window_id", 0)),
            "ts_start": window.get("start_ts"),
            "ts_end": window.get("end_ts"),
            "label_majority": label_majority,
            "predicted_label": decision.get("predicted_label"),
            "predicted_is_fall": decision.get("predicted_is_fall"),
            "detector_reason": decision.get("detector_reason"),
            "reasons": _as_reason_text(decision.get("reasons")),
            "stage_impact_pass": decision.get("stage_impact_pass"),
            "stage_support_pass": decision.get("stage_support_pass"),
            "stage_confirm_pass": decision.get("stage_confirm_pass"),
            "impact_check_peak_acc": impact_checks.get("impact_peak_acc"),
            "impact_check_peak_ratio": impact_checks.get("impact_peak_ratio"),
            "support_check_jerk_peak": support_checks.get("jerk_peak"),
            "support_check_gyro_peak": support_checks.get("gyro_peak"),
            "confirm_check_post_dyn_ratio_mean_max": confirm_checks.get("post_impact_dyn_ratio_mean_max"),
            "confirm_check_post_impact_dyn_mean_max": confirm_checks.get("post_impact_dyn_mean_max"),
            "confirm_check_post_impact_variance_max": confirm_checks.get("post_impact_variance_max"),
            "confirm_check_post_impact_jerk_rms_max": confirm_checks.get("post_impact_jerk_rms_max"),
            "confirm_check_post_impact_motion_ratio_max": confirm_checks.get("post_impact_motion_ratio_max"),
        }

        for key in feature_keys:
            if key in features:
                record[key] = features.get(key)
        records.append(record)

    debug_df = pd.DataFrame(records)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    debug_df.to_csv(out_csv, index=False)

    impact_threshold = float(detector_cfg.impact_peak_acc_threshold)
    peak_series = pd.to_numeric(debug_df.get("peak_acc"), errors="coerce") if not debug_df.empty else pd.Series([], dtype=float)
    impact_mask = peak_series >= impact_threshold if not peak_series.empty else pd.Series([], dtype=bool)
    label_series = debug_df.get("label_majority")
    if label_series is None:
        label_series = pd.Series([], dtype="string")
    else:
        label_series = label_series.astype("string")

    counts_by_true_label = Counter(label_series.fillna("unknown").astype(str).tolist())
    pred_series = debug_df.get("predicted_label")
    pred_series = pred_series.astype("string") if pred_series is not None else pd.Series([], dtype="string")
    counts_by_predicted_label = Counter(pred_series.fillna("unknown").astype(str).tolist())

    if not impact_mask.empty:
        counts_over_impact = int(impact_mask.sum())
        fall_over_impact = int(((label_series == "fall") & impact_mask).sum())
    else:
        counts_over_impact = 0
        fall_over_impact = 0

    top_windows = []
    if not debug_df.empty and "peak_acc" in debug_df.columns:
        ranked = debug_df.copy()
        ranked["peak_acc"] = pd.to_numeric(ranked["peak_acc"], errors="coerce")
        ranked = ranked[np.isfinite(ranked["peak_acc"])]
        ranked = ranked.sort_values("peak_acc", ascending=False).head(10)
        for _, row in ranked.iterrows():
            top_windows.append(
                {
                    "window_id": int(row["window_id"]),
                    "peak_acc": float(row["peak_acc"]),
                    "label_majority": row.get("label_majority"),
                    "ts_start": row.get("ts_start"),
                    "ts_end": row.get("ts_end"),
                }
            )

    summary = {
        "input_path": str(input_path),
        "window_params": {
            "target_rate": float(args.target_rate),
            "window_size": int(window_size),
            "step_size": int(step_size),
        },
        "fall_detector_config": asdict(detector_cfg),
        "counts_by_true_label": {str(k): int(v) for k, v in counts_by_true_label.items()},
        "counts_by_predicted_label": {str(k): int(v) for k, v in counts_by_predicted_label.items()},
        "impact_threshold": impact_threshold,
        "counts_of_windows_over_impact_threshold": counts_over_impact,
        "counts_of_fall_windows_over_threshold": fall_over_impact,
        "top_windows_by_peak_acc": top_windows,
    }

    _save_json(out_json, summary)
    print(f"Wrote debug CSV: {out_csv}")
    print(f"Wrote debug JSON: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
