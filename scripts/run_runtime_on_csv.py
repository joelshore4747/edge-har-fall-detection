#!/usr/bin/env python3
"""Run the Chapter 6 runtime stream on a CSV and write a JSON summary."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.har.baselines import heuristic_har_predict  # noqa: E402
from pipeline.features import build_feature_table  # noqa: E402
from pipeline.fall.threshold_detector import (  # noqa: E402
    FallThresholdConfig,
    default_fall_threshold_config,
    detect_fall_window,
)
from pipeline.preprocess import PreprocessConfig, append_derived_channels, resample_dataframe, window_dataframe  # noqa: E402
from pipeline.schema import COMMON_SCHEMA_COLUMNS  # noqa: E402


REQUIRED_BASE_COLUMNS = {"timestamp", "ax", "ay", "az"}
OPTIONAL_GYRO_COLUMNS = {"gx", "gy", "gz"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chapter 6 runtime event replay on a CSV")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument(
        "--db-url",
        default=None,
        help="Deprecated and ignored. Legacy runs/events DB logging has been removed.",
    )
    parser.add_argument("--target-rate", type=float, required=True)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--run-prefix",
        default=None,
        help="Optional prefix to auto-generate run name as <prefix>__<UTC timestamp>.",
    )
    parser.add_argument("--har-mode", choices=["heuristic", "rf"], default="heuristic")
    parser.add_argument("--har-model-path", default=None)
    parser.add_argument("--fall-mode", choices=["threshold"], default="threshold")
    # Fall-threshold overrides (optional)
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
    parser.add_argument(
        "--ts-origin",
        choices=["epoch", "now"],
        default="epoch",
        help="Deprecated and ignored. Retained only for CLI compatibility.",
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Deprecated no-op. This script now always writes JSON only.",
    )
    args = parser.parse_args()
    if not args.run_name:
        if args.run_prefix:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            args.run_name = f"{args.run_prefix}__{ts}"
        else:
            parser.error("--run-name is required unless --run-prefix is provided")
    return args


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


def _predict_har(
    feature_df: pd.DataFrame,
    *,
    mode: str,
    model_path: Path | None,
) -> pd.Series:
    if mode == "heuristic":
        return heuristic_har_predict(feature_df).astype("string")

    if model_path is None:
        raise ValueError("--har-model-path is required when --har-mode rf")

    try:
        import joblib  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("joblib is required to load HAR RF models") from exc

    obj = joblib.load(model_path)
    if isinstance(obj, dict):
        model = obj.get("model") or obj.get("rf_model")
        feature_cols = obj.get("feature_cols") or obj.get("feature_columns")
        fill_values = obj.get("imputer_fill_values") or {}
    else:
        model = obj
        feature_cols = getattr(model, "feature_names_in_", None)
        fill_values = {}

    if model is None or feature_cols is None:
        raise ValueError("HAR model payload missing model or feature columns")

    cols = [c for c in feature_cols if c in feature_df.columns]
    if not cols:
        raise ValueError("HAR model feature columns are not present in feature table")

    X = feature_df[cols].copy()
    if fill_values:
        X = X.fillna(fill_values).fillna(0.0)
    else:
        X = X.fillna(0.0)

    preds = model.predict(X)
    return pd.Series(preds, index=feature_df.index, dtype="string")


def _event_ts(window: dict[str, Any]) -> float | None:
    ts = window.get("end_ts")
    if ts is None:
        ts = window.get("start_ts")
    if ts is None:
        return None
    try:
        return float(ts)
    except Exception:
        return None


def _select_fall_feature_payload(features: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "peak_acc",
        "peak_over_mean_ratio",
        "post_impact_dyn_ratio_mean",
        "post_impact_motion_to_peak_ratio",
        "post_impact_motion",
        "post_impact_variance",
        "jerk_peak",
        "gyro_peak",
        "impact_time_offset_s",
    ]
    return {k: features.get(k) for k in keys if k in features}


def _apply_threshold_overrides(
    config: FallThresholdConfig, args: argparse.Namespace
) -> tuple[FallThresholdConfig, dict[str, Any]]:
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
    if not updates:
        return config, {}
    return replace(config, **updates), updates


def main() -> int:
    args = parse_args()
    input_path = _resolve_path(args.input)
    if not input_path.exists():
        print(f"ERROR: input CSV not found: {input_path}")
        return 1

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

    feature_df = build_feature_table(
        windows,
        filter_unacceptable=False,
        default_sampling_rate_hz=args.target_rate,
    )
    if feature_df.empty:
        print("ERROR: feature table is empty")
        return 1

    har_preds = _predict_har(feature_df, mode=args.har_mode, model_path=_resolve_path(args.har_model_path) if args.har_model_path else None)
    if "window_id" not in feature_df.columns:
        print("ERROR: feature table missing window_id")
        return 1

    har_pred_map = {int(wid): str(label) for wid, label in zip(feature_df["window_id"].tolist(), har_preds.tolist())}

    detector_cfg = default_fall_threshold_config()
    detector_cfg, threshold_updates = _apply_threshold_overrides(detector_cfg, args)

    sorted_windows = sorted(windows, key=lambda w: int(w.get("window_id", 0)))
    first_window = sorted_windows[0]
    first_window_id = int(first_window.get("window_id", 0))
    first_label = har_pred_map.get(first_window_id, "unknown")
    run_start_ts = first_window.get("start_ts") or first_window.get("end_ts") or _event_ts(first_window)
    if run_start_ts is None:
        print("ERROR: first window missing timestamps")
        return 1
    activity_ts = _event_ts(first_window) or run_start_ts

    events: list[dict[str, Any]] = [
        {
            "ts": run_start_ts,
            "event_type": "run_start",
            "label": "run_start",
            "confidence": None,
            "payload": {
                "window_id": first_window_id,
                "first_label": first_label,
            },
        },
        {
            "ts": activity_ts,
            "event_type": "activity_current",
            "label": first_label,
            "confidence": None,
            "payload": {
                "current_label": first_label,
                "window_id": first_window_id,
            },
        },
    ]
    prev_label: str | None = first_label

    for window in sorted_windows:
        window_id = int(window.get("window_id", 0))
        har_label = har_pred_map.get(window_id, "unknown")
        ts = _event_ts(window)

        if har_label != prev_label:
            events.append(
                {
                    "ts": ts,
                    "event_type": "activity_change",
                    "label": har_label,
                    "confidence": None,
                    "payload": {
                        "previous_label": prev_label,
                        "current_label": har_label,
                        "window_id": window_id,
                    },
                }
            )

        fall_result = detect_fall_window(
            window,
            config=detector_cfg,
            default_sampling_rate_hz=args.target_rate,
        )
        decision = fall_result.get("decision", {})
        if decision.get("predicted_is_fall"):
            features = fall_result.get("features", {})
            payload = {
                "detector_reason": decision.get("detector_reason"),
                "stage_impact_pass": decision.get("stage_impact_pass"),
                "stage_support_pass": decision.get("stage_support_pass"),
                "stage_confirm_pass": decision.get("stage_confirm_pass"),
                "features": _select_fall_feature_payload(features),
                "window_id": window_id,
            }
            events.append(
                {
                    "ts": ts,
                    "event_type": "fall_detected",
                    "label": "fall",
                    "confidence": None,
                    "payload": payload,
                }
            )

        prev_label = har_label

    # Build summary JSON
    event_type_counts = Counter([str(e["event_type"]) for e in events])
    label_counts = Counter([str(e["label"]) for e in events])

    ts_values = [e.get("ts") for e in events if e.get("ts") is not None]
    ts_min = float(min(ts_values)) if ts_values else None
    ts_max = float(max(ts_values)) if ts_values else None

    events_preview = events[:5]

    summary = {
        "run_name": args.run_name,
        "run_id": None,
        "input_path": str(input_path),
        "window_params": {
            "target_rate": float(args.target_rate),
            "window_size": int(window_size),
            "step_size": int(step_size),
        },
        "fall_detector_config": asdict(detector_cfg),
        "fall_threshold_overrides": threshold_updates,
        "counts_by_event_type": {str(k): int(v) for k, v in event_type_counts.items()},
        "counts_by_label": {str(k): int(v) for k, v in label_counts.items()},
        "ts_min": ts_min,
        "ts_max": ts_max,
        "events_preview": events_preview,
    }

    _save_json(out_json, summary)
    print(f"Wrote runtime summary: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
