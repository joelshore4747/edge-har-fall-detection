#!/usr/bin/env python3
"""Run combined HAR + fall artifact replay on one normalized sensor stream.

Supports:
- built-in dataset sources: ucihar, pamap2, mobifall, sisfall
- raw normalized CSV via runtime_phone_csv adapter
- multi-file phone-export folder via runtime_phone_folder adapter

Runs two branches on the same source stream:
- HAR: resample -> window -> feature table -> HAR artifact inference
- Fall: resample -> window -> threshold feature table -> fall artifact inference

Outputs:
- HAR replay CSV
- Fall replay CSV
- merged combined timeline CSV
- JSON report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.classification import compute_classification_metrics
from metrics.fall_metrics import compute_fall_detection_metrics
from models.fall.evaluate_threshold_fall import build_threshold_prediction_table
from models.fall.infer_fall import load_fall_model_artifact, predict_fall_from_artifact_path
from models.har.infer_har import predict_har_from_artifact_path
from models.har.train_har import load_har_model_artifact
from pipeline.fall.threshold_detector import default_fall_threshold_config
from pipeline.features import build_feature_table
from pipeline.ingest import load_mobifall, load_pamap2, load_sisfall, load_uci_har
from pipeline.ingest.runtime_phone_csv import RuntimePhoneCsvConfig, load_runtime_phone_csv
from pipeline.ingest.runtime_phone_folder import RuntimePhoneFolderConfig, load_runtime_phone_folder
from pipeline.preprocess import (
    PreprocessConfig,
    append_derived_channels,
    resample_dataframe,
    window_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run combined HAR + fall artifact replay")

    parser.add_argument(
        "--input-source",
        choices=["csv", "phone_folder", "ucihar", "pamap2", "mobifall", "sisfall"],
        required=True,
        help="Replay source type",
    )
    parser.add_argument("--input-path", required=True, help="Path to CSV, folder, or dataset root")

    parser.add_argument(
        "--har-artifact",
        default="artifacts/har/har_rf_ucihar.joblib",
        help="Path to exported HAR artifact",
    )
    parser.add_argument(
        "--fall-artifact",
        default="artifacts/fall/fall_meta_combined/model.joblib",
        help="Path to exported fall artifact model.joblib",
    )

    parser.add_argument(
        "--threshold-mode",
        choices=["shared", "dataset_presets"],
        default="shared",
        help="Threshold config mode for fall threshold table generation",
    )

    parser.add_argument("--session-id", default=None, help="Restrict replay to one session_id if present")
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=1,
        help="If session-id omitted, keep at most this many sessions",
    )

    # CSV adapter defaults
    parser.add_argument("--csv-task-type", default="runtime")
    parser.add_argument("--csv-dataset-name", default="PHONE_RUNTIME_CSV")
    parser.add_argument("--csv-subject-id", default="phone_subject")
    parser.add_argument("--csv-session-id", default="phone_session")
    parser.add_argument("--csv-placement", default="pocket")
    parser.add_argument("--csv-sampling-rate-hz", type=float, default=100.0)

    # phone folder defaults
    parser.add_argument("--phone-folder-task-type", default="runtime")
    parser.add_argument("--phone-folder-dataset-name", default="PHONE_RUNTIME_FOLDER")
    parser.add_argument("--phone-folder-subject-id", default="phone_subject")
    parser.add_argument("--phone-folder-session-id", default=None)
    parser.add_argument("--phone-folder-placement", default="pocket")
    parser.add_argument("--phone-folder-merge-tolerance-seconds", type=float, default=0.02)

    # HAR branch params
    parser.add_argument("--har-target-rate", type=float, default=None)
    parser.add_argument("--har-window-size", type=int, default=None)
    parser.add_argument("--har-step-size", type=int, default=None)

    # Fall branch params
    parser.add_argument("--fall-target-rate", type=float, default=100.0)
    parser.add_argument("--fall-window-size", type=int, default=128)
    parser.add_argument("--fall-step-size", type=int, default=64)

    parser.add_argument("--timeline-tolerance-seconds", type=float, default=1.0)

    parser.add_argument("--har-out", default="results/validation/combined_runtime_replay_har.csv")
    parser.add_argument("--fall-out", default="results/validation/combined_runtime_replay_fall.csv")
    parser.add_argument("--combined-out", default="results/validation/combined_runtime_replay_timeline.csv")
    parser.add_argument("--report-out", default="results/validation/combined_runtime_replay_report.json")

    return parser.parse_args()


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path

def _prediction_export_columns_har(df: pd.DataFrame) -> list[str]:
    preferred = [
        "window_id",
        "dataset_name",
        "subject_id",
        "session_id",
        "start_ts",
        "end_ts",
        "midpoint_ts",
        "label_mapped_majority",
        "predicted_label",
        "predicted_confidence",
    ]
    return [c for c in preferred if c in df.columns]


def _prediction_export_columns_fall(df: pd.DataFrame) -> list[str]:
    # Export the full fall threshold-feature table so phone adaptation has all engineered features.
    return df.columns.tolist()

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


def _load_source_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    input_path = _resolve_path(args.input_path)

    if args.input_source == "csv":
        cfg = RuntimePhoneCsvConfig(
            dataset_name=args.csv_dataset_name,
            task_type=args.csv_task_type,
            subject_id=args.csv_subject_id,
            session_id=args.csv_session_id,
            placement=args.csv_placement,
            sampling_rate_hz=float(args.csv_sampling_rate_hz),
        )
        return load_runtime_phone_csv(input_path, config=cfg)

    if args.input_source == "phone_folder":
        cfg = RuntimePhoneFolderConfig(
            dataset_name=args.phone_folder_dataset_name,
            task_type=args.phone_folder_task_type,
            subject_id=args.phone_folder_subject_id,
            session_id=args.phone_folder_session_id,
            placement=args.phone_folder_placement,
            merge_tolerance_seconds=float(args.phone_folder_merge_tolerance_seconds),
        )
        return load_runtime_phone_folder(input_path, config=cfg)

    if args.input_source == "ucihar":
        return load_uci_har(input_path)
    if args.input_source == "pamap2":
        return load_pamap2(input_path)
    if args.input_source == "mobifall":
        return load_mobifall(input_path)
    if args.input_source == "sisfall":
        return load_sisfall(input_path)

    raise ValueError(f"Unsupported input source: {args.input_source}")


def _restrict_sessions(df: pd.DataFrame, *, session_id: str | None, max_sessions: int) -> pd.DataFrame:
    if "session_id" not in df.columns:
        return df.copy()

    working = df.copy()
    working["session_id"] = working["session_id"].astype(str)

    if session_id:
        out = working[working["session_id"] == str(session_id)].copy()
        if out.empty:
            raise ValueError(f"Requested session_id not found: {session_id}")
        return out.reset_index(drop=True)

    if max_sessions <= 0:
        return working.reset_index(drop=True)

    keep_sessions = sorted(working["session_id"].dropna().astype(str).unique().tolist())[:max_sessions]
    out = working[working["session_id"].astype(str).isin(keep_sessions)].copy()
    return out.reset_index(drop=True)


def _make_detector_config(dataset_name: str | None, threshold_mode: str):
    if threshold_mode == "dataset_presets":
        return default_fall_threshold_config(dataset_name)
    return default_fall_threshold_config(None)


def _artifact_har_preprocess(har_artifact_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    artifact = load_har_model_artifact(har_artifact_path)
    meta = dict(artifact.get("metadata", {}))
    return {
        "target_rate_hz": float(args.har_target_rate if args.har_target_rate is not None else meta.get("target_rate_hz", 50.0)),
        "window_size": int(args.har_window-size if False else (args.har_window_size if args.har_window_size is not None else meta.get("window_size", 128))),
        "step_size": int(args.har_step_size if args.har_step_size is not None else meta.get("step_size", 64)),
        "artifact": artifact,
    }


def _has_ground_truth_labels(df: pd.DataFrame) -> bool:
    if "label_mapped" in df.columns and df["label_mapped"].dropna().astype(str).str.strip().ne("").any():
        return True
    if "label_raw" in df.columns and df["label_raw"].dropna().astype(str).str.strip().ne("").any():
        return True
    return False


def _prepare_har_branch(
    source_df: pd.DataFrame,
    *,
    har_artifact_path: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    har_cfg = _artifact_har_preprocess(har_artifact_path, args)

    resampled = resample_dataframe(source_df, target_rate_hz=har_cfg["target_rate_hz"])
    resampled = append_derived_channels(resampled)

    preprocess_cfg = PreprocessConfig(target_sampling_rate_hz=har_cfg["target_rate_hz"])
    windows = window_dataframe(
        resampled,
        window_size=har_cfg["window_size"],
        step_size=har_cfg["step_size"],
        config=preprocess_cfg,
    )

    has_labels = _has_ground_truth_labels(source_df)
    feature_df = build_feature_table(
        windows,
        filter_unacceptable=has_labels,
        default_sampling_rate_hz=har_cfg["target_rate_hz"],
    )
    if feature_df.empty:
        raise ValueError("HAR feature table is empty")

    preds = predict_har_from_artifact_path(feature_df, artifact_path=har_artifact_path)
    out = feature_df.copy()
    for col in preds.columns:
        out[col] = preds[col]

    if "start_ts" in out.columns and "end_ts" in out.columns:
        out["midpoint_ts"] = (
            pd.to_numeric(out["start_ts"], errors="coerce")
            + pd.to_numeric(out["end_ts"], errors="coerce")
        ) / 2.0
    else:
        out["midpoint_ts"] = np.arange(len(out), dtype=float)

    summary = {
        "target_rate_hz": har_cfg["target_rate_hz"],
        "window_size": har_cfg["window_size"],
        "step_size": har_cfg["step_size"],
        "rows_after_resample": int(len(resampled)),
        "windows_total": int(len(windows)),
        "prediction_rows": int(len(out)),
        "used_label_based_filtering": bool(has_labels),
        "predicted_label_counts": out["predicted_label"].astype(str).value_counts(dropna=False).to_dict(),
    }

    if "label_mapped_majority" in out.columns:
        try:
            labels_used = sorted(set(out["label_mapped_majority"].astype(str)) | set(out["predicted_label"].astype(str)))
            summary["metrics"] = compute_classification_metrics(
                out["label_mapped_majority"].astype(str).tolist(),
                out["predicted_label"].astype(str).tolist(),
                labels=labels_used,
            )
        except Exception as exc:
            summary["metrics_error"] = str(exc)

    return out, summary


def _prepare_fall_branch(
    source_df: pd.DataFrame,
    *,
    fall_artifact_path: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    artifact = load_fall_model_artifact(fall_artifact_path)

    resampled = resample_dataframe(source_df, target_rate_hz=args.fall_target_rate)
    resampled = append_derived_channels(resampled)

    preprocess_cfg = PreprocessConfig(target_sampling_rate_hz=args.fall_target_rate)
    windows = window_dataframe(
        resampled,
        window_size=args.fall_window_size,
        step_size=args.fall_step_size,
        config=preprocess_cfg,
    )

    dataset_name = None
    if "dataset_name" in source_df.columns and not source_df["dataset_name"].dropna().empty:
        dataset_name = str(source_df["dataset_name"].dropna().astype(str).iloc[0])

    detector_config = _make_detector_config(dataset_name, args.threshold_mode)
    has_labels = _has_ground_truth_labels(source_df)

    threshold_df = build_threshold_prediction_table(
        windows,
        detector_config=detector_config,
        filter_unacceptable=has_labels,
        default_sampling_rate_hz=args.fall_target_rate,
    )
    if threshold_df.empty:
        raise ValueError("Fall threshold prediction table is empty")

    preds = predict_fall_from_artifact_path(threshold_df, artifact_path=fall_artifact_path)
    out = threshold_df.copy()
    for col in preds.columns:
        out[col] = preds[col]

    if "start_ts" in out.columns and "end_ts" in out.columns:
        out["midpoint_ts"] = (
            pd.to_numeric(out["start_ts"], errors="coerce")
            + pd.to_numeric(out["end_ts"], errors="coerce")
        ) / 2.0
    else:
        out["midpoint_ts"] = np.arange(len(out), dtype=float)

    summary = {
        "target_rate_hz": float(args.fall_target_rate),
        "window_size": int(args.fall_window_size),
        "step_size": int(args.fall_step_size),
        "rows_after_resample": int(len(resampled)),
        "windows_total": int(len(windows)),
        "prediction_rows": int(len(out)),
        "used_label_based_filtering": bool(has_labels),
        "predicted_label_counts": out["predicted_label"].astype(str).value_counts(dropna=False).to_dict(),
        "probability_threshold_used": float(artifact["probability_threshold"]),
    }

    if "true_label" in out.columns:
        try:
            summary["metrics"] = compute_fall_detection_metrics(
                out["true_label"].astype(str).tolist(),
                out["predicted_label"].astype(str).tolist(),
                positive_label=str(artifact["positive_label"]),
                negative_label=str(artifact["negative_label"]),
            )
        except Exception as exc:
            summary["metrics_error"] = str(exc)

    return out, summary


def _timeline_subset_har(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "dataset_name",
        "subject_id",
        "session_id",
        "midpoint_ts",
        "window_id",
        "label_mapped_majority",
        "predicted_label",
        "predicted_confidence",
    ]
    out = df[[c for c in keep if c in df.columns]].copy()
    return out.rename(
        columns={
            "window_id": "har_window_id",
            "label_mapped_majority": "har_true_label",
            "predicted_label": "har_predicted_label",
            "predicted_confidence": "har_predicted_confidence",
        }
    )


def _timeline_subset_fall(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "dataset_name",
        "subject_id",
        "session_id",
        "midpoint_ts",
        "window_id",
        "true_label",
        "predicted_label",
        "predicted_probability",
        "predicted_is_fall",
    ]
    out = df[[c for c in keep if c in df.columns]].copy()
    return out.rename(
        columns={
            "window_id": "fall_window_id",
            "true_label": "fall_true_label",
            "predicted_label": "fall_predicted_label",
            "predicted_probability": "fall_predicted_probability",
            "predicted_is_fall": "fall_predicted_is_fall",
        }
    )


def _merge_timelines_by_session(
    har_df: pd.DataFrame,
    fall_df: pd.DataFrame,
    *,
    tolerance_seconds: float,
) -> pd.DataFrame:
    har_tl = _timeline_subset_har(har_df)
    fall_tl = _timeline_subset_fall(fall_df)

    if har_tl.empty and fall_tl.empty:
        return pd.DataFrame()
    if har_tl.empty:
        return fall_tl.copy()
    if fall_tl.empty:
        return har_tl.copy()

    if "session_id" not in har_tl.columns:
        har_tl["session_id"] = "unknown_session"
    if "session_id" not in fall_tl.columns:
        fall_tl["session_id"] = "unknown_session"

    session_ids = sorted(set(har_tl["session_id"].astype(str)) | set(fall_tl["session_id"].astype(str)))
    merged_parts: list[pd.DataFrame] = []

    for sid in session_ids:
        h = har_tl[har_tl["session_id"].astype(str) == sid].copy()
        f = fall_tl[fall_tl["session_id"].astype(str) == sid].copy()

        if h.empty:
            merged_parts.append(f)
            continue
        if f.empty:
            merged_parts.append(h)
            continue

        h = h.sort_values("midpoint_ts").reset_index(drop=True)
        f = f.sort_values("midpoint_ts").reset_index(drop=True)

        use_group_keys = {"dataset_name", "subject_id", "session_id"} <= (set(h.columns) & set(f.columns))
        merged = pd.merge_asof(
            h,
            f,
            on="midpoint_ts",
            by=["dataset_name", "subject_id", "session_id"] if use_group_keys else "session_id",
            direction="nearest",
            tolerance=float(tolerance_seconds),
        )
        merged_parts.append(merged)

    out = pd.concat(merged_parts, ignore_index=True, sort=False)
    out = out.sort_values(["session_id", "midpoint_ts"], kind="stable").reset_index(drop=True)
    return out


def _prediction_export_columns_har(df: pd.DataFrame) -> list[str]:
    preferred = [
        "window_id",
        "dataset_name",
        "subject_id",
        "session_id",
        "start_ts",
        "end_ts",
        "midpoint_ts",
        "label_mapped_majority",
        "predicted_label",
        "predicted_confidence",
    ]
    return [c for c in preferred if c in df.columns]


def _prediction_export_columns_fall(df: pd.DataFrame) -> list[str]:
    # Export the full fall table so phone adaptation has all engineered features.
    return df.columns.tolist()


def main() -> int:
    args = parse_args()

    input_path = _resolve_path(args.input_path)
    har_artifact_path = _resolve_path(args.har_artifact)
    fall_artifact_path = _resolve_path(args.fall_artifact)

    har_out = _resolve_path(args.har_out)
    fall_out = _resolve_path(args.fall_out)
    combined_out = _resolve_path(args.combined_out)
    report_out = _resolve_path(args.report_out)

    if not input_path.exists():
        print(f"ERROR: input path not found: {input_path}")
        return 1
    if not har_artifact_path.exists():
        print(f"ERROR: HAR artifact not found: {har_artifact_path}")
        return 1
    if not fall_artifact_path.exists():
        print(f"ERROR: fall artifact not found: {fall_artifact_path}")
        return 1

    print("Loading replay input...")
    source_df = _load_source_dataframe(args)
    source_df = _restrict_sessions(source_df, session_id=args.session_id, max_sessions=args.max_sessions)

    print("Running HAR branch...")
    har_df, har_summary = _prepare_har_branch(
        source_df,
        har_artifact_path=har_artifact_path,
        args=args,
    )

    print("Running fall branch...")
    fall_df, fall_summary = _prepare_fall_branch(
        source_df,
        fall_artifact_path=fall_artifact_path,
        args=args,
    )

    print("Merging combined timeline...")
    combined_df = _merge_timelines_by_session(
        har_df,
        fall_df,
        tolerance_seconds=args.timeline_tolerance_seconds,
    )

    har_out.parent.mkdir(parents=True, exist_ok=True)
    fall_out.parent.mkdir(parents=True, exist_ok=True)
    combined_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    har_df[_prediction_export_columns_har(har_df)].to_csv(har_out, index=False)
    fall_df[_prediction_export_columns_fall(fall_df)].to_csv(fall_out, index=False)
    combined_df.to_csv(combined_out, index=False)

    report = {
        "evaluation_name": "combined_runtime_replay",
        "input": {
            "input_source": args.input_source,
            "input_path": str(input_path),
            "rows_loaded_after_session_filter": int(len(source_df)),
            "dataset_name_counts": source_df["dataset_name"].astype(str).value_counts(dropna=False).to_dict()
            if "dataset_name" in source_df.columns else {},
            "session_counts": source_df["session_id"].astype(str).value_counts(dropna=False).to_dict()
            if "session_id" in source_df.columns else {},
        },
        "artifacts": {
            "har_artifact": str(har_artifact_path),
            "fall_artifact": str(fall_artifact_path),
        },
        "har_branch": har_summary,
        "fall_branch": fall_summary,
        "outputs": {
            "har_csv": str(har_out),
            "fall_csv": str(fall_out),
            "combined_csv": str(combined_out),
            "report_json": str(report_out),
        },
        "merge": {
            "timeline_tolerance_seconds": float(args.timeline_tolerance_seconds),
            "combined_rows": int(len(combined_df)),
        },
        "notes": [
            "HAR and fall are replayed as separate branches because they use different artifacts and feature paths.",
            "input_source=phone_folder supports exports with Accelerometer.csv and Gyroscope.csv.",
            "For unlabeled runtime inputs, label-based window filtering is disabled automatically.",
            "Fall CSV export includes the full threshold-feature table for later phone adaptation work.",
        ],
    }

    report_out.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")

    print()
    print(f"Saved HAR replay CSV to: {har_out}")
    print(f"Saved fall replay CSV to: {fall_out}")
    print(f"Saved combined timeline CSV to: {combined_out}")
    print(f"Saved replay report to: {report_out}")

    if "metrics" in har_summary:
        print(f"HAR macro-F1: {har_summary['metrics']['macro_f1']:.4f}")
    if "metrics" in fall_summary:
        print(f"Fall F1: {fall_summary['metrics']['f1']:.4f}")
    print(f"Combined timeline rows: {len(combined_df)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())