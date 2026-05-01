#!/usr/bin/env python3
"""Train/export a HAR artifact, reload it, and replay inference on a target dataset.

This is the bridge between:
- offline HAR evaluation
- saved model artifacts
- runtime/replay integration

Default behavior:
- train on one source (UCI HAR / PAMAP2 / WISDM / combined)
- replay on a target source
- optionally restrict both train and eval to shared labels for apples-to-apples cross-dataset replay
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.classification import compute_classification_metrics
from models.har.infer_har import predict_har_from_artifact_path
from models.har.train_har import (
    DEFAULT_HAR_ALLOWED_LABELS,
    subject_aware_group_split,
    train_and_export_har_model,
)
from pipeline.features import build_feature_table, feature_table_schema_summary
from pipeline.ingest import load_pamap2, load_uci_har, load_wisdm
from pipeline.preprocess import (
    PreprocessConfig,
    append_derived_channels,
    resample_dataframe,
    window_dataframe,
)

SOURCE_TO_DATASET_NAME = {"ucihar": "UCIHAR", "pamap2": "PAMAP2", "wisdm": "WISDM"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/export/reload/replay HAR artifact")
    parser.add_argument("--ucihar-path", default="data/raw/UCIHAR_Dataset")
    parser.add_argument("--pamap2-path", default="data/raw/PAMAP2_Dataset")
    parser.add_argument("--wisdm-path", default="data/raw/WISDM")
    parser.add_argument(
        "--wisdm-max-sessions",
        type=int,
        default=0,
        help="Optional maximum WISDM sessions to load for quick smoke tests; 0 uses all sessions.",
    )

    parser.add_argument(
        "--train-source",
        choices=["ucihar", "pamap2", "wisdm", "combined", "combined_all"],
        default="ucihar",
        help="Dataset used to train/export the HAR artifact",
    )
    parser.add_argument(
        "--eval-source",
        choices=["ucihar", "pamap2", "wisdm"],
        default="pamap2",
        help="Dataset used for replay/evaluation",
    )

    parser.add_argument(
        "--restrict-to-shared-labels",
        action="store_true",
        help=(
            "Restrict both training and evaluation to the shared label space between "
            "train and eval sources. Recommended for cross-dataset replay."
        ),
    )

    parser.add_argument(
        "--holdout-source",
        choices=["none", "ucihar", "pamap2", "wisdm"],
        default="none",
        help=(
            "Leave-one-dataset-out: when combined with --train-source combined_all, "
            "the chosen source is excluded from training and typically used as --eval-source."
        ),
    )
    parser.add_argument(
        "--eval-each-source",
        action="store_true",
        help=(
            "After training, additionally evaluate the exported artifact against each "
            "source's full feature table. Per-source metrics are embedded in the report."
        ),
    )
    parser.add_argument(
        "--artifact-version-tag",
        default=None,
        help="Free-form tag written to artifact metadata (e.g. 'v2_movement').",
    )
    parser.add_argument(
        "--internal-holdout-size",
        type=float,
        default=0.20,
        help="When using combined training with --holdout-source none, the subject-aware holdout fraction.",
    )

    parser.add_argument("--target-rate", type=float, default=50.0)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--step-size", type=int, default=64)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--rf-class-weight",
        choices=["balanced", "balanced_subsample", "none"],
        default=None,
        help=(
            "Optional RandomForest class_weight override. "
            "Omit to keep the project default."
        ),
    )
    parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=None,
        help="Optional RandomForest max_depth override.",
    )
    parser.add_argument(
        "--rf-min-samples-leaf",
        type=int,
        default=None,
        help="Optional RandomForest min_samples_leaf override.",
    )

    parser.add_argument(
        "--artifact-out",
        default="artifacts/har/har_rf_artifact.joblib",
        help="Where to save the exported HAR artifact",
    )
    parser.add_argument(
        "--report-out",
        default="results/validation/har_artifact_replay.json",
        help="Where to save replay metrics/report JSON",
    )
    parser.add_argument(
        "--predictions-out",
        default="results/validation/har_artifact_replay_predictions.csv",
        help="Where to save per-window predictions",
    )
    return parser.parse_args()


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _rf_param_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.rf_class_weight is not None:
        overrides["class_weight"] = None if args.rf_class_weight == "none" else args.rf_class_weight
    if args.rf_max_depth is not None:
        if args.rf_max_depth <= 0:
            raise ValueError("--rf-max-depth must be positive when provided")
        overrides["max_depth"] = int(args.rf_max_depth)
    if args.rf_min_samples_leaf is not None:
        if args.rf_min_samples_leaf <= 0:
            raise ValueError("--rf-min-samples-leaf must be positive when provided")
        overrides["min_samples_leaf"] = int(args.rf_min_samples_leaf)
    return overrides


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


def _prepare_feature_table(
    dataset_key: str,
    *,
    ucihar_path: Path,
    pamap2_path: Path,
    wisdm_path: Path,
    wisdm_max_sessions: int | None,
    target_rate: float,
    window_size: int,
    step_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if dataset_key == "ucihar":
        df = load_uci_har(ucihar_path)
        dataset_name = "UCIHAR"
    elif dataset_key == "pamap2":
        df = load_pamap2(pamap2_path)
        dataset_name = "PAMAP2"
    elif dataset_key == "wisdm":
        df = load_wisdm(wisdm_path, max_sessions=wisdm_max_sessions)
        dataset_name = "WISDM"
    else:
        raise ValueError(f"Unsupported dataset_key: {dataset_key}")

    resampled = resample_dataframe(df, target_rate_hz=target_rate)
    resampled = append_derived_channels(resampled)

    cfg = PreprocessConfig(target_sampling_rate_hz=target_rate)
    windows = window_dataframe(
        resampled,
        window_size=window_size,
        step_size=step_size,
        config=cfg,
    )
    feature_df = build_feature_table(
        windows,
        filter_unacceptable=True,
        default_sampling_rate_hz=target_rate,
    )
    if feature_df.empty:
        raise ValueError(f"Feature table is empty for dataset {dataset_key}")

    summary = {
        "dataset_key": dataset_key,
        "dataset_name": dataset_name,
        "rows_loaded": int(len(df)),
        "rows_after_resample": int(len(resampled)),
        "windows_total": int(len(windows)),
        "feature_rows": int(len(feature_df)),
        "feature_schema": feature_table_schema_summary(feature_df),
        "label_counts": feature_df["label_mapped_majority"].astype(str).value_counts(dropna=False).to_dict(),
    }
    return feature_df, summary


def _train_table(
    train_source: str,
    *,
    uci_df: pd.DataFrame | None,
    pamap_df: pd.DataFrame | None,
    wisdm_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if train_source == "ucihar":
        return _require_feature_table(uci_df, "ucihar").copy(), {
            "train_source": "ucihar",
            "train_dataset_names": ["UCIHAR"],
        }
    if train_source == "pamap2":
        return _require_feature_table(pamap_df, "pamap2").copy(), {
            "train_source": "pamap2",
            "train_dataset_names": ["PAMAP2"],
        }
    if train_source == "wisdm":
        return _require_feature_table(wisdm_df, "wisdm").copy(), {
            "train_source": "wisdm",
            "train_dataset_names": ["WISDM"],
        }
    if train_source == "combined":
        combined = pd.concat(
            [
                _require_feature_table(uci_df, "ucihar"),
                _require_feature_table(pamap_df, "pamap2"),
            ],
            ignore_index=True,
        )
        return combined, {
            "train_source": "combined",
            "train_dataset_names": sorted(
                combined["dataset_name"].dropna().astype(str).unique().tolist()
            ),
        }
    if train_source == "combined_all":
        combined = pd.concat(
            [
                _require_feature_table(uci_df, "ucihar"),
                _require_feature_table(pamap_df, "pamap2"),
                _require_feature_table(wisdm_df, "wisdm"),
            ],
            ignore_index=True,
        )
        return combined, {
            "train_source": "combined_all",
            "train_dataset_names": sorted(
                combined["dataset_name"].dropna().astype(str).unique().tolist()
            ),
        }
    raise ValueError(f"Unsupported train_source: {train_source}")


def _eval_table(
    eval_source: str,
    *,
    uci_df: pd.DataFrame | None,
    pamap_df: pd.DataFrame | None,
    wisdm_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, str]:
    if eval_source == "ucihar":
        return _require_feature_table(uci_df, "ucihar").copy(), "UCIHAR"
    if eval_source == "pamap2":
        return _require_feature_table(pamap_df, "pamap2").copy(), "PAMAP2"
    if eval_source == "wisdm":
        return _require_feature_table(wisdm_df, "wisdm").copy(), "WISDM"
    raise ValueError(f"Unsupported eval_source: {eval_source}")


def _require_feature_table(df: pd.DataFrame | None, dataset_key: str) -> pd.DataFrame:
    if df is None:
        raise ValueError(f"Feature table was not prepared for dataset {dataset_key}")
    return df


def _ordered_shared_labels(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> list[str]:
    train_labels = set(train_df["label_mapped_majority"].astype(str).tolist())
    eval_labels = set(eval_df["label_mapped_majority"].astype(str).tolist())
    shared = train_labels & eval_labels
    return sorted(shared)


def _restrict_to_labels(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    return df[df["label_mapped_majority"].astype(str).isin(labels)].reset_index(drop=True).copy()


def _select_prediction_export_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "window_id",
        "dataset_name",
        "subject_id",
        "session_id",
        "task_type",
        "label_mapped_majority",
        "predicted_label",
        "predicted_confidence",
    ]
    return [c for c in preferred if c in df.columns]


def _exclude_dataset(feature_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    mask = feature_df["dataset_name"].astype(str) != dataset_name
    return feature_df[mask].reset_index(drop=True).copy()


def _train_source_composition(train_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if "dataset_name" not in train_df.columns:
        return {}
    composition: dict[str, dict[str, Any]] = {}
    for name, group in train_df.groupby(train_df["dataset_name"].astype(str)):
        subjects = (
            group["subject_id"].astype(str).nunique(dropna=True)
            if "subject_id" in group.columns
            else 0
        )
        composition[name] = {
            "n_subjects": int(subjects),
            "n_windows": int(len(group)),
            "label_counts": group["label_mapped_majority"].astype(str).value_counts(dropna=False).to_dict(),
        }
    return composition


def _compute_per_dataset_feature_coverage(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, dict[str, float]]:
    if "dataset_name" not in train_df.columns or not feature_cols:
        return {}
    coverage: dict[str, dict[str, float]] = {}
    for name, group in train_df.groupby(train_df["dataset_name"].astype(str)):
        if group.empty:
            continue
        non_na = group[feature_cols].notna().mean(numeric_only=False)
        coverage[name] = {col: float(non_na.get(col, 0.0)) for col in feature_cols}
    return coverage


def _compute_per_source_metrics(
    artifact_path: Path,
    feature_tables_by_source: dict[str, pd.DataFrame],
    labels_for_metrics: list[str],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for source_name, feature_df in feature_tables_by_source.items():
        if feature_df is None or feature_df.empty:
            continue
        preds_df = predict_har_from_artifact_path(feature_df, artifact_path=artifact_path)
        y_true = feature_df["label_mapped_majority"].astype(str).tolist()
        y_pred = preds_df["predicted_label"].astype(str).tolist()
        metrics = compute_classification_metrics(y_true, y_pred, labels=labels_for_metrics)
        results[source_name] = {
            "n_rows": int(len(feature_df)),
            "label_counts": feature_df["label_mapped_majority"].astype(str).value_counts(dropna=False).to_dict(),
            "metrics": metrics,
        }
    return results


def _write_metadata_sidecar(joblib_path: Path, metadata: dict[str, Any]) -> Path:
    sidecar = joblib_path.with_name("metadata.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(_json_safe(metadata), indent=2), encoding="utf-8")
    return sidecar


def _rewrite_artifact_metadata(artifact_path: Path, extra_metadata: dict[str, Any]) -> dict[str, Any]:
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict):
        raise ValueError("Loaded HAR artifact is not a dictionary")
    current = dict(artifact.get("metadata") or {})
    current.update(extra_metadata)
    artifact["metadata"] = current
    joblib.dump(artifact, artifact_path)
    return current


def main() -> int:
    args = parse_args()

    ucihar_path = _resolve_path(args.ucihar_path)
    pamap2_path = _resolve_path(args.pamap2_path)
    wisdm_path = _resolve_path(args.wisdm_path)
    artifact_out = _resolve_path(args.artifact_out)
    report_out = _resolve_path(args.report_out)
    predictions_out = _resolve_path(args.predictions_out)
    wisdm_max_sessions = None if args.wisdm_max_sessions <= 0 else int(args.wisdm_max_sessions)
    rf_param_overrides = _rf_param_overrides(args)

    if args.holdout_source != "none" and args.train_source != "combined_all":
        raise ValueError("--holdout-source requires --train-source combined_all")

    required_sources: set[str] = {args.eval_source}
    if args.train_source == "combined":
        required_sources.update({"ucihar", "pamap2"})
    elif args.train_source == "combined_all":
        required_sources.update({"ucihar", "pamap2", "wisdm"})
    else:
        required_sources.add(args.train_source)

    if args.eval_each_source:
        required_sources.update({"ucihar", "pamap2", "wisdm"})

    uci_df: pd.DataFrame | None = None
    pamap_df: pd.DataFrame | None = None
    wisdm_df: pd.DataFrame | None = None
    dataset_summaries: dict[str, dict[str, Any]] = {}

    if "ucihar" in required_sources:
        print("Preparing UCI HAR feature table...")
        uci_df, dataset_summaries["UCIHAR"] = _prepare_feature_table(
            "ucihar",
            ucihar_path=ucihar_path,
            pamap2_path=pamap2_path,
            wisdm_path=wisdm_path,
            wisdm_max_sessions=wisdm_max_sessions,
            target_rate=args.target_rate,
            window_size=args.window_size,
            step_size=args.step_size,
        )

    if "pamap2" in required_sources:
        print("Preparing PAMAP2 feature table...")
        pamap_df, dataset_summaries["PAMAP2"] = _prepare_feature_table(
            "pamap2",
            ucihar_path=ucihar_path,
            pamap2_path=pamap2_path,
            wisdm_path=wisdm_path,
            wisdm_max_sessions=wisdm_max_sessions,
            target_rate=args.target_rate,
            window_size=args.window_size,
            step_size=args.step_size,
        )

    if "wisdm" in required_sources:
        print("Preparing WISDM feature table...")
        wisdm_df, dataset_summaries["WISDM"] = _prepare_feature_table(
            "wisdm",
            ucihar_path=ucihar_path,
            pamap2_path=pamap2_path,
            wisdm_path=wisdm_path,
            wisdm_max_sessions=wisdm_max_sessions,
            target_rate=args.target_rate,
            window_size=args.window_size,
            step_size=args.step_size,
        )

    train_df_full, train_meta = _train_table(
        args.train_source,
        uci_df=uci_df,
        pamap_df=pamap_df,
        wisdm_df=wisdm_df,
    )
    eval_df_full, eval_dataset_name = _eval_table(
        args.eval_source,
        uci_df=uci_df,
        pamap_df=pamap_df,
        wisdm_df=wisdm_df,
    )

    holdout_mode = "passthrough"
    internal_holdout_df: pd.DataFrame | None = None

    if args.holdout_source != "none":
        holdout_name = SOURCE_TO_DATASET_NAME[args.holdout_source]
        train_df_full = _exclude_dataset(train_df_full, holdout_name)
        train_meta = {
            "train_source": f"combined_all_ex_{args.holdout_source}",
            "train_dataset_names": sorted(
                train_df_full["dataset_name"].dropna().astype(str).unique().tolist()
            ),
            "holdout_source": args.holdout_source,
        }
        holdout_mode = f"lodo_{args.holdout_source}"
    elif (
        args.eval_each_source
        and args.train_source in {"combined", "combined_all"}
        and not args.restrict_to_shared_labels
    ):
        split_train_df, split_holdout_df = subject_aware_group_split(
            train_df_full,
            test_size=float(args.internal_holdout_size),
            random_state=int(args.random_state),
        )
        train_df_full = split_train_df
        eval_df_full = split_holdout_df
        eval_dataset_name = "COMBINED_HOLDOUT"
        internal_holdout_df = split_holdout_df
        holdout_mode = "internal_group_split"

    shared_labels = _ordered_shared_labels(train_df_full, eval_df_full)

    train_rows_before_filter = int(len(train_df_full))
    eval_rows_before_filter = int(len(eval_df_full))

    if args.restrict_to_shared_labels:
        if len(shared_labels) < 2:
            raise ValueError("Need at least 2 shared labels to use --restrict-to-shared-labels")
        train_df = _restrict_to_labels(train_df_full, shared_labels)
        eval_df = _restrict_to_labels(eval_df_full, shared_labels)
        labels_for_metrics = shared_labels
    else:
        train_df = train_df_full.copy()
        eval_df = eval_df_full.copy()
        labels_for_metrics = sorted(
            set(DEFAULT_HAR_ALLOWED_LABELS)
            | set(eval_df["label_mapped_majority"].astype(str).unique().tolist())
        )

    if train_df.empty:
        raise ValueError("Training feature table is empty after filtering")
    if eval_df.empty:
        raise ValueError("Evaluation feature table is empty after filtering")

    composition = _train_source_composition(train_df)

    print(f"Training/exporting HAR artifact from: {train_meta['train_source']}")
    artifact = train_and_export_har_model(
        train_df,
        output_path=artifact_out,
        random_state=args.random_state,
        rf_params=rf_param_overrides or None,
        metadata={
            "target_rate_hz": float(args.target_rate),
            "window_size": int(args.window_size),
            "step_size": int(args.step_size),
            "train_source": train_meta["train_source"],
            "train_dataset_names": train_meta["train_dataset_names"],
            "restrict_to_shared_labels": bool(args.restrict_to_shared_labels),
            "shared_labels": shared_labels,
            "holdout_mode": holdout_mode,
            "holdout_source": args.holdout_source,
            "rf_param_overrides": rf_param_overrides,
        },
    )

    feature_coverage = _compute_per_dataset_feature_coverage(train_df, artifact["feature_columns"])
    rf_params: dict[str, Any] = {}
    try:
        rf_params = {str(k): _json_safe(v) for k, v in artifact["model"].get_params().items()}
    except Exception:
        rf_params = {}

    extra_metadata: dict[str, Any] = {
        "artifact_version": args.artifact_version_tag,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "allowed_labels": list(DEFAULT_HAR_ALLOWED_LABELS),
        "train_source_composition": composition,
        "per_dataset_feature_coverage": feature_coverage,
        "training_config": {
            "target_rate_hz": float(args.target_rate),
            "window_size": int(args.window_size),
            "step_size": int(args.step_size),
            "random_state": int(args.random_state),
            "internal_holdout_size": float(args.internal_holdout_size)
            if holdout_mode == "internal_group_split"
            else None,
            "rf_params": rf_params,
            "rf_param_overrides": rf_param_overrides,
        },
        "library_versions": {
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    merged_metadata = _rewrite_artifact_metadata(artifact_out, extra_metadata)
    _write_metadata_sidecar(artifact_out, merged_metadata)

    print(f"Reloading artifact and replaying on: {args.eval_source}")
    preds_df = predict_har_from_artifact_path(eval_df, artifact_path=artifact_out)

    replay_df = eval_df.copy()
    replay_df["predicted_label"] = preds_df["predicted_label"].astype("string")
    if "predicted_confidence" in preds_df.columns:
        replay_df["predicted_confidence"] = preds_df["predicted_confidence"]
    for col in preds_df.columns:
        if col.startswith("proba_"):
            replay_df[col] = preds_df[col]

    y_true = replay_df["label_mapped_majority"].astype(str).copy()
    y_pred = replay_df["predicted_label"].astype(str).copy()

    metrics = compute_classification_metrics(
        y_true.tolist(),
        y_pred.tolist(),
        labels=labels_for_metrics,
    )

    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    export_cols = _select_prediction_export_columns(replay_df)
    replay_df[export_cols].to_csv(predictions_out, index=False)

    per_dataset_metrics: dict[str, dict[str, Any]] = {}
    if args.eval_each_source:
        feature_tables_by_source: dict[str, pd.DataFrame] = {}
        if uci_df is not None:
            feature_tables_by_source["UCIHAR"] = uci_df
        if pamap_df is not None:
            feature_tables_by_source["PAMAP2"] = pamap_df
        if wisdm_df is not None:
            feature_tables_by_source["WISDM"] = wisdm_df
        per_dataset_metrics = _compute_per_source_metrics(
            artifact_out,
            feature_tables_by_source,
            labels_for_metrics,
        )
        if internal_holdout_df is not None and "dataset_name" in internal_holdout_df.columns:
            holdout_preds = predict_har_from_artifact_path(
                internal_holdout_df, artifact_path=artifact_out
            )
            holdout_with_preds = internal_holdout_df.copy()
            holdout_with_preds["predicted_label"] = holdout_preds["predicted_label"].astype("string")
            per_dataset_metrics["_internal_holdout_by_dataset"] = {}
            for name, group in holdout_with_preds.groupby(
                holdout_with_preds["dataset_name"].astype(str)
            ):
                if group.empty:
                    continue
                group_metrics = compute_classification_metrics(
                    group["label_mapped_majority"].astype(str).tolist(),
                    group["predicted_label"].astype(str).tolist(),
                    labels=labels_for_metrics,
                )
                per_dataset_metrics["_internal_holdout_by_dataset"][name] = {
                    "n_rows": int(len(group)),
                    "label_counts": group["label_mapped_majority"].astype(str).value_counts(dropna=False).to_dict(),
                    "metrics": group_metrics,
                }

    payload = {
        "evaluation_name": "har_artifact_replay",
        "artifact_path": str(artifact_out),
        "artifact_version": args.artifact_version_tag,
        "holdout_mode": holdout_mode,
        "holdout_source": args.holdout_source,
        "predictions_path": str(predictions_out),
        "preprocessing": {
            "target_rate_hz": float(args.target_rate),
            "window_size": int(args.window_size),
            "step_size": int(args.step_size),
            "random_state": int(args.random_state),
            "wisdm_max_sessions": wisdm_max_sessions,
            "rf_param_overrides": rf_param_overrides,
        },
        "train": {
            **train_meta,
            "artifact_label_order": artifact["label_order"],
            "train_label_counts_before_filter": train_df_full["label_mapped_majority"].astype(str).value_counts(dropna=False).to_dict(),
            "train_label_counts_after_filter": train_df["label_mapped_majority"].astype(str).value_counts(dropna=False).to_dict(),
            "rows_before_filter": train_rows_before_filter,
            "rows_after_filter": int(len(train_df)),
            "source_composition": composition,
            "rf_params": rf_params,
            "rf_param_overrides": rf_param_overrides,
        },
        "eval": {
            "eval_source": args.eval_source,
            "eval_dataset_name": eval_dataset_name,
            "eval_label_counts_before_filter": eval_df_full["label_mapped_majority"].astype(str).value_counts(dropna=False).to_dict(),
            "eval_label_counts_after_filter": eval_df["label_mapped_majority"].astype(str).value_counts(dropna=False).to_dict(),
            "rows_before_filter": eval_rows_before_filter,
            "rows_after_filter": int(len(eval_df)),
            "shared_labels": shared_labels,
            "labels_used_for_metrics": labels_for_metrics,
            "restrict_to_shared_labels": bool(args.restrict_to_shared_labels),
        },
        "metrics": metrics,
        "per_dataset_metrics": per_dataset_metrics,
        "datasets": dataset_summaries,
        "notes": [
            "This script proves the exported HAR artifact can be reloaded and used for replay inference.",
            "With --restrict-to-shared-labels enabled, both training and evaluation are filtered to the shared cross-dataset label space before export/replay.",
            "With --holdout-source, the chosen dataset is excluded from training (leave-one-dataset-out).",
            "With --eval-each-source, per-source metrics against each full feature table are embedded under per_dataset_metrics.",
            "Predictions CSV contains a compact per-window output suitable for runtime/replay inspection.",
        ],
    }

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")

    print()
    print(f"Saved HAR artifact to: {artifact_out}")
    print(f"Saved replay report to: {report_out}")
    print(f"Saved replay predictions to: {predictions_out}")
    print("Replay macro-F1:")
    print(f"  {args.train_source} -> {args.eval_source}: {metrics['macro_f1']:.4f}")
    print(f"Shared labels: {shared_labels}")
    print(f"Labels used for metrics: {labels_for_metrics}")
    print(f"Restrict to shared labels: {args.restrict_to_shared_labels}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
