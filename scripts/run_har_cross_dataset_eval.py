#!/usr/bin/env python3
"""Run within-dataset and cross-dataset HAR evaluation across public HAR datasets."""

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

from metrics.classification import compute_classification_metrics
from models.har.baselines import (
    DEFAULT_HAR_LABEL_ORDER,
    get_feature_importances_dataframe,
    heuristic_har_predict,
    train_random_forest_classifier,
)
from models.har.evaluate_har import (
    run_har_baselines_on_feature_table,
    run_har_baselines_on_train_test_feature_tables,
)
from models.har.train_har import DEFAULT_HAR_ALLOWED_LABELS, filter_har_training_rows, select_feature_columns
from pipeline.features import build_feature_table, feature_table_schema_summary
from pipeline.ingest import load_mobiact_v2, load_pamap2, load_uci_har, load_wisdm
from pipeline.preprocess import (
    PreprocessConfig,
    append_derived_channels,
    resample_dataframe,
    window_dataframe,
)
from pipeline.validation import validate_ingestion_dataframe


DATASET_CANONICAL_NAMES = {
    "uci_har": "UCIHAR",
    "pamap2": "PAMAP2",
    "wisdm": "WISDM",
    "mobiact_v2": "MOBIACT_V2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HAR cross-dataset evaluation")
    parser.add_argument("--datasets", default="uci_har,pamap2,wisdm", help="Comma-separated dataset keys")
    parser.add_argument("--ucihar-path", default="data/raw/UCIHAR_Dataset")
    parser.add_argument("--pamap2-path", default="data/raw/PAMAP2_Dataset")
    parser.add_argument("--wisdm-path", default="data/raw/WISDM")
    parser.add_argument(
        "--mobiact-v2-path",
        default="data/raw/MobiAct_Dataset_v2.0/Annotated Data",
        help="Path to MobiAct v2 Annotated Data (used only when 'mobiact_v2' is in --datasets)",
    )
    parser.add_argument("--pamap2-include-optional", action="store_true")
    parser.add_argument("--target-rate", type=float, default=50.0)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--step-size", type=int, default=64)
    parser.add_argument("--keep-unacceptable", action="store_true")
    parser.add_argument("--ucihar-sample-limit", type=int, default=0, help="0 = full dataset")
    parser.add_argument("--pamap2-sample-limit", type=int, default=0, help="0 = full dataset")
    parser.add_argument("--wisdm-sample-limit", type=int, default=0, help="0 = full dataset")
    parser.add_argument("--mobiact-v2-sample-limit", type=int, default=0, help="0 = full dataset")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--out-json",
        default="results/validation/har_cross_dataset_eval.json",
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


def _parse_selected_datasets(raw_value: str) -> list[str]:
    selected: list[str] = []
    for part in str(raw_value).split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key not in DATASET_CANONICAL_NAMES:
            raise ValueError(f"Unsupported dataset key in --datasets: {key}")
        if key not in selected:
            selected.append(key)
    if len(selected) < 2:
        raise ValueError("Select at least two HAR datasets for cross-dataset evaluation")
    return selected


def _dataset_path(dataset_key: str, args: argparse.Namespace) -> Path:
    if dataset_key == "uci_har":
        return _resolve_path(args.ucihar_path)
    if dataset_key == "pamap2":
        return _resolve_path(args.pamap2_path)
    if dataset_key == "wisdm":
        return _resolve_path(args.wisdm_path)
    if dataset_key == "mobiact_v2":
        return _resolve_path(args.mobiact_v2_path)
    raise ValueError(dataset_key)


def _dataset_sample_limit(dataset_key: str, args: argparse.Namespace) -> int:
    if dataset_key == "uci_har":
        return int(args.ucihar_sample_limit)
    if dataset_key == "pamap2":
        return int(args.pamap2_sample_limit)
    if dataset_key == "wisdm":
        return int(args.wisdm_sample_limit)
    if dataset_key == "mobiact_v2":
        return int(args.mobiact_v2_sample_limit)
    raise ValueError(dataset_key)


def _load_dataset(
    *,
    dataset_key: str,
    path: Path,
    sample_limit: int,
    pamap2_include_optional: bool,
) -> pd.DataFrame:
    if dataset_key == "uci_har":
        max_windows = None if sample_limit <= 0 else int(sample_limit)
        return load_uci_har(path, max_windows_per_split=max_windows)
    if dataset_key == "pamap2":
        max_files = None if sample_limit <= 0 else int(sample_limit)
        return load_pamap2(path, max_files=max_files, include_optional=pamap2_include_optional)
    if dataset_key == "wisdm":
        max_sessions = None if sample_limit <= 0 else int(sample_limit)
        return load_wisdm(path, max_sessions=max_sessions)
    if dataset_key == "mobiact_v2":
        max_files = None if sample_limit <= 0 else int(sample_limit)
        return load_mobiact_v2(path, task="har", max_files=max_files)
    raise ValueError(f"Unsupported dataset: {dataset_key}")


def _load_wisdm_split(path: Path, *, split_name: str, sample_limit: int) -> pd.DataFrame:
    max_sessions = None if sample_limit <= 0 else int(sample_limit)
    return load_wisdm(path, split=split_name, max_sessions=max_sessions)


def _prepare_feature_table_from_df(
    *,
    dataset_key: str,
    path: Path,
    df: pd.DataFrame,
    target_rate: float,
    window_size: int,
    step_size: int,
    keep_unacceptable: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    validation = validate_ingestion_dataframe(df)

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
        filter_unacceptable=not keep_unacceptable,
        default_sampling_rate_hz=target_rate,
    )
    feature_df = filter_har_training_rows(
        feature_df,
        label_col="label_mapped_majority",
        allowed_labels=DEFAULT_HAR_ALLOWED_LABELS,
        require_acceptable=not keep_unacceptable,
    )
    if feature_df.empty:
        raise ValueError(f"Feature table is empty for dataset {dataset_key}")

    summary = {
        "dataset_key": dataset_key,
        "path": str(path),
        "rows_loaded": int(len(df)),
        "rows_after_resample": int(len(resampled)),
        "windows_total": int(len(windows)),
        "feature_rows": int(len(feature_df)),
        "validation_warnings": list(validation.warnings),
        "validation_errors": list(validation.errors),
        "feature_schema": feature_table_schema_summary(feature_df),
    }
    return feature_df, summary


def _prepare_feature_table(
    *,
    dataset_key: str,
    path: Path,
    sample_limit: int,
    pamap2_include_optional: bool,
    target_rate: float,
    window_size: int,
    step_size: int,
    keep_unacceptable: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = _load_dataset(
        dataset_key=dataset_key,
        path=path,
        sample_limit=sample_limit,
        pamap2_include_optional=pamap2_include_optional,
    )
    return _prepare_feature_table_from_df(
        dataset_key=dataset_key,
        path=path,
        df=df,
        target_rate=target_rate,
        window_size=window_size,
        step_size=step_size,
        keep_unacceptable=keep_unacceptable,
    )


def _prepare_wisdm_feature_tables(
    *,
    path: Path,
    sample_limit: int,
    target_rate: float,
    window_size: int,
    step_size: int,
    keep_unacceptable: bool,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    train_df = _load_wisdm_split(path, split_name="train", sample_limit=sample_limit)
    test_df = _load_wisdm_split(path, split_name="test", sample_limit=sample_limit)

    train_feature_df, train_summary = _prepare_feature_table_from_df(
        dataset_key="wisdm",
        path=path / "train.csv",
        df=train_df,
        target_rate=target_rate,
        window_size=window_size,
        step_size=step_size,
        keep_unacceptable=keep_unacceptable,
    )
    test_feature_df, test_summary = _prepare_feature_table_from_df(
        dataset_key="wisdm",
        path=path / "test.csv",
        df=test_df,
        target_rate=target_rate,
        window_size=window_size,
        step_size=step_size,
        keep_unacceptable=keep_unacceptable,
    )

    combined_feature_df = pd.concat([train_feature_df, test_feature_df], ignore_index=True)
    within_summary = _within_dataset_summary_from_explicit_split(
        train_df=train_feature_df,
        test_df=test_feature_df,
        dataset_name="WISDM",
        random_state=random_state,
    )
    dataset_summary = {
        "dataset_key": "wisdm",
        "path": str(path),
        "rows_loaded": int(len(train_df) + len(test_df)),
        "rows_after_resample": int(train_summary["rows_after_resample"] + test_summary["rows_after_resample"]),
        "windows_total": int(train_summary["windows_total"] + test_summary["windows_total"]),
        "feature_rows": int(len(combined_feature_df)),
        "validation_warnings": list(train_summary["validation_warnings"]) + list(test_summary["validation_warnings"]),
        "validation_errors": list(train_summary["validation_errors"]) + list(test_summary["validation_errors"]),
        "feature_schema": feature_table_schema_summary(combined_feature_df),
        "split_summaries": {
            "train": train_summary,
            "test": test_summary,
        },
        "notes": [
            "WISDM within-dataset evaluation uses the provided train/test export.",
            "This checked-in WISDM export has no subject identifiers, so within-dataset results are not subject-independent.",
        ],
    }
    return combined_feature_df, dataset_summary, within_summary


def _ordered_shared_labels(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    train_labels = set(train_df["label_mapped_majority"].astype(str).tolist())
    test_labels = set(test_df["label_mapped_majority"].astype(str).tolist())
    shared = train_labels & test_labels

    ordered = [label for label in DEFAULT_HAR_LABEL_ORDER if label in shared]
    ordered.extend(sorted(shared - set(ordered)))
    return ordered


def _filter_to_labels(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    return df[df["label_mapped_majority"].astype(str).isin(labels)].reset_index(drop=True).copy()


def _rf_cross_dataset_eval(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    labels: list[str],
    random_state: int,
) -> dict[str, Any]:
    train_use = _filter_to_labels(train_df, labels)
    test_use = _filter_to_labels(test_df, labels)

    feature_cols = select_feature_columns(train_use)
    feature_cols = [c for c in feature_cols if c in test_use.columns]
    if not feature_cols:
        raise ValueError("No overlapping feature columns for cross-dataset RF evaluation")

    X_train = train_use[feature_cols].copy()
    X_test = test_use[feature_cols].copy()
    y_train = train_use["label_mapped_majority"].astype(str).copy()
    y_test = test_use["label_mapped_majority"].astype(str).copy()

    fill_values = X_train.median(numeric_only=True).to_dict()
    X_train = X_train.fillna(fill_values).fillna(0.0)
    X_test = X_test.fillna(fill_values).fillna(0.0)

    model = train_random_forest_classifier(X_train, y_train, random_state=random_state)
    y_pred = pd.Series(model.predict(X_test), index=test_use.index, dtype="string")
    metrics = compute_classification_metrics(
        y_test.tolist(),
        y_pred.astype(str).tolist(),
        labels=labels,
    )
    importances = get_feature_importances_dataframe(model, feature_cols)

    return {
        "metrics": metrics,
        "train_rows": int(len(train_use)),
        "test_rows": int(len(test_use)),
        "train_label_counts": train_use["label_mapped_majority"].astype(str).value_counts(dropna=False).to_dict(),
        "test_label_counts": test_use["label_mapped_majority"].astype(str).value_counts(dropna=False).to_dict(),
        "feature_columns_count": int(len(feature_cols)),
        "top_feature_importances": importances.head(15),
    }


def _heuristic_cross_dataset_eval(
    *,
    test_df: pd.DataFrame,
    labels: list[str],
) -> dict[str, Any]:
    test_use = _filter_to_labels(test_df, labels)
    y_test = test_use["label_mapped_majority"].astype(str).copy()
    y_pred = heuristic_har_predict(test_use).astype(str)

    metrics = compute_classification_metrics(
        y_test.tolist(),
        y_pred.tolist(),
        labels=labels,
    )

    return {
        "metrics": metrics,
        "test_rows": int(len(test_use)),
        "test_label_counts": test_use["label_mapped_majority"].astype(str).value_counts(dropna=False).to_dict(),
    }


def _cross_dataset_block(
    *,
    source_name: str,
    source_df: pd.DataFrame,
    target_name: str,
    target_df: pd.DataFrame,
    random_state: int,
) -> dict[str, Any]:
    shared_labels = _ordered_shared_labels(source_df, target_df)
    if len(shared_labels) < 2:
        raise ValueError(
            f"Need at least 2 shared labels for cross-dataset evaluation: {source_name} -> {target_name}"
        )

    source_all = source_df["label_mapped_majority"].astype(str).value_counts(dropna=False).to_dict()
    target_all = target_df["label_mapped_majority"].astype(str).value_counts(dropna=False).to_dict()

    return {
        "source_dataset": source_name,
        "target_dataset": target_name,
        "shared_labels_used": shared_labels,
        "source_all_label_counts": source_all,
        "target_all_label_counts": target_all,
        "note": (
            "Cross-dataset HAR transfer is evaluated on shared labels only. "
            "This avoids penalizing the source-trained model for target-only classes."
        ),
        "heuristic": _heuristic_cross_dataset_eval(
            test_df=target_df,
            labels=shared_labels,
        ),
        "random_forest": _rf_cross_dataset_eval(
            train_df=source_df,
            test_df=target_df,
            labels=shared_labels,
            random_state=random_state,
        ),
    }


def _within_dataset_summary(
    *,
    feature_df: pd.DataFrame,
    dataset_name: str,
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    result = run_har_baselines_on_feature_table(
        feature_df,
        test_size=test_size,
        random_state=random_state,
    )
    return {
        "dataset_name": dataset_name,
        "label_order": result["label_order"],
        "split": result["split"],
        "heuristic": result["heuristic"]["metrics"],
        "random_forest": result["random_forest"]["metrics"],
    }


def _within_dataset_summary_from_explicit_split(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
    random_state: int,
) -> dict[str, Any]:
    result = run_har_baselines_on_train_test_feature_tables(
        train_df,
        test_df,
        random_state=random_state,
        split_extra={
            "provided_split_names": ["train", "test"],
            "split_note": (
                "WISDM is evaluated using the provided train/test export because this dataset copy has no subject identifiers."
            ),
        },
    )
    return {
        "dataset_name": dataset_name,
        "label_order": result["label_order"],
        "split": result["split"],
        "heuristic": result["heuristic"]["metrics"],
        "random_forest": result["random_forest"]["metrics"],
    }


def main() -> int:
    args = parse_args()
    selected = _parse_selected_datasets(args.datasets)
    out_json = _resolve_path(args.out_json)

    feature_tables: dict[str, pd.DataFrame] = {}
    dataset_summaries: dict[str, dict[str, Any]] = {}
    within_dataset: dict[str, dict[str, Any]] = {}

    for dataset_key in selected:
        dataset_path = _dataset_path(dataset_key, args)
        if not dataset_path.exists():
            print(f"ERROR: dataset path not found for {dataset_key}: {dataset_path}")
            return 1

        canonical_name = DATASET_CANONICAL_NAMES[dataset_key]
        sample_limit = _dataset_sample_limit(dataset_key, args)
        print(f"Preparing {canonical_name} feature table...")

        if dataset_key == "wisdm":
            feature_df, dataset_summary, within_summary = _prepare_wisdm_feature_tables(
                path=dataset_path,
                sample_limit=sample_limit,
                target_rate=args.target_rate,
                window_size=args.window_size,
                step_size=args.step_size,
                keep_unacceptable=args.keep_unacceptable,
                random_state=args.random_state,
            )
        else:
            feature_df, dataset_summary = _prepare_feature_table(
                dataset_key=dataset_key,
                path=dataset_path,
                sample_limit=sample_limit,
                pamap2_include_optional=args.pamap2_include_optional,
                target_rate=args.target_rate,
                window_size=args.window_size,
                step_size=args.step_size,
                keep_unacceptable=args.keep_unacceptable,
            )
            within_summary = _within_dataset_summary(
                feature_df=feature_df,
                dataset_name=canonical_name,
                test_size=args.test_size,
                random_state=args.random_state,
            )

        feature_tables[canonical_name] = feature_df
        dataset_summaries[canonical_name] = dataset_summary
        within_dataset[canonical_name] = within_summary

    cross_dataset: dict[str, dict[str, Any]] = {}
    for source_key in selected:
        for target_key in selected:
            if source_key == target_key:
                continue
            source_name = DATASET_CANONICAL_NAMES[source_key]
            target_name = DATASET_CANONICAL_NAMES[target_key]
            block_key = f"{source_name}_to_{target_name}"
            print(f"Running cross-dataset transfer: {block_key}")
            cross_dataset[block_key] = _cross_dataset_block(
                source_name=source_name,
                source_df=feature_tables[source_name],
                target_name=target_name,
                target_df=feature_tables[target_name],
                random_state=args.random_state,
            )

    payload = {
        "evaluation_name": "har_cross_dataset_eval",
        "selected_datasets": [DATASET_CANONICAL_NAMES[key] for key in selected],
        "preprocessing": {
            "target_rate_hz": float(args.target_rate),
            "window_size": int(args.window_size),
            "step_size": int(args.step_size),
            "keep_unacceptable": bool(args.keep_unacceptable),
            "test_size_within_dataset": float(args.test_size),
            "random_state": int(args.random_state),
        },
        "datasets": dataset_summaries,
        "within_dataset": within_dataset,
        "cross_dataset": cross_dataset,
        "notes": [
            "Cross-dataset HAR transfer uses shared labels only.",
            "Heuristic is evaluated directly on the target dataset feature table.",
            "Random Forest is trained on all source windows for the shared-label subset and tested on all target windows for the same shared-label subset.",
            "WISDM within-dataset results use the provided train/test split because the checked-in WISDM export has no subject IDs.",
        ],
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")

    print()
    print(f"Saved cross-dataset HAR evaluation to: {out_json}")
    print("Within-dataset macro-F1:")
    for dataset_name, summary in within_dataset.items():
        print(
            f"  {dataset_name} "
            f"heuristic={summary['heuristic']['macro_f1']:.4f} "
            f"rf={summary['random_forest']['macro_f1']:.4f}"
        )
    print("Cross-dataset macro-F1:")
    for block_key, summary in cross_dataset.items():
        print(
            f"  {block_key} "
            f"heuristic={summary['heuristic']['metrics']['macro_f1']:.4f} "
            f"rf={summary['random_forest']['metrics']['macro_f1']:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
