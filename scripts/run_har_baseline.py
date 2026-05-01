from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.har.train_har import filter_har_training_rows, DEFAULT_HAR_ALLOWED_LABELS
from analysis.confusion_matrix_plots import plot_confusion_matrix, save_confusion_matrix_csv
from models.har.evaluate_har import (
    run_har_baselines_on_feature_table,
    run_har_baselines_on_train_test_feature_tables,
)
from pipeline.features import build_feature_table, feature_table_schema_summary
from pipeline.ingest import load_pamap2, load_uci_har, load_wisdm
from pipeline.preprocess import (
    PreprocessConfig,
    append_derived_channels,
    resample_dataframe,
    window_dataframe, summarize_sampling_rate_by_group,
)
from pipeline.validation import validate_ingestion_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chapter 4 HAR baselines on a dataset")
    parser.add_argument("--dataset", required=True, choices=["uci_har", "pamap2", "wisdm"])
    parser.add_argument("--path", required=True, help="Dataset root/file path")
    parser.add_argument("--sample-limit", type=int, default=0, help="0 = all available files/windows, else sample-limited run")
    parser.add_argument("--pamap2-include-optional", action="store_true", help="Include PAMAP2 Optional/ files")
    parser.add_argument("--target-rate", type=float, default=50.0)
    parser.add_argument("--window-size", type=int, default=None, help="Window size in samples")
    parser.add_argument("--step-size", type=int, default=None, help="Step size in samples")
    parser.add_argument("--keep-unacceptable", action="store_true", help="Keep unacceptable windows in feature table")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--results-root", default="results/runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--skip-plots", action="store_true", help="Skip confusion matrix PNG generation")
    parser.add_argument("--feature-preview-rows", type=int, default=25)
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


def _load_dataset(args: argparse.Namespace, path: Path) -> pd.DataFrame:
    if args.dataset == "uci_har":
        max_windows = None if args.sample_limit <= 0 else int(args.sample_limit)
        return load_uci_har(path, max_windows_per_split=max_windows)
    if args.dataset == "pamap2":
        max_files = None if args.sample_limit <= 0 else int(args.sample_limit)
        return load_pamap2(path, max_files=max_files, include_optional=args.pamap2_include_optional)
    if args.dataset == "wisdm":
        max_sessions = None if args.sample_limit <= 0 else int(args.sample_limit)
        return load_wisdm(path, max_sessions=max_sessions)
    raise ValueError(args.dataset)


def _load_wisdm_split(args: argparse.Namespace, path: Path, *, split_name: str) -> pd.DataFrame:
    max_sessions = None if args.sample_limit <= 0 else int(args.sample_limit)
    return load_wisdm(path, split=split_name, max_sessions=max_sessions)


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
    max_group_size = None
    if group_cols and len(df) > 0:
        try:
            group_sizes = df.groupby(group_cols, dropna=False, sort=False).size()
            min_group_size = int(group_sizes.min())
            max_group_size = int(group_sizes.max())
        except Exception:
            min_group_size = None
            max_group_size = None

    # Keep the chapter defaults whenever at least some groups are long enough to form
    # standard windows. Tiny fragments are simply skipped later during windowing.
    if max_group_size is None or max_group_size >= cfg.window_size_samples:
        return cfg.window_size_samples, cfg.step_size_samples, None

    # Fixture/smoke-friendly fallback only when every group is short.
    w = max(2, min(32, max_group_size))
    s = int(step_size or max(1, w // 2))
    note = (
        f"Using short-group fallback window_size={w}, step_size={s} because "
        f"largest group length ({max_group_size}) is below the default {cfg.window_size_samples}."
    )
    return w, s, note


def _build_run_id(dataset: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"har_baseline_{dataset}__{ts}"


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, default=str), encoding="utf-8")


def _augment_feature_schema_summary(feature_schema: dict[str, Any], feature_df: pd.DataFrame, *, dataset_key: str) -> dict[str, Any]:
    out = dict(feature_schema)
    if "source_file" in feature_df.columns:
        out["source_files_count"] = int(feature_df["source_file"].nunique(dropna=True))
    else:
        out["source_files_count"] = 0

    if dataset_key == "uci_har":
        out["session_id_note"] = (
            "Session IDs are loader-derived provenance identifiers from flattened pre-windowed UCI HAR splits/windows."
        )
    elif dataset_key == "pamap2":
        out["session_id_note"] = (
            "Session IDs are loader-derived sequence identifiers from PAMAP2 Protocol subject files."
        )
    elif dataset_key == "wisdm":
        out["session_id_note"] = (
            "WISDM session IDs are derived from timestamp resets, large gaps, and label changes within the provided train/test export."
        )
    else:
        out["session_id_note"] = "Session ID semantics depend on the dataset loader."
    return out


def _validation_payload(validation) -> dict[str, Any]:
    return {
        "is_valid": bool(validation.is_valid),
        "errors": list(validation.errors),
        "warnings": list(validation.warnings),
    }


def _prepare_feature_bundle(
    df: pd.DataFrame,
    *,
    cfg: PreprocessConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    validation = validate_ingestion_dataframe(df)
    rate_summary = summarize_sampling_rate_by_group(df)

    resampled = resample_dataframe(
        df,
        target_rate_hz=cfg.target_sampling_rate_hz,
        interpolation_method=cfg.interpolation_method,
    )
    resampled = append_derived_channels(resampled)

    window_size, step_size, window_note = _effective_window_sizes(
        resampled,
        cfg,
        window_size=args.window_size,
        step_size=args.step_size,
    )

    windows = window_dataframe(resampled, window_size=window_size, step_size=step_size, config=cfg)
    feature_df = build_feature_table(
        windows,
        filter_unacceptable=not args.keep_unacceptable,
        default_sampling_rate_hz=args.target_rate,
    )
    feature_df = filter_har_training_rows(
        feature_df,
        label_col="label_mapped_majority",
        allowed_labels=DEFAULT_HAR_ALLOWED_LABELS,
        require_acceptable=not args.keep_unacceptable,
    )

    return {
        "validation": validation,
        "rate_summary": rate_summary,
        "resampled": resampled,
        "window_size": window_size,
        "step_size": step_size,
        "window_note": window_note,
        "windows": windows,
        "feature_df": feature_df,
    }


def main() -> int:
    args = parse_args()
    dataset_path = _resolve_path(args.path)
    if not dataset_path.exists():
        print(f"ERROR: dataset path not found: {dataset_path}")
        return 1

    run_id = args.run_id or _build_run_id(args.dataset)
    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = (REPO_ROOT / results_root).resolve()
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = PreprocessConfig(target_sampling_rate_hz=args.target_rate)
    split_ingestion_validation: dict[str, Any] | None = None
    split_preprocessing_summary: dict[str, Any] | None = None
    rows_loaded_total = 0

    if args.dataset == "wisdm":
        print(f"Loading dataset: {args.dataset} from {dataset_path}")
        train_df = _load_wisdm_split(args, dataset_path, split_name="train")
        test_df = _load_wisdm_split(args, dataset_path, split_name="test")

        train_bundle = _prepare_feature_bundle(train_df, cfg=cfg, args=args)
        test_bundle = _prepare_feature_bundle(test_df, cfg=cfg, args=args)

        for split_name, bundle, raw_df in [
            ("train", train_bundle, train_df),
            ("test", test_bundle, test_df),
        ]:
            validation = bundle["validation"]
            if validation.errors:
                print(f"WARNING: WISDM {split_name} ingestion validation reported errors before preprocessing:")
                for err in validation.errors:
                    print(f"  - {err}")
            for warn in validation.warnings:
                print(f"{split_name} validation warning: {warn}")

            rate_summary = bundle["rate_summary"]
            print(
                f"{split_name}: rows_loaded={len(raw_df)} "
                f"estimated_sampling_rate_hz={rate_summary.get('median_hz')} "
                f"min_hz={rate_summary.get('min_hz')} "
                f"max_hz={rate_summary.get('max_hz')} "
                f"groups_checked={rate_summary.get('groups_checked')}"
            )
            if bundle["window_note"]:
                print(f"{split_name} window_note: {bundle['window_note']}")
            print(
                f"{split_name}: rows_after_resample={len(bundle['resampled'])} "
                f"windows_total={len(bundle['windows'])} "
                f"feature_rows={len(bundle['feature_df'])}"
            )

        train_feature_df = train_bundle["feature_df"]
        test_feature_df = test_bundle["feature_df"]
        feature_df = pd.concat([train_feature_df, test_feature_df], ignore_index=True)
        if train_feature_df.empty or test_feature_df.empty:
            print("ERROR: WISDM feature table is empty for one of the provided splits")
            return 1

        try:
            eval_result = run_har_baselines_on_train_test_feature_tables(
                train_feature_df,
                test_feature_df,
                random_state=args.random_state,
                split_extra={
                    "provided_split_names": ["train", "test"],
                    "split_note": (
                        "WISDM is evaluated using the provided train/test export because this dataset copy has no subject identifiers."
                    ),
                },
            )
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: baseline evaluation failed: {type(exc).__name__}: {exc}")
            return 1

        split_ingestion_validation = {
            "train": _validation_payload(train_bundle["validation"]),
            "test": _validation_payload(test_bundle["validation"]),
        }
        split_preprocessing_summary = {
            "train": {
                "rows_loaded": int(len(train_df)),
                "sampling_rate_summary": train_bundle["rate_summary"],
                "rows_after_resampling": int(len(train_bundle["resampled"])),
                "windows_total": int(len(train_bundle["windows"])),
                "feature_rows": int(len(train_feature_df)),
            },
            "test": {
                "rows_loaded": int(len(test_df)),
                "sampling_rate_summary": test_bundle["rate_summary"],
                "rows_after_resampling": int(len(test_bundle["resampled"])),
                "windows_total": int(len(test_bundle["windows"])),
                "feature_rows": int(len(test_feature_df)),
            },
        }
        validation = train_bundle["validation"]
        rate_summary = train_bundle["rate_summary"]
        window_size = train_bundle["window_size"]
        step_size = train_bundle["step_size"]
        rows_loaded_total = int(len(train_df) + len(test_df))
        resampled = pd.concat([train_bundle["resampled"], test_bundle["resampled"]], ignore_index=True)
        windows = [*train_bundle["windows"], *test_bundle["windows"]]
    else:
        print(f"Loading dataset: {args.dataset} from {dataset_path}")
        df = _load_dataset(args, dataset_path)
        bundle = _prepare_feature_bundle(df, cfg=cfg, args=args)
        validation = bundle["validation"]
        if validation.errors:
            print("WARNING: ingestion validation reported errors before preprocessing:")
            for err in validation.errors:
                print(f"  - {err}")
        for warn in validation.warnings:
            print(f"validation warning: {warn}")

        rate_summary = bundle["rate_summary"]
        print(
            f"rows_loaded={len(df)} "
            f"estimated_sampling_rate_hz={rate_summary.get('median_hz')} "
            f"min_hz={rate_summary.get('min_hz')} "
            f"max_hz={rate_summary.get('max_hz')} "
            f"groups_checked={rate_summary.get('groups_checked')}"
        )

        if bundle["window_note"]:
            print(f"window_note: {bundle['window_note']}")

        feature_df = bundle["feature_df"]
        if feature_df.empty:
            print("ERROR: feature table is empty (likely no windows formed or all windows filtered out)")
            return 1

        rows_loaded_total = int(len(df))
        resampled = bundle["resampled"]
        windows = bundle["windows"]
        window_size = bundle["window_size"]
        step_size = bundle["step_size"]
        print(f"rows_after_resample={len(resampled)} windows_total={len(windows)} feature_rows={len(feature_df)}")

        try:
            eval_result = run_har_baselines_on_feature_table(
                feature_df,
                test_size=args.test_size,
                random_state=args.random_state,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: baseline evaluation failed: {type(exc).__name__}: {exc}")
            return 1

    heuristic_metrics = eval_result["heuristic"]["metrics"]
    rf_metrics = eval_result["random_forest"]["metrics"]
    labels = eval_result["label_order"]

    print("\nBaseline summary")
    print(
        f"heuristic: accuracy={heuristic_metrics['accuracy']:.4f} "
        f"macro_f1={heuristic_metrics['macro_f1']:.4f}"
    )
    print(
        f"random_forest: accuracy={rf_metrics['accuracy']:.4f} "
        f"macro_f1={rf_metrics['macro_f1']:.4f}"
    )
    print(
        f"split: train_rows={eval_result['split']['train_rows']} "
        f"test_rows={eval_result['split']['test_rows']} "
        f"train_subjects={eval_result['split']['train_subjects_count']} "
        f"test_subjects={eval_result['split']['test_subjects_count']}"
    )

    # Save results/artifacts
    feature_schema = _augment_feature_schema_summary(
        feature_table_schema_summary(feature_df),
        feature_df,
        dataset_key=args.dataset,
    )
    (run_dir / "feature_table_preview.csv").write_text(
        feature_df.head(args.feature_preview_rows).to_csv(index=False),
        encoding="utf-8",
    )
    _save_json(run_dir / "feature_table_schema_summary.json", feature_schema)

    plot_warnings: list[str] = []

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
            "pamap2_include_optional": bool(args.pamap2_include_optional),
        },
        "ingestion_validation": _validation_payload(validation),
        "preprocessing_summary": {
            "rows_loaded": rows_loaded_total,
            "sampling_rate_summary": rate_summary,
            "rows_after_resampling": int(len(resampled)),
            "windows_total": int(len(windows)),
            "feature_rows": int(len(feature_df)),
        },
        "split": eval_result["split"],
        "label_order": labels,
        "heuristic": {"metrics": heuristic_metrics},
        "random_forest": {"metrics": rf_metrics},
        "feature_schema_summary": feature_schema,
    }
    if split_ingestion_validation is not None:
        metrics_payload["split_ingestion_validation"] = split_ingestion_validation
    if split_preprocessing_summary is not None:
        metrics_payload["split_preprocessing_summary"] = split_preprocessing_summary

    save_confusion_matrix_csv(
        heuristic_metrics["confusion_matrix"],
        labels,
        run_dir / "confusion_matrix_heuristic.csv",
    )
    save_confusion_matrix_csv(
        rf_metrics["confusion_matrix"],
        labels,
        run_dir / "confusion_matrix_random_forest.csv",
    )

    rf_importances = eval_result["random_forest"]["feature_importances"]
    if isinstance(rf_importances, pd.DataFrame) and not rf_importances.empty:
        rf_importances.to_csv(run_dir / "feature_importances_random_forest.csv", index=False)

    if not args.skip_plots:
        try:
            plot_confusion_matrix(
                heuristic_metrics["confusion_matrix"],
                labels,
                title=f"Heuristic HAR Confusion Matrix ({args.dataset})",
                out_path=run_dir / "confusion_matrix_heuristic.png",
            )
            plot_confusion_matrix(
                rf_metrics["confusion_matrix"],
                labels,
                title=f"RF HAR Confusion Matrix ({args.dataset})",
                out_path=run_dir / "confusion_matrix_random_forest.png",
            )
        except Exception as exc:  # noqa: BLE001
            warning_msg = f"failed to generate confusion matrix PNGs: {type(exc).__name__}: {exc}"
            plot_warnings.append(warning_msg)
            print(f"plot warning: {warning_msg}")

    metrics_payload["artifact_status"] = {
        "confusion_matrix_heuristic_csv": str(run_dir / "confusion_matrix_heuristic.csv"),
        "confusion_matrix_random_forest_csv": str(run_dir / "confusion_matrix_random_forest.csv"),
        "confusion_matrix_heuristic_png": str(run_dir / "confusion_matrix_heuristic.png")
        if (run_dir / "confusion_matrix_heuristic.png").exists()
        else None,
        "confusion_matrix_random_forest_png": str(run_dir / "confusion_matrix_random_forest.png")
        if (run_dir / "confusion_matrix_random_forest.png").exists()
        else None,
        "feature_importances_random_forest_csv": str(run_dir / "feature_importances_random_forest.csv")
        if (run_dir / "feature_importances_random_forest.csv").exists()
        else None,
        "plot_warnings": plot_warnings,
    }
    _save_json(run_dir / "metrics.json", metrics_payload)

    # Save a reproducible run summary snapshot
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

    print(f"\nSaved HAR baseline artifacts to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
