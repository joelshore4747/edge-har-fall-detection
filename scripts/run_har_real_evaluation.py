#!/usr/bin/env python3
"""Run real-data Chapter 4 HAR evaluations (UCI HAR and/or PAMAP2).

This script orchestrates real-data HAR baseline runs using the existing
``scripts/run_har_baseline.py`` entrypoint, then parses artifacts into
dissertation-friendly JSON summaries for:
- UCI HAR real evaluation
- PAMAP2 real evaluation
- combined comparison summary

It intentionally reuses existing pipeline logic instead of duplicating training.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the existing validation helpers for artifact parsing and methodology checks.
from scripts.validate_lesson5_har import (  # noqa: E402
    _artifact_paths,
    _json_safe,
    _resolve_path,
    _run_subprocess,
    build_methodology_checks,
    summarize_evaluation_from_metrics,
    summarize_feature_table_artifacts,
)


DEFAULT_REAL_PATHS = {
    "uci_har": "data/raw/UCIHAR_Dataset/UCI-HAR Dataset",
    "pamap2": "data/raw/PAMAP2_Dataset",
}

DATASET_CANONICAL_NAMES = {
    "uci_har": "UCIHAR",
    "pamap2": "PAMAP2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-data Chapter 4 HAR evaluation and JSON reporting")
    parser.add_argument("--dataset", choices=["uci_har", "pamap2"], default=None, help="Run one real dataset evaluation")
    parser.add_argument("--all", action="store_true", help="Run both UCI HAR and PAMAP2 real evaluations")
    parser.add_argument("--ucihar-path", default=DEFAULT_REAL_PATHS["uci_har"])
    parser.add_argument("--pamap2-path", default=DEFAULT_REAL_PATHS["pamap2"])
    parser.add_argument("--pamap2-include-optional", action="store_true", help="Include PAMAP2 Optional/ files")
    parser.add_argument("--sample-limit", type=int, default=0, help="0 = full evaluation over discovered files/windows")
    parser.add_argument("--target-rate", type=float, default=50.0)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--step-size", type=int, default=64)
    parser.add_argument("--keep-unacceptable", action="store_true")
    parser.add_argument("--results-root", default="results/runs")
    parser.add_argument("--validation-root", default="results/validation")
    parser.add_argument("--output-json", default=None, help="Output path for combined JSON (or single dataset JSON if not --all)")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--stdout-preview-chars", type=int, default=4000)
    parser.add_argument("--stderr-preview-chars", type=int, default=2000)
    return parser.parse_args()


def _choose_datasets(args: argparse.Namespace) -> list[str]:
    if args.all:
        return ["uci_har", "pamap2"]
    if args.dataset:
        return [args.dataset]
    # Safe default for a real-eval runner: be explicit, but allow a helpful default.
    return ["uci_har"]


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _dataset_real_path(dataset_key: str, args: argparse.Namespace) -> Path:
    if dataset_key == "uci_har":
        return _resolve_path(args.ucihar_path)  # type: ignore[return-value]
    if dataset_key == "pamap2":
        return _resolve_path(args.pamap2_path)  # type: ignore[return-value]
    raise ValueError(dataset_key)


def _dataset_run_id(dataset_key: str) -> str:
    return f"lesson5_real_{dataset_key}__{_timestamp()}"


def _read_metrics_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _artifacts_summary_for_json(artifacts: dict[str, Path]) -> dict[str, Any]:
    # Prefer RF artifacts when a single path is requested in the output schema.
    files = {name: str(path) if path.exists() else None for name, path in artifacts.items()}
    return {
        "run_dir": str(next(iter(artifacts.values())).parent) if artifacts else None,
        "metrics_json": files.get("metrics_json"),
        "confusion_matrix_csv": files.get("confusion_matrix_random_forest_csv") or files.get("confusion_matrix_heuristic_csv"),
        "confusion_matrix_png": files.get("confusion_matrix_random_forest_png") or files.get("confusion_matrix_heuristic_png"),
        "feature_importances_csv": files.get("feature_importances_random_forest_csv"),
        "feature_table_preview_csv": files.get("feature_table_preview_csv"),
        "feature_table_schema_summary_json": files.get("feature_table_schema_summary_json"),
        "all_artifacts": files,
    }


def _extract_dataset_evaluation_section(
    *,
    metrics_payload: dict[str, Any] | None,
    artifacts: dict[str, Path],
) -> dict[str, Any]:
    artifacts_found = {name: path.exists() for name, path in artifacts.items()}
    eval_summary = summarize_evaluation_from_metrics(metrics_payload, artifacts_found=artifacts_found)
    split_node = ((metrics_payload or {}).get("split") or {}) if isinstance(metrics_payload, dict) else {}
    return {
        "split": {
            "strategy": split_node.get("strategy", "group_shuffle_split_by_subject"),
            "train_rows": split_node.get("train_rows"),
            "test_rows": split_node.get("test_rows"),
            "train_subjects": split_node.get("train_subjects_count"),
            "test_subjects": split_node.get("test_subjects_count"),
        },
        "heuristic_baseline": eval_summary.get("heuristic_baseline", {}),
        "rf_baseline": eval_summary.get("rf_baseline", {}),
        "confusion_matrix_available": bool(eval_summary.get("confusion_matrix_available")),
        "feature_importances_available": bool(artifacts_found.get("feature_importances_random_forest_csv")),
    }


def _build_real_dataset_summary(
    *,
    dataset_key: str,
    dataset_path: Path,
    run_dir: Path,
    pipeline_run_summary: dict[str, Any],
) -> dict[str, Any]:
    artifacts = _artifact_paths(run_dir)
    feature_table = summarize_feature_table_artifacts(
        feature_table_csv_path=artifacts["feature_table_preview_csv"],
        schema_summary_json_path=artifacts["feature_table_schema_summary_json"],
    )
    metrics_payload = _read_metrics_json(artifacts["metrics_json"])
    evaluation = _extract_dataset_evaluation_section(metrics_payload=metrics_payload, artifacts=artifacts)

    eval_for_checks = {
        "heuristic_baseline": evaluation.get("heuristic_baseline", {}),
        "rf_baseline": evaluation.get("rf_baseline", {}),
    }
    methodology_checks = build_methodology_checks(
        metrics_payload=metrics_payload,
        evaluation_summary=eval_for_checks,
        feature_table_summary=feature_table,
    )
    methodology_checks["real_data_used"] = bool(dataset_path.exists() and "tests/fixtures" not in str(dataset_path))

    canonical_name = DATASET_CANONICAL_NAMES[dataset_key]
    data_notes = []
    prewindowed = canonical_name == "UCIHAR"
    if prewindowed:
        data_notes.append("UCI HAR is pre-windowed in source form and flattened for the common sample-level schema.")
        data_notes.append("Session IDs are loader-derived provenance identifiers tied to source split/window provenance.")
    else:
        data_notes.append("PAMAP2 uses continuous Protocol files and the Chapter 3 grouped resampling/windowing pipeline.")
        data_notes.append("Session IDs are loader-derived sequence identifiers from PAMAP2 Protocol subject files.")

    notes: list[str] = []
    interpretation_notes: list[str] = []
    if not pipeline_run_summary.get("passed"):
        notes.append("Baseline pipeline execution failed for this dataset.")
    if metrics_payload is None:
        notes.append("metrics.json missing or unreadable; evaluation fields may be unavailable.")
    if feature_table.get("rows") is not None:
        try:
            if int(feature_table["rows"]) < 20:
                notes.append("Feature table is very small; results may not be representative.")
        except Exception:
            pass

    # Concise, factual interpretation notes for dissertation reporting.
    heur = evaluation.get("heuristic_baseline", {}) or {}
    rf = evaluation.get("rf_baseline", {}) or {}
    heur_acc = heur.get("accuracy")
    heur_f1 = heur.get("macro_f1")
    rf_acc = rf.get("accuracy")
    rf_f1 = rf.get("macro_f1")
    if heur.get("available") and rf.get("available"):
        try:
            if rf_f1 is not None and heur_f1 is not None:
                if float(rf_f1) > float(heur_f1):
                    interpretation_notes.append(
                        "Random Forest outperforms the heuristic baseline on macro-F1, indicating useful information in engineered features."
                    )
                elif float(rf_f1) == float(heur_f1):
                    interpretation_notes.append(
                        "Random Forest and heuristic baselines perform similarly on macro-F1 for this run; inspect class-level behavior and supports."
                    )
                else:
                    interpretation_notes.append(
                        "Heuristic baseline outperforms Random Forest on macro-F1 in this run; inspect feature table size, split composition, and class supports."
                    )
        except Exception:
            pass
        if heur_acc is not None and rf_acc is not None:
            try:
                if float(rf_acc) < 0.5:
                    interpretation_notes.append(
                        "Overall accuracy is low/moderate; class support imbalance and cross-subject variability may be affecting performance."
                    )
            except Exception:
                pass
    if prewindowed:
        interpretation_notes.append(
            "UCI HAR results are often cleaner due to the benchmark's structured protocol and pre-windowed source format."
        )
    else:
        interpretation_notes.append(
            "PAMAP2 is a harder setting because continuous signals require preprocessing and include broader activity variability."
        )
    if not bool(evaluation.get("confusion_matrix_available")):
        notes.append("Confusion matrix artifacts not found; check plot/csv generation step.")
    elif _artifacts_summary_for_json(artifacts).get("confusion_matrix_png") is None:
        notes.append("Confusion matrix CSVs are available but PNGs are missing (plots may have been skipped or plotting failed).")

    status = "ok" if (
        bool(pipeline_run_summary.get("passed"))
        and bool(feature_table.get("available"))
        and bool(methodology_checks.get("heuristic_baseline_present"))
        and bool(methodology_checks.get("rf_baseline_present"))
        and bool(methodology_checks.get("subject_aware_split_detected"))
        and bool(methodology_checks.get("real_data_used"))
    ) else "failed"

    return {
        "validation_name": "lesson5_har_real_evaluation",
        "dataset": canonical_name,
        "status": status,
        "repo_root": str(REPO_ROOT),
        "data_source": {
            "path": str(dataset_path),
            "prewindowed_source": prewindowed,
            "notes": data_notes,
        },
        "feature_table": feature_table,
        "evaluation": evaluation,
        "methodology_checks": methodology_checks,
        "artifacts": _artifacts_summary_for_json(artifacts),
        "interpretation_notes": interpretation_notes,
        "notes": notes,
        "pipeline_run": {
            "executed": pipeline_run_summary.get("executed"),
            "passed": pipeline_run_summary.get("passed"),
            "return_code": pipeline_run_summary.get("return_code"),
            "stdout_preview": pipeline_run_summary.get("stdout_preview", ""),
            "stderr_preview": pipeline_run_summary.get("stderr_preview", ""),
        },
    }


def build_comparison_summary(dataset_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    datasets = [str(d.get("dataset")) for d in dataset_summaries]
    metrics: dict[str, dict[str, Any]] = {}
    interpretation_notes: list[str] = []

    for summary in dataset_summaries:
        name = str(summary.get("dataset"))
        eval_node = summary.get("evaluation", {}) or {}
        heur = eval_node.get("heuristic_baseline", {}) or {}
        rf = eval_node.get("rf_baseline", {}) or {}
        metrics[name] = {
            "heuristic_accuracy": heur.get("accuracy"),
            "heuristic_macro_f1": heur.get("macro_f1"),
            "rf_accuracy": rf.get("accuracy"),
            "rf_macro_f1": rf.get("macro_f1"),
        }
        if summary.get("status") != "ok":
            interpretation_notes.append(f"{name} evaluation status is not ok; inspect per-dataset JSON for failures/notes.")
        if heur.get("available") and rf.get("available"):
            try:
                if (rf.get("macro_f1") is not None) and (heur.get("macro_f1") is not None) and float(rf["macro_f1"]) > float(heur["macro_f1"]):
                    interpretation_notes.append(f"{name}: Random Forest outperforms the heuristic baseline on macro-F1.")
            except Exception:
                pass

    if "UCIHAR" in metrics and "PAMAP2" in metrics:
        interpretation_notes.append(
            "Compare metrics cautiously: UCI HAR is pre-windowed in source form, while PAMAP2 requires continuous-to-window preprocessing."
        )
        interpretation_notes.append(
            "Both runs use subject-aware splitting; differences may still reflect dataset protocol and label-space composition, not only model quality."
        )

    status = "ok" if all(d.get("status") == "ok" for d in dataset_summaries) else "failed"
    return {
        "comparison_name": "lesson5_har_real_comparison",
        "status": status,
        "datasets": datasets,
        "metrics": metrics,
        "methodology_notes": [
            "UCI HAR is pre-windowed in source form.",
            "PAMAP2 uses continuous-to-window preprocessing.",
            "Subject-aware splitting is used to reduce subject leakage risk.",
        ],
        "interpretation_notes": interpretation_notes,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, default=str) + "\n", encoding="utf-8")


def _dataset_validation_output_path(validation_root: Path, dataset_key: str) -> Path:
    stem = "lesson5_uci_har_real.json" if dataset_key == "uci_har" else "lesson5_pamap2_real.json"
    return validation_root / stem


def _comparison_output_path(validation_root: Path) -> Path:
    return validation_root / "lesson5_har_comparison.json"


def _run_dataset_baseline(
    *,
    dataset_key: str,
    dataset_path: Path,
    args: argparse.Namespace,
    results_root: Path,
    run_id: str,
) -> dict[str, Any]:
    if not dataset_path.exists():
        return {
            "executed": False,
            "passed": False,
            "return_code": None,
            "command": [],
            "stdout_preview": "",
            "stderr_preview": "",
            "notes": [f"Dataset path missing: {dataset_path}"],
        }

    cmd = [
        sys.executable,
        "scripts/run_har_baseline.py",
        "--dataset",
        dataset_key,
        "--path",
        str(dataset_path),
        "--sample-limit",
        str(args.sample_limit),
        "--target-rate",
        str(args.target_rate),
        "--window-size",
        str(args.window_size),
        "--step-size",
        str(args.step_size),
        "--results-root",
        str(results_root),
        "--run-id",
        run_id,
    ]
    if args.skip_plots:
        cmd.append("--skip-plots")
    if args.keep_unacceptable:
        cmd.append("--keep-unacceptable")
    if dataset_key == "pamap2" and args.pamap2_include_optional:
        cmd.append("--pamap2-include-optional")

    return _run_subprocess(
        cmd,
        cwd=REPO_ROOT,
        stdout_preview_chars=args.stdout_preview_chars,
        stderr_preview_chars=args.stderr_preview_chars,
    )


def main() -> int:
    args = parse_args()
    dataset_keys = _choose_datasets(args)
    results_root = _resolve_path(args.results_root) or (REPO_ROOT / "results" / "runs")
    validation_root = _resolve_path(args.validation_root) or (REPO_ROOT / "results" / "validation")
    validation_root.mkdir(parents=True, exist_ok=True)

    per_dataset_summaries: list[dict[str, Any]] = []
    for dataset_key in dataset_keys:
        dataset_path = _dataset_real_path(dataset_key, args)
        run_id = _dataset_run_id(dataset_key)
        pipeline_run = _run_dataset_baseline(
            dataset_key=dataset_key,
            dataset_path=dataset_path,
            args=args,
            results_root=results_root,
            run_id=run_id,
        )
        summary = _build_real_dataset_summary(
            dataset_key=dataset_key,
            dataset_path=dataset_path,
            run_dir=results_root / run_id,
            pipeline_run_summary=pipeline_run,
        )

        out_path = _dataset_validation_output_path(validation_root, dataset_key)
        _write_json(out_path, summary)
        per_dataset_summaries.append(summary)

    if len(per_dataset_summaries) > 1:
        comparison = build_comparison_summary(per_dataset_summaries)
        comparison_out = _resolve_path(args.output_json) if args.output_json else _comparison_output_path(validation_root)
        assert comparison_out is not None
        _write_json(comparison_out, comparison)
        print(json.dumps(_json_safe(comparison), indent=2, default=str))
        return 0 if comparison.get("status") == "ok" else 1

    # Single dataset mode: print per-dataset summary and optionally write custom path.
    summary = per_dataset_summaries[0] if per_dataset_summaries else {
        "validation_name": "lesson5_har_real_evaluation",
        "status": "failed",
        "repo_root": str(REPO_ROOT),
        "notes": ["No dataset selected"],
    }
    if args.output_json:
        out_path = _resolve_path(args.output_json)
        assert out_path is not None
        _write_json(out_path, summary)
    print(json.dumps(_json_safe(summary), indent=2, default=str))
    return 0 if summary.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
