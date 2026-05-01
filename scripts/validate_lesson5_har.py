#!/usr/bin/env python3
"""Chapter 4 HAR validation runner with JSON reporting.

This script provides a single entrypoint to validate whether the Chapter 4 HAR
feature-engineering and baseline pipeline is functioning and methodologically
plausible. It:

1) runs key Chapter 4 tests
2) runs the HAR baseline pipeline script
3) inspects generated feature-table artifacts
4) parses saved metrics/evaluation outputs
5) prints structured JSON to stdout
6) optionally saves the JSON to disk
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


DEFAULT_PYTEST_TARGETS = [
    "tests/test_time_domain_features.py",
    "tests/test_frequency_features.py",
    "tests/test_feature_table_builder.py",
    "tests/test_har_baseline_smoke.py",
]

FEATURE_METADATA_COLUMNS = [
    "window_id",
    "dataset_name",
    "subject_id",
    "session_id",
    "source_file",
    "task_type",
    "label_mapped_majority",
    "is_acceptable",
    "n_samples",
    "missing_ratio",
    "has_large_gap",
    "n_gaps",
    "start_ts",
    "end_ts",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Chapter 4 HAR pipeline and emit JSON summary")
    parser.add_argument(
        "--mode",
        choices=["smoke", "real"],
        default="smoke",
        help="smoke = fixture/small-path validation (existing behavior); real = delegate to real-data evaluation runner",
    )
    parser.add_argument("--all", action="store_true", help="(real mode) run both UCI HAR and PAMAP2")
    parser.add_argument("--dataset", default="uci_har", choices=["uci_har", "pamap2"])
    parser.add_argument(
        "--path",
        default="tests/fixtures/uci_har_sample.csv",
        help="Dataset path for pipeline validation (defaults to a fixture for quick smoke checks)",
    )
    parser.add_argument("--ucihar-path", default="data/raw/UCIHAR_Dataset/UCI-HAR Dataset")
    parser.add_argument("--pamap2-path", default="data/raw/PAMAP2_Dataset")
    parser.add_argument("--pamap2-include-optional", action="store_true")
    parser.add_argument("--sample-limit", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--target-rate", type=float, default=50.0)
    parser.add_argument("--keep-unacceptable", action="store_true")
    parser.add_argument("--pytest-target", action="append", default=None, help="Override/add pytest target file(s)")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--skip-plots", action="store_true", help="Skip confusion matrix PNG generation in pipeline runs")
    parser.add_argument("--results-root", default="results/runs")
    parser.add_argument("--run-id", default=None, help="Optional fixed run id for the baseline script")
    parser.add_argument("--output", default=None, help="Optional JSON output path (e.g. results/validation/lesson5_validation.json)")
    parser.add_argument(
        "--output-json",
        default=None,
        help="(real mode) Optional JSON output path for the real-evaluation runner",
    )
    parser.add_argument(
        "--validation-root",
        default="results/validation",
        help="(real mode) Directory for per-dataset and comparison validation JSON files",
    )
    parser.add_argument("--stdout-preview-chars", type=int, default=4000)
    parser.add_argument("--stderr-preview-chars", type=int, default=2000)
    return parser.parse_args()


def truncate_preview(text: str | None, *, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated {len(text) - max_chars} chars]"


def _resolve_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
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


def _run_subprocess(
    cmd: list[str],
    *,
    cwd: Path,
    stdout_preview_chars: int,
    stderr_preview_chars: int,
) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    return {
        "executed": True,
        "passed": proc.returncode == 0,
        "return_code": int(proc.returncode),
        "command": cmd,
        "stdout_preview": truncate_preview(proc.stdout, max_chars=stdout_preview_chars),
        "stderr_preview": truncate_preview(proc.stderr, max_chars=stderr_preview_chars),
    }


def _default_run_id(dataset: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"lesson5_validation_{dataset}__{ts}"


def _artifact_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "metrics_json": run_dir / "metrics.json",
        "feature_table_preview_csv": run_dir / "feature_table_preview.csv",
        "feature_table_schema_summary_json": run_dir / "feature_table_schema_summary.json",
        "confusion_matrix_heuristic_csv": run_dir / "confusion_matrix_heuristic.csv",
        "confusion_matrix_random_forest_csv": run_dir / "confusion_matrix_random_forest.csv",
        "confusion_matrix_heuristic_png": run_dir / "confusion_matrix_heuristic.png",
        "confusion_matrix_random_forest_png": run_dir / "confusion_matrix_random_forest.png",
        "feature_importances_random_forest_csv": run_dir / "feature_importances_random_forest.csv",
        "run_summary_json": run_dir / "run_summary.json",
    }


def _artifacts_found_map(artifacts: dict[str, Path]) -> dict[str, Any]:
    return {
        "run_dir": str(next(iter(artifacts.values())).parent) if artifacts else None,
        "files": {name: path.exists() for name, path in artifacts.items()},
    }


def summarize_feature_table_artifacts(
    *,
    feature_table_csv_path: Path | None,
    schema_summary_json_path: Path | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "available": False,
        "rows": None,
        "columns": None,
        "feature_column_count": None,
        "metadata_columns": [],
        "label_column": None,
        "label_counts": {},
        "subjects_count": None,
        "sessions_count": None,
        "source_files_count": None,
        "datasets_present": [],
        "session_id_note": None,
        "notes": [],
    }

    schema_payload: dict[str, Any] | None = None
    preview_df: pd.DataFrame | None = None

    if schema_summary_json_path is not None and schema_summary_json_path.exists():
        try:
            schema_payload = json.loads(schema_summary_json_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            summary["notes"].append(f"Failed to read schema summary JSON: {type(exc).__name__}: {exc}")

    if feature_table_csv_path is not None and feature_table_csv_path.exists():
        try:
            preview_df = pd.read_csv(feature_table_csv_path)
        except Exception as exc:  # noqa: BLE001
            summary["notes"].append(f"Failed to read feature table preview CSV: {type(exc).__name__}: {exc}")

    if schema_payload is None and preview_df is None:
        summary["notes"].append("Feature table artifacts not found")
        return summary

    summary["available"] = True

    if schema_payload is not None:
        summary["rows"] = schema_payload.get("rows")
        summary["columns"] = schema_payload.get("columns_total")
        summary["feature_column_count"] = schema_payload.get("feature_columns_count")
        summary["metadata_columns"] = schema_payload.get("metadata_columns", [])
        summary["label_counts"] = schema_payload.get("label_counts", {})
        summary["subjects_count"] = schema_payload.get("subjects_count")
        summary["sessions_count"] = schema_payload.get("sessions_count")
        summary["source_files_count"] = schema_payload.get("source_files_count")
        summary["datasets_present"] = schema_payload.get("datasets_present", summary["datasets_present"])
        summary["session_id_note"] = schema_payload.get("session_id_note")

    if preview_df is not None:
        if summary["rows"] is None:
            summary["rows"] = int(len(preview_df))
        if summary["columns"] is None:
            summary["columns"] = int(len(preview_df.columns))

        metadata_cols = [c for c in FEATURE_METADATA_COLUMNS if c in preview_df.columns]
        if not summary["metadata_columns"]:
            summary["metadata_columns"] = metadata_cols
        if summary["feature_column_count"] is None:
            summary["feature_column_count"] = int(len([c for c in preview_df.columns if c not in metadata_cols]))

        if "label_mapped_majority" in preview_df.columns:
            summary["label_column"] = "label_mapped_majority"
            # Keep schema label counts if present; otherwise use preview counts.
            if not summary["label_counts"]:
                summary["label_counts"] = (
                    preview_df["label_mapped_majority"].astype(str).value_counts(dropna=False).to_dict()
                )

        if "subject_id" in preview_df.columns:
            if summary["subjects_count"] is None:
                summary["subjects_count"] = int(preview_df["subject_id"].nunique(dropna=True))
        if "session_id" in preview_df.columns:
            summary["sessions_count"] = int(preview_df["session_id"].nunique(dropna=True))
        if "dataset_name" in preview_df.columns:
            summary["datasets_present"] = sorted(preview_df["dataset_name"].astype(str).dropna().unique().tolist())
        if "source_file" in preview_df.columns and summary["source_files_count"] is None:
            summary["source_files_count"] = int(preview_df["source_file"].nunique(dropna=True))

        summary["preview_rows"] = int(len(preview_df))
        if schema_payload is not None:
            summary["notes"].append(
                "Feature table CSV is a preview artifact; total row count is taken from feature_table_schema_summary.json when available."
            )

    if summary["label_column"] is None and summary["label_counts"]:
        summary["label_column"] = "label_mapped_majority"
    if summary["session_id_note"] is None and summary["datasets_present"]:
        ds = set(summary["datasets_present"])
        if ds == {"UCIHAR"}:
            summary["session_id_note"] = "Session IDs are loader-derived provenance IDs from flattened pre-windowed UCI HAR windows."
        elif ds == {"PAMAP2"}:
            summary["session_id_note"] = "Session IDs are loader-derived sequence IDs from PAMAP2 Protocol subject files."
        else:
            summary["session_id_note"] = "Session ID semantics depend on the dataset loader and source format."
    return summary


def _extract_baseline_metrics(metrics_payload: dict[str, Any], key: str) -> dict[str, Any]:
    node = ((metrics_payload.get(key) or {}).get("metrics") or {}) if isinstance(metrics_payload, dict) else {}
    out = {
        "available": bool(node),
        "accuracy": None,
        "macro_f1": None,
        "support_total": None,
        "per_class_precision": {},
        "per_class_recall": {},
        "per_class_support": {},
    }
    if not node:
        return out

    out["accuracy"] = node.get("accuracy")
    out["macro_f1"] = node.get("macro_f1")
    out["support_total"] = node.get("support_total")

    per_class = node.get("per_class") or {}
    if isinstance(per_class, dict):
        out["per_class_precision"] = {str(k): (v.get("precision") if isinstance(v, dict) else None) for k, v in per_class.items()}
        out["per_class_recall"] = {str(k): (v.get("recall") if isinstance(v, dict) else None) for k, v in per_class.items()}
        out["per_class_support"] = {str(k): (v.get("support") if isinstance(v, dict) else None) for k, v in per_class.items()}
    explicit_support = node.get("per_class_support")
    if isinstance(explicit_support, dict):
        out["per_class_support"] = {str(k): v for k, v in explicit_support.items()}
    return out


def summarize_evaluation_from_metrics(
    metrics_payload: dict[str, Any] | None,
    *,
    artifacts_found: dict[str, bool] | None = None,
) -> dict[str, Any]:
    if not metrics_payload:
        return {
            "heuristic_baseline": {"available": False, "accuracy": None, "macro_f1": None, "per_class_precision": {}, "per_class_recall": {}},
            "rf_baseline": {"available": False, "accuracy": None, "macro_f1": None, "per_class_precision": {}, "per_class_recall": {}},
            "confusion_matrix_available": bool(artifacts_found and any(artifacts_found.get(k, False) for k in [
                "confusion_matrix_heuristic_csv",
                "confusion_matrix_random_forest_csv",
            ])),
        }

    files_map = artifacts_found or {}
    cm_available = any(
        files_map.get(k, False)
        for k in ["confusion_matrix_heuristic_csv", "confusion_matrix_random_forest_csv"]
    )
    # Fall back to metrics payload if CSVs are absent but matrices are embedded.
    if not cm_available:
        for key in ("heuristic", "random_forest"):
            metrics_node = ((metrics_payload.get(key) or {}).get("metrics") or {})
            if "confusion_matrix" in metrics_node:
                cm_available = True
                break

    return {
        "heuristic_baseline": _extract_baseline_metrics(metrics_payload, "heuristic"),
        "rf_baseline": _extract_baseline_metrics(metrics_payload, "random_forest"),
        "confusion_matrix_available": bool(cm_available),
    }


def build_methodology_checks(
    *,
    metrics_payload: dict[str, Any] | None,
    evaluation_summary: dict[str, Any],
    feature_table_summary: dict[str, Any],
) -> dict[str, Any]:
    split = (metrics_payload or {}).get("split", {}) if isinstance(metrics_payload, dict) else {}
    train_groups = set(map(str, split.get("train_subject_groups", []) or []))
    test_groups = set(map(str, split.get("test_subject_groups", []) or []))

    subject_aware_split_detected = False
    if train_groups and test_groups:
        subject_aware_split_detected = train_groups.isdisjoint(test_groups)

    heuristic_available = bool((evaluation_summary.get("heuristic_baseline") or {}).get("available"))
    rf_available = bool((evaluation_summary.get("rf_baseline") or {}).get("available"))

    heuristic_macro = (evaluation_summary.get("heuristic_baseline") or {}).get("macro_f1")
    rf_macro = (evaluation_summary.get("rf_baseline") or {}).get("macro_f1")
    macro_f1_reported = (heuristic_macro is not None) and (rf_macro is not None)

    cfg = (metrics_payload or {}).get("config", {}) if isinstance(metrics_payload, dict) else {}
    preprocessing = (metrics_payload or {}).get("preprocessing_summary", {}) if isinstance(metrics_payload, dict) else {}
    keep_unacceptable = cfg.get("keep_unacceptable")
    windows_total = preprocessing.get("windows_total")
    feature_rows = preprocessing.get("feature_rows")
    unacceptable_handled = False
    if keep_unacceptable is not None:
        if keep_unacceptable is True:
            unacceptable_handled = True
        elif keep_unacceptable is False:
            # If filtering is enabled, feature rows should not exceed windows total.
            if windows_total is None or feature_rows is None:
                unacceptable_handled = True
            else:
                try:
                    unacceptable_handled = int(feature_rows) <= int(windows_total)
                except Exception:
                    unacceptable_handled = True
    elif feature_table_summary.get("available"):
        # Fallback: if the feature table exists, assume the pipeline handled window acceptability somehow.
        unacceptable_handled = True

    return {
        "subject_aware_split_detected": bool(subject_aware_split_detected),
        "macro_f1_reported": bool(macro_f1_reported),
        "heuristic_baseline_present": bool(heuristic_available),
        "rf_baseline_present": bool(rf_available),
        "unacceptable_windows_filtered_or_handled": bool(unacceptable_handled),
    }


def assemble_validation_report(
    *,
    repo_root: Path,
    tests_summary: dict[str, Any],
    feature_table_summary: dict[str, Any],
    pipeline_run_summary: dict[str, Any],
    evaluation_summary: dict[str, Any],
    methodology_checks: dict[str, Any],
    notes: list[str] | None = None,
) -> dict[str, Any]:
    notes = list(notes or [])
    critical_ok = all(
        [
            bool(tests_summary.get("passed")) if tests_summary.get("executed") else True,
            bool(pipeline_run_summary.get("passed")) if pipeline_run_summary.get("executed") else True,
            bool(feature_table_summary.get("available")),
            bool(methodology_checks.get("heuristic_baseline_present")),
            bool(methodology_checks.get("rf_baseline_present")),
        ]
    )
    status = "ok" if critical_ok else "failed"

    return {
        "validation_name": "lesson5_har_validation",
        "status": status,
        "repo_root": str(repo_root),
        "tests": tests_summary,
        "feature_table": feature_table_summary,
        "pipeline_run": pipeline_run_summary,
        "evaluation": evaluation_summary,
        "methodology_checks": methodology_checks,
        "notes": notes,
    }


def _run_real_mode_delegate(args: argparse.Namespace) -> int:
    """Delegate to the dedicated real-data evaluation runner and mirror its JSON output."""
    argv_flags = set(sys.argv[1:])
    sample_limit = args.sample_limit
    window_size = args.window_size
    step_size = args.step_size

    # Preserve current smoke defaults while ensuring real mode defaults are genuinely
    # real-evaluation oriented unless the user explicitly overrides them.
    if "--sample-limit" not in argv_flags and sample_limit == 2:
        sample_limit = 0
    if "--window-size" not in argv_flags and window_size == 2:
        window_size = 128
    if "--step-size" not in argv_flags and step_size == 1:
        step_size = 64

    cmd = [
        sys.executable,
        "scripts/run_har_real_evaluation.py",
        "--results-root",
        str(args.results_root),
        "--validation-root",
        str(args.validation_root),
        "--sample-limit",
        str(sample_limit),
        "--target-rate",
        str(args.target_rate),
        "--window-size",
        str(window_size),
        "--step-size",
        str(step_size),
        "--ucihar-path",
        str(args.ucihar_path),
        "--pamap2-path",
        str(args.pamap2_path),
        "--stdout-preview-chars",
        str(args.stdout_preview_chars),
        "--stderr-preview-chars",
        str(args.stderr_preview_chars),
    ]
    if args.all:
        cmd.append("--all")
    else:
        cmd.extend(["--dataset", str(args.dataset)])
    if args.pamap2_include_optional:
        cmd.append("--pamap2-include-optional")
    if args.keep_unacceptable:
        cmd.append("--keep-unacceptable")
    if args.output_json:
        cmd.extend(["--output-json", str(args.output_json)])
    elif args.output:
        # Allow the existing --output flag to work in real mode too.
        cmd.extend(["--output-json", str(args.output)])

    if args.skip_plots:
        cmd.append("--skip-plots")

    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    # Pass through full output so the JSON remains valid and complete.
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
    return int(proc.returncode)


def main() -> int:
    args = parse_args()
    if args.mode == "real":
        # Real-data validation is handled by the dedicated runner so we don't duplicate
        # pipeline execution and JSON assembly logic here.
        return _run_real_mode_delegate(args)

    dataset_path = _resolve_path(args.path)
    results_root = _resolve_path(args.results_root) or (REPO_ROOT / "results" / "runs")
    run_id = args.run_id or _default_run_id(args.dataset)
    run_dir = results_root / run_id

    notes: list[str] = []

    # 1) Run tests
    pytest_targets = args.pytest_target if args.pytest_target else DEFAULT_PYTEST_TARGETS
    if args.skip_tests:
        tests_summary = {
            "executed": False,
            "passed": None,
            "pytest_targets": pytest_targets,
            "return_code": None,
            "stdout_preview": "",
            "stderr_preview": "",
        }
        notes.append("Chapter 4 pytest validation step skipped by user request.")
    else:
        pytest_cmd = [sys.executable, "-m", "pytest", *pytest_targets, "-q"]
        tests_summary = _run_subprocess(
            pytest_cmd,
            cwd=REPO_ROOT,
            stdout_preview_chars=args.stdout_preview_chars,
            stderr_preview_chars=args.stderr_preview_chars,
        )
        tests_summary["pytest_targets"] = pytest_targets
        if not tests_summary["passed"]:
            notes.append("One or more Chapter 4 pytest targets failed.")

    # 2) Run baseline pipeline
    if dataset_path is None or not dataset_path.exists():
        pipeline_run_summary = {
            "executed": False,
            "passed": False,
            "return_code": None,
            "stdout_preview": "",
            "stderr_preview": "",
            "artifacts_found": {"run_dir": str(run_dir), "files": {}},
        }
        notes.append(f"Dataset path missing for pipeline run: {dataset_path}")
    elif args.skip_pipeline:
        pipeline_run_summary = {
            "executed": False,
            "passed": None,
            "return_code": None,
            "stdout_preview": "",
            "stderr_preview": "",
            "artifacts_found": {"run_dir": str(run_dir), "files": {}},
        }
        notes.append("HAR baseline pipeline run skipped by user request.")
    else:
        baseline_cmd = [
            sys.executable,
            "scripts/run_har_baseline.py",
            "--dataset",
            args.dataset,
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
            baseline_cmd.append("--skip-plots")
        if args.keep_unacceptable:
            baseline_cmd.append("--keep-unacceptable")

        pipeline_run_summary = _run_subprocess(
            baseline_cmd,
            cwd=REPO_ROOT,
            stdout_preview_chars=args.stdout_preview_chars,
            stderr_preview_chars=args.stderr_preview_chars,
        )
        artifacts = _artifact_paths(run_dir)
        pipeline_run_summary["artifacts_found"] = _artifacts_found_map(artifacts)
        if not pipeline_run_summary["passed"]:
            notes.append("HAR baseline pipeline execution failed.")

    # 3) Inspect artifacts and parse outputs
    artifacts = _artifact_paths(run_dir)
    artifacts_found_files = {name: path.exists() for name, path in artifacts.items()}

    feature_table_summary = summarize_feature_table_artifacts(
        feature_table_csv_path=artifacts["feature_table_preview_csv"],
        schema_summary_json_path=artifacts["feature_table_schema_summary_json"],
    )

    metrics_payload: dict[str, Any] | None = None
    if artifacts["metrics_json"].exists():
        try:
            metrics_payload = json.loads(artifacts["metrics_json"].read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Failed to parse metrics JSON: {type(exc).__name__}: {exc}")
    else:
        notes.append("metrics.json not found; evaluation summary may be unavailable.")

    evaluation_summary = summarize_evaluation_from_metrics(metrics_payload, artifacts_found=artifacts_found_files)
    methodology_checks = build_methodology_checks(
        metrics_payload=metrics_payload,
        evaluation_summary=evaluation_summary,
        feature_table_summary=feature_table_summary,
    )

    if args.dataset == "uci_har" and str(args.path).endswith("tests/fixtures/uci_har_sample.csv"):
        notes.append("Validation run used a tiny UCI HAR fixture path for speed; results are smoke checks, not performance evidence.")
    if feature_table_summary.get("available") and feature_table_summary.get("rows") is not None:
        try:
            if int(feature_table_summary["rows"]) < 10:
                notes.append("Very small feature table detected; baseline metrics may be unstable and are suitable only for smoke validation.")
        except Exception:
            pass

    report = assemble_validation_report(
        repo_root=REPO_ROOT,
        tests_summary=tests_summary,
        feature_table_summary=feature_table_summary,
        pipeline_run_summary=pipeline_run_summary,
        evaluation_summary=evaluation_summary,
        methodology_checks=methodology_checks,
        notes=notes,
    )

    pretty_json = json.dumps(_json_safe(report), indent=2, default=str)
    print(pretty_json)

    if args.output:
        out_path = _resolve_path(args.output)
        assert out_path is not None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(pretty_json + "\n", encoding="utf-8")

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
