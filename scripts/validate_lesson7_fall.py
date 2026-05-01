#!/usr/bin/env python3
"""Chapter 5 validation runner with JSON reporting for threshold fall baseline."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Chapter 5 threshold-fall baseline and emit JSON summary")
    parser.add_argument("--dataset", required=True, choices=["mobifall", "sisfall"])
    parser.add_argument("--path", required=True, help="Dataset root/file path")
    parser.add_argument("--sample-limit", type=int, default=2, help="0 = full data, >0 = quick sample mode")
    parser.add_argument("--target-rate", type=float, default=50.0)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument("--keep-unacceptable", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--results-root", default="results/runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-run", action="store_true", help="Only assemble report from an existing metrics JSON")
    parser.add_argument("--metrics-json", default=None, help="Optional explicit metrics.json path when --skip-run is used")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument("--stdout-preview-chars", type=int, default=3000)
    parser.add_argument("--stderr-preview-chars", type=int, default=1500)
    return parser.parse_args()


def _resolve_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    p = Path(path_str)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p


def _default_run_id(dataset: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"lesson7_fall_validation_{dataset}__{ts}"


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


def truncate_preview(text: str | None, *, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated {len(text) - max_chars} chars]"


def _run_subprocess(
    cmd: list[str],
    *,
    cwd: Path,
    stdout_preview_chars: int,
    stderr_preview_chars: int,
) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return {
        "executed": True,
        "passed": proc.returncode == 0,
        "return_code": int(proc.returncode),
        "command": cmd,
        "stdout_preview": truncate_preview(proc.stdout, max_chars=stdout_preview_chars),
        "stderr_preview": truncate_preview(proc.stderr, max_chars=stderr_preview_chars),
    }


def _load_metrics_json(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None or not path.exists():
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload, None
    except Exception as exc:  # noqa: BLE001
        return None, f"Failed to parse metrics JSON ({path}): {type(exc).__name__}: {exc}"


def _extract_grouped_rate_summary(metrics_payload: dict[str, Any] | None) -> dict[str, Any]:
    default = {
        "groups_checked": 0,
        "median_hz": None,
        "min_hz": None,
        "max_hz": None,
        "estimated_rates_hz_preview": [],
    }
    if not metrics_payload:
        return default

    preprocessing = metrics_payload.get("preprocessing_summary") or {}
    grouped = preprocessing.get("grouped_sampling_rate_summary")
    if isinstance(grouped, dict):
        out = default.copy()
        out.update(
            {
                "groups_checked": int(grouped.get("groups_checked") or 0),
                "median_hz": grouped.get("median_hz"),
                "min_hz": grouped.get("min_hz"),
                "max_hz": grouped.get("max_hz"),
                "estimated_rates_hz_preview": list(grouped.get("estimated_rates_hz_preview") or []),
            }
        )
        return out

    # Backward-compatible fallback for older metrics payloads.
    fallback_rate = preprocessing.get("estimated_sampling_rate_hz")
    if fallback_rate is not None:
        out = default.copy()
        out["groups_checked"] = 1
        out["median_hz"] = fallback_rate
        out["min_hz"] = fallback_rate
        out["max_hz"] = fallback_rate
        out["estimated_rates_hz_preview"] = [fallback_rate]
        return out

    return default


def _extract_evaluation_summary(metrics_payload: dict[str, Any] | None, artifacts: dict[str, Any]) -> dict[str, Any]:
    metrics = (metrics_payload or {}).get("metrics") or {}
    cm_available = False
    if isinstance(metrics, dict) and metrics.get("confusion_matrix") is not None:
        cm_available = True
    cm_csv = artifacts.get("confusion_matrix_csv")
    cm_png = artifacts.get("confusion_matrix_png")
    if cm_csv and Path(str(cm_csv)).exists():
        cm_available = True
    if cm_png and Path(str(cm_png)).exists():
        cm_available = True

    return {
        "accuracy": metrics.get("accuracy"),
        "sensitivity": metrics.get("sensitivity"),
        "specificity": metrics.get("specificity"),
        "precision": metrics.get("precision"),
        "f1": metrics.get("f1"),
        "support_total": metrics.get("support_total"),
        "per_class_support": metrics.get("per_class_support", {}),
        "per_class_precision": metrics.get("per_class_precision", {}),
        "per_class_recall": metrics.get("per_class_recall", {}),
        "confusion_matrix_available": bool(cm_available),
    }


def build_lesson7_methodology_checks(
    *,
    dataset_path: Path,
    grouped_sampling_rate_summary: dict[str, Any],
    split_summary: dict[str, Any],
) -> dict[str, bool]:
    strategy = str(split_summary.get("strategy") or "").lower()
    train_groups = set(map(str, split_summary.get("train_subject_groups", []) or []))
    test_groups = set(map(str, split_summary.get("test_subject_groups", []) or []))

    subject_aware = False
    if train_groups and test_groups:
        subject_aware = train_groups.isdisjoint(test_groups)
    elif "group" in strategy or "subject" in strategy:
        subject_aware = True

    normalized_path = str(dataset_path).replace("\\", "/").lower()
    real_data_used = ("/tests/fixtures/" not in normalized_path) and dataset_path.exists()

    groups_checked = int(grouped_sampling_rate_summary.get("groups_checked") or 0)
    grouped_rate_used = groups_checked > 0 and grouped_sampling_rate_summary.get("median_hz") is not None

    return {
        "grouped_rate_estimation_used": bool(grouped_rate_used),
        "subject_aware_split_detected": bool(subject_aware),
        "real_data_used": bool(real_data_used),
    }


def assemble_lesson7_validation_report(
    *,
    repo_root: Path,
    dataset: str,
    dataset_path: Path,
    pipeline_run: dict[str, Any],
    grouped_sampling_rate_summary: dict[str, Any],
    split_summary: dict[str, Any],
    evaluation_summary: dict[str, Any],
    false_alarm_summary: dict[str, Any],
    methodology_checks: dict[str, bool],
    artifacts: dict[str, Any],
    notes: list[str] | None = None,
) -> dict[str, Any]:
    notes = list(notes or [])
    evaluation_ok = evaluation_summary.get("accuracy") is not None
    status = "ok" if bool(pipeline_run.get("passed")) and evaluation_ok else "failed"

    return {
        "validation_name": "lesson7_fall_validation",
        "dataset": dataset,
        "status": status,
        "repo_root": str(repo_root),
        "data_source": {
            "path": str(dataset_path),
        },
        "pipeline_run": pipeline_run,
        "grouped_sampling_rate_summary": grouped_sampling_rate_summary,
        "evaluation": evaluation_summary,
        "split": split_summary,
        "false_alarms": false_alarm_summary,
        "methodology_checks": methodology_checks,
        "artifacts": artifacts,
        "notes": notes,
    }


def main() -> int:
    args = parse_args()
    dataset_path = _resolve_path(args.path)
    if dataset_path is None:
        print("ERROR: --path is required")
        return 1

    results_root = _resolve_path(args.results_root) or (REPO_ROOT / "results" / "runs")
    run_id = args.run_id or _default_run_id(args.dataset)
    run_dir = results_root / run_id

    notes: list[str] = []

    if args.skip_run:
        pipeline_run = {
            "executed": False,
            "passed": None,
            "return_code": None,
            "command": [],
            "stdout_preview": "",
            "stderr_preview": "",
        }
    else:
        cmd = [
            sys.executable,
            "scripts/run_fall_threshold_baseline.py",
            "--dataset",
            args.dataset,
            "--path",
            str(dataset_path),
            "--sample-limit",
            str(args.sample_limit),
            "--target-rate",
            str(args.target_rate),
            "--results-root",
            str(results_root),
            "--run-id",
            run_id,
            "--test-size",
            str(args.test_size),
            "--random-state",
            str(args.random_state),
        ]
        if args.window_size is not None:
            cmd.extend(["--window-size", str(args.window_size)])
        if args.step_size is not None:
            cmd.extend(["--step-size", str(args.step_size)])
        if args.keep_unacceptable:
            cmd.append("--keep-unacceptable")
        if args.skip_plots:
            cmd.append("--skip-plots")

        pipeline_run = _run_subprocess(
            cmd,
            cwd=REPO_ROOT,
            stdout_preview_chars=args.stdout_preview_chars,
            stderr_preview_chars=args.stderr_preview_chars,
        )

    metrics_path = _resolve_path(args.metrics_json) if args.metrics_json else (run_dir / "metrics.json")
    metrics_payload, metrics_err = _load_metrics_json(metrics_path)
    if metrics_err:
        notes.append(metrics_err)
    if not metrics_payload:
        notes.append("metrics.json not found or unreadable")

    artifact_status = (metrics_payload or {}).get("artifact_status") or {}
    artifacts = {
        "run_dir": str(run_dir),
        "metrics_json": str(metrics_path) if metrics_path is not None else None,
        "predictions_windows_csv": artifact_status.get("predictions_windows_csv") or str(run_dir / "predictions_windows.csv"),
        "test_predictions_windows_csv": artifact_status.get("test_predictions_windows_csv") or str(run_dir / "test_predictions_windows.csv"),
        "false_alarms_csv": artifact_status.get("false_alarms_csv") or str(run_dir / "false_alarms.csv"),
        "confusion_matrix_csv": artifact_status.get("confusion_matrix_csv") or str(run_dir / "confusion_matrix_threshold_fall.csv"),
        "confusion_matrix_png": artifact_status.get("confusion_matrix_png") or str(run_dir / "confusion_matrix_threshold_fall.png"),
    }

    grouped_summary = _extract_grouped_rate_summary(metrics_payload)
    split_summary = (metrics_payload or {}).get("split") or {}
    evaluation_summary = _extract_evaluation_summary(metrics_payload, artifacts)

    false_alarm_count = None
    if isinstance((metrics_payload or {}).get("false_alarm_summary"), dict):
        false_alarm_count = (metrics_payload or {}).get("false_alarm_summary", {}).get("count")
    false_alarms = {
        "count": false_alarm_count,
        "artifact_path": artifacts.get("false_alarms_csv"),
        "artifact_exists": Path(str(artifacts.get("false_alarms_csv"))).exists() if artifacts.get("false_alarms_csv") else False,
    }

    methodology_checks = build_lesson7_methodology_checks(
        dataset_path=dataset_path,
        grouped_sampling_rate_summary=grouped_summary,
        split_summary=split_summary,
    )

    if not methodology_checks["grouped_rate_estimation_used"]:
        notes.append("Grouped sampling-rate summary missing or empty; check preprocessing_summary.grouped_sampling_rate_summary")
    if not methodology_checks["subject_aware_split_detected"]:
        notes.append("Subject-aware split was not detected from split metadata")
    if not methodology_checks["real_data_used"]:
        notes.append("Validation appears to use fixture/non-real path")

    report = assemble_lesson7_validation_report(
        repo_root=REPO_ROOT,
        dataset=args.dataset,
        dataset_path=dataset_path,
        pipeline_run=pipeline_run,
        grouped_sampling_rate_summary=grouped_summary,
        split_summary=split_summary,
        evaluation_summary=evaluation_summary,
        false_alarm_summary=false_alarms,
        methodology_checks=methodology_checks,
        artifacts=artifacts,
        notes=notes,
    )

    pretty = json.dumps(_json_safe(report), indent=2, default=str)
    print(pretty)

    if args.output:
        out_path = _resolve_path(args.output)
        assert out_path is not None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(pretty + "\n", encoding="utf-8")

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
