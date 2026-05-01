#!/usr/bin/env python3
"""Run SisFall Chapter 5 experiments and generate a report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results" / "runs"
VALIDATION_ROOT = REPO_ROOT / "results" / "validation"
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "raw" / "SISFALL_Dataset" / "SisFall_dataset"
DEFAULT_REPORT_JSON = VALIDATION_ROOT / "lesson7_sisfall_report.json"
DEFAULT_REPORT_MD = VALIDATION_ROOT / "lesson7_sisfall_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chapter 5 SisFall experiments and generate report.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--reuse-existing", action="store_true", help="Skip runs if target run dir already exists.")
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_cmd(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    return {
        "command": cmd,
        "return_code": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "passed": proc.returncode == 0,
    }


def _maybe_run(cmd: list[str], run_dir: Path, *, reuse_existing: bool) -> dict[str, Any]:
    if reuse_existing and run_dir.exists():
        return {
            "command": cmd,
            "return_code": 0,
            "stdout": f"skipped_existing={run_dir}",
            "stderr": "",
            "passed": True,
        }
    return _run_cmd(cmd)


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"ERROR: dataset path not found: {dataset_path}")
        return 1

    ts = _timestamp()
    results: dict[str, Any] = {"timestamp": ts, "runs": {}}

    validation_run_id = f"lesson7_fall_validation_sisfall__{ts}"
    validation_cmd = [
        sys.executable,
        "scripts/validate_lesson7_fall.py",
        "--dataset",
        "sisfall",
        "--path",
        str(dataset_path),
        "--sample-limit",
        "0",
        "--run-id",
        validation_run_id,
        "--output",
        str(VALIDATION_ROOT / "lesson7_sisfall_full_dyn.json"),
    ]
    validation_run_dir = RESULTS_ROOT / validation_run_id
    results["runs"]["validation"] = _maybe_run(validation_cmd, validation_run_dir, reuse_existing=args.reuse_existing)

    impact_only_id = f"sisfall_debug_impact_only__{ts}"
    impact_only_cmd = [
        sys.executable,
        "scripts/run_fall_threshold_baseline.py",
        "--dataset",
        "sisfall",
        "--path",
        str(dataset_path),
        "--sample-limit",
        "0",
        "--run-id",
        impact_only_id,
        "--impact-threshold",
        "490",
        "--disable-support-stage",
        "--disable-confirm-stage",
        "--skip-plots",
    ]
    results["runs"]["impact_only"] = _maybe_run(impact_only_cmd, RESULTS_ROOT / impact_only_id, reuse_existing=args.reuse_existing)

    permissive_id = f"sisfall_confirm_permissive__{ts}"
    permissive_cmd = [
        sys.executable,
        "scripts/run_fall_threshold_baseline.py",
        "--dataset",
        "sisfall",
        "--path",
        str(dataset_path),
        "--sample-limit",
        "0",
        "--run-id",
        permissive_id,
        "--impact-threshold",
        "490",
        "--post-dyn-ratio-threshold",
        "1000000000",
        "--post-var-threshold",
        "1000000000",
        "--post-motion-threshold",
        "1000000000",
        "--post-motion-ratio-threshold",
        "1000000000",
        "--skip-plots",
    ]
    results["runs"]["confirm_permissive"] = _maybe_run(
        permissive_cmd,
        RESULTS_ROOT / permissive_id,
        reuse_existing=args.reuse_existing,
    )

    strict_id = f"sisfall_confirm_strict__{ts}"
    strict_cmd = [
        sys.executable,
        "scripts/run_fall_threshold_baseline.py",
        "--dataset",
        "sisfall",
        "--path",
        str(dataset_path),
        "--sample-limit",
        "0",
        "--run-id",
        strict_id,
        "--impact-threshold",
        "490",
        "--post-dyn-ratio-threshold",
        "0.1",
        "--post-var-threshold",
        "0.1",
        "--post-motion-threshold",
        "0.1",
        "--post-motion-ratio-threshold",
        "0.1",
        "--skip-plots",
    ]
    results["runs"]["confirm_strict"] = _maybe_run(strict_cmd, RESULTS_ROOT / strict_id, reuse_existing=args.reuse_existing)

    var_only_id = f"fall_threshold_sweep_sisfall_VAR_ONLY__{ts}"
    var_only_cmd = [
        sys.executable,
        "scripts/sweep_fall_thresholds.py",
        "--dataset",
        "sisfall",
        "--path",
        str(dataset_path),
        "--run-id",
        var_only_id,
        "--impact-thresholds",
        "490,1470",
        "--confirm-post-dyn-ratio-mean-max",
        "1000000000",
        "--confirm-post-var-max",
        "25,50,100,500",
        "--jerk-thresholds",
        "0,30",
        "--random-state",
        "42",
        "--test-size",
        "0.3",
        "--skip-plots",
    ]
    results["runs"]["sweep_var_only"] = _maybe_run(
        var_only_cmd, RESULTS_ROOT / var_only_id, reuse_existing=args.reuse_existing
    )

    ratio_tuned_id = f"fall_threshold_sweep_sisfall_RATIO_TUNED__{ts}"
    ratio_tuned_cmd = [
        sys.executable,
        "scripts/sweep_fall_thresholds.py",
        "--dataset",
        "sisfall",
        "--path",
        str(dataset_path),
        "--run-id",
        ratio_tuned_id,
        "--impact-thresholds",
        "490,1470,1924,2922",
        "--confirm-post-dyn-ratio-mean-max",
        "100,200,300,400,500,700",
        "--confirm-post-var-max",
        "25,50,100,250",
        "--jerk-thresholds",
        "0,30,60",
        "--random-state",
        "42",
        "--test-size",
        "0.3",
        "--skip-plots",
    ]
    results["runs"]["sweep_ratio_tuned"] = _maybe_run(
        ratio_tuned_cmd, RESULTS_ROOT / ratio_tuned_id, reuse_existing=args.reuse_existing
    )

    report_cmd = [
        sys.executable,
        "scripts/lesson7_report.py",
        "--sisfall-validation-run-dir",
        str(validation_run_dir),
        "--sisfall-sweep-run-dirs",
        str(RESULTS_ROOT / var_only_id),
        str(RESULTS_ROOT / ratio_tuned_id),
        "--output-json",
        str(DEFAULT_REPORT_JSON),
        "--output-md",
        str(DEFAULT_REPORT_MD),
    ]
    results["runs"]["report"] = _run_cmd(report_cmd)

    print("How to run:")
    print("- python scripts/run_lesson7_sisfall_experiments.py")
    print("- python scripts/lesson7_report.py --help")
    print(f"- outputs: {DEFAULT_REPORT_JSON} and {DEFAULT_REPORT_MD}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
