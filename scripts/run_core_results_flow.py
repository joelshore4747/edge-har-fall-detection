#!/usr/bin/env python3
"""Build the dissertation's primary fall-vs-vulnerability comparison artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _run_python_script(script_path: Path, *args: str) -> int:
    cmd = [sys.executable, str(script_path), *args]
    completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    return int(completed.returncode)


def _run_vulnerability_eval_if_requested(
    *,
    input_csv: Path | None,
    output_dir: Path | None,
    event_profile: str,
    vulnerability_profile: str,
    promote_to_vulnerable_after_steps: int,
    promote_to_high_risk_after_steps: int,
    recovery_cooldown_steps: int,
    clear_event_after_normal_steps: int,
) -> int:
    if input_csv is None or output_dir is None:
        return 0

    script_path = REPO_ROOT / "scripts" / "run_vulnerability_eval.py"
    return _run_python_script(
        script_path,
        "--input-csv",
        str(input_csv),
        "--output-dir",
        str(output_dir),
        "--event-profile",
        event_profile,
        "--vulnerability-profile",
        vulnerability_profile,
        "--promote-to-vulnerable-after-steps",
        str(promote_to_vulnerable_after_steps),
        "--promote-to-high-risk-after-steps",
        str(promote_to_high_risk_after_steps),
        "--recovery-cooldown-steps",
        str(recovery_cooldown_steps),
        "--clear-event-after-normal-steps",
        str(clear_event_after_normal_steps),
    )


def _run_results_table(*, results_root: Path, output_dir: Path) -> int:
    script_path = REPO_ROOT / "scripts" / "build_results_table.py"
    return _run_python_script(
        script_path,
        "--results-root",
        str(results_root),
        "--output-dir",
        str(output_dir),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the dissertation's standard reporting flow. "
            "This flow treats threshold-only fall detection vs fused vulnerability "
            "assessment as the primary comparison."
        )
    )
    parser.add_argument(
        "--results-root",
        default="results/runs",
        help="Root directory containing threshold, meta-model, and vulnerability run artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/reports",
        help="Directory where consolidated report artifacts will be written.",
    )
    parser.add_argument(
        "--input-csv",
        default=None,
        help=(
            "Optional meta-model prediction CSV to evaluate with run_vulnerability_eval.py "
            "before building the report."
        ),
    )
    parser.add_argument(
        "--vulnerability-output-dir",
        default=None,
        help=(
            "Optional output directory for vulnerability evaluation artifacts. "
            "Required when --input-csv is provided."
        ),
    )
    parser.add_argument("--event-profile", choices=["balanced", "conservative"], default="balanced")
    parser.add_argument("--vulnerability-profile", choices=["balanced", "conservative"], default="balanced")
    parser.add_argument("--promote-to-vulnerable-after-steps", type=int, default=2)
    parser.add_argument("--promote-to-high-risk-after-steps", type=int, default=2)
    parser.add_argument("--recovery-cooldown-steps", type=int, default=2)
    parser.add_argument("--clear-event-after-normal-steps", type=int, default=2)
    args = parser.parse_args()

    results_root = _resolve_path(args.results_root)
    output_dir = _resolve_path(args.output_dir)
    input_csv = _resolve_path(args.input_csv) if args.input_csv else None
    vulnerability_output_dir = (
        _resolve_path(args.vulnerability_output_dir) if args.vulnerability_output_dir else None
    )

    if input_csv is not None and vulnerability_output_dir is None:
        raise ValueError("--vulnerability-output-dir is required when --input-csv is provided")

    if input_csv is not None:
        rc = _run_vulnerability_eval_if_requested(
            input_csv=input_csv,
            output_dir=vulnerability_output_dir,
            event_profile=str(args.event_profile),
            vulnerability_profile=str(args.vulnerability_profile),
            promote_to_vulnerable_after_steps=int(args.promote_to_vulnerable_after_steps),
            promote_to_high_risk_after_steps=int(args.promote_to_high_risk_after_steps),
            recovery_cooldown_steps=int(args.recovery_cooldown_steps),
            clear_event_after_normal_steps=int(args.clear_event_after_normal_steps),
        )
        if rc != 0:
            return rc

    return _run_results_table(results_root=results_root, output_dir=output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
