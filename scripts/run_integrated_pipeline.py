#!/usr/bin/env python3
"""Atomic orchestrator for the integrated Fall + HAR pipeline.

Three subcommands, each a thin wrapper that delegates to an existing script
or to the canonical-layout registry — no training or evaluation logic lives
here:

- ``status``    reads ``artifacts/{fall,har}/current/metadata.json`` via the
                :mod:`pipeline.artifacts` registry and prints a side-by-side
                summary of the active model kind, metrics, and promotion
                timestamp.
- ``promote``   runs ``scripts/organize_fall_artifacts.py --current-kind`` and
                ``scripts/organize_har_artifacts.py --current-variant`` in
                sequence. Exits non-zero if either step fails.
- ``evaluate``  invokes ``scripts/run_combined_runtime_replay.py`` against a
                user-supplied source using the canonical current/ artifacts,
                writing outputs to ``results/validation/integrated/<ts>/``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.artifacts import (  # noqa: E402
    ArtifactTask,
    load_current_metadata,
    resolve_current_artifact,
)

SCRIPT_DIR = _REPO_ROOT / "scripts"
DEFAULT_INTEGRATED_RESULTS = _REPO_ROOT / "results" / "validation" / "integrated"
RETRAIN_SCRIPT = SCRIPT_DIR / "retrain_from_phone.py"


# ---------------------------------------------------------------------- status


def _status_rows(task: ArtifactTask) -> dict[str, Any]:
    try:
        artifact_path = resolve_current_artifact(task)
        metadata = load_current_metadata(task)
    except FileNotFoundError as exc:
        return {"task": task, "error": str(exc)}

    heldout = metadata.get("heldout") or {}
    validation = metadata.get("validation") or {}
    return {
        "task": task,
        "artifact_id": metadata.get("artifact_id"),
        "model_kind": metadata.get("model_kind"),
        "status": metadata.get("status"),
        "artifact_path": str(artifact_path),
        "promoted_utc": metadata.get("promoted_utc"),
        "selection_metric": metadata.get("selection_metric"),
        # Fall stores f1/f2, HAR stores overall_accuracy/macro_f1 — show both.
        "heldout_f1": heldout.get("f1"),
        "heldout_f2": heldout.get("f2"),
        "heldout_accuracy": heldout.get("overall_accuracy"),
        "heldout_macro_f1": heldout.get("macro_f1"),
        "validation_f1": validation.get("f1"),
        "validation_f2": validation.get("f2"),
        "validation_accuracy": validation.get("overall_accuracy"),
        "validation_macro_f1": validation.get("macro_f1"),
    }


def _format_status(row: dict[str, Any]) -> str:
    if row.get("error"):
        return f"  {row['task']}: ERROR — {row['error']}"
    lines = [
        f"  {row['task']}:",
        f"    artifact_id       : {row.get('artifact_id')}",
        f"    model_kind        : {row.get('model_kind')}",
        f"    status            : {row.get('status')}",
        f"    promoted_utc      : {row.get('promoted_utc')}",
        f"    artifact_path     : {row.get('artifact_path')}",
    ]
    if row["task"] == "fall":
        lines += [
            f"    validation_f2     : {row.get('validation_f2')}",
            f"    heldout_f2        : {row.get('heldout_f2')}",
        ]
    else:
        lines += [
            f"    heldout_accuracy  : {row.get('heldout_accuracy')}",
            f"    heldout_macro_f1  : {row.get('heldout_macro_f1')}",
            f"    valid_accuracy    : {row.get('validation_accuracy')}",
            f"    valid_macro_f1    : {row.get('validation_macro_f1')}",
        ]
    return "\n".join(lines)


def cmd_status(args: argparse.Namespace) -> int:
    rows = [_status_rows("fall"), _status_rows("har")]
    exit_code = 1 if any(r.get("error") for r in rows) else 0
    if args.json:
        print(json.dumps(rows, indent=2))
        return exit_code

    print("Integrated pipeline status")
    for row in rows:
        print(_format_status(row))
    return exit_code


# --------------------------------------------------------------------- promote


def _run_subscript(args: list[str]) -> int:
    print(f"$ {' '.join(args)}")
    result = subprocess.run(args, cwd=str(_REPO_ROOT))
    return result.returncode


def cmd_promote(args: argparse.Namespace) -> int:
    steps = [
        [
            sys.executable,
            str(SCRIPT_DIR / "organize_fall_artifacts.py"),
            "--current-kind",
            args.fall_kind,
        ],
        [
            sys.executable,
            str(SCRIPT_DIR / "organize_har_artifacts.py"),
            "--current-variant",
            args.har_variant,
        ],
    ]
    for step in steps:
        rc = _run_subscript(step)
        if rc != 0:
            print(f"promote step failed (exit {rc}): {' '.join(step)}", file=sys.stderr)
            return rc
    print(
        f"Promoted fall={args.fall_kind} har={args.har_variant} "
        f"from {SCRIPT_DIR.relative_to(_REPO_ROOT)}"
    )
    return 0


# -------------------------------------------------------------------- evaluate


def cmd_evaluate(args: argparse.Namespace) -> int:
    # Fail fast if the canonical layout is incomplete.
    har_path = resolve_current_artifact("har")
    fall_path = resolve_current_artifact("fall")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_root) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_combined_runtime_replay.py"),
        "--input-source",
        args.input_source,
        "--input-path",
        str(args.input_path),
        "--har-artifact",
        str(har_path),
        "--fall-artifact",
        str(fall_path),
        "--timeline-tolerance-seconds",
        str(args.timeline_tolerance_seconds),
        "--har-out",
        str(out_dir / "har_predictions.csv"),
        "--fall-out",
        str(out_dir / "fall_predictions.csv"),
        "--combined-out",
        str(out_dir / "timeline.csv"),
        "--report-out",
        str(out_dir / "report.json"),
    ]
    if args.session_id:
        cmd += ["--session-id", args.session_id]
    if args.max_sessions is not None:
        cmd += ["--max-sessions", str(args.max_sessions)]
    if args.threshold_mode:
        cmd += ["--threshold-mode", args.threshold_mode]

    rc = _run_subscript(cmd)
    if rc != 0:
        return rc

    # Stamp a side-car pointer so operators can find the run without guessing.
    (out_dir / "evaluate_invocation.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "har_artifact": str(har_path),
                "fall_artifact": str(fall_path),
                "input_source": args.input_source,
                "input_path": str(args.input_path),
                "session_id": args.session_id,
                "max_sessions": args.max_sessions,
                "threshold_mode": args.threshold_mode,
                "timeline_tolerance_seconds": args.timeline_tolerance_seconds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Integrated evaluation -> {out_dir}")
    return 0


# -------------------------------------------------------------------- retrain


def cmd_retrain(args: argparse.Namespace) -> int:
    """Forward all passthrough args to scripts/retrain_from_phone.py.

    This is a thin wrapper — every flag the orchestrator exposes is reachable
    here by putting it after ``--`` on the command line, e.g.::

        run_integrated_pipeline.py retrain -- --server-url http://host:8000
    """
    cmd = [sys.executable, str(RETRAIN_SCRIPT), *args.passthrough]
    return _run_subscript(cmd)


# --------------------------------------------------------------------- wiring


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_p = subparsers.add_parser("status", help="Print active fall + HAR artifact summary")
    status_p.add_argument("--json", action="store_true", help="Emit JSON instead of the plain summary")
    status_p.set_defaults(func=cmd_status)

    promote_p = subparsers.add_parser(
        "promote",
        help="Promote a specific fall kind + HAR variant as the new current/ artifacts",
    )
    promote_p.add_argument("--fall-kind", default="xgb", choices=["hgb", "xgb", "rf"])
    promote_p.add_argument(
        "--har-variant",
        default="movement_v2",
        choices=[
            "movement_v2",
            "pamap2_shared_rf_balanced",
            "pamap2_shared_rf_balanced_reg",
        ],
    )
    promote_p.set_defaults(func=cmd_promote)

    evaluate_p = subparsers.add_parser(
        "evaluate",
        help="Run combined HAR + fall replay against a source; outputs land under results/validation/integrated/<ts>/",
    )
    evaluate_p.add_argument(
        "--input-source",
        required=True,
        choices=["csv", "phone_folder", "ucihar", "pamap2", "mobifall", "sisfall"],
    )
    evaluate_p.add_argument("--input-path", required=True)
    evaluate_p.add_argument("--session-id", default=None)
    evaluate_p.add_argument("--max-sessions", type=int, default=None)
    evaluate_p.add_argument(
        "--threshold-mode",
        choices=["shared", "dataset_presets"],
        default="shared",
    )
    evaluate_p.add_argument("--timeline-tolerance-seconds", type=float, default=1.0)
    evaluate_p.add_argument(
        "--output-root",
        default=str(DEFAULT_INTEGRATED_RESULTS),
        help="Parent directory for the timestamped run folder",
    )
    evaluate_p.set_defaults(func=cmd_evaluate)

    retrain_p = subparsers.add_parser(
        "retrain",
        help="Pull labelled phone recordings from the API and run the retrain pipeline",
    )
    retrain_p.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to scripts/retrain_from_phone.py (place after `--`)",
    )
    retrain_p.set_defaults(func=cmd_retrain)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())