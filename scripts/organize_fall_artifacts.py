#!/usr/bin/env python3
"""Create the canonical fall artifact layout from existing candidate outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._artifact_layout import (  # noqa: E402
    json_safe as _json_safe,
    load_json as _load_json,
    resolve_path as _resolve_path,
    safe_copy as _safe_copy,
)

from pipeline.artifacts.gates import gates_passed, run_gates  # noqa: E402


MODEL_KINDS = ("hgb", "xgb", "rf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-kind", default="xgb", choices=list(MODEL_KINDS))
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--legacy-report-dir", default="results/validation")
    parser.add_argument("--canonical-artifact-dir", default="artifacts/fall")
    parser.add_argument("--canonical-result-dir", default="results/validation/fall")
    parser.add_argument(
        "--update-runtime-copy",
        action="store_true",
        help="Also copy the current model to artifacts/fall_detector.joblib for compatibility.",
    )
    parser.add_argument(
        "--skip-gates",
        action="store_true",
        help="Promote even if the regression gates would refuse. Stamps an audit marker into metadata.",
    )
    parser.add_argument(
        "--gate-heldout-tolerance",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--gate-placement-tolerance",
        type=float,
        default=0.02,
    )
    return parser.parse_args()


def _fbeta_from_counts(metrics: dict[str, Any], beta: float = 2.0) -> float | None:
    if metrics.get("f2") is not None:
        return float(metrics["f2"])
    tp = metrics.get("tp")
    fp = metrics.get("fp")
    fn = metrics.get("fn")
    if tp is None or fp is None or fn is None:
        return None
    beta2 = beta * beta
    denom = ((1.0 + beta2) * float(tp)) + (beta2 * float(fn)) + float(fp)
    if denom == 0:
        return 0.0
    return ((1.0 + beta2) * float(tp)) / denom


def _candidate_metadata(
    *,
    kind: str,
    model_path: Path,
    report_path: Path,
    canonical_model_path: Path,
    canonical_report_path: Path,
    status: str,
) -> dict[str, Any]:
    report = _load_json(report_path) or {}
    threshold = None
    val_metrics: dict[str, Any] = {}
    heldout_metrics: dict[str, Any] = {}
    if report:
        threshold = report.get("artifact", {}).get("probability_threshold")
        val_metrics = report.get("threshold_tuning", {}).get("val_metrics", {})
        heldout_metrics = report.get("held_out_metrics", {}).get("combined", {})

    return {
        "artifact_id": f"fall_{kind}_candidate",
        "model_kind": kind,
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifact_path": str(model_path),
        "artifact_path": str(canonical_model_path),
        "source_report_path": str(report_path) if report_path.exists() else None,
        "report_path": str(canonical_report_path) if canonical_report_path.exists() else None,
        "selection_metric": "manual_current_baseline" if status == "current" else None,
        "probability_threshold": threshold,
        "validation": {
            "f1": val_metrics.get("f1"),
            "f2": _fbeta_from_counts(val_metrics),
            "recall": val_metrics.get("sensitivity"),
            "specificity": val_metrics.get("specificity"),
            "precision": val_metrics.get("precision"),
            "roc_auc": val_metrics.get("roc_auc"),
        },
        "heldout": {
            "f1": heldout_metrics.get("f1"),
            "f2": _fbeta_from_counts(heldout_metrics),
            "recall": heldout_metrics.get("sensitivity"),
            "specificity": heldout_metrics.get("specificity"),
            "precision": heldout_metrics.get("precision"),
            "roc_auc": heldout_metrics.get("roc_auc"),
            "false_positives": heldout_metrics.get("fp"),
            "false_negatives": heldout_metrics.get("fn"),
        },
    }


def _write_comparison(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "fall_detector_comparison.json"
    csv_path = out_dir / "fall_detector_comparison.csv"
    md_path = out_dir / "fall_detector_comparison.md"

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_note": (
            "The organizer uses --current-kind as an explicit temporary baseline. "
            "For future model selection, rank candidates by validation F2 first; "
            "held-out metrics are for reporting."
        ),
        "candidates": rows,
    }
    json_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")

    columns = [
        "model",
        "status",
        "artifact_path",
        "threshold",
        "validation_f1",
        "validation_f2",
        "heldout_f1",
        "heldout_f2",
        "recall",
        "specificity",
        "precision",
        "roc_auc",
        "false_positives",
        "false_negatives",
        "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    artifact_root = _resolve_path(args.artifact_root)
    legacy_report_dir = _resolve_path(args.legacy_report_dir)
    canonical_artifact_dir = _resolve_path(args.canonical_artifact_dir)
    canonical_result_dir = _resolve_path(args.canonical_result_dir)

    candidate_rows: list[dict[str, Any]] = []

    for kind in MODEL_KINDS:
        source_model = artifact_root / f"fall_detector_{kind}.joblib"
        source_report = legacy_report_dir / f"fall_artifact_eval_{kind}.json"
        source_predictions = legacy_report_dir / f"fall_artifact_eval_{kind}_predictions.csv"

        candidate_dir = canonical_artifact_dir / "candidates" / kind
        candidate_model = candidate_dir / "model.joblib"
        candidate_metadata = candidate_dir / "metadata.json"
        canonical_report = canonical_result_dir / "candidates" / f"{kind}.json"
        canonical_predictions = canonical_result_dir / "candidates" / f"{kind}_predictions.csv"

        copied_model = _safe_copy(source_model, candidate_model)
        copied_report = _safe_copy(source_report, canonical_report)
        _safe_copy(source_predictions, canonical_predictions)

        if not copied_model and not copied_report:
            continue

        status = "current" if kind == args.current_kind else "candidate"
        metadata = _candidate_metadata(
            kind=kind,
            model_path=source_model,
            report_path=source_report,
            canonical_model_path=candidate_model,
            canonical_report_path=canonical_report,
            status=status,
        )
        candidate_metadata.write_text(json.dumps(_json_safe(metadata), indent=2), encoding="utf-8")

        candidate_rows.append(
            {
                "model": kind,
                "status": status,
                "artifact_path": str(candidate_model) if copied_model else "",
                "threshold": metadata["probability_threshold"],
                "validation_f1": metadata["validation"]["f1"],
                "validation_f2": metadata["validation"]["f2"],
                "heldout_f1": metadata["heldout"]["f1"],
                "heldout_f2": metadata["heldout"]["f2"],
                "recall": metadata["heldout"]["recall"],
                "specificity": metadata["heldout"]["specificity"],
                "precision": metadata["heldout"]["precision"],
                "roc_auc": metadata["heldout"]["roc_auc"],
                "false_positives": metadata["heldout"]["false_positives"],
                "false_negatives": metadata["heldout"]["false_negatives"],
                "notes": "complete" if copied_model and copied_report else "incomplete",
            }
        )

    current_source = canonical_artifact_dir / "candidates" / args.current_kind / "model.joblib"
    current_metadata_source = canonical_artifact_dir / "candidates" / args.current_kind / "metadata.json"
    current_dir = canonical_artifact_dir / "current"
    existing_current_metadata = _load_json(current_dir / "metadata.json")
    candidate_metadata_before = _load_json(current_metadata_source) or {}
    gate_results = run_gates(
        candidate_metadata_before,
        existing_current_metadata,
        heldout_tolerance=args.gate_heldout_tolerance,
        placement_tolerance=args.gate_placement_tolerance,
    )
    gate_results_dicts = [r.to_dict() for r in gate_results]
    print("Promotion gates:")
    for r in gate_results:
        print(f"  - {r.name}: {'ok' if r.ok else 'FAIL'}")
        for reason in r.reasons:
            print(f"      {reason}")
    if not gates_passed(gate_results) and not args.skip_gates:
        print("Refusing promotion — one or more gates failed. Rerun with --skip-gates to override.")
        return 2

    if current_source.exists():
        _safe_copy(current_source, current_dir / "model.joblib")
        current_metadata = _load_json(current_metadata_source) or {}
        current_metadata["status"] = "current"
        current_metadata["artifact_path"] = str(current_dir / "model.joblib")
        current_metadata["source_candidate_artifact_path"] = str(current_source)
        current_metadata["promoted_utc"] = datetime.now(timezone.utc).isoformat()
        current_metadata["gate_results"] = gate_results_dicts
        if args.skip_gates and not gates_passed(gate_results):
            current_metadata["promoted_with_skip_gates"] = True
        (current_dir / "metadata.json").write_text(
            json.dumps(_json_safe(current_metadata), indent=2),
            encoding="utf-8",
        )
        if args.update_runtime_copy:
            _safe_copy(current_source, artifact_root / "fall_detector.joblib")

    pre_multi = artifact_root / "fall_detector.pre-multi-run.joblib"
    if pre_multi.exists():
        _safe_copy(pre_multi, canonical_artifact_dir / "archive" / "pre_multi_run" / "model.joblib")

    _write_comparison(candidate_rows, canonical_result_dir / "comparison")

    print(f"Canonical artifact dir -> {canonical_artifact_dir}")
    print(f"Canonical result dir -> {canonical_result_dir}")
    print(f"Current kind -> {args.current_kind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
