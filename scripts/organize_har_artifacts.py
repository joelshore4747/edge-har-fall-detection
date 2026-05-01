#!/usr/bin/env python3
"""Create the canonical HAR artifact layout from existing variant outputs.

Mirrors ``scripts/organize_fall_artifacts.py``: produces
``artifacts/har/{current,candidates,archive}/`` populated from the existing
flat variants directory and the matching evaluation reports under
``results/validation/``. Leaves the ``movement_v2/lodo/`` evaluation artifacts
in place; they are not promotable candidates.
"""

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


# Variant name -> (source model filename in artifacts/har/<variant>/, evaluation report filename in results/validation/).
# Listed in the order they should appear in the comparison report.
KNOWN_VARIANTS: dict[str, dict[str, str]] = {
    "movement_v2": {
        "model_filename": "model.joblib",
        "report_filename": "har_movement_v2_combined.json",
    },
    "pamap2_shared_rf_balanced": {
        "model_filename": "har_rf_artifact.joblib",
        "report_filename": "har_movement_v2_lodo_pamap2_shared_rf_balanced.json",
    },
    "pamap2_shared_rf_balanced_reg": {
        "model_filename": "har_rf_artifact.joblib",
        "report_filename": "har_movement_v2_lodo_pamap2_shared_rf_balanced_reg.json",
    },
}

# Directories under artifacts/har/ that are NOT promotable variant sources.
NON_VARIANT_DIRS = {"current", "candidates", "archive"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current-variant",
        default="movement_v2",
        choices=list(KNOWN_VARIANTS),
        help="Which candidate becomes the active artifacts/har/current/ model.",
    )
    parser.add_argument("--artifact-root", default="artifacts/har")
    parser.add_argument("--legacy-report-dir", default="results/validation")
    parser.add_argument("--canonical-result-dir", default="results/validation/har")
    parser.add_argument(
        "--skip-gates",
        action="store_true",
        help="Promote even if the regression gates would refuse. Stamps an audit marker into metadata.",
    )
    parser.add_argument("--gate-heldout-tolerance", type=float, default=0.01)
    parser.add_argument("--gate-placement-tolerance", type=float, default=0.02)
    return parser.parse_args()


def _extract_metrics_block(metrics: dict[str, Any]) -> dict[str, Any]:
    if not metrics:
        return {
            "overall_accuracy": None,
            "macro_f1": None,
            "support_total": None,
            "per_label": {},
        }
    per_class = metrics.get("per_class") or {}
    per_label = {
        label: {
            "precision": stats.get("precision"),
            "recall": stats.get("recall"),
            "f1": stats.get("f1"),
            "support": stats.get("support"),
        }
        for label, stats in per_class.items()
    }
    return {
        "overall_accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "support_total": metrics.get("support_total"),
        "per_label": per_label,
    }


def _combined_internal_holdout(report: dict[str, Any]) -> dict[str, Any]:
    """Aggregate the per-dataset internal-holdout metrics into a single block.

    movement_v2 trains with ``holdout_mode=internal_group_split`` and exposes
    per-dataset internal-holdout metrics under
    ``per_dataset_metrics._internal_holdout_by_dataset``. LODO variants do not
    have an internal holdout; return an empty block in that case.
    """
    per_dataset = report.get("per_dataset_metrics", {}) or {}
    internal = per_dataset.get("_internal_holdout_by_dataset") or {}
    if not internal:
        return _extract_metrics_block({})

    weighted_acc_num = 0.0
    weighted_f1_num = 0.0
    total_support = 0
    aggregated_per_label: dict[str, dict[str, Any]] = {}
    for _dataset, payload in internal.items():
        m = payload.get("metrics", {}) or {}
        support = m.get("support_total") or 0
        if not support:
            continue
        acc = m.get("accuracy")
        f1 = m.get("macro_f1")
        if acc is not None:
            weighted_acc_num += float(acc) * support
        if f1 is not None:
            weighted_f1_num += float(f1) * support
        total_support += support

        for label, stats in (m.get("per_class") or {}).items():
            bucket = aggregated_per_label.setdefault(
                label,
                {"precision_num": 0.0, "recall_num": 0.0, "f1_num": 0.0, "support": 0},
            )
            sup = stats.get("support") or 0
            bucket["support"] += sup
            for key, src in (("precision_num", "precision"), ("recall_num", "recall"), ("f1_num", "f1")):
                v = stats.get(src)
                if v is not None:
                    bucket[key] += float(v) * sup

    per_label: dict[str, dict[str, Any]] = {}
    for label, bucket in aggregated_per_label.items():
        sup = bucket["support"]
        per_label[label] = {
            "precision": bucket["precision_num"] / sup if sup else None,
            "recall": bucket["recall_num"] / sup if sup else None,
            "f1": bucket["f1_num"] / sup if sup else None,
            "support": sup,
        }

    return {
        "overall_accuracy": weighted_acc_num / total_support if total_support else None,
        "macro_f1": weighted_f1_num / total_support if total_support else None,
        "support_total": total_support or None,
        "per_label": per_label,
    }


def _candidate_metadata(
    *,
    variant: str,
    model_path: Path,
    source_metadata_path: Path,
    report_path: Path,
    canonical_model_path: Path,
    canonical_report_path: Path,
    status: str,
) -> dict[str, Any]:
    source_metadata = _load_json(source_metadata_path) or {}
    report = _load_json(report_path) or {}

    training_config = source_metadata.get("training_config") or {}
    target_rate = source_metadata.get("target_rate_hz") or training_config.get("target_rate_hz")
    window_size = source_metadata.get("window_size") or training_config.get("window_size")
    step_size = source_metadata.get("step_size") or training_config.get("step_size")

    label_order = report.get("train", {}).get("artifact_label_order") or source_metadata.get("allowed_labels") or []

    heldout = _extract_metrics_block(report.get("metrics") or {})
    validation = _combined_internal_holdout(report)

    return {
        "artifact_id": f"har_{variant}",
        "model_kind": "rf",
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifact_path": str(model_path),
        "artifact_path": str(canonical_model_path),
        "source_report_path": str(report_path) if report_path.exists() else None,
        "report_path": str(canonical_report_path) if canonical_report_path.exists() else None,
        "selection_metric": "manual_current_baseline" if status == "current" else None,
        "target_rate_hz": target_rate,
        "window_size": window_size,
        "step_size": step_size,
        "label_order": list(label_order),
        "n_features": source_metadata.get("n_features"),
        "restrict_to_shared_labels": source_metadata.get("restrict_to_shared_labels"),
        "holdout_mode": source_metadata.get("holdout_mode"),
        "holdout_source": source_metadata.get("holdout_source"),
        "train_source": source_metadata.get("train_source"),
        "train_dataset_names": source_metadata.get("train_dataset_names"),
        "train_source_composition": source_metadata.get("train_source_composition"),
        "validation": validation,
        "heldout": heldout,
    }


def _write_comparison(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "har_variant_comparison.json"
    csv_path = out_dir / "har_variant_comparison.csv"
    md_path = out_dir / "har_variant_comparison.md"

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_note": (
            "The organizer uses --current-variant as an explicit temporary baseline. "
            "For future variant selection, rank by held-out macro_f1 first; "
            "validation metrics are the per-dataset internal holdout where available."
        ),
        "candidates": rows,
    }
    json_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")

    columns = [
        "variant",
        "status",
        "artifact_path",
        "target_rate_hz",
        "window_size",
        "validation_accuracy",
        "validation_macro_f1",
        "heldout_accuracy",
        "heldout_macro_f1",
        "holdout_mode",
        "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})

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
    canonical_result_dir = _resolve_path(args.canonical_result_dir)

    if not artifact_root.exists():
        raise SystemExit(f"Artifact root not found: {artifact_root}")

    discovered = sorted(
        name
        for name in KNOWN_VARIANTS
        if (artifact_root / name).is_dir()
    )
    if not discovered:
        raise SystemExit(f"No HAR variants found under {artifact_root}")

    candidate_rows: list[dict[str, Any]] = []

    for variant in discovered:
        config = KNOWN_VARIANTS[variant]
        variant_dir = artifact_root / variant
        source_model = variant_dir / config["model_filename"]
        source_metadata = variant_dir / "metadata.json"
        source_report = legacy_report_dir / config["report_filename"]
        source_predictions = legacy_report_dir / (
            config["report_filename"].rsplit(".json", 1)[0] + "_predictions.csv"
        )

        candidate_dir = artifact_root / "candidates" / variant
        candidate_model = candidate_dir / "model.joblib"
        candidate_metadata_path = candidate_dir / "metadata.json"
        canonical_report = canonical_result_dir / "candidates" / f"{variant}.json"
        canonical_predictions = canonical_result_dir / "candidates" / f"{variant}_predictions.csv"

        copied_model = _safe_copy(source_model, candidate_model)
        copied_report = _safe_copy(source_report, canonical_report)
        _safe_copy(source_predictions, canonical_predictions)

        if not copied_model and not source_metadata.exists():
            continue

        status = "current" if variant == args.current_variant else "candidate"
        metadata = _candidate_metadata(
            variant=variant,
            model_path=source_model,
            source_metadata_path=source_metadata,
            report_path=source_report,
            canonical_model_path=candidate_model,
            canonical_report_path=canonical_report,
            status=status,
        )
        candidate_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_metadata_path.write_text(
            json.dumps(_json_safe(metadata), indent=2),
            encoding="utf-8",
        )

        candidate_rows.append(
            {
                "variant": variant,
                "status": status,
                "artifact_path": str(candidate_model) if copied_model else "",
                "target_rate_hz": metadata["target_rate_hz"],
                "window_size": metadata["window_size"],
                "validation_accuracy": metadata["validation"]["overall_accuracy"],
                "validation_macro_f1": metadata["validation"]["macro_f1"],
                "heldout_accuracy": metadata["heldout"]["overall_accuracy"],
                "heldout_macro_f1": metadata["heldout"]["macro_f1"],
                "holdout_mode": metadata["holdout_mode"],
                "notes": "complete" if copied_model and copied_report else "incomplete",
            }
        )

    current_candidate_model = artifact_root / "candidates" / args.current_variant / "model.joblib"
    current_candidate_metadata = artifact_root / "candidates" / args.current_variant / "metadata.json"
    current_dir = artifact_root / "current"
    existing_current_metadata = _load_json(current_dir / "metadata.json")
    candidate_metadata_before = _load_json(current_candidate_metadata) or {}
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

    if current_candidate_model.exists() or current_candidate_metadata.exists():
        _safe_copy(current_candidate_model, current_dir / "model.joblib")
        current_metadata = _load_json(current_candidate_metadata) or {}
        current_metadata["status"] = "current"
        current_metadata["artifact_path"] = str(current_dir / "model.joblib")
        current_metadata["source_candidate_artifact_path"] = str(current_candidate_model)
        current_metadata["promoted_utc"] = datetime.now(timezone.utc).isoformat()
        current_metadata["gate_results"] = gate_results_dicts
        if args.skip_gates and not gates_passed(gate_results):
            current_metadata["promoted_with_skip_gates"] = True
        (current_dir / "metadata.json").write_text(
            json.dumps(_json_safe(current_metadata), indent=2),
            encoding="utf-8",
        )

    archive_dir = artifact_root / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    _write_comparison(candidate_rows, canonical_result_dir / "comparison")

    print(f"Canonical HAR artifact dir -> {artifact_root}")
    print(f"Canonical HAR result dir -> {canonical_result_dir}")
    print(f"Current variant -> {args.current_variant}")
    print(f"Discovered variants -> {', '.join(discovered)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
