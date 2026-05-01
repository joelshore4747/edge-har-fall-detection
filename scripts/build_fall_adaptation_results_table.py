#!/usr/bin/env python3
"""Build one clean comparison table for fall adaptation experiments.

Purpose:
- read frozen adaptation/baseline JSON outputs
- extract the most important phone/public metrics
- produce a single CSV you can use in the dissertation results chapter

Recommended rows:
- baseline public-only
- adapted negatives-only
- adapted negatives+positives
"""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fall adaptation results table")
    parser.add_argument(
        "--baseline-json",
        default="results/validation/fall_phone_adaptation_comparison_mobifall_eval.json",
        help="JSON for baseline + negatives-only comparison run",
    )
    parser.add_argument(
        "--negatives-threshold-json",
        default="results/validation/phone_fall_operating_point_adapted.json",
        help="Threshold tuning JSON for negatives-only adapted model",
    )
    parser.add_argument(
        "--with-positives-json",
        default="results/validation/fall_phone_adaptation_comparison_with_positives.json",
        help="JSON for negatives+positives comparison run",
    )
    parser.add_argument(
        "--with-positives-threshold-json",
        default="results/validation/phone_fall_operating_point_with_positives.json",
        help="Threshold tuning JSON for negatives+positives adapted model",
    )
    parser.add_argument(
        "--out-csv",
        default="results/validation/fall_adaptation_results_table.csv",
        help="Output CSV comparison table",
    )
    parser.add_argument(
        "--out-md",
        default="results/validation/fall_adaptation_results_table.md",
        help="Optional markdown table output",
    )
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    return json.loads(path.read_text())


def _get(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _format_fp_breakdown(d: dict[str, Any] | None) -> str:
    if not d:
        return ""
    return "; ".join(f"{k}:{v}" for k, v in d.items())


def _baseline_row(baseline_data: dict[str, Any]) -> dict[str, Any]:
    phone = _get(baseline_data, "phone_eval", "baseline", default={}) or {}
    public = _get(baseline_data, "public_eval", "baseline", default={}) or {}
    return {
        "variant": "baseline_public_only",
        "phone_f1": phone.get("f1"),
        "phone_precision": phone.get("precision"),
        "phone_recall": phone.get("sensitivity"),
        "phone_specificity": phone.get("specificity"),
        "phone_fp": phone.get("fp"),
        "phone_tp": phone.get("tp"),
        "phone_fn": phone.get("fn"),
        "phone_threshold": phone.get("probability_threshold"),
        "phone_fp_breakdown": _format_fp_breakdown(
            _get(baseline_data, "phone_eval", "baseline_false_positive_breakdown", default={}) or {}
        ),
        "public_eval_source": _get(baseline_data, "public_eval", "eval_source"),
        "public_f1": public.get("f1"),
        "public_precision": public.get("precision"),
        "public_recall": public.get("sensitivity"),
        "public_specificity": public.get("specificity"),
        "selected_runtime_threshold": "",
        "selected_runtime_f1": "",
        "selected_runtime_precision": "",
        "selected_runtime_recall": "",
        "selected_runtime_specificity": "",
        "selected_runtime_fp": "",
        "selected_runtime_tp": "",
        "selected_runtime_fp_breakdown": "",
        "note": "Public-only baseline before phone adaptation.",
    }


def _negatives_only_row(
    baseline_data: dict[str, Any],
    threshold_data: dict[str, Any],
) -> dict[str, Any]:
    phone = _get(baseline_data, "phone_eval", "adapted", default={}) or {}
    public = _get(baseline_data, "public_eval", "adapted", default={}) or {}
    best = _get(threshold_data, "best_metrics", default={}) or {}
    return {
        "variant": "adapted_negatives_only",
        "phone_f1": phone.get("f1"),
        "phone_precision": phone.get("precision"),
        "phone_recall": phone.get("sensitivity"),
        "phone_specificity": phone.get("specificity"),
        "phone_fp": phone.get("fp"),
        "phone_tp": phone.get("tp"),
        "phone_fn": phone.get("fn"),
        "phone_threshold": phone.get("probability_threshold"),
        "phone_fp_breakdown": _format_fp_breakdown(
            _get(baseline_data, "phone_eval", "adapted_false_positive_breakdown", default={}) or {}
        ),
        "public_eval_source": _get(baseline_data, "public_eval", "eval_source"),
        "public_f1": public.get("f1"),
        "public_precision": public.get("precision"),
        "public_recall": public.get("sensitivity"),
        "public_specificity": public.get("specificity"),
        "selected_runtime_threshold": _get(threshold_data, "best_threshold"),
        "selected_runtime_f1": best.get("f1"),
        "selected_runtime_precision": best.get("precision"),
        "selected_runtime_recall": best.get("sensitivity"),
        "selected_runtime_specificity": best.get("specificity"),
        "selected_runtime_fp": best.get("fp"),
        "selected_runtime_tp": best.get("tp"),
        "selected_runtime_fp_breakdown": _format_fp_breakdown(
            _get(threshold_data, "best_false_positive_breakdown", default={}) or {}
        ),
        "note": "Phone hard negatives only; strongest deployment-oriented tradeoff.",
    }


def _with_positives_row(
    with_pos_data: dict[str, Any],
    threshold_data: dict[str, Any],
) -> dict[str, Any]:
    phone = _get(with_pos_data, "phone_eval", "adapted", default={}) or {}
    public = _get(with_pos_data, "public_eval", "adapted", default={}) or {}
    best = _get(threshold_data, "best_metrics", default={}) or {}
    return {
        "variant": "adapted_negatives_plus_positives",
        "phone_f1": phone.get("f1"),
        "phone_precision": phone.get("precision"),
        "phone_recall": phone.get("sensitivity"),
        "phone_specificity": phone.get("specificity"),
        "phone_fp": phone.get("fp"),
        "phone_tp": phone.get("tp"),
        "phone_fn": phone.get("fn"),
        "phone_threshold": phone.get("probability_threshold"),
        "phone_fp_breakdown": _format_fp_breakdown(
            _get(with_pos_data, "phone_eval", "adapted_false_positive_breakdown", default={}) or {}
        ),
        "public_eval_source": _get(with_pos_data, "public_eval", "eval_source"),
        "public_f1": public.get("f1"),
        "public_precision": public.get("precision"),
        "public_recall": public.get("sensitivity"),
        "public_specificity": public.get("specificity"),
        "selected_runtime_threshold": _get(threshold_data, "best_threshold"),
        "selected_runtime_f1": best.get("f1"),
        "selected_runtime_precision": best.get("precision"),
        "selected_runtime_recall": best.get("sensitivity"),
        "selected_runtime_specificity": best.get("specificity"),
        "selected_runtime_fp": best.get("fp"),
        "selected_runtime_tp": best.get("tp"),
        "selected_runtime_fp_breakdown": _format_fp_breakdown(
            _get(threshold_data, "best_false_positive_breakdown", default={}) or {}
        ),
        "note": "Adds phone positives; recovers some recall but increases false alarms and slightly weakens public retention.",
    }


def _round_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(4)
    return out


def main() -> int:
    args = parse_args()

    baseline_json = _resolve(args.baseline_json)
    negatives_threshold_json = _resolve(args.negatives_threshold_json)
    with_positives_json = _resolve(args.with_positives_json)
    with_positives_threshold_json = _resolve(args.with_positives_threshold_json)
    out_csv = _resolve(args.out_csv)
    out_md = _resolve(args.out_md)

    baseline_data = _load_json(baseline_json)
    negatives_threshold_data = _load_json(negatives_threshold_json)
    with_pos_data = _load_json(with_positives_json)
    with_pos_threshold_data = _load_json(with_positives_threshold_json)

    rows = [
        _baseline_row(baseline_data),
        _negatives_only_row(baseline_data, negatives_threshold_data),
        _with_positives_row(with_pos_data, with_pos_threshold_data),
    ]

    df = pd.DataFrame(rows)
    df = _round_numeric_columns(df)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_csv, index=False)

    md_df = df[
        [
            "variant",
            "phone_f1",
            "phone_precision",
            "phone_recall",
            "phone_specificity",
            "phone_fp",
            "public_f1",
            "selected_runtime_threshold",
            "selected_runtime_f1",
            "note",
        ]
    ].copy()

    def _to_simple_markdown(table_df: pd.DataFrame) -> str:
        cols = list(table_df.columns)
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = []
        for _, row in table_df.iterrows():
            vals = []
            for col in cols:
                val = row[col]
                if pd.isna(val):
                    vals.append("")
                else:
                    vals.append(str(val))
            rows.append("| " + " | ".join(vals) + " |")
        return "\n".join([header, sep] + rows)

    out_md.write_text(_to_simple_markdown(md_df), encoding="utf-8")

    print(f"Saved results table CSV to: {out_csv}")
    print(f"Saved results table markdown to: {out_md}")
    print()
    print(df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())