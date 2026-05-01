#!/usr/bin/env python3
"""Summarize Chapter 5 threshold-sweep CSV results."""

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


REQUIRED_COLUMNS = (
    "config_id",
    "impact_threshold",
    "confirm_post_dyn_mean_max",
    "jerk_threshold",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "false_alarms_count",
)
NUMERIC_COLUMNS = (
    "impact_threshold",
    "confirm_post_dyn_mean_max",
    "jerk_threshold",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "false_alarms_count",
)
DISPLAY_COLUMNS_BASE = [
    "config_id",
    "impact_threshold",
    "confirm_post_dyn_mean_max",
    "jerk_threshold",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "false_alarms_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize threshold_sweep_results.csv")
    parser.add_argument("--csv", required=True, help="Path to threshold_sweep_results.csv")
    parser.add_argument("--min-sensitivity", type=float, default=0.30)
    parser.add_argument("--min-specificity", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out-json", default=None, help="Optional output JSON path")
    parser.add_argument("--out-md", default=None, help="Optional output markdown path")
    return parser.parse_args()


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def normalize_sweep_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "confirm_post_dyn_mean_max" not in out.columns and "confirm_post_motion_max" in out.columns:
        out["confirm_post_dyn_mean_max"] = out["confirm_post_motion_max"]
    return out


def _coerce_numeric(df: pd.DataFrame, columns: tuple[str, ...] = NUMERIC_COLUMNS) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Sweep CSV missing required columns: {missing}")


def get_display_columns(df: pd.DataFrame) -> list[str]:
    cols = DISPLAY_COLUMNS_BASE.copy()
    if "confirm_post_var_max" in df.columns and "confirm_post_var_max" not in cols:
        cols.insert(3, "confirm_post_var_max")
    return [name for name in cols if name in df.columns]


def top_configs_by_f1(results_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    df = _coerce_numeric(results_df)
    df = df.dropna(subset=["f1", "specificity", "sensitivity"])
    if df.empty:
        return df
    df = df.sort_values(
        ["f1", "specificity", "sensitivity", "false_alarms_count"],
        ascending=[False, False, False, True],
        kind="stable",
    )
    return df.head(max(0, int(top_k)))


def filter_balanced_candidates(
    results_df: pd.DataFrame,
    *,
    min_sensitivity: float,
    min_specificity: float,
) -> pd.DataFrame:
    df = _coerce_numeric(results_df)
    df = df.dropna(subset=["sensitivity", "specificity", "f1"])
    filtered = df[(df["sensitivity"] >= float(min_sensitivity)) & (df["specificity"] >= float(min_specificity))]
    if filtered.empty:
        return filtered
    return filtered.sort_values(
        ["f1", "sensitivity", "specificity", "false_alarms_count"],
        ascending=[False, False, False, True],
        kind="stable",
    )


def filter_conservative_candidates(
    results_df: pd.DataFrame,
    *,
    min_specificity: float,
) -> pd.DataFrame:
    df = _coerce_numeric(results_df)
    df = df.dropna(subset=["specificity", "false_alarms_count", "sensitivity", "f1"])
    filtered = df[df["specificity"] >= float(min_specificity)]
    if filtered.empty:
        return filtered
    return filtered.sort_values(
        ["false_alarms_count", "sensitivity", "specificity", "f1"],
        ascending=[True, False, False, False],
        kind="stable",
    )


def _fmt_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:.4f}"
    return str(value)


def print_section(title: str, df: pd.DataFrame, *, columns: list[str], top_k: int) -> None:
    print(f"\n{title} (rows={len(df)})")
    if df.empty:
        print("No rows matched.")
        return
    preview = df[columns].head(max(0, int(top_k))).copy()
    print(preview.to_string(index=False, formatters={col: _fmt_cell for col in preview.columns}))


def _records(df: pd.DataFrame, *, columns: list[str]) -> list[dict[str, Any]]:
    if df.empty:
        return []
    return df[columns].to_dict(orient="records")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
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


def _to_markdown_table(df: pd.DataFrame, *, columns: list[str], top_k: int) -> str:
    if df.empty:
        return "_No rows matched._"
    preview = df[columns].head(max(0, int(top_k))).copy()
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in preview.iterrows():
        rows.append("| " + " | ".join(_fmt_cell(row[col]) for col in columns) + " |")
    return "\n".join([header, sep] + rows)


def write_markdown_summary(
    out_path: Path,
    *,
    csv_path: Path,
    top_by_f1: pd.DataFrame,
    balanced: pd.DataFrame,
    conservative: pd.DataFrame,
    columns: list[str],
    min_sensitivity: float,
    min_specificity: float,
    top_k: int,
) -> None:
    lines = [
        "# Threshold Sweep Selection Summary",
        "",
        f"- Source CSV: `{csv_path}`",
        f"- Constraints: sensitivity >= {min_sensitivity:.3f}, specificity >= {min_specificity:.3f}",
        f"- Top-K shown per section: {int(top_k)}",
        "",
        "## Top Configs by F1",
        "",
        _to_markdown_table(top_by_f1, columns=columns, top_k=top_k),
        "",
        "## Balanced Candidates",
        "",
        f"Total matched rows: {len(balanced)}",
        "",
        _to_markdown_table(balanced, columns=columns, top_k=top_k),
        "",
        "## Conservative Candidates",
        "",
        f"Total matched rows: {len(conservative)}",
        "",
        _to_markdown_table(conservative, columns=columns, top_k=top_k),
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    csv_path = _resolve_path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: sweep CSV not found: {csv_path}")
        return 1

    sweep_df = normalize_sweep_columns(pd.read_csv(csv_path))
    try:
        validate_required_columns(sweep_df)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    top_by_f1 = top_configs_by_f1(sweep_df, top_k=args.top_k)
    balanced = filter_balanced_candidates(
        sweep_df,
        min_sensitivity=args.min_sensitivity,
        min_specificity=args.min_specificity,
    )
    conservative = filter_conservative_candidates(
        sweep_df,
        min_specificity=args.min_specificity,
    )
    columns = get_display_columns(sweep_df)

    print(f"sweep_csv={csv_path}")
    print(f"rows={len(sweep_df)}")
    print_section("Top configs by F1", top_by_f1, columns=columns, top_k=args.top_k)
    print_section("Balanced candidates", balanced, columns=columns, top_k=args.top_k)
    print_section("Conservative candidates", conservative, columns=columns, top_k=args.top_k)

    summary_payload = {
        "summary_name": "lesson7_threshold_sweep_summary",
        "csv_path": str(csv_path),
        "constraint_values": {
            "min_sensitivity": float(args.min_sensitivity),
            "min_specificity": float(args.min_specificity),
            "top_k": int(args.top_k),
        },
        "counts": {
            "total_rows": int(len(sweep_df)),
            "top_by_f1": int(len(top_by_f1)),
            "balanced_candidates": int(len(balanced)),
            "conservative_candidates": int(len(conservative)),
        },
        "top_by_f1": _records(top_by_f1, columns=columns),
        "balanced_candidates": _records(balanced, columns=columns),
        "conservative_candidates": _records(conservative, columns=columns),
    }

    if args.out_json:
        out_json = _resolve_path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(_json_safe(summary_payload), indent=2, default=str) + "\n", encoding="utf-8")
        print(f"saved_json={out_json}")

    if args.out_md:
        out_md = _resolve_path(args.out_md)
        write_markdown_summary(
            out_md,
            csv_path=csv_path,
            top_by_f1=top_by_f1,
            balanced=balanced,
            conservative=conservative,
            columns=columns,
            min_sensitivity=args.min_sensitivity,
            min_specificity=args.min_specificity,
            top_k=args.top_k,
        )
        print(f"saved_markdown={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
