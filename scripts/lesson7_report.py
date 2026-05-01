#!/usr/bin/env python3
"""Generate a SisFall Chapter 5 report from existing run artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "runs"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "results" / "validation" / "lesson7_sisfall_report.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "results" / "validation" / "lesson7_sisfall_report.md"

QUANTILE_LEVELS = (0.50, 0.75, 0.90, 0.95, 0.99)
QUANTILE_KEYS = tuple(f"q{int(level * 100):02d}" for level in QUANTILE_LEVELS)
DEFAULT_QUANTILE_FEATURES = (
    "post_impact_dyn_ratio_mean",
    "post_impact_dyn_ratio_rms",
    "post_impact_variance",
    "peak_acc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Chapter 5 SisFall report JSON/Markdown.")
    parser.add_argument("--sisfall-validation-run-dir", default=None)
    parser.add_argument("--sisfall-sweep-run-dirs", nargs="*", default=None)
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    return parser.parse_args()


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, default=str) + "\n", encoding="utf-8")


def _parse_timestamp(name: str) -> datetime | None:
    match = re.search(r"(\\d{8}T\\d{6}Z)", name)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ")
        except ValueError:
            return None
    match = re.search(r"(\\d{8})", name)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d")
        except ValueError:
            return None
    return None


def _pick_newest_dir(candidates: list[Path]) -> Path | None:
    if not candidates:
        return None
    with_ts = [(path, _parse_timestamp(path.name)) for path in candidates]
    ts_candidates = [(path, ts) for path, ts in with_ts if ts is not None]
    if ts_candidates:
        return max(ts_candidates, key=lambda item: item[1])[0]
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _resolve_run_dir(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if path.is_file():
        return path.parent
    return path


def _glob_dirs(pattern: str) -> list[Path]:
    return [path for path in DEFAULT_RESULTS_ROOT.glob(pattern) if path.is_dir()]


def _filter_dirs_with_file(candidates: list[Path], filename: str) -> list[Path]:
    return [path for path in candidates if (path / filename).exists()]


def _normalize_labels(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace({"non-fall": "non_fall", "nonfall": "non_fall", "adl": "non_fall"})
    )


def _empty_quantiles() -> dict[str, float | None]:
    return {key: None for key in QUANTILE_KEYS}


def _quantiles(values: pd.Series) -> dict[str, float | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return _empty_quantiles()
    return {f"q{int(level * 100):02d}": float(numeric.quantile(level)) for level in QUANTILE_LEVELS}


def _compute_quantiles(df: pd.DataFrame, features: tuple[str, ...]) -> dict[str, Any]:
    if "true_label" not in df.columns:
        raise ValueError("Validation CSV missing true_label column.")
    labels = _normalize_labels(df["true_label"])
    output: dict[str, Any] = {"features": {}, "counts": {}}
    output["counts"]["overall"] = int(len(df))
    output["counts"]["fall"] = int((labels == "fall").sum())
    output["counts"]["non_fall"] = int((labels == "non_fall").sum())

    for feature in features:
        if feature not in df.columns:
            continue
        output["features"][feature] = {
            "fall": _quantiles(df.loc[labels == "fall", feature]),
            "non_fall": _quantiles(df.loc[labels == "non_fall", feature]),
            "overall": _quantiles(df[feature]),
        }
    return output


def _load_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _best_row(df: pd.DataFrame, sort_cols: list[str], *, ascending: list[bool]) -> dict[str, Any] | None:
    if df.empty:
        return None
    return df.sort_values(sort_cols, ascending=ascending).iloc[0].to_dict()


def _select_balanced(df: pd.DataFrame) -> tuple[dict[str, Any] | None, str]:
    if df.empty:
        return None, "No sweep results available."
    filtered = df.loc[(df["specificity"] >= 0.75) & (df["sensitivity"] >= 0.20)]
    if not filtered.empty:
        return (
            _best_row(filtered, ["f1", "sensitivity"], ascending=[False, False]),
            "Selected config with specificity>=0.75 and sensitivity>=0.20 (best f1).",
        )
    return _best_row(df, ["f1", "sensitivity"], ascending=[False, False]), "No config met specificity>=0.75 & sensitivity>=0.20; chose best f1."


def _select_conservative(df: pd.DataFrame) -> tuple[dict[str, Any] | None, str]:
    if df.empty:
        return None, "No sweep results available."
    return _best_row(df, ["specificity", "f1"], ascending=[False, False]), "Selected highest specificity (tie-break by f1)."


def _select_aggressive(df: pd.DataFrame) -> tuple[dict[str, Any] | None, str]:
    if df.empty:
        return None, "No sweep results available."
    return _best_row(df, ["sensitivity", "f1"], ascending=[False, False]), "Selected highest sensitivity (tie-break by f1)."


def _summarize_sweep(results_df: pd.DataFrame) -> dict[str, Any]:
    if results_df.empty:
        return {
            "max_sensitivity": None,
            "max_f1": None,
            "best_by_f1": None,
            "best_by_sensitivity": None,
        }
    return {
        "max_sensitivity": float(results_df["sensitivity"].max()),
        "max_f1": float(results_df["f1"].max()),
        "best_by_f1": _best_row(results_df, ["f1", "sensitivity"], ascending=[False, False]),
        "best_by_sensitivity": _best_row(results_df, ["sensitivity", "f1"], ascending=[False, False]),
    }


def _format_table(rows: list[list[str]], header: list[str]) -> str:
    col_widths = [len(h) for h in header]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))
    lines = []
    header_line = " | ".join(h.ljust(col_widths[idx]) for idx, h in enumerate(header))
    sep_line = " | ".join("-" * col_widths[idx] for idx in range(len(header)))
    lines.append(header_line)
    lines.append(sep_line)
    for row in rows:
        lines.append(" | ".join(row[idx].ljust(col_widths[idx]) for idx in range(len(header))))
    return "\n".join(lines)


def _fmt_value(value: float | None) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    value = float(value)
    if abs(value) >= 1000:
        return f"{value:,.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _build_markdown(report: dict[str, Any]) -> str:
    quantiles = report.get("quantiles", {})
    quant_rows: list[list[str]] = []
    for feature, by_label in (quantiles.get("features") or {}).items():
        for label in ("fall", "non_fall", "overall"):
            stats = (by_label or {}).get(label, {})
            quant_rows.append(
                [
                    label,
                    feature,
                    _fmt_value(stats.get("q50")),
                    _fmt_value(stats.get("q75")),
                    _fmt_value(stats.get("q90")),
                    _fmt_value(stats.get("q95")),
                    _fmt_value(stats.get("q99")),
                ]
            )

    quant_table = _format_table(
        quant_rows,
        ["label", "feature", "q50", "q75", "q90", "q95", "q99"],
    )

    rec_rows: list[list[str]] = []
    rec = report.get("recommended_configs", {})
    for key in ("conservative", "balanced", "aggressive"):
        entry = rec.get(key) or {}
        config = entry.get("config") or {}
        rec_rows.append(
            [
                key,
                str(entry.get("source_sweep") or "-"),
                str(config.get("impact_threshold") or "-"),
                str(config.get("confirm_post_dyn_ratio_mean_max") or "-"),
                str(config.get("confirm_post_var_max") or "-"),
                str(config.get("jerk_threshold") or "-"),
                _fmt_value(config.get("sensitivity")),
                _fmt_value(config.get("specificity")),
                _fmt_value(config.get("f1")),
            ]
        )
    rec_table = _format_table(
        rec_rows,
        [
            "type",
            "source",
            "impact_thr",
            "ratio_max",
            "var_max",
            "jerk_thr",
            "sens",
            "spec",
            "f1",
        ],
    )

    artifact_lines = []
    for key, value in (report.get("artifacts") or {}).items():
        artifact_lines.append(f"- {key}: {value}")

    return "\n".join(
        [
            "# Chapter 5 SisFall Report",
            "",
            "## Quantiles",
            "",
            quant_table,
            "",
            "## Recommended Configs",
            "",
            rec_table,
            "",
            "## Artifacts",
            "",
            *artifact_lines,
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    errors: list[str] = []

    validation_dir = _resolve_run_dir(args.sisfall_validation_run_dir)
    if validation_dir is None:
        validation_candidates = _glob_dirs("lesson7_fall_validation_sisfall__*")
        validation_candidates = _filter_dirs_with_file(validation_candidates, "test_predictions_windows.csv")
        validation_dir = _pick_newest_dir(validation_candidates)
    if validation_dir is None:
        errors.append("No validation run directories found for SisFall.")
    elif not validation_dir.exists():
        errors.append(f"Validation run dir not found: {validation_dir}")

    sweep_dirs: list[Path] = []
    if args.sisfall_sweep_run_dirs:
        for value in args.sisfall_sweep_run_dirs:
            resolved = _resolve_run_dir(value)
            if resolved is None or not resolved.exists():
                errors.append(f"Sweep run dir not found: {value}")
            else:
                sweep_dirs.append(resolved)
    else:
        for pattern in ("fall_threshold_sweep_sisfall_VAR_ONLY__*", "fall_threshold_sweep_sisfall_RATIO_TUNED__*"):
            candidates = _filter_dirs_with_file(_glob_dirs(pattern), "threshold_sweep_results.csv")
            newest = _pick_newest_dir(candidates)
            if newest is not None:
                sweep_dirs.append(newest)

    if not sweep_dirs:
        errors.append("No sweep run directories found for SisFall.")

    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        return 1

    validation_csv = validation_dir / "test_predictions_windows.csv"
    if not validation_csv.exists():
        print(f"ERROR: validation CSV not found: {validation_csv}")
        return 1

    validation_df = pd.read_csv(validation_csv)
    quantiles = _compute_quantiles(validation_df, DEFAULT_QUANTILE_FEATURES)

    dataset_stats: dict[str, Any] = {
        "validation_run_dir": str(validation_dir),
        "rows": int(len(validation_df)),
        "columns": list(validation_df.columns),
    }
    metrics_payload = _load_metrics(validation_dir / "metrics.json")
    if metrics_payload:
        dataset_stats["preprocessing_summary"] = metrics_payload.get("preprocessing_summary", {})
        dataset_stats["split_summary"] = metrics_payload.get("split", {})
        dataset_stats["metrics"] = metrics_payload.get("metrics", {})

    baseline_names = [
        "sisfall_confirm_permissive",
        "sisfall_confirm_strict",
        "sisfall_debug_impact_only",
    ]
    baseline_metrics: dict[str, Any] = {}
    for name in baseline_names:
        baseline_dir = DEFAULT_RESULTS_ROOT / name
        if not baseline_dir.exists():
            candidates = _glob_dirs(f"{name}__*")
            baseline_dir = _pick_newest_dir(candidates) if candidates else baseline_dir
        if baseline_dir and baseline_dir.exists():
            payload = _load_metrics(baseline_dir / "metrics.json")
            if payload and payload.get("metrics"):
                baseline_metrics[name] = {
                    "run_dir": str(baseline_dir),
                    "metrics": payload.get("metrics"),
                }

    sweep_summaries: list[dict[str, Any]] = []
    combined_results: list[pd.DataFrame] = []
    for sweep_dir in sweep_dirs:
        results_csv = sweep_dir / "threshold_sweep_results.csv"
        if not results_csv.exists():
            print(f"ERROR: sweep CSV missing: {results_csv}")
            return 1
        results_df = pd.read_csv(results_csv)
        summary = _summarize_sweep(results_df)
        sweep_summaries.append(
            {
                "run_dir": str(sweep_dir),
                "results_csv": str(results_csv),
                "results_json": str(sweep_dir / "threshold_sweep_results.json"),
                "summary": summary,
            }
        )
        combined_results.append(results_df)

    combined_df = pd.concat(combined_results, ignore_index=True) if combined_results else pd.DataFrame()

    conservative_cfg, conservative_note = _select_conservative(combined_df)
    balanced_cfg, balanced_note = _select_balanced(combined_df)
    aggressive_cfg, aggressive_note = _select_aggressive(combined_df)

    def _annotate_source(cfg: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str | None]:
        if cfg is None:
            return None, None
        config_id = cfg.get("config_id")
        for sweep in sweep_summaries:
            csv_path = Path(sweep["results_csv"])
            if not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            if "config_id" in df.columns and config_id in df["config_id"].values:
                return cfg, Path(sweep["run_dir"]).name
        return cfg, None

    conservative_cfg, conservative_source = _annotate_source(conservative_cfg)
    balanced_cfg, balanced_source = _annotate_source(balanced_cfg)
    aggressive_cfg, aggressive_source = _annotate_source(aggressive_cfg)

    report = {
        "report_name": "lesson7_sisfall_report",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "sisfall",
        "dataset_stats": dataset_stats,
        "quantiles": quantiles,
        "baseline_metrics": baseline_metrics,
        "sweep_summaries": sweep_summaries,
        "recommended_configs": {
            "conservative": {
                "config": conservative_cfg,
                "note": conservative_note,
                "source_sweep": conservative_source,
            },
            "balanced": {
                "config": balanced_cfg,
                "note": balanced_note,
                "source_sweep": balanced_source,
            },
            "aggressive": {
                "config": aggressive_cfg,
                "note": aggressive_note,
                "source_sweep": aggressive_source,
            },
        },
        "failure_explanation": "Wrong threshold scale (0–1) rejected all falls; SisFall confirm ratios are on a large scale.",
        "artifacts": {
            "validation_csv": str(validation_csv),
            "output_json": str(Path(args.output_json).resolve()),
            "output_md": str(Path(args.output_md).resolve()),
        },
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    _write_json(output_json, report)
    output_md.write_text(_build_markdown(report), encoding="utf-8")

    print(f"report_json={output_json}")
    print(f"report_md={output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
