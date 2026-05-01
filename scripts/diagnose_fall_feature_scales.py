#!/usr/bin/env python3
"""Diagnose feature-scale differences for fall vs non-fall windows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


QUANTILE_LEVELS = (0.50, 0.75, 0.90, 0.95, 0.99)
QUANTILE_KEYS = tuple(f"q{int(level * 100):02d}" for level in QUANTILE_LEVELS)
FEATURES_TO_PROFILE = (
    "peak_acc",
    "jerk_peak",
    "post_impact_dyn_mean",
    "post_impact_dyn_rms",
    "post_impact_dyn_ratio_mean",
    "post_impact_dyn_ratio_rms",
)
LABEL_ORDER = ("fall", "non_fall")
REQUIRED_COLUMNS = ("true_label", "peak_acc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose feature scales from prediction CSV artifacts")
    parser.add_argument("--predictions-csv", required=True, help="Path to test_predictions_windows.csv (or compatible prediction CSV)")
    parser.add_argument("--out-json", default=None, help="Optional output JSON path")
    return parser.parse_args()


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _normalize_label_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace(
            {
                "non-fall": "non_fall",
                "nonfall": "non_fall",
                "adl": "non_fall",
            }
        )
    )


def _empty_quantiles() -> dict[str, float | None]:
    return {key: None for key in QUANTILE_KEYS}


def _compute_quantiles(values: pd.Series) -> dict[str, float | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return _empty_quantiles()
    return {f"q{int(level * 100):02d}": float(numeric.quantile(level)) for level in QUANTILE_LEVELS}


def build_quantiles_by_label(
    df: pd.DataFrame,
    *,
    features: list[str] | None = None,
    labels: tuple[str, ...] = LABEL_ORDER,
) -> dict[str, dict[str, dict[str, float | None]]]:
    if "true_label" not in df.columns:
        raise ValueError("Input dataframe must include column: true_label")
    selected_features = [name for name in (features or list(FEATURES_TO_PROFILE)) if name in df.columns]
    normalized_label = _normalize_label_series(df["true_label"])

    output: dict[str, dict[str, dict[str, float | None]]] = {}
    for label in labels:
        subset = df.loc[normalized_label == label]
        output[label] = {feature: _compute_quantiles(subset[feature]) for feature in selected_features}
    return output


def _is_finite(value: Any) -> bool:
    return value is not None and isinstance(value, (float, int)) and math.isfinite(float(value))


def infer_peak_acc_unit_hint(fall_peak_quantiles: dict[str, float | None]) -> tuple[str, str]:
    p50 = fall_peak_quantiles.get("q50")
    p99 = fall_peak_quantiles.get("q99")
    if not _is_finite(p50):
        return "unknown", "Could not infer units because fall peak_acc quantiles are unavailable."

    p50f = float(p50)
    p99f = float(p99) if _is_finite(p99) else p50f
    if 0.5 <= p50f <= 4.0 and p99f < 20.0:
        return "g_like", "Fall peak_acc appears g-like (median around 1-3g)."
    if p50f >= 8.0 and p99f <= 120.0:
        return "m_s2_like", "Fall peak_acc appears m/s^2-like (magnitudes typically 10+)."
    if p50f >= 120.0 or p99f > 200.0:
        return "large_scale", "Fall peak_acc is very high; values are likely scaled m/s^2 or raw sensor units."
    return "unclear", "Peak_acc units are ambiguous; verify dataset preprocessing and accelerometer scaling."


def _integer_band(low: int, high: int, *, max_points: int = 12) -> list[int]:
    if high < low:
        return [low]
    span = high - low
    if span <= max_points:
        step = 1
    elif span <= max_points * 2:
        step = 2
    elif span <= max_points * 5:
        step = 5
    elif span <= max_points * 10:
        step = 10
    else:
        step = max(1, int(math.ceil(span / max_points)))

    out = list(range(low, high + 1, step))
    if out and out[-1] != high:
        out.append(high)
    if not out:
        out = [low]
    return out


def suggest_impact_thresholds(fall_peak_quantiles: dict[str, float | None]) -> tuple[list[float], str, list[str]]:
    seed_values = [fall_peak_quantiles.get("q75"), fall_peak_quantiles.get("q90"), fall_peak_quantiles.get("q95"), fall_peak_quantiles.get("q99")]
    rounded_seed = [int(round(float(value))) for value in seed_values if _is_finite(value)]
    unit_hint, unit_note = infer_peak_acc_unit_hint(fall_peak_quantiles)

    suggestions: list[float]
    notes = [unit_note]
    if not rounded_seed:
        notes.append("Suggested thresholds unavailable because fall peak_acc quantiles are missing.")
        return [], unit_hint, notes

    if unit_hint == "g_like":
        anchors = [fall_peak_quantiles.get("q50"), fall_peak_quantiles.get("q95"), fall_peak_quantiles.get("q99")]
        finite_anchors = [float(value) for value in anchors if _is_finite(value)]
        low_anchor = max(0.5, min(finite_anchors) - 1.0) if finite_anchors else max(0.5, min(rounded_seed) - 1.0)
        high_anchor = max(finite_anchors) + 1.0 if finite_anchors else float(max(rounded_seed) + 1)
        start = math.floor(low_anchor * 2.0) / 2.0
        stop = math.ceil(high_anchor * 2.0) / 2.0
        half_step_grid: list[float] = []
        current = start
        while current <= stop + 1e-9:
            half_step_grid.append(round(current, 2))
            current += 0.5
        suggestions = sorted({float(value) for value in rounded_seed + half_step_grid if value > 0})
        notes.append("Added +/-0.5 step g-like thresholds around fall quantiles.")
        return suggestions, unit_hint, notes

    low_quantile = fall_peak_quantiles.get("q75")
    high_quantile = fall_peak_quantiles.get("q99")
    low_int = int(math.floor(float(low_quantile))) if _is_finite(low_quantile) else min(rounded_seed)
    high_int = int(math.ceil(float(high_quantile))) if _is_finite(high_quantile) else max(rounded_seed)
    low_int = max(1, low_int)
    high_int = max(low_int, high_int)
    band = _integer_band(low_int, high_int)
    suggestions = sorted({float(int(value)) for value in (rounded_seed + band) if float(value) > 0})

    if unit_hint in {"m_s2_like", "large_scale"}:
        notes.append("Using integer thresholds for m/s^2-like scale.")
    return suggestions, unit_hint, notes


def _fmt_value(value: float | None) -> str:
    if value is None:
        return "-"
    numeric = float(value)
    magnitude = abs(numeric)
    if magnitude >= 1000:
        return f"{numeric:,.1f}"
    if magnitude >= 10:
        return f"{numeric:.2f}"
    return f"{numeric:.3f}"


def print_quantile_table(
    quantiles_by_label: dict[str, dict[str, dict[str, float | None]]],
    *,
    features: list[str],
    labels: tuple[str, ...] = LABEL_ORDER,
) -> None:
    header = ["label", "feature", "q50", "q75", "q90", "q95", "q99"]
    rows: list[list[str]] = []
    for label in labels:
        for feature in features:
            stats = quantiles_by_label.get(label, {}).get(feature, _empty_quantiles())
            rows.append(
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

    col_widths = [len(col) for col in header]
    for row in rows:
        for idx, item in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(item))

    def _join(items: list[str]) -> str:
        return "  ".join(item.ljust(col_widths[idx]) for idx, item in enumerate(items))

    print(_join(header))
    print(_join(["-" * width for width in col_widths]))
    for row in rows:
        print(_join(row))


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


def main() -> int:
    args = parse_args()
    predictions_csv = _resolve_path(args.predictions_csv)
    if not predictions_csv.exists():
        print(f"ERROR: predictions CSV not found: {predictions_csv}")
        return 1

    df = pd.read_csv(predictions_csv)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        print(f"ERROR: missing required columns in predictions CSV: {missing}")
        return 1

    features = [name for name in FEATURES_TO_PROFILE if name in df.columns]
    quantiles_by_label = build_quantiles_by_label(df, features=features)
    fall_peak_quantiles = quantiles_by_label.get("fall", {}).get("peak_acc", _empty_quantiles())
    suggestions, unit_hint, notes = suggest_impact_thresholds(fall_peak_quantiles)

    label_counts = _normalize_label_series(df["true_label"]).value_counts(dropna=False).to_dict()
    print(f"predictions_csv={predictions_csv}")
    print(f"rows={len(df)}")
    print(f"label_counts={label_counts}")
    print(f"profiled_features={features}")
    print("\nFeature quantiles by true_label:")
    print_quantile_table(quantiles_by_label, features=features)
    print("\nSuggested impact_thresholds for sweep:")
    print(suggestions)
    print(f"likely_peak_acc_units={unit_hint}")
    for note in notes:
        print(f"note: {note}")

    payload = {
        "analysis_name": "lesson7_feature_scale_diagnosis",
        "predictions_csv": str(predictions_csv),
        "rows": int(len(df)),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "quantiles_by_label": quantiles_by_label,
        "suggested_impact_thresholds": suggestions,
        "likely_peak_acc_units": unit_hint,
        "notes": notes,
    }

    if args.out_json:
        out_json = _resolve_path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(_json_safe(payload), indent=2, default=str) + "\n", encoding="utf-8")
        print(f"saved_json={out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
