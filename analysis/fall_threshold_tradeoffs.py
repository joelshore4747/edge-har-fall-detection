"""Threshold sweep trade-off helpers for Chapter 5 tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


RESULT_SORT_COLUMNS = ["f1", "sensitivity", "specificity", "precision", "accuracy"]


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def best_config_by_f1(results_df: pd.DataFrame) -> dict[str, Any] | None:
    if results_df.empty or "f1" not in results_df.columns:
        return None
    df = _coerce_numeric(results_df, ["f1", "specificity", "sensitivity", "false_alarms_count"]) \
        .sort_values(["f1", "specificity", "sensitivity"], ascending=[False, False, False], kind="stable")
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def best_config_by_sensitivity_with_specificity_floor(
    results_df: pd.DataFrame,
    *,
    specificity_floor: float,
) -> dict[str, Any] | None:
    if results_df.empty:
        return None
    df = _coerce_numeric(results_df, ["sensitivity", "specificity", "f1", "false_alarms_count"]) \
        .dropna(subset=["sensitivity", "specificity"])
    df = df[df["specificity"] >= float(specificity_floor)]
    if df.empty:
        return None
    df = df.sort_values(
        ["sensitivity", "f1", "false_alarms_count"],
        ascending=[False, False, True],
        kind="stable",
    )
    return df.iloc[0].to_dict()


def best_config_by_false_alarms_with_sensitivity_floor(
    results_df: pd.DataFrame,
    *,
    sensitivity_floor: float,
) -> dict[str, Any] | None:
    if results_df.empty:
        return None
    df = _coerce_numeric(results_df, ["false_alarms_count", "sensitivity", "specificity", "f1"]) \
        .dropna(subset=["false_alarms_count", "sensitivity"])
    df = df[df["sensitivity"] >= float(sensitivity_floor)]
    if df.empty:
        return None
    df = df.sort_values(
        ["false_alarms_count", "specificity", "f1"],
        ascending=[True, False, False],
        kind="stable",
    )
    return df.iloc[0].to_dict()


def save_tradeoff_plot(
    results_df: pd.DataFrame,
    out_path: str | Path,
    *,
    x_metric: str = "false_alarms_count",
    y_metric: str = "sensitivity",
    title: str = "Threshold Sweep Trade-off",
) -> Path:
    """Save a simple scatter plot for sweep trade-offs.

    Raises ImportError if matplotlib is unavailable so callers can decide whether
    plotting should be skipped.
    """
    if results_df.empty:
        raise ValueError("Cannot plot trade-offs for an empty results dataframe")
    if x_metric not in results_df.columns or y_metric not in results_df.columns:
        raise ValueError(f"Missing plotting columns: {x_metric}, {y_metric}")

    import matplotlib.pyplot as plt

    df = _coerce_numeric(results_df, [x_metric, y_metric, "f1"]).copy()
    df = df.dropna(subset=[x_metric, y_metric])
    if df.empty:
        raise ValueError("No numeric rows available for plotting")

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    scatter = ax.scatter(
        df[x_metric],
        df[y_metric],
        c=df["f1"] if "f1" in df.columns else None,
        cmap="viridis",
        alpha=0.85,
        edgecolors="black",
        linewidths=0.3,
    )
    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(title)
    ax.grid(alpha=0.25)

    if "f1" in df.columns:
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("F1")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
