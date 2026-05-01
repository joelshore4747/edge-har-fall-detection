#!/usr/bin/env python3
"""Render the dissertation figure for HAR domain-adaptation deltas."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


METHOD_ORDER = [
    "rf_target_zscore",
    "rf_subject_zscore",
    "rf_coral",
    "rf_subspace_align",
    "rf_importance_weighted",
]

METHOD_LABELS = {
    "rf_target_zscore": "Target z-score",
    "rf_subject_zscore": "Subject z-score",
    "rf_coral": "CORAL",
    "rf_subspace_align": "Subspace Alignment",
    "rf_importance_weighted": "Importance weighting",
}

DIRECTION_LABELS = {
    "UCIHAR_to_PAMAP2": "UCIHAR -> PAMAP2",
    "PAMAP2_to_UCIHAR": "PAMAP2 -> UCIHAR",
}

METHOD_COLOURS = {
    "rf_target_zscore": "#2f6f9f",
    "rf_subject_zscore": "#8a8f98",
    "rf_coral": "#167c5c",
    "rf_subspace_align": "#c77718",
    "rf_importance_weighted": "#b64b4b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in-csv",
        default="results/validation/har_domain_adaptation_eval.csv",
        help="Summary CSV emitted by scripts/run_har_domain_adaptation_eval.py",
    )
    parser.add_argument(
        "--out-svg",
        default="results/validation/har_domain_adaptation_delta.svg",
        help="Canonical dissertation SVG output",
    )
    parser.add_argument(
        "--admin-svg",
        default="apps/admin/public/figures/har_domain_adaptation_delta.svg",
        help="Optional copy for the admin/static figure library",
    )
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _load_adaptation_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "direction",
        "method",
        "macro_f1",
        "delta_macro_f1_vs_baseline",
        "delta_macro_f1_ci_lower",
        "delta_macro_f1_ci_upper",
        "uses_target_labels",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise SystemExit(f"{path} is missing required columns: {', '.join(missing)}")

    adapted = df[df["method"].isin(METHOD_ORDER)].copy()
    if adapted.empty:
        raise SystemExit(f"{path} contains no adaptation rows")
    return adapted


def _axis_limit(df: pd.DataFrame) -> float:
    cols = [
        "delta_macro_f1_vs_baseline",
        "delta_macro_f1_ci_lower",
        "delta_macro_f1_ci_upper",
    ]
    max_abs = float(np.nanmax(np.abs(df[cols].to_numpy(dtype=float))))
    return max(0.08, np.ceil((max_abs * 1.12) / 0.05) * 0.05)


def render(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    directions = [direction for direction in DIRECTION_LABELS if direction in set(df["direction"])]
    x_limit = _axis_limit(df)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.titlesize": 14,
        }
    )
    fig, axes = plt.subplots(
        1,
        len(directions),
        figsize=(11.5, 5.4),
        sharex=True,
        constrained_layout=False,
    )
    if len(directions) == 1:
        axes = [axes]

    for ax, direction in zip(axes, directions):
        group = df[df["direction"] == direction].set_index("method")
        ordered_methods = [method for method in METHOD_ORDER if method in group.index]
        plot_df = group.loc[ordered_methods]
        y_pos = np.arange(len(plot_df))
        deltas = plot_df["delta_macro_f1_vs_baseline"].to_numpy(dtype=float)
        ci_low = plot_df["delta_macro_f1_ci_lower"].to_numpy(dtype=float)
        ci_high = plot_df["delta_macro_f1_ci_upper"].to_numpy(dtype=float)
        xerr = np.vstack([deltas - ci_low, ci_high - deltas])
        colours = [METHOD_COLOURS[method] for method in ordered_methods]

        ax.barh(y_pos, deltas, color=colours, height=0.58)
        ax.errorbar(
            deltas,
            y_pos,
            xerr=xerr,
            fmt="none",
            ecolor="#202833",
            elinewidth=1.2,
            capsize=3,
            capthick=1.2,
            zorder=3,
        )
        ax.axvline(0, color="#1f2933", linewidth=1.0)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([METHOD_LABELS[method] for method in ordered_methods])
        ax.invert_yaxis()
        ax.set_title(DIRECTION_LABELS[direction])
        ax.set_xlabel("Macro-F1 delta vs unadapted RF")
        ax.set_xlim(-x_limit, x_limit)
        ax.grid(axis="x", color="#d7dde5", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#99a3af")

        for y, delta in zip(y_pos, deltas):
            label = f"{delta:+.3f}"
            offset = 0.008 if delta >= 0 else -0.008
            ha = "left" if delta >= 0 else "right"
            ax.text(delta + offset, y, label, va="center", ha=ha, fontsize=9, color="#202833")

    fig.suptitle("Offline HAR Domain-Adaptation Deltas")
    fig.text(
        0.5,
        0.035,
        (
            "Error bars are paired bootstrap 95% CIs over target windows "
            "(2,000 resamples). Transforms use target-domain feature statistics only, not labels."
        ),
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.03, 0.08, 0.99, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg", metadata={"Date": None})
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_csv = _resolve(args.in_csv)
    out_svg = _resolve(args.out_svg)
    admin_svg = _resolve(args.admin_svg) if args.admin_svg else None
    df = _load_adaptation_rows(input_csv)
    render(df, out_svg)
    print(f"Wrote {out_svg}")
    if admin_svg is not None:
        render(df, admin_svg)
        print(f"Wrote {admin_svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
