"""Render PNG figures from a run's experiments CSVs.

Reads ``runs/<id>/experiments/*.csv`` and writes PNGs into
``runs/<id>/figures/``. Designed to be cheap and dependency-light: only
matplotlib is required, no seaborn. Run after the experiments script has
populated the CSVs:

    python scripts/render_results_figures.py --run current
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lib.run_registry import resolve_current_run  # noqa: E402

logger = logging.getLogger("render_results_figures")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run",
        default="current",
        help="Run id; ``current`` follows the symlink",
    )
    p.add_argument(
        "--runs-root",
        type=Path,
        default=Path("artifacts/unifallmonitor/runs"),
    )
    return p.parse_args()


def _resolve_run_dir(*, run: str, runs_root: Path) -> Path:
    runs_root = runs_root.resolve()
    if run == "current":
        target = resolve_current_run(runs_root=runs_root)
        if target is None:
            raise SystemExit(
                f"No 'current' run resolved under {runs_root.parent}. "
                f"Pass --run <id> explicitly."
            )
        return target
    candidate = runs_root / run
    if not candidate.exists():
        raise SystemExit(f"Run {run} not found under {runs_root}")
    return candidate.resolve()


def _confusion(actual: pd.Series, predicted: pd.Series, labels: list[str]) -> np.ndarray:
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    for a, p in zip(actual, predicted):
        if a in label_to_idx and p in label_to_idx:
            cm[label_to_idx[a], label_to_idx[p]] += 1
    return cm


def render_confusion(ax, cm: np.ndarray, labels: list[str], *, title: str) -> None:
    ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title)
    vmax = cm.max() if cm.size else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = int(cm[i, j])
            color = "white" if v > vmax / 2 else "black"
            ax.text(j, i, str(v), ha="center", va="center", color=color)


def figure_per_session_confusion(
    csv_path: Path, *, title: str, out_path: Path
) -> None:
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    if df.empty:
        return
    labels = sorted(set(df["actual"]).union(set(df["predicted"])))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    cm_raw = _confusion(df["actual"], df["predicted"], labels)
    render_confusion(axes[0], cm_raw, labels, title=f"{title} — raw majority")
    if "predicted_smoothed" in df.columns:
        cm_sm = _confusion(df["actual"], df["predicted_smoothed"], labels)
        render_confusion(axes[1], cm_sm, labels, title=f"{title} — smoothed")
    else:
        axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def figure_threshold_sweep(csv_path: Path, *, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if "score" in df.columns:
        for score, group in df.groupby("score"):
            ax.plot(group["threshold"], group["f1"], marker="o", label=score)
    else:
        ax.plot(df["threshold"], df["f1"], marker="o", label="F1")

    # F1-optimal point per score series.
    if "score" in df.columns:
        for score, group in df.groupby("score"):
            valid = group.dropna(subset=["f1"])
            if valid.empty:
                continue
            best = valid.loc[valid["f1"].idxmax()]
            ax.scatter([best["threshold"]], [best["f1"]], s=80, marker="*",
                       label=f"{score} best F1={best['f1']:.2f} @ t={best['threshold']:.2f}")
    ax.set_xlabel("threshold on per-session max P(fall)")
    ax.set_ylabel("F1")
    ax.set_title("Fall-event threshold sweep")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def figure_pr_curve(csv_path: Path, *, out_path: Path) -> None:
    """Precision-recall curve from the per-session score CSV.

    Plots both raw and smoothed scores, sweeping a fine threshold grid so
    the curve isn't restricted to the 0.05-step sweep used in the report.
    """
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    if df.empty:
        return
    y = df["is_fall_session"].astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    for col, label in (("max_p_fall", "raw"), ("smoothed_max_p_fall", "smoothed")):
        if col not in df.columns:
            continue
        scores = df[col].to_numpy(dtype=float)
        thresholds = np.unique(np.concatenate([scores, np.linspace(0, 1, 101)]))
        precision = []
        recall = []
        for thr in thresholds:
            pred = (scores >= thr).astype(int)
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            p = tp / (tp + fp) if (tp + fp) else 1.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            precision.append(p)
            recall.append(r)
        order = np.argsort(recall)
        ax.plot(np.array(recall)[order], np.array(precision)[order], label=label)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Fall-event detection (per-session) — PR curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def figure_activity_by_placement(csv_path: Path, *, out_path: Path) -> None:
    """Heatmap: per-(placement, actual-activity) accuracy on experiment C."""
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    if df.empty:
        return
    placements = sorted(df["placement"].dropna().unique())
    activities = sorted(df["actual"].dropna().unique())
    grid = np.full((len(placements), len(activities)), np.nan)
    counts = np.zeros((len(placements), len(activities)), dtype=int)
    for i, pl in enumerate(placements):
        for j, act in enumerate(activities):
            sub = df[(df["placement"] == pl) & (df["actual"] == act)]
            if sub.empty:
                continue
            counts[i, j] = len(sub)
            grid[i, j] = sub["agree"].mean()

    fig, ax = plt.subplots(figsize=(6 + 0.5 * len(activities), 1 + 0.6 * len(placements)))
    img = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(activities)))
    ax.set_yticks(range(len(placements)))
    ax.set_xticklabels(activities, rotation=30, ha="right")
    ax.set_yticklabels(placements)
    ax.set_xlabel("actual activity")
    ax.set_ylabel("placement")
    ax.set_title("Activity-by-placement LOSO accuracy")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            n = counts[i, j]
            if n == 0:
                continue
            v = grid[i, j]
            ax.text(j, i, f"{v:.0%}\n(n={n})", ha="center", va="center", fontsize=8)
    fig.colorbar(img, ax=ax, fraction=0.04, pad=0.04, label="accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s :: %(message)s")
    args = parse_args()
    run_dir = _resolve_run_dir(run=args.run, runs_root=args.runs_root)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir = run_dir / "experiments"

    if (experiments_dir / "A_placement_per_session.csv").exists():
        figure_per_session_confusion(
            experiments_dir / "A_placement_per_session.csv",
            title="Placement (LOSO)",
            out_path=figures_dir / "A_placement_confusion.png",
        )
    if (experiments_dir / "C_activity_5class_per_session.csv").exists():
        figure_per_session_confusion(
            experiments_dir / "C_activity_5class_per_session.csv",
            title="Activity 5-class (LOSO)",
            out_path=figures_dir / "C_activity_confusion.png",
        )
        figure_activity_by_placement(
            experiments_dir / "C_activity_5class_per_session.csv",
            out_path=figures_dir / "C_activity_by_placement.png",
        )
    if (experiments_dir / "B_fall_threshold_sweep.csv").exists():
        figure_threshold_sweep(
            experiments_dir / "B_fall_threshold_sweep.csv",
            out_path=figures_dir / "B_fall_threshold_sweep.png",
        )
    if (experiments_dir / "B_fall_event_per_session.csv").exists():
        figure_pr_curve(
            experiments_dir / "B_fall_event_per_session.csv",
            out_path=figures_dir / "B_fall_pr_curve.png",
        )

    logger.info("Figures written to %s", figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
