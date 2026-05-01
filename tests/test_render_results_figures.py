"""Smoke tests for ``scripts.render_results_figures``."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402

from scripts.render_results_figures import (
    figure_activity_by_placement,
    figure_per_session_confusion,
    figure_pr_curve,
    figure_threshold_sweep,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_per_session_confusion_renders(tmp_path: Path):
    csv_path = tmp_path / "A.csv"
    _write_csv(csv_path, [
        {"actual": "pocket", "predicted": "pocket", "predicted_smoothed": "pocket", "agree": 1, "agree_smoothed": 1, "placement": "pocket"},
        {"actual": "hand", "predicted": "hand", "predicted_smoothed": "hand", "agree": 1, "agree_smoothed": 1, "placement": "hand"},
        {"actual": "hand", "predicted": "pocket", "predicted_smoothed": "hand", "agree": 0, "agree_smoothed": 1, "placement": "hand"},
    ])
    out = tmp_path / "fig.png"
    figure_per_session_confusion(csv_path, title="Test", out_path=out)
    assert out.exists() and out.stat().st_size > 0


def test_threshold_sweep_renders(tmp_path: Path):
    csv_path = tmp_path / "sweep.csv"
    _write_csv(csv_path, [
        {"score": "max_p_fall", "threshold": 0.3, "tp": 1, "fp": 1, "fn": 1, "tn": 1, "precision": 0.5, "recall": 0.5, "f1": 0.5},
        {"score": "max_p_fall", "threshold": 0.5, "tp": 2, "fp": 0, "fn": 0, "tn": 2, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        {"score": "smoothed_max_p_fall", "threshold": 0.3, "tp": 2, "fp": 1, "fn": 0, "tn": 1, "precision": 0.66, "recall": 1.0, "f1": 0.8},
    ])
    out = tmp_path / "sweep.png"
    figure_threshold_sweep(csv_path, out_path=out)
    assert out.exists() and out.stat().st_size > 0


def test_pr_curve_renders(tmp_path: Path):
    csv_path = tmp_path / "B.csv"
    _write_csv(csv_path, [
        {"is_fall_session": 1, "max_p_fall": 0.8, "smoothed_max_p_fall": 0.7},
        {"is_fall_session": 0, "max_p_fall": 0.2, "smoothed_max_p_fall": 0.1},
        {"is_fall_session": 1, "max_p_fall": 0.6, "smoothed_max_p_fall": 0.55},
        {"is_fall_session": 0, "max_p_fall": 0.4, "smoothed_max_p_fall": 0.3},
    ])
    out = tmp_path / "pr.png"
    figure_pr_curve(csv_path, out_path=out)
    assert out.exists() and out.stat().st_size > 0


def test_activity_by_placement_renders(tmp_path: Path):
    csv_path = tmp_path / "C.csv"
    _write_csv(csv_path, [
        {"placement": "hand", "actual": "walking", "predicted": "walking", "agree": 1},
        {"placement": "hand", "actual": "walking", "predicted": "stairs", "agree": 0},
        {"placement": "pocket", "actual": "walking", "predicted": "walking", "agree": 1},
        {"placement": "pocket", "actual": "stairs", "predicted": "walking", "agree": 0},
    ])
    out = tmp_path / "heatmap.png"
    figure_activity_by_placement(csv_path, out_path=out)
    assert out.exists() and out.stat().st_size > 0
