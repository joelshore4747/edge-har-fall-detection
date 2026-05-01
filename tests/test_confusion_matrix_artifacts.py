from pathlib import Path

import pytest

from analysis.confusion_matrix_plots import plot_confusion_matrix, save_confusion_matrix_csv


def test_confusion_matrix_csv_and_png_artifacts_are_written(tmp_path: Path):
    pytest.importorskip("matplotlib")

    labels = ["static", "locomotion"]
    cm = [[3, 1], [2, 4]]

    csv_path = tmp_path / "cm.csv"
    png_path = tmp_path / "cm.png"

    save_confusion_matrix_csv(cm, labels, csv_path)
    out_png = plot_confusion_matrix(cm, labels, title="Test CM", out_path=png_path)

    assert csv_path.exists()
    assert csv_path.stat().st_size > 0
    assert out_png == png_path
    assert png_path.exists()
    assert png_path.stat().st_size > 0
