from pathlib import Path

import pandas as pd
import pytest

from pipeline.preprocess.config import PreprocessConfig
from pipeline.preprocess.window import assign_majority_label, sliding_window_indices, window_dataframe

FIX = Path("tests/fixtures")


def test_sliding_window_indices_count_and_overlap():
    idx = sliding_window_indices(n_samples=12, window_size=4, step_size=2)
    assert idx == [(0, 4), (2, 6), (4, 8), (6, 10), (8, 12)]


def test_sliding_window_indices_drops_incomplete_trailing_window():
    idx = sliding_window_indices(n_samples=10, window_size=4, step_size=3)
    assert idx == [(0, 4), (3, 7), (6, 10)]


def test_majority_label_assignment_and_tie_break_is_deterministic():
    df = pd.DataFrame({"label_mapped": ["static", "locomotion", "static", "locomotion"]})
    # Tie: earliest occurrence wins (static).
    assert assign_majority_label(df) == "static"


def test_window_dataframe_outputs_expected_contract():
    df = pd.read_csv(FIX / "timeseries_regular.csv")
    cfg = PreprocessConfig()
    windows = window_dataframe(df, window_size=4, step_size=2, config=cfg)

    assert len(windows) == 5
    first = windows[0]
    assert first["window_id"] == 0
    assert first["dataset_name"] == "TESTSEQ"
    assert first["subject_id"] == "S1"
    assert first["task_type"] == "har"
    assert first["n_samples"] == 4
    assert first["start_ts"] == pytest.approx(0.0)
    assert first["end_ts"] == pytest.approx(0.06)
    assert first["label_mapped_majority"] in {"locomotion", "static"}
    assert "sensor_payload" in first
    assert len(first["sensor_payload"]["ax"]) == 4
