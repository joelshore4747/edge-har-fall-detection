from pathlib import Path

import pandas as pd

from pipeline.preprocess.config import PreprocessConfig
from pipeline.preprocess.quality import compute_missing_ratio, detect_large_time_gaps, is_window_acceptable, window_quality_summary

FIX = Path("tests/fixtures")


def test_missing_segment_fixture_triggers_missing_and_gap_detection():
    df = pd.read_csv(FIX / "timeseries_missing_segments.csv")
    missing_ratio = compute_missing_ratio(df)
    gap_summary = detect_large_time_gaps(df)

    assert missing_ratio > 0.0
    assert gap_summary["has_large_gap"] is True
    assert gap_summary["n_gaps"] >= 1
    assert gap_summary["max_gap"] is not None


def test_window_acceptability_distinguishes_clean_and_dirty_windows():
    df = pd.read_csv(FIX / "timeseries_missing_segments.csv")
    cfg = PreprocessConfig(max_missing_ratio_per_window=0.20)

    clean_window = df.iloc[7:11].copy()  # mostly dense and after the gap
    dirty_window = df.iloc[2:8].copy()   # contains missing values and the large gap transition

    clean_summary = window_quality_summary(clean_window, cfg)
    dirty_summary = window_quality_summary(dirty_window, cfg)

    assert clean_summary["is_acceptable"] is True
    assert dirty_summary["is_acceptable"] is False
    assert is_window_acceptable(clean_window, cfg) is True
    assert is_window_acceptable(dirty_window, cfg) is False
