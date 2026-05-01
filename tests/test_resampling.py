from pathlib import Path

import pandas as pd
import pytest

from pipeline.preprocess.resample import build_uniform_timeline, estimate_sampling_rate, resample_dataframe

FIX = Path("tests/fixtures")


def test_regular_fixture_remains_consistent_after_resampling():
    df = pd.read_csv(FIX / "timeseries_regular.csv")
    est = estimate_sampling_rate(df)
    assert est == pytest.approx(50.0, rel=1e-2)

    out = resample_dataframe(df, target_rate_hz=50.0)
    assert len(out) == len(df)
    assert out["timestamp"].iloc[0] == pytest.approx(df["timestamp"].iloc[0])
    assert out["timestamp"].iloc[-1] == pytest.approx(df["timestamp"].iloc[-1])
    assert out["ax"].iloc[0] == pytest.approx(df["ax"].iloc[0])
    assert out["label_mapped"].notna().all()


def test_irregular_fixture_resamples_to_expected_timeline_length_and_preserves_metadata():
    df = pd.read_csv(FIX / "timeseries_irregular.csv")
    out = resample_dataframe(df, target_rate_hz=50.0)

    expected_timeline = build_uniform_timeline(df["timestamp"].min(), df["timestamp"].max(), 50.0)
    assert len(out) == len(expected_timeline)

    for col in ["dataset_name", "task_type", "subject_id", "session_id", "placement", "source_file"]:
        assert col in out.columns
        assert out[col].nunique(dropna=False) == 1

    assert out["label_raw"].notna().all()
    assert out["label_mapped"].notna().all()
    assert (out["sampling_rate_hz"] == 50.0).all()


def test_resampling_populates_labels_without_numeric_interpolation():
    df = pd.read_csv(FIX / "timeseries_irregular.csv")
    out = resample_dataframe(df, target_rate_hz=50.0)
    assert out["label_mapped"].isin(["locomotion", "static"]).all()
