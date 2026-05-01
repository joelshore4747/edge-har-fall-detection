import pandas as pd
import pytest

from pipeline.preprocess.dejitter import (
    PHANTOM_GAP_THRESHOLD_SECONDS,
    drop_phantom_leading_samples,
)
from pipeline.preprocess.resample import resample_group_to_rate


def _frame(timestamps):
    return pd.DataFrame(
        {"timestamp": timestamps, "ax": [0.0] * len(timestamps),
         "ay": [0.0] * len(timestamps), "az": [9.81] * len(timestamps)}
    )


def test_drops_phantom_zero_sample_with_large_gap():
    df = _frame([0.0, 83.5, 83.52, 83.54])
    cleaned = drop_phantom_leading_samples(df)
    assert len(cleaned) == 3
    assert cleaned["timestamp"].iloc[0] == pytest.approx(83.5)


def test_keeps_clean_session_with_small_starting_gap():
    df = _frame([1764.7, 1764.72, 1764.74, 1764.76])
    cleaned = drop_phantom_leading_samples(df)
    assert len(cleaned) == 4


def test_threshold_is_strictly_greater_than():
    df = _frame([0.0, PHANTOM_GAP_THRESHOLD_SECONDS, 5.02, 5.04])
    cleaned = drop_phantom_leading_samples(df)
    assert len(cleaned) == 4


def test_handles_too_short_input():
    assert len(drop_phantom_leading_samples(_frame([0.0]))) == 1
    assert len(drop_phantom_leading_samples(_frame([]))) == 0


def test_resample_drops_phantom_before_resampling():
    df = _frame([0.0, 100.0, 100.02, 100.04, 100.06, 100.08])
    out = resample_group_to_rate(df, target_rate_hz=50.0)
    assert float(out["timestamp"].min()) == pytest.approx(100.0, abs=0.05)
    span = float(out["timestamp"].max()) - float(out["timestamp"].min())
    assert span < 1.0
