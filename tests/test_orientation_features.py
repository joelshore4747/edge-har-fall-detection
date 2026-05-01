import pandas as pd
import pytest

from pipeline.preprocess.orientation import append_acc_magnitude, append_derived_channels, append_gyro_magnitude


def test_acc_magnitude_computed_correctly():
    df = pd.DataFrame({"ax": [3.0], "ay": [4.0], "az": [12.0]})
    out = append_acc_magnitude(df)
    assert out["acc_magnitude"].iloc[0] == pytest.approx(13.0)


def test_gyro_magnitude_computed_correctly_when_present():
    df = pd.DataFrame({"gx": [1.0], "gy": [2.0], "gz": [2.0]})
    out = append_gyro_magnitude(df)
    assert out["gyro_magnitude"].iloc[0] == pytest.approx(3.0)


def test_gyro_magnitude_added_safely_when_gyro_absent():
    df = pd.DataFrame({"ax": [0.0], "ay": [0.0], "az": [1.0]})
    out = append_gyro_magnitude(df)
    assert "gyro_magnitude" in out.columns
    assert out["gyro_magnitude"].isna().all()


def test_append_derived_channels_combined_helper():
    df = pd.DataFrame({
        "ax": [0.0, 3.0],
        "ay": [4.0, 0.0],
        "az": [0.0, 0.0],
        "gx": [0.0, 1.0],
        "gy": [0.0, 2.0],
        "gz": [0.0, 2.0],
    })
    out = append_derived_channels(df)
    assert "acc_magnitude" in out.columns
    assert "gyro_magnitude" in out.columns
    assert out["acc_magnitude"].iloc[0] == pytest.approx(4.0)
    assert out["gyro_magnitude"].iloc[1] == pytest.approx(3.0)
