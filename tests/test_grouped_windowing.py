from pathlib import Path

import pandas as pd

from pipeline.preprocess.config import PreprocessConfig
from pipeline.preprocess.resample import resample_dataframe
from pipeline.preprocess.window import window_dataframe

FIX = Path("tests/fixtures")


def _make_two_sequence_dataframe() -> pd.DataFrame:
    base = pd.read_csv(FIX / "timeseries_regular.csv").iloc[:6].copy()

    seq_a = base.copy()
    seq_a["subject_id"] = "S_A"
    seq_a["session_id"] = "sess_A"
    seq_a["source_file"] = "seq_a.csv"
    seq_a["ax"] = seq_a["ax"] + 0.0

    seq_b = base.copy()
    seq_b["subject_id"] = "S_B"
    seq_b["session_id"] = "sess_B"
    seq_b["source_file"] = "seq_b.csv"
    seq_b["timestamp"] = seq_b["timestamp"] + 10.0
    seq_b["ax"] = seq_b["ax"] + 1000.0  # make mixed windows obvious if grouping breaks

    return pd.concat([seq_a, seq_b], ignore_index=True)


def test_grouped_resampling_does_not_mix_sequences():
    df = _make_two_sequence_dataframe()
    out = resample_dataframe(df, target_rate_hz=50.0)

    # Two logical groups should remain after grouped resampling.
    groups = out.groupby(["dataset_name", "subject_id", "session_id", "source_file"], dropna=False, sort=False)
    assert groups.ngroups == 2
    assert set(out["subject_id"].astype(str).unique().tolist()) == {"S_A", "S_B"}

    # Each sequence covers the same time duration -> same resampled row count.
    group_sizes = sorted(len(g) for _, g in groups)
    assert len(group_sizes) == 2
    assert group_sizes[0] == group_sizes[1]


def test_grouped_windowing_never_spans_multiple_sequences():
    df = _make_two_sequence_dataframe()
    cfg = PreprocessConfig(target_sampling_rate_hz=50.0)
    windows = window_dataframe(df, window_size=4, step_size=2, config=cfg)

    # 6 samples per sequence -> windows (0:4), (2:6) => 2 each => total 4.
    assert len(windows) == 4

    # If grouping were broken, a boundary-crossing window would mix ax near 0 and ~1000.
    for w in windows:
        ax = w["sensor_payload"]["ax"]
        assert float(ax.max() - ax.min()) < 100.0
        assert w["subject_id"] in {"S_A", "S_B"}
        assert w["session_id"] in {"sess_A", "sess_B"}
        assert w["source_file"] in {"seq_a.csv", "seq_b.csv"}
