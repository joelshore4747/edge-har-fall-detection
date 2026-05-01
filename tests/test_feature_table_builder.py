import numpy as np

from pipeline.features.build_feature_table import build_feature_table, feature_table_schema_summary, infer_window_sampling_rate_hz


def _make_window(window_id: int, *, acceptable: bool, label: str = "locomotion") -> dict:
    n = 8
    t = np.arange(n, dtype=float) / 50.0
    ax = np.linspace(0.0, 0.7, n)
    ay = np.linspace(0.1, 0.8, n)
    az = np.linspace(9.6, 9.9, n)
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    return {
        "window_id": window_id,
        "dataset_name": "TESTSEQ",
        "subject_id": "S1",
        "session_id": "sessA",
        "source_file": "fixture.csv",
        "task_type": "har",
        "start_ts": float(t[0]),
        "end_ts": float(t[-1]),
        "label_mapped_majority": label,
        "n_samples": n,
        "missing_ratio": 0.0,
        "is_acceptable": acceptable,
        "has_large_gap": False,
        "n_gaps": 0,
        "quality_summary": {"is_acceptable": acceptable},
        "sensor_payload": {
            "ax": ax,
            "ay": ay,
            "az": az,
            "acc_magnitude": acc_mag,
        },
    }


def test_infer_window_sampling_rate_from_window_metadata():
    w = _make_window(0, acceptable=True)
    fs = infer_window_sampling_rate_hz(w)
    assert fs is not None
    assert abs(fs - 50.0) < 1e-6


def test_build_feature_table_preserves_metadata_and_features():
    windows = [_make_window(0, acceptable=True), _make_window(1, acceptable=True, label="static")]
    feature_df = build_feature_table(windows, filter_unacceptable=True)

    assert len(feature_df) == 2
    for col in ["window_id", "dataset_name", "subject_id", "label_mapped_majority", "is_acceptable"]:
        assert col in feature_df.columns
    assert "ax_mean" in feature_df.columns
    assert "acc_magnitude_rms" in feature_df.columns
    assert "acc_magnitude_dominant_freq_hz" in feature_df.columns

    summary = feature_table_schema_summary(feature_df)
    assert summary["rows"] == 2
    assert summary["subjects_count"] == 1


def test_build_feature_table_filters_unacceptable_windows():
    windows = [_make_window(0, acceptable=True), _make_window(1, acceptable=False)]
    filtered = build_feature_table(windows, filter_unacceptable=True)
    unfiltered = build_feature_table(windows, filter_unacceptable=False)
    assert len(filtered) == 1
    assert len(unfiltered) == 2
