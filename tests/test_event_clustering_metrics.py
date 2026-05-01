import pandas as pd

from metrics.fall_event_metrics import compute_event_level_metrics


def test_event_level_metrics_clusters_consecutive_windows_and_counts_overlaps():
    df = pd.DataFrame(
        {
            "dataset_name": ["D"] * 6,
            "subject_id": ["S1"] * 6,
            "session_id": ["sess1"] * 6,
            "source_file": ["f1"] * 6,
            "window_id": [0, 1, 2, 3, 4, 5],
            "start_ts": [0, 1, 2, 3, 4, 5],
            "end_ts": [1, 2, 3, 4, 5, 6],
            "true_label": ["non_fall", "fall", "fall", "non_fall", "fall", "non_fall"],
            "predicted_label": ["non_fall", "fall", "non_fall", "non_fall", "fall", "fall"],
        }
    )

    out = compute_event_level_metrics(df)

    assert out["predicted_fall_events_count"] == 2
    assert out["true_fall_events_count"] == 2
    assert out["true_positive_events_count"] == 2
    assert out["false_positive_events_count"] == 0
    assert out["false_negative_events_count"] == 0
    assert out["event_precision"] == 1.0
    assert out["event_recall"] == 1.0


def test_event_level_metrics_counts_false_positive_event():
    df = pd.DataFrame(
        {
            "dataset_name": ["D"] * 5,
            "subject_id": ["S1"] * 5,
            "session_id": ["sess1"] * 5,
            "source_file": ["f1"] * 5,
            "window_id": [0, 1, 2, 3, 4],
            "start_ts": [0, 1, 2, 3, 4],
            "end_ts": [1, 2, 3, 4, 5],
            "true_label": ["non_fall", "fall", "fall", "non_fall", "non_fall"],
            "predicted_label": ["non_fall", "non_fall", "non_fall", "fall", "fall"],
        }
    )

    out = compute_event_level_metrics(df)

    assert out["predicted_fall_events_count"] == 1
    assert out["true_fall_events_count"] == 1
    assert out["true_positive_events_count"] == 0
    assert out["false_positive_events_count"] == 1
    assert out["false_negative_events_count"] == 1
    assert out["event_precision"] == 0.0
    assert out["event_recall"] == 0.0
