import pandas as pd

from models.har.evaluate_har import (
    run_har_baselines_on_feature_table,
    run_har_baselines_on_train_test_feature_tables,
)


def _synthetic_feature_table() -> pd.DataFrame:
    rows = []
    labels = ["static", "locomotion", "stairs", "other"]
    for subject_idx in range(1, 9):  # 8 subjects
        for label_idx, label in enumerate(labels):
            # Create separable but slightly noisy feature patterns.
            base_std = [0.05, 0.40, 0.55, 0.25][label_idx]
            base_range = [0.10, 0.90, 1.20, 0.40][label_idx]
            base_dom = [0.2, 1.8, 2.2, 0.6][label_idx]
            base_jerk = [1.0, 8.0, 14.0, 3.0][label_idx]
            base_az_std = [0.08, 0.20, 0.65, 0.25][label_idx]
            noise = 0.01 * subject_idx

            rows.append(
                {
                    "window_id": len(rows),
                    "dataset_name": "SYNTH",
                    "subject_id": f"S{subject_idx:02d}",
                    "session_id": f"sess_{subject_idx:02d}",
                    "source_file": f"synth_subject_{subject_idx:02d}.csv",
                    "task_type": "har",
                    "label_mapped_majority": label,
                    "is_acceptable": True,
                    "n_samples": 128,
                    "missing_ratio": 0.0,
                    "has_large_gap": False,
                    "n_gaps": 0,
                    "start_ts": 0.0,
                    "end_ts": 2.54,
                    "ax_mean": 0.1 * label_idx + noise,
                    "az_std": base_az_std + noise,
                    "acc_magnitude_std": base_std + noise,
                    "acc_magnitude_range": base_range + noise,
                    "acc_magnitude_dominant_freq_hz": base_dom + 0.02 * subject_idx,
                    "acc_magnitude_spectral_energy": 5.0 + label_idx + noise,
                    "acc_magnitude_jerk_mean_abs": base_jerk + noise,
                    "acc_magnitude_mean_abs_diff": (base_jerk / 50.0) + noise,
                    "acc_sma": 9.0 + label_idx + noise,
                    "window_sampling_rate_hz": 50.0,
                }
            )
    return pd.DataFrame(rows)


def test_har_baselines_run_end_to_end_on_synthetic_feature_table():
    feature_df = _synthetic_feature_table()
    result = run_har_baselines_on_feature_table(feature_df, test_size=0.25, random_state=7)

    assert "heuristic" in result
    assert "random_forest" in result
    assert "split" in result
    assert "feature_columns" in result
    assert result["split"]["train_rows"] > 0
    assert result["split"]["test_rows"] > 0

    train_groups = set(result["split"]["train_subject_groups"])
    test_groups = set(result["split"]["test_subject_groups"])
    assert train_groups.isdisjoint(test_groups)

    h_metrics = result["heuristic"]["metrics"]
    rf_metrics = result["random_forest"]["metrics"]
    assert 0.0 <= h_metrics["accuracy"] <= 1.0
    assert 0.0 <= h_metrics["macro_f1"] <= 1.0
    assert 0.0 <= rf_metrics["accuracy"] <= 1.0
    assert 0.0 <= rf_metrics["macro_f1"] <= 1.0

    cm = rf_metrics["confusion_matrix"]
    labels = rf_metrics["labels"]
    assert len(cm) == len(labels)
    assert all(len(row) == len(labels) for row in cm)

    rf_importances = result["random_forest"]["feature_importances"]
    assert not rf_importances.empty
    assert {"feature", "importance"}.issubset(rf_importances.columns)


def test_har_baselines_run_on_explicit_train_test_split():
    feature_df = _synthetic_feature_table()
    train_df = feature_df[feature_df["subject_id"].isin(["S01", "S02", "S03", "S04"])].reset_index(drop=True)
    test_df = feature_df[feature_df["subject_id"].isin(["S05", "S06"])].reset_index(drop=True)

    result = run_har_baselines_on_train_test_feature_tables(
        train_df,
        test_df,
        random_state=7,
    )

    assert result["split"]["strategy"] == "explicit_train_test_split"
    assert result["split"]["train_rows"] == len(train_df)
    assert result["split"]["test_rows"] == len(test_df)
    assert 0.0 <= result["heuristic"]["metrics"]["macro_f1"] <= 1.0
    assert 0.0 <= result["random_forest"]["metrics"]["macro_f1"] <= 1.0
