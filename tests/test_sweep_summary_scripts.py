import pandas as pd
import pytest

from scripts.diagnose_fall_feature_scales import build_quantiles_by_label
from scripts.summarize_threshold_sweep import (
    filter_balanced_candidates,
    filter_conservative_candidates,
)


def test_build_quantiles_by_label_creates_expected_quantiles():
    df = pd.DataFrame(
        {
            "true_label": [
                "Fall",
                "fall",
                "FALL",
                "fall ",
                "non_fall",
                "non-fall",
                "ADL",
                "nonfall",
            ],
            "peak_acc": [2.0, 4.0, 6.0, 8.0, 1.0, 2.0, 3.0, 4.0],
            "jerk_peak": [10.0, 20.0, 30.0, 40.0, 5.0, 6.0, 7.0, 8.0],
        }
    )

    quantiles = build_quantiles_by_label(df, features=["peak_acc", "jerk_peak"])

    assert quantiles["fall"]["peak_acc"]["q50"] == pytest.approx(5.0)
    assert quantiles["fall"]["peak_acc"]["q75"] == pytest.approx(6.5)
    assert quantiles["non_fall"]["peak_acc"]["q50"] == pytest.approx(2.5)
    assert quantiles["non_fall"]["jerk_peak"]["q95"] == pytest.approx(7.85)


def test_candidate_filters_return_sorted_balanced_and_conservative_rows():
    df = pd.DataFrame(
        {
            "config_id": [1, 2, 3, 4, 5],
            "impact_threshold": [10, 12, 14, 16, 18],
            "confirm_post_dyn_mean_max": [0.4, 0.5, 0.5, 0.7, 0.8],
            "jerk_threshold": [0, 20, 20, 30, 30],
            "sensitivity": ["0.20", "0.35", "0.40", "0.32", "0.38"],
            "specificity": ["0.90", "0.86", "0.84", "0.92", "0.88"],
            "precision": [0.50, 0.60, 0.70, 0.55, 0.63],
            "f1": ["0.29", "0.47", "0.50", "0.41", "0.45"],
            "false_alarms_count": [5, 10, 3, 2, 2],
        }
    )

    balanced = filter_balanced_candidates(df, min_sensitivity=0.30, min_specificity=0.85)
    conservative = filter_conservative_candidates(df, min_specificity=0.85)

    assert list(balanced["config_id"]) == [2, 5, 4]
    assert list(conservative["config_id"]) == [5, 4, 1, 2]
