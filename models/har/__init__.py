"""HAR baseline training and evaluation utilities (Chapter 4)."""

from models.har.baselines import (
    DEFAULT_HAR_LABEL_ORDER,
    get_feature_importances_dataframe,
    heuristic_har_predict,
    train_random_forest_classifier,
)
from models.har.evaluate_har import (
    run_har_baselines_on_feature_table,
    run_har_baselines_on_train_test_feature_tables,
)
from models.har.train_har import (
    build_group_labels,
    prepare_feature_matrices,
    select_feature_columns,
    subject_aware_group_split,
)

__all__ = [
    "DEFAULT_HAR_LABEL_ORDER",
    "heuristic_har_predict",
    "train_random_forest_classifier",
    "get_feature_importances_dataframe",
    "build_group_labels",
    "select_feature_columns",
    "subject_aware_group_split",
    "prepare_feature_matrices",
    "run_har_baselines_on_feature_table",
    "run_har_baselines_on_train_test_feature_tables",
]
