"""Chapter 4 HAR feature engineering utilities."""

from pipeline.features.build_feature_table import (
    build_feature_table,
    feature_table_schema_summary,
    infer_window_sampling_rate_hz,
)
from pipeline.features.frequency_domain import (
    compute_frequency_features,
    extract_frequency_features_for_window,
)
from pipeline.features.magnitude_features import (
    extract_magnitude_features_for_window,
    signal_magnitude_area,
)
from pipeline.features.time_domain import (
    compute_time_domain_features,
    extract_time_domain_features_for_window,
)

__all__ = [
    "compute_time_domain_features",
    "extract_time_domain_features_for_window",
    "signal_magnitude_area",
    "extract_magnitude_features_for_window",
    "compute_frequency_features",
    "extract_frequency_features_for_window",
    "build_feature_table",
    "infer_window_sampling_rate_hz",
    "feature_table_schema_summary",
]
