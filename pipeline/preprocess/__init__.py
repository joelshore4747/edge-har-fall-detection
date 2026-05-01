"""Chapter 3 preprocessing utilities (resampling, quality, orientation, windowing)."""

from pipeline.preprocess.config import PreprocessConfig, default_preprocess_config
from pipeline.preprocess.dejitter import drop_phantom_leading_samples
from pipeline.preprocess.orientation import append_acc_magnitude, append_derived_channels, append_gyro_magnitude
from pipeline.preprocess.prepare import prepare_windowed_sequences
from pipeline.preprocess.quality import compute_missing_ratio, detect_large_time_gaps, is_window_acceptable, window_quality_summary
from pipeline.preprocess.resample import (
    build_uniform_timeline,
    default_resample_group_cols,
    default_rate_summary_group_cols,
    estimate_sampling_rate,
    summarize_sampling_rate_by_group,
    resample_dataframe,
    resample_group_to_rate,
)
from pipeline.preprocess.quality import infer_active_sensor_columns
from pipeline.preprocess.window import assign_majority_label, sliding_window_indices, window_dataframe

__all__ = [
    "PreprocessConfig",
    "default_preprocess_config",
    "drop_phantom_leading_samples",
    "estimate_sampling_rate",
    "build_uniform_timeline",
    "default_resample_group_cols",
    "default_rate_summary_group_cols",
    "summarize_sampling_rate_by_group",
    "resample_group_to_rate",
    "resample_dataframe",
    "append_acc_magnitude",
    "append_gyro_magnitude",
    "append_derived_channels",
    "prepare_windowed_sequences",
    "compute_missing_ratio",
    "infer_active_sensor_columns",
    "detect_large_time_gaps",
    "window_quality_summary",
    "is_window_acceptable",
    "assign_majority_label",
    "sliding_window_indices",
    "window_dataframe",
]
