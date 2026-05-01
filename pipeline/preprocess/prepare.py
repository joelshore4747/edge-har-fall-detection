from __future__ import annotations

from typing import Any

import pandas as pd

from pipeline.preprocess.config import PreprocessConfig, default_preprocess_config
from pipeline.preprocess.resample import resample_dataframe
from pipeline.preprocess.window import window_dataframe


def prepare_windowed_sequences(
    df: pd.DataFrame,
    *,
    config: PreprocessConfig | None = None,
    label_col: str = "label_mapped",
    group_cols: list[str] | None = None,
    timestamp_col: str = "timestamp",
) -> list[dict[str, Any]]:
    cfg = config or default_preprocess_config()

    resampled = resample_dataframe(
        df,
        target_rate_hz=cfg.target_sampling_rate_hz,
        timestamp_col=timestamp_col,
        interpolation_method=cfg.interpolation_method,
        group_cols=group_cols,
    )

    return window_dataframe(
        resampled,
        window_size=cfg.window_size_samples,
        step_size=cfg.step_size_samples,
        label_col=label_col,
        config=cfg,
        group_cols=group_cols,
    )