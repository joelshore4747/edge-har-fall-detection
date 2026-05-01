"""False-alarm inspection helpers shared by evaluation and reporting code."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


FALSE_ALARM_CORE_COLUMNS = [
    "window_id",
    "dataset_name",
    "subject_id",
    "session_id",
    "source_file",
    "peak_acc",
    "post_impact_motion",
    "predicted_label",
    "true_label",
    "detector_reason",
    "stage_impact_pass",
    "stage_support_pass",
    "stage_confirm_pass",
]


def build_false_alarm_table(predictions_df: pd.DataFrame) -> pd.DataFrame:
    if predictions_df.empty:
        return predictions_df.copy()

    required = {"predicted_label", "true_label"}
    if not required.issubset(predictions_df.columns):
        raise ValueError("Prediction dataframe must contain predicted_label and true_label")

    mask = (
        predictions_df["predicted_label"].astype(str).eq("fall")
        & predictions_df["true_label"].astype(str).eq("non_fall")
    )
    out = predictions_df.loc[mask].copy()
    cols = [c for c in FALSE_ALARM_CORE_COLUMNS if c in out.columns]
    extra = [c for c in out.columns if c not in cols]
    return out[cols + sorted(extra)].reset_index(drop=True)


def save_false_alarm_csv(false_alarm_df: pd.DataFrame, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    false_alarm_df.to_csv(out, index=False)
    return out
