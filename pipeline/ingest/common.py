from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from pipeline.schema import COMMON_SCHEMA_COLUMNS, TASK_FALL, TASK_HAR
from pipeline.validation import validate_ingestion_dataframe

_HAR_EXACT_MAP: Dict[str, str] = {
    "walking": "walking",
    "walk": "walking",
    "jogging": "walking",
    "running": "walking",

    "sitting": "static",
    "standing": "static",
    "lying": "static",
    "laying": "static",
    "lying_down": "static",

    "ascending_stairs": "stairs",
    "descending_stairs": "stairs",
    "walking_upstairs": "stairs",
    "walking_downstairs": "stairs",
    "stairs_up": "stairs",
    "stairs_down": "stairs",

    "cycling": "other",
    "nordic_walking": "other",

    "watching_tv": "static",
    "computer_work": "static",
    "car_driving": "static",
    "vacuum_cleaning": "other",
    "ironing": "other",
    "folding_laundry": "other",
    "house_cleaning": "other",
    "playing_soccer": "other",
    "rope_jumping": "other",
    "other_transition": "other",
}

_FALL_EXACT_MAP: Dict[str, str] = {
    "fall": "fall",
    "fallen": "fall",
    "non_fall": "non_fall",
    "nonfall": "non_fall",
    "adl": "non_fall",
    "walk": "non_fall",
    "walking": "non_fall",
    "sit": "non_fall",
    "standing": "non_fall",
    "standing_up": "non_fall",
    "sitting_down": "non_fall",
    "jogging": "non_fall",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        key = key.replace("-", "_")
        renamed[col] = key
    return df.rename(columns=renamed)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def add_missing_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        "session_id": pd.NA,
        "timestamp": np.nan,
        "gx": np.nan,
        "gy": np.nan,
        "gz": np.nan,
        "placement": pd.NA,
        "sampling_rate_hz": np.nan,
    }
    for col, default_value in defaults.items():
        if col not in df.columns:
            df[col] = default_value
    return df


def ensure_common_schema_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = add_missing_optional_columns(df)
    for col in COMMON_SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[COMMON_SCHEMA_COLUMNS].copy()


def _normalize_label_token(label: object) -> str:
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return ""
    text = str(label).strip().lower()
    for ch in ["-", "/", "\\", "(", ")", "."]:
        text = text.replace(ch, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def map_har_label(raw_label: object, *, unknown_strategy: str = "other") -> str:
    token = _normalize_label_token(raw_label)
    if token in _HAR_EXACT_MAP:
        return _HAR_EXACT_MAP[token]

    # Keyword-based fallbacks for less standardized labels.
    if "stair" in token or "upstairs" in token or "downstairs" in token:
        return "stairs"

    if any(k in token for k in ["walk", "run", "jog"]):
        return "walking"

    if any(k in token for k in ["sit", "stand", "lay", "lie", "lying"]):
        return "static"

    if unknown_strategy == "other":
        return "other"
    if unknown_strategy == "raise":
        raise ValueError(f"Unknown HAR raw label: {raw_label!r}")
    raise ValueError(f"Unsupported unknown_strategy for HAR labels: {unknown_strategy}")

def map_fall_label(raw_label: object, *, unknown_strategy: str = "non_fall") -> str:
    token = _normalize_label_token(raw_label)
    if token in _FALL_EXACT_MAP:
        return _FALL_EXACT_MAP[token]

    if "fall" in token and "non" not in token:
        return "fall"
    if any(k in token for k in ["adl", "walk", "sit", "stand", "jog", "run"]):
        return "non_fall"

    if unknown_strategy == "non_fall":
        return "non_fall"
    if unknown_strategy == "other":
        return "non_fall"
    if unknown_strategy == "raise":
        raise ValueError(f"Unknown fall raw label: {raw_label!r}")
    raise ValueError(f"Unsupported unknown_strategy for fall labels: {unknown_strategy}")


def map_label(raw_label: object, task_type: str, *, unknown_strategy: str | None = None) -> str:
    if task_type == TASK_HAR:
        return map_har_label(raw_label, unknown_strategy=unknown_strategy or "other")
    if task_type == TASK_FALL:
        return map_fall_label(raw_label, unknown_strategy=unknown_strategy or "non_fall")
    raise ValueError(f"Unsupported task_type for label mapping: {task_type}")


def apply_label_mapping(
    df: pd.DataFrame,
    *,
    task_type: str,
    raw_label_col: str = "label_raw",
    mapped_label_col: str = "label_mapped",
    unknown_strategy: str | None = None,
) -> pd.DataFrame:
    df[mapped_label_col] = df[raw_label_col].apply(
        lambda v: map_label(v, task_type, unknown_strategy=unknown_strategy)
    )
    return df


def apply_numeric_coercions(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = safe_numeric(df[col])
    return df


def inject_metadata(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    task_type: str,
    source_file: str | Path,
    placement: str | None = None,
    sampling_rate_hz: float | None = None,
) -> pd.DataFrame:
    df["dataset_name"] = dataset_name
    df["task_type"] = task_type
    if placement is not None and "placement" in df.columns:
        df["placement"] = df["placement"].fillna(placement) if hasattr(df["placement"], "fillna") else placement
    elif placement is not None:
        df["placement"] = placement
    if sampling_rate_hz is not None and "sampling_rate_hz" in df.columns:
        df["sampling_rate_hz"] = df["sampling_rate_hz"].fillna(float(sampling_rate_hz))
    elif sampling_rate_hz is not None:
        df["sampling_rate_hz"] = float(sampling_rate_hz)
    if "source_file" in df.columns:
        df["source_file"] = df["source_file"].fillna(str(source_file))
    else:
        df["source_file"] = str(source_file)
    if "row_index" not in df.columns:
        df["row_index"] = np.arange(len(df), dtype=int)
    return df


def finalize_ingest_dataframe(df: pd.DataFrame, *, validate: bool = True) -> pd.DataFrame:
    df = ensure_common_schema_columns(df)

    apply_numeric_coercions(df, ["timestamp", "ax", "ay", "az", "gx", "gy", "gz", "sampling_rate_hz", "row_index"])

    for col in ["dataset_name", "task_type", "subject_id", "session_id", "label_raw", "label_mapped", "placement", "source_file"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    if validate:
        result = validate_ingestion_dataframe(df)
        result.raise_for_errors()
    return df


def read_csv_any(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)
