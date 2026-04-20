"""Unified dataset schema for Chapter 3 ingestion scaffolding.

This module defines the common record-level schema used by dataset-specific loaders.
The schema is intentionally simple and dataframe-oriented for an undergraduate project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

TASK_HAR = "har"
TASK_FALL = "fall"
ALLOWED_TASK_TYPES = {TASK_HAR, TASK_FALL}

HAR_MAPPED_LABELS = {"static", "locomotion", "stairs", "other"}
FALL_MAPPED_LABELS = {"fall", "non_fall"}

COMMON_SCHEMA_COLUMNS: List[str] = [
    "dataset_name",
    "task_type",
    "subject_id",
    "session_id",
    "timestamp",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
    "label_raw",
    "label_mapped",
    "placement",
    "sampling_rate_hz",
    "source_file",
    "row_index",
]

REQUIRED_CORE_COLUMNS: Tuple[str, ...] = (
    "dataset_name",
    "task_type",
    "subject_id",
    "ax",
    "ay",
    "az",
    "label_raw",
    "label_mapped",
    "source_file",
    "row_index",
)

NULLABLE_SCHEMA_COLUMNS = {
    "session_id",
    "timestamp",
    "gx",
    "gy",
    "gz",
    "placement",
    "sampling_rate_hz",
}

NUMERIC_COLUMNS = {
    "timestamp",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
    "sampling_rate_hz",
    "row_index",
}


@dataclass(frozen=True)
class DatasetIngestMetadata:
    dataset_name: str
    task_type: str
    row_count: int
    source_path: str


@dataclass(frozen=True)
class SchemaCheckResult:
    ok: bool
    missing_columns: Tuple[str, ...]
    unexpected_columns: Tuple[str, ...]
    invalid_task_types: Tuple[str, ...]


def build_empty_common_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=COMMON_SCHEMA_COLUMNS)


def check_common_schema_columns(df: pd.DataFrame) -> SchemaCheckResult:
    cols = list(df.columns)
    missing = tuple(col for col in COMMON_SCHEMA_COLUMNS if col not in cols)
    unexpected = tuple(col for col in cols if col not in COMMON_SCHEMA_COLUMNS)

    invalid_task_types: Tuple[str, ...] = tuple()
    if "task_type" in df.columns:
        values = {str(v) for v in df["task_type"].dropna().astype(str).unique().tolist()}
        invalid = sorted(v for v in values if v not in ALLOWED_TASK_TYPES)
        invalid_task_types = tuple(invalid)

    return SchemaCheckResult(
        ok=(len(missing) == 0 and len(invalid_task_types) == 0),
        missing_columns=missing,
        unexpected_columns=unexpected,
        invalid_task_types=invalid_task_types,
    )


def require_common_schema(df: pd.DataFrame) -> None:
    result = check_common_schema_columns(df)
    errors: List[str] = []
    if result.missing_columns:
        errors.append(f"Missing common schema columns: {', '.join(result.missing_columns)}")
    if result.invalid_task_types:
        errors.append(f"Invalid task_type values: {', '.join(result.invalid_task_types)}")
    if errors:
        raise ValueError("; ".join(errors))


def ordered_common_columns(columns: Iterable[str]) -> List[str]:
    column_list = list(columns)
    ordered = [c for c in COMMON_SCHEMA_COLUMNS if c in column_list]
    ordered.extend([c for c in column_list if c not in ordered])
    return ordered


__all__ = [
    "TASK_HAR",
    "TASK_FALL",
    "ALLOWED_TASK_TYPES",
    "HAR_MAPPED_LABELS",
    "FALL_MAPPED_LABELS",
    "COMMON_SCHEMA_COLUMNS",
    "REQUIRED_CORE_COLUMNS",
    "NULLABLE_SCHEMA_COLUMNS",
    "NUMERIC_COLUMNS",
    "DatasetIngestMetadata",
    "SchemaCheckResult",
    "build_empty_common_frame",
    "check_common_schema_columns",
    "require_common_schema",
    "ordered_common_columns",
]
