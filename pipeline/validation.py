from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd

from pipeline.schema import (
    ALLOWED_TASK_TYPES,
    COMMON_SCHEMA_COLUMNS,
    NUMERIC_COLUMNS,
    REQUIRED_CORE_COLUMNS,
    check_common_schema_columns,
)


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError(" | ".join(self.errors))


def _coercion_failure_count(series: pd.Series, *, allow_nulls: bool = True) -> int:
    coerced = pd.to_numeric(series, errors="coerce")
    original_non_null = series.notna()
    if allow_nulls:
        return int(((coerced.isna()) & original_non_null).sum())
    return int(coerced.isna().sum())


def validate_ingestion_dataframe(df: pd.DataFrame) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []

    schema_check = check_common_schema_columns(df)
    if schema_check.missing_columns:
        errors.append(f"Missing common schema columns: {', '.join(schema_check.missing_columns)}")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    if schema_check.invalid_task_types:
        errors.append(f"Invalid task_type values: {', '.join(schema_check.invalid_task_types)}")

    for col in REQUIRED_CORE_COLUMNS:
        if col not in df.columns:
            continue
        if df[col].isna().any():
            errors.append(f"Column '{col}' contains null values")
            continue
        if df[col].dtype == object:
            empty_count = int(df[col].astype(str).str.strip().eq("").sum())
            if empty_count > 0:
                errors.append(f"Column '{col}' contains empty strings ({empty_count} rows)")

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        allow_nulls = col in {"timestamp", "gx", "gy", "gz", "sampling_rate_hz"}
        bad_count = _coercion_failure_count(df[col], allow_nulls=allow_nulls)
        if bad_count > 0:
            errors.append(f"Column '{col}' has {bad_count} non-numeric values")

    for axis in ("ax", "ay", "az"):
        if axis in df.columns and df[axis].isna().any():
            errors.append(f"Axis column '{axis}' contains null values")

    for col in ("label_raw", "label_mapped"):
        if col in df.columns and df[col].astype(str).str.strip().eq("").any():
            errors.append(f"Column '{col}' contains empty labels")

    for gyro_col in ("gx", "gy", "gz"):
        if gyro_col in df.columns:
            null_pct = float(df[gyro_col].isna().mean() * 100) if len(df) else 0.0
            if null_pct >= 50:
                warnings.append(f"Column '{gyro_col}' is {null_pct:.1f}% null (expected for accelerometer-only datasets)")

    if "timestamp" in df.columns:
        ts_null_pct = float(df["timestamp"].isna().mean() * 100) if len(df) else 0.0
        if ts_null_pct > 0:
            warnings.append(f"Column 'timestamp' is {ts_null_pct:.1f}% null")

    if "task_type" in df.columns:
        seen = sorted(set(df["task_type"].dropna().astype(str).tolist()))
        unknown = [v for v in seen if v not in ALLOWED_TASK_TYPES]
        if unknown:
            errors.append(f"Unexpected task_type values found: {', '.join(unknown)}")

    if list(df.columns) != COMMON_SCHEMA_COLUMNS:
        warnings.append("DataFrame column order differs from COMMON_SCHEMA_COLUMNS")

    if bool(getattr(df, "attrs", {}).get("is_prewindowed", False)):
        warnings.append("DataFrame source is pre-windowed (flattened to sample rows); treat as non-continuous origin")

    loader_notes = getattr(df, "attrs", {}).get("loader_notes")
    if isinstance(loader_notes, (list, tuple)):
        for note in loader_notes:
            if note:
                warnings.append(f"loader_note: {note}")

    return ValidationResult(is_valid=(len(errors) == 0), errors=errors, warnings=warnings)


def assert_valid_ingestion_dataframe(df: pd.DataFrame) -> None:
    result = validate_ingestion_dataframe(df)
    result.raise_for_errors()
