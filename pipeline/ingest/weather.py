"""Lightweight weather CSV loaders for context ingestion readiness (Chapter 3 patch).

This module is intentionally simple and CSV-focused so it can be used for:
- Open-Meteo style CSVs (e.g., ``time,pressure_msl``)
- Meteostat exports with a time column and numeric measurements

The goal is not full context fusion yet; it is to provide a clean, testable loader for
later alignment and feature engineering lessons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _infer_location_name_from_filename(path: Path) -> str:
    stem = path.stem
    for prefix in ["weather_", "open_meteo_", "meteostat_hourly_"]:
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
    return stem


def load_weather_csv(
    path: str | Path,
    *,
    location_name: str | None = None,
    time_col: str = "time",
) -> pd.DataFrame:
    """Load a single weather CSV and attach minimal metadata."""
    src_path = Path(path)
    df = pd.read_csv(src_path)

    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # Coerce non-time, non-string metadata columns to numeric where practical.
    for col in df.columns:
        if col == time_col:
            continue
        if col in {"location_name", "source_file"}:
            continue
        if df[col].dtype == object:
            coerced = pd.to_numeric(df[col], errors="coerce")
            # Keep original text column if coercion would destroy all values.
            if not coerced.isna().all():
                df[col] = coerced

    df["location_name"] = location_name or _infer_location_name_from_filename(src_path)
    df["source_file"] = str(src_path)
    return df


def load_weather_csvs(
    paths: Iterable[str | Path],
    *,
    time_col: str = "time",
) -> pd.DataFrame:
    """Load and concatenate multiple weather CSVs."""
    frames: list[pd.DataFrame] = []
    for path in paths:
        frames.append(load_weather_csv(path, time_col=time_col))
    if not frames:
        return pd.DataFrame(columns=[time_col, "location_name", "source_file"])
    return pd.concat(frames, ignore_index=True)


__all__ = ["load_weather_csv", "load_weather_csvs"]
