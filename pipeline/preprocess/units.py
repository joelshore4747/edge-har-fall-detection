"""Sensor unit harmonization helpers.

Purpose:
- keep accelerometer and gyroscope units consistent across datasets
- provide one reusable normalization path for loaders
- attach lightweight normalization metadata to DataFrame attrs

Canonical target units for the dissertation pipeline:
- accelerometer: m/s^2
- gyroscope: rad/s
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

GRAVITY_M_S2 = 9.80665
DEG_TO_RAD = np.pi / 180.0

ACCEL_AXES = ("ax", "ay", "az")
GYRO_AXES = ("gx", "gy", "gz")


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _convert_axes(df: pd.DataFrame, axes: tuple[str, ...], factor: float) -> pd.DataFrame:
    out = df.copy()
    for col in axes:
        if col in out.columns:
            out[col] = _safe_numeric(out[col]) * float(factor)
    return out


def convert_accel_g_to_m_s2(df: pd.DataFrame) -> pd.DataFrame:
    """Convert accelerometer axes from g to m/s^2."""
    return _convert_axes(df, ACCEL_AXES, GRAVITY_M_S2)


def convert_accel_m_s2_to_g(df: pd.DataFrame) -> pd.DataFrame:
    """Convert accelerometer axes from m/s^2 to g."""
    return _convert_axes(df, ACCEL_AXES, 1.0 / GRAVITY_M_S2)


def convert_gyro_deg_s_to_rad_s(df: pd.DataFrame) -> pd.DataFrame:
    """Convert gyroscope axes from deg/s to rad/s."""
    return _convert_axes(df, GYRO_AXES, DEG_TO_RAD)


def convert_gyro_rad_s_to_deg_s(df: pd.DataFrame) -> pd.DataFrame:
    """Convert gyroscope axes from rad/s to deg/s."""
    return _convert_axes(df, GYRO_AXES, 1.0 / DEG_TO_RAD)


def _append_loader_note(df: pd.DataFrame, note: str) -> None:
    notes = list(df.attrs.get("loader_notes", []))
    notes.append(note)
    df.attrs["loader_notes"] = notes


def _record_unit_metadata(
    df: pd.DataFrame,
    *,
    source_accel_unit: str | None,
    source_gyro_unit: str | None,
    target_accel_unit: str,
    target_gyro_unit: str,
) -> None:
    unit_meta: dict[str, Any] = dict(df.attrs.get("unit_normalization", {}))
    unit_meta.update(
        {
            "source_accel_unit": source_accel_unit,
            "source_gyro_unit": source_gyro_unit,
            "target_accel_unit": target_accel_unit,
            "target_gyro_unit": target_gyro_unit,
        }
    )
    df.attrs["unit_normalization"] = unit_meta


def normalize_sensor_units(
    df: pd.DataFrame,
    *,
    source_accel_unit: str | None = None,
    source_gyro_unit: str | None = None,
    target_accel_unit: str = "m_s2",
    target_gyro_unit: str = "rad_s",
) -> pd.DataFrame:
    """Normalize sensor units onto the dissertation pipeline defaults.

    Supported source units:
    - accelerometer: "g", "m_s2"
    - gyroscope: "deg_s", "rad_s"

    If a source unit is None, that sensor family is left unchanged.
    """
    out = df.copy()

    if source_accel_unit is not None and source_accel_unit != target_accel_unit:
        if source_accel_unit == "g" and target_accel_unit == "m_s2":
            out = convert_accel_g_to_m_s2(out)
            _append_loader_note(out, "Converted accelerometer axes from g to m/s^2.")
        elif source_accel_unit == "m_s2" and target_accel_unit == "g":
            out = convert_accel_m_s2_to_g(out)
            _append_loader_note(out, "Converted accelerometer axes from m/s^2 to g.")
        else:
            raise ValueError(
                f"Unsupported accelerometer conversion: {source_accel_unit} -> {target_accel_unit}"
            )

    if source_gyro_unit is not None and source_gyro_unit != target_gyro_unit:
        if source_gyro_unit == "deg_s" and target_gyro_unit == "rad_s":
            out = convert_gyro_deg_s_to_rad_s(out)
            _append_loader_note(out, "Converted gyroscope axes from deg/s to rad/s.")
        elif source_gyro_unit == "rad_s" and target_gyro_unit == "deg_s":
            out = convert_gyro_rad_s_to_deg_s(out)
            _append_loader_note(out, "Converted gyroscope axes from rad/s to deg/s.")
        else:
            raise ValueError(
                f"Unsupported gyroscope conversion: {source_gyro_unit} -> {target_gyro_unit}"
            )

    _record_unit_metadata(
        out,
        source_accel_unit=source_accel_unit,
        source_gyro_unit=source_gyro_unit,
        target_accel_unit=target_accel_unit,
        target_gyro_unit=target_gyro_unit,
    )
    return out


def vector_magnitude(df: pd.DataFrame, axes: tuple[str, ...]) -> pd.Series:
    """Convenience helper for quick loader-side diagnostics/tests."""
    available = [col for col in axes if col in df.columns]
    if not available:
        return pd.Series(dtype=float)

    squared = pd.DataFrame({col: _safe_numeric(df[col]).pow(2) for col in available})
    return np.sqrt(squared.sum(axis=1))


def accel_magnitude(df: pd.DataFrame) -> pd.Series:
    return vector_magnitude(df, ACCEL_AXES)


def gyro_magnitude(df: pd.DataFrame) -> pd.Series:
    return vector_magnitude(df, GYRO_AXES)