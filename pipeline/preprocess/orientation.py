"""Orientation-robust derived channels for Chapter 3 preprocessing.

Uses simple magnitude features as a first step toward orientation robustness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def append_acc_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = ["ax", "ay", "az"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"append_acc_magnitude requires columns: {', '.join(required)}")
    out["acc_magnitude"] = np.sqrt(
        pd.to_numeric(out["ax"], errors="coerce") ** 2
        + pd.to_numeric(out["ay"], errors="coerce") ** 2
        + pd.to_numeric(out["az"], errors="coerce") ** 2
    )
    return out


def append_gyro_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    gyro_cols = ["gx", "gy", "gz"]
    if not all(col in out.columns for col in gyro_cols):
        out["gyro_magnitude"] = np.nan
        return out

    gx = pd.to_numeric(out["gx"], errors="coerce")
    gy = pd.to_numeric(out["gy"], errors="coerce")
    gz = pd.to_numeric(out["gz"], errors="coerce")
    out["gyro_magnitude"] = np.sqrt(gx**2 + gy**2 + gz**2)
    return out


def append_derived_channels(df: pd.DataFrame, *, include_acc: bool = True, include_gyro: bool = True) -> pd.DataFrame:
    out = df.copy()
    if include_acc:
        out = append_acc_magnitude(out)
    if include_gyro:
        out = append_gyro_magnitude(out)
    return out
