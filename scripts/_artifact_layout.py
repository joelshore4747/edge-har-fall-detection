"""Shared helpers for canonical artifact-layout scripts.

Used by scripts/organize_fall_artifacts.py and scripts/organize_har_artifacts.py
so both promotion workflows share the same file I/O and JSON-safety semantics
without a cross-import.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def safe_copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def json_safe(value: Any) -> Any:
    """Convert numpy/pandas scalars and containers to JSON-serialisable values.

    Imported lazily so this module stays importable in environments that don't
    have pandas/numpy installed (e.g. lightweight CI jobs that only read
    metadata).
    """
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)

    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        np = None
        pd = None

    if pd is not None:
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")
        if isinstance(value, pd.Series):
            return json_safe(value.to_dict())

    if np is not None:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            f = float(value)
            return f if np.isfinite(f) else None
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.ndarray):
            return json_safe(value.tolist())

    if isinstance(value, float):
        import math

        if not math.isfinite(value):
            return None

    if pd is not None:
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass

    return value


def ensure_repo_on_syspath() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
