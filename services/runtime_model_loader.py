"""Schema-aware loader for the deployable runtime model bundle.

The deployable bundle (``artifacts/unifallmonitor/deployable/runtime_v1.joblib``)
is what the API and mobile consumers load at startup. This module hides the
schema-version dispatch so callers don't have to know the bundle layout.

If the schema version is unknown, the loader refuses rather than silently
mis-decoding — the alternative is a model running with a feature column
order that doesn't match the trained model, producing nonsense predictions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

SUPPORTED_SCHEMA_VERSIONS = {"v1"}


@dataclass(frozen=True)
class RuntimeBundle:
    schema_version: str
    model: Any
    feature_cols: tuple[str, ...]
    labels: tuple[str, ...]
    smoothing: Mapping[str, Any]
    fall_threshold: float
    metadata: Mapping[str, Any]
    trained_at_iso: str
    git_sha: str
    run_id: str

    def predict_fall_probabilities(self, X) -> Any:
        """Return P(fall) for each input row.

        The model is expected to expose ``predict_proba``; we look up the
        column matching the ``"fall"`` label rather than assuming a fixed
        index, so a label reordering in a future bundle doesn't silently
        flip the threshold sweep.
        """
        proba = self.model.predict_proba(X)
        if "fall" not in self.labels:
            raise RuntimeError("Bundle does not include a 'fall' label")
        col = self.labels.index("fall")
        return proba[:, col]


_REQUIRED_METADATA_KEYS = ("target_rate_hz", "window_size", "step_size")


def load_bundle(path: Path | str) -> RuntimeBundle:
    """Load and validate a deployable bundle from disk.

    Validates the ``metadata`` block carries the training-time preprocess
    params the inference path needs (rate / window / step). The 2026-04-30
    runtime_v1 incident was caused by a bundle missing this block: the
    legacy loader silently shimmed it, inference fell back to default
    ``window_size=128``, and HAR labels degenerated.
    """
    import joblib

    path = Path(path)
    raw = joblib.load(path)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: bundle must be a dict, got {type(raw).__name__}")

    version = raw.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"{path}: unsupported schema_version={version!r}; "
            f"this build supports {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )

    required = ("model", "feature_cols", "labels", "fall_threshold", "metadata")
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"{path}: bundle missing required keys: {missing}")

    metadata = raw["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError(
            f"{path}: 'metadata' must be a dict, got {type(metadata).__name__}"
        )
    missing_meta = [k for k in _REQUIRED_METADATA_KEYS if metadata.get(k) is None]
    if missing_meta:
        raise ValueError(
            f"{path}: bundle metadata missing required preprocess keys: "
            f"{missing_meta}. Re-export with scripts/export_runtime_artifact.py."
        )

    return RuntimeBundle(
        schema_version=str(version),
        model=raw["model"],
        feature_cols=tuple(raw["feature_cols"]),
        labels=tuple(raw["labels"]),
        smoothing=dict(raw.get("smoothing") or {}),
        fall_threshold=float(raw["fall_threshold"]),
        metadata=dict(metadata),
        trained_at_iso=str(raw.get("trained_at_iso", "")),
        git_sha=str(raw.get("git_sha", "")),
        run_id=str(raw.get("run_id", "")),
    )
