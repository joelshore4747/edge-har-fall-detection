"""End-to-end smoke test for the deployable runtime bundle.

Skips cleanly when no bundle has been built (CI on a fresh checkout
won't have one). When a bundle exists, this test guards against the
single highest-risk class of deployment regression: train-serve skew
in the schema-versioned bundle layout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from services.runtime_model_loader import SUPPORTED_SCHEMA_VERSIONS, load_bundle

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_PATH = REPO_ROOT / "artifacts" / "unifallmonitor" / "deployable" / "runtime_v1.joblib"


@pytest.fixture(scope="module")
def bundle():
    if not BUNDLE_PATH.exists():
        pytest.skip(
            f"runtime bundle not built at {BUNDLE_PATH.relative_to(REPO_ROOT)} — "
            "run `python scripts/export_runtime_artifact.py --run current` to build one"
        )
    return load_bundle(BUNDLE_PATH)


def test_bundle_schema_is_supported(bundle):
    assert bundle.schema_version in SUPPORTED_SCHEMA_VERSIONS


def test_bundle_carries_provenance(bundle):
    # The whole point of the schema is that "what's deployed" is traceable.
    assert bundle.run_id, "bundle missing run_id provenance"
    assert bundle.trained_at_iso, "bundle missing trained_at_iso provenance"
    # git_sha is required for runs minted by the registry, but legacy
    # migrated runs (run_id suffix "_legacy") predate the SHA capture
    # and are allowed through.
    if not bundle.run_id.endswith("_legacy"):
        assert bundle.git_sha, "non-legacy bundle missing git_sha provenance"


def test_bundle_feature_cols_and_labels_are_frozen_tuples(bundle):
    # Frozen tuples (not lists) is what stops accidental mutation
    # leading to feature-order drift between calls.
    assert isinstance(bundle.feature_cols, tuple)
    assert isinstance(bundle.labels, tuple)
    assert len(bundle.feature_cols) > 0
    assert len(bundle.labels) > 0


def test_bundle_predicts_on_synthetic_row(bundle):
    # Construct a single zero-valued row matching the bundle's feature
    # contract; we don't care what it predicts, only that the shape and
    # the fall-probability lookup work end-to-end.
    X = np.zeros((1, len(bundle.feature_cols)), dtype=float)
    proba = bundle.model.predict_proba(X)
    assert proba.shape == (1, len(bundle.labels))


def test_bundle_predict_fall_probabilities_returns_one_per_row(bundle):
    if "fall" not in bundle.labels:
        pytest.skip("bundle does not include a 'fall' label")
    X = np.zeros((3, len(bundle.feature_cols)), dtype=float)
    p_fall = bundle.predict_fall_probabilities(X)
    assert p_fall.shape == (3,)
    assert ((p_fall >= 0.0) & (p_fall <= 1.0)).all()


def test_bundle_fall_threshold_is_sane(bundle):
    assert 0.0 <= bundle.fall_threshold <= 1.0


def test_bundle_exposes_metadata_via_schema_loader(bundle):
    """The schema-aware loader must surface the metadata block.

    Inference paths that load through ``runtime_model_loader.load_bundle``
    rely on ``bundle.metadata`` to source training-time preprocess params.
    """
    assert isinstance(bundle.metadata, dict)
    for key in ("target_rate_hz", "window_size", "step_size"):
        assert key in bundle.metadata, f"bundle.metadata missing {key}"


def test_load_bundle_rejects_missing_metadata(tmp_path):
    """Schema-aware loader fails loudly when metadata is missing or partial.

    Without this guard, the legacy HAR loader's v1 shim silently fills in
    defaults and inference computes features over the wrong window length
    (the 2026-04-30 runtime_v1 incident).
    """
    import joblib

    from services.runtime_model_loader import load_bundle

    bad = tmp_path / "bad.joblib"
    joblib.dump(
        {
            "schema_version": "v1",
            "model": object(),
            "feature_cols": ["a", "b"],
            "labels": ["fall", "static"],
            "smoothing": {"mode": "none", "window": 1},
            "fall_threshold": 0.5,
            # metadata intentionally absent
        },
        bad,
    )
    with pytest.raises(ValueError, match="missing required keys"):
        load_bundle(bad)

    partial = tmp_path / "partial.joblib"
    joblib.dump(
        {
            "schema_version": "v1",
            "model": object(),
            "feature_cols": ["a", "b"],
            "labels": ["fall", "static"],
            "smoothing": {"mode": "none", "window": 1},
            "fall_threshold": 0.5,
            "metadata": {"target_rate_hz": 50.0},  # window_size / step_size missing
        },
        partial,
    )
    with pytest.raises(ValueError, match="missing required preprocess keys"):
        load_bundle(partial)


def test_bundle_metadata_drives_inference_window_params():
    """Regression for the runtime_v1 incident on 2026-04-30.

    The bundle must carry its training-time preprocess params under
    ``metadata`` so ``services.runtime_inference._artifact_har_preprocess``
    windows the live signal at the same rate/size the model saw during
    training. When this block is missing, the legacy loader silently
    falls back to ``window_size=128``, every feature is computed over the
    wrong window length, and HAR labels degenerate (a static phone gets
    classified as ``stairs``/blank, which silently disables the
    static-gate that suppresses fall false positives).
    """
    if not BUNDLE_PATH.exists():
        pytest.skip("runtime bundle not built")

    import joblib

    from services.runtime_inference import (
        RuntimeInferenceConfig,
        _artifact_har_preprocess,
    )

    raw = joblib.load(BUNDLE_PATH)
    assert isinstance(raw, dict)
    metadata = raw.get("metadata")
    assert isinstance(metadata, dict), (
        "bundle missing top-level 'metadata' block — inference will silently "
        "fall back to default window_size=128"
    )
    for key in ("target_rate_hz", "window_size", "step_size"):
        assert key in metadata, f"bundle metadata missing required key: {key}"

    resolved = _artifact_har_preprocess(BUNDLE_PATH, RuntimeInferenceConfig())
    assert resolved["target_rate_hz"] == float(metadata["target_rate_hz"])
    assert resolved["window_size"] == int(metadata["window_size"])
    assert resolved["step_size"] == int(metadata["step_size"])
