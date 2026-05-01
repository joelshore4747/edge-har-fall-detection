"""Smoke test for ``scripts/train_unifallmonitor_har`` end-to-end.

Mirrors ``tests/test_har_baseline_smoke.py``: synthesise a few raw session
payloads in the shape returned by ``GET /v1/sessions/{id}/raw``, run them
through ``samples_to_dataframe`` → ``build_features_for_dataframe`` →
``train_and_evaluate``, and assert the returned model + report look sane.

This protects against:
  - the canonical-label helper drifting from ``apps/api/schemas``,
  - ``select_feature_columns`` / ``prepare_feature_matrices`` integration breaking,
  - the train/test split branches throwing on small synthetic corpora.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.har.train_har import select_feature_columns
from scripts.train_unifallmonitor_har import (
    build_features_for_dataframe,
    canonicalize_activity_label,
    samples_to_dataframe,
    train_and_evaluate,
)


_RATE_HZ = 50.0
_DURATION_SEC = 10.0


def _make_session_samples(*, label: str, seed: int) -> list[dict]:
    """Create separable synthetic IMU samples for a given canonical label."""
    rng = np.random.default_rng(seed)
    n = int(_RATE_HZ * _DURATION_SEC)
    timestamps = np.arange(n) / _RATE_HZ

    if label == "walking":
        ax = 0.5 * np.sin(2 * np.pi * 1.7 * timestamps) + rng.normal(0, 0.05, n)
        ay = 0.4 * np.sin(2 * np.pi * 1.7 * timestamps + 0.3) + rng.normal(0, 0.05, n)
        az = 9.81 + 0.3 * np.sin(2 * np.pi * 1.7 * timestamps) + rng.normal(0, 0.05, n)
    elif label == "stairs":
        ax = 0.6 * np.sin(2 * np.pi * 1.0 * timestamps) + rng.normal(0, 0.08, n)
        ay = 0.5 * np.sin(2 * np.pi * 1.0 * timestamps + 0.5) + rng.normal(0, 0.08, n)
        az = 9.81 + 0.8 * np.sin(2 * np.pi * 1.0 * timestamps) + rng.normal(0, 0.08, n)
    else:  # static / other
        ax = rng.normal(0, 0.02, n)
        ay = rng.normal(0, 0.02, n)
        az = 9.81 + rng.normal(0, 0.02, n)

    return [
        {
            "timestamp": float(t),
            "ax": float(a1),
            "ay": float(a2),
            "az": float(a3),
            "gx": 0.0,
            "gy": 0.0,
            "gz": 0.0,
        }
        for t, a1, a2, a3 in zip(timestamps, ax, ay, az)
    ]


def _make_raw_payload(*, subject_id: str, label: str, placement: str, seed: int) -> dict:
    return {
        "stored_at": "2026-04-27T00:00:00Z",
        "request": {
            "metadata": {
                "session_id": f"{subject_id}_{label}_{seed}",
                "subject_id": subject_id,
                "placement": placement,
                "dataset_name": "TEST",
                "device_platform": "test",
                "task_type": "har",
                "activity_label": label,
                "sampling_rate_hz": _RATE_HZ,
            },
            "samples": _make_session_samples(label=label, seed=seed),
        },
    }


def _build_feature_table(payloads: list[dict]) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for index, payload in enumerate(payloads):
        request = payload["request"]
        canonical = canonicalize_activity_label(request["metadata"]["activity_label"])
        assert canonical is not None, "synthetic label should canonicalize"
        df = samples_to_dataframe(
            metadata=request["metadata"],
            samples=request["samples"],
            canonical_label=canonical,
            session_index=index,
        )
        assert df is not None and not df.empty
        feats = build_features_for_dataframe(
            df, target_rate_hz=_RATE_HZ, window_size=100, step_size=50
        )
        if not feats.empty:
            chunks.append(feats)
    return pd.concat(chunks, ignore_index=True)


def test_canonicalize_activity_label_matches_api_aliases():
    assert canonicalize_activity_label("walking") == "walking"
    assert canonicalize_activity_label("WALK") == "walking"
    assert canonicalize_activity_label("downstairs") == "stairs"
    assert canonicalize_activity_label("sitting") == "static"
    assert canonicalize_activity_label("falling") == "fall"
    assert canonicalize_activity_label("unknown") is None
    assert canonicalize_activity_label("") is None
    assert canonicalize_activity_label(None) is None
    assert canonicalize_activity_label("not-a-label") is None


def test_train_and_evaluate_end_to_end_on_synthetic_payloads():
    pairs = [
        ("subj_a", "walking"),
        ("subj_a", "static"),
        ("subj_a", "walking"),
        ("subj_b", "walking"),
        ("subj_b", "static"),
        ("subj_b", "static"),
    ]
    payloads = [
        _make_raw_payload(subject_id=sid, label=lbl, placement="pocket", seed=i)
        for i, (sid, lbl) in enumerate(pairs)
    ]

    feature_df = _build_feature_table(payloads)
    assert not feature_df.empty
    assert feature_df["label_mapped_majority"].nunique() >= 2

    result = train_and_evaluate(
        feature_df,
        label_col="label_mapped_majority",
        test_size=0.3,
        random_state=42,
        n_estimators=50,
    )

    # Shape of the result.
    assert "model" in result
    assert "feature_cols" in result
    assert "labels" in result
    assert "split_mode" in result
    assert "confusion_matrix" in result
    assert "report" in result

    # Feature columns we use must be a subset of the canonical numeric set —
    # no metadata leakage as features.
    canonical = set(select_feature_columns(feature_df))
    assert set(result["feature_cols"]).issubset(canonical)

    # Metrics in [0, 1].
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["macro_f1"] <= 1.0

    # Confusion matrix shape matches labels.
    cm = result["confusion_matrix"]
    assert len(cm) == len(result["labels"])
    assert all(len(row) == len(result["labels"]) for row in cm)

    # Train/test row counts add up.
    assert result["train_rows"] + result["test_rows"] == len(feature_df)


@pytest.mark.parametrize("split_mode", ["stratified", "auto"])
def test_train_and_evaluate_split_modes(split_mode):
    payloads = [
        _make_raw_payload(subject_id=f"s{i}", label=lbl, placement="pocket", seed=i)
        for i, lbl in enumerate(["walking", "static", "walking", "static"])
    ]
    feature_df = _build_feature_table(payloads)

    result = train_and_evaluate(
        feature_df,
        label_col="label_mapped_majority",
        test_size=0.3,
        random_state=7,
        n_estimators=25,
        split_mode=split_mode,
    )
    assert result["split_mode"] in {"stratified", "subject", "stratified_fallback"}


def test_train_and_evaluate_isotonic_calibration_keeps_probs_valid():
    # Enough rows so the 20% calibration holdout fires.
    pairs = [
        ("subj_a", "walking"),
        ("subj_a", "static"),
        ("subj_b", "walking"),
        ("subj_b", "static"),
        ("subj_c", "walking"),
        ("subj_c", "static"),
        ("subj_d", "walking"),
        ("subj_d", "static"),
        ("subj_e", "walking"),
        ("subj_e", "static"),
    ]
    payloads = [
        _make_raw_payload(subject_id=sid, label=lbl, placement="pocket", seed=i)
        for i, (sid, lbl) in enumerate(pairs)
    ]
    feature_df = _build_feature_table(payloads)
    if feature_df.empty:
        pytest.skip("synthetic feature table empty")

    result = train_and_evaluate(
        feature_df,
        label_col="label_mapped_majority",
        test_size=0.25,
        random_state=42,
        n_estimators=25,
        split_mode="stratified",
        calibrate="isotonic",
    )

    # Either isotonic calibration was applied, or it was logged as skipped
    # because the holdout would have been too small. Both are valid.
    assert result["calibration"] in {"isotonic", "none"}

    # Calibrated model still exposes predict_proba and produces values in [0,1]
    # that sum to 1 per row.
    proba = result["model"].predict_proba(feature_df[result["feature_cols"]].fillna(0.0))
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0
    import numpy as np
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_train_and_evaluate_calibrate_none_skips_holdout():
    payloads = [
        _make_raw_payload(subject_id=f"s{i}", label=lbl, placement="pocket", seed=i)
        for i, lbl in enumerate(["walking", "static", "walking", "static",
                                  "walking", "static"])
    ]
    feature_df = _build_feature_table(payloads)
    if feature_df.empty:
        pytest.skip("synthetic feature table empty")
    result = train_and_evaluate(
        feature_df,
        label_col="label_mapped_majority",
        test_size=0.25,
        random_state=42,
        n_estimators=25,
        split_mode="stratified",
        calibrate="none",
    )
    assert result["calibration"] == "none"
