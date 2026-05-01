from __future__ import annotations

import pytest

from fusion.fall_event import (
    FallEventInputs,
    FallEventState,
    FallEventThresholds,
    classify_fall_event,
    confirm_score,
    determine_state,
    impact_score,
    low_motion_score,
    low_variance_score,
    meta_probability_score,
    recovery_penalty,
)


def test_meta_probability_score() -> None:
    assert meta_probability_score(None) == 0.0
    assert meta_probability_score(0.0) == 0.0
    assert meta_probability_score(0.42) == 0.42
    assert meta_probability_score(1.0) == 1.0


def test_impact_score_no_event() -> None:
    thresholds = FallEventThresholds()
    assert impact_score(peak_acc=0.0, stage_impact_pass=False, thresholds=thresholds) == 0.0


def test_impact_score_stage_flag_gives_minimum_signal() -> None:
    thresholds = FallEventThresholds()
    assert impact_score(peak_acc=100.0, stage_impact_pass=True, thresholds=thresholds) == 0.6


def test_impact_score_at_and_above_thresholds() -> None:
    thresholds = FallEventThresholds()
    assert impact_score(
        peak_acc=thresholds.impact_threshold,
        stage_impact_pass=False,
        thresholds=thresholds,
    ) == 0.5
    assert impact_score(
        peak_acc=thresholds.strong_impact_threshold,
        stage_impact_pass=False,
        thresholds=thresholds,
    ) == 1.0


@pytest.mark.parametrize(
    ("available", "ratio", "expected"),
    [
        (False, 0.0, 0.0),
        (True, 0.10, 1.0),
        (True, 0.30, 0.7),
        (True, 0.50, 0.35),
        (True, 0.80, 0.0),
    ],
)
def test_low_motion_score(available: bool, ratio: float, expected: float) -> None:
    thresholds = FallEventThresholds()
    assert low_motion_score(
        post_impact_available=available,
        post_impact_motion_to_peak_ratio=ratio,
        thresholds=thresholds,
    ) == expected


@pytest.mark.parametrize(
    ("available", "variance", "expected"),
    [
        (False, 0.0, 0.0),
        (True, 100.0, 1.0),
        (True, 1000.0, 0.7),
        (True, 5000.0, 0.35),
        (True, 50000.0, 0.0),
    ],
)
def test_low_variance_score(available: bool, variance: float, expected: float) -> None:
    thresholds = FallEventThresholds()
    assert low_variance_score(
        post_impact_available=available,
        post_impact_variance=variance,
        thresholds=thresholds,
    ) == expected


@pytest.mark.parametrize(
    ("confirm_pass", "support_pass", "expected"),
    [
        (False, False, 0.0),
        (True, False, 0.7),
        (False, True, 0.3),
        (True, True, 1.0),
    ],
)
def test_confirm_score(confirm_pass: bool, support_pass: bool, expected: float) -> None:
    assert confirm_score(confirm_pass, support_pass) == expected


@pytest.mark.parametrize(
    ("recovery_detected", "available", "ratio", "expected"),
    [
        (False, False, 0.0, 0.0),
        (False, True, 0.20, 0.0),
        (False, True, 0.50, 0.15),
        (False, True, 0.80, 0.3),
        (True, False, 0.0, 0.7),
        (True, True, 0.50, 0.85),
        (True, True, 0.80, 1.0),
    ],
)
def test_recovery_penalty(
    recovery_detected: bool,
    available: bool,
    ratio: float,
    expected: float,
) -> None:
    thresholds = FallEventThresholds()
    assert recovery_penalty(
        recovery_detected=recovery_detected,
        post_impact_available=available,
        post_impact_motion_to_peak_ratio=ratio,
        thresholds=thresholds,
    ) == expected


@pytest.mark.parametrize(
    ("score", "impact_component", "meta_component", "expected"),
    [
        (0.10, 0.05, 0.10, FallEventState.NO_EVENT),
        (0.25, 0.30, 0.00, FallEventState.IMPACT_ONLY),
        (0.55, 0.30, 0.00, FallEventState.POSSIBLE_FALL),
        (0.80, 0.30, 0.00, FallEventState.PROBABLE_FALL),
        (0.58, 0.00, 0.20, FallEventState.POSSIBLE_FALL),
    ],
)
def test_determine_state(
    score: float,
    impact_component: float,
    meta_component: float,
    expected: FallEventState,
) -> None:
    thresholds = FallEventThresholds()
    assert determine_state(score, impact_component, meta_component, thresholds) == expected


def test_classify_fall_event_probable_fall_from_learned_and_threshold_evidence() -> None:
    result = classify_fall_event(
        FallEventInputs(
            peak_acc=1800.0,
            stage_impact_pass=True,
            stage_confirm_pass=True,
            stage_support_pass=False,
            post_impact_available=True,
            post_impact_motion_to_peak_ratio=0.10,
            post_impact_variance=150.0,
            post_impact_dyn_ratio_mean=200.0,
            recovery_detected=False,
            meta_probability=0.93,
            meta_predicted_is_fall=True,
        )
    )

    assert result.state == FallEventState.PROBABLE_FALL
    assert result.confidence >= 0.75
    assert result.is_suspicious is True
    assert result.is_likely_fall is True
    assert "high learned fall probability" in result.reasons
    assert "meta-model classified event as fall" in result.reasons
    assert "impact stage passed" in result.reasons
    assert "strong impact magnitude observed" in result.reasons
    assert "very low post-impact motion" in result.reasons
    assert "very low post-impact variance" in result.reasons
    assert "confirm stage passed" in result.reasons


def test_classify_fall_event_possible_fall_from_meta_probability() -> None:
    result = classify_fall_event(
        FallEventInputs(
            peak_acc=900.0,
            stage_impact_pass=True,
            stage_confirm_pass=False,
            stage_support_pass=False,
            post_impact_available=True,
            post_impact_motion_to_peak_ratio=0.28,
            post_impact_variance=1800.0,
            post_impact_dyn_ratio_mean=400.0,
            recovery_detected=False,
            meta_probability=0.61,
            meta_predicted_is_fall=True,
        )
    )

    assert result.state in {FallEventState.POSSIBLE_FALL, FallEventState.PROBABLE_FALL}
    assert result.is_suspicious is True
    assert "moderate learned fall probability" in result.reasons
    assert "meta-model classified event as fall" in result.reasons


def test_classify_fall_event_impact_only_when_recovery_present() -> None:
    result = classify_fall_event(
        FallEventInputs(
            peak_acc=1600.0,
            stage_impact_pass=True,
            stage_confirm_pass=False,
            stage_support_pass=False,
            post_impact_available=True,
            post_impact_motion_to_peak_ratio=0.75,
            post_impact_variance=12000.0,
            post_impact_dyn_ratio_mean=900.0,
            recovery_detected=True,
            meta_probability=0.38,
            meta_predicted_is_fall=False,
        )
    )

    assert result.state in {FallEventState.IMPACT_ONLY, FallEventState.NO_EVENT}
    assert "recovery detected" in result.reasons
    assert "substantial post-impact motion" in result.reasons


def test_classify_fall_event_no_event() -> None:
    result = classify_fall_event(
        FallEventInputs(
            peak_acc=150.0,
            stage_impact_pass=False,
            stage_confirm_pass=False,
            stage_support_pass=False,
            post_impact_available=False,
            post_impact_motion_to_peak_ratio=0.0,
            post_impact_variance=0.0,
            post_impact_dyn_ratio_mean=0.0,
            recovery_detected=False,
            meta_probability=0.04,
            meta_predicted_is_fall=False,
        )
    )

    assert result.state == FallEventState.NO_EVENT
    assert result.confidence < 0.20
    assert result.is_suspicious is False
    assert result.is_likely_fall is False


def test_classify_fall_event_can_be_suspicious_from_meta_probability_without_strong_impact() -> None:
    result = classify_fall_event(
        FallEventInputs(
            peak_acc=50.0,
            stage_impact_pass=False,
            stage_confirm_pass=False,
            stage_support_pass=False,
            post_impact_available=False,
            post_impact_motion_to_peak_ratio=0.0,
            post_impact_variance=0.0,
            post_impact_dyn_ratio_mean=0.0,
            recovery_detected=False,
            meta_probability=0.74,
            meta_predicted_is_fall=True,
        )
    )

    assert result.state in {FallEventState.POSSIBLE_FALL, FallEventState.PROBABLE_FALL}
    assert result.is_suspicious is True
    assert "moderate learned fall probability" in result.reasons or "high learned fall probability" in result.reasons


def test_inputs_validate_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="peak_acc"):
        FallEventInputs(peak_acc=-1.0).validate()

    with pytest.raises(ValueError, match="post_impact_variance"):
        FallEventInputs(post_impact_variance=-1.0).validate()

    with pytest.raises(ValueError, match="post_impact_dyn_ratio_mean"):
        FallEventInputs(post_impact_dyn_ratio_mean=-1.0).validate()


def test_inputs_validate_rejects_bad_motion_ratio() -> None:
    with pytest.raises(ValueError, match="post_impact_motion_to_peak_ratio"):
        FallEventInputs(post_impact_motion_to_peak_ratio=1.5).validate()


def test_inputs_validate_rejects_bad_meta_probability() -> None:
    with pytest.raises(ValueError, match="meta_probability"):
        FallEventInputs(meta_probability=1.2).validate()