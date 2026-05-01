from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class FallEventState(str, Enum):
    NO_EVENT = "no_event"
    IMPACT_ONLY = "impact_only"
    POSSIBLE_FALL = "possible_fall"
    PROBABLE_FALL = "probable_fall"


@dataclass(slots=True)
class FallEventThresholds:
    """
    Thresholds and weights for staged fall-event reasoning.

    Main design update:
    - meta_probability from the second-stage logistic model is now the
      primary fall-confidence signal when available
    - threshold-derived evidence remains useful as supporting evidence
    - recovery evidence suppresses escalation
    """

    impact_threshold: float = 490.0
    strong_impact_threshold: float = 1470.0

    low_motion_ratio_threshold: float = 0.20
    medium_motion_ratio_threshold: float = 0.40
    high_motion_ratio_threshold: float = 0.60

    low_variance_threshold: float = 500.0
    medium_variance_threshold: float = 2500.0
    high_variance_threshold: float = 10000.0

    probable_fall_score_threshold: float = 0.75
    possible_fall_score_threshold: float = 0.50
    impact_only_score_threshold: float = 0.20

    meta_probability_weight: float = 0.50
    impact_weight: float = 0.22
    motion_weight: float = 0.12
    variance_weight: float = 0.08
    confirm_weight: float = 0.08
    meta_predicted_bonus: float = 0.05
    recovery_penalty_weight: float = 0.30


@dataclass(slots=True)
class FallEventInputs:
    """
    Evidence from the fall pipeline.

    These values can come from:
    - threshold detector outputs
    - engineered post-impact features
    - second-stage probabilistic meta-model
    """

    peak_acc: float = 0.0

    stage_impact_pass: bool = False
    stage_confirm_pass: bool = False
    stage_support_pass: bool = False

    post_impact_available: bool = False
    post_impact_motion_to_peak_ratio: float = 0.0
    post_impact_variance: float = 0.0
    post_impact_dyn_ratio_mean: float = 0.0

    recovery_detected: bool = False

    meta_probability: Optional[float] = None
    meta_predicted_is_fall: bool = False

    def validate(self) -> None:
        if self.peak_acc < 0:
            raise ValueError("peak_acc must be >= 0")
        if self.post_impact_variance < 0:
            raise ValueError("post_impact_variance must be >= 0")
        if self.post_impact_dyn_ratio_mean < 0:
            raise ValueError("post_impact_dyn_ratio_mean must be >= 0")

        value = self.post_impact_motion_to_peak_ratio
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"post_impact_motion_to_peak_ratio must be between 0.0 and 1.0, got {value}"
            )

        if self.meta_probability is not None and not 0.0 <= self.meta_probability <= 1.0:
            raise ValueError(
                f"meta_probability must be between 0.0 and 1.0, got {self.meta_probability}"
            )


@dataclass(slots=True)
class FallEventResult:
    state: FallEventState
    confidence: float
    reasons: List[str] = field(default_factory=list)
    contributions: Dict[str, float] = field(default_factory=dict)

    @property
    def is_suspicious(self) -> bool:
        return self.state in {
            FallEventState.IMPACT_ONLY,
            FallEventState.POSSIBLE_FALL,
            FallEventState.PROBABLE_FALL,
        }

    @property
    def is_likely_fall(self) -> bool:
        return self.state in {
            FallEventState.POSSIBLE_FALL,
            FallEventState.PROBABLE_FALL,
        }


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def meta_probability_score(meta_probability: Optional[float]) -> float:
    if meta_probability is None:
        return 0.0
    return clamp(float(meta_probability))


def impact_score(
    peak_acc: float,
    stage_impact_pass: bool,
    thresholds: FallEventThresholds,
) -> float:
    """
    Convert impact evidence into a 0..1 score.

    Uses both raw peak acceleration and the existing stage_impact_pass flag.
    """
    if peak_acc <= 0 and not stage_impact_pass:
        return 0.0

    if peak_acc >= thresholds.strong_impact_threshold:
        base = 1.0
    elif peak_acc >= thresholds.impact_threshold:
        span = thresholds.strong_impact_threshold - thresholds.impact_threshold
        base = 0.5 if span <= 0 else 0.5 + 0.5 * (
            (peak_acc - thresholds.impact_threshold) / span
        )
    else:
        base = 0.0

    if stage_impact_pass:
        base = max(base, 0.6)

    return clamp(base)


def low_motion_score(
    post_impact_available: bool,
    post_impact_motion_to_peak_ratio: float,
    thresholds: FallEventThresholds,
) -> float:
    """
    Higher score means less movement after the event.
    """
    if not post_impact_available:
        return 0.0

    ratio = post_impact_motion_to_peak_ratio

    if ratio <= thresholds.low_motion_ratio_threshold:
        return 1.0
    if ratio <= thresholds.medium_motion_ratio_threshold:
        return 0.7
    if ratio <= thresholds.high_motion_ratio_threshold:
        return 0.35
    return 0.0


def low_variance_score(
    post_impact_available: bool,
    post_impact_variance: float,
    thresholds: FallEventThresholds,
) -> float:
    """
    Higher score means post-impact variance is low, which can indicate
    stillness after an incident.
    """
    if not post_impact_available:
        return 0.0

    variance = post_impact_variance

    if variance <= thresholds.low_variance_threshold:
        return 1.0
    if variance <= thresholds.medium_variance_threshold:
        return 0.7
    if variance <= thresholds.high_variance_threshold:
        return 0.35
    return 0.0


def confirm_score(stage_confirm_pass: bool, stage_support_pass: bool) -> float:
    """
    Confirmation evidence should boost confidence, not hard-gate the event.
    """
    score = 0.0
    if stage_confirm_pass:
        score += 0.7
    if stage_support_pass:
        score += 0.3
    return clamp(score)


def recovery_penalty(
    recovery_detected: bool,
    post_impact_available: bool,
    post_impact_motion_to_peak_ratio: float,
    thresholds: FallEventThresholds,
) -> float:
    """
    Penalty for clear recovery / resumed movement.
    """
    penalty = 0.0

    if recovery_detected:
        penalty += 0.7

    if post_impact_available:
        ratio = post_impact_motion_to_peak_ratio
        if ratio >= thresholds.high_motion_ratio_threshold:
            penalty += 0.3
        elif ratio >= thresholds.medium_motion_ratio_threshold:
            penalty += 0.15

    return clamp(penalty)


def determine_state(
    score: float,
    impact_component: float,
    meta_component: float,
    thresholds: FallEventThresholds,
) -> FallEventState:
    """
    Require either:
    - meaningful impact evidence, or
    - meaningful learned fall probability
    before calling the event suspicious.
    """
    if impact_component < 0.10 and meta_component < 0.15:
        return FallEventState.NO_EVENT

    if score >= thresholds.probable_fall_score_threshold:
        return FallEventState.PROBABLE_FALL

    # Strong learned evidence can promote to POSSIBLE_FALL even when impact is weak.
    if meta_component >= 0.35 and score >= 0.40:
        return FallEventState.POSSIBLE_FALL

    if score >= thresholds.possible_fall_score_threshold:
        return FallEventState.POSSIBLE_FALL
    if score >= thresholds.impact_only_score_threshold:
        return FallEventState.IMPACT_ONLY
    return FallEventState.NO_EVENT


def classify_fall_event(
    inputs: FallEventInputs,
    thresholds: Optional[FallEventThresholds] = None,
) -> FallEventResult:
    """
    Classify a fall-related event using combined learned + threshold evidence.

    Design update:
    - meta_probability is now the primary fall-confidence signal when present
    - threshold/event evidence provides interpretable support
    - recovery reduces escalation confidence
    """
    inputs.validate()
    thresholds = thresholds or FallEventThresholds()

    reasons: List[str] = []
    contributions: Dict[str, float] = {}

    meta_component = thresholds.meta_probability_weight * meta_probability_score(inputs.meta_probability)
    contributions["meta_probability_component"] = round(meta_component, 4)

    impact_component = thresholds.impact_weight * impact_score(
        peak_acc=inputs.peak_acc,
        stage_impact_pass=inputs.stage_impact_pass,
        thresholds=thresholds,
    )
    contributions["impact_component"] = round(impact_component, 4)

    motion_component = thresholds.motion_weight * low_motion_score(
        post_impact_available=inputs.post_impact_available,
        post_impact_motion_to_peak_ratio=inputs.post_impact_motion_to_peak_ratio,
        thresholds=thresholds,
    )
    contributions["low_motion_component"] = round(motion_component, 4)

    variance_component = thresholds.variance_weight * low_variance_score(
        post_impact_available=inputs.post_impact_available,
        post_impact_variance=inputs.post_impact_variance,
        thresholds=thresholds,
    )
    contributions["low_variance_component"] = round(variance_component, 4)

    confirm_component = thresholds.confirm_weight * confirm_score(
        stage_confirm_pass=inputs.stage_confirm_pass,
        stage_support_pass=inputs.stage_support_pass,
    )
    contributions["confirm_component"] = round(confirm_component, 4)

    predicted_bonus = thresholds.meta_predicted_bonus if inputs.meta_predicted_is_fall else 0.0
    contributions["meta_predicted_bonus"] = round(predicted_bonus, 4)

    penalty_component = thresholds.recovery_penalty_weight * recovery_penalty(
        recovery_detected=inputs.recovery_detected,
        post_impact_available=inputs.post_impact_available,
        post_impact_motion_to_peak_ratio=inputs.post_impact_motion_to_peak_ratio,
        thresholds=thresholds,
    )
    contributions["recovery_penalty"] = round(-penalty_component, 4)

    raw_score = (
        meta_component
        + impact_component
        + motion_component
        + variance_component
        + confirm_component
        + predicted_bonus
        - penalty_component
    )
    final_score = clamp(raw_score)

    state = determine_state(
        score=final_score,
        impact_component=impact_component,
        meta_component=meta_component,
        thresholds=thresholds,
    )

    if inputs.meta_probability is not None:
        if inputs.meta_probability >= 0.80:
            reasons.append("high learned fall probability")
        elif inputs.meta_probability >= 0.50:
            reasons.append("moderate learned fall probability")
        elif inputs.meta_probability >= 0.30:
            reasons.append("low-to-moderate learned fall probability")

    if inputs.meta_predicted_is_fall:
        reasons.append("meta-model classified event as fall")

    if inputs.stage_impact_pass:
        reasons.append("impact stage passed")
    elif inputs.peak_acc >= thresholds.impact_threshold:
        reasons.append("peak acceleration exceeded impact threshold")

    if inputs.peak_acc >= thresholds.strong_impact_threshold:
        reasons.append("strong impact magnitude observed")

    if inputs.post_impact_available:
        if inputs.post_impact_motion_to_peak_ratio <= thresholds.low_motion_ratio_threshold:
            reasons.append("very low post-impact motion")
        elif inputs.post_impact_motion_to_peak_ratio <= thresholds.medium_motion_ratio_threshold:
            reasons.append("reduced post-impact motion")
        elif inputs.post_impact_motion_to_peak_ratio >= thresholds.high_motion_ratio_threshold:
            reasons.append("substantial post-impact motion")

        if inputs.post_impact_variance <= thresholds.low_variance_threshold:
            reasons.append("very low post-impact variance")
        elif inputs.post_impact_variance <= thresholds.medium_variance_threshold:
            reasons.append("reduced post-impact variance")

    if inputs.stage_confirm_pass:
        reasons.append("confirm stage passed")
    if inputs.stage_support_pass:
        reasons.append("support stage passed")

    if inputs.recovery_detected:
        reasons.append("recovery detected")

    return FallEventResult(
        state=state,
        confidence=round(final_score, 4),
        reasons=reasons,
        contributions=contributions,
    )