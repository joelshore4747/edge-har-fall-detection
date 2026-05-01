from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from pipeline.fall.features import extract_fall_window_features


@dataclass(frozen=True)
class FallThresholdConfig:
    # Stage 1: impact candidate
    impact_peak_acc_threshold: float
    impact_peak_ratio_threshold: float = 1.20

    # Stage 2: optional support (legacy-compatible)
    require_support_stage: bool = False
    jerk_peak_threshold: float | None = None
    gyro_peak_threshold: float | None = None
    support_logic: str = "any"  # "any" or "all"

    # Stage 3: optional confirm stage (legacy-compatible)
    require_confirm_stage: bool = True
    confirm_logic: str = "all"  # "all" or "any"
    confirm_post_dyn_ratio_mean_max: float | None = 0.10
    confirm_requires_post_impact: bool = True
    confirm_post_dyn_mean_max: float | None = None
    confirm_post_var_max: float | None = None
    confirm_post_jerk_rms_max: float | None = None

    # Legacy aliases retained for scripts/tests
    post_impact_motion_max: float | None = None
    post_impact_variance_max: float | None = None
    post_impact_motion_ratio_max: float | None = 1.0

    # Feature extraction helper config
    post_impact_skip_samples: int = 2

    # New soft-sidecar fields
    probable_fall_score_threshold: float = 0.72
    possible_fall_score_threshold: float = 0.48


def default_fall_threshold_config(dataset_name: str | None = None) -> FallThresholdConfig:
    """
    Return a transparent threshold preset.

    Important change:
    defaults are now more permissive than the old strict baseline so the
    baseline is at least usable, while still preserving the old fields so
    scripts/tests keep working.
    """
    name = (dataset_name or "").upper()

    if name == "MOBIFALL":
        return FallThresholdConfig(
            impact_peak_acc_threshold=9.5,
            impact_peak_ratio_threshold=1.10,
            require_support_stage=False,
            jerk_peak_threshold=20.0,
            gyro_peak_threshold=6.0,
            support_logic="any",
            require_confirm_stage=True,
            confirm_logic="any",
            confirm_post_dyn_ratio_mean_max=None,
            confirm_requires_post_impact=True,
            confirm_post_dyn_mean_max=None,
            confirm_post_var_max=8.0,
            confirm_post_jerk_rms_max=None,
            post_impact_motion_max=None,
            post_impact_variance_max=8.0,
            post_impact_motion_ratio_max=None,
            post_impact_skip_samples=2,
            probable_fall_score_threshold=0.72,
            possible_fall_score_threshold=0.48,
        )

    if name == "SISFALL":
        return FallThresholdConfig(
            impact_peak_acc_threshold=490.0,
            impact_peak_ratio_threshold=1.05,
            require_support_stage=False,
            jerk_peak_threshold=0.0,
            gyro_peak_threshold=900.0,
            support_logic="any",
            require_confirm_stage=False,
            confirm_logic="any",
            # Keep this field for compatibility, but do not let tiny values kill
            # all falls when the feature scale is clearly much larger.
            confirm_post_dyn_ratio_mean_max=700.0,
            confirm_requires_post_impact=False,
            confirm_post_dyn_mean_max=None,
            confirm_post_var_max=25000.0,
            confirm_post_jerk_rms_max=None,
            post_impact_motion_max=None,
            post_impact_variance_max=25000.0,
            post_impact_motion_ratio_max=1.05,
            post_impact_skip_samples=2,
            probable_fall_score_threshold=0.72,
            possible_fall_score_threshold=0.48,
        )

    return FallThresholdConfig(
        impact_peak_acc_threshold=14.0,
        impact_peak_ratio_threshold=1.20,
        require_support_stage=False,
        jerk_peak_threshold=20.0,
        gyro_peak_threshold=None,
        support_logic="any",
        require_confirm_stage=False,
        confirm_logic="any",
        confirm_post_dyn_ratio_mean_max=0.25,
        confirm_requires_post_impact=False,
        confirm_post_dyn_mean_max=None,
        confirm_post_var_max=None,
        confirm_post_jerk_rms_max=None,
        post_impact_motion_max=None,
        post_impact_variance_max=None,
        post_impact_motion_ratio_max=1.0,
        post_impact_skip_samples=2,
        probable_fall_score_threshold=0.72,
        possible_fall_score_threshold=0.48,
    )


def _is_finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _safe_fraction_true(checks: dict[str, bool]) -> float:
    if not checks:
        return 0.0
    return float(sum(bool(v) for v in checks.values()) / len(checks))


def _impact_component(peak_acc: float, peak_ratio: float, config: FallThresholdConfig) -> tuple[float, dict[str, bool]]:
    stage1_checks = {
        "impact_peak_acc": (_is_finite_number(peak_acc) and peak_acc >= float(config.impact_peak_acc_threshold)),
        "impact_peak_ratio": (_is_finite_number(peak_ratio) and peak_ratio >= float(config.impact_peak_ratio_threshold)),
    }

    score = 0.0
    if stage1_checks["impact_peak_acc"]:
        score += 0.70
    if stage1_checks["impact_peak_ratio"]:
        score += 0.30

    return _clamp(score), stage1_checks


def _support_component(jerk_peak: float, gyro_peak: float, config: FallThresholdConfig) -> tuple[float, dict[str, bool]]:
    support_checks: dict[str, bool] = {}

    if config.jerk_peak_threshold is not None:
        support_checks["jerk_peak"] = _is_finite_number(jerk_peak) and jerk_peak >= float(config.jerk_peak_threshold)
    if config.gyro_peak_threshold is not None:
        support_checks["gyro_peak"] = _is_finite_number(gyro_peak) and gyro_peak >= float(config.gyro_peak_threshold)

    return _safe_fraction_true(support_checks), support_checks


def _confirm_checks_from_features(
    *,
    post_dyn_ratio_mean: float,
    post_dyn_mean: float,
    post_motion: float,
    post_var: float,
    jerk_rms: float,
    peak_acc: float,
    config: FallThresholdConfig,
) -> dict[str, bool]:
    confirm_checks: dict[str, bool] = {}

    # Ratio-mean threshold
    ratio_threshold = config.confirm_post_dyn_ratio_mean_max
    if ratio_threshold is not None:
        if _is_finite_number(post_dyn_ratio_mean):
            # Important: protect against obvious scale mismatch, especially on SisFall.
            if float(ratio_threshold) <= 5.0 and float(post_dyn_ratio_mean) > 10.0:
                # Treat as unavailable / non-binding instead of auto-failing everything.
                confirm_checks["post_impact_dyn_ratio_mean_max"] = not bool(config.confirm_requires_post_impact)
            else:
                confirm_checks["post_impact_dyn_ratio_mean_max"] = post_dyn_ratio_mean <= float(ratio_threshold)
        else:
            confirm_checks["post_impact_dyn_ratio_mean_max"] = not bool(config.confirm_requires_post_impact)

    # Legacy motion alias
    dyn_mean_threshold = config.confirm_post_dyn_mean_max
    if dyn_mean_threshold is None and config.post_impact_motion_max is not None:
        dyn_mean_threshold = float(config.post_impact_motion_max)
    if dyn_mean_threshold is not None:
        dyn_motion_value = post_dyn_mean if _is_finite_number(post_dyn_mean) else post_motion
        confirm_checks["post_impact_dyn_mean_max"] = _is_finite_number(dyn_motion_value) and dyn_motion_value <= float(dyn_mean_threshold)

    # Variance threshold / legacy alias
    var_threshold = config.confirm_post_var_max
    if var_threshold is None and config.post_impact_variance_max is not None:
        var_threshold = float(config.post_impact_variance_max)
    if var_threshold is not None:
        confirm_checks["post_impact_variance_max"] = _is_finite_number(post_var) and post_var <= float(var_threshold)

    if config.confirm_post_jerk_rms_max is not None:
        confirm_checks["post_impact_jerk_rms_max"] = _is_finite_number(jerk_rms) and jerk_rms <= float(config.confirm_post_jerk_rms_max)

    if config.post_impact_motion_ratio_max is not None:
        ratio = float("nan")
        if _is_finite_number(post_motion) and _is_finite_number(peak_acc) and peak_acc != 0:
            ratio = float(post_motion / peak_acc)
        confirm_checks["post_impact_motion_ratio_max"] = _is_finite_number(ratio) and ratio <= float(config.post_impact_motion_ratio_max)

    return confirm_checks


def _confirm_component(confirm_checks: dict[str, bool]) -> float:
    return _safe_fraction_true(confirm_checks)


def _event_state_from_confidence(score: float, impact_pass: bool, config: FallThresholdConfig) -> str:
    if not impact_pass:
        return "no_event"
    if score >= config.probable_fall_score_threshold:
        return "probable_fall"
    if score >= config.possible_fall_score_threshold:
        return "possible_fall"
    return "impact_only"


def detect_fall_from_features(features: dict[str, Any], config: FallThresholdConfig) -> dict[str, Any]:
    peak_acc = float(features.get("peak_acc", np.nan))
    mean_acc = float(features.get("mean_acc", np.nan))
    peak_ratio = float(features.get("peak_over_mean_ratio", np.nan))
    post_dyn_mean = float(features.get("post_impact_dyn_mean", np.nan))
    post_dyn_ratio_mean = float(features.get("post_impact_dyn_ratio_mean", np.nan))
    post_motion = float(features.get("post_impact_motion", np.nan))
    post_var = float(features.get("post_impact_variance", np.nan))
    jerk_rms = float(features.get("jerk_rms", np.nan))
    jerk_peak = float(features.get("jerk_peak", np.nan))
    gyro_peak = float(features.get("gyro_peak", np.nan))

    reasons: list[str] = []

    # Legacy-compatible hard stage checks
    impact_component, stage1_checks = _impact_component(peak_acc, peak_ratio, config)
    impact_pass = bool(all(stage1_checks.values()))
    if impact_pass:
        reasons.append("impact_candidate")

    support_component, support_checks = _support_component(jerk_peak, gyro_peak, config)
    if config.require_support_stage:
        if not support_checks:
            support_pass = True
        elif config.support_logic == "all":
            support_pass = bool(all(support_checks.values()))
        else:
            support_pass = bool(any(support_checks.values()))
    else:
        support_pass = True
    if support_pass and support_checks:
        reasons.append("support_stage")

    confirm_checks = _confirm_checks_from_features(
        post_dyn_ratio_mean=post_dyn_ratio_mean,
        post_dyn_mean=post_dyn_mean,
        post_motion=post_motion,
        post_var=post_var,
        jerk_rms=jerk_rms,
        peak_acc=peak_acc,
        config=config,
    )
    confirm_component = _confirm_component(confirm_checks)

    if config.require_confirm_stage:
        finite_confirm_checks = [v for v in confirm_checks.values()]
        if not finite_confirm_checks:
            confirm_pass = False
        elif str(config.confirm_logic).lower() == "any":
            confirm_pass = bool(any(finite_confirm_checks))
        else:
            confirm_pass = bool(all(finite_confirm_checks))
    else:
        confirm_pass = True
    if confirm_pass and confirm_checks:
        reasons.append("confirm_stage")

    # Legacy-compatible predicted label for old evaluation/tests/scripts
    predicted_is_fall = bool(impact_pass and support_pass and confirm_pass)
    predicted_label = "fall" if predicted_is_fall else "non_fall"

    # New soft sidecar for the newer fusion pipeline
    recovery_penalty = 0.0
    ratio = float("nan")
    if _is_finite_number(post_motion) and _is_finite_number(peak_acc) and peak_acc != 0:
        ratio = float(post_motion / peak_acc)
        features.setdefault("post_impact_motion_to_peak_ratio", ratio)
        if config.post_impact_motion_ratio_max is not None:
            loose_high_motion_cutoff = max(float(config.post_impact_motion_ratio_max), 0.60)
            if ratio >= loose_high_motion_cutoff:
                recovery_penalty = 0.20

    fall_confidence = _clamp(
        (0.55 * impact_component)
        + (0.15 * support_component)
        + (0.30 * confirm_component)
        - recovery_penalty
    )
    event_state = _event_state_from_confidence(fall_confidence, impact_pass, config)

    if not impact_pass:
        detector_reason = "failed_impact_stage"
    elif not support_pass:
        detector_reason = "failed_support_stage"
    elif not confirm_pass:
        detector_reason = "failed_confirm_stage"
    else:
        detector_reason = "fall_detected"

    return {
        "predicted_label": predicted_label,
        "predicted_is_fall": predicted_is_fall,
        "fall_confidence": fall_confidence,
        "event_state": event_state,
        "stage_impact_pass": impact_pass,
        "stage_support_pass": support_pass,
        "stage_confirm_pass": confirm_pass,
        "impact_checks": stage1_checks,
        "support_checks": support_checks,
        "confirm_checks": confirm_checks,
        "impact_thresholds": {
            "impact_peak_acc_threshold": float(config.impact_peak_acc_threshold),
            "impact_peak_ratio_threshold": float(config.impact_peak_ratio_threshold),
        },
        "confirm_thresholds": {
            "confirm_post_dyn_ratio_mean_max": config.confirm_post_dyn_ratio_mean_max,
            "confirm_requires_post_impact": config.confirm_requires_post_impact,
            "confirm_post_dyn_mean_max": config.confirm_post_dyn_mean_max,
            "confirm_post_var_max": config.confirm_post_var_max,
            "confirm_post_jerk_rms_max": config.confirm_post_jerk_rms_max,
            "confirm_logic": config.confirm_logic,
        },
        "contributions": {
            "impact_component": round(impact_component, 4),
            "support_component": round(support_component, 4),
            "confirm_component": round(confirm_component, 4),
            "recovery_penalty": round(recovery_penalty, 4),
        },
        "detector_reason": detector_reason,
        "reasons": reasons,
    }


def detect_fall_window(
    window: dict[str, Any],
    *,
    config: FallThresholdConfig,
    default_sampling_rate_hz: float | None = None,
) -> dict[str, Any]:
    features = extract_fall_window_features(
        window,
        default_sampling_rate_hz=default_sampling_rate_hz,
        post_impact_skip_samples=config.post_impact_skip_samples,
    )
    decision = detect_fall_from_features(features, config)
    return {
        "features": features,
        "decision": decision,
        "config": asdict(config),
    }