from __future__ import annotations

from fusion.fall_event import FallEventThresholds


def get_app_runtime_balanced_event_thresholds() -> FallEventThresholds:
    """
    Balanced runtime profile for live phone captures.

    APP_RUNTIME sessions use phone-scale acceleration values like MobiFall, but
    the live demo payload was slightly too noisy for the raw MobiFall impact
    cutoffs. This keeps the phone-scale weighting pattern while raising the
    impact gates enough to avoid promoting the clean session prefix.
    """
    return FallEventThresholds(
        impact_threshold=11.5,
        strong_impact_threshold=16.5,
        low_motion_ratio_threshold=0.15,
        medium_motion_ratio_threshold=0.35,
        high_motion_ratio_threshold=0.60,
        low_variance_threshold=8.0,
        medium_variance_threshold=20.0,
        high_variance_threshold=60.0,
        probable_fall_score_threshold=0.75,
        possible_fall_score_threshold=0.50,
        impact_only_score_threshold=0.20,
        meta_probability_weight=0.50,
        impact_weight=0.22,
        motion_weight=0.12,
        variance_weight=0.08,
        confirm_weight=0.08,
        meta_predicted_bonus=0.05,
        recovery_penalty_weight=0.30,
    )


def get_app_runtime_conservative_event_thresholds() -> FallEventThresholds:
    """
    More conservative runtime profile for live phone captures.

    This preserves phone-scale units but requires a stronger impact before the
    event layer promotes benign carry motion into a fall-like state.
    """
    return FallEventThresholds(
        impact_threshold=12.5,
        strong_impact_threshold=17.5,
        low_motion_ratio_threshold=0.15,
        medium_motion_ratio_threshold=0.35,
        high_motion_ratio_threshold=0.55,
        low_variance_threshold=8.0,
        medium_variance_threshold=18.0,
        high_variance_threshold=50.0,
        probable_fall_score_threshold=0.80,
        possible_fall_score_threshold=0.55,
        impact_only_score_threshold=0.25,
        meta_probability_weight=0.48,
        impact_weight=0.24,
        motion_weight=0.10,
        variance_weight=0.10,
        confirm_weight=0.08,
        meta_predicted_bonus=0.04,
        recovery_penalty_weight=0.32,
    )


def get_mobifall_balanced_event_thresholds() -> FallEventThresholds:
    """
    Balanced MobiFall profile.

    Designed to preserve the strong MobiFall meta-model signal while keeping
    event promotion reasonably conservative.
    """
    return FallEventThresholds(
        impact_threshold=9.5,
        strong_impact_threshold=14.0,
        low_motion_ratio_threshold=0.15,
        medium_motion_ratio_threshold=0.35,
        high_motion_ratio_threshold=0.60,
        low_variance_threshold=8.0,
        medium_variance_threshold=20.0,
        high_variance_threshold=60.0,
        probable_fall_score_threshold=0.75,
        possible_fall_score_threshold=0.50,
        impact_only_score_threshold=0.20,
        meta_probability_weight=0.50,
        impact_weight=0.22,
        motion_weight=0.12,
        variance_weight=0.08,
        confirm_weight=0.08,
        meta_predicted_bonus=0.05,
        recovery_penalty_weight=0.30,
    )


def get_mobifall_conservative_event_thresholds() -> FallEventThresholds:
    """
    More conservative MobiFall profile.

    Use this when you want fewer event promotions and stronger precision.
    """
    return FallEventThresholds(
        impact_threshold=10.5,
        strong_impact_threshold=15.0,
        low_motion_ratio_threshold=0.15,
        medium_motion_ratio_threshold=0.35,
        high_motion_ratio_threshold=0.55,
        low_variance_threshold=8.0,
        medium_variance_threshold=18.0,
        high_variance_threshold=50.0,
        probable_fall_score_threshold=0.80,
        possible_fall_score_threshold=0.55,
        impact_only_score_threshold=0.25,
        meta_probability_weight=0.48,
        impact_weight=0.24,
        motion_weight=0.10,
        variance_weight=0.10,
        confirm_weight=0.08,
        meta_predicted_bonus=0.04,
        recovery_penalty_weight=0.32,
    )


def get_sisfall_balanced_event_thresholds() -> FallEventThresholds:
    """
    Balanced SisFall profile.

    This profile is intentionally less conservative than the original generic
    event configuration because the SisFall meta-model was being suppressed too
    strongly by the event layer.
    """
    return FallEventThresholds(
        impact_threshold=490.0,
        strong_impact_threshold=1470.0,
        low_motion_ratio_threshold=0.20,
        medium_motion_ratio_threshold=0.40,
        high_motion_ratio_threshold=0.60,
        low_variance_threshold=500.0,
        medium_variance_threshold=2500.0,
        high_variance_threshold=10000.0,
        probable_fall_score_threshold=0.68,
        possible_fall_score_threshold=0.42,
        impact_only_score_threshold=0.18,
        meta_probability_weight=0.60,
        impact_weight=0.18,
        motion_weight=0.07,
        variance_weight=0.05,
        confirm_weight=0.05,
        meta_predicted_bonus=0.08,
        recovery_penalty_weight=0.25,
    )


def get_sisfall_conservative_event_thresholds() -> FallEventThresholds:
    """
    More conservative SisFall profile.

    Keeps more of the learned probability than the generic profile, but is
    stricter than the balanced SisFall setting.
    """
    return FallEventThresholds(
        impact_threshold=490.0,
        strong_impact_threshold=1470.0,
        low_motion_ratio_threshold=0.20,
        medium_motion_ratio_threshold=0.40,
        high_motion_ratio_threshold=0.60,
        low_variance_threshold=500.0,
        medium_variance_threshold=2500.0,
        high_variance_threshold=10000.0,
        probable_fall_score_threshold=0.75,
        possible_fall_score_threshold=0.50,
        impact_only_score_threshold=0.20,
        meta_probability_weight=0.55,
        impact_weight=0.20,
        motion_weight=0.08,
        variance_weight=0.07,
        confirm_weight=0.07,
        meta_predicted_bonus=0.06,
        recovery_penalty_weight=0.28,
    )


def get_fall_event_thresholds(
    dataset_name: str | None,
    profile: str = "balanced",
) -> FallEventThresholds:
    """
    Select dataset-aware fall-event thresholds.

    Supported profiles:
    - balanced
    - conservative
    """
    name = (dataset_name or "").upper()
    profile_name = profile.strip().lower()

    if profile_name not in {"balanced", "conservative"}:
        raise ValueError(
            f"Unsupported fall-event profile: {profile!r}. "
            "Expected 'balanced' or 'conservative'."
        )

    if name.startswith("APP_RUNTIME"):
        if profile_name == "conservative":
            return get_app_runtime_conservative_event_thresholds()
        return get_app_runtime_balanced_event_thresholds()

    if name == "MOBIFALL":
        if profile_name == "conservative":
            return get_mobifall_conservative_event_thresholds()
        return get_mobifall_balanced_event_thresholds()

    if name == "SISFALL":
        if profile_name == "conservative":
            return get_sisfall_conservative_event_thresholds()
        return get_sisfall_balanced_event_thresholds()

    # Fallback: keep the historical generic defaults for unknown datasets.
    # Runtime phone captures should route through the explicit APP_RUNTIME
    # family above instead of relying on these SisFall-scale thresholds.
    return FallEventThresholds()
