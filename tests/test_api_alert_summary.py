"""Unit tests for ``_derive_alert_summary`` HAR-gated suppression.

These exercise the pure alert-decision logic without spinning up the inference
pipeline. The rule under test: when HAR majority is ``static`` *or* a
confident locomotion class (walking / running / stairs / dynamic /
locomotion) with a sufficient fraction, fall-like events get downgraded one
severity step — the same treatment that ``placement_state`` already receives.
The locomotion arm mirrors the session-level cap applied per-window by
``fusion.vulnerability_score._dynamic_har_attenuation``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from apps.api.main import _derive_alert_summary
from apps.api.schemas import (
    FallSummary,
    HarSummary,
    PlacementSummary,
    VulnerabilitySummary,
)


@dataclass
class _Result:
    """Minimal stand-in for the inference result — only ``placement_summary`` matters."""

    placement_summary: dict | None = None


def _make_inputs(
    *,
    har_top: str | None,
    har_fraction: float | None,
    placement_state: str,
    likely_fall: bool = True,
    top_prob: float = 0.9,
    alert_worthy_windows: int = 0,
    latest_level: str | None = None,
    latest_state: str | None = None,
):
    result = _Result(placement_summary={"placement_state": placement_state})
    placement = PlacementSummary(
        placement_state=placement_state,
        placement_confidence=0.9,
    )
    har = HarSummary(top_label=har_top, top_label_fraction=har_fraction, label_counts={}, total_windows=0)
    fall = FallSummary(
        likely_fall_detected=likely_fall,
        positive_window_count=5 if likely_fall else 0,
        grouped_event_count=1 if likely_fall else 0,
        top_fall_probability=top_prob,
    )
    vulnerability = VulnerabilitySummary(
        enabled=True,
        alert_worthy_window_count=alert_worthy_windows,
        latest_vulnerability_level=latest_level,
        latest_monitoring_state=latest_state,
    )
    return result, placement, har, fall, vulnerability


def test_likely_fall_with_locomotion_har_majority_downgrades_to_medium():
    """Mirror of the session-level cap in
    ``fusion.vulnerability_score._dynamic_har_attenuation``: a confident
    locomotion majority should downgrade the alert one step, not stay
    high. Real walking-then-fall scenarios are still surfaced via the
    per-window grouped fall events; only the session-level alert is
    capped.
    """
    summary = _derive_alert_summary(
        *_make_inputs(har_top="locomotion", har_fraction=0.8, placement_state="pocket"),
    )
    assert summary.warning_level == "medium"
    assert summary.likely_fall_detected is True
    assert "locomotion" in summary.recommended_message.lower()


def test_likely_fall_with_walking_har_majority_downgrades_to_medium():
    """Same gate as locomotion, exercised on a concrete activity label."""
    summary = _derive_alert_summary(
        *_make_inputs(har_top="walking", har_fraction=0.8, placement_state="pocket"),
    )
    assert summary.warning_level == "medium"
    assert "walking" in summary.recommended_message.lower()


def test_likely_fall_with_locomotion_har_below_threshold_stays_high():
    """Noisy locomotion HAR should not veto — only a clear majority triggers
    the gate. This guards against accidentally suppressing a fall that
    happened to have a few walking-labelled windows around it."""
    summary = _derive_alert_summary(
        *_make_inputs(har_top="walking", har_fraction=0.4, placement_state="pocket"),
    )
    assert summary.warning_level == "high"


def test_likely_fall_with_static_har_majority_downgrades_to_medium():
    summary = _derive_alert_summary(
        *_make_inputs(har_top="static", har_fraction=0.9, placement_state="pocket"),
    )
    assert summary.warning_level == "medium"
    assert "static" in summary.recommended_message.lower()


def test_likely_fall_with_static_har_below_threshold_stays_high():
    """Noisy HAR should not veto — only a clear majority triggers the gate."""
    summary = _derive_alert_summary(
        *_make_inputs(har_top="static", har_fraction=0.4, placement_state="pocket"),
    )
    assert summary.warning_level == "high"


def test_likely_fall_with_missing_har_stays_high():
    summary = _derive_alert_summary(
        *_make_inputs(har_top=None, har_fraction=None, placement_state="pocket"),
    )
    assert summary.warning_level == "high"


def test_placement_and_har_gates_compose_not_stack():
    """Both gates firing should downgrade exactly one step, not two."""
    summary = _derive_alert_summary(
        *_make_inputs(har_top="static", har_fraction=0.9, placement_state="on_surface"),
    )
    # Without either gate the level would be "high"; with gate(s), "medium".
    # The test guards against a future change that accidentally downgrades twice.
    assert summary.warning_level == "medium"


def test_high_risk_vulnerability_pattern_also_downgrades_on_static_har():
    summary = _derive_alert_summary(
        *_make_inputs(
            har_top="static",
            har_fraction=0.9,
            placement_state="pocket",
            latest_state="high_risk_vulnerable_state",
            latest_level="high",
        ),
    )
    assert summary.warning_level == "medium"
    assert "static" in summary.recommended_message.lower()