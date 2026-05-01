from __future__ import annotations

import pytest

from fusion.fall_event import FallEventState
from fusion.state_machine import (
    MonitoringState,
    StateMachineConfig,
    StateMachineInputs,
    StateMachineState,
    step_state_machine,
)
from fusion.vulnerability_score import VulnerabilityLevel


def make_inputs(
    *,
    fall_event_state: FallEventState = FallEventState.NO_EVENT,
    fall_event_confidence: float = 0.0,
    vulnerability_level: VulnerabilityLevel = VulnerabilityLevel.LOW,
    vulnerability_score: float = 0.0,
    har_label: str | None = None,
    recovery_detected: bool = False,
) -> StateMachineInputs:
    return StateMachineInputs(
        fall_event_state=fall_event_state,
        fall_event_confidence=fall_event_confidence,
        vulnerability_level=vulnerability_level,
        vulnerability_score=vulnerability_score,
        har_label=har_label,
        recovery_detected=recovery_detected,
    )


def test_inputs_validate_rejects_bad_confidence_ranges() -> None:
    with pytest.raises(ValueError, match="fall_event_confidence"):
        make_inputs(fall_event_confidence=1.5).validate()

    with pytest.raises(ValueError, match="vulnerability_score"):
        make_inputs(vulnerability_score=-0.1).validate()


def test_normal_to_monitoring_event_on_suspicious_input() -> None:
    state = StateMachineState()

    result = step_state_machine(
        state,
        make_inputs(
            fall_event_state=FallEventState.IMPACT_ONLY,
            fall_event_confidence=0.35,
            vulnerability_level=VulnerabilityLevel.LOW,
            vulnerability_score=0.22,
            har_label="static",
        ),
    )

    assert result.previous_state == MonitoringState.NORMAL
    assert result.current_state == MonitoringState.MONITORING_EVENT
    assert result.escalated is True
    assert result.deescalated is False
    assert "entered monitoring after suspicious event" in result.reasons
    assert state.consecutive_suspicious_steps == 1
    assert len(state.history) == 1


def test_normal_to_high_risk_on_immediate_high_risk_input() -> None:
    state = StateMachineState()

    result = step_state_machine(
        state,
        make_inputs(
            fall_event_state=FallEventState.PROBABLE_FALL,
            fall_event_confidence=0.92,
            vulnerability_level=VulnerabilityLevel.HIGH,
            vulnerability_score=0.88,
            har_label="static",
        ),
    )

    assert result.previous_state == MonitoringState.NORMAL
    assert result.current_state == MonitoringState.HIGH_RISK_VULNERABLE_STATE
    assert result.escalated is True
    assert "high-risk evidence triggered immediate escalation" in result.reasons


def test_monitoring_event_to_suspected_incident_after_repeated_suspicion() -> None:
    state = StateMachineState(
        state=MonitoringState.MONITORING_EVENT,
        consecutive_suspicious_steps=1,
    )

    result = step_state_machine(
        state,
        make_inputs(
            fall_event_state=FallEventState.IMPACT_ONLY,
            fall_event_confidence=0.40,
            vulnerability_level=VulnerabilityLevel.LOW,
            vulnerability_score=0.30,
            har_label="static",
        ),
    )

    assert result.previous_state == MonitoringState.MONITORING_EVENT
    assert result.current_state == MonitoringState.SUSPECTED_INCIDENT
    assert result.escalated is True
    assert "monitoring escalated to suspected incident" in result.reasons
    assert state.consecutive_suspicious_steps == 2


def test_monitoring_event_to_suspected_incident_on_medium_vulnerability() -> None:
    state = StateMachineState(state=MonitoringState.MONITORING_EVENT)

    result = step_state_machine(
        state,
        make_inputs(
            fall_event_state=FallEventState.IMPACT_ONLY,
            fall_event_confidence=0.33,
            vulnerability_level=VulnerabilityLevel.MEDIUM,
            vulnerability_score=0.55,
            har_label="static",
        ),
    )

    assert result.current_state == MonitoringState.SUSPECTED_INCIDENT
    assert result.escalated is True


def test_suspected_incident_to_vulnerable_state() -> None:
    state = StateMachineState(state=MonitoringState.SUSPECTED_INCIDENT)

    result = step_state_machine(
        state,
        make_inputs(
            fall_event_state=FallEventState.IMPACT_ONLY,
            fall_event_confidence=0.36,
            vulnerability_level=VulnerabilityLevel.MEDIUM,
            vulnerability_score=0.58,
            har_label="static",
        ),
    )

    assert result.previous_state == MonitoringState.SUSPECTED_INCIDENT
    assert result.current_state == MonitoringState.VULNERABLE_STATE
    assert result.escalated is True
    assert "medium vulnerability promoted to vulnerable state" in result.reasons


def test_vulnerable_state_to_high_risk_vulnerable_state() -> None:
    state = StateMachineState(state=MonitoringState.VULNERABLE_STATE)

    result = step_state_machine(
        state,
        make_inputs(
            fall_event_state=FallEventState.PROBABLE_FALL,
            fall_event_confidence=0.91,
            vulnerability_level=VulnerabilityLevel.HIGH,
            vulnerability_score=0.90,
            har_label="static",
        ),
    )

    assert result.previous_state == MonitoringState.VULNERABLE_STATE
    assert result.current_state == MonitoringState.HIGH_RISK_VULNERABLE_STATE
    assert result.escalated is True
    assert "vulnerable state escalated to high risk" in result.reasons


def test_suspected_incident_to_recovery_observed() -> None:
    state = StateMachineState(state=MonitoringState.SUSPECTED_INCIDENT)

    result = step_state_machine(
        state,
        make_inputs(
            fall_event_state=FallEventState.NO_EVENT,
            fall_event_confidence=0.0,
            vulnerability_level=VulnerabilityLevel.LOW,
            vulnerability_score=0.10,
            har_label="locomotion",
            recovery_detected=True,
        ),
    )

    assert result.previous_state == MonitoringState.SUSPECTED_INCIDENT
    assert result.current_state == MonitoringState.RECOVERY_OBSERVED
    assert result.deescalated is True
    assert "recovery observed after suspected incident" in result.reasons


def test_high_risk_needs_sustained_recovery_before_downgrade() -> None:
    config = StateMachineConfig(recovery_cooldown_steps=2)
    state = StateMachineState(state=MonitoringState.HIGH_RISK_VULNERABLE_STATE)

    first = step_state_machine(
        state,
        make_inputs(
            fall_event_state=FallEventState.NO_EVENT,
            vulnerability_level=VulnerabilityLevel.LOW,
            vulnerability_score=0.12,
            har_label="locomotion",
            recovery_detected=True,
        ),
        config=config,
    )
    assert first.current_state == MonitoringState.HIGH_RISK_VULNERABLE_STATE
    assert first.deescalated is False

    second = step_state_machine(
        state,
        make_inputs(
            fall_event_state=FallEventState.NO_EVENT,
            vulnerability_level=VulnerabilityLevel.LOW,
            vulnerability_score=0.08,
            har_label="locomotion",
            recovery_detected=True,
        ),
        config=config,
    )
    assert second.current_state == MonitoringState.RECOVERY_OBSERVED
    assert second.deescalated is True
    assert "high-risk state downgraded after sustained recovery" in second.reasons


def test_recovery_observed_returns_to_normal_after_stable_recovery() -> None:
    config = StateMachineConfig(recovery_cooldown_steps=2)
    state = StateMachineState(
        state=MonitoringState.RECOVERY_OBSERVED,
        consecutive_normal_steps=1,
    )

    result = step_state_machine(
        state,
        make_inputs(
            fall_event_state=FallEventState.NO_EVENT,
            vulnerability_level=VulnerabilityLevel.LOW,
            vulnerability_score=0.05,
            har_label="locomotion",
            recovery_detected=False,
        ),
        config=config,
    )

    assert result.previous_state == MonitoringState.RECOVERY_OBSERVED
    assert result.current_state == MonitoringState.NORMAL
    assert result.deescalated is True
    assert "returned to normal after stable recovery" in result.reasons


def test_recovery_observed_reescalates_on_new_high_risk_event() -> None:
    state = StateMachineState(state=MonitoringState.RECOVERY_OBSERVED)

    result = step_state_machine(
        state,
        make_inputs(
            fall_event_state=FallEventState.PROBABLE_FALL,
            fall_event_confidence=0.95,
            vulnerability_level=VulnerabilityLevel.HIGH,
            vulnerability_score=0.92,
            har_label="static",
        ),
    )

    assert result.previous_state == MonitoringState.RECOVERY_OBSERVED
    assert result.current_state == MonitoringState.HIGH_RISK_VULNERABLE_STATE
    assert result.escalated is True
    assert "recovery interrupted by high-risk evidence" in result.reasons


def test_monitoring_event_clears_after_normal_behaviour() -> None:
    config = StateMachineConfig(clear_event_after_normal_steps=2)
    state = StateMachineState(
        state=MonitoringState.MONITORING_EVENT,
        consecutive_normal_steps=1,
    )

    result = step_state_machine(
        state,
        make_inputs(
            fall_event_state=FallEventState.NO_EVENT,
            vulnerability_level=VulnerabilityLevel.LOW,
            vulnerability_score=0.02,
            har_label="locomotion",
            recovery_detected=False,
        ),
        config=config,
    )

    assert result.current_state == MonitoringState.NORMAL
    assert result.deescalated is True
    assert "monitoring cleared after normal behaviour" in result.reasons


def test_history_is_appended_with_transition_summary() -> None:
    state = StateMachineState()

    step_state_machine(
        state,
        make_inputs(
            fall_event_state=FallEventState.IMPACT_ONLY,
            fall_event_confidence=0.30,
            vulnerability_level=VulnerabilityLevel.LOW,
            vulnerability_score=0.20,
            har_label="static",
        ),
    )

    assert len(state.history) == 1
    assert "normal -> monitoring_event" in state.history[0]
    assert "fall=impact_only" in state.history[0]