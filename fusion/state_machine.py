# fusion/state_machine.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from fusion.fall_event import FallEventState
from fusion.vulnerability_score import VulnerabilityLevel


class MonitoringState(str, Enum):
    NORMAL = "normal"
    MONITORING_EVENT = "monitoring_event"
    SUSPECTED_INCIDENT = "suspected_incident"
    RECOVERY_OBSERVED = "recovery_observed"
    VULNERABLE_STATE = "vulnerable_state"
    HIGH_RISK_VULNERABLE_STATE = "high_risk_vulnerable_state"


@dataclass(slots=True)
class StateMachineInputs:
    """
    Inputs for one decision step.

    These should come from:
    - fusion.fall_event.classify_fall_event(...)
    - fusion.vulnerability_score.score_vulnerability(...)
    - upstream movement / recovery logic
    """

    fall_event_state: FallEventState = FallEventState.NO_EVENT
    fall_event_confidence: float = 0.0

    vulnerability_level: VulnerabilityLevel = VulnerabilityLevel.LOW
    vulnerability_score: float = 0.0

    har_label: Optional[str] = None
    recovery_detected: bool = False

    def validate(self) -> None:
        if not 0.0 <= self.fall_event_confidence <= 1.0:
            raise ValueError(
                f"fall_event_confidence must be between 0.0 and 1.0, got {self.fall_event_confidence}"
            )
        if not 0.0 <= self.vulnerability_score <= 1.0:
            raise ValueError(
                f"vulnerability_score must be between 0.0 and 1.0, got {self.vulnerability_score}"
            )


@dataclass(slots=True)
class StateMachineConfig:
    """
    Small, interpretable transition settings.

    These are intentionally simple and can be tuned later once you start
    replaying real sequences.
    """

    promote_to_vulnerable_after_steps: int = 2
    promote_to_high_risk_after_steps: int = 2
    recovery_cooldown_steps: int = 2
    clear_event_after_normal_steps: int = 2


@dataclass(slots=True)
class StateMachineState:
    state: MonitoringState = MonitoringState.NORMAL
    consecutive_suspicious_steps: int = 0
    consecutive_recovery_steps: int = 0
    consecutive_normal_steps: int = 0

    last_fall_event_state: FallEventState = FallEventState.NO_EVENT
    last_vulnerability_level: VulnerabilityLevel = VulnerabilityLevel.LOW

    history: List[str] = field(default_factory=list)


@dataclass(slots=True)
class StateTransitionResult:
    previous_state: MonitoringState
    current_state: MonitoringState
    escalated: bool
    deescalated: bool
    reasons: List[str] = field(default_factory=list)
    counters: Dict[str, int] = field(default_factory=dict)


def _is_stationary_label(har_label: Optional[str]) -> bool:
    if not har_label:
        return False
    return har_label.strip().lower() in {"static", "stationary", "sitting", "standing", "lying", "laying"}


def _is_moving_label(har_label: Optional[str]) -> bool:
    if not har_label:
        return False
    return har_label.strip().lower() in {"locomotion", "walking", "running", "stairs", "dynamic"}


def _is_suspicious(inputs: StateMachineInputs) -> bool:
    return (
        inputs.fall_event_state in {FallEventState.IMPACT_ONLY, FallEventState.POSSIBLE_FALL, FallEventState.PROBABLE_FALL}
        or inputs.vulnerability_level in {VulnerabilityLevel.MEDIUM, VulnerabilityLevel.HIGH}
    )


def _is_high_risk(inputs: StateMachineInputs) -> bool:
    return (
        inputs.fall_event_state == FallEventState.PROBABLE_FALL
        or inputs.vulnerability_level == VulnerabilityLevel.HIGH
        or (
            inputs.fall_event_state == FallEventState.POSSIBLE_FALL
            and inputs.vulnerability_score >= 0.70
        )
    )


def _is_normalising(inputs: StateMachineInputs) -> bool:
    return (
        inputs.fall_event_state == FallEventState.NO_EVENT
        and inputs.vulnerability_level == VulnerabilityLevel.LOW
        and (
            inputs.recovery_detected
            or _is_moving_label(inputs.har_label)
        )
    )


def step_state_machine(
    current: StateMachineState,
    inputs: StateMachineInputs,
    config: Optional[StateMachineConfig] = None,
) -> StateTransitionResult:
    """
    Apply one step of transition logic.

    The main idea:
    - suspicious event -> monitoring
    - repeated suspicious / medium vulnerability -> vulnerable_state
    - probable fall / high vulnerability -> high_risk_vulnerable_state
    - recovery -> recovery_observed
    - repeated normalising behaviour -> normal
    """
    inputs.validate()
    config = config or StateMachineConfig()

    previous_state = current.state
    reasons: List[str] = []
    escalated = False
    deescalated = False

    suspicious = _is_suspicious(inputs)
    high_risk = _is_high_risk(inputs)
    normalising = _is_normalising(inputs)

    if suspicious:
        current.consecutive_suspicious_steps += 1
        current.consecutive_normal_steps = 0
        reasons.append("suspicious event evidence present")
    else:
        current.consecutive_suspicious_steps = 0

    if inputs.recovery_detected:
        current.consecutive_recovery_steps += 1
        reasons.append("recovery detected")
    elif normalising:
        current.consecutive_recovery_steps += 1
        reasons.append("normalising movement observed")
    else:
        current.consecutive_recovery_steps = 0

    if not suspicious and normalising:
        current.consecutive_normal_steps += 1
    elif not suspicious and inputs.fall_event_state == FallEventState.NO_EVENT:
        current.consecutive_normal_steps += 1
    else:
        current.consecutive_normal_steps = 0

    if current.state == MonitoringState.NORMAL:
        if high_risk:
            current.state = MonitoringState.HIGH_RISK_VULNERABLE_STATE
            reasons.append("high-risk evidence triggered immediate escalation")
            escalated = True
        elif suspicious:
            current.state = MonitoringState.MONITORING_EVENT
            reasons.append("entered monitoring after suspicious event")
            escalated = True

    elif current.state == MonitoringState.MONITORING_EVENT:
        if high_risk:
            current.state = MonitoringState.HIGH_RISK_VULNERABLE_STATE
            reasons.append("monitoring escalated to high risk")
            escalated = True
        elif (
            inputs.vulnerability_level == VulnerabilityLevel.MEDIUM
            or inputs.fall_event_state == FallEventState.POSSIBLE_FALL
            or current.consecutive_suspicious_steps >= config.promote_to_vulnerable_after_steps
        ):
            current.state = MonitoringState.SUSPECTED_INCIDENT
            reasons.append("monitoring escalated to suspected incident")
            escalated = True
        elif current.consecutive_normal_steps >= config.clear_event_after_normal_steps:
            current.state = MonitoringState.NORMAL
            reasons.append("monitoring cleared after normal behaviour")
            deescalated = True

    elif current.state == MonitoringState.SUSPECTED_INCIDENT:
        if high_risk:
            current.state = MonitoringState.HIGH_RISK_VULNERABLE_STATE
            reasons.append("suspected incident escalated to high-risk vulnerable state")
            escalated = True
        elif inputs.vulnerability_level == VulnerabilityLevel.MEDIUM:
            current.state = MonitoringState.VULNERABLE_STATE
            reasons.append("medium vulnerability promoted to vulnerable state")
            escalated = True
        elif current.consecutive_recovery_steps >= 1:
            current.state = MonitoringState.RECOVERY_OBSERVED
            reasons.append("recovery observed after suspected incident")
            deescalated = True
        elif current.consecutive_normal_steps >= config.clear_event_after_normal_steps:
            current.state = MonitoringState.NORMAL
            reasons.append("suspected incident cleared after normal behaviour")
            deescalated = True

    elif current.state == MonitoringState.VULNERABLE_STATE:
        if (
            high_risk
            or current.consecutive_suspicious_steps >= config.promote_to_high_risk_after_steps
        ):
            current.state = MonitoringState.HIGH_RISK_VULNERABLE_STATE
            reasons.append("vulnerable state escalated to high risk")
            escalated = True
        elif current.consecutive_recovery_steps >= 1:
            current.state = MonitoringState.RECOVERY_OBSERVED
            reasons.append("recovery observed from vulnerable state")
            deescalated = True

    elif current.state == MonitoringState.HIGH_RISK_VULNERABLE_STATE:
        if current.consecutive_recovery_steps >= config.recovery_cooldown_steps:
            current.state = MonitoringState.RECOVERY_OBSERVED
            reasons.append("high-risk state downgraded after sustained recovery")
            deescalated = True

    elif current.state == MonitoringState.RECOVERY_OBSERVED:
        if high_risk:
            current.state = MonitoringState.HIGH_RISK_VULNERABLE_STATE
            reasons.append("recovery interrupted by high-risk evidence")
            escalated = True
        elif suspicious and not normalising:
            current.state = MonitoringState.MONITORING_EVENT
            reasons.append("new suspicious evidence after recovery")
            escalated = True
        elif current.consecutive_normal_steps >= config.recovery_cooldown_steps:
            current.state = MonitoringState.NORMAL
            reasons.append("returned to normal after stable recovery")
            deescalated = True

    current.last_fall_event_state = inputs.fall_event_state
    current.last_vulnerability_level = inputs.vulnerability_level

    history_entry = (
        f"{previous_state.value} -> {current.state.value} | "
        f"fall={inputs.fall_event_state.value} "
        f"fall_conf={inputs.fall_event_confidence:.3f} "
        f"vuln={inputs.vulnerability_level.value} "
        f"vuln_score={inputs.vulnerability_score:.3f} "
        f"recovery={inputs.recovery_detected}"
    )
    current.history.append(history_entry)

    return StateTransitionResult(
        previous_state=previous_state,
        current_state=current.state,
        escalated=escalated,
        deescalated=deescalated,
        reasons=reasons,
        counters={
            "consecutive_suspicious_steps": current.consecutive_suspicious_steps,
            "consecutive_recovery_steps": current.consecutive_recovery_steps,
            "consecutive_normal_steps": current.consecutive_normal_steps,
        },
    )