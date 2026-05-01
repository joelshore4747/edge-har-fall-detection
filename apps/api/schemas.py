from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PlacementType(str, Enum):
    pocket = "pocket"
    hand = "hand"
    desk = "desk"
    bag = "bag"
    unknown = "unknown"


class SourceType(str, Enum):
    mobile_app = "mobile_app"
    csv = "csv"
    phone_folder = "phone_folder"
    debug = "debug"


class WarningLevel(str, Enum):
    none = "none"
    low = "low"
    medium = "medium"
    high = "high"


class TaskType(str, Enum):
    runtime = "runtime"
    evaluation = "evaluation"
    import_session = "import_session"
    debug = "debug"


class RecordingMode(str, Enum):
    live_capture = "live_capture"
    import_session = "import_session"
    replay = "replay"
    demo = "demo"


class RuntimeMode(str, Enum):
    mobile_live = "mobile_live"
    desktop_demo = "desktop_demo"
    session_replay = "session_replay"
    debug = "debug"


class UserFeedbackType(str, Enum):
    confirmed_fall = "confirmed_fall"
    false_alarm = "false_alarm"
    uncertain = "uncertain"
    corrected_label = "corrected_label"


class FeedbackTargetType(str, Enum):
    session = "session"
    grouped_fall_event = "grouped_fall_event"
    timeline_event = "timeline_event"
    transition_event = "transition_event"
    window = "window"


class CanonicalSessionLabel(str, Enum):
    static = "static"
    walking = "walking"
    stairs = "stairs"
    fall = "fall"
    other = "other"
    unknown = "unknown"


class SessionAnnotationSource(str, Enum):
    mobile = "mobile"
    admin = "admin"
    system = "system"


_CANONICAL_SESSION_LABEL_ALIASES = {
    "static": CanonicalSessionLabel.static,
    "stationary": CanonicalSessionLabel.static,
    "standing": CanonicalSessionLabel.static,
    "sitting": CanonicalSessionLabel.static,
    "lying": CanonicalSessionLabel.static,
    "laying": CanonicalSessionLabel.static,
    "walking": CanonicalSessionLabel.walking,
    "walk": CanonicalSessionLabel.walking,
    "jogging": CanonicalSessionLabel.walking,
    "running": CanonicalSessionLabel.walking,
    "stairs": CanonicalSessionLabel.stairs,
    "upstairs": CanonicalSessionLabel.stairs,
    "downstairs": CanonicalSessionLabel.stairs,
    "walking_upstairs": CanonicalSessionLabel.stairs,
    "walking_downstairs": CanonicalSessionLabel.stairs,
    "ascending_stairs": CanonicalSessionLabel.stairs,
    "descending_stairs": CanonicalSessionLabel.stairs,
    "stairs_up": CanonicalSessionLabel.stairs,
    "stairs_down": CanonicalSessionLabel.stairs,
    "fall": CanonicalSessionLabel.fall,
    "falling": CanonicalSessionLabel.fall,
    "other": CanonicalSessionLabel.other,
    "transition": CanonicalSessionLabel.other,
    "unknown": CanonicalSessionLabel.unknown,
}


def _finite_number(value: float | None, *, field_name: str) -> float | None:
    if value is None:
        return None
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{field_name} must be finite")
    return value


def normalise_canonical_session_label(
    value: str | CanonicalSessionLabel | None,
    *,
    fallback: CanonicalSessionLabel | None = None,
) -> CanonicalSessionLabel | None:
    if value is None:
        return fallback
    if isinstance(value, CanonicalSessionLabel):
        return value

    normalized = value.strip().lower()
    if not normalized:
        return fallback

    resolved = _CANONICAL_SESSION_LABEL_ALIASES.get(normalized)
    if resolved is None:
        raise ValueError(
            "label must be one of: static, walking, stairs, fall, other, unknown"
        )
    return resolved


class RequestContext(BaseModel):
    request_id: UUID = Field(default_factory=uuid4)
    trace_id: str | None = None
    client_version: str | None = None

    model_config = ConfigDict(extra="allow")


class SelfServiceRegistrationRequest(BaseModel):
    username: str = Field(..., min_length=6, max_length=64)
    password: str = Field(..., min_length=8, max_length=128)
    subject_id: str = Field(..., min_length=6, max_length=80)
    display_name: str | None = Field(default=None, max_length=120)
    device_platform: str = Field(default="unknown", max_length=40)
    device_model: str | None = Field(default=None, max_length=120)
    app_version: str | None = Field(default=None, max_length=40)
    app_build: str | None = Field(default=None, max_length=40)
    request_context: RequestContext | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("username", "subject_id")
    @classmethod
    def validate_registration_identifier(cls, value: str, info) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError(f"{info.field_name} must not be blank")
        return normalized

    @field_validator("display_name", "device_platform", "device_model", "app_version", "app_build")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class SelfServiceRegistrationResponse(BaseModel):
    status: str
    username: str
    subject_id: str
    display_name: str | None = None
    role: str = "user"
    created: bool = True

    model_config = ConfigDict(extra="ignore")


class AuthenticatedUserResponse(BaseModel):
    status: str
    username: str | None = None
    subject_id: str | None = None
    role: str = "anonymous"
    auth_required: bool = True

    model_config = ConfigDict(extra="ignore")


class AdminAuthLoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=128)
    password: str = Field(..., min_length=1, max_length=128)

    model_config = ConfigDict(extra="forbid")

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("username must not be blank")
        return normalized


class AdminAuthSessionResponse(BaseModel):
    status: str
    username: str | None = None
    subject_id: str | None = None
    role: str = "anonymous"

    model_config = ConfigDict(extra="ignore")


class SessionAnnotationRequest(BaseModel):
    label: CanonicalSessionLabel
    notes: str | None = Field(default=None, max_length=2000)
    reviewer_id: str | None = Field(default=None, max_length=120)
    source: SessionAnnotationSource | None = None
    request_context: RequestContext | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("label", mode="before")
    @classmethod
    def validate_label(cls, value: str | CanonicalSessionLabel | None) -> CanonicalSessionLabel:
        normalized = normalise_canonical_session_label(value)
        if normalized is None:
            raise ValueError("label is required")
        return normalized

    @field_validator("notes", "reviewer_id")
    @classmethod
    def validate_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class SessionAnnotationRecord(BaseModel):
    annotation_id: UUID
    app_session_id: UUID
    label: CanonicalSessionLabel
    source: SessionAnnotationSource
    reviewer_identifier: str | None = None
    auth_account_id: UUID | None = None
    created_by_username: str | None = None
    request_id: UUID | None = None
    notes: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(extra="ignore")


class SessionAnnotationListResponse(BaseModel):
    app_session_id: UUID
    annotations: list[SessionAnnotationRecord] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class SensorSample(BaseModel):
    timestamp: float = Field(..., description="Seconds from session start")
    ax: float
    ay: float
    az: float
    gx: float | None = None
    gy: float | None = None
    gz: float | None = None

    model_config = ConfigDict(extra="ignore")

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: float) -> float:
        _finite_number(value, field_name="timestamp")
        if value < 0:
            raise ValueError("timestamp must be >= 0")
        return value

    @field_validator("ax", "ay", "az", "gx", "gy", "gz")
    @classmethod
    def validate_sensor_axis(cls, value: float | None, info) -> float | None:
        return _finite_number(value, field_name=info.field_name)


class SessionMetadata(BaseModel):
    session_id: str = Field(..., min_length=1)
    subject_id: str = Field(default="anonymous_user")
    placement: PlacementType = Field(default=PlacementType.pocket)
    task_type: TaskType = Field(default=TaskType.runtime)
    dataset_name: str = Field(default="APP_RUNTIME")
    source_type: SourceType = Field(default=SourceType.mobile_app)
    device_platform: str = Field(default="unknown")
    device_model: str | None = None
    device_id: str | None = Field(default=None, max_length=160)
    installation_id: str | None = Field(default=None, max_length=160)
    sampling_rate_hz: float | None = Field(default=None, gt=0)
    activity_label: CanonicalSessionLabel | None = None
    notes: str | None = None
    app_version: str | None = None
    app_build: str | None = None
    recording_mode: RecordingMode = Field(default=RecordingMode.live_capture)
    runtime_mode: RuntimeMode = Field(default=RuntimeMode.mobile_live)

    model_config = ConfigDict(extra="allow")

    @field_validator("activity_label", mode="before")
    @classmethod
    def validate_activity_label(
        cls,
        value: str | CanonicalSessionLabel | None,
    ) -> CanonicalSessionLabel | None:
        return normalise_canonical_session_label(value, fallback=None)

    @field_validator("device_platform")
    @classmethod
    def validate_device_platform(cls, value: str | None) -> str:
        normalized = (value or "").strip()
        return normalized or "unknown"

    @field_validator(
        "device_model",
        "device_id",
        "installation_id",
        "notes",
        "app_version",
        "app_build",
    )
    @classmethod
    def validate_metadata_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class RuntimeSessionRequest(BaseModel):
    metadata: SessionMetadata
    samples: list[SensorSample] = Field(..., min_length=1)
    include_har_windows: bool = False
    include_fall_windows: bool = False
    include_vulnerability_windows: bool = False
    include_combined_timeline: bool = True
    include_grouped_fall_events: bool = True
    include_point_timeline: bool = False
    include_timeline_events: bool = True
    include_transition_events: bool = True
    request_context: RequestContext | None = None

    model_config = ConfigDict(extra="forbid")


class SourceSummary(BaseModel):
    input_sample_count: int = 0
    session_duration_seconds: float | None = None
    has_gyro: bool = False
    estimated_sampling_rate_hz: float | None = None
    source_type: SourceType = SourceType.mobile_app
    device_platform: str = "unknown"

    model_config = ConfigDict(extra="ignore")


class PlacementSummary(BaseModel):
    placement_state: str
    placement_confidence: float | None = None
    state_fraction: float | None = None
    state_counts: dict[str, int] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")


class HarSummary(BaseModel):
    top_label: str | None = None
    top_label_fraction: float | None = None
    label_counts: dict[str, int] = Field(default_factory=dict)
    total_windows: int = 0

    model_config = ConfigDict(extra="ignore")


class FallSummary(BaseModel):
    likely_fall_detected: bool
    positive_window_count: int = 0
    grouped_event_count: int = 0
    top_fall_probability: float | None = None
    mean_fall_probability: float | None = None

    model_config = ConfigDict(extra="ignore")


class VulnerabilitySummary(BaseModel):
    enabled: bool = True
    event_profile: str | None = None
    vulnerability_profile: str | None = None
    window_count: int = 0
    session_count: int = 0
    alert_worthy_window_count: int = 0
    fall_event_state_counts: dict[str, int] = Field(default_factory=dict)
    vulnerability_level_counts: dict[str, int] = Field(default_factory=dict)
    monitoring_state_counts: dict[str, int] = Field(default_factory=dict)
    top_vulnerability_score: float | None = None
    mean_vulnerability_score: float | None = None
    latest_vulnerability_score: float | None = None
    latest_vulnerability_level: str | None = None
    latest_monitoring_state: str | None = None
    latest_fall_event_state: str | None = None
    top_vulnerability_reasons: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class RuntimeAlertSummary(BaseModel):
    warning_level: WarningLevel
    likely_fall_detected: bool
    top_har_label: str | None = None
    top_har_fraction: float | None = None
    grouped_fall_event_count: int = 0
    top_fall_probability: float | None = None
    top_vulnerability_score: float | None = None
    latest_vulnerability_level: str | None = None
    latest_monitoring_state: str | None = None
    latest_fall_event_state: str | None = None
    recommended_message: str

    model_config = ConfigDict(extra="ignore")


class RuntimeDebugSummary(BaseModel):
    input_sample_count: int = 0
    has_gyro: bool = False
    estimated_sampling_rate_hz: float | None = None
    runtime_mode: RuntimeMode = RuntimeMode.mobile_live
    processing_notes: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class ModelInfo(BaseModel):
    har_model_name: str | None = None
    har_model_version: str | None = None
    fall_model_name: str | None = None
    fall_model_version: str | None = None
    api_version: str

    model_config = ConfigDict(extra="ignore")


class HarWindowPrediction(BaseModel):
    window_id: str | None = None
    start_ts: float | None = None
    end_ts: float | None = None
    midpoint_ts: float | None = None
    predicted_label: str
    predicted_confidence: float | None = None

    model_config = ConfigDict(extra="ignore")


class FallWindowPrediction(BaseModel):
    window_id: str | None = None
    start_ts: float | None = None
    end_ts: float | None = None
    midpoint_ts: float | None = None
    predicted_label: str
    predicted_probability: float | None = None
    predicted_is_fall: bool | None = None

    model_config = ConfigDict(extra="ignore")


class VulnerabilityWindowPrediction(BaseModel):
    window_id: str | None = None
    start_ts: float | None = None
    end_ts: float | None = None
    midpoint_ts: float | None = None
    har_label: str | None = None
    har_confidence: float | None = None
    fall_probability: float | None = None
    fall_predicted_label: str | None = None
    fall_predicted_is_fall: bool | None = None
    fall_event_state: str | None = None
    fall_event_confidence: float | None = None
    fall_event_reasons: list[str] = Field(default_factory=list)
    fall_event_contributions: dict[str, float] = Field(default_factory=dict)
    vulnerability_level: str | None = None
    vulnerability_score: float | None = None
    vulnerability_reasons: list[str] = Field(default_factory=list)
    vulnerability_contributions: dict[str, float] = Field(default_factory=dict)
    monitoring_state: str | None = None
    escalated: bool = False
    deescalated: bool = False
    state_machine_reasons: list[str] = Field(default_factory=list)
    state_machine_counters: dict[str, int] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")


class GroupedFallEvent(BaseModel):
    event_id: str
    event_start_ts: float
    event_end_ts: float
    event_duration_seconds: float
    n_positive_windows: int
    peak_probability: float | None = None
    mean_probability: float | None = None
    median_probability: float | None = None

    model_config = ConfigDict(extra="ignore")


class CombinedTimelinePoint(BaseModel):
    timestamp: float
    har_label: str | None = None
    har_confidence: float | None = None
    fall_probability: float | None = None
    fall_detected: bool | None = None
    window_id: str | None = None

    model_config = ConfigDict(extra="ignore")


class PointTimelinePoint(BaseModel):
    midpoint_ts: float
    activity_label: str | None = None
    placement_label: str | None = None
    activity_confidence: float | None = None
    placement_confidence: float | None = None
    fall_probability: float | None = None
    elevated_fall: bool | None = None

    model_config = ConfigDict(extra="ignore")


class TimelineEvent(BaseModel):
    event_id: str
    start_ts: float
    end_ts: float
    duration_seconds: float
    midpoint_ts: float | None = None
    point_count: int = 0
    activity_label: str
    placement_label: str
    activity_confidence_mean: float | None = None
    placement_confidence_mean: float | None = None
    fall_probability_peak: float | None = None
    fall_probability_mean: float | None = None
    likely_fall: bool = False
    event_kind: str
    related_grouped_fall_event_ids: list[str] = Field(default_factory=list)
    description: str

    model_config = ConfigDict(extra="ignore")


class TransitionEvent(BaseModel):
    transition_id: str
    transition_ts: float
    from_event_id: str
    to_event_id: str
    transition_kind: str
    from_activity_label: str | None = None
    to_activity_label: str | None = None
    from_placement_label: str | None = None
    to_placement_label: str | None = None
    description: str

    model_config = ConfigDict(extra="ignore")


class SessionNarrativeSummary(BaseModel):
    session_id: str
    dataset_name: str
    subject_id: str
    total_duration_seconds: float
    event_count: int
    transition_count: int
    fall_event_count: int
    dominant_activity_label: str
    dominant_placement_label: str
    highest_fall_probability: float | None = None
    summary_text: str

    # Vulnerability + HAR-attenuation rollup. ``highest_fall_probability``
    # is the raw model output and is intentionally NOT attenuated by the
    # walking/stairs gate in fusion.vulnerability_score, so it overstates
    # confident-locomotion sessions. Admin clients should prefer
    # ``peak_vulnerability_score`` as the headline number and badge the
    # session when ``har_attenuation_applied`` is true.
    peak_vulnerability_score: float | None = None
    mean_vulnerability_score: float | None = None
    dominant_vulnerability_level: str | None = None
    har_attenuation_applied: bool = False
    har_attenuation_window_count: int = 0
    har_attenuation_label: str | None = None
    har_attenuation_confidence_mean: float | None = None

    model_config = ConfigDict(extra="ignore")


class RuntimeSessionResponse(BaseModel):
    request_id: UUID | None = None
    session_id: str
    persisted_user_id: UUID | None = None
    persisted_session_id: UUID | None = None
    persisted_inference_id: UUID | None = None
    source_summary: SourceSummary
    placement_summary: PlacementSummary
    har_summary: HarSummary
    fall_summary: FallSummary
    vulnerability_summary: VulnerabilitySummary
    alert_summary: RuntimeAlertSummary
    debug_summary: RuntimeDebugSummary
    model_info: ModelInfo

    grouped_fall_events: list[GroupedFallEvent] = Field(default_factory=list)
    har_windows: list[HarWindowPrediction] = Field(default_factory=list)
    fall_windows: list[FallWindowPrediction] = Field(default_factory=list)
    vulnerability_windows: list[VulnerabilityWindowPrediction] = Field(default_factory=list)
    combined_timeline: list[CombinedTimelinePoint] = Field(default_factory=list)

    point_timeline: list[PointTimelinePoint] = Field(default_factory=list)
    timeline_events: list[TimelineEvent] = Field(default_factory=list)
    transition_events: list[TransitionEvent] = Field(default_factory=list)
    session_narrative_summary: SessionNarrativeSummary | None = None
    narrative_summary: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")


class HealthResponse(BaseModel):
    status: str
    service_name: str
    version: str

    model_config = ConfigDict(extra="ignore")


class ErrorResponse(BaseModel):
    request_id: UUID | None = None
    error_code: str
    message: str
    details: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")


class PredictionFeedbackRequest(BaseModel):
    request_context: RequestContext | None = None
    session_id: str = Field(..., min_length=1)
    subject_id: str | None = None
    persisted_session_id: UUID | None = None
    persisted_inference_id: UUID | None = None
    target_type: FeedbackTargetType | None = None
    event_id: str | None = None
    window_id: str | None = None
    user_feedback: UserFeedbackType
    corrected_label: str | None = None
    reviewer_id: str | None = None
    notes: str | None = None

    model_config = ConfigDict(extra="forbid")


class PredictionFeedbackResponse(BaseModel):
    request_id: UUID | None = None
    session_id: str
    persisted_session_id: UUID | None = None
    persisted_inference_id: UUID | None = None
    persisted_feedback_id: UUID | None = None
    target_type: FeedbackTargetType | None = None
    event_id: str | None = None
    window_id: str | None = None
    user_feedback: UserFeedbackType
    message: str
    status: str = "accepted"
    recorded_at: str | None = None

    model_config = ConfigDict(extra="ignore")


class PersistedSessionRecord(BaseModel):
    app_session_id: UUID
    user_id: UUID
    device_id: UUID | None = None
    subject_id: str
    client_session_id: str
    request_id: UUID | None = None
    trace_id: str | None = None
    client_version: str | None = None
    dataset_name: str
    source_type: str
    task_type: str
    placement_declared: str
    device_platform: str
    device_model: str | None = None
    app_version: str | None = None
    app_build: str | None = None
    recording_mode: str
    runtime_mode: str
    recording_started_at: datetime | None = None
    recording_ended_at: datetime | None = None
    uploaded_at: datetime
    sampling_rate_hz: float | None = None
    sample_count: int
    has_gyro: bool
    duration_seconds: float | None = None
    session_name: str | None = None
    activity_label: str | None = None
    notes: str | None = None
    raw_storage_uri: str | None = None
    raw_storage_format: str | None = None
    raw_payload_sha256: str | None = None
    raw_payload_bytes: int | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(extra="ignore")


class PersistedSessionListItem(BaseModel):
    session: PersistedSessionRecord
    latest_inference_id: UUID | None = None
    latest_inference_request_id: UUID | None = None
    latest_inference_created_at: datetime | None = None
    latest_status: str | None = None
    latest_warning_level: str | None = None
    latest_likely_fall_detected: bool | None = None
    latest_top_har_label: str | None = None
    latest_top_fall_probability: float | None = None
    latest_grouped_fall_event_count: int | None = None
    latest_annotation_label: CanonicalSessionLabel | None = None
    latest_annotation_source: SessionAnnotationSource | None = None
    latest_annotation_created_at: datetime | None = None

    model_config = ConfigDict(extra="ignore")


class PersistedSessionListResponse(BaseModel):
    sessions: list[PersistedSessionListItem] = Field(default_factory=list)
    total_count: int
    limit: int
    offset: int

    model_config = ConfigDict(extra="ignore")


class PersistedInferenceDetail(BaseModel):
    inference_id: UUID
    request_id: UUID | None = None
    status: str
    error_message: str | None = None
    started_at: datetime
    completed_at: datetime | None = None
    created_at: datetime
    response: RuntimeSessionResponse

    model_config = ConfigDict(extra="ignore")


class PersistedFeedbackRecord(BaseModel):
    feedback_id: UUID
    app_session_id: UUID
    inference_id: UUID | None = None
    target_type: FeedbackTargetType
    target_event_key: str | None = None
    window_id: str | None = None
    feedback_type: UserFeedbackType
    corrected_label: str | None = None
    reviewer_identifier: str | None = None
    subject_key: str | None = None
    notes: str | None = None
    request_id: UUID | None = None
    recorded_at: datetime

    model_config = ConfigDict(extra="ignore")


class PersistedSessionDetailResponse(BaseModel):
    session: PersistedSessionRecord
    latest_inference: PersistedInferenceDetail | None = None
    feedback: list[PersistedFeedbackRecord] = Field(default_factory=list)
    annotations: list[SessionAnnotationRecord] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class AdminKpiTotals(BaseModel):
    users: int = 0
    sessions: int = 0
    inferences: int = 0
    grouped_fall_events: int = 0

    model_config = ConfigDict(extra="ignore")


class AdminRecentActivity(BaseModel):
    sessions_last_7_days: int = 0
    sessions_with_likely_fall: int = 0

    model_config = ConfigDict(extra="ignore")


class AdminChartDatum(BaseModel):
    label: str
    value: int = 0

    model_config = ConfigDict(extra="ignore")


class AdminOverviewCharts(BaseModel):
    sessions_by_day: list[AdminChartDatum] = Field(default_factory=list)
    fall_events_by_day: list[AdminChartDatum] = Field(default_factory=list)
    warning_level_distribution: list[AdminChartDatum] = Field(default_factory=list)
    top_har_labels: list[AdminChartDatum] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class AdminOverviewSessionRecord(BaseModel):
    app_session_id: UUID
    subject_id: str
    client_session_id: str
    device_platform: str
    device_model: str | None = None
    uploaded_at: datetime
    duration_seconds: float | None = None
    session_name: str | None = None
    activity_label: str | None = None

    model_config = ConfigDict(extra="ignore")


class AdminOverviewSessionItem(BaseModel):
    session: AdminOverviewSessionRecord
    latest_status: str | None = None
    latest_warning_level: str | None = None
    latest_likely_fall_detected: bool | None = None
    latest_top_har_label: str | None = None
    latest_top_fall_probability: float | None = None
    latest_grouped_fall_event_count: int | None = None

    model_config = ConfigDict(extra="ignore")


class AdminSessionListRecord(BaseModel):
    app_session_id: UUID
    subject_id: str
    client_session_id: str
    device_platform: str
    device_model: str | None = None
    uploaded_at: datetime
    duration_seconds: float | None = None
    session_name: str | None = None
    activity_label: str | None = None
    notes: str | None = None

    model_config = ConfigDict(extra="ignore")


class AdminSessionListItem(BaseModel):
    session: AdminSessionListRecord
    latest_status: str | None = None
    latest_warning_level: str | None = None
    latest_likely_fall_detected: bool | None = None
    latest_top_har_label: str | None = None
    latest_top_fall_probability: float | None = None
    latest_grouped_fall_event_count: int | None = None
    latest_annotation_label: CanonicalSessionLabel | None = None

    model_config = ConfigDict(extra="ignore")


class AdminOverviewResponse(BaseModel):
    totals: AdminKpiTotals
    recent_activity: AdminRecentActivity
    charts: AdminOverviewCharts
    recent_sessions: list[AdminOverviewSessionItem] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class AdminSessionListResponse(BaseModel):
    sessions: list[AdminSessionListItem] = Field(default_factory=list)
    total_count: int
    page: int
    page_size: int
    total_pages: int

    model_config = ConfigDict(extra="ignore")


class AdminSessionEvidenceCounts(BaseModel):
    grouped_fall_events: int = 0
    timeline_events: int = 0
    transition_events: int = 0
    feedback: int = 0
    annotations: int = 0

    model_config = ConfigDict(extra="ignore")


class AdminRuntimeSessionSummary(BaseModel):
    request_id: UUID | None = None
    session_id: str
    persisted_user_id: UUID | None = None
    persisted_session_id: UUID | None = None
    persisted_inference_id: UUID | None = None
    source_summary: SourceSummary
    placement_summary: PlacementSummary
    har_summary: HarSummary
    fall_summary: FallSummary
    vulnerability_summary: VulnerabilitySummary
    alert_summary: RuntimeAlertSummary
    model_info: ModelInfo
    session_narrative_summary: SessionNarrativeSummary | None = None
    narrative_summary: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")


class AdminPersistedInferenceSummary(BaseModel):
    inference_id: UUID
    request_id: UUID | None = None
    status: str
    error_message: str | None = None
    started_at: datetime
    completed_at: datetime | None = None
    created_at: datetime
    response: AdminRuntimeSessionSummary

    model_config = ConfigDict(extra="ignore")


class AdminSessionDetailSummaryResponse(BaseModel):
    session: PersistedSessionRecord
    latest_inference: AdminPersistedInferenceSummary | None = None
    latest_feedback: PersistedFeedbackRecord | None = None
    latest_annotation: SessionAnnotationRecord | None = None
    evidence_counts: AdminSessionEvidenceCounts

    model_config = ConfigDict(extra="ignore")


class AdminSessionEvidenceResponse(BaseModel):
    loaded_sections: list[str] = Field(default_factory=list)
    grouped_fall_events: list[GroupedFallEvent] = Field(default_factory=list)
    timeline_events: list[TimelineEvent] = Field(default_factory=list)
    transition_events: list[TransitionEvent] = Field(default_factory=list)
    feedback: list[PersistedFeedbackRecord] = Field(default_factory=list)
    annotations: list[SessionAnnotationRecord] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class DeleteSessionResponse(BaseModel):
    app_session_id: UUID
    deleted: bool
    message: str

    model_config = ConfigDict(extra="ignore")
