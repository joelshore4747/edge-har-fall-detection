from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from apps.api.schemas import (
    CanonicalSessionLabel,
    FeedbackTargetType,
    PredictionFeedbackRequest,
    RuntimeSessionRequest,
    RuntimeSessionResponse,
    SessionAnnotationRequest,
    SessionAnnotationSource,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSION_STORAGE_DIR = REPO_ROOT / "artifacts" / "runtime_sessions"


class RuntimePersistenceError(RuntimeError):
    """Raised when runtime persistence fails."""

    def __init__(
        self,
        message: str,
        *,
        operation: str | None = None,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.diagnostics = dict(diagnostics or {})


@dataclass(slots=True)
class StoredPayloadInfo:
    path: Path
    sha256: str
    size_bytes: int


@dataclass(slots=True)
class RuntimePersistenceResult:
    user_id: UUID
    app_session_id: UUID
    inference_id: UUID


@dataclass(slots=True)
class FeedbackPersistenceResult:
    feedback_id: UUID
    app_session_id: UUID
    inference_id: UUID | None
    target_type: FeedbackTargetType
    recorded_at: datetime


def database_url_from_env(db_url: str | None = None) -> str | None:
    if db_url is not None and db_url.strip():
        return db_url.strip()
    env_value = os.getenv("DATABASE_URL")
    if env_value is None or not env_value.strip():
        return None
    return env_value.strip()


def persistence_enabled(db_url: str | None = None) -> bool:
    return database_url_from_env(db_url) is not None


def _require_db_url(db_url: str | None = None) -> str:
    resolved = database_url_from_env(db_url)
    if resolved is None:
        raise RuntimePersistenceError("DATABASE_URL is not configured.")
    return resolved


def _resolve_storage_dir(storage_dir: Path | None = None) -> Path:
    if storage_dir is not None:
        return storage_dir.expanduser().resolve()

    env_value = os.getenv("SESSION_STORAGE_DIR")
    if env_value and env_value.strip():
        return Path(env_value).expanduser().resolve()

    return DEFAULT_SESSION_STORAGE_DIR


def _slug(value: str | None, *, fallback: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return fallback
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    cleaned = cleaned.strip("._-")
    return cleaned[:80] or fallback


def _request_context_json(req: RuntimeSessionRequest | PredictionFeedbackRequest) -> dict[str, Any]:
    request_context = getattr(req, "request_context", None)
    if request_context is None:
        return {}
    return request_context.model_dump(mode="json")


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc).isoformat()
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_compatible(item())
        except Exception:
            pass

    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return str(value)
    return coerced if math.isfinite(coerced) else None


def _jsonb(value: Any) -> Jsonb:
    return Jsonb(_json_compatible(value))


def _parse_optional_timestamptz(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _db_exception_diagnostics(exc: BaseException) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, psycopg.Error):
            if current.sqlstate:
                fields["db_sqlstate"] = current.sqlstate
            diag = getattr(current, "diag", None)
            if diag is not None:
                for attr in (
                    "severity",
                    "message_primary",
                    "message_detail",
                    "message_hint",
                    "schema_name",
                    "table_name",
                    "column_name",
                    "constraint_name",
                ):
                    value = getattr(diag, attr, None)
                    if value:
                        fields[f"db_{attr}"] = value
            break
        current = current.__cause__ or current.__context__
    return fields


def _resolve_session_lifecycle(req: RuntimeSessionRequest) -> tuple[datetime | None, datetime | None, datetime | None]:
    metadata_json = req.metadata.model_dump(mode="json")

    def _pick(*keys: str) -> datetime | None:
        for key in keys:
            parsed = _parse_optional_timestamptz(metadata_json.get(key))
            if parsed is not None:
                return parsed
        return None

    recording_started_at = _pick(
        "recording_started_at",
        "capture_started_at",
        "session_started_at",
    )
    recording_ended_at = _pick(
        "recording_ended_at",
        "capture_ended_at",
        "session_ended_at",
    )
    uploaded_at = _pick(
        "uploaded_at",
        "ingested_at",
        "saved_at",
    )
    if uploaded_at is None:
        uploaded_at = datetime.now(timezone.utc)

    return recording_started_at, recording_ended_at, uploaded_at


def _write_session_payload(
    req: RuntimeSessionRequest,
    *,
    storage_dir: Path | None = None,
) -> StoredPayloadInfo:
    resolved_dir = _resolve_storage_dir(storage_dir)
    subject_dir = _slug(req.metadata.subject_id, fallback="anonymous_user")
    session_token = _slug(req.metadata.session_id, fallback="session")
    request_token = str(req.request_context.request_id) if req.request_context is not None else str(uuid4())

    payload_dir = resolved_dir / subject_dir
    payload_dir.mkdir(parents=True, exist_ok=True)
    payload_path = payload_dir / f"{session_token}__{request_token}.json"

    payload = {
        "stored_at": datetime.now(timezone.utc).isoformat(),
        "request": req.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    payload_path.write_bytes(encoded)

    return StoredPayloadInfo(
        path=payload_path,
        sha256=hashlib.sha256(encoded).hexdigest(),
        size_bytes=len(encoded),
    )


def _remove_stored_payload(stored_payload: StoredPayloadInfo | None) -> None:
    if stored_payload is None:
        return
    try:
        stored_payload.path.unlink(missing_ok=True)
    except Exception:
        # Best-effort cleanup only; persistence errors should report the primary failure.
        return


def _sample_count(req: RuntimeSessionRequest) -> int:
    return len(req.samples)


def _has_gyro(req: RuntimeSessionRequest) -> bool:
    return any(sample.gx is not None or sample.gy is not None or sample.gz is not None for sample in req.samples)


def _duration_seconds(req: RuntimeSessionRequest, resp: RuntimeSessionResponse) -> float | None:
    duration = resp.source_summary.session_duration_seconds
    if duration is not None:
        return duration
    if len(req.samples) < 2:
        return None
    return float(req.samples[-1].timestamp - req.samples[0].timestamp)


def _request_options_json(req: RuntimeSessionRequest) -> dict[str, Any]:
    return {
        "include_har_windows": bool(req.include_har_windows),
        "include_fall_windows": bool(req.include_fall_windows),
        "include_vulnerability_windows": bool(req.include_vulnerability_windows),
        "include_combined_timeline": bool(req.include_combined_timeline),
        "include_grouped_fall_events": bool(req.include_grouped_fall_events),
        "include_point_timeline": bool(req.include_point_timeline),
        "include_timeline_events": bool(req.include_timeline_events),
        "include_transition_events": bool(req.include_transition_events),
    }


def _upsert_user(cur: psycopg.Cursor[Any], req: RuntimeSessionRequest) -> UUID:
    subject_key = (req.metadata.subject_id or "anonymous_user").strip() or "anonymous_user"
    display_name = subject_key
    cur.execute(
        """
        INSERT INTO app_users (
            subject_key,
            display_name,
            latest_device_platform,
            latest_device_model
        )
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (subject_key) DO UPDATE
        SET
            display_name = COALESCE(EXCLUDED.display_name, app_users.display_name),
            latest_device_platform = EXCLUDED.latest_device_platform,
            latest_device_model = COALESCE(EXCLUDED.latest_device_model, app_users.latest_device_model),
            updated_at = NOW(),
            last_seen_at = NOW()
        RETURNING user_id
        """,
        (
            subject_key,
            display_name,
            req.metadata.device_platform,
            req.metadata.device_model,
        ),
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimePersistenceError("Failed to upsert app user.")
    return row[0]


def _resolve_device_identity(req: RuntimeSessionRequest) -> tuple[str, str]:
    metadata_json = req.metadata.model_dump(mode="json")
    for key in (
        "device_id",
        "installation_id",
        "device_identifier",
        "installation_identifier",
    ):
        raw_value = metadata_json.get(key)
        if raw_value is None:
            continue
        normalized = str(raw_value).strip()
        if normalized:
            return normalized, "explicit"

    derived_seed = "|".join(
        [
            str(req.metadata.device_platform or "unknown"),
            str(req.metadata.device_model or ""),
            str(req.metadata.app_version or ""),
            str(req.metadata.app_build or ""),
        ]
    )
    derived_identifier = hashlib.sha256(derived_seed.encode("utf-8")).hexdigest()[:24]
    return f"derived:{derived_identifier}", "derived"


def _upsert_device(
    cur: psycopg.Cursor[Any],
    *,
    user_id: UUID,
    req: RuntimeSessionRequest,
) -> UUID:
    device_identifier, identifier_source = _resolve_device_identity(req)
    cur.execute(
        """
        INSERT INTO app_devices (
            user_id,
            device_identifier,
            identifier_source,
            device_platform,
            device_model,
            app_version,
            app_build,
            first_seen_at,
            last_seen_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
        ON CONFLICT (user_id, device_identifier) DO UPDATE
        SET
            identifier_source = EXCLUDED.identifier_source,
            device_platform = EXCLUDED.device_platform,
            device_model = COALESCE(EXCLUDED.device_model, app_devices.device_model),
            app_version = COALESCE(EXCLUDED.app_version, app_devices.app_version),
            app_build = COALESCE(EXCLUDED.app_build, app_devices.app_build),
            last_seen_at = NOW(),
            updated_at = NOW()
        RETURNING device_id
        """,
        (
            user_id,
            device_identifier,
            identifier_source,
            req.metadata.device_platform,
            req.metadata.device_model,
            req.metadata.app_version,
            req.metadata.app_build,
        ),
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimePersistenceError("Failed to upsert app device.")
    return row[0]


def _upsert_model_registry(
    cur: psycopg.Cursor[Any],
    *,
    model_type: str,
    model_name: str | None,
    model_version: str | None,
    metadata: Mapping[str, Any] | None = None,
) -> UUID | None:
    normalized_name = (model_name or "").strip()
    if not normalized_name:
        return None

    normalized_version = (model_version or "").strip()
    metadata_payload = dict(metadata or {})
    cur.execute(
        """
        INSERT INTO app_model_registry (
            model_type,
            model_name,
            model_version,
            metadata_json,
            first_seen_at,
            last_seen_at
        )
        VALUES (%s, %s, %s, %s, NOW(), NOW())
        ON CONFLICT (model_type, model_name, model_version) DO UPDATE
        SET
            metadata_json = CASE
                WHEN EXCLUDED.metadata_json <> '{}'::jsonb THEN EXCLUDED.metadata_json
                ELSE app_model_registry.metadata_json
            END,
            last_seen_at = NOW(),
            updated_at = NOW()
        RETURNING model_registry_id
        """,
        (
            model_type,
            normalized_name,
            normalized_version,
            _jsonb(metadata_payload),
        ),
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimePersistenceError("Failed to upsert app model registry row.")
    return row[0]


def _upsert_session(
    cur: psycopg.Cursor[Any],
    *,
    user_id: UUID,
    device_id: UUID | None,
    req: RuntimeSessionRequest,
    resp: RuntimeSessionResponse,
    stored_payload: StoredPayloadInfo,
) -> UUID:
    request_context = _request_context_json(req)
    metadata_json = req.metadata.model_dump(mode="json")
    request_id = req.request_context.request_id if req.request_context is not None else None
    trace_id = req.request_context.trace_id if req.request_context is not None else None
    client_version = req.request_context.client_version if req.request_context is not None else None
    recording_started_at, recording_ended_at, uploaded_at = _resolve_session_lifecycle(req)

    cur.execute(
        """
        INSERT INTO app_sessions (
            user_id,
            device_id,
            client_session_id,
            request_id,
            trace_id,
            client_version,
            dataset_name,
            source_type,
            task_type,
            placement_declared,
            device_platform,
            device_model,
            app_version,
            app_build,
            recording_mode,
            runtime_mode,
            recording_started_at,
            recording_ended_at,
            uploaded_at,
            sampling_rate_hz,
            sample_count,
            has_gyro,
            duration_seconds,
            notes,
            metadata_json,
            request_context_json,
            raw_storage_uri,
            raw_storage_format,
            raw_payload_sha256,
            raw_payload_bytes
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (user_id, client_session_id) DO UPDATE
        SET
            device_id = COALESCE(EXCLUDED.device_id, app_sessions.device_id),
            request_id = EXCLUDED.request_id,
            trace_id = EXCLUDED.trace_id,
            client_version = EXCLUDED.client_version,
            dataset_name = EXCLUDED.dataset_name,
            source_type = EXCLUDED.source_type,
            task_type = EXCLUDED.task_type,
            placement_declared = EXCLUDED.placement_declared,
            device_platform = EXCLUDED.device_platform,
            device_model = COALESCE(EXCLUDED.device_model, app_sessions.device_model),
            app_version = COALESCE(EXCLUDED.app_version, app_sessions.app_version),
            app_build = COALESCE(EXCLUDED.app_build, app_sessions.app_build),
            recording_mode = EXCLUDED.recording_mode,
            runtime_mode = EXCLUDED.runtime_mode,
            recording_started_at = COALESCE(EXCLUDED.recording_started_at, app_sessions.recording_started_at),
            recording_ended_at = COALESCE(EXCLUDED.recording_ended_at, app_sessions.recording_ended_at),
            uploaded_at = EXCLUDED.uploaded_at,
            sampling_rate_hz = EXCLUDED.sampling_rate_hz,
            sample_count = EXCLUDED.sample_count,
            has_gyro = EXCLUDED.has_gyro,
            duration_seconds = EXCLUDED.duration_seconds,
            notes = COALESCE(EXCLUDED.notes, app_sessions.notes),
            metadata_json = EXCLUDED.metadata_json,
            request_context_json = EXCLUDED.request_context_json,
            raw_storage_uri = EXCLUDED.raw_storage_uri,
            raw_storage_format = EXCLUDED.raw_storage_format,
            raw_payload_sha256 = EXCLUDED.raw_payload_sha256,
            raw_payload_bytes = EXCLUDED.raw_payload_bytes,
            updated_at = NOW()
        RETURNING app_session_id
        """,
        (
            user_id,
            device_id,
            req.metadata.session_id,
            request_id,
            trace_id,
            client_version,
            req.metadata.dataset_name,
            req.metadata.source_type.value,
            req.metadata.task_type.value,
            req.metadata.placement.value,
            req.metadata.device_platform,
            req.metadata.device_model,
            req.metadata.app_version,
            req.metadata.app_build,
            req.metadata.recording_mode.value,
            req.metadata.runtime_mode.value,
            recording_started_at,
            recording_ended_at,
            uploaded_at,
            req.metadata.sampling_rate_hz,
            _sample_count(req),
            _has_gyro(req),
            _duration_seconds(req, resp),
            req.metadata.notes,
            _jsonb(metadata_json),
            _jsonb(request_context),
            str(stored_payload.path),
            "json",
            stored_payload.sha256,
            stored_payload.size_bytes,
        ),
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimePersistenceError("Failed to upsert app session.")
    return row[0]


def _lookup_existing_inference_id(
    cur: psycopg.Cursor[Any],
    *,
    app_session_id: UUID,
    request_id: UUID | None,
) -> UUID | None:
    if request_id is None:
        return None

    cur.execute(
        """
        SELECT app_session_id, inference_id
        FROM app_session_inferences
        WHERE request_id = %s
        """,
        (request_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None

    existing_session_id, inference_id = row
    if existing_session_id != app_session_id:
        raise RuntimePersistenceError(
            "request_id is already associated with a different app_session_id."
        )
    return inference_id


def _insert_inference(
    cur: psycopg.Cursor[Any],
    *,
    app_session_id: UUID,
    req: RuntimeSessionRequest,
    resp: RuntimeSessionResponse,
) -> tuple[UUID, bool]:
    request_id = resp.request_id or (req.request_context.request_id if req.request_context is not None else None)
    session_narrative_summary = (
        resp.session_narrative_summary.model_dump(mode="json")
        if resp.session_narrative_summary is not None
        else None
    )
    existing_inference_id = _lookup_existing_inference_id(
        cur,
        app_session_id=app_session_id,
        request_id=request_id,
    )
    if existing_inference_id is not None:
        return existing_inference_id, False

    model_registry_metadata = {"api_version": resp.model_info.api_version}
    har_model_registry_id = _upsert_model_registry(
        cur,
        model_type="har",
        model_name=resp.model_info.har_model_name,
        model_version=resp.model_info.har_model_version,
        metadata=model_registry_metadata,
    )
    fall_model_registry_id = _upsert_model_registry(
        cur,
        model_type="fall",
        model_name=resp.model_info.fall_model_name,
        model_version=resp.model_info.fall_model_version,
        metadata=model_registry_metadata,
    )

    inference_id = uuid4()
    started_at = datetime.now(timezone.utc)
    completed_at = started_at
    alert_summary_payload = resp.alert_summary.model_dump(mode="json")
    alert_summary_payload["vulnerability_summary"] = resp.vulnerability_summary.model_dump(mode="json")

    cur.execute(
        """
        INSERT INTO app_session_inferences (
            inference_id,
            app_session_id,
            request_id,
            har_model_registry_id,
            fall_model_registry_id,
            api_version,
            har_model_name,
            har_model_version,
            fall_model_name,
            fall_model_version,
            status,
            error_message,
            started_at,
            completed_at,
            warning_level,
            likely_fall_detected,
            top_har_label,
            top_har_fraction,
            grouped_fall_event_count,
            top_fall_probability,
            timeline_event_count,
            transition_event_count,
            request_options_json,
            source_summary_json,
            placement_summary_json,
            har_summary_json,
            fall_summary_json,
            alert_summary_json,
            debug_summary_json,
            model_info_json,
            session_narrative_summary_json,
            narrative_summary_json
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """,
        (
            inference_id,
            app_session_id,
            request_id,
            har_model_registry_id,
            fall_model_registry_id,
            resp.model_info.api_version,
            resp.model_info.har_model_name,
            resp.model_info.har_model_version,
            resp.model_info.fall_model_name,
            resp.model_info.fall_model_version,
            "completed",
            None,
            started_at,
            completed_at,
            resp.alert_summary.warning_level.value,
            resp.alert_summary.likely_fall_detected,
            resp.alert_summary.top_har_label,
            resp.alert_summary.top_har_fraction,
            resp.alert_summary.grouped_fall_event_count,
            resp.alert_summary.top_fall_probability,
            len(resp.timeline_events),
            len(resp.transition_events),
            _jsonb(_request_options_json(req)),
            _jsonb(resp.source_summary.model_dump(mode="json")),
            _jsonb(resp.placement_summary.model_dump(mode="json")),
            _jsonb(resp.har_summary.model_dump(mode="json")),
            _jsonb(resp.fall_summary.model_dump(mode="json")),
            _jsonb(alert_summary_payload),
            _jsonb(resp.debug_summary.model_dump(mode="json")),
            _jsonb(resp.model_info.model_dump(mode="json")),
            _jsonb(session_narrative_summary) if session_narrative_summary is not None else None,
            _jsonb(resp.narrative_summary),
        ),
    )
    return inference_id, True


def _insert_grouped_fall_events(
    cur: psycopg.Cursor[Any],
    *,
    app_session_id: UUID,
    inference_id: UUID,
    resp: RuntimeSessionResponse,
) -> None:
    rows = [
        (
            app_session_id,
            inference_id,
            event.event_id,
            event.event_start_ts,
            event.event_end_ts,
            event.event_duration_seconds,
            event.n_positive_windows,
            event.peak_probability,
            event.mean_probability,
            event.median_probability,
        )
        for event in resp.grouped_fall_events
    ]
    if not rows:
        return

    cur.executemany(
        """
        INSERT INTO app_grouped_fall_events (
            app_session_id,
            inference_id,
            event_key,
            event_start_ts,
            event_end_ts,
            event_duration_seconds,
            positive_window_count,
            peak_probability,
            mean_probability,
            median_probability
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (inference_id, event_key) DO NOTHING
        """,
        rows,
    )


def _insert_timeline_events(
    cur: psycopg.Cursor[Any],
    *,
    app_session_id: UUID,
    inference_id: UUID,
    resp: RuntimeSessionResponse,
) -> None:
    rows = [
        (
            app_session_id,
            inference_id,
            event.event_id,
            event.start_ts,
            event.end_ts,
            event.duration_seconds,
            event.midpoint_ts,
            event.point_count,
            event.activity_label,
            event.placement_label,
            event.activity_confidence_mean,
            event.placement_confidence_mean,
            event.fall_probability_peak,
            event.fall_probability_mean,
            event.likely_fall,
            event.event_kind,
            _jsonb(list(event.related_grouped_fall_event_ids)),
            event.description,
        )
        for event in resp.timeline_events
    ]
    if not rows:
        return

    cur.executemany(
        """
        INSERT INTO app_timeline_events (
            app_session_id,
            inference_id,
            event_key,
            start_ts,
            end_ts,
            duration_seconds,
            midpoint_ts,
            point_count,
            activity_label,
            placement_label,
            activity_confidence_mean,
            placement_confidence_mean,
            fall_probability_peak,
            fall_probability_mean,
            likely_fall,
            event_kind,
            related_grouped_event_ids_json,
            description
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (inference_id, event_key) DO NOTHING
        """,
        rows,
    )


def _insert_transition_events(
    cur: psycopg.Cursor[Any],
    *,
    app_session_id: UUID,
    inference_id: UUID,
    resp: RuntimeSessionResponse,
) -> None:
    rows = [
        (
            app_session_id,
            inference_id,
            transition.transition_id,
            transition.transition_ts,
            transition.from_event_id,
            transition.to_event_id,
            transition.transition_kind,
            transition.from_activity_label,
            transition.to_activity_label,
            transition.from_placement_label,
            transition.to_placement_label,
            transition.description,
        )
        for transition in resp.transition_events
    ]
    if not rows:
        return

    cur.executemany(
        """
        INSERT INTO app_transition_events (
            app_session_id,
            inference_id,
            transition_key,
            transition_ts,
            from_event_key,
            to_event_key,
            transition_kind,
            from_activity_label,
            to_activity_label,
            from_placement_label,
            to_placement_label,
            description
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (inference_id, transition_key) DO NOTHING
        """,
        rows,
    )


def persist_runtime_session(
    req: RuntimeSessionRequest,
    resp: RuntimeSessionResponse,
    *,
    db_url: str | None = None,
    storage_dir: Path | None = None,
) -> RuntimePersistenceResult:
    resolved_db_url = _require_db_url(db_url)
    stored_payload: StoredPayloadInfo | None = None

    try:
        stored_payload = _write_session_payload(req, storage_dir=storage_dir)
        with psycopg.connect(resolved_db_url) as conn:
            with conn.cursor() as cur:
                user_id = _upsert_user(cur, req)
                device_id = _upsert_device(
                    cur,
                    user_id=user_id,
                    req=req,
                )
                app_session_id = _upsert_session(
                    cur,
                    user_id=user_id,
                    device_id=device_id,
                    req=req,
                    resp=resp,
                    stored_payload=stored_payload,
                )
                inference_id, inference_created = _insert_inference(
                    cur,
                    app_session_id=app_session_id,
                    req=req,
                    resp=resp,
                )
                if inference_created:
                    _insert_grouped_fall_events(
                        cur,
                        app_session_id=app_session_id,
                        inference_id=inference_id,
                        resp=resp,
                    )
                    _insert_timeline_events(
                        cur,
                        app_session_id=app_session_id,
                        inference_id=inference_id,
                        resp=resp,
                    )
                    _insert_transition_events(
                        cur,
                        app_session_id=app_session_id,
                        inference_id=inference_id,
                        resp=resp,
                    )
            conn.commit()
    except Exception as exc:  # noqa: BLE001
        _remove_stored_payload(stored_payload)
        raise RuntimePersistenceError(
            f"Failed to persist runtime session: {exc}",
            operation="persist_runtime_session",
            diagnostics=_db_exception_diagnostics(exc),
        ) from exc

    return RuntimePersistenceResult(
        user_id=user_id,
        app_session_id=app_session_id,
        inference_id=inference_id,
    )


def _lookup_feedback_session(
    cur: psycopg.Cursor[Any],
    req: PredictionFeedbackRequest,
    *,
    owner_user_id: UUID | None = None,
) -> tuple[UUID, UUID | None]:
    session_row: tuple[Any, ...] | None = None
    resolved_inference_id: UUID | None = getattr(req, "persisted_inference_id", None)

    persisted_session_id = getattr(req, "persisted_session_id", None)
    subject_key = getattr(req, "subject_id", None)

    if persisted_session_id is not None and resolved_inference_id is not None:
        cur.execute(
            """
            SELECT app_session_id, inference_id
            FROM app_session_inferences
            WHERE app_session_id = %s AND inference_id = %s
            """,
            (persisted_session_id, resolved_inference_id),
        )
        session_row = cur.fetchone()
        if session_row is None:
            raise RuntimePersistenceError(
                "persisted_session_id and persisted_inference_id do not refer to the same inference row."
            )
    elif persisted_session_id is not None:
        cur.execute(
            "SELECT app_session_id FROM app_sessions WHERE app_session_id = %s",
            (persisted_session_id,),
        )
        session_row = cur.fetchone()
    elif resolved_inference_id is not None:
        cur.execute(
            """
            SELECT app_session_id
            FROM app_session_inferences
            WHERE inference_id = %s
            """,
            (resolved_inference_id,),
        )
        session_row = cur.fetchone()
    elif subject_key:
        cur.execute(
            """
            SELECT s.app_session_id
            FROM app_sessions AS s
            JOIN app_users AS u ON u.user_id = s.user_id
            WHERE s.client_session_id = %s AND u.subject_key = %s
            ORDER BY s.created_at DESC
            LIMIT 1
            """,
            (req.session_id, subject_key),
        )
        session_row = cur.fetchone()
    else:
        cur.execute(
            """
            SELECT app_session_id
            FROM app_sessions
            WHERE client_session_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (req.session_id,),
        )
        session_row = cur.fetchone()

    if session_row is None:
        raise RuntimePersistenceError(
            f"No persisted session was found for client session_id={req.session_id!r}."
        )

    app_session_id = session_row[0]
    if owner_user_id is not None:
        cur.execute(
            """
            SELECT 1
            FROM app_sessions
            WHERE app_session_id = %s AND user_id = %s
            """,
            (app_session_id, owner_user_id),
        )
        if cur.fetchone() is None:
            raise RuntimePersistenceError(
                "Authenticated user does not own the requested session."
            )

    if resolved_inference_id is not None:
        return app_session_id, resolved_inference_id

    cur.execute(
        """
        SELECT inference_id
        FROM app_session_inferences
        WHERE app_session_id = %s
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (app_session_id,),
    )
    inference_row = cur.fetchone()
    return app_session_id, inference_row[0] if inference_row is not None else None


def _derive_feedback_target_type(req: PredictionFeedbackRequest) -> FeedbackTargetType:
    if req.target_type is not None:
        return req.target_type
    if req.window_id:
        return FeedbackTargetType.window
    if req.event_id:
        return FeedbackTargetType.timeline_event
    return FeedbackTargetType.session


def _lookup_existing_feedback_by_request(
    cur: psycopg.Cursor[Any],
    *,
    request_id: UUID | None,
) -> tuple[UUID, UUID, UUID | None, str, datetime] | None:
    if request_id is None:
        return None

    cur.execute(
        """
        SELECT feedback_id, app_session_id, inference_id, target_type, recorded_at
        FROM app_feedback
        WHERE request_id = %s
        """,
        (request_id,),
    )
    return cur.fetchone()


def persist_feedback(
    req: PredictionFeedbackRequest,
    *,
    db_url: str | None = None,
    owner_user_id: UUID | None = None,
) -> FeedbackPersistenceResult:
    resolved_db_url = _require_db_url(db_url)
    feedback_id = uuid4()
    recorded_at = datetime.now(timezone.utc)
    target_type = _derive_feedback_target_type(req)

    try:
        with psycopg.connect(resolved_db_url) as conn:
            with conn.cursor() as cur:
                request_id = req.request_context.request_id if req.request_context is not None else None
                existing_feedback = _lookup_existing_feedback_by_request(cur, request_id=request_id)
                if existing_feedback is not None:
                    (
                        existing_feedback_id,
                        existing_session_id,
                        existing_inference_id,
                        existing_target_type,
                        existing_recorded_at,
                    ) = existing_feedback
                    if owner_user_id is not None:
                        cur.execute(
                            """
                            SELECT 1
                            FROM app_sessions
                            WHERE app_session_id = %s AND user_id = %s
                            """,
                            (existing_session_id, owner_user_id),
                        )
                        if cur.fetchone() is None:
                            raise RuntimePersistenceError(
                                "Authenticated user does not own the existing feedback session."
                            )
                    return FeedbackPersistenceResult(
                        feedback_id=existing_feedback_id,
                        app_session_id=existing_session_id,
                        inference_id=existing_inference_id,
                        target_type=FeedbackTargetType(existing_target_type),
                        recorded_at=existing_recorded_at,
                    )

                app_session_id, inference_id = _lookup_feedback_session(
                    cur,
                    req,
                    owner_user_id=owner_user_id,
                )
                subject_key = getattr(req, "subject_id", None)

                cur.execute(
                    """
                    INSERT INTO app_feedback (
                        feedback_id,
                        app_session_id,
                        inference_id,
                        target_type,
                        target_event_key,
                        window_id,
                        feedback_type,
                        corrected_label,
                        reviewer_identifier,
                        subject_key,
                        notes,
                        request_id,
                        recorded_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        feedback_id,
                        app_session_id,
                        inference_id,
                        target_type.value,
                        req.event_id,
                        req.window_id,
                        req.user_feedback.value,
                        req.corrected_label,
                        req.reviewer_id,
                        subject_key,
                        req.notes,
                        request_id,
                        recorded_at,
                    ),
                )
            conn.commit()
    except Exception as exc:  # noqa: BLE001
        raise RuntimePersistenceError(
            f"Failed to persist feedback: {exc}",
            operation="persist_feedback",
            diagnostics=_db_exception_diagnostics(exc),
        ) from exc

    return FeedbackPersistenceResult(
        feedback_id=feedback_id,
        app_session_id=app_session_id,
        inference_id=inference_id,
        target_type=target_type,
        recorded_at=recorded_at,
    )


def _session_record_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "app_session_id": row["app_session_id"],
        "user_id": row["user_id"],
        "device_id": row.get("device_id"),
        "subject_id": row["subject_key"],
        "client_session_id": row["client_session_id"],
        "request_id": row["request_id"],
        "trace_id": row["trace_id"],
        "client_version": row["client_version"],
        "dataset_name": row["dataset_name"],
        "source_type": row["source_type"],
        "task_type": row["task_type"],
        "placement_declared": row["placement_declared"],
        "device_platform": row["device_platform"],
        "device_model": row["device_model"],
        "app_version": row["app_version"],
        "app_build": row["app_build"],
        "recording_mode": row["recording_mode"],
        "runtime_mode": row["runtime_mode"],
        "recording_started_at": row["recording_started_at"],
        "recording_ended_at": row["recording_ended_at"],
        "uploaded_at": row["uploaded_at"],
        "sampling_rate_hz": row["sampling_rate_hz"],
        "sample_count": row["sample_count"],
        "has_gyro": row["has_gyro"],
        "duration_seconds": row["duration_seconds"],
        "session_name": row.get("session_name"),
        "activity_label": row.get("activity_label"),
        "notes": row["notes"],
        "raw_storage_uri": row["raw_storage_uri"],
        "raw_storage_format": row["raw_storage_format"],
        "raw_payload_sha256": row["raw_payload_sha256"],
        "raw_payload_bytes": row["raw_payload_bytes"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _grouped_fall_event_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_id": row["event_key"],
        "event_start_ts": row["event_start_ts"],
        "event_end_ts": row["event_end_ts"],
        "event_duration_seconds": row["event_duration_seconds"],
        "n_positive_windows": row["positive_window_count"],
        "peak_probability": row["peak_probability"],
        "mean_probability": row["mean_probability"],
        "median_probability": row["median_probability"],
    }


def _timeline_event_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_id": row["event_key"],
        "start_ts": row["start_ts"],
        "end_ts": row["end_ts"],
        "duration_seconds": row["duration_seconds"],
        "midpoint_ts": row["midpoint_ts"],
        "point_count": row["point_count"],
        "activity_label": row["activity_label"],
        "placement_label": row["placement_label"],
        "activity_confidence_mean": row["activity_confidence_mean"],
        "placement_confidence_mean": row["placement_confidence_mean"],
        "fall_probability_peak": row["fall_probability_peak"],
        "fall_probability_mean": row["fall_probability_mean"],
        "likely_fall": row["likely_fall"],
        "event_kind": row["event_kind"],
        "related_grouped_fall_event_ids": list(row["related_grouped_event_ids_json"] or []),
        "description": row["description"],
    }


def _transition_event_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "transition_id": row["transition_key"],
        "transition_ts": row["transition_ts"],
        "from_event_id": row["from_event_key"],
        "to_event_id": row["to_event_key"],
        "transition_kind": row["transition_kind"],
        "from_activity_label": row["from_activity_label"],
        "to_activity_label": row["to_activity_label"],
        "from_placement_label": row["from_placement_label"],
        "to_placement_label": row["to_placement_label"],
        "description": row["description"],
    }


def _feedback_record_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "feedback_id": row["feedback_id"],
        "app_session_id": row["app_session_id"],
        "inference_id": row["inference_id"],
        "target_type": row["target_type"],
        "target_event_key": row["target_event_key"],
        "window_id": row["window_id"],
        "feedback_type": row["feedback_type"],
        "corrected_label": row["corrected_label"],
        "reviewer_identifier": row["reviewer_identifier"],
        "subject_key": row["subject_key"],
        "notes": row["notes"],
        "request_id": row["request_id"],
        "recorded_at": row["recorded_at"],
    }


def _session_annotation_record_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "annotation_id": row["annotation_id"],
        "app_session_id": row["app_session_id"],
        "label": row["annotation_label"],
        "source": row["source"],
        "reviewer_identifier": row["reviewer_identifier"],
        "auth_account_id": row["auth_account_id"],
        "created_by_username": row["created_by_username"],
        "request_id": row["request_id"],
        "notes": row["notes"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _runtime_response_from_rows(
    session_row: Mapping[str, Any],
    inference_row: Mapping[str, Any],
    *,
    grouped_rows: list[Mapping[str, Any]],
    timeline_rows: list[Mapping[str, Any]],
    transition_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    alert_summary = dict(inference_row["alert_summary_json"] or {})
    vulnerability_summary = {}
    raw_vulnerability_summary = alert_summary.get("vulnerability_summary")
    if isinstance(raw_vulnerability_summary, Mapping):
        vulnerability_summary = dict(raw_vulnerability_summary)
    alert_summary.pop("vulnerability_summary", None)

    return {
        "request_id": inference_row["request_id"],
        "session_id": session_row["client_session_id"],
        "persisted_user_id": session_row["user_id"],
        "persisted_session_id": session_row["app_session_id"],
        "persisted_inference_id": inference_row["inference_id"],
        "source_summary": dict(inference_row["source_summary_json"] or {}),
        "placement_summary": dict(inference_row["placement_summary_json"] or {}),
        "har_summary": dict(inference_row["har_summary_json"] or {}),
        "fall_summary": dict(inference_row["fall_summary_json"] or {}),
        "vulnerability_summary": vulnerability_summary,
        "alert_summary": alert_summary,
        "debug_summary": dict(inference_row["debug_summary_json"] or {}),
        "model_info": dict(inference_row["model_info_json"] or {}),
        "grouped_fall_events": [_grouped_fall_event_from_row(row) for row in grouped_rows],
        "har_windows": [],
        "fall_windows": [],
        "vulnerability_windows": [],
        "combined_timeline": [],
        "point_timeline": [],
        "timeline_events": [_timeline_event_from_row(row) for row in timeline_rows],
        "transition_events": [_transition_event_from_row(row) for row in transition_rows],
        "session_narrative_summary": inference_row["session_narrative_summary_json"],
        "narrative_summary": dict(inference_row["narrative_summary_json"] or {}),
    }


def _runtime_summary_from_rows(
    session_row: Mapping[str, Any],
    inference_row: Mapping[str, Any],
) -> dict[str, Any]:
    alert_summary = dict(inference_row["alert_summary_json"] or {})
    vulnerability_summary = {}
    raw_vulnerability_summary = alert_summary.get("vulnerability_summary")
    if isinstance(raw_vulnerability_summary, Mapping):
        vulnerability_summary = dict(raw_vulnerability_summary)
    alert_summary.pop("vulnerability_summary", None)

    return {
        "request_id": inference_row["request_id"],
        "session_id": session_row["client_session_id"],
        "persisted_user_id": session_row["user_id"],
        "persisted_session_id": session_row["app_session_id"],
        "persisted_inference_id": inference_row["inference_id"],
        "source_summary": dict(inference_row["source_summary_json"] or {}),
        "placement_summary": dict(inference_row["placement_summary_json"] or {}),
        "har_summary": dict(inference_row["har_summary_json"] or {}),
        "fall_summary": dict(inference_row["fall_summary_json"] or {}),
        "vulnerability_summary": vulnerability_summary,
        "alert_summary": alert_summary,
        "model_info": dict(inference_row["model_info_json"] or {}),
        "session_narrative_summary": inference_row["session_narrative_summary_json"],
        "narrative_summary": dict(inference_row["narrative_summary_json"] or {}),
    }


def _session_list_item_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "session": _session_record_from_row(row),
        "latest_inference_id": row["latest_inference_id"],
        "latest_inference_request_id": row["latest_inference_request_id"],
        "latest_inference_created_at": row["latest_inference_created_at"],
        "latest_status": row["latest_status"],
        "latest_warning_level": row["latest_warning_level"],
        "latest_likely_fall_detected": row["latest_likely_fall_detected"],
        "latest_top_har_label": row["latest_top_har_label"],
        "latest_top_fall_probability": row["latest_top_fall_probability"],
        "latest_grouped_fall_event_count": row["latest_grouped_fall_event_count"],
        "latest_annotation_label": row.get("latest_annotation_label"),
        "latest_annotation_source": row.get("latest_annotation_source"),
        "latest_annotation_created_at": row.get("latest_annotation_created_at"),
    }


def _admin_overview_session_record_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "app_session_id": row["app_session_id"],
        "subject_id": row["subject_key"],
        "client_session_id": row["client_session_id"],
        "device_platform": row["device_platform"],
        "device_model": row["device_model"],
        "uploaded_at": row["uploaded_at"],
        "duration_seconds": row["duration_seconds"],
        "session_name": row.get("session_name"),
        "activity_label": row.get("activity_label"),
    }


def _admin_overview_session_item_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "session": _admin_overview_session_record_from_row(row),
        "latest_status": row["latest_status"],
        "latest_warning_level": row["latest_warning_level"],
        "latest_likely_fall_detected": row["latest_likely_fall_detected"],
        "latest_top_har_label": row["latest_top_har_label"],
        "latest_top_fall_probability": row["latest_top_fall_probability"],
        "latest_grouped_fall_event_count": row["latest_grouped_fall_event_count"],
    }


def _admin_session_list_record_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "app_session_id": row["app_session_id"],
        "subject_id": row["subject_key"],
        "client_session_id": row["client_session_id"],
        "device_platform": row["device_platform"],
        "device_model": row["device_model"],
        "uploaded_at": row["uploaded_at"],
        "duration_seconds": row["duration_seconds"],
        "session_name": row.get("session_name"),
        "activity_label": row.get("activity_label"),
        "notes": row["notes"],
    }


def _admin_session_list_item_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "session": _admin_session_list_record_from_row(row),
        "latest_status": row["latest_status"],
        "latest_warning_level": row["latest_warning_level"],
        "latest_likely_fall_detected": row["latest_likely_fall_detected"],
        "latest_top_har_label": row["latest_top_har_label"],
        "latest_top_fall_probability": row["latest_top_fall_probability"],
        "latest_grouped_fall_event_count": row["latest_grouped_fall_event_count"],
        "latest_annotation_label": row.get("latest_annotation_label"),
    }


def _latest_inference_join_sql(session_alias: str = "s") -> str:
    return f"""
    LEFT JOIN LATERAL (
        SELECT
            i.inference_id,
            i.request_id,
            i.created_at,
            i.status,
            i.warning_level,
            i.likely_fall_detected,
            i.top_har_label,
            i.top_fall_probability,
            i.grouped_fall_event_count
        FROM app_session_inferences AS i
        WHERE i.app_session_id = {session_alias}.app_session_id
        ORDER BY i.created_at DESC
        LIMIT 1
    ) AS li ON TRUE
    """


def _latest_annotation_join_sql(session_alias: str = "s") -> str:
    return f"""
    LEFT JOIN LATERAL (
        SELECT
            a.annotation_label,
            a.source,
            a.created_at
        FROM app_session_annotations AS a
        WHERE a.app_session_id = {session_alias}.app_session_id
        ORDER BY a.created_at DESC, a.annotation_id DESC
        LIMIT 1
    ) AS la ON TRUE
    """


def _latest_inference_snapshot_cte_sql(alias: str = "latest_inferences") -> str:
    return f"""
    {alias} AS (
        SELECT DISTINCT ON (i.app_session_id)
            i.app_session_id,
            i.warning_level,
            i.top_har_label
        FROM app_session_inferences AS i
        ORDER BY i.app_session_id, i.created_at DESC
    )
    """


def _session_is_accessible(
    cur: psycopg.Cursor[Any],
    *,
    app_session_id: UUID,
    owner_user_id: UUID | None = None,
) -> bool:
    cur.execute(
        f"""
        SELECT 1
        FROM app_sessions
        WHERE app_session_id = %s
        {"AND user_id = %s" if owner_user_id is not None else ""}
        """,
        (app_session_id, owner_user_id)
        if owner_user_id is not None
        else (app_session_id,),
    )
    return cur.fetchone() is not None


def _fetch_session_annotation_rows(
    cur: psycopg.Cursor[Any],
    *,
    app_session_id: UUID,
) -> list[Mapping[str, Any]]:
    cur.execute(
        """
        SELECT
            a.annotation_id,
            a.app_session_id,
            a.annotation_label,
            a.source,
            a.reviewer_identifier,
            a.auth_account_id,
            aa.login_username AS created_by_username,
            a.request_id,
            a.notes,
            a.created_at,
            a.updated_at
        FROM app_session_annotations AS a
        LEFT JOIN app_auth_accounts AS aa ON aa.auth_account_id = a.auth_account_id
        WHERE a.app_session_id = %s
        ORDER BY a.created_at DESC, a.annotation_id DESC
        """,
        (app_session_id,),
    )
    return list(cur.fetchall())


def _fetch_latest_session_annotation_row(
    cur: psycopg.Cursor[Any],
    *,
    app_session_id: UUID,
) -> Mapping[str, Any] | None:
    cur.execute(
        """
        SELECT
            a.annotation_id,
            a.app_session_id,
            a.annotation_label,
            a.source,
            a.reviewer_identifier,
            a.auth_account_id,
            aa.login_username AS created_by_username,
            a.request_id,
            a.notes,
            a.created_at,
            a.updated_at
        FROM app_session_annotations AS a
        LEFT JOIN app_auth_accounts AS aa ON aa.auth_account_id = a.auth_account_id
        WHERE a.app_session_id = %s
        ORDER BY a.created_at DESC, a.annotation_id DESC
        LIMIT 1
        """,
        (app_session_id,),
    )
    return cur.fetchone()


def _count_session_annotation_rows(
    cur: psycopg.Cursor[Any],
    *,
    app_session_id: UUID,
) -> int:
    cur.execute(
        """
        SELECT COUNT(*) AS total_count
        FROM app_session_annotations
        WHERE app_session_id = %s
        """,
        (app_session_id,),
    )
    row = cur.fetchone()
    return int(row["total_count"] if row is not None else 0)


def _fetch_latest_feedback_row(
    cur: psycopg.Cursor[Any],
    *,
    app_session_id: UUID,
) -> Mapping[str, Any] | None:
    cur.execute(
        """
        SELECT
            feedback_id,
            app_session_id,
            inference_id,
            target_type,
            target_event_key,
            window_id,
            feedback_type,
            corrected_label,
            reviewer_identifier,
            subject_key,
            notes,
            request_id,
            recorded_at
        FROM app_feedback
        WHERE app_session_id = %s
        ORDER BY recorded_at DESC, feedback_id DESC
        LIMIT 1
        """,
        (app_session_id,),
    )
    return cur.fetchone()


def _count_session_feedback_rows(
    cur: psycopg.Cursor[Any],
    *,
    app_session_id: UUID,
) -> int:
    cur.execute(
        """
        SELECT COUNT(*) AS total_count
        FROM app_feedback
        WHERE app_session_id = %s
        """,
        (app_session_id,),
    )
    row = cur.fetchone()
    return int(row["total_count"] if row is not None else 0)


def _fetch_session_annotation_by_id(
    cur: psycopg.Cursor[Any],
    *,
    annotation_id: UUID,
) -> Mapping[str, Any] | None:
    cur.execute(
        """
        SELECT
            a.annotation_id,
            a.app_session_id,
            a.annotation_label,
            a.source,
            a.reviewer_identifier,
            a.auth_account_id,
            aa.login_username AS created_by_username,
            a.request_id,
            a.notes,
            a.created_at,
            a.updated_at
        FROM app_session_annotations AS a
        LEFT JOIN app_auth_accounts AS aa ON aa.auth_account_id = a.auth_account_id
        WHERE a.annotation_id = %s
        """,
        (annotation_id,),
    )
    return cur.fetchone()


def _lookup_existing_annotation_by_request(
    cur: psycopg.Cursor[Any],
    *,
    request_id: UUID | None,
) -> Mapping[str, Any] | None:
    if request_id is None:
        return None
    cur.execute(
        """
        SELECT annotation_id, app_session_id
        FROM app_session_annotations
        WHERE request_id = %s
        """,
        (request_id,),
    )
    return cur.fetchone()


def _fill_daily_buckets(
    rows: list[Mapping[str, Any]],
    *,
    days: int,
    value_key: str = "value",
) -> list[dict[str, Any]]:
    start_day = date.today() - timedelta(days=days - 1)
    values = {
        row["bucket_day"].isoformat(): int(row[value_key] or 0)
        for row in rows
    }
    return [
        {
            "label": (start_day + timedelta(days=offset)).isoformat(),
            "value": values.get((start_day + timedelta(days=offset)).isoformat(), 0),
        }
        for offset in range(days)
    ]


def get_admin_overview(
    *,
    db_url: str | None = None,
    recent_session_limit: int = 8,
    chart_days: int = 7,
) -> dict[str, Any]:
    resolved_db_url = _require_db_url(db_url)

    try:
        with psycopg.connect(resolved_db_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM app_users) AS users,
                        (SELECT COUNT(*) FROM app_sessions) AS sessions,
                        (SELECT COUNT(*) FROM app_session_inferences) AS inferences,
                        (SELECT COUNT(*) FROM app_grouped_fall_events) AS grouped_fall_events,
                        (
                            SELECT COUNT(*)
                            FROM app_sessions
                            WHERE created_at >= NOW() - INTERVAL '7 days'
                        ) AS sessions_last_7_days,
                        (
                            SELECT COUNT(DISTINCT app_session_id)
                            FROM app_session_inferences
                            WHERE likely_fall_detected = TRUE
                        ) AS sessions_with_likely_fall
                    """
                )
                totals_row = cur.fetchone()

                cur.execute(
                    """
                    SELECT
                        date_trunc('day', created_at)::date AS bucket_day,
                        COUNT(*)::int AS value
                    FROM app_sessions
                    WHERE created_at >= CURRENT_DATE - (%s::int - 1) * INTERVAL '1 day'
                    GROUP BY bucket_day
                    ORDER BY bucket_day ASC
                    """,
                    (chart_days,),
                )
                sessions_by_day_rows = list(cur.fetchall())

                cur.execute(
                    """
                    SELECT
                        date_trunc('day', created_at)::date AS bucket_day,
                        COUNT(*)::int AS value
                    FROM app_grouped_fall_events
                    WHERE created_at >= CURRENT_DATE - (%s::int - 1) * INTERVAL '1 day'
                    GROUP BY bucket_day
                    ORDER BY bucket_day ASC
                    """,
                    (chart_days,),
                )
                fall_events_by_day_rows = list(cur.fetchall())

                cur.execute(
                    f"""
                    WITH
                    {_latest_inference_snapshot_cte_sql()},
                    warning_distribution AS (
                        SELECT
                            COALESCE(li.warning_level, 'unknown') AS label,
                            COUNT(*)::int AS value
                        FROM app_sessions AS s
                        LEFT JOIN latest_inferences AS li ON li.app_session_id = s.app_session_id
                        GROUP BY label
                    ),
                    ranked_har_distribution AS (
                        SELECT
                            li.top_har_label AS label,
                            COUNT(*)::int AS value,
                            ROW_NUMBER() OVER (
                                ORDER BY COUNT(*) DESC, li.top_har_label ASC
                            ) AS row_rank
                        FROM latest_inferences AS li
                        WHERE li.top_har_label IS NOT NULL
                          AND li.top_har_label <> ''
                        GROUP BY li.top_har_label
                    )
                    SELECT
                        metric,
                        label,
                        value
                    FROM (
                        SELECT
                            'warning_level_distribution' AS metric,
                            wd.label,
                            wd.value
                        FROM warning_distribution AS wd
                        UNION ALL
                        SELECT
                            'top_har_labels' AS metric,
                            rhd.label,
                            rhd.value
                        FROM ranked_har_distribution AS rhd
                        WHERE rhd.row_rank <= 6
                    ) AS aggregated
                    ORDER BY metric ASC, value DESC, label ASC
                    """
                )
                distribution_rows = list(cur.fetchall())

                cur.execute(
                    f"""
                    SELECT
                        s.app_session_id,
                        u.subject_key,
                        s.client_session_id,
                        s.device_platform,
                        s.device_model,
                        s.uploaded_at,
                        s.duration_seconds,
                        COALESCE(s.metadata_json ->> 'session_name', s.metadata_json ->> 'file_name')
                            AS session_name,
                        s.metadata_json ->> 'activity_label' AS activity_label,
                        li.status AS latest_status,
                        li.warning_level AS latest_warning_level,
                        li.likely_fall_detected AS latest_likely_fall_detected,
                        li.top_har_label AS latest_top_har_label,
                        li.top_fall_probability AS latest_top_fall_probability,
                        li.grouped_fall_event_count AS latest_grouped_fall_event_count
                    FROM app_sessions AS s
                    JOIN app_users AS u ON u.user_id = s.user_id
                    {_latest_inference_join_sql()}
                    ORDER BY s.created_at DESC
                    LIMIT %s
                    """,
                    (recent_session_limit,),
                )
                recent_session_rows = list(cur.fetchall())
    except Exception as exc:  # noqa: BLE001
        raise RuntimePersistenceError(
            f"Failed to load admin overview: {exc}",
            operation="get_admin_overview",
            diagnostics=_db_exception_diagnostics(exc),
        ) from exc

    return {
        "totals": {
            "users": int(totals_row["users"] or 0),
            "sessions": int(totals_row["sessions"] or 0),
            "inferences": int(totals_row["inferences"] or 0),
            "grouped_fall_events": int(totals_row["grouped_fall_events"] or 0),
        },
        "recent_activity": {
            "sessions_last_7_days": int(totals_row["sessions_last_7_days"] or 0),
            "sessions_with_likely_fall": int(totals_row["sessions_with_likely_fall"] or 0),
        },
        "charts": {
            "sessions_by_day": _fill_daily_buckets(sessions_by_day_rows, days=chart_days),
            "fall_events_by_day": _fill_daily_buckets(fall_events_by_day_rows, days=chart_days),
            "warning_level_distribution": [
                {"label": row["label"], "value": int(row["value"] or 0)}
                for row in distribution_rows
                if row["metric"] == "warning_level_distribution"
            ],
            "top_har_labels": [
                {"label": row["label"], "value": int(row["value"] or 0)}
                for row in distribution_rows
                if row["metric"] == "top_har_labels"
            ],
        },
        "recent_sessions": [
            _admin_overview_session_item_from_row(row) for row in recent_session_rows
        ],
    }


def list_admin_sessions(
    *,
    page: int = 1,
    page_size: int = 25,
    search: str | None = None,
    subject_id: str | None = None,
    warning_level: str | None = None,
    device_platform: str | None = None,
    likely_fall: bool | None = None,
    status: str | None = None,
    dataset_name: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    sort_by: str = "created_at",
    sort_dir: str = "desc",
    db_url: str | None = None,
) -> dict[str, Any]:
    resolved_db_url = _require_db_url(db_url)
    base_filters: list[str] = []
    base_params: list[Any] = []
    latest_inference_filters: list[str] = []
    latest_inference_params: list[Any] = []
    search_filter_sql: str | None = None
    search_params: list[Any] = []

    normalized_search = (search or "").strip()
    if normalized_search:
        like_value = f"%{normalized_search}%"
        search_filter_sql = (
            """
            (
                s.client_session_id ILIKE %s
                OR u.subject_key ILIKE %s
                OR COALESCE(s.metadata_json ->> 'session_name', s.metadata_json ->> 'file_name', '') ILIKE %s
                OR COALESCE(la.annotation_label, '') ILIKE %s
            )
            """
        )
        search_params.extend([like_value, like_value, like_value, like_value])

    normalized_subject_id = (subject_id or "").strip()
    if normalized_subject_id:
        base_filters.append("u.subject_key = %s")
        base_params.append(normalized_subject_id)

    normalized_warning_level = (warning_level or "").strip()
    if normalized_warning_level:
        latest_inference_filters.append("li.warning_level = %s")
        latest_inference_params.append(normalized_warning_level)

    normalized_device_platform = (device_platform or "").strip()
    if normalized_device_platform:
        base_filters.append("s.device_platform = %s")
        base_params.append(normalized_device_platform)

    if likely_fall is not None:
        latest_inference_filters.append("li.likely_fall_detected = %s")
        latest_inference_params.append(likely_fall)

    normalized_status = (status or "").strip()
    if normalized_status:
        latest_inference_filters.append("li.status = %s")
        latest_inference_params.append(normalized_status)

    normalized_dataset_name = (dataset_name or "").strip()
    if normalized_dataset_name:
        base_filters.append("s.dataset_name = %s")
        base_params.append(normalized_dataset_name)

    if date_from is not None:
        base_filters.append("COALESCE(s.uploaded_at, s.created_at) >= %s")
        base_params.append(
            datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc)
        )

    if date_to is not None:
        base_filters.append("COALESCE(s.uploaded_at, s.created_at) < %s")
        base_params.append(
            datetime.combine(date_to + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
        )

    sort_fields = {
        "created_at": "s.created_at",
        "uploaded_at": "COALESCE(s.uploaded_at, s.created_at)",
        "subject_id": "u.subject_key",
        "dataset_name": "s.dataset_name",
        "sample_count": "s.sample_count",
        "duration_seconds": "s.duration_seconds",
        "device_platform": "s.device_platform",
        "status": "li.status",
        "warning_level": "li.warning_level",
        "annotation_label": "la.annotation_label",
        "top_fall_probability": "li.top_fall_probability",
        "grouped_fall_event_count": "li.grouped_fall_event_count",
    }
    latest_lookup_sort_fields = {
        "status",
        "warning_level",
        "annotation_label",
        "top_fall_probability",
        "grouped_fall_event_count",
    }
    order_column = sort_fields.get(sort_by, "s.created_at")
    order_direction = "ASC" if sort_dir.lower() == "asc" else "DESC"
    offset = max(page - 1, 0) * page_size
    count_filters = [*base_filters]
    count_params = [*base_params]
    if search_filter_sql is not None:
        count_filters.append(search_filter_sql)
        count_params.extend(search_params)
    count_filters.extend(latest_inference_filters)
    count_params.extend(latest_inference_params)

    count_from_parts = ["FROM app_sessions AS s"]
    if normalized_search or normalized_subject_id:
        count_from_parts.append("JOIN app_users AS u ON u.user_id = s.user_id")
    if latest_inference_filters:
        count_from_parts.append(_latest_inference_join_sql())
    if search_filter_sql is not None:
        count_from_parts.append(_latest_annotation_join_sql())
    count_from_sql = "\n".join(count_from_parts)
    count_where_sql = f"WHERE {' AND '.join(count_filters)}" if count_filters else ""

    row_filters = [*base_filters]
    row_params = [*base_params]
    if search_filter_sql is not None:
        row_filters.append(search_filter_sql)
        row_params.extend(search_params)
    row_filters.extend(latest_inference_filters)
    row_params.extend(latest_inference_params)
    row_where_sql = f"WHERE {' AND '.join(row_filters)}" if row_filters else ""

    can_defer_latest_lookups = (
        search_filter_sql is None
        and not latest_inference_filters
        and sort_by not in latest_lookup_sort_fields
    )

    try:
        with psycopg.connect(resolved_db_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT COUNT(*)
                    {count_from_sql}
                    {count_where_sql}
                    """,
                    count_params,
                )
                total_count = int(cur.fetchone()["count"] or 0)

                if can_defer_latest_lookups:
                    # Page the base session records first, then resolve latest
                    # inference/annotation data only for the visible rows.
                    cur.execute(
                        f"""
                        WITH paged_sessions AS (
                            SELECT
                                s.app_session_id,
                                u.subject_key,
                                s.client_session_id,
                                s.device_platform,
                                s.device_model,
                                s.uploaded_at,
                                s.duration_seconds,
                                COALESCE(s.metadata_json ->> 'session_name', s.metadata_json ->> 'file_name')
                                    AS session_name,
                                s.metadata_json ->> 'activity_label' AS activity_label,
                                s.notes,
                                ROW_NUMBER() OVER (
                                    ORDER BY {order_column} {order_direction} NULLS LAST, s.created_at DESC
                                ) AS page_rank
                            FROM app_sessions AS s
                            JOIN app_users AS u ON u.user_id = s.user_id
                            {row_where_sql}
                            ORDER BY {order_column} {order_direction} NULLS LAST, s.created_at DESC
                            LIMIT %s OFFSET %s
                        )
                        SELECT
                            p.app_session_id,
                            p.subject_key,
                            p.client_session_id,
                            p.device_platform,
                            p.device_model,
                            p.uploaded_at,
                            p.duration_seconds,
                            p.session_name,
                            p.activity_label,
                            p.notes,
                            li.status AS latest_status,
                            li.warning_level AS latest_warning_level,
                            li.likely_fall_detected AS latest_likely_fall_detected,
                            li.top_har_label AS latest_top_har_label,
                            li.top_fall_probability AS latest_top_fall_probability,
                            li.grouped_fall_event_count AS latest_grouped_fall_event_count,
                            la.annotation_label AS latest_annotation_label
                        FROM paged_sessions AS p
                        {_latest_inference_join_sql("p")}
                        {_latest_annotation_join_sql("p")}
                        ORDER BY p.page_rank ASC
                        """,
                        [*row_params, page_size, offset],
                    )
                    rows = cur.fetchall()
                else:
                    cur.execute(
                        f"""
                        SELECT
                            s.app_session_id,
                            u.subject_key,
                            s.client_session_id,
                            s.device_platform,
                            s.device_model,
                            s.uploaded_at,
                            s.duration_seconds,
                            COALESCE(s.metadata_json ->> 'session_name', s.metadata_json ->> 'file_name')
                                AS session_name,
                            s.metadata_json ->> 'activity_label' AS activity_label,
                            s.notes,
                            li.status AS latest_status,
                            li.warning_level AS latest_warning_level,
                            li.likely_fall_detected AS latest_likely_fall_detected,
                            li.top_har_label AS latest_top_har_label,
                            li.top_fall_probability AS latest_top_fall_probability,
                            li.grouped_fall_event_count AS latest_grouped_fall_event_count,
                            la.annotation_label AS latest_annotation_label
                        FROM app_sessions AS s
                        JOIN app_users AS u ON u.user_id = s.user_id
                        {_latest_inference_join_sql()}
                        {_latest_annotation_join_sql()}
                        {row_where_sql}
                        ORDER BY {order_column} {order_direction} NULLS LAST, s.created_at DESC
                        LIMIT %s OFFSET %s
                        """,
                        [*row_params, page_size, offset],
                    )
                    rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001
        raise RuntimePersistenceError(
            f"Failed to list admin sessions: {exc}",
            operation="list_admin_sessions",
            diagnostics=_db_exception_diagnostics(exc),
        ) from exc

    total_pages = int(math.ceil(total_count / page_size)) if total_count else 1
    return {
        "sessions": [_admin_session_list_item_from_row(row) for row in rows],
        "total_count": total_count,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


def list_persisted_sessions(
    *,
    subject_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db_url: str | None = None,
    owner_user_id: UUID | None = None,
) -> dict[str, Any]:
    resolved_db_url = _require_db_url(db_url)
    filters: list[str] = []
    params: list[Any] = []

    if subject_id is not None and subject_id.strip():
        filters.append("u.subject_key = %s")
        params.append(subject_id.strip())
    if owner_user_id is not None:
        filters.append("s.user_id = %s")
        params.append(owner_user_id)

    where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""

    try:
        with psycopg.connect(resolved_db_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM app_sessions AS s
                    JOIN app_users AS u ON u.user_id = s.user_id
                    {where_sql}
                    """,
                    params,
                )
                total_count = int(cur.fetchone()["count"] or 0)

                cur.execute(
                    f"""
                    SELECT
                        s.app_session_id,
                        s.user_id,
                        s.device_id,
                        u.subject_key,
                        s.client_session_id,
                        s.request_id,
                        s.trace_id,
                        s.client_version,
                        s.dataset_name,
                        s.source_type,
                        s.task_type,
                        s.placement_declared,
                        s.device_platform,
                        s.device_model,
                        s.app_version,
                        s.app_build,
                        s.recording_mode,
                        s.runtime_mode,
                        s.recording_started_at,
                        s.recording_ended_at,
                        s.uploaded_at,
                        s.sampling_rate_hz,
                        s.sample_count,
                        s.has_gyro,
                        s.duration_seconds,
                        COALESCE(s.metadata_json ->> 'session_name', s.metadata_json ->> 'file_name')
                            AS session_name,
                        s.metadata_json ->> 'activity_label' AS activity_label,
                        s.notes,
                        s.raw_storage_uri,
                        s.raw_storage_format,
                        s.raw_payload_sha256,
                        s.raw_payload_bytes,
                        s.created_at,
                        s.updated_at,
                        li.inference_id AS latest_inference_id,
                        li.request_id AS latest_inference_request_id,
                        li.created_at AS latest_inference_created_at,
                        li.status AS latest_status,
                        li.warning_level AS latest_warning_level,
                        li.likely_fall_detected AS latest_likely_fall_detected,
                        li.top_har_label AS latest_top_har_label,
                        li.top_fall_probability AS latest_top_fall_probability,
                        li.grouped_fall_event_count AS latest_grouped_fall_event_count,
                        la.annotation_label AS latest_annotation_label,
                        la.source AS latest_annotation_source,
                        la.created_at AS latest_annotation_created_at
                    FROM app_sessions AS s
                    JOIN app_users AS u ON u.user_id = s.user_id
                    {_latest_inference_join_sql()}
                    {_latest_annotation_join_sql()}
                    {where_sql}
                    ORDER BY s.created_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    [*params, limit, offset],
                )
                rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001
        raise RuntimePersistenceError(
            f"Failed to list persisted sessions: {exc}",
            operation="list_persisted_sessions",
            diagnostics=_db_exception_diagnostics(exc),
        ) from exc

    sessions = [_session_list_item_from_row(row) for row in rows]
    return {
        "sessions": sessions,
        "total_count": total_count,
        "limit": limit,
        "offset": offset,
    }


def list_session_annotations(
    app_session_id: UUID,
    *,
    db_url: str | None = None,
    owner_user_id: UUID | None = None,
) -> dict[str, Any] | None:
    resolved_db_url = _require_db_url(db_url)

    try:
        with psycopg.connect(resolved_db_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                if not _session_is_accessible(
                    cur,
                    app_session_id=app_session_id,
                    owner_user_id=owner_user_id,
                ):
                    return None
                annotation_rows = _fetch_session_annotation_rows(
                    cur,
                    app_session_id=app_session_id,
                )
    except Exception as exc:  # noqa: BLE001
        raise RuntimePersistenceError(
            f"Failed to list session annotations: {exc}",
            operation="list_session_annotations",
            diagnostics=_db_exception_diagnostics(exc),
        ) from exc

    return {
        "app_session_id": app_session_id,
        "annotations": [
            _session_annotation_record_from_row(row)
            for row in annotation_rows
        ],
    }


def create_session_annotation(
    app_session_id: UUID,
    req: SessionAnnotationRequest,
    *,
    source: SessionAnnotationSource,
    auth_account_id: UUID | None = None,
    db_url: str | None = None,
    owner_user_id: UUID | None = None,
) -> dict[str, Any] | None:
    resolved_db_url = _require_db_url(db_url)
    request_id = req.request_context.request_id if req.request_context is not None else None

    try:
        with psycopg.connect(resolved_db_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                if not _session_is_accessible(
                    cur,
                    app_session_id=app_session_id,
                    owner_user_id=owner_user_id,
                ):
                    return None

                existing = _lookup_existing_annotation_by_request(
                    cur,
                    request_id=request_id,
                )
                if existing is not None:
                    if existing["app_session_id"] != app_session_id:
                        raise RuntimePersistenceError(
                            "request_id is already associated with a different session annotation."
                        )
                    annotation_row = _fetch_session_annotation_by_id(
                        cur,
                        annotation_id=existing["annotation_id"],
                    )
                    conn.commit()
                    if annotation_row is None:
                        raise RuntimePersistenceError(
                            "Existing session annotation could not be reloaded."
                        )
                    return _session_annotation_record_from_row(annotation_row)

                cur.execute(
                    """
                    INSERT INTO app_session_annotations (
                        app_session_id,
                        annotation_label,
                        source,
                        reviewer_identifier,
                        auth_account_id,
                        request_id,
                        notes
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING annotation_id
                    """,
                    (
                        app_session_id,
                        req.label.value,
                        source.value,
                        req.reviewer_id,
                        auth_account_id,
                        request_id,
                        req.notes,
                    ),
                )
                row = cur.fetchone()
                if row is None:
                    raise RuntimePersistenceError("Failed to create session annotation.")
                annotation_row = _fetch_session_annotation_by_id(
                    cur,
                    annotation_id=row["annotation_id"],
                )
            conn.commit()
    except Exception as exc:  # noqa: BLE001
        raise RuntimePersistenceError(
            f"Failed to create session annotation: {exc}",
            operation="create_session_annotation",
            diagnostics=_db_exception_diagnostics(exc),
        ) from exc

    if annotation_row is None:
        raise RuntimePersistenceError("Created session annotation could not be reloaded.")
    return _session_annotation_record_from_row(annotation_row)


def get_session_raw_storage_location(
    app_session_id: UUID,
    *,
    db_url: str | None = None,
    owner_user_id: UUID | None = None,
) -> dict[str, Any] | None:
    """Return the on-disk location of a session's raw payload, scoped by owner.

    Returns a mapping with ``raw_storage_uri`` / ``raw_storage_format`` /
    ``raw_payload_sha256`` / ``raw_payload_bytes``, or ``None`` if the session
    does not exist or is not owned by ``owner_user_id`` (when provided).
    """
    resolved_db_url = _require_db_url(db_url)

    try:
        with psycopg.connect(resolved_db_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        raw_storage_uri,
                        raw_storage_format,
                        raw_payload_sha256,
                        raw_payload_bytes
                    FROM app_sessions
                    WHERE app_session_id = %s
                    {"AND user_id = %s" if owner_user_id is not None else ""}
                    """,
                    (app_session_id, owner_user_id)
                    if owner_user_id is not None
                    else (app_session_id,),
                )
                row = cur.fetchone()
    except Exception as exc:  # noqa: BLE001
        raise RuntimePersistenceError(
            f"Failed to load session raw storage location: {exc}",
            operation="get_session_raw_storage_location",
            diagnostics=_db_exception_diagnostics(exc),
        ) from exc

    if row is None:
        return None
    return {
        "raw_storage_uri": row["raw_storage_uri"],
        "raw_storage_format": row["raw_storage_format"],
        "raw_payload_sha256": row["raw_payload_sha256"],
        "raw_payload_bytes": row["raw_payload_bytes"],
    }


def get_persisted_session_detail(
    app_session_id: UUID,
    *,
    db_url: str | None = None,
    owner_user_id: UUID | None = None,
) -> dict[str, Any] | None:
    resolved_db_url = _require_db_url(db_url)

    try:
        with psycopg.connect(resolved_db_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        s.app_session_id,
                        s.user_id,
                        s.device_id,
                        u.subject_key,
                        s.client_session_id,
                        s.request_id,
                        s.trace_id,
                        s.client_version,
                        s.dataset_name,
                        s.source_type,
                        s.task_type,
                        s.placement_declared,
                        s.device_platform,
                        s.device_model,
                        s.app_version,
                        s.app_build,
                        s.recording_mode,
                        s.runtime_mode,
                        s.recording_started_at,
                        s.recording_ended_at,
                        s.uploaded_at,
                        s.sampling_rate_hz,
                        s.sample_count,
                        s.has_gyro,
                        s.duration_seconds,
                        COALESCE(s.metadata_json ->> 'session_name', s.metadata_json ->> 'file_name')
                            AS session_name,
                        s.metadata_json ->> 'activity_label' AS activity_label,
                        s.notes,
                        s.raw_storage_uri,
                        s.raw_storage_format,
                        s.raw_payload_sha256,
                        s.raw_payload_bytes,
                        s.created_at,
                        s.updated_at
                    FROM app_sessions AS s
                    JOIN app_users AS u ON u.user_id = s.user_id
                    WHERE s.app_session_id = %s
                    {"AND s.user_id = %s" if owner_user_id is not None else ""}
                    """,
                    (app_session_id, owner_user_id)
                    if owner_user_id is not None
                    else (app_session_id,),
                )
                session_row = cur.fetchone()
                if session_row is None:
                    return None

                cur.execute(
                    """
                    SELECT
                        inference_id,
                        request_id,
                        status,
                        error_message,
                        started_at,
                        completed_at,
                        created_at,
                        source_summary_json,
                        placement_summary_json,
                        har_summary_json,
                        fall_summary_json,
                        alert_summary_json,
                        debug_summary_json,
                        model_info_json,
                        session_narrative_summary_json,
                        narrative_summary_json
                    FROM app_session_inferences
                    WHERE app_session_id = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (app_session_id,),
                )
                inference_row = cur.fetchone()

                grouped_rows: list[Mapping[str, Any]] = []
                timeline_rows: list[Mapping[str, Any]] = []
                transition_rows: list[Mapping[str, Any]] = []
                if inference_row is not None:
                    inference_id = inference_row["inference_id"]

                    cur.execute(
                        """
                        SELECT
                            event_key,
                            event_start_ts,
                            event_end_ts,
                            event_duration_seconds,
                            positive_window_count,
                            peak_probability,
                            mean_probability,
                            median_probability
                        FROM app_grouped_fall_events
                        WHERE app_session_id = %s AND inference_id = %s
                        ORDER BY event_start_ts ASC, event_key ASC
                        """,
                        (app_session_id, inference_id),
                    )
                    grouped_rows = list(cur.fetchall())

                    cur.execute(
                        """
                        SELECT
                            event_key,
                            start_ts,
                            end_ts,
                            duration_seconds,
                            midpoint_ts,
                            point_count,
                            activity_label,
                            placement_label,
                            activity_confidence_mean,
                            placement_confidence_mean,
                            fall_probability_peak,
                            fall_probability_mean,
                            likely_fall,
                            event_kind,
                            related_grouped_event_ids_json,
                            description
                        FROM app_timeline_events
                        WHERE app_session_id = %s AND inference_id = %s
                        ORDER BY start_ts ASC, event_key ASC
                        """,
                        (app_session_id, inference_id),
                    )
                    timeline_rows = list(cur.fetchall())

                    cur.execute(
                        """
                        SELECT
                            transition_key,
                            transition_ts,
                            from_event_key,
                            to_event_key,
                            transition_kind,
                            from_activity_label,
                            to_activity_label,
                            from_placement_label,
                            to_placement_label,
                            description
                        FROM app_transition_events
                        WHERE app_session_id = %s AND inference_id = %s
                        ORDER BY transition_ts ASC, transition_key ASC
                        """,
                        (app_session_id, inference_id),
                    )
                    transition_rows = list(cur.fetchall())

                cur.execute(
                    """
                    SELECT
                        feedback_id,
                        app_session_id,
                        inference_id,
                        target_type,
                        target_event_key,
                        window_id,
                        feedback_type,
                        corrected_label,
                        reviewer_identifier,
                        subject_key,
                        notes,
                        request_id,
                        recorded_at
                    FROM app_feedback
                    WHERE app_session_id = %s
                    ORDER BY recorded_at DESC, feedback_id DESC
                    """,
                    (app_session_id,),
                )
                feedback_rows = cur.fetchall()
                annotation_rows = _fetch_session_annotation_rows(
                    cur,
                    app_session_id=app_session_id,
                )
    except Exception as exc:  # noqa: BLE001
        raise RuntimePersistenceError(
            f"Failed to load persisted session detail: {exc}",
            operation="get_persisted_session_detail",
            diagnostics=_db_exception_diagnostics(exc),
        ) from exc

    latest_inference = None
    if inference_row is not None:
        latest_inference = {
            "inference_id": inference_row["inference_id"],
            "request_id": inference_row["request_id"],
            "status": inference_row["status"],
            "error_message": inference_row["error_message"],
            "started_at": inference_row["started_at"],
            "completed_at": inference_row["completed_at"],
            "created_at": inference_row["created_at"],
            "response": _runtime_response_from_rows(
                session_row,
                inference_row,
                grouped_rows=grouped_rows,
                timeline_rows=timeline_rows,
                transition_rows=transition_rows,
            ),
        }

    return {
        "session": _session_record_from_row(session_row),
        "latest_inference": latest_inference,
        "feedback": [_feedback_record_from_row(row) for row in feedback_rows],
        "annotations": [
            _session_annotation_record_from_row(row)
            for row in annotation_rows
        ],
    }


def get_admin_session_detail_summary(
    app_session_id: UUID,
    *,
    db_url: str | None = None,
) -> dict[str, Any] | None:
    resolved_db_url = _require_db_url(db_url)

    try:
        with psycopg.connect(resolved_db_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        s.app_session_id,
                        s.user_id,
                        s.device_id,
                        u.subject_key,
                        s.client_session_id,
                        s.request_id,
                        s.trace_id,
                        s.client_version,
                        s.dataset_name,
                        s.source_type,
                        s.task_type,
                        s.placement_declared,
                        s.device_platform,
                        s.device_model,
                        s.app_version,
                        s.app_build,
                        s.recording_mode,
                        s.runtime_mode,
                        s.recording_started_at,
                        s.recording_ended_at,
                        s.uploaded_at,
                        s.sampling_rate_hz,
                        s.sample_count,
                        s.has_gyro,
                        s.duration_seconds,
                        COALESCE(s.metadata_json ->> 'session_name', s.metadata_json ->> 'file_name')
                            AS session_name,
                        s.metadata_json ->> 'activity_label' AS activity_label,
                        s.notes,
                        s.raw_storage_uri,
                        s.raw_storage_format,
                        s.raw_payload_sha256,
                        s.raw_payload_bytes,
                        s.created_at,
                        s.updated_at
                    FROM app_sessions AS s
                    JOIN app_users AS u ON u.user_id = s.user_id
                    WHERE s.app_session_id = %s
                    """,
                    (app_session_id,),
                )
                session_row = cur.fetchone()
                if session_row is None:
                    return None

                cur.execute(
                    """
                    SELECT
                        inference_id,
                        request_id,
                        status,
                        error_message,
                        started_at,
                        completed_at,
                        created_at,
                        source_summary_json,
                        placement_summary_json,
                        har_summary_json,
                        fall_summary_json,
                        alert_summary_json,
                        model_info_json,
                        session_narrative_summary_json,
                        narrative_summary_json,
                        grouped_fall_event_count,
                        timeline_event_count,
                        transition_event_count
                    FROM app_session_inferences
                    WHERE app_session_id = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (app_session_id,),
                )
                inference_row = cur.fetchone()
                latest_feedback_row = _fetch_latest_feedback_row(
                    cur,
                    app_session_id=app_session_id,
                )
                feedback_count = _count_session_feedback_rows(
                    cur,
                    app_session_id=app_session_id,
                )
                latest_annotation_row = _fetch_latest_session_annotation_row(
                    cur,
                    app_session_id=app_session_id,
                )
                annotation_count = _count_session_annotation_rows(
                    cur,
                    app_session_id=app_session_id,
                )
    except Exception as exc:  # noqa: BLE001
        raise RuntimePersistenceError(
            f"Failed to load admin session detail summary: {exc}",
            operation="get_admin_session_detail_summary",
            diagnostics=_db_exception_diagnostics(exc),
        ) from exc

    evidence_counts = {
        "grouped_fall_events": 0,
        "timeline_events": 0,
        "transition_events": 0,
        "feedback": feedback_count,
        "annotations": annotation_count,
    }
    latest_inference = None
    if inference_row is not None:
        evidence_counts["grouped_fall_events"] = int(
            inference_row["grouped_fall_event_count"] or 0
        )
        evidence_counts["timeline_events"] = int(
            inference_row["timeline_event_count"] or 0
        )
        evidence_counts["transition_events"] = int(
            inference_row["transition_event_count"] or 0
        )
        latest_inference = {
            "inference_id": inference_row["inference_id"],
            "request_id": inference_row["request_id"],
            "status": inference_row["status"],
            "error_message": inference_row["error_message"],
            "started_at": inference_row["started_at"],
            "completed_at": inference_row["completed_at"],
            "created_at": inference_row["created_at"],
            "response": _runtime_summary_from_rows(
                session_row,
                inference_row,
            ),
        }

    return {
        "session": _session_record_from_row(session_row),
        "latest_inference": latest_inference,
        "latest_feedback": _feedback_record_from_row(latest_feedback_row)
        if latest_feedback_row is not None
        else None,
        "latest_annotation": _session_annotation_record_from_row(
            latest_annotation_row
        )
        if latest_annotation_row is not None
        else None,
        "evidence_counts": evidence_counts,
    }


def get_admin_session_evidence(
    app_session_id: UUID,
    *,
    sections: list[str] | None = None,
    db_url: str | None = None,
) -> dict[str, Any] | None:
    resolved_db_url = _require_db_url(db_url)
    allowed_sections = (
        "grouped_fall_events",
        "timeline_events",
        "transition_events",
        "feedback",
        "annotations",
    )
    requested_sections = []
    for section in sections or list(allowed_sections):
        if section not in allowed_sections:
            raise ValueError(f"Unsupported evidence section: {section}")
        if section not in requested_sections:
            requested_sections.append(section)

    grouped_rows: list[Mapping[str, Any]] = []
    timeline_rows: list[Mapping[str, Any]] = []
    transition_rows: list[Mapping[str, Any]] = []
    feedback_rows: list[Mapping[str, Any]] = []
    annotation_rows: list[Mapping[str, Any]] = []
    needs_inference = any(
        section in {"grouped_fall_events", "timeline_events", "transition_events"}
        for section in requested_sections
    )

    try:
        with psycopg.connect(resolved_db_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                if not _session_is_accessible(cur, app_session_id=app_session_id):
                    return None

                inference_row = None
                if needs_inference:
                    cur.execute(
                        """
                        SELECT inference_id
                        FROM app_session_inferences
                        WHERE app_session_id = %s
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        (app_session_id,),
                    )
                    inference_row = cur.fetchone()

                inference_id = (
                    inference_row["inference_id"] if inference_row is not None else None
                )
                if inference_id is not None and "grouped_fall_events" in requested_sections:
                    cur.execute(
                        """
                        SELECT
                            event_key,
                            event_start_ts,
                            event_end_ts,
                            event_duration_seconds,
                            positive_window_count,
                            peak_probability,
                            mean_probability,
                            median_probability
                        FROM app_grouped_fall_events
                        WHERE app_session_id = %s AND inference_id = %s
                        ORDER BY event_start_ts ASC, event_key ASC
                        """,
                        (app_session_id, inference_id),
                    )
                    grouped_rows = list(cur.fetchall())

                if inference_id is not None and "timeline_events" in requested_sections:
                    cur.execute(
                        """
                        SELECT
                            event_key,
                            start_ts,
                            end_ts,
                            duration_seconds,
                            midpoint_ts,
                            point_count,
                            activity_label,
                            placement_label,
                            activity_confidence_mean,
                            placement_confidence_mean,
                            fall_probability_peak,
                            fall_probability_mean,
                            likely_fall,
                            event_kind,
                            related_grouped_event_ids_json,
                            description
                        FROM app_timeline_events
                        WHERE app_session_id = %s AND inference_id = %s
                        ORDER BY start_ts ASC, event_key ASC
                        """,
                        (app_session_id, inference_id),
                    )
                    timeline_rows = list(cur.fetchall())

                if inference_id is not None and "transition_events" in requested_sections:
                    cur.execute(
                        """
                        SELECT
                            transition_key,
                            transition_ts,
                            from_event_key,
                            to_event_key,
                            transition_kind,
                            from_activity_label,
                            to_activity_label,
                            from_placement_label,
                            to_placement_label,
                            description
                        FROM app_transition_events
                        WHERE app_session_id = %s AND inference_id = %s
                        ORDER BY transition_ts ASC, transition_key ASC
                        """,
                        (app_session_id, inference_id),
                    )
                    transition_rows = list(cur.fetchall())

                if "feedback" in requested_sections:
                    cur.execute(
                        """
                        SELECT
                            feedback_id,
                            app_session_id,
                            inference_id,
                            target_type,
                            target_event_key,
                            window_id,
                            feedback_type,
                            corrected_label,
                            reviewer_identifier,
                            subject_key,
                            notes,
                            request_id,
                            recorded_at
                        FROM app_feedback
                        WHERE app_session_id = %s
                        ORDER BY recorded_at DESC, feedback_id DESC
                        """,
                        (app_session_id,),
                    )
                    feedback_rows = list(cur.fetchall())

                if "annotations" in requested_sections:
                    annotation_rows = _fetch_session_annotation_rows(
                        cur,
                        app_session_id=app_session_id,
                    )
    except Exception as exc:  # noqa: BLE001
        raise RuntimePersistenceError(
            f"Failed to load admin session evidence: {exc}",
            operation="get_admin_session_evidence",
            diagnostics=_db_exception_diagnostics(exc),
        ) from exc

    return {
        "loaded_sections": requested_sections,
        "grouped_fall_events": [
            _grouped_fall_event_from_row(row) for row in grouped_rows
        ],
        "timeline_events": [_timeline_event_from_row(row) for row in timeline_rows],
        "transition_events": [
            _transition_event_from_row(row) for row in transition_rows
        ],
        "feedback": [_feedback_record_from_row(row) for row in feedback_rows],
        "annotations": [
            _session_annotation_record_from_row(row)
            for row in annotation_rows
        ],
    }


def _remove_raw_storage_uri(raw_storage_uri: str | None) -> None:
    if raw_storage_uri is None or not raw_storage_uri.strip():
        return
    try:
        Path(raw_storage_uri).unlink(missing_ok=True)
    except Exception:
        return


def delete_persisted_session(
    app_session_id: UUID,
    *,
    db_url: str | None = None,
    owner_user_id: UUID | None = None,
) -> bool:
    resolved_db_url = _require_db_url(db_url)
    raw_storage_uri: str | None = None

    try:
        with psycopg.connect(resolved_db_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    DELETE FROM app_sessions
                    WHERE app_session_id = %s
                    {"AND user_id = %s" if owner_user_id is not None else ""}
                    RETURNING raw_storage_uri
                    """,
                    (app_session_id, owner_user_id)
                    if owner_user_id is not None
                    else (app_session_id,),
                )
                row = cur.fetchone()
            conn.commit()
    except Exception as exc:  # noqa: BLE001
        raise RuntimePersistenceError(
            f"Failed to delete persisted session: {exc}",
            operation="delete_persisted_session",
            diagnostics=_db_exception_diagnostics(exc),
        ) from exc

    if row is None:
        return False

    raw_storage_uri = row.get("raw_storage_uri")
    _remove_raw_storage_uri(raw_storage_uri)
    return True
