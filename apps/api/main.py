from __future__ import annotations

import json
import logging
import os
import secrets
import threading
from collections import deque
from collections.abc import Mapping
from datetime import date, datetime, timezone
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any
from uuid import UUID, uuid4

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.middleware.sessions import SessionMiddleware

from apps.api.schemas import (
    AdminAuthLoginRequest,
    AdminAuthSessionResponse,
    AdminOverviewResponse,
    AdminSessionDetailSummaryResponse,
    AdminSessionEvidenceResponse,
    AdminSessionListResponse,
    AuthenticatedUserResponse,
    CombinedTimelinePoint,
    DeleteSessionResponse,
    ErrorResponse,
    FallSummary,
    FallWindowPrediction,
    GroupedFallEvent,
    HarSummary,
    HarWindowPrediction,
    HealthResponse,
    ModelInfo,
    PlacementSummary,
    PersistedSessionDetailResponse,
    PersistedSessionListResponse,
    PointTimelinePoint,
    PredictionFeedbackRequest,
    PredictionFeedbackResponse,
    RuntimeAlertSummary,
    RuntimeDebugSummary,
    SessionAnnotationListResponse,
    SessionAnnotationRecord,
    SessionAnnotationRequest,
    SessionAnnotationSource,
    SelfServiceRegistrationRequest,
    SelfServiceRegistrationResponse,
    RuntimeSessionRequest,
    RuntimeSessionResponse,
    SessionNarrativeSummary,
    SourceSummary,
    TimelineEvent,
    TransitionEvent,
    VulnerabilitySummary,
    VulnerabilityWindowPrediction,
)
from pipeline.artifacts import resolve_current_artifact
from pipeline.preprocess.dejitter import drop_phantom_leading_samples
from services.runtime_inference import (
    RuntimeArtifacts,
    RuntimeInferenceConfig,
    run_runtime_inference_from_dataframe,
)
from services.runtime_auth import (
    AuthenticatedPrincipal,
    RuntimeAuthenticationError,
    RuntimeAuthorizationError,
    RuntimeRegistrationConflictError,
    RuntimeRegistrationError,
    auth_required,
    authenticate_basic_credentials,
    load_authenticated_principal_by_account_id,
    normalize_subject_for_principal,
    owner_user_id_for_principal,
    register_self_service_user,
    self_service_registration_enabled,
)
from services.runtime_logging import configure_runtime_logging, log_event
from services.runtime_persistence import (
    RuntimePersistenceError,
    create_session_annotation,
    delete_persisted_session,
    get_admin_session_detail_summary,
    get_admin_session_evidence,
    get_admin_overview,
    get_persisted_session_detail,
    get_session_raw_storage_location,
    list_session_annotations,
    list_admin_sessions,
    list_persisted_sessions,
    persist_feedback as persist_feedback_record,
    persist_runtime_session,
    persistence_enabled,
)
from services.runtime_persistence import _resolve_storage_dir as _resolve_session_storage_dir

API_VERSION = "1.1.0"
APP_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_HAR_ARTIFACT = APP_ROOT / "artifacts" / "har" / "current" / "model.joblib"
DEFAULT_FALL_ARTIFACT = APP_ROOT / "artifacts" / "fall" / "current" / "model.joblib"
DEFAULT_FEEDBACK_STORE = APP_ROOT / "artifacts" / "feedback" / "prediction_feedback.jsonl"
DEFAULT_ADMIN_WEB_DIST = APP_ROOT / "apps" / "admin" / "dist"


def _path_from_env(env_var: str, default_path: Path) -> tuple[Path, bool]:
    """Return (resolved_path, override_active) for an optional path env var."""
    raw_value = os.getenv(env_var)
    if raw_value is None or not raw_value.strip():
        return default_path, False
    return Path(raw_value).expanduser().resolve(), True


HAR_ARTIFACT_PATH, HAR_ARTIFACT_OVERRIDE = _path_from_env("HAR_ARTIFACT_PATH", DEFAULT_HAR_ARTIFACT)
FALL_ARTIFACT_PATH, FALL_ARTIFACT_OVERRIDE = _path_from_env("FALL_ARTIFACT_PATH", DEFAULT_FALL_ARTIFACT)
FEEDBACK_STORE_PATH, _ = _path_from_env("FEEDBACK_STORE_PATH", DEFAULT_FEEDBACK_STORE)
ADMIN_WEB_DIST_PATH, _ = _path_from_env("ADMIN_WEB_DIST_PATH", DEFAULT_ADMIN_WEB_DIST)
ADMIN_WEB_ENABLED = os.getenv("ADMIN_WEB_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
configure_runtime_logging(service_name="runtime-inference-api", log_level=LOG_LEVEL)
logger = logging.getLogger("runtime_inference_api")
security = HTTPBasic(auto_error=False)

ADMIN_SESSION_EVIDENCE_SECTIONS = (
    "grouped_fall_events",
    "timeline_events",
    "transition_events",
    "feedback",
    "annotations",
)


def _parse_allowed_origins() -> list[str]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "*")
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins or ["*"]


ALLOWED_ORIGINS = _parse_allowed_origins()

_ADMIN_SESSION_SECRET_DEV_FALLBACK = "unifallmonitor-dev-session-secret-change-me"


def _resolve_admin_session_secret() -> str:
    raw = os.getenv("ADMIN_SESSION_SECRET", "").strip()
    if not raw:
        raise RuntimeError(
            "ADMIN_SESSION_SECRET is not set. Refusing to start with an unsigned "
            "or default session key. Set ADMIN_SESSION_SECRET to a long random "
            "value before launching the API."
        )
    if raw == _ADMIN_SESSION_SECRET_DEV_FALLBACK:
        raise RuntimeError(
            "ADMIN_SESSION_SECRET is set to the documented placeholder. Replace "
            "it with a long random value before launching the API."
        )
    return raw


ADMIN_SESSION_SECRET = _resolve_admin_session_secret()
ADMIN_SESSION_COOKIE = os.getenv(
    "ADMIN_SESSION_COOKIE_NAME",
    "unifallmonitor_admin_session",
)
_DEFAULT_ADMIN_SESSION_MAX_AGE_SECONDS = 12 * 60 * 60
ADMIN_SESSION_MAX_AGE = int(
    os.getenv("ADMIN_SESSION_MAX_AGE_SECONDS", str(_DEFAULT_ADMIN_SESSION_MAX_AGE_SECONDS))
)
ADMIN_SESSION_SAME_SITE = os.getenv("ADMIN_SESSION_SAME_SITE", "lax")
ADMIN_SESSION_HTTPS_ONLY = os.getenv("ADMIN_SESSION_HTTPS_ONLY", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ADMIN_SESSION_KEY = "admin_session"
CSRF_COOKIE_NAME = os.getenv("ADMIN_CSRF_COOKIE_NAME", "unifallmonitor_csrf")
CSRF_HEADER_NAME = "X-CSRF-Token"
_CSRF_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_CSRF_EXEMPT_PATHS = {
    "/v1/admin/auth/login",
}
MIN_RUNTIME_SESSION_DURATION_SECONDS = 3.0

REGISTRATION_RATE_LIMIT_PER_MINUTE = max(
    0, int(os.getenv("RATE_LIMIT_REGISTRATION_PER_MINUTE", "5"))
)
_REGISTRATION_RATE_LIMIT_WINDOW_SECONDS = 60.0
_registration_rate_limiter_lock = threading.Lock()
_registration_rate_limiter_buckets: dict[str, deque[float]] = {}


def _enforce_registration_rate_limit(client_ip: str | None) -> None:
    if REGISTRATION_RATE_LIMIT_PER_MINUTE <= 0:
        return
    key = client_ip or "unknown"
    now = monotonic()
    cutoff = now - _REGISTRATION_RATE_LIMIT_WINDOW_SECONDS
    with _registration_rate_limiter_lock:
        bucket = _registration_rate_limiter_buckets.setdefault(key, deque())
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= REGISTRATION_RATE_LIMIT_PER_MINUTE:
            retry_after = max(1, int(bucket[0] + _REGISTRATION_RATE_LIMIT_WINDOW_SECONDS - now))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many registration attempts. Please retry shortly.",
                headers={"Retry-After": str(retry_after)},
            )
        bucket.append(now)

app = FastAPI(
    title="Runtime Fall Detection API",
    version=API_VERSION,
    description="Mobile session inference API for HAR + fall detection.",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOWED_ORIGINS != ["*"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
app.add_middleware(
    GZipMiddleware,
    minimum_size=500,
)
app.add_middleware(
    SessionMiddleware,
    secret_key=ADMIN_SESSION_SECRET,
    session_cookie=ADMIN_SESSION_COOKIE,
    max_age=ADMIN_SESSION_MAX_AGE,
    same_site=ADMIN_SESSION_SAME_SITE,
    https_only=ADMIN_SESSION_HTTPS_ONLY,
)


@app.on_event("startup")
def _probe_canonical_artifacts() -> None:
    """Confirm the resolved artifact path is present and non-empty.

    Always stat the path that will actually be loaded at request time —
    whether it came from env override or the canonical artifacts/<task>/current
    layout. A missing or 0-byte file fails fast at startup instead of
    surfacing as an opaque inference error on the first request.
    """
    for task, override_active, override_path in (
        ("fall", FALL_ARTIFACT_OVERRIDE, FALL_ARTIFACT_PATH),
        ("har", HAR_ARTIFACT_OVERRIDE, HAR_ARTIFACT_PATH),
    ):
        if override_active:
            resolved = override_path
            source = "env_override"
        else:
            resolved = resolve_current_artifact(task)
            source = "canonical_current"

        if not resolved.is_file():
            raise RuntimeError(
                f"Artifact for task '{task}' is missing at {resolved} (source={source})."
            )
        if resolved.stat().st_size == 0:
            raise RuntimeError(
                f"Artifact for task '{task}' is 0 bytes at {resolved} (source={source})."
            )

        log_event(
            logger,
            logging.INFO,
            "artifact_probe_ok",
            task=task,
            artifact_path=str(resolved),
            source=source,
            size_bytes=resolved.stat().st_size,
        )


def _principal_log_fields(principal: AuthenticatedPrincipal | None) -> dict[str, Any]:
    if principal is None:
        return {}
    return {
        "auth_username": principal.login_username,
        "auth_role": principal.role,
        "auth_user_id": principal.user_id,
        "auth_subject_key": principal.subject_key,
    }


def _set_request_state(
    request: Request,
    *,
    request_id: UUID | None = None,
    trace_id: str | None = None,
    session_id: str | None = None,
    subject_id: str | None = None,
    principal: AuthenticatedPrincipal | None = None,
) -> None:
    if request_id is not None:
        request.state.request_id = request_id
    if trace_id is not None:
        request.state.trace_id = trace_id
    if session_id is not None:
        request.state.session_id = session_id
    if subject_id is not None:
        request.state.subject_id = subject_id
    if principal is not None:
        request.state.auth_username = principal.login_username
        request.state.auth_role = principal.role
        request.state.auth_user_id = principal.user_id
        request.state.auth_subject_key = principal.subject_key


def _effective_request_id(request: Request) -> UUID | None:
    return getattr(request.state, "request_id", None) or getattr(request.state, "http_request_id", None)


def _client_ip(request: Request) -> str | None:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip() or None
    client = request.client
    return client.host if client is not None else None


@app.middleware("http")
async def csrf_middleware(request: Request, call_next):
    """Double-submit CSRF protection for cookie-authenticated admin requests.

    Threat model: an admin signed in via the session cookie visits a
    malicious page that tries to POST to the API. SameSite=lax blocks the
    obvious form POST but not every variant; this middleware adds
    defence-in-depth by requiring an X-CSRF-Token header that matches the
    csrf cookie on every state-changing request that carries the admin
    session cookie. Mobile/Basic-auth callers don't send that cookie and
    aren't subject to the check.
    """
    method = request.method.upper()
    path = request.url.path
    has_session_cookie = bool(request.cookies.get(ADMIN_SESSION_COOKIE))
    needs_check = (
        method not in _CSRF_SAFE_METHODS
        and has_session_cookie
        and path not in _CSRF_EXEMPT_PATHS
    )

    if needs_check:
        cookie_token = request.cookies.get(CSRF_COOKIE_NAME, "")
        header_token = request.headers.get(CSRF_HEADER_NAME, "")
        if (
            not cookie_token
            or not header_token
            or not secrets.compare_digest(cookie_token, header_token)
        ):
            log_event(
                logger,
                logging.WARNING,
                "csrf_token_mismatch",
                method=method,
                path=path,
                client_ip=_client_ip(request),
            )
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "request_id": str(uuid4()),
                    "error_code": "csrf_token_mismatch",
                    "message": "CSRF token missing or invalid.",
                    "details": {},
                },
            )

    response = await call_next(request)

    issue_csrf_cookie = request.cookies.get(CSRF_COOKIE_NAME) is None and (
        has_session_cookie
        or (
            path == "/v1/admin/auth/login"
            and method == "POST"
            and 200 <= response.status_code < 300
        )
    )
    if issue_csrf_cookie:
        token = secrets.token_urlsafe(32)
        response.set_cookie(
            key=CSRF_COOKIE_NAME,
            value=token,
            max_age=ADMIN_SESSION_MAX_AGE,
            httponly=False,
            secure=ADMIN_SESSION_HTTPS_ONLY,
            samesite=ADMIN_SESSION_SAME_SITE,
        )

    return response


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    started_at = perf_counter()
    raw_header_request_id = request.headers.get("x-request-id")
    try:
        request.state.http_request_id = UUID(raw_header_request_id) if raw_header_request_id else uuid4()
    except ValueError:
        request.state.http_request_id = uuid4()

    header_trace_id = request.headers.get("x-trace-id")
    if header_trace_id:
        request.state.trace_id = header_trace_id

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = round((perf_counter() - started_at) * 1000.0, 3)
        log_event(
            logger,
            logging.ERROR,
            "http_request_failed",
            exc_info=True,
            request_id=_effective_request_id(request),
            http_request_id=request.state.http_request_id,
            trace_id=getattr(request.state, "trace_id", None),
            method=request.method,
            path=request.url.path,
            query_string=request.url.query or None,
            duration_ms=duration_ms,
            client_ip=_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            auth_username=getattr(request.state, "auth_username", None),
            auth_role=getattr(request.state, "auth_role", None),
            subject_id=getattr(request.state, "subject_id", None),
            session_id=getattr(request.state, "session_id", None),
        )
        raise

    duration_ms = round((perf_counter() - started_at) * 1000.0, 3)
    effective_request_id = _effective_request_id(request)
    response.headers["X-Request-ID"] = str(effective_request_id or request.state.http_request_id)

    status_code = response.status_code
    log_level = logging.INFO
    if status_code >= 500:
        log_level = logging.ERROR
    elif status_code >= 400:
        log_level = logging.WARNING

    log_event(
        logger,
        log_level,
        "http_request_completed",
        request_id=effective_request_id,
        http_request_id=request.state.http_request_id,
        trace_id=getattr(request.state, "trace_id", None),
        method=request.method,
        path=request.url.path,
        query_string=request.url.query or None,
        status_code=status_code,
        duration_ms=duration_ms,
        client_ip=_client_ip(request),
        user_agent=request.headers.get("user-agent"),
        auth_username=getattr(request.state, "auth_username", None),
        auth_role=getattr(request.state, "auth_role", None),
        subject_id=getattr(request.state, "subject_id", None),
        session_id=getattr(request.state, "session_id", None),
    )
    return response


def _basic_auth_challenge(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Basic"},
    )


def _resolve_authenticated_principal(
    request: Request,
    credentials: HTTPBasicCredentials | None = Depends(security),
) -> AuthenticatedPrincipal | None:
    if not auth_required():
        return None

    if credentials is None:
        log_event(
            logger,
            logging.WARNING,
            "authentication_required",
            method=request.method,
            path=request.url.path,
        )
        raise _basic_auth_challenge("Authentication required.")

    try:
        principal = authenticate_basic_credentials(
            credentials.username,
            credentials.password,
        )
    except RuntimeAuthenticationError as exc:
        log_event(
            logger,
            logging.ERROR,
            "authentication_backend_failure",
            exc_info=True,
            method=request.method,
            path=request.url.path,
            auth_username=credentials.username.strip() or None,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication backend error.",
        ) from exc

    if principal is None:
        log_event(
            logger,
            logging.WARNING,
            "authentication_failed",
            method=request.method,
            path=request.url.path,
            auth_username=credentials.username.strip() or None,
        )
        raise _basic_auth_challenge("Invalid username or password.")
    _set_request_state(request, principal=principal)
    return principal


def _admin_session_data(request: Request) -> dict[str, Any]:
    raw_session = request.session.get(ADMIN_SESSION_KEY)
    if isinstance(raw_session, Mapping):
        return dict(raw_session)
    return {}


def _store_admin_session(request: Request, principal: AuthenticatedPrincipal) -> None:
    request.session[ADMIN_SESSION_KEY] = {
        "auth_account_id": str(principal.auth_account_id),
        "username": principal.login_username,
        "role": principal.role,
    }


def _clear_admin_session(request: Request) -> None:
    request.session.pop(ADMIN_SESSION_KEY, None)


def _resolve_admin_session_principal(request: Request) -> AuthenticatedPrincipal:
    session_data = _admin_session_data(request)
    auth_account_id = session_data.get("auth_account_id")
    if not auth_account_id:
        log_event(
            logger,
            logging.WARNING,
            "admin_session_missing",
            method=request.method,
            path=request.url.path,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin authentication required.",
        )

    try:
        principal = load_authenticated_principal_by_account_id(auth_account_id)
    except RuntimeAuthenticationError as exc:
        log_event(
            logger,
            logging.ERROR,
            "admin_session_lookup_failed",
            exc_info=True,
            method=request.method,
            path=request.url.path,
            auth_account_id=auth_account_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication backend error.",
        ) from exc

    if principal is None:
        _clear_admin_session(request)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin session has expired.",
        )
    if not principal.is_admin:
        _clear_admin_session(request)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access is required.",
        )

    _set_request_state(request, principal=principal)
    return principal


def _normalize_runtime_request_for_principal(
    req: RuntimeSessionRequest,
    principal: AuthenticatedPrincipal | None,
) -> RuntimeSessionRequest:
    if principal is None or principal.is_admin:
        return req

    subject_key = normalize_subject_for_principal(principal, req.metadata.subject_id)
    if subject_key == req.metadata.subject_id:
        return req
    return req.model_copy(
        update={
            "metadata": req.metadata.model_copy(
                update={"subject_id": subject_key},
            )
        }
    )


def _normalize_feedback_request_for_principal(
    req: PredictionFeedbackRequest,
    principal: AuthenticatedPrincipal | None,
) -> PredictionFeedbackRequest:
    if principal is None or principal.is_admin:
        return req

    subject_key = normalize_subject_for_principal(principal, req.subject_id)
    if subject_key == req.subject_id:
        return req
    return req.model_copy(update={"subject_id": subject_key})


def _annotation_source_for_principal(
    principal: AuthenticatedPrincipal | None,
) -> SessionAnnotationSource:
    if principal is not None and principal.is_admin:
        return SessionAnnotationSource.admin
    return SessionAnnotationSource.mobile


def _request_id_from_runtime_request(req: RuntimeSessionRequest) -> UUID | None:
    if req.request_context is None:
        return None
    return req.request_context.request_id


def _request_id_from_feedback_request(req: PredictionFeedbackRequest) -> UUID | None:
    if req.request_context is None:
        return None
    return req.request_context.request_id


def _build_error_response(
        *,
        request_id: UUID | None,
        error_code: str,
        message: str,
        details: dict[str, str] | None = None,
) -> dict[str, Any]:
    payload = ErrorResponse(
        request_id=request_id,
        error_code=error_code,
        message=message,
        details=details or {},
    )
    return payload.model_dump(mode="json")


def _summarize_validation_errors(
    errors: list[dict[str, Any]],
    *,
    limit: int = 20,
) -> list[dict[str, str]]:
    summary: list[dict[str, str]] = []
    for error in errors[:limit]:
        raw_loc = error.get("loc", ())
        if isinstance(raw_loc, (list, tuple)):
            loc_parts = [str(part) for part in raw_loc if part != "body"]
        else:
            loc_parts = [str(raw_loc)]
        summary.append(
            {
                "field": ".".join(loc_parts) or "request",
                "type": str(error.get("type", "validation_error")),
                "message": str(error.get("msg", "Invalid request value.")),
            }
        )
    return summary


def _persistence_error_fields(exc: RuntimePersistenceError) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    operation = getattr(exc, "operation", None)
    if operation:
        fields["persistence_operation"] = operation
    diagnostics = getattr(exc, "diagnostics", None)
    if isinstance(diagnostics, Mapping):
        fields.update(diagnostics)
    return fields


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
) -> JSONResponse:
    request_id = _effective_request_id(request)
    validation_errors = _summarize_validation_errors(exc.errors())
    log_event(
        logger,
        logging.WARNING,
        "request_validation_failed",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        error_count=len(exc.errors()),
        validation_errors=validation_errors,
    )
    payload = _build_error_response(
        request_id=request_id,
        error_code="validation_error",
        message="Request validation failed.",
        details={"errors": json.dumps(validation_errors, ensure_ascii=True)},
    )
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=payload)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    message = str(exc.detail) if exc.detail is not None else "HTTP error."
    payload = _build_error_response(
        request_id=_effective_request_id(request),
        error_code=f"http_{exc.status_code}",
        message=message,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=payload,
        headers=exc.headers,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = _effective_request_id(request)
    log_event(
        logger,
        logging.ERROR,
        "unhandled_api_error",
        exc_info=True,
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    )
    payload = _build_error_response(
        request_id=request_id,
        error_code="internal_server_error",
        message="An unexpected internal server error occurred.",
    )
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=payload)


def _validate_artifact_paths() -> None:
    missing: list[str] = []

    if not HAR_ARTIFACT_PATH.is_file():
        missing.append(str(HAR_ARTIFACT_PATH))
    if not FALL_ARTIFACT_PATH.is_file():
        missing.append(str(FALL_ARTIFACT_PATH))

    if missing:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Required model artifact(s) not found: {', '.join(missing)}",
        )


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if pd.isna(result):
        return None
    return result


def _to_int_or_default(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    try:
        return bool(value)
    except Exception:
        return False


def _to_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _to_dataframe(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()

    if isinstance(value, pd.DataFrame):
        return value.copy()

    if isinstance(value, list):
        return pd.DataFrame(value)

    if isinstance(value, tuple):
        return pd.DataFrame(list(value))

    if isinstance(value, Mapping):
        try:
            return pd.DataFrame([dict(value)])
        except Exception:
            return pd.DataFrame()

    try:
        return pd.DataFrame(value)
    except Exception:
        return pd.DataFrame()


def _clean_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return str(value)


def _clean_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def _has_gyro(df: pd.DataFrame) -> bool:
    gyro_cols = [col for col in ("gx", "gy", "gz") if col in df.columns]
    if not gyro_cols:
        return False
    return df[gyro_cols].notna().any().any()


def _estimate_sampling_rate_hz(df: pd.DataFrame) -> float | None:
    if df.empty or "timestamp" not in df.columns or len(df) < 2:
        return None

    start_ts = _to_float_or_none(df["timestamp"].iloc[0])
    end_ts = _to_float_or_none(df["timestamp"].iloc[-1])

    if start_ts is None or end_ts is None:
        return None

    duration = end_ts - start_ts
    if duration <= 0:
        return None

    return float((len(df) - 1) / duration)


def _runtime_session_duration_seconds(df: pd.DataFrame) -> float | None:
    if df.empty or "timestamp" not in df.columns or len(df) < 2:
        return None

    start_ts = _to_float_or_none(df["timestamp"].iloc[0])
    end_ts = _to_float_or_none(df["timestamp"].iloc[-1])
    if start_ts is None or end_ts is None:
        return None

    duration = end_ts - start_ts
    if duration <= 0:
        return None
    return float(duration)


def _validate_runtime_session_duration(df: pd.DataFrame) -> None:
    duration_seconds = _runtime_session_duration_seconds(df)
    if duration_seconds is not None and duration_seconds >= MIN_RUNTIME_SESSION_DURATION_SECONDS:
        return

    actual_duration = (
        f"{duration_seconds:.3f}s" if duration_seconds is not None else "unavailable duration"
    )
    raise ValueError(
        "Runtime inference requires at least "
        f"{MIN_RUNTIME_SESSION_DURATION_SECONDS:.1f} seconds of sensor data "
        f"(got {actual_duration} across {len(df)} samples)."
    )


def _request_to_dataframe(req: RuntimeSessionRequest) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for sample in req.samples:
        rows.append(
            {
                "timestamp": sample.timestamp,
                "ax": sample.ax,
                "ay": sample.ay,
                "az": sample.az,
                "gx": sample.gx,
                "gy": sample.gy,
                "gz": sample.gz,
                "dataset_name": req.metadata.dataset_name,
                "subject_id": req.metadata.subject_id,
                "session_id": req.metadata.session_id,
                "task_type": req.metadata.task_type.value,
                "placement": req.metadata.placement.value,
                "sampling_rate_hz": req.metadata.sampling_rate_hz,
                "source_file": req.metadata.source_type.value,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No samples were provided.")

    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    df = _normalise_runtime_timestamps(df)

    for col in ("gx", "gy", "gz"):
        if col not in df.columns:
            df[col] = pd.NA

    return df


def _normalise_runtime_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Remove a phantom first sample and keep API timestamps session-relative."""

    if df.empty or "timestamp" not in df.columns:
        return df

    cleaned = drop_phantom_leading_samples(df, timestamp_col="timestamp").copy()
    if cleaned.empty:
        return cleaned

    first_ts = _to_float_or_none(cleaned["timestamp"].iloc[0])
    if first_ts is None:
        return cleaned

    cleaned["timestamp"] = pd.to_numeric(cleaned["timestamp"], errors="coerce") - first_ts
    return cleaned.reset_index(drop=True)


def _normalise_public_placement_state(raw_state: Any) -> str:
    value = str(raw_state).strip().lower() if raw_state is not None else "unknown"
    allowed = {"pocket", "hand", "desk", "bag", "unknown"}
    return value if value in allowed else "unknown"


def _build_source_summary(
        result: Any,
        req: RuntimeSessionRequest,
        input_df: pd.DataFrame,
) -> SourceSummary:
    raw = _to_mapping(getattr(result, "source_summary", None))

    duration = _to_float_or_none(raw.get("session_duration_seconds"))
    if duration is None and len(input_df) >= 2:
        start_ts = _to_float_or_none(input_df["timestamp"].iloc[0])
        end_ts = _to_float_or_none(input_df["timestamp"].iloc[-1])
        if start_ts is not None and end_ts is not None and end_ts >= start_ts:
            duration = end_ts - start_ts

    estimated_rate = _to_float_or_none(raw.get("estimated_sampling_rate_hz"))
    if estimated_rate is None:
        estimated_rate = req.metadata.sampling_rate_hz or _estimate_sampling_rate_hz(input_df)

    return SourceSummary(
        input_sample_count=_to_int_or_default(raw.get("input_sample_count"), len(input_df)),
        session_duration_seconds=duration,
        has_gyro=_to_bool(raw.get("has_gyro")) if "has_gyro" in raw else _has_gyro(input_df),
        estimated_sampling_rate_hz=estimated_rate,
        source_type=req.metadata.source_type,
        device_platform=req.metadata.device_platform,
    )


def _build_placement_summary(result: Any) -> PlacementSummary:
    raw = _to_mapping(getattr(result, "placement_summary", None))
    state_counts_raw = raw.get("state_counts", {})
    state_counts: dict[str, int] = {}

    if isinstance(state_counts_raw, Mapping):
        for key, value in state_counts_raw.items():
            state_counts[str(key)] = max(0, _to_int_or_default(value))

    public_state = _normalise_public_placement_state(raw.get("placement_state", "unknown"))

    return PlacementSummary(
        placement_state=public_state,
        placement_confidence=_to_float_or_none(raw.get("placement_confidence")),
        state_fraction=_to_float_or_none(raw.get("state_fraction")),
        state_counts=state_counts,
    )


def _build_har_summary(result: Any) -> HarSummary:
    raw = _to_mapping(getattr(result, "har_summary", None))
    har_windows_df = _to_dataframe(getattr(result, "har_windows", None))

    label_counts: dict[str, int] = {}
    counts_mapping = raw.get("label_counts") or raw.get("predicted_label_counts")
    if isinstance(counts_mapping, Mapping):
        for key, value in dict(counts_mapping).items():
            label_counts[str(key)] = max(0, _to_int_or_default(value))

    if not label_counts and not har_windows_df.empty and "predicted_label" in har_windows_df.columns:
        counts = har_windows_df["predicted_label"].astype(str).value_counts(dropna=False)
        label_counts = {str(k): int(v) for k, v in counts.to_dict().items()}

    total_windows = _to_int_or_default(raw.get("total_windows"), len(har_windows_df))
    top_label = raw.get("top_label")
    top_fraction = _to_float_or_none(raw.get("top_label_fraction"))

    if (top_label is None or top_fraction is None) and label_counts:
        sorted_counts = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)
        top_label = sorted_counts[0][0]
        if total_windows > 0:
            top_fraction = float(sorted_counts[0][1] / total_windows)

    return HarSummary(
        top_label=str(top_label) if top_label is not None else None,
        top_label_fraction=top_fraction,
        label_counts=label_counts,
        total_windows=total_windows,
    )


def _build_fall_summary(result: Any) -> FallSummary:
    raw = _to_mapping(getattr(result, "fall_summary", None))
    fall_windows_df = _to_dataframe(getattr(result, "fall_windows", None))
    grouped_events_df = _to_dataframe(getattr(result, "grouped_fall_events", None))

    top_prob = _to_float_or_none(raw.get("top_fall_probability"))
    if top_prob is None:
        if not grouped_events_df.empty and "peak_probability" in grouped_events_df.columns:
            top_prob = _to_float_or_none(
                pd.to_numeric(grouped_events_df["peak_probability"], errors="coerce").max()
            )
        elif not fall_windows_df.empty and "predicted_probability" in fall_windows_df.columns:
            top_prob = _to_float_or_none(
                pd.to_numeric(fall_windows_df["predicted_probability"], errors="coerce").max()
            )

    mean_prob = _to_float_or_none(raw.get("mean_fall_probability"))
    if mean_prob is None and not fall_windows_df.empty and "predicted_probability" in fall_windows_df.columns:
        mean_prob = _to_float_or_none(
            pd.to_numeric(fall_windows_df["predicted_probability"], errors="coerce").mean()
        )

    positive_window_count = _to_int_or_default(raw.get("positive_window_count"))
    if positive_window_count == 0 and not fall_windows_df.empty:
        if "predicted_is_fall" in fall_windows_df.columns:
            positive_window_count = int(fall_windows_df["predicted_is_fall"].fillna(False).astype(bool).sum())
        elif "predicted_probability" in fall_windows_df.columns:
            positive_window_count = int(
                (pd.to_numeric(fall_windows_df["predicted_probability"], errors="coerce") >= 0.5).sum()
            )

    grouped_event_count = _to_int_or_default(raw.get("grouped_event_count"), len(grouped_events_df))

    likely_fall_detected = _to_bool(raw.get("likely_fall_detected"))
    if not likely_fall_detected:
        likely_fall_detected = grouped_event_count > 0 and (top_prob is not None and top_prob >= 0.75)

    return FallSummary(
        likely_fall_detected=likely_fall_detected,
        positive_window_count=positive_window_count,
        grouped_event_count=grouped_event_count,
        top_fall_probability=top_prob,
        mean_fall_probability=mean_prob,
    )


def _build_vulnerability_summary(result: Any) -> VulnerabilitySummary:
    raw = _to_mapping(getattr(result, "vulnerability_summary", None))
    vulnerability_windows_df = _to_dataframe(getattr(result, "vulnerability_windows", None))

    def _counts_from_raw(key: str) -> dict[str, int]:
        mapping = raw.get(key)
        if not isinstance(mapping, Mapping):
            return {}
        return {str(k): max(0, _to_int_or_default(v)) for k, v in dict(mapping).items()}

    fall_event_state_counts = _counts_from_raw("fall_event_state_counts")
    if not fall_event_state_counts and not vulnerability_windows_df.empty and "fall_event_state" in vulnerability_windows_df.columns:
        counts = vulnerability_windows_df["fall_event_state"].astype(str).value_counts(dropna=False)
        fall_event_state_counts = {str(k): int(v) for k, v in counts.to_dict().items()}

    vulnerability_level_counts = _counts_from_raw("vulnerability_level_counts")
    if (
        not vulnerability_level_counts
        and not vulnerability_windows_df.empty
        and "vulnerability_level" in vulnerability_windows_df.columns
    ):
        counts = vulnerability_windows_df["vulnerability_level"].astype(str).value_counts(dropna=False)
        vulnerability_level_counts = {str(k): int(v) for k, v in counts.to_dict().items()}

    monitoring_state_counts = _counts_from_raw("monitoring_state_counts")
    if (
        not monitoring_state_counts
        and not vulnerability_windows_df.empty
        and "monitoring_state" in vulnerability_windows_df.columns
    ):
        counts = vulnerability_windows_df["monitoring_state"].astype(str).value_counts(dropna=False)
        monitoring_state_counts = {str(k): int(v) for k, v in counts.to_dict().items()}

    top_score = _to_float_or_none(raw.get("top_vulnerability_score"))
    mean_score = _to_float_or_none(raw.get("mean_vulnerability_score"))
    latest_score = _to_float_or_none(raw.get("latest_vulnerability_score"))
    latest_level = raw.get("latest_vulnerability_level")
    latest_state = raw.get("latest_monitoring_state")
    latest_event_state = raw.get("latest_fall_event_state")

    if not vulnerability_windows_df.empty:
        sorted_df = vulnerability_windows_df.copy()
        sort_cols = [c for c in ["session_id", "start_ts", "end_ts", "midpoint_ts"] if c in sorted_df.columns]
        if sort_cols:
            sorted_df = sorted_df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
        latest_row = sorted_df.iloc[-1]
        if latest_score is None:
            latest_score = _to_float_or_none(latest_row.get("vulnerability_score"))
        if latest_level is None:
            latest_level = latest_row.get("vulnerability_level")
        if latest_state is None:
            latest_state = latest_row.get("monitoring_state")
        if latest_event_state is None:
            latest_event_state = latest_row.get("fall_event_state")

        if top_score is None and "vulnerability_score" in vulnerability_windows_df.columns:
            top_score = _to_float_or_none(
                pd.to_numeric(vulnerability_windows_df["vulnerability_score"], errors="coerce").max()
            )
        if mean_score is None and "vulnerability_score" in vulnerability_windows_df.columns:
            mean_score = _to_float_or_none(
                pd.to_numeric(vulnerability_windows_df["vulnerability_score"], errors="coerce").mean()
            )

    return VulnerabilitySummary(
        enabled=_to_bool(raw.get("enabled")) if raw else not vulnerability_windows_df.empty,
        event_profile=str(raw.get("event_profile")) if raw.get("event_profile") is not None else None,
        vulnerability_profile=(
            str(raw.get("vulnerability_profile")) if raw.get("vulnerability_profile") is not None else None
        ),
        window_count=_to_int_or_default(raw.get("window_count"), len(vulnerability_windows_df)),
        session_count=_to_int_or_default(raw.get("session_count")),
        alert_worthy_window_count=_to_int_or_default(raw.get("alert_worthy_window_count")),
        fall_event_state_counts=fall_event_state_counts,
        vulnerability_level_counts=vulnerability_level_counts,
        monitoring_state_counts=monitoring_state_counts,
        top_vulnerability_score=top_score,
        mean_vulnerability_score=mean_score,
        latest_vulnerability_score=latest_score,
        latest_vulnerability_level=str(latest_level) if latest_level is not None else None,
        latest_monitoring_state=str(latest_state) if latest_state is not None else None,
        latest_fall_event_state=str(latest_event_state) if latest_event_state is not None else None,
        top_vulnerability_reasons=[str(item) for item in raw.get("top_vulnerability_reasons", [])]
        if isinstance(raw.get("top_vulnerability_reasons"), list)
        else [],
    )


def _derive_alert_summary(
        result: Any,
        placement_summary: PlacementSummary,
        har_summary: HarSummary,
        fall_summary: FallSummary,
        vulnerability_summary: VulnerabilitySummary,
) -> RuntimeAlertSummary:
    raw_placement_summary = _to_mapping(getattr(result, "placement_summary", None))
    raw_placement_state = str(raw_placement_summary.get("placement_state", "unknown")).strip().lower()

    likely_fall = fall_summary.likely_fall_detected or vulnerability_summary.alert_worthy_window_count > 0
    top_prob = fall_summary.top_fall_probability
    event_count = fall_summary.grouped_event_count
    top_vulnerability_score = vulnerability_summary.top_vulnerability_score
    latest_level = (vulnerability_summary.latest_vulnerability_level or "").strip().lower()
    latest_state = (vulnerability_summary.latest_monitoring_state or "").strip().lower()
    latest_fall_event_state = vulnerability_summary.latest_fall_event_state
    placement_reduced_confidence = raw_placement_state in {"repositioning", "on_surface"}

    # HAR-gated fall suppression: a phone that HAR says has been static for
    # the bulk of the session cannot have fallen. Treat that the same way as
    # placement_reduced_confidence — downgrade one step, don't veto outright.
    # Gate is intentionally cautious (0.6 majority) because noisy HAR on a few
    # windows shouldn't suppress a genuine fall.
    har_top = (har_summary.top_label or "").strip().lower()
    har_top_fraction = float(har_summary.top_label_fraction or 0.0)
    har_static_reduced = har_top == "static" and har_top_fraction >= 0.6
    # Session-level mirror of fusion.vulnerability_score._dynamic_har_attenuation:
    # if the dominant HAR label is confident locomotion, the session-level
    # alert should match the per-window attenuation that already capped the
    # vulnerability score. Without this, walking/stairs sessions can still
    # surface as warning_level="high" via the likely_fall / top_prob branches
    # below even though the vulnerability layer has already capped them.
    har_locomotion_reduced = (
        har_top in {"walking", "running", "stairs", "dynamic", "locomotion"}
        and har_top_fraction >= 0.6
    )
    har_reduced_confidence = har_static_reduced or har_locomotion_reduced
    reduced_confidence = placement_reduced_confidence or har_reduced_confidence
    if placement_reduced_confidence:
        reduced_reason = "placement state suggests reduced confidence"
    elif har_locomotion_reduced:
        reduced_reason = (
            f"HAR indicates the session was predominantly {har_top}, "
            "so fall-like windows were attenuated"
        )
    else:
        reduced_reason = "HAR indicates the phone was static for most of the session"

    if latest_state == "high_risk_vulnerable_state" or latest_level == "high":
        if reduced_confidence:
            warning_level = "medium"
            message = f"High-risk vulnerability pattern detected, but {reduced_reason}."
        else:
            warning_level = "high"
            message = "High-risk vulnerability pattern detected. Review immediately."
    elif latest_state in {"vulnerable_state", "suspected_incident"} or latest_level == "medium":
        warning_level = "medium"
        message = "Elevated vulnerability pattern detected. Review the session."
    elif reduced_confidence and likely_fall:
        warning_level = "medium"
        message = f"Fall-like event detected, but {reduced_reason}."
    elif likely_fall:
        warning_level = "high"
        message = "High-confidence fall-like event detected. Review immediately."
    elif top_prob is not None and top_prob >= 0.50:
        warning_level = "medium"
        message = "Moderate fall risk detected. Review the session."
    elif event_count > 0:
        warning_level = "low"
        message = "Fall-like motion detected, but confidence is limited."
    else:
        warning_level = "none"
        message = "No strong fall-like event detected."

    return RuntimeAlertSummary(
        warning_level=warning_level,
        likely_fall_detected=likely_fall,
        top_har_label=har_summary.top_label,
        top_har_fraction=har_summary.top_label_fraction,
        grouped_fall_event_count=fall_summary.grouped_event_count,
        top_fall_probability=fall_summary.top_fall_probability,
        top_vulnerability_score=top_vulnerability_score,
        latest_vulnerability_level=vulnerability_summary.latest_vulnerability_level,
        latest_monitoring_state=vulnerability_summary.latest_monitoring_state,
        latest_fall_event_state=str(latest_fall_event_state) if latest_fall_event_state is not None else None,
        recommended_message=message,
    )


def _build_debug_summary(
        result: Any,
        req: RuntimeSessionRequest,
        input_df: pd.DataFrame,
) -> RuntimeDebugSummary:
    raw = _to_mapping(getattr(result, "debug_summary", None))
    processing_notes_raw = raw.get("processing_notes", [])
    processing_notes = [str(note) for note in processing_notes_raw] if isinstance(processing_notes_raw, list) else []

    estimated_rate = _to_float_or_none(raw.get("estimated_sampling_rate_hz"))
    if estimated_rate is None:
        estimated_rate = req.metadata.sampling_rate_hz or _estimate_sampling_rate_hz(input_df)

    return RuntimeDebugSummary(
        input_sample_count=_to_int_or_default(raw.get("input_sample_count"), len(input_df)),
        has_gyro=_to_bool(raw.get("has_gyro")) if "has_gyro" in raw else _has_gyro(input_df),
        estimated_sampling_rate_hz=estimated_rate,
        runtime_mode=req.metadata.runtime_mode,
        processing_notes=processing_notes,
    )


def _artifact_name_and_version(path: Path) -> tuple[str | None, str | None]:
    if not path.name:
        return None, None

    stem = path.stem
    parent = path.parent.name if path.parent else ""

    if stem.lower() in {"model", "artifact"} and parent:
        return parent, parent

    if parent and parent not in {"har", "fall", "artifacts"}:
        return stem, parent

    return stem, None


def _build_model_info() -> ModelInfo:
    har_name, har_version = _artifact_name_and_version(HAR_ARTIFACT_PATH)
    fall_name, fall_version = _artifact_name_and_version(FALL_ARTIFACT_PATH)

    return ModelInfo(
        har_model_name=har_name,
        har_model_version=har_version,
        fall_model_name=fall_name,
        fall_model_version=fall_version,
        api_version=API_VERSION,
    )


def _to_har_models(df: pd.DataFrame) -> list[HarWindowPrediction]:
    if df.empty:
        return []

    keep = [
        col
        for col in (
            "window_id",
            "start_ts",
            "end_ts",
            "midpoint_ts",
            "predicted_label",
            "predicted_confidence",
        )
        if col in df.columns
    ]

    cleaned: list[HarWindowPrediction] = []
    for row in df[keep].to_dict(orient="records"):
        row["window_id"] = _clean_optional_str(row.get("window_id"))
        row["predicted_label"] = str(row.get("predicted_label", "unknown"))
        row["start_ts"] = _to_float_or_none(row.get("start_ts"))
        row["end_ts"] = _to_float_or_none(row.get("end_ts"))
        row["midpoint_ts"] = _to_float_or_none(row.get("midpoint_ts"))
        row["predicted_confidence"] = _to_float_or_none(row.get("predicted_confidence"))
        cleaned.append(HarWindowPrediction(**row))

    return cleaned


def _to_fall_models(df: pd.DataFrame) -> list[FallWindowPrediction]:
    if df.empty:
        return []

    keep = [
        col
        for col in (
            "window_id",
            "start_ts",
            "end_ts",
            "midpoint_ts",
            "predicted_label",
            "predicted_probability",
            "predicted_is_fall",
        )
        if col in df.columns
    ]

    cleaned: list[FallWindowPrediction] = []
    for row in df[keep].to_dict(orient="records"):
        row["window_id"] = _clean_optional_str(row.get("window_id"))
        row["predicted_label"] = str(row.get("predicted_label", "unknown"))
        row["start_ts"] = _to_float_or_none(row.get("start_ts"))
        row["end_ts"] = _to_float_or_none(row.get("end_ts"))
        row["midpoint_ts"] = _to_float_or_none(row.get("midpoint_ts"))
        row["predicted_probability"] = _to_float_or_none(row.get("predicted_probability"))
        row["predicted_is_fall"] = _clean_optional_bool(row.get("predicted_is_fall"))
        cleaned.append(FallWindowPrediction(**row))

    return cleaned


def _to_vulnerability_models(df: pd.DataFrame) -> list[VulnerabilityWindowPrediction]:
    if df.empty:
        return []

    cleaned: list[VulnerabilityWindowPrediction] = []
    for row in df.to_dict(orient="records"):
        fall_event_reasons = row.get("fall_event_reasons")
        if not isinstance(fall_event_reasons, list):
            fall_event_reasons = []

        vulnerability_reasons = row.get("vulnerability_reasons")
        if not isinstance(vulnerability_reasons, list):
            vulnerability_reasons = []

        state_machine_reasons = row.get("state_machine_reasons")
        if not isinstance(state_machine_reasons, list):
            state_machine_reasons = []

        fall_event_contributions = row.get("fall_event_contributions")
        if not isinstance(fall_event_contributions, Mapping):
            fall_event_contributions = {}

        vulnerability_contributions = row.get("vulnerability_contributions")
        if not isinstance(vulnerability_contributions, Mapping):
            vulnerability_contributions = {}

        state_machine_counters = row.get("state_machine_counters")
        if not isinstance(state_machine_counters, Mapping):
            state_machine_counters = {}

        cleaned.append(
            VulnerabilityWindowPrediction(
                window_id=_clean_optional_str(row.get("window_id")),
                start_ts=_to_float_or_none(row.get("start_ts")),
                end_ts=_to_float_or_none(row.get("end_ts")),
                midpoint_ts=_to_float_or_none(row.get("midpoint_ts")),
                har_label=_clean_optional_str(row.get("har_label")),
                har_confidence=_to_float_or_none(row.get("har_confidence")),
                fall_probability=_to_float_or_none(row.get("fall_probability")),
                fall_predicted_label=_clean_optional_str(row.get("fall_predicted_label")),
                fall_predicted_is_fall=_clean_optional_bool(row.get("fall_predicted_is_fall")),
                fall_event_state=_clean_optional_str(row.get("fall_event_state")),
                fall_event_confidence=_to_float_or_none(row.get("fall_event_confidence")),
                fall_event_reasons=[str(item) for item in fall_event_reasons],
                fall_event_contributions={
                    str(k): _to_float_or_none(v) or 0.0 for k, v in dict(fall_event_contributions).items()
                },
                vulnerability_level=_clean_optional_str(row.get("vulnerability_level")),
                vulnerability_score=_to_float_or_none(row.get("vulnerability_score")),
                vulnerability_reasons=[str(item) for item in vulnerability_reasons],
                vulnerability_contributions={
                    str(k): _to_float_or_none(v) or 0.0 for k, v in dict(vulnerability_contributions).items()
                },
                monitoring_state=_clean_optional_str(row.get("monitoring_state")),
                escalated=_to_bool(row.get("escalated")),
                deescalated=_to_bool(row.get("deescalated")),
                state_machine_reasons=[str(item) for item in state_machine_reasons],
                state_machine_counters={str(k): _to_int_or_default(v) for k, v in dict(state_machine_counters).items()},
            )
        )

    return cleaned


def _to_grouped_event_models(df: pd.DataFrame) -> list[GroupedFallEvent]:
    if df.empty:
        return []

    keep = [
        col
        for col in (
            "event_id",
            "event_start_ts",
            "event_end_ts",
            "event_duration_seconds",
            "n_positive_windows",
            "peak_probability",
            "mean_probability",
            "median_probability",
        )
        if col in df.columns
    ]

    cleaned: list[GroupedFallEvent] = []
    for row in df[keep].to_dict(orient="records"):
        row["event_id"] = str(row.get("event_id", ""))
        row["event_start_ts"] = _to_float_or_none(row.get("event_start_ts")) or 0.0
        row["event_end_ts"] = _to_float_or_none(row.get("event_end_ts")) or 0.0
        row["event_duration_seconds"] = _to_float_or_none(row.get("event_duration_seconds")) or 0.0
        row["n_positive_windows"] = _to_int_or_default(row.get("n_positive_windows"))
        row["peak_probability"] = _to_float_or_none(row.get("peak_probability"))
        row["mean_probability"] = _to_float_or_none(row.get("mean_probability"))
        row["median_probability"] = _to_float_or_none(row.get("median_probability"))
        cleaned.append(GroupedFallEvent(**row))

    return cleaned


def _to_combined_timeline_models(df: pd.DataFrame) -> list[CombinedTimelinePoint]:
    if df.empty:
        return []

    cleaned: list[CombinedTimelinePoint] = []
    for row in df.to_dict(orient="records"):
        timestamp = _to_float_or_none(row.get("timestamp"))
        if timestamp is None:
            timestamp = _to_float_or_none(row.get("midpoint_ts"))
        if timestamp is None:
            continue

        window_id = _clean_optional_str(row.get("window_id"))
        if window_id is None:
            window_id = _clean_optional_str(row.get("har_window_id")) or _clean_optional_str(
                row.get("fall_window_id")
            )

        har_label = _clean_optional_str(row.get("har_label"))
        if har_label is None:
            har_label = _clean_optional_str(row.get("har_predicted_label"))

        har_confidence = _to_float_or_none(row.get("har_confidence"))
        if har_confidence is None:
            har_confidence = _to_float_or_none(row.get("har_predicted_confidence"))

        fall_probability = _to_float_or_none(row.get("fall_probability"))
        if fall_probability is None:
            fall_probability = _to_float_or_none(row.get("fall_predicted_probability"))

        fall_detected = _clean_optional_bool(row.get("fall_detected"))
        if fall_detected is None:
            fall_detected = _clean_optional_bool(row.get("fall_predicted_is_fall"))

        cleaned.append(
            CombinedTimelinePoint(
                timestamp=timestamp,
                har_label=har_label,
                har_confidence=har_confidence,
                fall_probability=fall_probability,
                fall_detected=fall_detected,
                window_id=window_id,
            )
        )

    return cleaned


def _to_point_timeline_models(df: pd.DataFrame) -> list[PointTimelinePoint]:
    if df.empty:
        return []

    cleaned: list[PointTimelinePoint] = []
    for row in df.to_dict(orient="records"):
        midpoint_ts = _to_float_or_none(row.get("midpoint_ts"))
        if midpoint_ts is None:
            continue

        cleaned.append(
            PointTimelinePoint(
                midpoint_ts=midpoint_ts,
                activity_label=_clean_optional_str(
                    row.get("smoothed_activity_label") or row.get("activity_label")
                ),
                placement_label=_clean_optional_str(
                    row.get("smoothed_placement_label") or row.get("placement_label")
                ),
                activity_confidence=_to_float_or_none(row.get("activity_confidence")),
                placement_confidence=_to_float_or_none(row.get("placement_confidence")),
                fall_probability=_to_float_or_none(row.get("fall_probability")),
                elevated_fall=_clean_optional_bool(row.get("elevated_fall")),
            )
        )

    return cleaned


def _to_timeline_event_models(df: pd.DataFrame) -> list[TimelineEvent]:
    if df.empty:
        return []

    cleaned: list[TimelineEvent] = []
    for row in df.to_dict(orient="records"):
        related_ids = row.get("related_grouped_fall_event_ids")
        if not isinstance(related_ids, list):
            related_ids = []

        cleaned.append(
            TimelineEvent(
                event_id=str(row.get("event_id", "")),
                start_ts=_to_float_or_none(row.get("start_ts")) or 0.0,
                end_ts=_to_float_or_none(row.get("end_ts")) or 0.0,
                duration_seconds=_to_float_or_none(row.get("duration_seconds")) or 0.0,
                midpoint_ts=_to_float_or_none(row.get("midpoint_ts")),
                point_count=_to_int_or_default(row.get("point_count")),
                activity_label=str(row.get("activity_label", "unknown")),
                placement_label=str(row.get("placement_label", "unknown")),
                activity_confidence_mean=_to_float_or_none(row.get("activity_confidence_mean")),
                placement_confidence_mean=_to_float_or_none(row.get("placement_confidence_mean")),
                fall_probability_peak=_to_float_or_none(row.get("fall_probability_peak")),
                fall_probability_mean=_to_float_or_none(row.get("fall_probability_mean")),
                likely_fall=_to_bool(row.get("likely_fall")),
                event_kind=str(row.get("event_kind", "activity")),
                related_grouped_fall_event_ids=[str(item) for item in related_ids],
                description=str(row.get("description", "")),
            )
        )

    return cleaned


def _to_transition_event_models(df: pd.DataFrame) -> list[TransitionEvent]:
    if df.empty:
        return []

    cleaned: list[TransitionEvent] = []
    for row in df.to_dict(orient="records"):
        cleaned.append(
            TransitionEvent(
                transition_id=str(row.get("transition_id", "")),
                transition_ts=_to_float_or_none(row.get("transition_ts")) or 0.0,
                from_event_id=str(row.get("from_event_id", "")),
                to_event_id=str(row.get("to_event_id", "")),
                transition_kind=str(row.get("transition_kind", "transition")),
                from_activity_label=_clean_optional_str(row.get("from_activity_label")),
                to_activity_label=_clean_optional_str(row.get("to_activity_label")),
                from_placement_label=_clean_optional_str(row.get("from_placement_label")),
                to_placement_label=_clean_optional_str(row.get("to_placement_label")),
                description=str(row.get("description", "")),
            )
        )

    return cleaned


def _build_session_narrative_summary(result: Any) -> SessionNarrativeSummary | None:
    session_summaries_df = _to_dataframe(getattr(result, "session_summaries", None))
    if session_summaries_df.empty:
        return None

    row = session_summaries_df.iloc[0].to_dict()
    har_attenuation_label_raw = row.get("har_attenuation_label")
    har_attenuation_label = (
        str(har_attenuation_label_raw)
        if har_attenuation_label_raw not in (None, "")
        and not (isinstance(har_attenuation_label_raw, float) and pd.isna(har_attenuation_label_raw))
        else None
    )
    dominant_vulnerability_level_raw = row.get("dominant_vulnerability_level")
    dominant_vulnerability_level = (
        str(dominant_vulnerability_level_raw)
        if dominant_vulnerability_level_raw not in (None, "")
        and not (isinstance(dominant_vulnerability_level_raw, float) and pd.isna(dominant_vulnerability_level_raw))
        else None
    )
    return SessionNarrativeSummary(
        session_id=str(row.get("session_id", "")),
        dataset_name=str(row.get("dataset_name", "")),
        subject_id=str(row.get("subject_id", "")),
        total_duration_seconds=_to_float_or_none(row.get("total_duration_seconds")) or 0.0,
        event_count=_to_int_or_default(row.get("event_count")),
        transition_count=_to_int_or_default(row.get("transition_count")),
        fall_event_count=_to_int_or_default(row.get("fall_event_count")),
        dominant_activity_label=str(row.get("dominant_activity_label", "unknown")),
        dominant_placement_label=str(row.get("dominant_placement_label", "unknown")),
        highest_fall_probability=_to_float_or_none(row.get("highest_fall_probability")),
        summary_text=str(row.get("summary_text", "")),
        peak_vulnerability_score=_to_float_or_none(row.get("peak_vulnerability_score")),
        mean_vulnerability_score=_to_float_or_none(row.get("mean_vulnerability_score")),
        dominant_vulnerability_level=dominant_vulnerability_level,
        har_attenuation_applied=_to_bool(row.get("har_attenuation_applied")),
        har_attenuation_window_count=_to_int_or_default(row.get("har_attenuation_window_count")),
        har_attenuation_label=har_attenuation_label,
        har_attenuation_confidence_mean=_to_float_or_none(row.get("har_attenuation_confidence_mean")),
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")


def _default_feedback_target_type(req: PredictionFeedbackRequest) -> str:
    if req.target_type is not None:
        return req.target_type.value
    if req.window_id:
        return "window"
    if req.event_id:
        return "timeline_event"
    return "session"


log_event(
    logger,
    logging.INFO,
    "api_configuration_loaded",
    api_version=API_VERSION,
    log_level=LOG_LEVEL,
    persistence_enabled=persistence_enabled(),
    auth_required=auth_required(),
    self_service_registration_enabled=self_service_registration_enabled(),
    allowed_origins=ALLOWED_ORIGINS,
    har_artifact_path=HAR_ARTIFACT_PATH,
    fall_artifact_path=FALL_ARTIFACT_PATH,
    session_storage_dir=os.getenv("SESSION_STORAGE_DIR"),
    feedback_store_path=FEEDBACK_STORE_PATH,
    admin_web_enabled=ADMIN_WEB_ENABLED,
    admin_web_dist_path=ADMIN_WEB_DIST_PATH,
)


@app.get("/livez", response_model=HealthResponse)
def livez() -> HealthResponse:
    """Liveness probe: returns 200 as long as the process is up.

    Does NOT touch artifacts or the database — orchestrators should
    only restart the container when this fails.
    """
    return HealthResponse(
        status="ok",
        service_name="runtime-inference-api",
        version=API_VERSION,
    )


@app.get("/readyz", response_model=HealthResponse)
def readyz() -> HealthResponse:
    """Readiness probe: returns 503 when artifacts or the DB are not
    available. Load balancers should drain traffic on failure but
    NOT restart the container.
    """
    _validate_artifact_paths()
    if persistence_enabled():
        try:
            db_url = os.getenv("DATABASE_URL")
            if db_url:
                import psycopg

                with psycopg.connect(db_url, connect_timeout=2) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                        cur.fetchone()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Database unavailable: {type(exc).__name__}",
            ) from exc
    return HealthResponse(
        status="ok",
        service_name="runtime-inference-api",
        version=API_VERSION,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Back-compat alias for /readyz. New deployments should point
    liveness probes at /livez and readiness probes at /readyz.
    """
    return readyz()


@app.post("/v1/auth/register", response_model=SelfServiceRegistrationResponse)
def register_self_service_account(
    request: Request,
    req: SelfServiceRegistrationRequest,
) -> SelfServiceRegistrationResponse:
    if not self_service_registration_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Self-service registration is disabled.",
        )

    _enforce_registration_rate_limit(_client_ip(request))

    request_id = req.request_context.request_id if req.request_context is not None else None
    trace_id = req.request_context.trace_id if req.request_context is not None else None
    _set_request_state(
        request,
        request_id=request_id,
        trace_id=trace_id,
        subject_id=req.subject_id,
    )

    try:
        registered = register_self_service_user(
            req.username,
            req.password,
            subject_key=req.subject_id,
            display_name=req.display_name,
            device_platform=req.device_platform,
            device_model=req.device_model,
        )
    except RuntimeRegistrationConflictError as exc:
        log_event(
            logger,
            logging.WARNING,
            "self_service_registration_conflict",
            request_id=request_id,
            trace_id=trace_id,
            auth_username=req.username,
            subject_id=req.subject_id,
            error=str(exc),
            client_ip=_client_ip(request),
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except RuntimeRegistrationError as exc:
        log_event(
            logger,
            logging.ERROR,
            "self_service_registration_failed",
            exc_info=True,
            request_id=request_id,
            trace_id=trace_id,
            auth_username=req.username,
            subject_id=req.subject_id,
            error=str(exc),
            client_ip=_client_ip(request),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register runtime user.",
        ) from exc

    log_event(
        logger,
        logging.INFO,
        "self_service_registration_succeeded",
        request_id=request_id,
        trace_id=trace_id,
        auth_username=registered.login_username,
        auth_role="user",
        auth_user_id=registered.user_id,
        auth_subject_key=registered.subject_key,
        subject_id=registered.subject_key,
        created=registered.created,
        client_ip=_client_ip(request),
    )
    return SelfServiceRegistrationResponse(
        status="registered" if registered.created else "updated",
        username=registered.login_username,
        subject_id=registered.subject_key,
        display_name=registered.display_name,
        role="user",
        created=registered.created,
    )


@app.get("/v1/auth/me", response_model=AuthenticatedUserResponse)
def get_authenticated_user(
    request: Request,
    principal: AuthenticatedPrincipal | None = Depends(_resolve_authenticated_principal),
) -> AuthenticatedUserResponse:
    _set_request_state(request, principal=principal)
    if principal is None:
        return AuthenticatedUserResponse(
            status="anonymous",
            username=None,
            subject_id="anonymous_user",
            role="anonymous",
            auth_required=auth_required(),
        )

    log_event(
        logger,
        logging.INFO,
        "authenticated_user_loaded",
        auth_username=principal.login_username,
        auth_role=principal.role,
        auth_user_id=principal.user_id,
        auth_subject_key=principal.subject_key,
    )
    return AuthenticatedUserResponse(
        status="authenticated",
        username=principal.login_username,
        subject_id=principal.subject_key,
        role=principal.role,
        auth_required=auth_required(),
    )


@app.post("/v1/admin/auth/login", response_model=AdminAuthSessionResponse)
def admin_login(
    request: Request,
    req: AdminAuthLoginRequest,
) -> AdminAuthSessionResponse:
    try:
        principal = authenticate_basic_credentials(req.username, req.password)
    except RuntimeAuthenticationError as exc:
        log_event(
            logger,
            logging.ERROR,
            "admin_login_backend_failure",
            exc_info=True,
            method=request.method,
            path=request.url.path,
            auth_username=req.username,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication backend error.",
        ) from exc

    if principal is None:
        log_event(
            logger,
            logging.WARNING,
            "admin_login_failed",
            method=request.method,
            path=request.url.path,
            auth_username=req.username,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )
    if not principal.is_admin:
        log_event(
            logger,
            logging.WARNING,
            "admin_login_forbidden",
            method=request.method,
            path=request.url.path,
            auth_username=principal.login_username,
            auth_role=principal.role,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access is required.",
        )

    request.session.clear()
    _store_admin_session(request, principal)
    _set_request_state(request, principal=principal)
    log_event(
        logger,
        logging.INFO,
        "admin_login_succeeded",
        auth_username=principal.login_username,
        auth_role=principal.role,
        auth_user_id=principal.user_id,
        auth_subject_key=principal.subject_key,
    )
    return AdminAuthSessionResponse(
        status="authenticated",
        username=principal.login_username,
        subject_id=principal.subject_key,
        role=principal.role,
    )


@app.post("/v1/admin/auth/logout", response_model=AdminAuthSessionResponse)
def admin_logout(request: Request) -> AdminAuthSessionResponse:
    session_data = _admin_session_data(request)
    _clear_admin_session(request)
    log_event(
        logger,
        logging.INFO,
        "admin_logout_completed",
        auth_username=session_data.get("username"),
        auth_role=session_data.get("role"),
    )
    return AdminAuthSessionResponse(
        status="signed_out",
        username=None,
        subject_id=None,
        role="anonymous",
    )


@app.get("/v1/admin/auth/me", response_model=AdminAuthSessionResponse)
def get_admin_authenticated_user(
    request: Request,
    principal: AuthenticatedPrincipal = Depends(_resolve_admin_session_principal),
) -> AdminAuthSessionResponse:
    _set_request_state(request, principal=principal)
    return AdminAuthSessionResponse(
        status="authenticated",
        username=principal.login_username,
        subject_id=principal.subject_key,
        role=principal.role,
    )


@app.post("/v1/infer/session", response_model=RuntimeSessionResponse)
def infer_session(
    request: Request,
    req: RuntimeSessionRequest,
    principal: AuthenticatedPrincipal | None = Depends(_resolve_authenticated_principal),
) -> RuntimeSessionResponse:
    request_id = _request_id_from_runtime_request(req)
    trace_id = req.request_context.trace_id if req.request_context is not None else None
    _set_request_state(
        request,
        request_id=request_id,
        trace_id=trace_id,
        session_id=req.metadata.session_id,
        subject_id=req.metadata.subject_id,
        principal=principal,
    )

    try:
        req = _normalize_runtime_request_for_principal(req, principal)
    except RuntimeAuthorizationError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc

    _set_request_state(request, subject_id=req.metadata.subject_id)
    started_at = perf_counter()
    log_event(
        logger,
        logging.INFO,
        "runtime_inference_requested",
        request_id=request_id,
        trace_id=trace_id,
        session_id=req.metadata.session_id,
        subject_id=req.metadata.subject_id,
        dataset_name=req.metadata.dataset_name,
        source_type=req.metadata.source_type.value,
        device_platform=req.metadata.device_platform,
        sample_count=len(req.samples),
        include_vulnerability_windows=req.include_vulnerability_windows,
        include_grouped_fall_events=req.include_grouped_fall_events,
        include_timeline_events=req.include_timeline_events,
        include_transition_events=req.include_transition_events,
        **_principal_log_fields(principal),
    )

    try:
        _validate_artifact_paths()

        input_df = _request_to_dataframe(req)
        _validate_runtime_session_duration(input_df)

        artifacts = RuntimeArtifacts(
            har_artifact_path=HAR_ARTIFACT_PATH,
            fall_artifact_path=FALL_ARTIFACT_PATH,
        )
        config = RuntimeInferenceConfig()

        result = run_runtime_inference_from_dataframe(
            input_df,
            artifacts=artifacts,
            config=config,
        )

        har_windows_df = _to_dataframe(getattr(result, "har_windows", None))
        fall_windows_df = _to_dataframe(getattr(result, "fall_windows", None))
        vulnerability_windows_df = _to_dataframe(getattr(result, "vulnerability_windows", None))
        grouped_events_df = _to_dataframe(getattr(result, "grouped_fall_events", None))
        combined_timeline_df = _to_dataframe(getattr(result, "combined_timeline", None))
        point_timeline_df = _to_dataframe(getattr(result, "point_timeline", None))
        timeline_events_df = _to_dataframe(getattr(result, "timeline_events", None))
        transition_events_df = _to_dataframe(getattr(result, "transition_events", None))

        source_summary = _build_source_summary(result, req, input_df)
        placement_summary = _build_placement_summary(result)
        har_summary = _build_har_summary(result)
        fall_summary = _build_fall_summary(result)
        vulnerability_summary = _build_vulnerability_summary(result)
        alert_summary = _derive_alert_summary(
            result,
            placement_summary,
            har_summary,
            fall_summary,
            vulnerability_summary,
        )
        debug_summary = _build_debug_summary(result, req, input_df)
        model_info = _build_model_info()
        session_narrative_summary = _build_session_narrative_summary(result)
        narrative_summary = _to_mapping(getattr(result, "narrative_summary", None))

        response = RuntimeSessionResponse(
            request_id=request_id,
            session_id=req.metadata.session_id,
            source_summary=source_summary,
            placement_summary=placement_summary,
            har_summary=har_summary,
            fall_summary=fall_summary,
            vulnerability_summary=vulnerability_summary,
            alert_summary=alert_summary,
            debug_summary=debug_summary,
            model_info=model_info,
            grouped_fall_events=_to_grouped_event_models(grouped_events_df)
            if req.include_grouped_fall_events
            else [],
            har_windows=_to_har_models(har_windows_df) if req.include_har_windows else [],
            fall_windows=_to_fall_models(fall_windows_df) if req.include_fall_windows else [],
            vulnerability_windows=(
                _to_vulnerability_models(vulnerability_windows_df)
                if req.include_vulnerability_windows
                else []
            ),
            combined_timeline=_to_combined_timeline_models(combined_timeline_df)
            if req.include_combined_timeline
            else [],
            point_timeline=_to_point_timeline_models(point_timeline_df)
            if req.include_point_timeline
            else [],
            timeline_events=_to_timeline_event_models(timeline_events_df)
            if req.include_timeline_events
            else [],
            transition_events=_to_transition_event_models(transition_events_df)
            if req.include_transition_events
            else [],
            session_narrative_summary=session_narrative_summary,
            narrative_summary=narrative_summary,
        )

        if persistence_enabled():
            persistence_started_at = perf_counter()
            log_event(
                logger,
                logging.INFO,
                "runtime_session_persistence_started",
                request_id=request_id,
                trace_id=trace_id,
                session_id=req.metadata.session_id,
                subject_id=req.metadata.subject_id,
                sample_count=len(req.samples),
                timeline_event_count=len(response.timeline_events),
                transition_event_count=len(response.transition_events),
                grouped_fall_event_count=len(response.grouped_fall_events),
                **_principal_log_fields(principal),
            )
            persistence_result = persist_runtime_session(req, response)
            response = response.model_copy(
                update={
                    "persisted_user_id": persistence_result.user_id,
                    "persisted_session_id": persistence_result.app_session_id,
                    "persisted_inference_id": persistence_result.inference_id,
                }
            )
            log_event(
                logger,
                logging.INFO,
                "runtime_session_persisted",
                request_id=request_id,
                trace_id=trace_id,
                session_id=req.metadata.session_id,
                subject_id=req.metadata.subject_id,
                persisted_user_id=response.persisted_user_id,
                persisted_session_id=response.persisted_session_id,
                persisted_inference_id=response.persisted_inference_id,
                persistence_duration_ms=round(
                    (perf_counter() - persistence_started_at) * 1000.0,
                    3,
                ),
                **_principal_log_fields(principal),
            )

        log_event(
            logger,
            logging.INFO,
            "runtime_inference_completed",
            request_id=request_id,
            trace_id=trace_id,
            session_id=req.metadata.session_id,
            subject_id=req.metadata.subject_id,
            persisted_user_id=response.persisted_user_id,
            persisted_session_id=response.persisted_session_id,
            persisted_inference_id=response.persisted_inference_id,
            duration_ms=round((perf_counter() - started_at) * 1000.0, 3),
            sample_count=response.source_summary.input_sample_count,
            estimated_sampling_rate_hz=response.source_summary.estimated_sampling_rate_hz,
            likely_fall_detected=response.alert_summary.likely_fall_detected,
            warning_level=response.alert_summary.warning_level.value,
            top_har_label=response.alert_summary.top_har_label,
            top_fall_probability=response.alert_summary.top_fall_probability,
            top_vulnerability_score=response.alert_summary.top_vulnerability_score,
            latest_vulnerability_level=response.alert_summary.latest_vulnerability_level,
            latest_monitoring_state=response.alert_summary.latest_monitoring_state,
            grouped_fall_event_count=response.alert_summary.grouped_fall_event_count,
            timeline_event_count=len(response.timeline_events),
            transition_event_count=len(response.transition_events),
            har_model_name=response.model_info.har_model_name,
            har_model_version=response.model_info.har_model_version,
            fall_model_name=response.model_info.fall_model_name,
            fall_model_version=response.model_info.fall_model_version,
            **_principal_log_fields(principal),
        )

        return response

    except HTTPException:
        raise
    except RuntimePersistenceError as exc:
        log_event(
            logger,
            logging.ERROR,
            "runtime_inference_persistence_failed",
            exc_info=True,
            trace_id=trace_id,
            request_id=request_id,
            session_id=req.metadata.session_id,
            subject_id=req.metadata.subject_id,
            duration_ms=round((perf_counter() - started_at) * 1000.0, 3),
            error=str(exc),
            **_persistence_error_fields(exc),
            **_principal_log_fields(principal),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Runtime persistence error.",
        ) from exc
    except ValueError as exc:
        log_event(
            logger,
            logging.WARNING,
            "runtime_inference_bad_request",
            request_id=request_id,
            trace_id=trace_id,
            session_id=req.metadata.session_id,
            subject_id=req.metadata.subject_id,
            error=str(exc),
            **_principal_log_fields(principal),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        log_event(
            logger,
            logging.ERROR,
            "runtime_inference_failed",
            exc_info=True,
            request_id=request_id,
            trace_id=trace_id,
            session_id=req.metadata.session_id,
            subject_id=req.metadata.subject_id,
            duration_ms=round((perf_counter() - started_at) * 1000.0, 3),
            error=str(exc),
            **_principal_log_fields(principal),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal inference error.",
        ) from exc


@app.get("/v1/sessions", response_model=PersistedSessionListResponse)
def get_sessions(
    request: Request,
    subject_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    principal: AuthenticatedPrincipal | None = Depends(_resolve_authenticated_principal),
) -> PersistedSessionListResponse:
    _set_request_state(request, subject_id=subject_id, principal=principal)
    if not persistence_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database persistence is not configured.",
        )

    try:
        effective_subject_id = subject_id
        if principal is not None and not principal.is_admin and subject_id is not None:
            effective_subject_id = normalize_subject_for_principal(principal, subject_id)
        payload = list_persisted_sessions(
            subject_id=effective_subject_id,
            limit=limit,
            offset=offset,
            owner_user_id=owner_user_id_for_principal(principal)
            if principal is not None
            else None,
        )
        _set_request_state(request, subject_id=effective_subject_id)
        log_event(
            logger,
            logging.INFO,
            "persisted_sessions_listed",
            subject_id=effective_subject_id,
            limit=limit,
            offset=offset,
            returned_count=len(payload["sessions"]),
            total_count=payload["total_count"],
            **_principal_log_fields(principal),
        )
        return PersistedSessionListResponse(**payload)
    except RuntimeAuthorizationError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    except RuntimePersistenceError as exc:
        log_event(
            logger,
            logging.ERROR,
            "persisted_sessions_list_failed",
            exc_info=True,
            subject_id=subject_id,
            limit=limit,
            offset=offset,
            error=str(exc),
            **_persistence_error_fields(exc),
            **_principal_log_fields(principal),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load persisted sessions.",
        ) from exc


@app.get("/v1/sessions/{app_session_id}", response_model=PersistedSessionDetailResponse)
def get_session_detail(
    request: Request,
    app_session_id: UUID,
    principal: AuthenticatedPrincipal | None = Depends(_resolve_authenticated_principal),
) -> PersistedSessionDetailResponse:
    _set_request_state(request, session_id=str(app_session_id), principal=principal)
    if not persistence_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database persistence is not configured.",
        )

    try:
        payload = get_persisted_session_detail(
            app_session_id,
            owner_user_id=owner_user_id_for_principal(principal)
            if principal is not None
            else None,
        )
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Persisted session was not found.",
            )
        _set_request_state(
            request,
            subject_id=payload["session"]["subject_id"],
            session_id=payload["session"]["client_session_id"],
        )
        log_event(
            logger,
            logging.INFO,
            "persisted_session_detail_loaded",
            app_session_id=app_session_id,
            client_session_id=payload["session"]["client_session_id"],
            subject_id=payload["session"]["subject_id"],
            has_inference=payload["latest_inference"] is not None,
            feedback_count=len(payload["feedback"]),
            annotation_count=len(payload["annotations"]),
            **_principal_log_fields(principal),
        )
        return PersistedSessionDetailResponse(**payload)
    except HTTPException:
        raise
    except RuntimePersistenceError as exc:
        log_event(
            logger,
            logging.ERROR,
            "persisted_session_detail_failed",
            exc_info=True,
            app_session_id=app_session_id,
            error=str(exc),
            **_persistence_error_fields(exc),
            **_principal_log_fields(principal),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load persisted session detail.",
        ) from exc


@app.get("/v1/sessions/{app_session_id}/raw")
def get_session_raw_payload(
    request: Request,
    app_session_id: UUID,
    principal: AuthenticatedPrincipal | None = Depends(_resolve_authenticated_principal),
) -> FileResponse:
    _set_request_state(request, session_id=str(app_session_id), principal=principal)
    if not persistence_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database persistence is not configured.",
        )

    try:
        location = get_session_raw_storage_location(
            app_session_id,
            owner_user_id=owner_user_id_for_principal(principal)
            if principal is not None
            else None,
        )
    except RuntimePersistenceError as exc:
        log_event(
            logger,
            logging.ERROR,
            "persisted_session_raw_failed",
            exc_info=True,
            app_session_id=app_session_id,
            error=str(exc),
            **_persistence_error_fields(exc),
            **_principal_log_fields(principal),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load session raw payload.",
        ) from exc

    if location is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Persisted session was not found.",
        )

    raw_uri = (location.get("raw_storage_uri") or "").strip()
    if not raw_uri:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session has no stored raw payload.",
        )

    try:
        payload_path = Path(raw_uri).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stored raw payload path is invalid.",
        ) from exc

    storage_root = _resolve_session_storage_dir()
    try:
        payload_path.relative_to(storage_root)
    except ValueError:
        log_event(
            logger,
            logging.ERROR,
            "persisted_session_raw_outside_storage_root",
            app_session_id=app_session_id,
            payload_path=str(payload_path),
            storage_root=str(storage_root),
            **_principal_log_fields(principal),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stored raw payload is outside the configured storage root.",
        )

    if not payload_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stored raw payload file is missing.",
        )

    log_event(
        logger,
        logging.INFO,
        "persisted_session_raw_served",
        app_session_id=app_session_id,
        payload_bytes=location.get("raw_payload_bytes"),
        payload_sha256=location.get("raw_payload_sha256"),
        **_principal_log_fields(principal),
    )
    return FileResponse(
        path=str(payload_path),
        media_type=location.get("raw_storage_format") or "application/json",
        filename=payload_path.name,
    )


@app.get(
    "/v1/sessions/{app_session_id}/annotations",
    response_model=SessionAnnotationListResponse,
)
def get_session_annotations(
    request: Request,
    app_session_id: UUID,
    principal: AuthenticatedPrincipal | None = Depends(_resolve_authenticated_principal),
) -> SessionAnnotationListResponse:
    _set_request_state(request, session_id=str(app_session_id), principal=principal)
    if not persistence_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database persistence is not configured.",
        )

    try:
        payload = list_session_annotations(
            app_session_id,
            owner_user_id=owner_user_id_for_principal(principal)
            if principal is not None
            else None,
        )
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Persisted session was not found.",
            )
        log_event(
            logger,
            logging.INFO,
            "session_annotations_listed",
            app_session_id=app_session_id,
            annotation_count=len(payload["annotations"]),
            **_principal_log_fields(principal),
        )
        return SessionAnnotationListResponse(**payload)
    except HTTPException:
        raise
    except RuntimePersistenceError as exc:
        log_event(
            logger,
            logging.ERROR,
            "session_annotations_list_failed",
            exc_info=True,
            app_session_id=app_session_id,
            error=str(exc),
            **_persistence_error_fields(exc),
            **_principal_log_fields(principal),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load session annotations.",
        ) from exc


@app.post(
    "/v1/sessions/{app_session_id}/annotations",
    response_model=SessionAnnotationRecord,
)
def create_session_annotation_route(
    request: Request,
    app_session_id: UUID,
    req: SessionAnnotationRequest,
    principal: AuthenticatedPrincipal | None = Depends(_resolve_authenticated_principal),
) -> SessionAnnotationRecord:
    request_id = req.request_context.request_id if req.request_context is not None else None
    _set_request_state(
        request,
        request_id=request_id,
        session_id=str(app_session_id),
        principal=principal,
    )
    if not persistence_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database persistence is not configured.",
        )

    resolved_req = req
    if principal is not None and req.reviewer_id is None:
        resolved_req = req.model_copy(update={"reviewer_id": principal.login_username})

    resolved_source = req.source or _annotation_source_for_principal(principal)

    try:
        payload = create_session_annotation(
            app_session_id,
            resolved_req,
            source=resolved_source,
            auth_account_id=principal.auth_account_id if principal is not None else None,
            owner_user_id=owner_user_id_for_principal(principal)
            if principal is not None
            else None,
        )
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Persisted session was not found.",
            )
        log_event(
            logger,
            logging.INFO,
            "session_annotation_created",
            app_session_id=app_session_id,
            annotation_id=payload["annotation_id"],
            annotation_label=payload["label"],
            annotation_source=payload["source"],
            **_principal_log_fields(principal),
        )
        return SessionAnnotationRecord(**payload)
    except HTTPException:
        raise
    except RuntimePersistenceError as exc:
        log_event(
            logger,
            logging.ERROR,
            "session_annotation_create_failed",
            exc_info=True,
            app_session_id=app_session_id,
            error=str(exc),
            **_persistence_error_fields(exc),
            **_principal_log_fields(principal),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create session annotation.",
        ) from exc


@app.get("/v1/admin/overview", response_model=AdminOverviewResponse)
def get_admin_dashboard_overview(
    request: Request,
    principal: AuthenticatedPrincipal = Depends(_resolve_admin_session_principal),
) -> AdminOverviewResponse:
    _set_request_state(request, principal=principal)
    if not persistence_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database persistence is not configured.",
        )

    try:
        payload = get_admin_overview()
        log_event(
            logger,
            logging.INFO,
            "admin_overview_loaded",
            recent_session_count=len(payload["recent_sessions"]),
            total_sessions=payload["totals"]["sessions"],
            total_users=payload["totals"]["users"],
            **_principal_log_fields(principal),
        )
        return AdminOverviewResponse(**payload)
    except RuntimePersistenceError as exc:
        log_event(
            logger,
            logging.ERROR,
            "admin_overview_failed",
            exc_info=True,
            error=str(exc),
            **_persistence_error_fields(exc),
            **_principal_log_fields(principal),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load admin overview.",
        ) from exc


@app.get("/v1/admin/sessions", response_model=AdminSessionListResponse)
def get_admin_sessions(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
    search: str | None = None,
    subject_id: str | None = None,
    warning_level: str | None = None,
    device_platform: str | None = None,
    likely_fall: bool | None = None,
    status_filter: str | None = Query(default=None, alias="status"),
    dataset_name: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    sort_by: str = Query(default="created_at"),
    sort_dir: str = Query(default="desc"),
    principal: AuthenticatedPrincipal = Depends(_resolve_admin_session_principal),
) -> AdminSessionListResponse:
    _set_request_state(request, principal=principal, subject_id=subject_id)
    if not persistence_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database persistence is not configured.",
        )

    try:
        payload = list_admin_sessions(
            page=page,
            page_size=page_size,
            search=search,
            subject_id=subject_id,
            warning_level=warning_level,
            device_platform=device_platform,
            likely_fall=likely_fall,
            status=status_filter,
            dataset_name=dataset_name,
            date_from=date_from,
            date_to=date_to,
            sort_by=sort_by,
            sort_dir=sort_dir,
        )
        log_event(
            logger,
            logging.INFO,
            "admin_sessions_listed",
            page=page,
            page_size=page_size,
            returned_count=len(payload["sessions"]),
            total_count=payload["total_count"],
            search=search,
            subject_id=subject_id,
            warning_level=warning_level,
            device_platform=device_platform,
            likely_fall=likely_fall,
            status=status_filter,
            dataset_name=dataset_name,
            sort_by=sort_by,
            sort_dir=sort_dir,
            **_principal_log_fields(principal),
        )
        return AdminSessionListResponse(**payload)
    except RuntimePersistenceError as exc:
        log_event(
            logger,
            logging.ERROR,
            "admin_sessions_list_failed",
            exc_info=True,
            error=str(exc),
            page=page,
            page_size=page_size,
            **_persistence_error_fields(exc),
            **_principal_log_fields(principal),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load admin sessions.",
        ) from exc


def _normalize_admin_session_evidence_sections(raw_sections: str | None) -> list[str]:
    if raw_sections is None:
        return list(ADMIN_SESSION_EVIDENCE_SECTIONS)

    normalized = raw_sections.strip()
    if not normalized or normalized.lower() == "all":
        return list(ADMIN_SESSION_EVIDENCE_SECTIONS)

    resolved_sections: list[str] = []
    invalid_sections: list[str] = []
    for raw_section in normalized.split(","):
        section = raw_section.strip()
        if not section:
            continue
        if section not in ADMIN_SESSION_EVIDENCE_SECTIONS:
            invalid_sections.append(section)
            continue
        if section not in resolved_sections:
            resolved_sections.append(section)

    if invalid_sections:
        allowed_sections = ", ".join(ADMIN_SESSION_EVIDENCE_SECTIONS)
        invalid_label = ", ".join(invalid_sections)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Unsupported admin evidence sections: {invalid_label}. "
                f"Allowed values: {allowed_sections}."
            ),
        )

    return resolved_sections or list(ADMIN_SESSION_EVIDENCE_SECTIONS)


@app.get(
    "/v1/admin/sessions/{app_session_id}",
    response_model=AdminSessionDetailSummaryResponse,
)
def get_admin_session_detail(
    request: Request,
    app_session_id: UUID,
    principal: AuthenticatedPrincipal = Depends(_resolve_admin_session_principal),
) -> AdminSessionDetailSummaryResponse:
    _set_request_state(request, session_id=str(app_session_id), principal=principal)
    if not persistence_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database persistence is not configured.",
        )

    try:
        payload = get_admin_session_detail_summary(app_session_id)
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Persisted session was not found.",
            )
        _set_request_state(
            request,
            subject_id=payload["session"]["subject_id"],
            session_id=payload["session"]["client_session_id"],
            principal=principal,
        )
        log_event(
            logger,
            logging.INFO,
            "admin_session_detail_loaded",
            app_session_id=app_session_id,
            client_session_id=payload["session"]["client_session_id"],
            subject_id=payload["session"]["subject_id"],
            has_inference=payload["latest_inference"] is not None,
            grouped_fall_event_count=payload["evidence_counts"]["grouped_fall_events"],
            timeline_event_count=payload["evidence_counts"]["timeline_events"],
            transition_event_count=payload["evidence_counts"]["transition_events"],
            feedback_count=payload["evidence_counts"]["feedback"],
            annotation_count=payload["evidence_counts"]["annotations"],
            **_principal_log_fields(principal),
        )
        return AdminSessionDetailSummaryResponse(**payload)
    except HTTPException:
        raise
    except RuntimePersistenceError as exc:
        log_event(
            logger,
            logging.ERROR,
            "admin_session_detail_failed",
            exc_info=True,
            app_session_id=app_session_id,
            error=str(exc),
            **_persistence_error_fields(exc),
            **_principal_log_fields(principal),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load admin session detail.",
        ) from exc


@app.get(
    "/v1/admin/sessions/{app_session_id}/evidence",
    response_model=AdminSessionEvidenceResponse,
)
def get_admin_session_evidence_route(
    request: Request,
    app_session_id: UUID,
    sections: str | None = Query(default=None),
    principal: AuthenticatedPrincipal = Depends(_resolve_admin_session_principal),
) -> AdminSessionEvidenceResponse:
    _set_request_state(request, session_id=str(app_session_id), principal=principal)
    if not persistence_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database persistence is not configured.",
        )

    resolved_sections = _normalize_admin_session_evidence_sections(sections)

    try:
        payload = get_admin_session_evidence(
            app_session_id,
            sections=resolved_sections,
        )
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Persisted session was not found.",
            )
        _set_request_state(request, session_id=str(app_session_id), principal=principal)
        log_event(
            logger,
            logging.INFO,
            "admin_session_evidence_loaded",
            app_session_id=app_session_id,
            loaded_sections=payload["loaded_sections"],
            grouped_fall_event_count=len(payload["grouped_fall_events"]),
            timeline_event_count=len(payload["timeline_events"]),
            transition_event_count=len(payload["transition_events"]),
            feedback_count=len(payload["feedback"]),
            annotation_count=len(payload["annotations"]),
            **_principal_log_fields(principal),
        )
        return AdminSessionEvidenceResponse(**payload)
    except HTTPException:
        raise
    except RuntimePersistenceError as exc:
        log_event(
            logger,
            logging.ERROR,
            "admin_session_evidence_failed",
            exc_info=True,
            app_session_id=app_session_id,
            requested_sections=resolved_sections,
            error=str(exc),
            **_persistence_error_fields(exc),
            **_principal_log_fields(principal),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load admin session evidence.",
        ) from exc


@app.delete("/v1/sessions/{app_session_id}", response_model=DeleteSessionResponse)
def delete_session(
    request: Request,
    app_session_id: UUID,
    principal: AuthenticatedPrincipal | None = Depends(_resolve_authenticated_principal),
) -> DeleteSessionResponse:
    _set_request_state(request, session_id=str(app_session_id), principal=principal)
    if not persistence_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database persistence is not configured.",
        )

    try:
        deleted = delete_persisted_session(
            app_session_id,
            owner_user_id=owner_user_id_for_principal(principal)
            if principal is not None
            else None,
        )
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Persisted session was not found.",
            )
        log_event(
            logger,
            logging.INFO,
            "persisted_session_deleted",
            app_session_id=app_session_id,
            **_principal_log_fields(principal),
        )
        return DeleteSessionResponse(
            app_session_id=app_session_id,
            deleted=True,
            message="Persisted session deleted.",
        )
    except HTTPException:
        raise
    except RuntimePersistenceError as exc:
        log_event(
            logger,
            logging.ERROR,
            "persisted_session_delete_failed",
            exc_info=True,
            app_session_id=app_session_id,
            error=str(exc),
            **_persistence_error_fields(exc),
            **_principal_log_fields(principal),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete persisted session.",
        ) from exc


@app.post(
    "/v1/feedback",
    response_model=PredictionFeedbackResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def submit_feedback(
    request: Request,
    req: PredictionFeedbackRequest,
    principal: AuthenticatedPrincipal | None = Depends(_resolve_authenticated_principal),
) -> PredictionFeedbackResponse:
    request_id = _request_id_from_feedback_request(req)
    _set_request_state(
        request,
        request_id=request_id,
        session_id=req.session_id,
        subject_id=req.subject_id,
        principal=principal,
    )

    try:
        req = _normalize_feedback_request_for_principal(req, principal)
    except RuntimeAuthorizationError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc

    _set_request_state(request, subject_id=req.subject_id)
    recorded_at = datetime.now(timezone.utc).isoformat()

    try:
        persistence_result = None
        if persistence_enabled():
            persistence_result = persist_feedback_record(
                req,
                owner_user_id=owner_user_id_for_principal(principal)
                if principal is not None
                else None,
            )
            recorded_at = persistence_result.recorded_at.isoformat()
        else:
            payload = req.model_dump(mode="json")
            payload["target_type"] = _default_feedback_target_type(req)
            payload["request_id"] = str(request_id) if request_id is not None else None
            payload["recorded_at"] = recorded_at
            _append_jsonl(FEEDBACK_STORE_PATH, payload)

        log_event(
            logger,
            logging.INFO,
            "feedback_recorded",
            request_id=request_id,
            session_id=req.session_id,
            subject_id=req.subject_id,
            persisted_session_id=persistence_result.app_session_id if persistence_result is not None else None,
            persisted_inference_id=persistence_result.inference_id if persistence_result is not None else None,
            persisted_feedback_id=persistence_result.feedback_id if persistence_result is not None else None,
            target_type=(persistence_result.target_type if persistence_result is not None else _default_feedback_target_type(req)),
            event_id=req.event_id,
            window_id=req.window_id,
            feedback_type=req.user_feedback.value,
            reviewer_id=req.reviewer_id,
            **_principal_log_fields(principal),
        )

        return PredictionFeedbackResponse(
            request_id=request_id,
            session_id=req.session_id,
            persisted_session_id=persistence_result.app_session_id if persistence_result is not None else None,
            persisted_inference_id=persistence_result.inference_id if persistence_result is not None else None,
            persisted_feedback_id=persistence_result.feedback_id if persistence_result is not None else None,
            target_type=persistence_result.target_type if persistence_result is not None else _default_feedback_target_type(req),
            event_id=req.event_id,
            window_id=req.window_id,
            user_feedback=req.user_feedback,
            message="Feedback recorded successfully",
            status="accepted",
            recorded_at=recorded_at,
        )

    except RuntimePersistenceError as exc:
        log_event(
            logger,
            logging.ERROR,
            "feedback_persistence_failed",
            exc_info=True,
            request_id=request_id,
            session_id=req.session_id,
            subject_id=req.subject_id,
            error=str(exc),
            **_persistence_error_fields(exc),
            **_principal_log_fields(principal),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record feedback.",
        ) from exc

    except Exception as exc:  # noqa: BLE001
        log_event(
            logger,
            logging.ERROR,
            "feedback_failed",
            exc_info=True,
            request_id=request_id,
            session_id=req.session_id,
            subject_id=req.subject_id,
            error=str(exc),
            **_principal_log_fields(principal),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record feedback.",
        ) from exc


def _admin_web_cache_headers(full_path: str, *, is_index: bool = False) -> dict[str, str]:
    normalized_path = full_path.lstrip("/")

    if is_index or not normalized_path:
        return {"Cache-Control": "no-cache"}

    if normalized_path.startswith("assets/"):
        return {"Cache-Control": "public, max-age=31536000, immutable"}

    if normalized_path.startswith("figures/"):
        return {"Cache-Control": "public, max-age=86400"}

    if normalized_path in {"favicon.svg", "icons.svg"}:
        return {"Cache-Control": "public, max-age=86400"}

    return {"Cache-Control": "public, max-age=3600"}


@app.api_route("/{full_path:path}", methods=["GET", "HEAD"], include_in_schema=False)
def serve_admin_web(full_path: str) -> FileResponse:
    """Serve the built Solid admin app for browser routes on the public domain."""
    api_like_prefixes = ("v1/", "api/")
    if full_path == "health" or full_path.startswith(api_like_prefixes):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")
    if not ADMIN_WEB_ENABLED:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Admin web app is disabled.")

    dist_dir = ADMIN_WEB_DIST_PATH.resolve()
    index_path = dist_dir / "index.html"
    if not index_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Admin web app is not built.",
        )

    requested_path = (dist_dir / full_path).resolve() if full_path else index_path
    if requested_path == dist_dir or dist_dir in requested_path.parents:
        if requested_path.is_file():
            return FileResponse(
                requested_path,
                headers=_admin_web_cache_headers(full_path),
            )

    return FileResponse(
        index_path,
        headers=_admin_web_cache_headers("", is_index=True),
    )
