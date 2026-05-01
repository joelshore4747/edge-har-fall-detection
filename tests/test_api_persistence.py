from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault(
    "HAR_ARTIFACT_PATH",
    str(PROJECT_ROOT / "artifacts" / "har" / "har_rf_ucihar.joblib"),
)
os.environ.setdefault(
    "FALL_ARTIFACT_PATH",
    str(
        PROJECT_ROOT / "artifacts" / "fall" / "fall_meta_phone_negatives_v1" / "model.joblib"
    ),
)

from apps.api import main
from apps.api.schemas import FeedbackTargetType, RuntimeSessionRequest
from services.runtime_persistence import (
    FeedbackPersistenceResult,
    RuntimePersistenceResult,
    _json_compatible,
    _resolve_session_lifecycle,
)

DEMO_PAYLOAD_PATH = (
    PROJECT_ROOT / "apps" / "mobile" / "app_mobile" / "assets" / "demo_session_phone1.json"
)
HAR_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "har" / "har_rf_ucihar.joblib"
FALL_ARTIFACT_PATH = (
    PROJECT_ROOT / "artifacts" / "fall" / "fall_meta_phone_negatives_v1" / "model.joblib"
)


def _require_demo_and_artifacts() -> None:
    missing: list[str] = []
    for path in (DEMO_PAYLOAD_PATH, HAR_ARTIFACT_PATH, FALL_ARTIFACT_PATH):
        if not path.exists():
            missing.append(str(path))
    if missing:
        pytest.skip(f"Required test assets are missing: {missing}")


def _load_demo_payload() -> dict:
    _require_demo_and_artifacts()
    return json.loads(DEMO_PAYLOAD_PATH.read_text(encoding="utf-8"))


def test_infer_session_adds_persisted_ids_when_db_persistence_is_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    payload = _load_demo_payload()

    expected_user_id = uuid4()
    expected_session_id = uuid4()
    expected_inference_id = uuid4()

    monkeypatch.setattr(main, "persistence_enabled", lambda: True)
    monkeypatch.setattr(
        main,
        "persist_runtime_session",
        lambda req, resp: RuntimePersistenceResult(
            user_id=expected_user_id,
            app_session_id=expected_session_id,
            inference_id=expected_inference_id,
        ),
    )

    with TestClient(main.app) as client:
        response = client.post("/v1/infer/session", json=payload)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["persisted_user_id"] == str(expected_user_id)
    assert body["persisted_session_id"] == str(expected_session_id)
    assert body["persisted_inference_id"] == str(expected_inference_id)


def test_feedback_uses_database_persistence_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    expected_feedback_id = uuid4()
    expected_session_id = uuid4()
    expected_inference_id = uuid4()

    monkeypatch.setattr(main, "persistence_enabled", lambda: True)
    monkeypatch.setattr(
        main,
        "persist_feedback_record",
        lambda req, owner_user_id=None: FeedbackPersistenceResult(
            feedback_id=expected_feedback_id,
            app_session_id=expected_session_id,
            inference_id=expected_inference_id,
            target_type=FeedbackTargetType.session,
            recorded_at=main.datetime.now(main.timezone.utc),
        ),
    )

    with TestClient(main.app) as client:
        response = client.post(
            "/v1/feedback",
            json={
                "session_id": "session_123",
                "user_feedback": "false_alarm",
            },
        )

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["persisted_session_id"] == str(expected_session_id)
    assert body["persisted_inference_id"] == str(expected_inference_id)
    assert body["persisted_feedback_id"] == str(expected_feedback_id)
    assert body["target_type"] == "session"


def test_feedback_falls_back_to_jsonl_when_db_persistence_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    feedback_path = tmp_path / "feedback.jsonl"

    monkeypatch.setattr(main, "persistence_enabled", lambda: False)
    monkeypatch.setattr(main, "FEEDBACK_STORE_PATH", feedback_path)

    with TestClient(main.app) as client:
        response = client.post(
            "/v1/feedback",
            json={
                "session_id": "session_456",
                "user_feedback": "confirmed_fall",
                "notes": "follow-up review",
            },
        )

    assert response.status_code == 202, response.text
    assert feedback_path.exists()

    lines = feedback_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["session_id"] == "session_456"
    assert payload["user_feedback"] == "confirmed_fall"
    assert payload["notes"] == "follow-up review"
    assert payload["target_type"] == "session"


def test_validation_errors_include_request_id_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "false")

    with TestClient(main.app) as client:
        response = client.post("/v1/feedback", json={})

    assert response.status_code == 422, response.text
    body = response.json()
    assert body["request_id"]
    assert response.headers["X-Request-ID"] == body["request_id"]


def test_auth_me_reports_anonymous_when_auth_is_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "false")

    with TestClient(main.app) as client:
        response = client.get("/v1/auth/me")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "anonymous"
    assert body["subject_id"] == "anonymous_user"
    assert body["auth_required"] is False


def test_delete_session_uses_database_persistence_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    expected_session_id = uuid4()
    calls: list[tuple[str, object | None]] = []

    monkeypatch.setattr(main, "persistence_enabled", lambda: True)
    monkeypatch.setattr(
        main,
        "delete_persisted_session",
        lambda app_session_id, owner_user_id=None: calls.append(
            (str(app_session_id), owner_user_id)
        )
        or True,
    )

    with TestClient(main.app) as client:
        response = client.delete(f"/v1/sessions/{expected_session_id}")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["app_session_id"] == str(expected_session_id)
    assert body["deleted"] is True
    assert body["message"] == "Persisted session deleted."
    assert calls == [(str(expected_session_id), None)]


def test_delete_session_returns_404_when_session_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    missing_session_id = uuid4()

    monkeypatch.setattr(main, "persistence_enabled", lambda: True)
    monkeypatch.setattr(
        main,
        "delete_persisted_session",
        lambda app_session_id, owner_user_id=None: False,
    )

    with TestClient(main.app) as client:
        response = client.delete(f"/v1/sessions/{missing_session_id}")

    assert response.status_code == 404, response.text
    assert response.json()["message"] == "Persisted session was not found."


def test_raw_session_returns_stored_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "false")

    storage_root = tmp_path / "runtime_sessions"
    subject_dir = storage_root / "subject_abc"
    subject_dir.mkdir(parents=True)
    payload_path = subject_dir / "session_001__req.json"
    payload_body = {"stored_at": "2026-04-24T00:00:00+00:00", "request": {"demo": True}}
    payload_path.write_text(json.dumps(payload_body), encoding="utf-8")

    monkeypatch.setattr(main, "persistence_enabled", lambda: True)
    monkeypatch.setattr(main, "_resolve_session_storage_dir", lambda: storage_root.resolve())
    monkeypatch.setattr(
        main,
        "get_session_raw_storage_location",
        lambda app_session_id, owner_user_id=None: {
            "raw_storage_uri": str(payload_path),
            "raw_storage_format": "application/json",
            "raw_payload_sha256": "deadbeef",
            "raw_payload_bytes": payload_path.stat().st_size,
        },
    )

    session_id = uuid4()
    with TestClient(main.app) as client:
        response = client.get(f"/v1/sessions/{session_id}/raw")

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == payload_body


def test_raw_session_returns_404_when_session_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    monkeypatch.setattr(main, "persistence_enabled", lambda: True)
    monkeypatch.setattr(
        main,
        "get_session_raw_storage_location",
        lambda app_session_id, owner_user_id=None: None,
    )

    with TestClient(main.app) as client:
        response = client.get(f"/v1/sessions/{uuid4()}/raw")

    assert response.status_code == 404, response.text
    assert response.json()["message"] == "Persisted session was not found."


def test_raw_session_returns_404_when_payload_file_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    storage_root = tmp_path / "runtime_sessions"
    storage_root.mkdir()
    missing_path = storage_root / "subject_abc" / "never_written.json"

    monkeypatch.setattr(main, "persistence_enabled", lambda: True)
    monkeypatch.setattr(main, "_resolve_session_storage_dir", lambda: storage_root.resolve())
    monkeypatch.setattr(
        main,
        "get_session_raw_storage_location",
        lambda app_session_id, owner_user_id=None: {
            "raw_storage_uri": str(missing_path),
            "raw_storage_format": "application/json",
            "raw_payload_sha256": None,
            "raw_payload_bytes": None,
        },
    )

    with TestClient(main.app) as client:
        response = client.get(f"/v1/sessions/{uuid4()}/raw")

    assert response.status_code == 404, response.text
    assert response.json()["message"] == "Stored raw payload file is missing."


def test_raw_session_rejects_payload_outside_storage_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    storage_root = tmp_path / "runtime_sessions"
    storage_root.mkdir()
    escaped_path = tmp_path / "outside.json"
    escaped_path.write_text("{\"leak\": true}", encoding="utf-8")

    monkeypatch.setattr(main, "persistence_enabled", lambda: True)
    monkeypatch.setattr(main, "_resolve_session_storage_dir", lambda: storage_root.resolve())
    monkeypatch.setattr(
        main,
        "get_session_raw_storage_location",
        lambda app_session_id, owner_user_id=None: {
            "raw_storage_uri": str(escaped_path),
            "raw_storage_format": "application/json",
            "raw_payload_sha256": None,
            "raw_payload_bytes": None,
        },
    )

    with TestClient(main.app) as client:
        response = client.get(f"/v1/sessions/{uuid4()}/raw")

    assert response.status_code == 404, response.text
    assert response.json()["message"] == (
        "Stored raw payload is outside the configured storage root."
    )


def test_raw_session_returns_503_when_persistence_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    monkeypatch.setattr(main, "persistence_enabled", lambda: False)

    with TestClient(main.app) as client:
        response = client.get(f"/v1/sessions/{uuid4()}/raw")

    assert response.status_code == 503, response.text


def test_json_compatible_replaces_non_finite_values_for_jsonb_storage() -> None:
    payload = {
        "nan_value": float("nan"),
        "inf_value": float("inf"),
        "nested": {
            "negative_inf": float("-inf"),
            "series": [1.5, float("nan"), {"ts": datetime(2026, 4, 20, tzinfo=timezone.utc)}],
        },
    }

    sanitized = _json_compatible(payload)

    assert sanitized["nan_value"] is None
    assert sanitized["inf_value"] is None
    assert sanitized["nested"]["negative_inf"] is None
    assert sanitized["nested"]["series"][0] == 1.5
    assert sanitized["nested"]["series"][1] is None
    assert sanitized["nested"]["series"][2]["ts"] == "2026-04-20T00:00:00+00:00"


def test_runtime_session_lifecycle_defaults_uploaded_at_for_mobile_payload() -> None:
    payload = {
        "metadata": {
            "session_id": "session_mobile_missing_uploaded_at",
            "subject_id": "subject_mobile",
            "placement": "pocket",
            "task_type": "runtime",
            "dataset_name": "APP_RUNTIME",
            "source_type": "mobile_app",
            "device_platform": "android",
        },
        "samples": [
            {"timestamp": index / 50.0, "ax": 0.1, "ay": 0.2, "az": 9.8}
            for index in range(32)
        ],
    }

    request = RuntimeSessionRequest.model_validate(payload)
    before = datetime.now(timezone.utc)
    recording_started_at, recording_ended_at, uploaded_at = _resolve_session_lifecycle(
        request
    )
    after = datetime.now(timezone.utc)

    assert recording_started_at is None
    assert recording_ended_at is None
    assert uploaded_at is not None
    assert before <= uploaded_at <= after
