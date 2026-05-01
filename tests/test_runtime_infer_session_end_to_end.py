from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote
from uuid import uuid4

from fastapi.testclient import TestClient
import psycopg
import pytest

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

MIGRATION_PATHS = [
    PROJECT_ROOT / "db" / "migrations" / "002_app_runtime_persistence.sql",
    PROJECT_ROOT / "db" / "migrations" / "003_harden_app_runtime_schema.sql",
    PROJECT_ROOT / "db" / "migrations" / "004_add_basic_auth_accounts.sql",
    PROJECT_ROOT / "db" / "migrations" / "005_add_annotations_devices_and_model_registry.sql",
]
DEFAULT_DB_URL = (
    os.environ.get("SCHEMA_TEST_DATABASE_URL")
    or os.environ.get("DATABASE_URL")
    or "postgresql://summit:summit@localhost:5433/summit"
)
DEMO_PAYLOAD_PATH = (
    PROJECT_ROOT / "apps" / "mobile" / "app_mobile" / "assets" / "demo_session_phone1.json"
)
HAR_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "har" / "har_rf_ucihar.joblib"
FALL_ARTIFACT_PATH = (
    PROJECT_ROOT / "artifacts" / "fall" / "fall_meta_phone_negatives_v1" / "model.joblib"
)


def _db_reachable(db_url: str) -> bool:
    try:
        with psycopg.connect(db_url, connect_timeout=2):
            return True
    except Exception:
        return False


def _schema_db_url(base_url: str, schema: str) -> str:
    options = quote(f"-csearch_path={schema}", safe="")
    joiner = "&" if "?" in base_url else "?"
    return f"{base_url}{joiner}options={options}"


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


def _build_demo_payload(*, suffix: str | None = None) -> dict:
    payload = _load_demo_payload()
    token = suffix or uuid4().hex[:10]
    payload["metadata"]["subject_id"] = f"subject_{token}"
    payload["metadata"]["session_id"] = f"session_{token}"
    payload["metadata"]["recording_started_at"] = "2026-04-15T10:00:00Z"
    payload["metadata"]["recording_ended_at"] = "2026-04-15T10:00:12Z"
    payload["metadata"]["uploaded_at"] = "2026-04-15T10:00:15Z"
    payload.setdefault("request_context", {})
    payload["request_context"]["request_id"] = str(uuid4())
    payload["request_context"]["trace_id"] = f"trace_{token}"
    payload["request_context"]["client_version"] = "integration-test"
    return payload


def _persist_demo_session(client: TestClient, *, suffix: str | None = None) -> tuple[dict, dict]:
    payload = _build_demo_payload(suffix=suffix)
    payload["include_vulnerability_windows"] = True
    response = client.post("/v1/infer/session", json=payload)
    assert response.status_code == 200, response.text
    return payload, response.json()


def _create_auth_account(
    schema_db_url: str,
    *,
    username: str,
    password: str,
    subject_key: str | None = None,
    role: str = "user",
) -> None:
    with psycopg.connect(schema_db_url) as conn:
        with conn.cursor() as cur:
            user_id = None
            if role == "user":
                if subject_key is None:
                    raise ValueError("subject_key is required for role=user")
                cur.execute(
                    """
                    INSERT INTO app_users (
                        subject_key,
                        display_name,
                        latest_device_platform
                    )
                    VALUES (%s, %s, %s)
                    ON CONFLICT (subject_key) DO UPDATE
                    SET
                        display_name = COALESCE(EXCLUDED.display_name, app_users.display_name),
                        updated_at = NOW(),
                        last_seen_at = NOW()
                    RETURNING user_id
                    """,
                    (subject_key, subject_key, "auth_test"),
                )
                user_id = cur.fetchone()[0]

            cur.execute(
                """
                INSERT INTO app_auth_accounts (
                    user_id,
                    login_username,
                    password_hash,
                    role,
                    is_active
                )
                VALUES (%s, %s, public.crypt(%s, public.gen_salt('bf')), %s, %s)
                """,
                (user_id, username, password, role, True),
            )
        conn.commit()


@pytest.fixture
def runtime_db_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[str, Path]:
    base_db_url = DEFAULT_DB_URL
    if not _db_reachable(base_db_url):
        pytest.skip("PostgreSQL is not reachable for runtime end-to-end persistence tests.")

    schema = f"test_runtime_e2e_{uuid4().hex[:10]}"
    schema_db_url = _schema_db_url(base_db_url, schema)
    storage_dir = tmp_path / "runtime_sessions"

    with psycopg.connect(base_db_url) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f'CREATE SCHEMA "{schema}"')

    try:
        with psycopg.connect(schema_db_url) as conn:
            with conn.cursor() as cur:
                for path in MIGRATION_PATHS:
                    cur.execute(path.read_text(encoding="utf-8"))
            conn.commit()

        monkeypatch.setenv("DATABASE_URL", schema_db_url)
        monkeypatch.setenv("SESSION_STORAGE_DIR", str(storage_dir))
        monkeypatch.setenv("AUTH_REQUIRED", "false")
        yield schema_db_url, storage_dir
    finally:
        with psycopg.connect(base_db_url) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


def test_infer_session_persists_full_runtime_flow(runtime_db_env: tuple[str, Path]) -> None:
    schema_db_url, storage_dir = runtime_db_env

    with TestClient(main.app) as client:
        payload, body = _persist_demo_session(client)

    persisted_user_id = body["persisted_user_id"]
    persisted_session_id = body["persisted_session_id"]
    persisted_inference_id = body["persisted_inference_id"]
    assert persisted_user_id
    assert persisted_session_id
    assert persisted_inference_id

    with psycopg.connect(schema_db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT subject_key
                FROM app_users
                WHERE user_id = %s
                """,
                (persisted_user_id,),
            )
            user_row = cur.fetchone()

            cur.execute(
                """
                SELECT
                    client_session_id,
                    raw_storage_uri,
                    raw_payload_sha256,
                    raw_payload_bytes,
                    sample_count,
                    recording_started_at,
                    recording_ended_at,
                    uploaded_at
                FROM app_sessions
                WHERE app_session_id = %s
                """,
                (persisted_session_id,),
            )
            session_row = cur.fetchone()

            cur.execute(
                """
                SELECT
                    app_session_id,
                    request_id,
                    status,
                    grouped_fall_event_count,
                    timeline_event_count,
                    transition_event_count
                FROM app_session_inferences
                WHERE inference_id = %s
                """,
                (persisted_inference_id,),
            )
            inference_row = cur.fetchone()

            cur.execute(
                "SELECT COUNT(*) FROM app_grouped_fall_events WHERE inference_id = %s",
                (persisted_inference_id,),
            )
            grouped_count = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM app_timeline_events WHERE inference_id = %s",
                (persisted_inference_id,),
            )
            timeline_count = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM app_transition_events WHERE inference_id = %s",
                (persisted_inference_id,),
            )
            transition_count = cur.fetchone()[0]

    assert user_row is not None
    assert user_row[0] == payload["metadata"]["subject_id"]

    assert session_row is not None
    (
        client_session_id,
        raw_storage_uri,
        raw_payload_sha256,
        raw_payload_bytes,
        sample_count,
        recording_started_at,
        recording_ended_at,
        uploaded_at,
    ) = session_row
    assert client_session_id == payload["metadata"]["session_id"]
    assert sample_count == len(payload["samples"])
    assert recording_started_at is not None
    assert recording_ended_at is not None
    assert uploaded_at is not None

    assert inference_row is not None
    (
        inference_session_id,
        request_id,
        status,
        grouped_fall_event_count,
        timeline_event_count,
        transition_event_count,
    ) = inference_row
    assert str(inference_session_id) == persisted_session_id
    assert str(request_id) == payload["request_context"]["request_id"]
    assert status == "completed"

    assert grouped_count == len(body["grouped_fall_events"])
    assert timeline_count == len(body["timeline_events"])
    assert transition_count == len(body["transition_events"])
    assert grouped_fall_event_count == grouped_count
    assert timeline_event_count == timeline_count
    assert transition_event_count == transition_count

    assert raw_storage_uri is not None
    payload_path = Path(raw_storage_uri)
    assert payload_path.is_file()
    assert storage_dir in payload_path.parents
    assert raw_payload_bytes == payload_path.stat().st_size

    payload_bytes = payload_path.read_bytes()
    stored_payload = json.loads(payload_bytes.decode("utf-8"))
    assert stored_payload["request"]["metadata"]["session_id"] == payload["metadata"]["session_id"]
    assert stored_payload["request"]["metadata"]["subject_id"] == payload["metadata"]["subject_id"]
    assert raw_payload_sha256 == hashlib.sha256(payload_bytes).hexdigest()


def test_list_sessions_returns_latest_inference_summary_and_supports_subject_filter(
    runtime_db_env: tuple[str, Path],
) -> None:
    with TestClient(main.app) as client:
        payload_one, response_one = _persist_demo_session(client, suffix="list_a")
        _payload_two, _response_two = _persist_demo_session(client, suffix="list_b")

        response = client.get(
            "/v1/sessions",
            params={
                "subject_id": payload_one["metadata"]["subject_id"],
                "limit": 10,
                "offset": 0,
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_count"] == 1
    assert body["limit"] == 10
    assert body["offset"] == 0
    assert len(body["sessions"]) == 1

    session_item = body["sessions"][0]
    assert session_item["session"]["subject_id"] == payload_one["metadata"]["subject_id"]
    assert session_item["session"]["client_session_id"] == payload_one["metadata"]["session_id"]
    assert session_item["latest_inference_id"] == response_one["persisted_inference_id"]
    assert session_item["latest_status"] == "completed"
    assert (
        session_item["latest_grouped_fall_event_count"]
        == len(response_one["grouped_fall_events"])
    )


def test_session_detail_returns_latest_inference_and_feedback(
    runtime_db_env: tuple[str, Path],
) -> None:
    with TestClient(main.app) as client:
        payload, inference_response = _persist_demo_session(client, suffix="detail")

        feedback_response = client.post(
            "/v1/feedback",
            json={
                "session_id": payload["metadata"]["session_id"],
                "subject_id": payload["metadata"]["subject_id"],
                "persisted_session_id": inference_response["persisted_session_id"],
                "persisted_inference_id": inference_response["persisted_inference_id"],
                "target_type": "session",
                "user_feedback": "false_alarm",
                "notes": "integration detail check",
            },
        )
        assert feedback_response.status_code == 202, feedback_response.text

        response = client.get(f"/v1/sessions/{inference_response['persisted_session_id']}")

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["session"]["app_session_id"] == inference_response["persisted_session_id"]
    assert body["session"]["subject_id"] == payload["metadata"]["subject_id"]
    assert body["session"]["client_session_id"] == payload["metadata"]["session_id"]

    latest_inference = body["latest_inference"]
    assert latest_inference is not None
    assert latest_inference["inference_id"] == inference_response["persisted_inference_id"]
    assert latest_inference["status"] == "completed"
    assert latest_inference["response"]["persisted_session_id"] == inference_response["persisted_session_id"]
    assert latest_inference["response"]["persisted_inference_id"] == inference_response["persisted_inference_id"]
    assert (
        len(latest_inference["response"]["grouped_fall_events"])
        == len(inference_response["grouped_fall_events"])
    )
    assert len(latest_inference["response"]["timeline_events"]) == len(inference_response["timeline_events"])
    assert (
        len(latest_inference["response"]["transition_events"])
        == len(inference_response["transition_events"])
    )
    assert latest_inference["response"]["vulnerability_summary"]["top_vulnerability_score"] is not None
    assert latest_inference["response"]["vulnerability_summary"]["latest_vulnerability_level"] is not None
    assert latest_inference["response"]["vulnerability_windows"] == []
    assert latest_inference["response"]["alert_summary"]["top_vulnerability_score"] is not None

    assert len(body["feedback"]) == 1
    feedback = body["feedback"][0]
    assert feedback["app_session_id"] == inference_response["persisted_session_id"]
    assert feedback["inference_id"] == inference_response["persisted_inference_id"]
    assert feedback["target_type"] == "session"
    assert feedback["feedback_type"] == "false_alarm"
    assert feedback["notes"] == "integration detail check"
    assert body["annotations"] == []


def test_session_annotations_are_persisted_and_surface_in_detail_and_list(
    runtime_db_env: tuple[str, Path],
) -> None:
    with TestClient(main.app) as client:
        payload, inference_response = _persist_demo_session(client, suffix="annotated")
        app_session_id = inference_response["persisted_session_id"]

        create_response = client.post(
            f"/v1/sessions/{app_session_id}/annotations",
            json={
                "label": "standing",
                "notes": "reviewed from mobile client",
                "request_context": {
                    "request_id": str(uuid4()),
                    "client_version": "integration-test",
                },
            },
        )
        assert create_response.status_code == 200, create_response.text
        created = create_response.json()
        assert created["app_session_id"] == app_session_id
        assert created["label"] == "static"
        assert created["source"] == "mobile"
        assert created["notes"] == "reviewed from mobile client"

        list_response = client.get(f"/v1/sessions/{app_session_id}/annotations")
        assert list_response.status_code == 200, list_response.text
        annotations_payload = list_response.json()
        assert annotations_payload["app_session_id"] == app_session_id
        assert len(annotations_payload["annotations"]) == 1
        assert annotations_payload["annotations"][0]["label"] == "static"

        detail_response = client.get(f"/v1/sessions/{app_session_id}")
        assert detail_response.status_code == 200, detail_response.text
        detail_payload = detail_response.json()
        assert len(detail_payload["annotations"]) == 1
        assert detail_payload["annotations"][0]["label"] == "static"

        sessions_response = client.get(
            "/v1/sessions",
            params={
                "subject_id": payload["metadata"]["subject_id"],
                "limit": 10,
                "offset": 0,
            },
        )
        assert sessions_response.status_code == 200, sessions_response.text
        session_item = sessions_response.json()["sessions"][0]
        assert session_item["latest_annotation_label"] == "static"
        assert session_item["latest_annotation_source"] == "mobile"
        assert session_item["latest_annotation_created_at"] is not None


def test_admin_cookie_session_routes_work_end_to_end(
    runtime_db_env: tuple[str, Path],
) -> None:
    schema_db_url, _storage_dir = runtime_db_env
    _create_auth_account(
        schema_db_url,
        username="admin_user",
        password="admin_password_123",
        role="admin",
    )

    with TestClient(main.app) as client:
        payload, inference_response = _persist_demo_session(client, suffix="admin_routes")
        app_session_id = inference_response["persisted_session_id"]

        feedback_response = client.post(
            "/v1/feedback",
            json={
                "session_id": payload["metadata"]["session_id"],
                "subject_id": payload["metadata"]["subject_id"],
                "persisted_session_id": app_session_id,
                "persisted_inference_id": inference_response["persisted_inference_id"],
                "target_type": "session",
                "user_feedback": "false_alarm",
                "notes": "admin evidence review",
            },
        )
        assert feedback_response.status_code == 202, feedback_response.text

        annotation_response = client.post(
            f"/v1/sessions/{app_session_id}/annotations",
            json={
                "label": "static",
                "source": "mobile",
                "notes": "reviewed from mobile client",
                "request_context": {
                    "request_id": str(uuid4()),
                    "client_version": "integration-test",
                },
            },
        )
        assert annotation_response.status_code == 200, annotation_response.text

        login_response = client.post(
            "/v1/admin/auth/login",
            json={"username": "admin_user", "password": "admin_password_123"},
        )
        assert login_response.status_code == 200, login_response.text
        login_payload = login_response.json()
        assert login_payload["status"] == "authenticated"
        assert login_payload["username"] == "admin_user"
        assert login_payload["role"] == "admin"

        me_response = client.get("/v1/admin/auth/me")
        assert me_response.status_code == 200, me_response.text
        me_payload = me_response.json()
        assert me_payload["username"] == "admin_user"
        assert me_payload["role"] == "admin"

        overview_response = client.get("/v1/admin/overview")
        assert overview_response.status_code == 200, overview_response.text
        overview_payload = overview_response.json()
        assert overview_payload["totals"]["sessions"] >= 1
        assert overview_payload["totals"]["inferences"] >= 1
        overview_recent_session = next(
            item
            for item in overview_payload["recent_sessions"]
            if item["session"]["app_session_id"] == app_session_id
        )
        assert any(
            item["session"]["app_session_id"] == app_session_id
            for item in overview_payload["recent_sessions"]
        )
        assert "dataset_name" not in overview_recent_session["session"]
        assert "notes" not in overview_recent_session["session"]
        assert "latest_annotation_label" not in overview_recent_session

        sessions_response = client.get(
            "/v1/admin/sessions",
            params={"page": 1, "page_size": 25},
        )
        assert sessions_response.status_code == 200, sessions_response.text
        sessions_payload = sessions_response.json()
        assert sessions_payload["total_count"] >= 1
        listed_session = next(
            item
            for item in sessions_payload["sessions"]
            if item["session"]["app_session_id"] == app_session_id
        )
        assert any(
            item["session"]["app_session_id"] == app_session_id
            for item in sessions_payload["sessions"]
        )
        assert "notes" in listed_session["session"]
        assert "dataset_name" not in listed_session["session"]
        assert "raw_storage_uri" not in listed_session["session"]
        assert listed_session["latest_annotation_label"] == "static"
        assert "latest_annotation_source" not in listed_session

        annotation_search_response = client.get(
            "/v1/admin/sessions",
            params={"page": 1, "page_size": 25, "search": "static"},
        )
        assert annotation_search_response.status_code == 200, annotation_search_response.text
        annotation_search_payload = annotation_search_response.json()
        assert annotation_search_payload["total_count"] == 1
        assert annotation_search_payload["sessions"][0]["session"]["app_session_id"] == app_session_id

        status_filter_response = client.get(
            "/v1/admin/sessions",
            params={"page": 1, "page_size": 25, "status": "completed"},
        )
        assert status_filter_response.status_code == 200, status_filter_response.text
        status_filter_payload = status_filter_response.json()
        assert status_filter_payload["total_count"] == 1
        assert status_filter_payload["sessions"][0]["session"]["app_session_id"] == app_session_id

        detail_response = client.get(f"/v1/admin/sessions/{app_session_id}")
        assert detail_response.status_code == 200, detail_response.text
        detail_payload = detail_response.json()
        assert detail_payload["session"]["app_session_id"] == app_session_id
        assert detail_payload["session"]["subject_id"] == payload["metadata"]["subject_id"]
        assert detail_payload["latest_inference"]["inference_id"] == inference_response["persisted_inference_id"]
        assert detail_payload["latest_feedback"]["notes"] == "admin evidence review"
        assert detail_payload["latest_annotation"]["label"] == "static"
        assert "grouped_fall_events" not in detail_payload["latest_inference"]["response"]
        assert "timeline_events" not in detail_payload["latest_inference"]["response"]
        assert "transition_events" not in detail_payload["latest_inference"]["response"]
        assert (
            detail_payload["evidence_counts"]["grouped_fall_events"]
            == len(inference_response["grouped_fall_events"])
        )
        assert (
            detail_payload["evidence_counts"]["timeline_events"]
            == len(inference_response["timeline_events"])
        )
        assert (
            detail_payload["evidence_counts"]["transition_events"]
            == len(inference_response["transition_events"])
        )
        assert detail_payload["evidence_counts"]["feedback"] == 1
        assert detail_payload["evidence_counts"]["annotations"] == 1
        assert "dataset_name" in detail_payload["session"]
        assert "raw_storage_uri" in detail_payload["session"]

        evidence_response = client.get(f"/v1/admin/sessions/{app_session_id}/evidence")
        assert evidence_response.status_code == 200, evidence_response.text
        evidence_payload = evidence_response.json()
        assert evidence_payload["loaded_sections"] == [
            "grouped_fall_events",
            "timeline_events",
            "transition_events",
            "feedback",
            "annotations",
        ]
        assert (
            len(evidence_payload["grouped_fall_events"])
            == len(inference_response["grouped_fall_events"])
        )
        assert len(evidence_payload["timeline_events"]) == len(inference_response["timeline_events"])
        assert (
            len(evidence_payload["transition_events"])
            == len(inference_response["transition_events"])
        )
        assert len(evidence_payload["feedback"]) == 1
        assert evidence_payload["feedback"][0]["notes"] == "admin evidence review"
        assert len(evidence_payload["annotations"]) == 1
        assert evidence_payload["annotations"][0]["label"] == "static"


def test_delete_session_removes_database_rows_and_raw_payload(
    runtime_db_env: tuple[str, Path],
) -> None:
    schema_db_url, _storage_dir = runtime_db_env

    with TestClient(main.app) as client:
        _payload, inference_response = _persist_demo_session(client, suffix="delete")
        persisted_session_id = inference_response["persisted_session_id"]
        persisted_inference_id = inference_response["persisted_inference_id"]

        with psycopg.connect(schema_db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT raw_storage_uri FROM app_sessions WHERE app_session_id = %s",
                    (persisted_session_id,),
                )
                raw_storage_uri = cur.fetchone()[0]

        raw_payload_path = Path(raw_storage_uri)
        assert raw_payload_path.exists()

        response = client.delete(f"/v1/sessions/{persisted_session_id}")
        assert response.status_code == 200, response.text
        assert response.json()["deleted"] is True

        detail_response = client.get(f"/v1/sessions/{persisted_session_id}")
        assert detail_response.status_code == 404, detail_response.text

    with psycopg.connect(schema_db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM app_sessions WHERE app_session_id = %s",
                (persisted_session_id,),
            )
            assert cur.fetchone()[0] == 0

            cur.execute(
                "SELECT COUNT(*) FROM app_session_inferences WHERE inference_id = %s",
                (persisted_inference_id,),
            )
            assert cur.fetchone()[0] == 0

            cur.execute(
                "SELECT COUNT(*) FROM app_timeline_events WHERE inference_id = %s",
                (persisted_inference_id,),
            )
            assert cur.fetchone()[0] == 0

    assert not raw_payload_path.exists()


def test_basic_auth_blocks_unauthenticated_requests(
    runtime_db_env: tuple[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "true")

    with TestClient(main.app) as client:
        response = client.get("/v1/sessions")

    assert response.status_code == 401, response.text
    assert response.headers["www-authenticate"] == "Basic"


def test_basic_auth_user_is_limited_to_own_subject_sessions(
    runtime_db_env: tuple[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema_db_url, _storage_dir = runtime_db_env
    monkeypatch.setenv("AUTH_REQUIRED", "true")

    _create_auth_account(
        schema_db_url,
        username="alice",
        password="password123",
        subject_key="subject_alice",
    )
    _create_auth_account(
        schema_db_url,
        username="bob",
        password="password456",
        subject_key="subject_bob",
    )

    with TestClient(main.app) as client:
        alice_payload = _build_demo_payload(suffix="auth_alice")
        alice_payload["metadata"]["subject_id"] = "subject_alice"
        alice_response = client.post(
            "/v1/infer/session",
            json=alice_payload,
            auth=("alice", "password123"),
        )
        assert alice_response.status_code == 200, alice_response.text

        bob_payload = _build_demo_payload(suffix="auth_bob")
        bob_payload["metadata"]["subject_id"] = "subject_bob"
        bob_response = client.post(
            "/v1/infer/session",
            json=bob_payload,
            auth=("bob", "password456"),
        )
        assert bob_response.status_code == 200, bob_response.text

        mismatch_payload = _build_demo_payload(suffix="auth_mismatch")
        mismatch_payload["metadata"]["subject_id"] = "subject_bob"
        mismatch_response = client.post(
            "/v1/infer/session",
            json=mismatch_payload,
            auth=("alice", "password123"),
        )
        assert mismatch_response.status_code == 403, mismatch_response.text

        alice_list = client.get("/v1/sessions", auth=("alice", "password123"))
        assert alice_list.status_code == 200, alice_list.text
        alice_body = alice_list.json()
        assert alice_body["total_count"] == 1
        assert len(alice_body["sessions"]) == 1
        assert alice_body["sessions"][0]["session"]["subject_id"] == "subject_alice"

        other_detail = client.get(
            f"/v1/sessions/{bob_response.json()['persisted_session_id']}",
            auth=("alice", "password123"),
        )
        assert other_detail.status_code == 404, other_detail.text
