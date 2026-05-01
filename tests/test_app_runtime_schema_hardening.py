from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from uuid import uuid4

import psycopg
from psycopg.errors import CheckViolation, ForeignKeyViolation, UniqueViolation
from psycopg.types.json import Jsonb
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def _db_reachable(db_url: str) -> bool:
    try:
        with psycopg.connect(db_url, connect_timeout=2):
            return True
    except Exception:
        return False


def _connect(db_url: str, schema: str | None = None) -> psycopg.Connection:
    if schema is None:
        return psycopg.connect(db_url)
    return psycopg.connect(db_url, options=f"-c search_path={schema}")


@pytest.fixture
def schema_db() -> tuple[str, str]:
    db_url = DEFAULT_DB_URL
    if not _db_reachable(db_url):
        pytest.skip("PostgreSQL is not reachable for schema hardening tests.")

    schema = f"test_schema_{uuid4().hex[:10]}"
    with _connect(db_url) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f'CREATE SCHEMA "{schema}"')

    try:
        with _connect(db_url, schema=schema) as conn:
            with conn.cursor() as cur:
                for path in MIGRATION_PATHS:
                    cur.execute(path.read_text(encoding="utf-8"))
            conn.commit()
        yield db_url, schema
    finally:
        with _connect(db_url) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


def _seed_user_and_session(conn: psycopg.Connection, *, subject_key: str | None = None) -> tuple[str, str]:
    user_id = str(uuid4())
    session_id = str(uuid4())
    subject_key = subject_key or f"subject_{uuid4().hex[:8]}"

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO app_users (
                user_id,
                subject_key,
                display_name,
                latest_device_platform
            )
            VALUES (%s, %s, %s, %s)
            """,
            (user_id, subject_key, subject_key, "android"),
        )
        cur.execute(
            """
            INSERT INTO app_sessions (
                app_session_id,
                user_id,
                client_session_id,
                dataset_name,
                source_type,
                task_type,
                placement_declared,
                device_platform,
                recording_mode,
                runtime_mode,
                uploaded_at,
                sample_count,
                has_gyro,
                metadata_json,
                request_context_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                session_id,
                user_id,
                f"client_{session_id}",
                "APP_RUNTIME",
                "mobile_app",
                "runtime",
                "pocket",
                "android",
                "live_capture",
                "mobile_live",
                datetime.now(timezone.utc),
                256,
                True,
                Jsonb({}),
                Jsonb({}),
            ),
        )
    return user_id, session_id


def _seed_inference(
    conn: psycopg.Connection,
    *,
    app_session_id: str,
    status: str = "completed",
    error_message: str | None = None,
) -> str:
    inference_id = str(uuid4())
    started_at = datetime.now(timezone.utc)
    completed_at = started_at if status in {"completed", "failed"} else None

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO app_session_inferences (
                inference_id,
                app_session_id,
                api_version,
                status,
                error_message,
                started_at,
                completed_at,
                warning_level,
                request_options_json,
                source_summary_json,
                placement_summary_json,
                har_summary_json,
                fall_summary_json,
                alert_summary_json,
                debug_summary_json,
                model_info_json,
                narrative_summary_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                inference_id,
                app_session_id,
                "1.1.0",
                status,
                error_message,
                started_at,
                completed_at,
                "high",
                Jsonb({}),
                Jsonb({}),
                Jsonb({}),
                Jsonb({}),
                Jsonb({}),
                Jsonb({}),
                Jsonb({}),
                Jsonb({}),
                Jsonb({}),
            ),
        )
    return inference_id


def test_valid_and_invalid_session_inference_pairs(schema_db: tuple[str, str]) -> None:
    db_url, schema = schema_db

    with _connect(db_url, schema=schema) as conn:
        _, session_a = _seed_user_and_session(conn, subject_key="subject_a")
        _, session_b = _seed_user_and_session(conn, subject_key="subject_b")
        inference_a = _seed_inference(conn, app_session_id=session_a)
        inference_b = _seed_inference(conn, app_session_id=session_b)

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO app_grouped_fall_events (
                    app_session_id,
                    inference_id,
                    event_key,
                    event_start_ts,
                    event_end_ts,
                    event_duration_seconds,
                    positive_window_count
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (session_a, inference_a, "event_ok", 1.0, 2.0, 1.0, 2),
            )
        conn.commit()

        with conn.cursor() as cur:
            with pytest.raises(ForeignKeyViolation):
                cur.execute(
                    """
                    INSERT INTO app_grouped_fall_events (
                        app_session_id,
                        inference_id,
                        event_key,
                        event_start_ts,
                        event_end_ts,
                        event_duration_seconds,
                        positive_window_count
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (session_a, inference_b, "event_bad", 3.0, 4.0, 1.0, 1),
                )


def test_updated_at_trigger_updates_users_and_sessions(schema_db: tuple[str, str]) -> None:
    db_url, schema = schema_db

    with _connect(db_url, schema=schema) as conn:
        user_id, session_id = _seed_user_and_session(conn, subject_key="trigger_subject")

        with conn.cursor() as cur:
            cur.execute("SELECT updated_at FROM app_users WHERE user_id = %s", (user_id,))
            user_before = cur.fetchone()[0]
            cur.execute("SELECT updated_at FROM app_sessions WHERE app_session_id = %s", (session_id,))
            session_before = cur.fetchone()[0]

            cur.execute("SELECT pg_sleep(0.02)")

            cur.execute("UPDATE app_users SET display_name = %s WHERE user_id = %s", ("updated", user_id))
            cur.execute("UPDATE app_sessions SET notes = %s WHERE app_session_id = %s", ("changed", session_id))

            cur.execute("SELECT updated_at FROM app_users WHERE user_id = %s", (user_id,))
            user_after = cur.fetchone()[0]
            cur.execute("SELECT updated_at FROM app_sessions WHERE app_session_id = %s", (session_id,))
            session_after = cur.fetchone()[0]

        assert user_after > user_before
        assert session_after > session_before


def test_feedback_target_type_accepts_valid_values_and_rejects_invalid(schema_db: tuple[str, str]) -> None:
    db_url, schema = schema_db

    with _connect(db_url, schema=schema) as conn:
        _, session_id = _seed_user_and_session(conn, subject_key="feedback_subject")
        inference_id = _seed_inference(conn, app_session_id=session_id)

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO app_feedback (
                    app_session_id,
                    inference_id,
                    target_type,
                    feedback_type,
                    request_id
                )
                VALUES (%s, %s, %s, %s, %s)
                """,
                (session_id, inference_id, "session", "uncertain", str(uuid4())),
            )
        conn.commit()

        with conn.cursor() as cur:
            with pytest.raises(CheckViolation):
                cur.execute(
                    """
                    INSERT INTO app_feedback (
                        app_session_id,
                        inference_id,
                        target_type,
                        feedback_type,
                        request_id
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (session_id, inference_id, "bogus_target", "uncertain", str(uuid4())),
                )


def test_inference_status_fields_are_persisted(schema_db: tuple[str, str]) -> None:
    db_url, schema = schema_db

    with _connect(db_url, schema=schema) as conn:
        _, session_id = _seed_user_and_session(conn, subject_key="status_subject")
        inference_id = _seed_inference(
            conn,
            app_session_id=session_id,
            status="failed",
            error_message="model artifact missing",
        )

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT status, error_message, started_at, completed_at
                FROM app_session_inferences
                WHERE inference_id = %s
                """,
                (inference_id,),
            )
            row = cur.fetchone()

        assert row is not None
        status, error_message, started_at, completed_at = row
        assert status == "failed"
        assert error_message == "model artifact missing"
        assert started_at is not None
        assert completed_at is not None


def test_session_annotations_enforce_canonical_labels_and_unique_request_ids(
    schema_db: tuple[str, str],
) -> None:
    db_url, schema = schema_db

    with _connect(db_url, schema=schema) as conn:
        _, session_id = _seed_user_and_session(conn, subject_key="annotation_subject")
        request_id = str(uuid4())

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO app_session_annotations (
                    app_session_id,
                    annotation_label,
                    source,
                    request_id
                )
                VALUES (%s, %s, %s, %s)
                """,
                (session_id, "static", "mobile", request_id),
            )
        conn.commit()

        with conn.cursor() as cur:
            with pytest.raises(CheckViolation):
                cur.execute(
                    """
                    INSERT INTO app_session_annotations (
                        app_session_id,
                        annotation_label,
                        source
                    )
                    VALUES (%s, %s, %s)
                    """,
                    (session_id, "standing", "mobile"),
                )
        conn.rollback()

        with conn.cursor() as cur:
            with pytest.raises(UniqueViolation):
                cur.execute(
                    """
                    INSERT INTO app_session_annotations (
                        app_session_id,
                        annotation_label,
                        source,
                        request_id
                    )
                    VALUES (%s, %s, %s, %s)
                    """,
                    (session_id, "walking", "mobile", request_id),
                )
        conn.rollback()


def test_devices_and_model_registry_support_session_and_inference_links(
    schema_db: tuple[str, str],
) -> None:
    db_url, schema = schema_db

    with _connect(db_url, schema=schema) as conn:
        user_id, session_id = _seed_user_and_session(conn, subject_key="registry_subject")

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO app_devices (
                    user_id,
                    device_identifier,
                    identifier_source,
                    device_platform,
                    device_model
                )
                VALUES (%s, %s, %s, %s, %s)
                RETURNING device_id
                """,
                (user_id, "device-123", "explicit", "android", "pixel"),
            )
            device_id = cur.fetchone()[0]

            cur.execute(
                """
                UPDATE app_sessions
                SET device_id = %s
                WHERE app_session_id = %s
                """,
                (device_id, session_id),
            )

            cur.execute(
                """
                INSERT INTO app_model_registry (
                    model_type,
                    model_name,
                    model_version
                )
                VALUES (%s, %s, %s)
                RETURNING model_registry_id
                """,
                ("har", "movement_v2", "2026-04"),
            )
            har_model_registry_id = cur.fetchone()[0]

            cur.execute(
                """
                INSERT INTO app_model_registry (
                    model_type,
                    model_name,
                    model_version
                )
                VALUES (%s, %s, %s)
                RETURNING model_registry_id
                """,
                ("fall", "fall_meta_phone_negatives_v1", "2026-04"),
            )
            fall_model_registry_id = cur.fetchone()[0]

            cur.execute(
                """
                INSERT INTO app_session_inferences (
                    inference_id,
                    app_session_id,
                    api_version,
                    har_model_registry_id,
                    fall_model_registry_id,
                    status,
                    started_at,
                    completed_at,
                    warning_level,
                    request_options_json,
                    source_summary_json,
                    placement_summary_json,
                    har_summary_json,
                    fall_summary_json,
                    alert_summary_json,
                    debug_summary_json,
                    model_info_json,
                    narrative_summary_json
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    str(uuid4()),
                    session_id,
                    "1.1.0",
                    har_model_registry_id,
                    fall_model_registry_id,
                    "completed",
                    datetime.now(timezone.utc),
                    datetime.now(timezone.utc),
                    "low",
                    Jsonb({}),
                    Jsonb({}),
                    Jsonb({}),
                    Jsonb({}),
                    Jsonb({}),
                    Jsonb({}),
                    Jsonb({}),
                    Jsonb({}),
                    Jsonb({}),
                ),
            )
        conn.commit()

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT device_id
                FROM app_sessions
                WHERE app_session_id = %s
                """,
                (session_id,),
            )
            stored_device_id = cur.fetchone()[0]

            cur.execute(
                """
                SELECT har_model_registry_id, fall_model_registry_id
                FROM app_session_inferences
                WHERE app_session_id = %s
                """,
                (session_id,),
            )
            registry_row = cur.fetchone()

        assert str(stored_device_id) == str(device_id)
        assert registry_row is not None
        assert str(registry_row[0]) == str(har_model_registry_id)
        assert str(registry_row[1]) == str(fall_model_registry_id)
