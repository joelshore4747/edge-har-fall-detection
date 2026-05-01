from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any
from uuid import UUID

import bcrypt
import psycopg
from psycopg.rows import dict_row


def _verify_bcrypt(password: str, password_hash: str | None) -> bool:
    """Verify a plaintext password against a bcrypt hash in Python.

    Compatible with hashes produced by pgcrypto's `crypt(..., gen_salt('bf'))`,
    which writes the standard `$2a$`/`$2b$`/`$2y$` format.
    """
    if not password or not password_hash:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


class RuntimeAuthenticationError(RuntimeError):
    """Raised when authentication cannot be completed."""


class RuntimeAuthorizationError(RuntimeAuthenticationError):
    """Raised when an authenticated principal is not allowed to access a resource."""


class RuntimeRegistrationError(RuntimeError):
    """Raised when self-service registration cannot be completed."""


class RuntimeRegistrationConflictError(RuntimeRegistrationError):
    """Raised when a self-service registration collides with another account."""


@dataclass(slots=True)
class AuthenticatedPrincipal:
    auth_account_id: UUID
    login_username: str
    role: str
    user_id: UUID | None
    subject_key: str | None

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


@dataclass(slots=True)
class RegisteredPrincipal:
    auth_account_id: UUID
    user_id: UUID
    login_username: str
    subject_key: str
    display_name: str | None
    created: bool


_SELF_SERVICE_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{5,79}$")


def auth_required() -> bool:
    raw = os.getenv("AUTH_REQUIRED", "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def self_service_registration_enabled() -> bool:
    raw = os.getenv("SELF_SERVICE_REGISTRATION_ENABLED", "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _database_url_from_env(db_url: str | None = None) -> str | None:
    if db_url is not None and db_url.strip():
        return db_url.strip()
    env_value = os.getenv("DATABASE_URL")
    if env_value is None or not env_value.strip():
        return None
    return env_value.strip()


def _require_db_url(db_url: str | None = None) -> str:
    resolved = _database_url_from_env(db_url)
    if resolved is None:
        raise RuntimeAuthenticationError("DATABASE_URL is not configured.")
    return resolved


def _normalize_identifier(value: str, *, field_name: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        raise RuntimeRegistrationError(f"{field_name} must not be blank.")
    if not _SELF_SERVICE_IDENTIFIER_PATTERN.fullmatch(normalized):
        raise RuntimeRegistrationError(
            f"{field_name} may only contain lowercase letters, digits, '.', '_' or '-'."
        )
    return normalized


def _normalize_display_name(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _validate_registration_password(password: str) -> str:
    if len(password) < 8:
        raise RuntimeRegistrationError("password must be at least 8 characters long.")
    if len(password) > 128:
        raise RuntimeRegistrationError("password must be at most 128 characters long.")
    return password


def _principal_from_row(row: dict[str, Any]) -> AuthenticatedPrincipal:
    if row["role"] != "admin" and row["user_id"] is None:
        raise RuntimeAuthenticationError(
            "User auth account is missing a linked app_users row."
        )

    return AuthenticatedPrincipal(
        auth_account_id=row["auth_account_id"],
        login_username=row["login_username"],
        role=row["role"],
        user_id=row["user_id"],
        subject_key=row["subject_key"],
    )


def register_self_service_user(
    username: str,
    password: str,
    *,
    subject_key: str,
    display_name: str | None = None,
    device_platform: str | None = None,
    device_model: str | None = None,
    db_url: str | None = None,
) -> RegisteredPrincipal:
    resolved_db_url = _require_db_url(db_url)
    normalized_username = _normalize_identifier(username, field_name="username")
    normalized_subject_key = _normalize_identifier(subject_key, field_name="subject_id")
    normalized_display_name = _normalize_display_name(display_name)
    normalized_password = _validate_registration_password(password)
    normalized_device_platform = _normalize_display_name(device_platform)
    normalized_device_model = _normalize_display_name(device_model)

    try:
        with psycopg.connect(resolved_db_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        a.auth_account_id,
                        a.user_id,
                        a.role,
                        u.subject_key,
                        u.display_name
                    FROM app_auth_accounts AS a
                    LEFT JOIN app_users AS u ON u.user_id = a.user_id
                    WHERE a.login_username = %s
                    """,
                    (normalized_username,),
                )
                existing = cur.fetchone()
                if existing is not None:
                    if existing["role"] != "user" or existing["user_id"] is None:
                        raise RuntimeRegistrationConflictError(
                            "username is already reserved for another account."
                        )
                    if (existing["subject_key"] or "").strip().lower() != normalized_subject_key:
                        raise RuntimeRegistrationConflictError(
                            "username is already associated with another subject."
                        )

                    cur.execute(
                        """
                        UPDATE app_users
                        SET
                            display_name = COALESCE(%s, display_name),
                            latest_device_platform = COALESCE(%s, latest_device_platform),
                            latest_device_model = COALESCE(%s, latest_device_model),
                            updated_at = NOW(),
                            last_seen_at = NOW()
                        WHERE user_id = %s
                        RETURNING display_name
                        """,
                        (
                            normalized_display_name,
                            normalized_device_platform,
                            normalized_device_model,
                            existing["user_id"],
                        ),
                    )
                    user_row = cur.fetchone()

                    cur.execute(
                        """
                        UPDATE app_auth_accounts
                        SET
                            password_hash = public.crypt(%s, public.gen_salt('bf')),
                            is_active = TRUE,
                            updated_at = NOW()
                        WHERE auth_account_id = %s
                        RETURNING auth_account_id
                        """,
                        (
                            normalized_password,
                            existing["auth_account_id"],
                        ),
                    )
                    auth_row = cur.fetchone()
                    conn.commit()
                    return RegisteredPrincipal(
                        auth_account_id=auth_row["auth_account_id"],
                        user_id=existing["user_id"],
                        login_username=normalized_username,
                        subject_key=normalized_subject_key,
                        display_name=(user_row or {}).get("display_name") or existing["display_name"],
                        created=False,
                    )

                cur.execute(
                    """
                    SELECT
                        u.user_id,
                        u.display_name,
                        a.auth_account_id
                    FROM app_users AS u
                    LEFT JOIN app_auth_accounts AS a ON a.user_id = u.user_id
                    WHERE u.subject_key = %s
                    """,
                    (normalized_subject_key,),
                )
                subject_row = cur.fetchone()
                if subject_row is not None and subject_row["auth_account_id"] is not None:
                    raise RuntimeRegistrationConflictError(
                        "subject_id is already associated with another account."
                    )

                if subject_row is None:
                    cur.execute(
                        """
                        INSERT INTO app_users (
                            subject_key,
                            display_name,
                            latest_device_platform,
                            latest_device_model
                        )
                        VALUES (%s, %s, %s, %s)
                        RETURNING user_id, display_name
                        """,
                        (
                            normalized_subject_key,
                            normalized_display_name or normalized_subject_key,
                            normalized_device_platform or "self_register",
                            normalized_device_model,
                        ),
                    )
                    user_row = cur.fetchone()
                else:
                    cur.execute(
                        """
                        UPDATE app_users
                        SET
                            display_name = COALESCE(%s, display_name),
                            latest_device_platform = COALESCE(%s, latest_device_platform),
                            latest_device_model = COALESCE(%s, latest_device_model),
                            updated_at = NOW(),
                            last_seen_at = NOW()
                        WHERE user_id = %s
                        RETURNING user_id, display_name
                        """,
                        (
                            normalized_display_name,
                            normalized_device_platform,
                            normalized_device_model,
                            subject_row["user_id"],
                        ),
                    )
                    user_row = cur.fetchone()

                cur.execute(
                    """
                    INSERT INTO app_auth_accounts (
                        user_id,
                        login_username,
                        password_hash,
                        role,
                        is_active
                    )
                    VALUES (
                        %s,
                        %s,
                        public.crypt(%s, public.gen_salt('bf')),
                        'user',
                        TRUE
                    )
                    RETURNING auth_account_id
                    """,
                    (
                        user_row["user_id"],
                        normalized_username,
                        normalized_password,
                    ),
                )
                auth_row = cur.fetchone()
            conn.commit()
    except RuntimeRegistrationError:
        raise
    except psycopg.Error as exc:
        raise RuntimeRegistrationError(f"Failed to register self-service user: {exc}") from exc

    return RegisteredPrincipal(
        auth_account_id=auth_row["auth_account_id"],
        user_id=user_row["user_id"],
        login_username=normalized_username,
        subject_key=normalized_subject_key,
        display_name=user_row["display_name"],
        created=True,
    )


def authenticate_basic_credentials(
    username: str,
    password: str,
    *,
    db_url: str | None = None,
) -> AuthenticatedPrincipal | None:
    resolved_db_url = _require_db_url(db_url)
    normalized_username = username.strip()
    if not normalized_username or not password:
        return None

    try:
        with psycopg.connect(resolved_db_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        a.auth_account_id,
                        a.user_id,
                        a.login_username,
                        a.role,
                        a.password_hash,
                        u.subject_key
                    FROM app_auth_accounts AS a
                    LEFT JOIN app_users AS u ON u.user_id = a.user_id
                    WHERE a.login_username = %s
                      AND a.is_active = TRUE
                    """,
                    (normalized_username,),
                )
                row = cur.fetchone()
                if row is None:
                    return None

                if not _verify_bcrypt(password, row.get("password_hash")):
                    return None

                cur.execute(
                    """
                    UPDATE app_auth_accounts
                    SET last_login_at = NOW()
                    WHERE auth_account_id = %s
                    """,
                    (row["auth_account_id"],),
                )
            conn.commit()
    except RuntimeAuthenticationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeAuthenticationError(
            f"Failed to authenticate API user: {exc}"
        ) from exc

    row.pop("password_hash", None)
    return _principal_from_row(row)


def load_authenticated_principal_by_account_id(
    auth_account_id: UUID | str,
    *,
    db_url: str | None = None,
) -> AuthenticatedPrincipal | None:
    resolved_db_url = _require_db_url(db_url)

    try:
        with psycopg.connect(resolved_db_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        a.auth_account_id,
                        a.user_id,
                        a.login_username,
                        a.role,
                        u.subject_key
                    FROM app_auth_accounts AS a
                    LEFT JOIN app_users AS u ON u.user_id = a.user_id
                    WHERE a.auth_account_id = %s
                      AND a.is_active = TRUE
                    """,
                    (str(auth_account_id),),
                )
                row = cur.fetchone()
                if row is None:
                    return None
    except RuntimeAuthenticationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeAuthenticationError(
            f"Failed to load authenticated principal: {exc}"
        ) from exc

    return _principal_from_row(row)


def normalize_subject_for_principal(
    principal: AuthenticatedPrincipal,
    subject_key: str | None,
) -> str:
    if principal.is_admin:
        normalized = (subject_key or "").strip()
        return normalized or "anonymous_user"

    owner_subject_key = (principal.subject_key or "").strip()
    if not owner_subject_key:
        raise RuntimeAuthorizationError(
            "Authenticated user is missing a linked subject_key."
        )

    normalized = (subject_key or "").strip()
    if not normalized or normalized == "anonymous_user":
        return owner_subject_key
    if normalized != owner_subject_key:
        raise RuntimeAuthorizationError(
            "Authenticated user cannot access another subject."
        )
    return normalized


def owner_user_id_for_principal(principal: AuthenticatedPrincipal) -> UUID | None:
    if principal.is_admin:
        return None
    return principal.user_id
