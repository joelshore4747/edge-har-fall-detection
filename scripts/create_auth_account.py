#!/usr/bin/env python3
"""Create or update a Basic-auth account for the runtime API."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import psycopg


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or update a runtime API auth account")
    parser.add_argument("--db-url", default=None, help="Database URL (fallback: DATABASE_URL)")
    parser.add_argument("--username", required=True, help="Login username for Basic auth")
    parser.add_argument("--password", required=True, help="Plaintext password to hash and store")
    parser.add_argument(
        "--role",
        choices=("user", "admin"),
        default="user",
        help="Auth account role",
    )
    parser.add_argument(
        "--subject-key",
        default=None,
        help="Linked app_users.subject_key for role=user accounts",
    )
    parser.add_argument(
        "--display-name",
        default=None,
        help="Optional display name when creating a linked app_users row",
    )
    parser.add_argument(
        "--inactive",
        action="store_true",
        help="Create or update the account as inactive",
    )
    return parser.parse_args()


def _require_db_url(raw_value: str | None) -> str:
    resolved = raw_value or os.getenv("DATABASE_URL")
    if resolved is None or not resolved.strip():
        raise SystemExit("ERROR: DATABASE_URL env var or --db-url is required")
    return resolved.strip()


def _resolve_user_id(
    cur: psycopg.Cursor[Any],
    *,
    subject_key: str,
    display_name: str | None,
) -> str:
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
        (
            subject_key,
            display_name or subject_key,
            "auth_bootstrap",
        ),
    )
    row = cur.fetchone()
    if row is None:
        raise SystemExit("ERROR: failed to create or resolve linked app user")
    return str(row[0])


def main() -> int:
    args = parse_args()
    db_url = _require_db_url(args.db_url)

    subject_key = (args.subject_key or "").strip()
    if args.role == "user" and not subject_key:
        raise SystemExit("ERROR: --subject-key is required for role=user")

    user_id: str | None = None
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            if args.role == "user":
                user_id = _resolve_user_id(
                    cur,
                    subject_key=subject_key,
                    display_name=args.display_name,
                )

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
                    %s,
                    %s
                )
                ON CONFLICT (login_username) DO UPDATE
                SET
                    user_id = EXCLUDED.user_id,
                    password_hash = EXCLUDED.password_hash,
                    role = EXCLUDED.role,
                    is_active = EXCLUDED.is_active,
                    updated_at = NOW()
                RETURNING auth_account_id
                """,
                (
                    user_id,
                    args.username.strip(),
                    args.password,
                    args.role,
                    not args.inactive,
                ),
            )
            row = cur.fetchone()
        conn.commit()

    auth_account_id = row[0] if row is not None else None
    print("Auth account upserted")
    print(f"  auth_account_id={auth_account_id}")
    print(f"  username={args.username.strip()}")
    print(f"  role={args.role}")
    print(f"  subject_key={subject_key or '-'}")
    print(f"  active={not args.inactive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
