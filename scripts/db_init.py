#!/usr/bin/env python3
"""Apply database migrations in db/migrations.

Tracks applied migrations in a `schema_migrations(version, applied_at)`
table so each `*.sql` file runs exactly once. Re-running this script is a
no-op once every migration has been recorded.

For a database that already had migrations applied before the ledger
existed, the bootstrap step below records every migration as applied
when it detects the canonical baseline tables (`app_users`,
`app_auth_accounts`) — this prevents migrations from re-running against
a live production schema that the ledger has not yet seen.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize the Postgres database schema")
    parser.add_argument("--db-url", default=None, help="Database URL (fallback: DATABASE_URL)")
    parser.add_argument("--migrations-dir", default="db/migrations", help="Directory containing SQL migrations")
    return parser.parse_args()


def _resolve_dir(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _collect_migrations(migrations_dir: Path) -> List[Path]:
    return sorted([p for p in migrations_dir.glob("*.sql") if p.is_file()])


def _ensure_schema_migrations_table(cur) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )


def _bootstrap_legacy_schema(cur, all_versions: List[str]) -> int:
    """Record every migration as applied if the ledger is empty but the
    canonical baseline tables already exist (i.e. this DB was migrated
    before the ledger was introduced).

    Returns the number of versions back-filled.
    """
    cur.execute("SELECT COUNT(*) FROM schema_migrations")
    count = cur.fetchone()[0]
    if count > 0:
        return 0

    cur.execute(
        """
        SELECT to_regclass('public.app_users') IS NOT NULL
           AND to_regclass('public.app_auth_accounts') IS NOT NULL
        """
    )
    legacy_present = bool(cur.fetchone()[0])
    if not legacy_present:
        return 0

    for version in all_versions:
        cur.execute(
            "INSERT INTO schema_migrations (version) VALUES (%s) ON CONFLICT DO NOTHING",
            (version,),
        )
    return len(all_versions)


def _already_applied(cur, version: str) -> bool:
    cur.execute("SELECT 1 FROM schema_migrations WHERE version = %s", (version,))
    return cur.fetchone() is not None


def _record_applied(cur, version: str) -> None:
    cur.execute(
        "INSERT INTO schema_migrations (version) VALUES (%s) ON CONFLICT DO NOTHING",
        (version,),
    )


def main() -> int:
    args = parse_args()
    db_url = args.db_url or os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL env var or --db-url is required")
        return 1

    migrations_dir = _resolve_dir(args.migrations_dir)
    if not migrations_dir.exists():
        print(f"ERROR: migrations directory not found: {migrations_dir}")
        return 1

    migrations = _collect_migrations(migrations_dir)
    if not migrations:
        print(f"No migrations found under {migrations_dir}")
        return 0

    try:
        import psycopg  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: psycopg is required to run migrations: {type(exc).__name__}: {exc}")
        return 1

    versions = [path.name for path in migrations]
    applied: List[str] = []
    skipped: List[str] = []
    backfilled = 0

    with psycopg.connect(db_url) as conn:
        conn.autocommit = False
        with conn.cursor() as cur:
            _ensure_schema_migrations_table(cur)
            backfilled = _bootstrap_legacy_schema(cur, versions)
        conn.commit()

        for path in migrations:
            sql = path.read_text(encoding="utf-8").strip()
            if not sql:
                continue
            with conn.cursor() as cur:
                if _already_applied(cur, path.name):
                    skipped.append(path.name)
                    continue
                cur.execute(sql)
                _record_applied(cur, path.name)
            conn.commit()
            applied.append(path.name)

    print("Migrations applied")
    print(f"  db_url={db_url}")
    print(f"  migrations_dir={migrations_dir}")
    print(f"  migrations_found={len(migrations)}")
    print(f"  migrations_applied={len(applied)}")
    print(f"  migrations_skipped={len(skipped)}")
    print(f"  legacy_backfilled={backfilled}")
    for name in applied:
        print(f"    +applied  {name}")
    for name in skipped:
        print(f"    =skipped  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
