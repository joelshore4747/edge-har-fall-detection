-- 004_add_basic_auth_accounts.sql
--
-- Intentionally a no-op as of the schema_migrations ledger introduction.
--
-- Earlier revisions of this file re-asserted the app_auth_accounts table,
-- its UNIQUE/CHECK constraints, the trg_app_auth_accounts_set_updated_at
-- trigger, and the idx_app_auth_accounts_user_id index. All of those are
-- already created by 002_app_runtime_persistence.sql. The duplicate
-- statements were idempotent thanks to IF NOT EXISTS guards, but they
-- made the migration narrative harder to read.
--
-- The file is preserved as a placeholder so the schema_migrations ledger
-- continues to record the version on systems that already applied it.
-- Future authentication-related schema changes should land in a new
-- migration file (006_* or later).

SELECT 1;
