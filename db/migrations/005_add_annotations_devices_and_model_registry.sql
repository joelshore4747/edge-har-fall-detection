CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = clock_timestamp();
    RETURN NEW;
END;
$$;

CREATE TABLE IF NOT EXISTS app_devices (
    device_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES app_users(user_id) ON DELETE CASCADE,
    device_identifier TEXT NOT NULL,
    identifier_source TEXT NOT NULL DEFAULT 'derived'
        CHECK (identifier_source IN ('explicit', 'derived')),
    device_platform TEXT NOT NULL,
    device_model TEXT,
    app_version TEXT,
    app_build TEXT,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_app_devices_user_identifier UNIQUE (user_id, device_identifier)
);

CREATE TABLE IF NOT EXISTS app_model_registry (
    model_registry_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_type TEXT NOT NULL CHECK (model_type IN ('har', 'fall')),
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL DEFAULT '',
    artifact_path TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_app_model_registry_identity UNIQUE (model_type, model_name, model_version)
);

CREATE TABLE IF NOT EXISTS app_session_annotations (
    annotation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    app_session_id UUID NOT NULL REFERENCES app_sessions(app_session_id) ON DELETE CASCADE,
    annotation_label TEXT NOT NULL
        CHECK (annotation_label IN ('static', 'walking', 'stairs', 'fall', 'other', 'unknown')),
    source TEXT NOT NULL DEFAULT 'mobile'
        CHECK (source IN ('mobile', 'admin', 'system')),
    reviewer_identifier TEXT,
    auth_account_id UUID REFERENCES app_auth_accounts(auth_account_id) ON DELETE SET NULL,
    request_id UUID,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE app_sessions
    ADD COLUMN IF NOT EXISTS device_id UUID REFERENCES app_devices(device_id) ON DELETE SET NULL;

ALTER TABLE app_session_inferences
    ADD COLUMN IF NOT EXISTS har_model_registry_id UUID REFERENCES app_model_registry(model_registry_id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS fall_model_registry_id UUID REFERENCES app_model_registry(model_registry_id) ON DELETE SET NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'uq_app_session_annotations_request_id'
          AND conrelid = 'app_session_annotations'::regclass
    ) THEN
        ALTER TABLE app_session_annotations
            ADD CONSTRAINT uq_app_session_annotations_request_id
            UNIQUE (request_id);
    END IF;
END;
$$;

DROP TRIGGER IF EXISTS trg_app_devices_set_updated_at ON app_devices;
CREATE TRIGGER trg_app_devices_set_updated_at
BEFORE UPDATE ON app_devices
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_app_model_registry_set_updated_at ON app_model_registry;
CREATE TRIGGER trg_app_model_registry_set_updated_at
BEFORE UPDATE ON app_model_registry
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_app_session_annotations_set_updated_at ON app_session_annotations;
CREATE TRIGGER trg_app_session_annotations_set_updated_at
BEFORE UPDATE ON app_session_annotations
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

CREATE INDEX IF NOT EXISTS idx_app_sessions_uploaded_at
    ON app_sessions (uploaded_at DESC);

CREATE INDEX IF NOT EXISTS idx_app_sessions_dataset_uploaded_at
    ON app_sessions (dataset_name, uploaded_at DESC);

CREATE INDEX IF NOT EXISTS idx_app_sessions_device_platform_uploaded_at
    ON app_sessions (device_platform, uploaded_at DESC);

CREATE INDEX IF NOT EXISTS idx_app_sessions_device_id_uploaded_at
    ON app_sessions (device_id, uploaded_at DESC);

CREATE INDEX IF NOT EXISTS idx_app_session_inferences_warning_created
    ON app_session_inferences (warning_level, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_app_session_inferences_likely_fall_created
    ON app_session_inferences (likely_fall_detected, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_app_session_annotations_session_created
    ON app_session_annotations (app_session_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_app_session_annotations_label_created
    ON app_session_annotations (annotation_label, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_app_devices_user_last_seen
    ON app_devices (user_id, last_seen_at DESC);

CREATE INDEX IF NOT EXISTS idx_app_devices_platform_model
    ON app_devices (device_platform, device_model);

CREATE INDEX IF NOT EXISTS idx_app_model_registry_type_last_seen
    ON app_model_registry (model_type, last_seen_at DESC);
