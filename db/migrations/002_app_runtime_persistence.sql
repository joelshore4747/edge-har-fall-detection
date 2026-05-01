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

CREATE TABLE IF NOT EXISTS app_users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject_key TEXT NOT NULL UNIQUE,
    display_name TEXT,
    latest_device_platform TEXT,
    latest_device_model TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS app_auth_accounts (
    auth_account_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES app_users(user_id) ON DELETE SET NULL,
    login_username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login_at TIMESTAMPTZ,
    UNIQUE (user_id),
    CHECK (role = 'admin' OR user_id IS NOT NULL)
);

CREATE TABLE IF NOT EXISTS app_sessions (
    app_session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES app_users(user_id) ON DELETE CASCADE,
    client_session_id TEXT NOT NULL,
    request_id UUID,
    trace_id TEXT,
    client_version TEXT,
    dataset_name TEXT NOT NULL,
    source_type TEXT NOT NULL,
    task_type TEXT NOT NULL,
    placement_declared TEXT NOT NULL,
    device_platform TEXT NOT NULL,
    device_model TEXT,
    app_version TEXT,
    app_build TEXT,
    recording_mode TEXT NOT NULL,
    runtime_mode TEXT NOT NULL,
    recording_started_at TIMESTAMPTZ,
    recording_ended_at TIMESTAMPTZ,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    sampling_rate_hz DOUBLE PRECISION,
    sample_count INTEGER NOT NULL DEFAULT 0 CHECK (sample_count >= 0),
    has_gyro BOOLEAN NOT NULL DEFAULT FALSE,
    duration_seconds DOUBLE PRECISION,
    notes TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    request_context_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    raw_storage_uri TEXT,
    raw_storage_format TEXT,
    raw_payload_sha256 TEXT,
    raw_payload_bytes BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, client_session_id),
    CHECK (recording_ended_at IS NULL OR recording_started_at IS NULL OR recording_ended_at >= recording_started_at)
);

CREATE TABLE IF NOT EXISTS app_session_inferences (
    inference_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    app_session_id UUID NOT NULL REFERENCES app_sessions(app_session_id) ON DELETE CASCADE,
    request_id UUID,
    api_version TEXT NOT NULL,
    har_model_name TEXT,
    har_model_version TEXT,
    fall_model_name TEXT,
    fall_model_version TEXT,
    status TEXT NOT NULL DEFAULT 'completed' CHECK (status IN ('queued', 'processing', 'completed', 'failed')),
    error_message TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    warning_level TEXT NOT NULL CHECK (warning_level IN ('none', 'low', 'medium', 'high')),
    likely_fall_detected BOOLEAN NOT NULL DEFAULT FALSE,
    top_har_label TEXT,
    top_har_fraction DOUBLE PRECISION,
    grouped_fall_event_count INTEGER NOT NULL DEFAULT 0 CHECK (grouped_fall_event_count >= 0),
    top_fall_probability DOUBLE PRECISION,
    timeline_event_count INTEGER NOT NULL DEFAULT 0 CHECK (timeline_event_count >= 0),
    transition_event_count INTEGER NOT NULL DEFAULT 0 CHECK (transition_event_count >= 0),
    request_options_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    source_summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    placement_summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    har_summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    fall_summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    alert_summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    debug_summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    model_info_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    session_narrative_summary_json JSONB,
    narrative_summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_app_session_inferences_request_id UNIQUE (request_id),
    CONSTRAINT uq_app_session_inferences_session_inference UNIQUE (app_session_id, inference_id),
    CHECK (top_har_fraction IS NULL OR (top_har_fraction >= 0 AND top_har_fraction <= 1)),
    CHECK (top_fall_probability IS NULL OR (top_fall_probability >= 0 AND top_fall_probability <= 1)),
    CHECK (completed_at IS NULL OR completed_at >= started_at),
    CHECK (status <> 'failed' OR error_message IS NOT NULL)
);

CREATE TABLE IF NOT EXISTS app_grouped_fall_events (
    grouped_fall_event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    app_session_id UUID NOT NULL REFERENCES app_sessions(app_session_id) ON DELETE CASCADE,
    inference_id UUID NOT NULL,
    event_key TEXT NOT NULL,
    event_start_ts DOUBLE PRECISION NOT NULL,
    event_end_ts DOUBLE PRECISION NOT NULL,
    event_duration_seconds DOUBLE PRECISION NOT NULL,
    positive_window_count INTEGER NOT NULL CHECK (positive_window_count >= 0),
    peak_probability DOUBLE PRECISION,
    mean_probability DOUBLE PRECISION,
    median_probability DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (inference_id, event_key),
    CONSTRAINT fk_app_grouped_fall_events_session_inference FOREIGN KEY (app_session_id, inference_id)
        REFERENCES app_session_inferences(app_session_id, inference_id)
        ON DELETE CASCADE,
    CHECK (event_end_ts >= event_start_ts),
    CHECK (peak_probability IS NULL OR (peak_probability >= 0 AND peak_probability <= 1)),
    CHECK (mean_probability IS NULL OR (mean_probability >= 0 AND mean_probability <= 1)),
    CHECK (median_probability IS NULL OR (median_probability >= 0 AND median_probability <= 1))
);

CREATE TABLE IF NOT EXISTS app_timeline_events (
    timeline_event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    app_session_id UUID NOT NULL REFERENCES app_sessions(app_session_id) ON DELETE CASCADE,
    inference_id UUID NOT NULL,
    event_key TEXT NOT NULL,
    start_ts DOUBLE PRECISION NOT NULL,
    end_ts DOUBLE PRECISION NOT NULL,
    duration_seconds DOUBLE PRECISION NOT NULL,
    midpoint_ts DOUBLE PRECISION,
    point_count INTEGER NOT NULL DEFAULT 0 CHECK (point_count >= 0),
    activity_label TEXT NOT NULL,
    placement_label TEXT NOT NULL,
    activity_confidence_mean DOUBLE PRECISION,
    placement_confidence_mean DOUBLE PRECISION,
    fall_probability_peak DOUBLE PRECISION,
    fall_probability_mean DOUBLE PRECISION,
    likely_fall BOOLEAN NOT NULL DEFAULT FALSE,
    event_kind TEXT NOT NULL,
    related_grouped_event_ids_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    description TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (inference_id, event_key),
    CONSTRAINT fk_app_timeline_events_session_inference FOREIGN KEY (app_session_id, inference_id)
        REFERENCES app_session_inferences(app_session_id, inference_id)
        ON DELETE CASCADE,
    CHECK (end_ts >= start_ts),
    CHECK (activity_confidence_mean IS NULL OR (activity_confidence_mean >= 0 AND activity_confidence_mean <= 1)),
    CHECK (placement_confidence_mean IS NULL OR (placement_confidence_mean >= 0 AND placement_confidence_mean <= 1)),
    CHECK (fall_probability_peak IS NULL OR (fall_probability_peak >= 0 AND fall_probability_peak <= 1)),
    CHECK (fall_probability_mean IS NULL OR (fall_probability_mean >= 0 AND fall_probability_mean <= 1))
);

CREATE TABLE IF NOT EXISTS app_transition_events (
    app_transition_event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    app_session_id UUID NOT NULL REFERENCES app_sessions(app_session_id) ON DELETE CASCADE,
    inference_id UUID NOT NULL,
    transition_key TEXT NOT NULL,
    transition_ts DOUBLE PRECISION NOT NULL,
    from_event_key TEXT NOT NULL,
    to_event_key TEXT NOT NULL,
    transition_kind TEXT NOT NULL,
    from_activity_label TEXT,
    to_activity_label TEXT,
    from_placement_label TEXT,
    to_placement_label TEXT,
    description TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (inference_id, transition_key),
    CONSTRAINT fk_app_transition_events_session_inference FOREIGN KEY (app_session_id, inference_id)
        REFERENCES app_session_inferences(app_session_id, inference_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS app_feedback (
    feedback_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    app_session_id UUID NOT NULL REFERENCES app_sessions(app_session_id) ON DELETE CASCADE,
    inference_id UUID,
    target_type TEXT NOT NULL DEFAULT 'session'
        CHECK (target_type IN ('session', 'grouped_fall_event', 'timeline_event', 'transition_event', 'window')),
    target_event_key TEXT,
    window_id TEXT,
    feedback_type TEXT NOT NULL CHECK (feedback_type IN ('confirmed_fall', 'false_alarm', 'uncertain', 'corrected_label')),
    corrected_label TEXT,
    reviewer_identifier TEXT,
    subject_key TEXT,
    notes TEXT,
    request_id UUID,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_app_feedback_request_id UNIQUE (request_id),
    CONSTRAINT fk_app_feedback_session_inference FOREIGN KEY (app_session_id, inference_id)
        REFERENCES app_session_inferences(app_session_id, inference_id)
        ON DELETE CASCADE
);

ALTER TABLE app_sessions
    ADD COLUMN IF NOT EXISTS recording_started_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS recording_ended_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS uploaded_at TIMESTAMPTZ;

ALTER TABLE app_session_inferences
    ADD COLUMN IF NOT EXISTS status TEXT,
    ADD COLUMN IF NOT EXISTS error_message TEXT,
    ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS completed_at TIMESTAMPTZ;

ALTER TABLE app_feedback
    ADD COLUMN IF NOT EXISTS target_type TEXT;

DROP TRIGGER IF EXISTS trg_app_users_set_updated_at ON app_users;
CREATE TRIGGER trg_app_users_set_updated_at
BEFORE UPDATE ON app_users
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_app_auth_accounts_set_updated_at ON app_auth_accounts;
CREATE TRIGGER trg_app_auth_accounts_set_updated_at
BEFORE UPDATE ON app_auth_accounts
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_app_sessions_set_updated_at ON app_sessions;
CREATE TRIGGER trg_app_sessions_set_updated_at
BEFORE UPDATE ON app_sessions
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

CREATE INDEX IF NOT EXISTS idx_app_users_subject_key ON app_users (subject_key);
CREATE INDEX IF NOT EXISTS idx_app_auth_accounts_user_id ON app_auth_accounts (user_id);
CREATE INDEX IF NOT EXISTS idx_app_sessions_user_created ON app_sessions (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_app_sessions_client_session ON app_sessions (client_session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_app_session_inferences_session_created ON app_session_inferences (app_session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_app_session_inferences_status_created ON app_session_inferences (status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_app_grouped_fall_events_session_start ON app_grouped_fall_events (app_session_id, event_start_ts);
CREATE INDEX IF NOT EXISTS idx_app_timeline_events_session_start ON app_timeline_events (app_session_id, start_ts);
CREATE INDEX IF NOT EXISTS idx_app_transition_events_session_ts ON app_transition_events (app_session_id, transition_ts);
CREATE INDEX IF NOT EXISTS idx_app_feedback_session_recorded ON app_feedback (app_session_id, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_app_feedback_target_type_recorded ON app_feedback (target_type, recorded_at DESC);
