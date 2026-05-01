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

ALTER TABLE app_sessions
    ADD COLUMN IF NOT EXISTS recording_started_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS recording_ended_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS uploaded_at TIMESTAMPTZ;

UPDATE app_sessions
SET uploaded_at = COALESCE(uploaded_at, created_at, NOW())
WHERE uploaded_at IS NULL;

ALTER TABLE app_sessions
    ALTER COLUMN uploaded_at SET DEFAULT NOW(),
    ALTER COLUMN uploaded_at SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'ck_app_sessions_recording_window'
          AND conrelid = 'app_sessions'::regclass
    ) THEN
        ALTER TABLE app_sessions
            ADD CONSTRAINT ck_app_sessions_recording_window
            CHECK (recording_ended_at IS NULL OR recording_started_at IS NULL OR recording_ended_at >= recording_started_at);
    END IF;
END;
$$;

ALTER TABLE app_session_inferences
    ADD COLUMN IF NOT EXISTS status TEXT,
    ADD COLUMN IF NOT EXISTS error_message TEXT,
    ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS completed_at TIMESTAMPTZ;

UPDATE app_session_inferences
SET
    status = COALESCE(status, 'completed'),
    started_at = COALESCE(started_at, created_at, NOW()),
    completed_at = CASE
        WHEN completed_at IS NOT NULL THEN completed_at
        WHEN COALESCE(status, 'completed') IN ('completed', 'failed') THEN COALESCE(created_at, NOW())
        ELSE NULL
    END
WHERE status IS NULL
   OR started_at IS NULL
   OR (completed_at IS NULL AND COALESCE(status, 'completed') IN ('completed', 'failed'));

ALTER TABLE app_session_inferences
    ALTER COLUMN status SET DEFAULT 'completed',
    ALTER COLUMN status SET NOT NULL,
    ALTER COLUMN started_at SET DEFAULT NOW(),
    ALTER COLUMN started_at SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'uq_app_session_inferences_session_inference'
          AND conrelid = 'app_session_inferences'::regclass
    ) THEN
        ALTER TABLE app_session_inferences
            ADD CONSTRAINT uq_app_session_inferences_session_inference
            UNIQUE (app_session_id, inference_id);
    END IF;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'uq_app_session_inferences_request_id'
          AND conrelid = 'app_session_inferences'::regclass
    ) THEN
        ALTER TABLE app_session_inferences
            ADD CONSTRAINT uq_app_session_inferences_request_id
            UNIQUE (request_id);
    END IF;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'ck_app_session_inferences_completed_window'
          AND conrelid = 'app_session_inferences'::regclass
    ) THEN
        ALTER TABLE app_session_inferences
            ADD CONSTRAINT ck_app_session_inferences_completed_window
            CHECK (completed_at IS NULL OR completed_at >= started_at);
    END IF;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'ck_app_session_inferences_failed_requires_error'
          AND conrelid = 'app_session_inferences'::regclass
    ) THEN
        ALTER TABLE app_session_inferences
            ADD CONSTRAINT ck_app_session_inferences_failed_requires_error
            CHECK (status <> 'failed' OR error_message IS NOT NULL);
    END IF;
END;
$$;

ALTER TABLE app_feedback
    ADD COLUMN IF NOT EXISTS target_type TEXT;

UPDATE app_feedback AS f
SET target_type = CASE
    WHEN f.window_id IS NOT NULL THEN 'window'
    WHEN f.target_event_key IS NULL THEN 'session'
    WHEN EXISTS (
        SELECT 1
        FROM app_grouped_fall_events AS g
        WHERE g.app_session_id = f.app_session_id
          AND g.inference_id = f.inference_id
          AND g.event_key = f.target_event_key
    ) THEN 'grouped_fall_event'
    WHEN EXISTS (
        SELECT 1
        FROM app_transition_events AS t
        WHERE t.app_session_id = f.app_session_id
          AND t.inference_id = f.inference_id
          AND t.transition_key = f.target_event_key
    ) THEN 'transition_event'
    WHEN EXISTS (
        SELECT 1
        FROM app_timeline_events AS e
        WHERE e.app_session_id = f.app_session_id
          AND e.inference_id = f.inference_id
          AND e.event_key = f.target_event_key
    ) THEN 'timeline_event'
    ELSE 'timeline_event'
END
WHERE target_type IS NULL;

ALTER TABLE app_feedback
    ALTER COLUMN target_type SET DEFAULT 'session',
    ALTER COLUMN target_type SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'ck_app_feedback_target_type'
          AND conrelid = 'app_feedback'::regclass
    ) THEN
        ALTER TABLE app_feedback
            ADD CONSTRAINT ck_app_feedback_target_type
            CHECK (target_type IN ('session', 'grouped_fall_event', 'timeline_event', 'transition_event', 'window'));
    END IF;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'uq_app_feedback_request_id'
          AND conrelid = 'app_feedback'::regclass
    ) THEN
        ALTER TABLE app_feedback
            ADD CONSTRAINT uq_app_feedback_request_id
            UNIQUE (request_id);
    END IF;
END;
$$;

DO $$
DECLARE
    table_name TEXT;
    constraint_name TEXT;
BEGIN
    FOR table_name IN
        SELECT unnest(ARRAY[
            'app_grouped_fall_events',
            'app_timeline_events',
            'app_transition_events',
            'app_feedback'
        ])
    LOOP
        FOR constraint_name IN
            SELECT c.conname
            FROM pg_constraint AS c
            JOIN pg_attribute AS att
              ON att.attrelid = c.conrelid
             AND att.attname = 'inference_id'
            WHERE c.conrelid = to_regclass(table_name)
              AND c.confrelid = 'app_session_inferences'::regclass
              AND c.contype = 'f'
              AND c.conkey = ARRAY[att.attnum]
        LOOP
            EXECUTE format('ALTER TABLE %I DROP CONSTRAINT %I', table_name, constraint_name);
        END LOOP;
    END LOOP;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_app_grouped_fall_events_session_inference'
          AND conrelid = 'app_grouped_fall_events'::regclass
    ) THEN
        ALTER TABLE app_grouped_fall_events
            ADD CONSTRAINT fk_app_grouped_fall_events_session_inference
            FOREIGN KEY (app_session_id, inference_id)
            REFERENCES app_session_inferences(app_session_id, inference_id)
            ON DELETE CASCADE;
    END IF;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_app_timeline_events_session_inference'
          AND conrelid = 'app_timeline_events'::regclass
    ) THEN
        ALTER TABLE app_timeline_events
            ADD CONSTRAINT fk_app_timeline_events_session_inference
            FOREIGN KEY (app_session_id, inference_id)
            REFERENCES app_session_inferences(app_session_id, inference_id)
            ON DELETE CASCADE;
    END IF;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_app_transition_events_session_inference'
          AND conrelid = 'app_transition_events'::regclass
    ) THEN
        ALTER TABLE app_transition_events
            ADD CONSTRAINT fk_app_transition_events_session_inference
            FOREIGN KEY (app_session_id, inference_id)
            REFERENCES app_session_inferences(app_session_id, inference_id)
            ON DELETE CASCADE;
    END IF;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_app_feedback_session_inference'
          AND conrelid = 'app_feedback'::regclass
    ) THEN
        ALTER TABLE app_feedback
            ADD CONSTRAINT fk_app_feedback_session_inference
            FOREIGN KEY (app_session_id, inference_id)
            REFERENCES app_session_inferences(app_session_id, inference_id)
            ON DELETE CASCADE;
    END IF;
END;
$$;

DROP TRIGGER IF EXISTS trg_app_users_set_updated_at ON app_users;
CREATE TRIGGER trg_app_users_set_updated_at
BEFORE UPDATE ON app_users
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_app_sessions_set_updated_at ON app_sessions;
CREATE TRIGGER trg_app_sessions_set_updated_at
BEFORE UPDATE ON app_sessions
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

CREATE INDEX IF NOT EXISTS idx_app_session_inferences_status_created ON app_session_inferences (status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_app_feedback_target_type_recorded ON app_feedback (target_type, recorded_at DESC);
