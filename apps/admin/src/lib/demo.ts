const demoSessions = [
  {
    session: {
      app_session_id: 'ufm-demo-session-001',
      user_id: 'demo-user-014',
      subject_id: 'SUBJECT-014',
      client_session_id: 'runtime_capture_2026_04_21_0912',
      dataset_name: 'UFM_RUNTIME_EVAL',
      source_type: 'phone_folder',
      task_type: 'walking_fall_sequence',
      placement_declared: 'trouser_pocket',
      device_platform: 'ios',
      device_model: 'iPhone 15 Pro',
      runtime_mode: 'runtime_capture',
      recording_mode: 'buffered_upload',
      uploaded_at: '2026-04-21T09:12:00Z',
      sample_count: 7680,
      duration_seconds: 384,
      session_name: 'Campus Walkway Trial',
      activity_label: 'walking',
      notes: 'Demo session highlighting clustered fall evidence after a walking sequence.',
      created_at: '2026-04-21T09:13:04Z',
    },
    latest_inference_id: 'demo-inference-001',
    latest_inference_request_id: 'demo-request-001',
    latest_inference_created_at: '2026-04-21T09:13:36Z',
    latest_status: 'completed',
    latest_warning_level: 'high',
    latest_likely_fall_detected: true,
    latest_top_har_label: 'walking',
    latest_top_fall_probability: 0.93,
    latest_grouped_fall_event_count: 2,
  },
  {
    session: {
      app_session_id: 'ufm-demo-session-002',
      user_id: 'demo-user-008',
      subject_id: 'SUBJECT-008',
      client_session_id: 'runtime_capture_2026_04_20_1610',
      dataset_name: 'UFM_RUNTIME_EVAL',
      source_type: 'phone_folder',
      task_type: 'adl_mixed_sequence',
      placement_declared: 'jacket_pocket',
      device_platform: 'android',
      device_model: 'Pixel 8',
      runtime_mode: 'runtime_capture',
      recording_mode: 'buffered_upload',
      uploaded_at: '2026-04-20T16:10:00Z',
      sample_count: 6120,
      duration_seconds: 306,
      session_name: 'Library Corridor ADL',
      activity_label: 'sitting',
      notes: 'Mostly low-risk daily activity with a short balance disturbance.',
      created_at: '2026-04-20T16:11:18Z',
    },
    latest_inference_id: 'demo-inference-002',
    latest_inference_request_id: 'demo-request-002',
    latest_inference_created_at: '2026-04-20T16:12:07Z',
    latest_status: 'completed',
    latest_warning_level: 'medium',
    latest_likely_fall_detected: false,
    latest_top_har_label: 'sitting',
    latest_top_fall_probability: 0.34,
    latest_grouped_fall_event_count: 0,
  },
  {
    session: {
      app_session_id: 'ufm-demo-session-003',
      user_id: 'demo-user-021',
      subject_id: 'SUBJECT-021',
      client_session_id: 'runtime_capture_2026_04_19_1118',
      dataset_name: 'UFM_VALIDATION',
      source_type: 'phone_folder',
      task_type: 'standing_transition_sequence',
      placement_declared: 'handheld',
      device_platform: 'ios',
      device_model: 'iPhone 14',
      runtime_mode: 'runtime_capture',
      recording_mode: 'buffered_upload',
      uploaded_at: '2026-04-19T11:18:00Z',
      sample_count: 4980,
      duration_seconds: 249,
      session_name: 'Office Standing Trial',
      activity_label: 'standing',
      notes: 'Stable standing and sit-to-stand transitions with low alert confidence.',
      created_at: '2026-04-19T11:18:51Z',
    },
    latest_inference_id: 'demo-inference-003',
    latest_inference_request_id: 'demo-request-003',
    latest_inference_created_at: '2026-04-19T11:19:22Z',
    latest_status: 'completed',
    latest_warning_level: 'low',
    latest_likely_fall_detected: false,
    latest_top_har_label: 'standing',
    latest_top_fall_probability: 0.12,
    latest_grouped_fall_event_count: 0,
  },
  {
    session: {
      app_session_id: 'ufm-demo-session-004',
      user_id: 'demo-user-005',
      subject_id: 'SUBJECT-005',
      client_session_id: 'runtime_capture_2026_04_18_1435',
      dataset_name: 'UFM_RUNTIME_EVAL',
      source_type: 'phone_folder',
      task_type: 'stair_transition_sequence',
      placement_declared: 'waist_band',
      device_platform: 'android',
      device_model: 'Samsung S24',
      runtime_mode: 'runtime_capture',
      recording_mode: 'buffered_upload',
      uploaded_at: '2026-04-18T14:35:00Z',
      sample_count: 7020,
      duration_seconds: 351,
      session_name: 'Stairwell Descent Trial',
      activity_label: 'descending_stairs',
      notes: 'Includes abrupt posture change and one high-confidence fall-like segment.',
      created_at: '2026-04-18T14:35:44Z',
    },
    latest_inference_id: 'demo-inference-004',
    latest_inference_request_id: 'demo-request-004',
    latest_inference_created_at: '2026-04-18T14:36:20Z',
    latest_status: 'completed',
    latest_warning_level: 'high',
    latest_likely_fall_detected: true,
    latest_top_har_label: 'descending_stairs',
    latest_top_fall_probability: 0.88,
    latest_grouped_fall_event_count: 1,
  },
  {
    session: {
      app_session_id: 'ufm-demo-session-005',
      user_id: 'demo-user-019',
      subject_id: 'SUBJECT-019',
      client_session_id: 'runtime_capture_2026_04_17_1004',
      dataset_name: 'UFM_VALIDATION',
      source_type: 'phone_folder',
      task_type: 'balance_recovery_sequence',
      placement_declared: 'trouser_pocket',
      device_platform: 'ios',
      device_model: 'iPhone 13',
      runtime_mode: 'runtime_capture',
      recording_mode: 'buffered_upload',
      uploaded_at: '2026-04-17T10:04:00Z',
      sample_count: 5890,
      duration_seconds: 295,
      session_name: 'Balance Recovery Validation',
      activity_label: 'walking',
      notes: 'Processing demo item used to show intermediate operational states.',
      created_at: '2026-04-17T10:05:06Z',
    },
    latest_inference_id: 'demo-inference-005',
    latest_inference_request_id: 'demo-request-005',
    latest_inference_created_at: '2026-04-17T10:05:44Z',
    latest_status: 'processing',
    latest_warning_level: 'medium',
    latest_likely_fall_detected: false,
    latest_top_har_label: 'walking',
    latest_top_fall_probability: 0.51,
    latest_grouped_fall_event_count: 0,
  },
  {
    session: {
      app_session_id: 'ufm-demo-session-006',
      user_id: 'demo-user-031',
      subject_id: 'SUBJECT-031',
      client_session_id: 'runtime_capture_2026_04_15_0831',
      dataset_name: 'UFM_RUNTIME_EVAL',
      source_type: 'phone_folder',
      task_type: 'sensor_dropout_check',
      placement_declared: 'bag',
      device_platform: 'android',
      device_model: 'OnePlus 12',
      runtime_mode: 'runtime_capture',
      recording_mode: 'buffered_upload',
      uploaded_at: '2026-04-15T08:31:00Z',
      sample_count: 0,
      duration_seconds: null,
      session_name: 'Dropped Sensor Packet Review',
      activity_label: null,
      notes: 'Failed import retained for admin review and QA follow-up.',
      created_at: '2026-04-15T08:31:39Z',
    },
    latest_inference_id: null,
    latest_inference_request_id: null,
    latest_inference_created_at: null,
    latest_status: 'failed',
    latest_warning_level: null,
    latest_likely_fall_detected: null,
    latest_top_har_label: null,
    latest_top_fall_probability: null,
    latest_grouped_fall_event_count: null,
  },
]

type DemoSessionListItem = (typeof demoSessions)[number]
type DemoSortValue = string | number | null | undefined
type DemoSessionsParams = {
  page: number
  pageSize: number
  search?: string
  warningLevel?: string
  devicePlatform?: string
  dateFrom?: string
  dateTo?: string
  likelyFall?: boolean
  status?: string
  sortBy?: string
  sortDir?: 'asc' | 'desc'
}

const demoSessionDetails = {
  'ufm-demo-session-001': {
    session: demoSessions[0].session,
    latest_inference: {
      inference_id: 'demo-inference-001',
      request_id: 'demo-request-001',
      status: 'completed',
      error_message: null,
      started_at: '2026-04-21T09:13:05Z',
      completed_at: '2026-04-21T09:13:36Z',
      created_at: '2026-04-21T09:13:36Z',
      response: {
        request_id: 'demo-request-001',
        session_id: 'ufm-demo-session-001',
        persisted_user_id: 'demo-user-014',
        persisted_session_id: 'ufm-demo-session-001',
        persisted_inference_id: 'demo-inference-001',
        source_summary: {
          input_sample_count: 7680,
          session_duration_seconds: 384,
          estimated_sampling_rate_hz: 20,
        },
        placement_summary: {
          placement_state: 'trouser_pocket',
          placement_confidence: 0.94,
        },
        har_summary: {
          top_label: 'walking',
          top_label_fraction: 0.41,
          total_windows: 128,
        },
        fall_summary: {
          likely_fall_detected: true,
          positive_window_count: 9,
          grouped_event_count: 2,
          top_fall_probability: 0.93,
          mean_fall_probability: 0.34,
        },
        alert_summary: {
          warning_level: 'high',
          likely_fall_detected: true,
          top_har_label: 'walking',
          top_har_fraction: 0.41,
          grouped_fall_event_count: 2,
          top_fall_probability: 0.93,
          top_vulnerability_score: 0.88,
          latest_vulnerability_level: 'high',
          latest_monitoring_state: 'alert',
          latest_fall_event_state: 'escalated',
          recommended_message: 'Immediate review recommended due to clustered high-probability fall evidence.',
        },
        model_info: {
          har_model_name: 'UniFall HAR Transformer',
          har_model_version: '2026.1',
          fall_model_name: 'UniFall FallNet Lite',
          fall_model_version: '2026.1',
          api_version: '1.0.0',
        },
        grouped_fall_events: [
          {
            event_id: 'gfe-001',
            event_start_ts: 162.2,
            event_end_ts: 171.8,
            event_duration_seconds: 9.6,
            n_positive_windows: 5,
            peak_probability: 0.93,
            mean_probability: 0.82,
            median_probability: 0.84,
          },
          {
            event_id: 'gfe-002',
            event_start_ts: 214.6,
            event_end_ts: 219.4,
            event_duration_seconds: 4.8,
            n_positive_windows: 4,
            peak_probability: 0.79,
            mean_probability: 0.73,
            median_probability: 0.74,
          },
        ],
        timeline_events: [
          {
            event_id: 'tle-001',
            start_ts: 0,
            end_ts: 142.5,
            duration_seconds: 142.5,
            midpoint_ts: 71.2,
            point_count: 48,
            activity_label: 'walking',
            placement_label: 'trouser_pocket',
            activity_confidence_mean: 0.88,
            placement_confidence_mean: 0.94,
            fall_probability_peak: 0.19,
            fall_probability_mean: 0.08,
            likely_fall: false,
            event_kind: 'normal_activity',
            related_grouped_fall_event_ids: [],
            description: 'Steady walking segment before the alert region begins.',
          },
          {
            event_id: 'tle-002',
            start_ts: 162.2,
            end_ts: 171.8,
            duration_seconds: 9.6,
            midpoint_ts: 167.0,
            point_count: 16,
            activity_label: 'fall_like_motion',
            placement_label: 'trouser_pocket',
            activity_confidence_mean: 0.63,
            placement_confidence_mean: 0.92,
            fall_probability_peak: 0.93,
            fall_probability_mean: 0.82,
            likely_fall: true,
            event_kind: 'grouped_fall_event',
            related_grouped_fall_event_ids: ['gfe-001'],
            description: 'Rapid deceleration and impact-like motion with strong fall confidence.',
          },
          {
            event_id: 'tle-003',
            start_ts: 214.6,
            end_ts: 219.4,
            duration_seconds: 4.8,
            midpoint_ts: 217.0,
            point_count: 10,
            activity_label: 'recovery_motion',
            placement_label: 'trouser_pocket',
            activity_confidence_mean: 0.55,
            placement_confidence_mean: 0.9,
            fall_probability_peak: 0.79,
            fall_probability_mean: 0.73,
            likely_fall: true,
            event_kind: 'grouped_fall_event',
            related_grouped_fall_event_ids: ['gfe-002'],
            description: 'Short secondary event consistent with post-impact recovery movement.',
          },
        ],
        transition_events: [
          {
            transition_id: 'tte-001',
            transition_ts: 160.8,
            from_event_id: 'tle-001',
            to_event_id: 'tle-002',
            transition_kind: 'walking_to_fall',
            from_activity_label: 'walking',
            to_activity_label: 'fall_like_motion',
            from_placement_label: 'trouser_pocket',
            to_placement_label: 'trouser_pocket',
            description: 'Shift from normal gait into high-acceleration fall-like movement.',
          },
          {
            transition_id: 'tte-002',
            transition_ts: 212.9,
            from_event_id: 'tle-002',
            to_event_id: 'tle-003',
            transition_kind: 'impact_to_recovery',
            from_activity_label: 'fall_like_motion',
            to_activity_label: 'recovery_motion',
            from_placement_label: 'trouser_pocket',
            to_placement_label: 'trouser_pocket',
            description: 'Post-impact recovery phase detected after the primary event cluster.',
          },
        ],
        session_narrative_summary: {
          session_id: 'ufm-demo-session-001',
          dataset_name: 'UFM_RUNTIME_EVAL',
          subject_id: 'SUBJECT-014',
          total_duration_seconds: 384,
          event_count: 3,
          transition_count: 2,
          fall_event_count: 2,
          dominant_activity_label: 'walking',
          dominant_placement_label: 'trouser_pocket',
          highest_fall_probability: 0.93,
          summary_text: 'This session is dominated by walking activity before two clustered fall-like events appear, including one high-confidence impact pattern followed by a short recovery sequence.',
        },
        narrative_summary: {
          summary_text: 'Walking dominates the session until a brief, high-confidence fall region and recovery motion trigger a high warning level.',
        },
      },
    },
    feedback: [
      {
        feedback_id: 'feedback-001',
        target_type: 'grouped_fall_event',
        target_event_key: 'gfe-001',
        window_id: null,
        feedback_type: 'confirmed',
        corrected_label: 'fall',
        reviewer_identifier: 'demo-reviewer',
        subject_key: 'SUBJECT-014',
        notes: 'Primary grouped event aligns with the annotated impact section in the evaluation notes.',
        request_id: 'demo-request-001',
        recorded_at: '2026-04-21T10:02:00Z',
      },
    ],
    annotations: [
      {
        annotation_id: 'annotation-001',
        app_session_id: 'ufm-demo-session-001',
        label: 'fall',
        source: 'admin',
        reviewer_identifier: 'demo-reviewer',
        auth_account_id: null,
        created_by_username: 'demo-admin',
        request_id: 'demo-request-001',
        notes: 'Confirmed clustered fall evidence after narrative review and grouped-event inspection.',
        created_at: '2026-04-21T10:05:00Z',
        updated_at: '2026-04-21T10:05:00Z',
      },
    ],
  },
  'ufm-demo-session-002': {
    session: demoSessions[1].session,
    latest_inference: {
      inference_id: 'demo-inference-002',
      request_id: 'demo-request-002',
      status: 'completed',
      error_message: null,
      started_at: '2026-04-20T16:11:22Z',
      completed_at: '2026-04-20T16:12:07Z',
      created_at: '2026-04-20T16:12:07Z',
      response: {
        request_id: 'demo-request-002',
        session_id: 'ufm-demo-session-002',
        persisted_user_id: 'demo-user-008',
        persisted_session_id: 'ufm-demo-session-002',
        persisted_inference_id: 'demo-inference-002',
        source_summary: {
          input_sample_count: 6120,
          session_duration_seconds: 306,
          estimated_sampling_rate_hz: 20,
        },
        placement_summary: {
          placement_state: 'jacket_pocket',
          placement_confidence: 0.89,
        },
        har_summary: {
          top_label: 'sitting',
          top_label_fraction: 0.52,
          total_windows: 102,
        },
        fall_summary: {
          likely_fall_detected: false,
          positive_window_count: 1,
          grouped_event_count: 0,
          top_fall_probability: 0.34,
          mean_fall_probability: 0.11,
        },
        alert_summary: {
          warning_level: 'medium',
          likely_fall_detected: false,
          top_har_label: 'sitting',
          top_har_fraction: 0.52,
          grouped_fall_event_count: 0,
          top_fall_probability: 0.34,
          top_vulnerability_score: 0.46,
          latest_vulnerability_level: 'guarded',
          latest_monitoring_state: 'watch',
          latest_fall_event_state: 'none',
          recommended_message: 'Review optional: slight instability pattern detected, but no grouped fall evidence.',
        },
        model_info: {
          har_model_name: 'UniFall HAR Transformer',
          har_model_version: '2026.1',
          fall_model_name: 'UniFall FallNet Lite',
          fall_model_version: '2026.1',
          api_version: '1.0.0',
        },
        grouped_fall_events: [],
        timeline_events: [
          {
            event_id: 'tle-101',
            start_ts: 0,
            end_ts: 188,
            duration_seconds: 188,
            midpoint_ts: 94,
            point_count: 63,
            activity_label: 'sitting',
            placement_label: 'jacket_pocket',
            activity_confidence_mean: 0.86,
            placement_confidence_mean: 0.89,
            fall_probability_peak: 0.12,
            fall_probability_mean: 0.05,
            likely_fall: false,
            event_kind: 'normal_activity',
            related_grouped_fall_event_ids: [],
            description: 'Extended sitting segment with stable placement and low fall confidence.',
          },
          {
            event_id: 'tle-102',
            start_ts: 188,
            end_ts: 227,
            duration_seconds: 39,
            midpoint_ts: 207.5,
            point_count: 13,
            activity_label: 'standing',
            placement_label: 'jacket_pocket',
            activity_confidence_mean: 0.67,
            placement_confidence_mean: 0.84,
            fall_probability_peak: 0.34,
            fall_probability_mean: 0.18,
            likely_fall: false,
            event_kind: 'balance_disturbance',
            related_grouped_fall_event_ids: [],
            description: 'Short postural disturbance with no clustered fall-event evidence.',
          },
        ],
        transition_events: [
          {
            transition_id: 'tte-101',
            transition_ts: 187.4,
            from_event_id: 'tle-101',
            to_event_id: 'tle-102',
            transition_kind: 'sit_to_stand',
            from_activity_label: 'sitting',
            to_activity_label: 'standing',
            from_placement_label: 'jacket_pocket',
            to_placement_label: 'jacket_pocket',
            description: 'Transition out of seated posture into a short standing disturbance.',
          },
        ],
        session_narrative_summary: {
          session_id: 'ufm-demo-session-002',
          dataset_name: 'UFM_RUNTIME_EVAL',
          subject_id: 'SUBJECT-008',
          total_duration_seconds: 306,
          event_count: 2,
          transition_count: 1,
          fall_event_count: 0,
          dominant_activity_label: 'sitting',
          dominant_placement_label: 'jacket_pocket',
          highest_fall_probability: 0.34,
          summary_text: 'The session is largely seated activity with one brief balance disturbance, resulting in a medium warning level but no grouped fall event.',
        },
        narrative_summary: {
          summary_text: 'Mostly normal activity with a short period of mild instability and no confirmed fall evidence.',
        },
      },
    },
    feedback: [],
  },
  'ufm-demo-session-003': {
    session: demoSessions[2].session,
    latest_inference: {
      inference_id: 'demo-inference-003',
      request_id: 'demo-request-003',
      status: 'completed',
      error_message: null,
      started_at: '2026-04-19T11:18:56Z',
      completed_at: '2026-04-19T11:19:22Z',
      created_at: '2026-04-19T11:19:22Z',
      response: {
        request_id: 'demo-request-003',
        session_id: 'ufm-demo-session-003',
        persisted_user_id: 'demo-user-021',
        persisted_session_id: 'ufm-demo-session-003',
        persisted_inference_id: 'demo-inference-003',
        source_summary: {
          input_sample_count: 4980,
          session_duration_seconds: 249,
          estimated_sampling_rate_hz: 20,
        },
        placement_summary: {
          placement_state: 'handheld',
          placement_confidence: 0.78,
        },
        har_summary: {
          top_label: 'standing',
          top_label_fraction: 0.57,
          total_windows: 83,
        },
        fall_summary: {
          likely_fall_detected: false,
          positive_window_count: 0,
          grouped_event_count: 0,
          top_fall_probability: 0.12,
          mean_fall_probability: 0.04,
        },
        alert_summary: {
          warning_level: 'low',
          likely_fall_detected: false,
          top_har_label: 'standing',
          top_har_fraction: 0.57,
          grouped_fall_event_count: 0,
          top_fall_probability: 0.12,
          top_vulnerability_score: 0.21,
          latest_vulnerability_level: 'low',
          latest_monitoring_state: 'stable',
          latest_fall_event_state: 'none',
          recommended_message: 'No intervention needed. Session appears stable.',
        },
        model_info: {
          har_model_name: 'UniFall HAR Transformer',
          har_model_version: '2026.1',
          fall_model_name: 'UniFall FallNet Lite',
          fall_model_version: '2026.1',
          api_version: '1.0.0',
        },
        grouped_fall_events: [],
        timeline_events: [
          {
            event_id: 'tle-201',
            start_ts: 0,
            end_ts: 96,
            duration_seconds: 96,
            midpoint_ts: 48,
            point_count: 32,
            activity_label: 'standing',
            placement_label: 'handheld',
            activity_confidence_mean: 0.84,
            placement_confidence_mean: 0.78,
            fall_probability_peak: 0.09,
            fall_probability_mean: 0.03,
            likely_fall: false,
            event_kind: 'normal_activity',
            related_grouped_fall_event_ids: [],
            description: 'Stable standing with low fall probability.',
          },
          {
            event_id: 'tle-202',
            start_ts: 96,
            end_ts: 121,
            duration_seconds: 25,
            midpoint_ts: 108.5,
            point_count: 8,
            activity_label: 'sit_to_stand',
            placement_label: 'handheld',
            activity_confidence_mean: 0.72,
            placement_confidence_mean: 0.74,
            fall_probability_peak: 0.12,
            fall_probability_mean: 0.07,
            likely_fall: false,
            event_kind: 'transition',
            related_grouped_fall_event_ids: [],
            description: 'Expected sit-to-stand transition with low risk profile.',
          },
        ],
        transition_events: [
          {
            transition_id: 'tte-201',
            transition_ts: 95.8,
            from_event_id: 'tle-201',
            to_event_id: 'tle-202',
            transition_kind: 'standing_transition',
            from_activity_label: 'standing',
            to_activity_label: 'sit_to_stand',
            from_placement_label: 'handheld',
            to_placement_label: 'handheld',
            description: 'Routine transition with consistent placement confidence.',
          },
        ],
        session_narrative_summary: {
          session_id: 'ufm-demo-session-003',
          dataset_name: 'UFM_VALIDATION',
          subject_id: 'SUBJECT-021',
          total_duration_seconds: 249,
          event_count: 2,
          transition_count: 1,
          fall_event_count: 0,
          dominant_activity_label: 'standing',
          dominant_placement_label: 'handheld',
          highest_fall_probability: 0.12,
          summary_text: 'This validation run is dominated by standing and expected posture transitions with a consistently low-risk inference profile.',
        },
        narrative_summary: {
          summary_text: 'Stable validation session with no meaningful fall evidence.',
        },
      },
    },
    feedback: [],
  },
  'ufm-demo-session-004': {
    session: demoSessions[3].session,
    latest_inference: {
      inference_id: 'demo-inference-004',
      request_id: 'demo-request-004',
      status: 'completed',
      error_message: null,
      started_at: '2026-04-18T14:35:51Z',
      completed_at: '2026-04-18T14:36:20Z',
      created_at: '2026-04-18T14:36:20Z',
      response: {
        request_id: 'demo-request-004',
        session_id: 'ufm-demo-session-004',
        persisted_user_id: 'demo-user-005',
        persisted_session_id: 'ufm-demo-session-004',
        persisted_inference_id: 'demo-inference-004',
        source_summary: {
          input_sample_count: 7020,
          session_duration_seconds: 351,
          estimated_sampling_rate_hz: 20,
        },
        placement_summary: {
          placement_state: 'waist_band',
          placement_confidence: 0.86,
        },
        har_summary: {
          top_label: 'descending_stairs',
          top_label_fraction: 0.47,
          total_windows: 117,
        },
        fall_summary: {
          likely_fall_detected: true,
          positive_window_count: 4,
          grouped_event_count: 1,
          top_fall_probability: 0.88,
          mean_fall_probability: 0.28,
        },
        alert_summary: {
          warning_level: 'high',
          likely_fall_detected: true,
          top_har_label: 'descending_stairs',
          top_har_fraction: 0.47,
          grouped_fall_event_count: 1,
          top_fall_probability: 0.88,
          top_vulnerability_score: 0.81,
          latest_vulnerability_level: 'high',
          latest_monitoring_state: 'alert',
          latest_fall_event_state: 'raised',
          recommended_message: 'High-priority review suggested for abrupt stair-descending anomaly.',
        },
        model_info: {
          har_model_name: 'UniFall HAR Transformer',
          har_model_version: '2026.1',
          fall_model_name: 'UniFall FallNet Lite',
          fall_model_version: '2026.1',
          api_version: '1.0.0',
        },
        grouped_fall_events: [
          {
            event_id: 'gfe-401',
            event_start_ts: 141.4,
            event_end_ts: 147.6,
            event_duration_seconds: 6.2,
            n_positive_windows: 4,
            peak_probability: 0.88,
            mean_probability: 0.76,
            median_probability: 0.77,
          },
        ],
        timeline_events: [
          {
            event_id: 'tle-401',
            start_ts: 0,
            end_ts: 138,
            duration_seconds: 138,
            midpoint_ts: 69,
            point_count: 46,
            activity_label: 'descending_stairs',
            placement_label: 'waist_band',
            activity_confidence_mean: 0.83,
            placement_confidence_mean: 0.86,
            fall_probability_peak: 0.22,
            fall_probability_mean: 0.09,
            likely_fall: false,
            event_kind: 'normal_activity',
            related_grouped_fall_event_ids: [],
            description: 'Consistent stair descent before the abrupt anomaly.',
          },
          {
            event_id: 'tle-402',
            start_ts: 141.4,
            end_ts: 147.6,
            duration_seconds: 6.2,
            midpoint_ts: 144.5,
            point_count: 10,
            activity_label: 'fall_like_motion',
            placement_label: 'waist_band',
            activity_confidence_mean: 0.61,
            placement_confidence_mean: 0.84,
            fall_probability_peak: 0.88,
            fall_probability_mean: 0.76,
            likely_fall: true,
            event_kind: 'grouped_fall_event',
            related_grouped_fall_event_ids: ['gfe-401'],
            description: 'Abrupt event during stair descent with high impact signature.',
          },
        ],
        transition_events: [
          {
            transition_id: 'tte-401',
            transition_ts: 140.9,
            from_event_id: 'tle-401',
            to_event_id: 'tle-402',
            transition_kind: 'stair_descent_to_fall',
            from_activity_label: 'descending_stairs',
            to_activity_label: 'fall_like_motion',
            from_placement_label: 'waist_band',
            to_placement_label: 'waist_band',
            description: 'Sudden shift out of stable stair descent into a fall-like event.',
          },
        ],
        session_narrative_summary: {
          session_id: 'ufm-demo-session-004',
          dataset_name: 'UFM_RUNTIME_EVAL',
          subject_id: 'SUBJECT-005',
          total_duration_seconds: 351,
          event_count: 2,
          transition_count: 1,
          fall_event_count: 1,
          dominant_activity_label: 'descending_stairs',
          dominant_placement_label: 'waist_band',
          highest_fall_probability: 0.88,
          summary_text: 'Most of the session contains ordinary stair descent, followed by a single high-confidence anomaly that pushes the warning level to high.',
        },
        narrative_summary: {
          summary_text: 'Stair descent stays stable until one concentrated high-probability fall-like event appears.',
        },
      },
    },
    feedback: [
      {
        feedback_id: 'feedback-401',
        target_type: 'session',
        target_event_key: null,
        window_id: null,
        feedback_type: 'reviewed',
        corrected_label: null,
        reviewer_identifier: 'demo-reviewer',
        subject_key: 'SUBJECT-005',
        notes: 'Useful dissertation example of a high-warning session with a single grouped event.',
        request_id: 'demo-request-004',
        recorded_at: '2026-04-18T15:12:00Z',
      },
    ],
    annotations: [
      {
        annotation_id: 'annotation-401',
        app_session_id: 'ufm-demo-session-004',
        label: 'fall',
        source: 'admin',
        reviewer_identifier: 'demo-reviewer',
        auth_account_id: null,
        created_by_username: 'demo-admin',
        request_id: 'demo-request-004',
        notes: 'Retained as a strong dissertation example of a single-event high-warning session.',
        created_at: '2026-04-18T15:14:00Z',
        updated_at: '2026-04-18T15:14:00Z',
      },
    ],
  },
  'ufm-demo-session-005': {
    session: demoSessions[4].session,
    latest_inference: {
      inference_id: 'demo-inference-005',
      request_id: 'demo-request-005',
      status: 'processing',
      error_message: null,
      started_at: '2026-04-17T10:05:12Z',
      completed_at: null,
      created_at: '2026-04-17T10:05:44Z',
      response: {
        request_id: 'demo-request-005',
        session_id: 'ufm-demo-session-005',
        persisted_user_id: 'demo-user-019',
        persisted_session_id: 'ufm-demo-session-005',
        persisted_inference_id: 'demo-inference-005',
        source_summary: {
          input_sample_count: 5890,
          session_duration_seconds: 295,
          estimated_sampling_rate_hz: 20,
        },
        placement_summary: {
          placement_state: 'trouser_pocket',
          placement_confidence: 0.91,
        },
        har_summary: {
          top_label: 'walking',
          top_label_fraction: 0.38,
          total_windows: 98,
        },
        fall_summary: {
          likely_fall_detected: false,
          positive_window_count: 2,
          grouped_event_count: 0,
          top_fall_probability: 0.51,
          mean_fall_probability: 0.17,
        },
        alert_summary: {
          warning_level: 'medium',
          likely_fall_detected: false,
          top_har_label: 'walking',
          top_har_fraction: 0.38,
          grouped_fall_event_count: 0,
          top_fall_probability: 0.51,
          top_vulnerability_score: 0.58,
          latest_vulnerability_level: 'guarded',
          latest_monitoring_state: 'processing',
          latest_fall_event_state: 'pending',
          recommended_message: 'Inference still processing. Review once the final session summary is saved.',
        },
        model_info: {
          har_model_name: 'UniFall HAR Transformer',
          har_model_version: '2026.1',
          fall_model_name: 'UniFall FallNet Lite',
          fall_model_version: '2026.1',
          api_version: '1.0.0',
        },
        grouped_fall_events: [],
        timeline_events: [
          {
            event_id: 'tle-501',
            start_ts: 0,
            end_ts: 124,
            duration_seconds: 124,
            midpoint_ts: 62,
            point_count: 41,
            activity_label: 'walking',
            placement_label: 'trouser_pocket',
            activity_confidence_mean: 0.75,
            placement_confidence_mean: 0.91,
            fall_probability_peak: 0.31,
            fall_probability_mean: 0.12,
            likely_fall: false,
            event_kind: 'processing_window',
            related_grouped_fall_event_ids: [],
            description: 'Intermediate processing output captured before final review is complete.',
          },
        ],
        transition_events: [],
        session_narrative_summary: {
          session_id: 'ufm-demo-session-005',
          dataset_name: 'UFM_VALIDATION',
          subject_id: 'SUBJECT-019',
          total_duration_seconds: 295,
          event_count: 1,
          transition_count: 0,
          fall_event_count: 0,
          dominant_activity_label: 'walking',
          dominant_placement_label: 'trouser_pocket',
          highest_fall_probability: 0.51,
          summary_text: 'The inference is still in progress, but early windows suggest normal walking with one moderate-confidence anomaly that should be reviewed after completion.',
        },
        narrative_summary: {
          summary_text: 'Processing is still underway, so this page shows the intermediate saved state.',
        },
      },
    },
    feedback: [],
  },
  'ufm-demo-session-006': {
    session: demoSessions[5].session,
    latest_inference: null,
    feedback: [
      {
        feedback_id: 'feedback-601',
        target_type: 'session',
        target_event_key: null,
        window_id: null,
        feedback_type: 'flagged',
        corrected_label: null,
        reviewer_identifier: 'demo-reviewer',
        subject_key: 'SUBJECT-031',
        notes: 'Sensor import failed before inference. Retained here to show dashboard coverage of QA edge cases.',
        request_id: null,
        recorded_at: '2026-04-15T09:02:00Z',
      },
    ],
    annotations: [
      {
        annotation_id: 'annotation-601',
        app_session_id: 'ufm-demo-session-006',
        label: 'unknown',
        source: 'system',
        reviewer_identifier: null,
        auth_account_id: null,
        created_by_username: null,
        request_id: null,
        notes: 'Marked unknown because ingestion failed before a reviewable inference could be produced.',
        created_at: '2026-04-15T09:03:00Z',
        updated_at: '2026-04-15T09:03:00Z',
      },
    ],
  },
}

type DemoSessionDetail = (typeof demoSessionDetails)[keyof typeof demoSessionDetails]
type DemoEvidenceSection =
  | 'grouped_fall_events'
  | 'timeline_events'
  | 'transition_events'
  | 'feedback'
  | 'annotations'

const ALL_DEMO_EVIDENCE_SECTIONS: DemoEvidenceSection[] = [
  'grouped_fall_events',
  'timeline_events',
  'transition_events',
  'feedback',
  'annotations',
]

function compareValues(left: DemoSortValue, right: DemoSortValue) {
  if (left === right) {
    return 0
  }
  if (left === null || left === undefined) {
    return -1
  }
  if (right === null || right === undefined) {
    return 1
  }
  if (typeof left === 'number' && typeof right === 'number') {
    return left - right
  }
  return String(left).localeCompare(String(right))
}

function getSortValue(item: DemoSessionListItem, sortBy: string): DemoSortValue {
  switch (sortBy) {
    case 'uploaded_at':
      return item.session.uploaded_at
    case 'sample_count':
      return item.session.sample_count
    case 'device_platform':
      return item.session.device_platform
    case 'status':
      return item.latest_status
    case 'warning_level':
      return item.latest_warning_level
    case 'top_fall_probability':
      return item.latest_top_fall_probability
    case 'grouped_fall_event_count':
      return item.latest_grouped_fall_event_count
    case 'created_at':
    default:
      return item.session.created_at
  }
}

export function buildDemoAdminOverview() {
  return {
    totals: {
      users: 6,
      sessions: demoSessions.length,
      inferences: 5,
      grouped_fall_events: 3,
    },
    recent_activity: {
      sessions_last_7_days: 5,
      sessions_with_likely_fall: 2,
    },
    charts: {
      sessions_by_day: [
        { label: '2026-04-15', value: 1 },
        { label: '2026-04-16', value: 0 },
        { label: '2026-04-17', value: 1 },
        { label: '2026-04-18', value: 1 },
        { label: '2026-04-19', value: 1 },
        { label: '2026-04-20', value: 1 },
        { label: '2026-04-21', value: 1 },
      ],
      fall_events_by_day: [
        { label: '2026-04-15', value: 0 },
        { label: '2026-04-16', value: 0 },
        { label: '2026-04-17', value: 0 },
        { label: '2026-04-18', value: 1 },
        { label: '2026-04-19', value: 0 },
        { label: '2026-04-20', value: 0 },
        { label: '2026-04-21', value: 2 },
      ],
      warning_level_distribution: [
        { label: 'high', value: 2 },
        { label: 'medium', value: 2 },
        { label: 'low', value: 1 },
        { label: 'none', value: 1 },
      ],
      top_har_labels: [
        { label: 'walking', value: 2 },
        { label: 'standing', value: 1 },
        { label: 'sitting', value: 1 },
        { label: 'descending_stairs', value: 1 },
      ],
    },
    recent_sessions: [...demoSessions]
      .sort((left: DemoSessionListItem, right: DemoSessionListItem) =>
        compareValues(right.session.uploaded_at, left.session.uploaded_at),
      )
      .slice(0, 4),
  }
}

export function buildDemoSessionListResponse(params: DemoSessionsParams) {
  const normalizedSearch = params.search?.trim().toLowerCase()

  const filtered = demoSessions
    .filter((item) => {
      if (!normalizedSearch) {
        return true
      }

      const haystack = [
        item.session.app_session_id,
        item.session.subject_id,
        item.session.client_session_id,
        item.session.session_name,
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase()

      return haystack.includes(normalizedSearch)
    })
    .filter((item) => {
      if (!params.warningLevel) {
        return true
      }

      if (params.warningLevel === 'none') {
        return !item.latest_warning_level
      }

      return item.latest_warning_level === params.warningLevel
    })
    .filter((item) => {
      if (!params.devicePlatform) {
        return true
      }

      return item.session.device_platform === params.devicePlatform
    })
    .filter((item) => {
      if (!params.dateFrom) {
        return true
      }

      return new Date(item.session.uploaded_at) >= new Date(`${params.dateFrom}T00:00:00Z`)
    })
    .filter((item) => {
      if (!params.dateTo) {
        return true
      }

      return new Date(item.session.uploaded_at) < new Date(`${params.dateTo}T23:59:59.999Z`)
    })
    .filter((item) => {
      if (params.status === undefined || params.status === '') {
        return true
      }
      return item.latest_status === params.status
    })
    .filter((item) => {
      if (params.likelyFall === undefined) {
        return true
      }
      return item.latest_likely_fall_detected === params.likelyFall
    })

  const sortBy = params.sortBy || 'created_at'
  const sortDir = params.sortDir || 'desc'
  const sorted = [...filtered].sort((left: DemoSessionListItem, right: DemoSessionListItem) => {
    const comparison = compareValues(getSortValue(left, sortBy), getSortValue(right, sortBy))
    return sortDir === 'asc' ? comparison : -comparison
  })

  const totalCount = sorted.length
  const pageSize = Math.max(1, params.pageSize || 10)
  const totalPages = Math.max(1, Math.ceil(totalCount / pageSize))
  const page = Math.min(Math.max(1, params.page || 1), totalPages)
  const startIndex = (page - 1) * pageSize

  return {
    sessions: sorted.slice(startIndex, startIndex + pageSize),
    total_count: totalCount,
    page,
    page_size: pageSize,
    total_pages: totalPages,
  }
}

export function getDemoSessionDetail(sessionId: string): DemoSessionDetail | null {
  const sessionDetails = demoSessionDetails as Record<string, DemoSessionDetail>
  return sessionDetails[sessionId] || null
}

function getDemoFeedbackRecords(detail: DemoSessionDetail) {
  const feedbackItems = 'feedback' in detail && Array.isArray(detail.feedback) ? detail.feedback : []

  return feedbackItems.map((item) => ({
    ...item,
    app_session_id:
      'app_session_id' in item && typeof item.app_session_id === 'string'
        ? item.app_session_id
        : detail.session.app_session_id,
    inference_id:
      'inference_id' in item && (typeof item.inference_id === 'string' || item.inference_id === null)
        ? item.inference_id
        : detail.latest_inference?.inference_id || null,
  }))
}

function getDemoAnnotationRecords(detail: DemoSessionDetail) {
  return 'annotations' in detail && Array.isArray(detail.annotations) ? detail.annotations : []
}

function buildDemoSessionDetailSummary(detail: DemoSessionDetail) {
  const latestInference = detail.latest_inference
  const feedback = getDemoFeedbackRecords(detail)
  const annotations = getDemoAnnotationRecords(detail)
  const groupedFallEvents = latestInference?.response?.grouped_fall_events || []
  const timelineEvents = latestInference?.response?.timeline_events || []
  const transitionEvents = latestInference?.response?.transition_events || []

  return {
    session: detail.session,
    latest_inference: latestInference
      ? {
          ...latestInference,
          response: (() => {
            const {
              grouped_fall_events: _groupedFallEvents,
              timeline_events: _timelineEvents,
              transition_events: _transitionEvents,
              ...summaryResponse
            } = latestInference.response
            return summaryResponse
          })(),
        }
      : null,
    latest_feedback: feedback[0] || null,
    latest_annotation: annotations[0] || null,
    evidence_counts: {
      grouped_fall_events: groupedFallEvents.length,
      timeline_events: timelineEvents.length,
      transition_events: transitionEvents.length,
      feedback: feedback.length,
      annotations: annotations.length,
    },
  }
}

export function getDemoSessionDetailSummary(sessionId: string) {
  const detail = getDemoSessionDetail(sessionId)
  return detail ? buildDemoSessionDetailSummary(detail) : null
}

export function getDemoSessionEvidence(
  sessionId: string,
  sections: DemoEvidenceSection[] = ALL_DEMO_EVIDENCE_SECTIONS,
) {
  const detail = getDemoSessionDetail(sessionId)
  if (!detail) {
    return null
  }

  const resolvedSections = Array.from(new Set(sections.length ? sections : ALL_DEMO_EVIDENCE_SECTIONS))
  const feedback = getDemoFeedbackRecords(detail)
  const annotations = getDemoAnnotationRecords(detail)
  const response = detail.latest_inference?.response

  return {
    loaded_sections: resolvedSections,
    grouped_fall_events: resolvedSections.includes('grouped_fall_events') ? response?.grouped_fall_events || [] : [],
    timeline_events: resolvedSections.includes('timeline_events') ? response?.timeline_events || [] : [],
    transition_events: resolvedSections.includes('transition_events') ? response?.transition_events || [] : [],
    feedback: resolvedSections.includes('feedback') ? feedback : [],
    annotations: resolvedSections.includes('annotations') ? annotations : [],
  }
}
