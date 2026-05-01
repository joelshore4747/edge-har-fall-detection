const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')
export const DEMO_MODE_ENABLED = ['1', 'true', 'yes', 'on'].includes(
  (import.meta.env.VITE_DEMO_MODE || '').toLowerCase(),
)
const DEMO_SESSION_STORAGE_KEY = 'unifallmonitor.admin.demo-session'
// Artificial delay used to make demo API calls feel like network calls.
// Kept low because real demo flows chain several calls (login → me →
// overview → dashboard data) and stacked latency was making the login
// page feel sluggish on perception checks.
const DEMO_NETWORK_DELAY_MS = 20
type DemoApiModule = typeof import('./demo')

let demoApiModulePromise: Promise<DemoApiModule> | null = null

export class ApiError extends Error {
  status: number
  details: unknown

  constructor(status: number, message: string, details?: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.details = details
  }
}

export type AdminAuthSession = {
  status: string
  username: string | null
  subject_id: string | null
  role: string
}

export type PersistedSessionRecord = {
  app_session_id: string
  user_id: string
  device_id?: string | null
  subject_id: string
  client_session_id: string
  dataset_name: string
  source_type: string
  task_type: string
  placement_declared: string
  device_platform: string
  device_model: string | null
  runtime_mode: string
  recording_mode: string
  uploaded_at: string
  sample_count: number
  duration_seconds: number | null
  session_name: string | null
  activity_label: string | null
  notes: string | null
  raw_storage_uri?: string | null
  raw_storage_format?: string | null
  raw_payload_sha256?: string | null
  raw_payload_bytes?: number | null
  created_at: string
  updated_at?: string
}

export type AdminOverviewSessionRecord = {
  app_session_id: string
  subject_id: string
  client_session_id: string
  device_platform: string
  device_model: string | null
  uploaded_at: string
  duration_seconds: number | null
  session_name: string | null
  activity_label: string | null
}

export type AdminOverviewSessionItem = {
  session: AdminOverviewSessionRecord
  latest_status: string | null
  latest_warning_level: string | null
  latest_likely_fall_detected: boolean | null
  latest_top_har_label: string | null
  latest_top_fall_probability: number | null
  latest_grouped_fall_event_count: number | null
}

export type AdminSessionListRecord = {
  app_session_id: string
  subject_id: string
  client_session_id: string
  device_platform: string
  device_model: string | null
  uploaded_at: string
  duration_seconds: number | null
  session_name: string | null
  activity_label: string | null
  notes: string | null
}

export type AdminSessionListItem = {
  session: AdminSessionListRecord
  latest_status: string | null
  latest_warning_level: string | null
  latest_likely_fall_detected: boolean | null
  latest_top_har_label: string | null
  latest_top_fall_probability: number | null
  latest_grouped_fall_event_count: number | null
  latest_annotation_label?: string | null
}

export type SessionAnnotationRecord = {
  annotation_id: string
  app_session_id: string
  label: string
  source: string
  reviewer_identifier: string | null
  auth_account_id: string | null
  created_by_username: string | null
  request_id: string | null
  notes: string | null
  created_at: string
  updated_at: string
}

export type PersistedFeedbackRecord = {
  feedback_id: string
  app_session_id: string
  inference_id: string | null
  target_type: string
  target_event_key: string | null
  window_id: string | null
  feedback_type: string
  corrected_label: string | null
  reviewer_identifier: string | null
  subject_key: string | null
  notes: string | null
  request_id: string | null
  recorded_at: string
}

export type AdminOverviewResponse = {
  totals: {
    users: number
    sessions: number
    inferences: number
    grouped_fall_events: number
  }
  recent_activity: {
    sessions_last_7_days: number
    sessions_with_likely_fall: number
  }
  charts: {
    sessions_by_day: Array<{ label: string; value: number }>
    fall_events_by_day: Array<{ label: string; value: number }>
    warning_level_distribution: Array<{ label: string; value: number }>
    top_har_labels: Array<{ label: string; value: number }>
  }
  recent_sessions: AdminOverviewSessionItem[]
}

export type AdminSessionListResponse = {
  sessions: AdminSessionListItem[]
  total_count: number
  page: number
  page_size: number
  total_pages: number
}

export type GroupedFallEvent = {
  event_id: string
  event_start_ts: number
  event_end_ts: number
  event_duration_seconds: number
  n_positive_windows: number
  peak_probability: number | null
  mean_probability: number | null
  median_probability: number | null
}

export type TimelineEvent = {
  event_id: string
  start_ts: number
  end_ts: number
  duration_seconds: number
  midpoint_ts: number | null
  point_count: number
  activity_label: string
  placement_label: string
  activity_confidence_mean: number | null
  placement_confidence_mean: number | null
  fall_probability_peak: number | null
  fall_probability_mean: number | null
  likely_fall: boolean
  event_kind: string
  related_grouped_fall_event_ids: string[]
  description: string
}

export type TransitionEvent = {
  transition_id: string
  transition_ts: number
  from_event_id: string
  to_event_id: string
  transition_kind: string
  from_activity_label: string | null
  to_activity_label: string | null
  from_placement_label: string | null
  to_placement_label: string | null
  description: string
}

export type SessionNarrativeSummary = {
  session_id: string
  dataset_name: string
  subject_id: string
  total_duration_seconds: number
  event_count: number
  transition_count: number
  fall_event_count: number
  dominant_activity_label: string
  dominant_placement_label: string
  highest_fall_probability: number | null
  summary_text: string
  // Vulnerability + HAR-attenuation rollup. Prefer peak_vulnerability_score
  // as the headline number; highest_fall_probability is the raw model
  // output and is intentionally NOT attenuated by the walking/stairs gate.
  peak_vulnerability_score?: number | null
  mean_vulnerability_score?: number | null
  dominant_vulnerability_level?: string | null
  har_attenuation_applied?: boolean
  har_attenuation_window_count?: number
  har_attenuation_label?: string | null
  har_attenuation_confidence_mean?: number | null
}

export type RuntimeSessionResponse = {
  request_id: string | null
  session_id: string
  persisted_user_id: string | null
  persisted_session_id: string | null
  persisted_inference_id: string | null
  source_summary: {
    input_sample_count: number
    session_duration_seconds: number | null
    estimated_sampling_rate_hz: number | null
  }
  placement_summary: {
    placement_state: string
    placement_confidence: number | null
  }
  har_summary: {
    top_label: string | null
    top_label_fraction: number | null
    total_windows: number
  }
  fall_summary: {
    likely_fall_detected: boolean
    positive_window_count: number
    grouped_event_count: number
    top_fall_probability: number | null
    mean_fall_probability: number | null
  }
  alert_summary: {
    warning_level: string
    likely_fall_detected: boolean
    top_har_label: string | null
    top_har_fraction: number | null
    grouped_fall_event_count: number
    top_fall_probability: number | null
    top_vulnerability_score: number | null
    latest_vulnerability_level: string | null
    latest_monitoring_state: string | null
    latest_fall_event_state: string | null
    recommended_message: string
  }
  model_info: {
    har_model_name: string | null
    har_model_version: string | null
    fall_model_name: string | null
    fall_model_version: string | null
    api_version: string
  }
  grouped_fall_events: GroupedFallEvent[]
  timeline_events: TimelineEvent[]
  transition_events: TransitionEvent[]
  session_narrative_summary: SessionNarrativeSummary | null
  narrative_summary: Record<string, unknown>
}

export type PersistedSessionDetailResponse = {
  session: PersistedSessionRecord
  latest_inference: {
    inference_id: string
    request_id: string | null
    status: string
    error_message: string | null
    started_at: string
    completed_at: string | null
    created_at: string
    response: RuntimeSessionResponse
  } | null
  feedback: PersistedFeedbackRecord[]
  annotations?: SessionAnnotationRecord[]
}

export type AdminEvidenceSection =
  | 'grouped_fall_events'
  | 'timeline_events'
  | 'transition_events'
  | 'feedback'
  | 'annotations'

export type AdminSessionEvidenceCounts = {
  grouped_fall_events: number
  timeline_events: number
  transition_events: number
  feedback: number
  annotations: number
}

export type AdminRuntimeSessionSummary = Omit<
  RuntimeSessionResponse,
  'grouped_fall_events' | 'timeline_events' | 'transition_events'
>

export type AdminPersistedInferenceSummary = {
  inference_id: string
  request_id: string | null
  status: string
  error_message: string | null
  started_at: string
  completed_at: string | null
  created_at: string
  response: AdminRuntimeSessionSummary
}

export type AdminSessionDetailSummaryResponse = {
  session: PersistedSessionRecord
  latest_inference: AdminPersistedInferenceSummary | null
  latest_feedback: PersistedFeedbackRecord | null
  latest_annotation: SessionAnnotationRecord | null
  evidence_counts: AdminSessionEvidenceCounts
}

export type AdminSessionEvidenceResponse = {
  loaded_sections: AdminEvidenceSection[]
  grouped_fall_events: GroupedFallEvent[]
  timeline_events: TimelineEvent[]
  transition_events: TransitionEvent[]
  feedback: PersistedFeedbackRecord[]
  annotations: SessionAnnotationRecord[]
}

export type AdminSessionsParams = {
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

function buildQueryString(params: Record<string, string | number | boolean | undefined>) {
  const searchParams = new URLSearchParams()
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === '') {
      return
    }
    searchParams.set(key, String(value))
  })
  const rendered = searchParams.toString()
  return rendered ? `?${rendered}` : ''
}

function createAbortError() {
  return new DOMException('The operation was aborted.', 'AbortError')
}

function withDemoDelay<T>(resolver: () => T, signal?: AbortSignal) {
  return new Promise<T>((resolve, reject) => {
    if (signal?.aborted) {
      reject(createAbortError())
      return
    }

    const timeoutId = globalThis.setTimeout(() => {
      cleanupAbortListener()
      try {
        resolve(resolver())
      } catch (error) {
        reject(error)
      }
    }, DEMO_NETWORK_DELAY_MS)

    const handleAbort = () => {
      globalThis.clearTimeout(timeoutId)
      cleanupAbortListener()
      reject(createAbortError())
    }

    const cleanupAbortListener = () => {
      signal?.removeEventListener('abort', handleAbort)
    }

    signal?.addEventListener('abort', handleAbort, { once: true })
  })
}

function loadDemoApiModule() {
  if (!demoApiModulePromise) {
    // Keep demo-only seed data out of the normal production entry chunk.
    demoApiModulePromise = import('./demo')
  }

  return demoApiModulePromise
}

function writeDemoSessionState(value: 'active' | 'signed_out') {
  if (typeof window === 'undefined') {
    return
  }
  window.localStorage.setItem(DEMO_SESSION_STORAGE_KEY, value)
}

function readDemoSessionState() {
  if (typeof window === 'undefined') {
    return 'active' as const
  }

  const stored = window.localStorage.getItem(DEMO_SESSION_STORAGE_KEY)
  if (!stored) {
    writeDemoSessionState('active')
    return 'active' as const
  }

  return stored === 'signed_out' ? ('signed_out' as const) : ('active' as const)
}

function getDemoAdminSession(): AdminAuthSession {
  if (readDemoSessionState() === 'signed_out') {
    throw new ApiError(
      401,
      'Demo mode is enabled, but the local demo session is signed out. Use demo access to reopen the dashboard.',
    )
  }

  return {
    status: 'authenticated',
    username: 'demo-admin',
    subject_id: 'UFM-DEMO-ADMIN',
    role: 'admin',
  }
}

export function getDemoAdminSessionSnapshot(): AdminAuthSession | null {
  if (!DEMO_MODE_ENABLED) {
    return null
  }

  try {
    return getDemoAdminSession()
  } catch {
    return null
  }
}

// Public editorial pages (Results, Briefing, Architecture, Docs) must be
// readable without an explicit demo "sign-in" click. Calling this restores
// the demo session to 'active' even if the visitor previously signed out,
// so the editorial gate never bounces a marker to /login in demo deploys.
export function ensureDemoAdminSessionActive(): AdminAuthSession | null {
  if (!DEMO_MODE_ENABLED) {
    return null
  }
  writeDemoSessionState('active')
  return getDemoAdminSessionSnapshot()
}

const CSRF_COOKIE_NAME = 'unifallmonitor_csrf'
const CSRF_HEADER_NAME = 'X-CSRF-Token'
const CSRF_SAFE_METHODS = new Set(['GET', 'HEAD', 'OPTIONS'])

function readCsrfCookie(): string | null {
  if (typeof document === 'undefined') {
    return null
  }
  const prefix = `${CSRF_COOKIE_NAME}=`
  for (const part of document.cookie.split(';')) {
    const trimmed = part.trim()
    if (trimmed.startsWith(prefix)) {
      return decodeURIComponent(trimmed.slice(prefix.length))
    }
  }
  return null
}

async function apiFetch<T>(path: string, init: RequestInit = {}) {
  const headers = new Headers(init.headers)
  if (!headers.has('Accept')) {
    headers.set('Accept', 'application/json')
  }
  if (init.body && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }

  const method = (init.method || 'GET').toUpperCase()
  if (!CSRF_SAFE_METHODS.has(method) && !headers.has(CSRF_HEADER_NAME)) {
    const token = readCsrfCookie()
    if (token) {
      headers.set(CSRF_HEADER_NAME, token)
    }
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    credentials: 'include',
    headers,
  })

  const contentType = response.headers.get('content-type') || ''
  let payload: unknown = null

  if (contentType.includes('application/json')) {
    payload = await response.json()
  } else {
    const text = await response.text()
    payload = text ? { detail: text } : null
  }

  if (!response.ok) {
    const message =
      typeof payload === 'object' &&
      payload !== null &&
      'message' in payload &&
      (payload as { message: unknown }).message != null
        ? String((payload as { message: unknown }).message)
        : typeof payload === 'object' && payload !== null && 'detail' in payload
          ? String((payload as { detail: unknown }).detail)
        : `Request failed with status ${response.status}`
    throw new ApiError(response.status, message, payload)
  }

  return payload as T
}

export function getErrorMessage(error: unknown, fallback = 'Something went wrong.') {
  if (error instanceof ApiError) {
    return error.message
  }
  if (error instanceof Error && error.message) {
    return error.message
  }
  return fallback
}

export function loginAdmin(payload: { username: string; password: string }) {
  if (DEMO_MODE_ENABLED) {
    void payload
    return withDemoDelay<AdminAuthSession>(() => {
      writeDemoSessionState('active')
      return getDemoAdminSession()
    })
  }

  return apiFetch<AdminAuthSession>('/v1/admin/auth/login', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function logoutAdmin() {
  if (DEMO_MODE_ENABLED) {
    return withDemoDelay<AdminAuthSession>(() => {
      writeDemoSessionState('signed_out')
      return {
        status: 'signed_out',
        username: null,
        subject_id: null,
        role: 'admin',
      }
    })
  }

  return apiFetch<AdminAuthSession>('/v1/admin/auth/logout', {
    method: 'POST',
  })
}

export function getAdminMe() {
  if (DEMO_MODE_ENABLED) {
    return withDemoDelay<AdminAuthSession>(() => getDemoAdminSession())
  }

  return apiFetch<AdminAuthSession>('/v1/admin/auth/me')
}

export async function getAdminOverview() {
  if (DEMO_MODE_ENABLED) {
    const demoApiModule = await loadDemoApiModule()

    return withDemoDelay<AdminOverviewResponse>(() => {
      getDemoAdminSession()
      return demoApiModule.buildDemoAdminOverview()
    })
  }

  return apiFetch<AdminOverviewResponse>('/v1/admin/overview')
}

export async function listAdminSessions(
  params: AdminSessionsParams,
  options: { signal?: AbortSignal } = {},
) {
  if (DEMO_MODE_ENABLED) {
    const demoApiModule = await loadDemoApiModule()

    return withDemoDelay<AdminSessionListResponse>(() => {
      getDemoAdminSession()
      return demoApiModule.buildDemoSessionListResponse(params)
    }, options.signal)
  }

  return apiFetch<AdminSessionListResponse>(
    `/v1/admin/sessions${buildQueryString({
      page: params.page,
      page_size: params.pageSize,
      search: params.search,
      warning_level: params.warningLevel,
      device_platform: params.devicePlatform,
      date_from: params.dateFrom,
      date_to: params.dateTo,
      likely_fall: params.likelyFall,
      status: params.status,
      sort_by: params.sortBy,
      sort_dir: params.sortDir,
    })}`,
    { signal: options.signal },
  )
}

export async function getAdminSessionDetail(
  sessionId: string,
  options: { signal?: AbortSignal } = {},
) {
  if (DEMO_MODE_ENABLED) {
    const demoApiModule = await loadDemoApiModule()

    return withDemoDelay<AdminSessionDetailSummaryResponse>(() => {
      getDemoAdminSession()
      const detail = demoApiModule.getDemoSessionDetailSummary(sessionId)
      if (!detail) {
        throw new ApiError(404, `Demo session ${sessionId} was not found.`)
      }
      return detail
    }, options.signal)
  }

  return apiFetch<AdminSessionDetailSummaryResponse>(`/v1/admin/sessions/${sessionId}`, {
    signal: options.signal,
  })
}

export async function getAdminSessionEvidence(
  sessionId: string,
  sections: AdminEvidenceSection[],
  options: { signal?: AbortSignal } = {},
) {
  if (DEMO_MODE_ENABLED) {
    const demoApiModule = await loadDemoApiModule()

    return withDemoDelay<AdminSessionEvidenceResponse>(() => {
      getDemoAdminSession()
      const evidence = demoApiModule.getDemoSessionEvidence(sessionId, sections)
      if (!evidence) {
        throw new ApiError(404, `Demo session ${sessionId} was not found.`)
      }
      return evidence
    }, options.signal)
  }

  return apiFetch<AdminSessionEvidenceResponse>(
    `/v1/admin/sessions/${sessionId}/evidence${buildQueryString({
      sections: sections.join(','),
    })}`,
    { signal: options.signal },
  )
}
