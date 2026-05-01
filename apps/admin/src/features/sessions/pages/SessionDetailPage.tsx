import { createQuery } from '@tanstack/solid-query'
import { Show } from 'solid-js'
import { DataState, Gcard, Metric, WarningTag } from '../../../components/v6'
import { getAdminSessionDetail, getAdminSessionEvidence } from '../../../lib/api'
import { SessionTimelineChart } from '../components/SessionTimelineChart'

export function SessionDetailPage(props: { sessionId: string }) {
  const detailQuery = createQuery(() => ({
    queryKey: ['admin', 'session', props.sessionId],
    queryFn: () => getAdminSessionDetail(props.sessionId),
  }))

  const evidenceQuery = createQuery(() => ({
    queryKey: [
      'admin',
      'session',
      props.sessionId,
      'evidence',
      ['grouped_fall_events', 'timeline_events'],
    ],
    queryFn: () =>
      getAdminSessionEvidence(props.sessionId, ['grouped_fall_events', 'timeline_events']),
  }))

  return (
    <Show
      when={detailQuery.data}
      fallback={
        <DataState
          title={detailQuery.isLoading ? 'Loading session' : 'Session not found'}
          description={
            detailQuery.isLoading
              ? 'Fetching the persisted session, inference rows, and grouped fall events.'
              : `No persisted record for ${props.sessionId}. Confirm the session id and try again.`
          }
        />
      }
    >
      {(data) => {
        const session = () => data().session
        const inference = () => data().latest_inference
        const alert = () => inference()?.response.alert_summary
        const sourceSummary = () => inference()?.response.source_summary
        const persistedDuration = () => session().duration_seconds
        const inferenceDuration = () => sourceSummary()?.session_duration_seconds ?? null
        const durationMismatch = () => {
          const stored = persistedDuration()
          const inferred = inferenceDuration()
          if (stored == null || inferred == null) return false
          return Math.abs(stored - inferred) > 1
        }
        const warning = () => alert()?.warning_level ?? 'none'
        const tag = () =>
          warning() === 'medium' ? 'med' : (warning() as 'high' | 'low' | 'none')

        return (
          <>
            <div class="metrics" style={{ 'grid-template-columns': 'repeat(4, 1fr)' }}>
              <Metric
                label="Session id"
                value={session().app_session_id.slice(0, 10)}
                sub={session().client_session_id ?? '—'}
              />
              <Metric
                label="Subject"
                value={`subj_${session().subject_id}`}
                sub={`${session().device_platform} · ${session().placement_declared}`}
              />
              <Metric
                label="Warning"
                alert={warning() === 'high'}
                value={warning()}
                sub={alert()?.likely_fall_detected ? 'likely fall detected' : 'no fall flagged'}
              />
              <Metric
                label="Top fall p"
                value={(alert()?.top_fall_probability ?? 0).toFixed(3)}
                sub={`${session().sample_count.toLocaleString()} samples`}
              />
            </div>

            <div class="an-row r2">
              <Gcard
                title="Session metadata"
                sub={`${session().task_type} · ${session().recording_mode}`}
              >
                <table class="tbl">
                  <tbody>
                    <tr>
                      <td class="mono">app_session_id</td>
                      <td class="id">{session().app_session_id}</td>
                    </tr>
                    <tr>
                      <td class="mono">subject_id</td>
                      <td>{session().subject_id}</td>
                    </tr>
                    <tr>
                      <td class="mono">dataset_name</td>
                      <td>{session().dataset_name}</td>
                    </tr>
                    <tr>
                      <td class="mono">placement_declared</td>
                      <td>{session().placement_declared}</td>
                    </tr>
                    <tr>
                      <td class="mono">device</td>
                      <td>
                        {session().device_platform} · {session().device_model ?? '—'}
                      </td>
                    </tr>
                    <tr>
                      <td class="mono">stored_duration</td>
                      <td>
                        {persistedDuration()
                          ? `${persistedDuration()!.toFixed(1)} s`
                          : '—'}
                      </td>
                    </tr>
                    <tr>
                      <td class="mono">inference_duration</td>
                      <td>
                        {inferenceDuration()
                          ? `${inferenceDuration()!.toFixed(1)} s`
                          : '—'}
                        <Show when={durationMismatch()}>
                          <span
                            class="mono"
                            style={{
                              color: 'var(--terracotta)',
                              'margin-left': '8px',
                              'font-size': '11px',
                            }}
                          >
                            differs from stored row
                          </span>
                        </Show>
                      </td>
                    </tr>
                    <tr>
                      <td class="mono">estimated_rate</td>
                      <td class="mono">
                        {sourceSummary()?.estimated_sampling_rate_hz
                          ? `${sourceSummary()!.estimated_sampling_rate_hz!.toFixed(2)} Hz`
                          : '—'}
                      </td>
                    </tr>
                    <tr>
                      <td class="mono">sample_count</td>
                      <td class="mono">{session().sample_count.toLocaleString()}</td>
                    </tr>
                    <tr>
                      <td class="mono">uploaded_at</td>
                      <td class="mono">
                        {new Date(session().uploaded_at).toLocaleString('en-GB')}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </Gcard>

              <Gcard
                title="Latest inference"
                sub={inference() ? 'persisted · most recent' : 'no inference recorded'}
                pill={
                  <Show when={inference()} fallback={<>—</>}>
                    <WarningTag level={tag()}>{warning()}</WarningTag>
                  </Show>
                }
              >
                <Show
                  when={inference()}
                  fallback={
                    <p style={{ color: 'var(--text-3)', 'font-size': '13px' }}>
                      Trigger an inference run to populate this card.
                    </p>
                  }
                >
                  {(inf) => {
                    const resp = () => inf().response
                    return (
                      <table class="tbl">
                        <tbody>
                          <tr>
                            <td class="mono">inference_id</td>
                            <td class="id">{inf().inference_id}</td>
                          </tr>
                          <tr>
                            <td class="mono">status</td>
                            <td>{inf().status}</td>
                          </tr>
                          <tr>
                            <td class="mono">har_model</td>
                            <td>
                              {resp().model_info.har_model_name ?? '—'} ·{' '}
                              {resp().model_info.har_model_version ?? ''}
                            </td>
                          </tr>
                          <tr>
                            <td class="mono">fall_model</td>
                            <td>
                              {resp().model_info.fall_model_name ?? '—'} ·{' '}
                              {resp().model_info.fall_model_version ?? ''}
                            </td>
                          </tr>
                          <tr>
                            <td class="mono">top_har_label</td>
                            <td>
                              {resp().har_summary.top_label ?? '—'}{' '}
                              <span class="mono">
                                ({(resp().har_summary.top_label_fraction ?? 0).toFixed(2)})
                              </span>
                            </td>
                          </tr>
                          <tr>
                            <td class="mono">top_fall_probability</td>
                            <td class="mono">
                              {(resp().fall_summary.top_fall_probability ?? 0).toFixed(3)}
                            </td>
                          </tr>
                          <tr>
                            <td class="mono">positive_windows</td>
                            <td class="mono">{resp().fall_summary.positive_window_count}</td>
                          </tr>
                          <tr>
                            <td class="mono">grouped_events</td>
                            <td class="mono">{resp().fall_summary.grouped_event_count}</td>
                          </tr>
                        </tbody>
                      </table>
                    )
                  }}
                </Show>
              </Gcard>
            </div>

            <div class="an-row" style={{ 'grid-template-columns': '1fr' }}>
              <Gcard
                bare
                title="Per-segment fall probability"
                sub={
                  (evidenceQuery.data?.timeline_events ?? []).length > 0
                    ? `${evidenceQuery.data!.timeline_events.length} segments · activity colour bands · grouped fall events overlaid`
                    : 'timeline evidence required for graph'
                }
              >
                <Show
                  when={(evidenceQuery.data?.timeline_events ?? []).length > 0}
                  fallback={
                    <div style={{ padding: '32px' }}>
                      <DataState
                        title={evidenceQuery.isLoading ? 'Loading graph data' : 'No graph data yet'}
                        description={
                          evidenceQuery.isLoading
                            ? 'Fetching persisted timeline events for this session.'
                            : 'Re-upload or rescore this session to generate per-window timeline events for the graph.'
                        }
                      />
                    </div>
                  }
                >
                  <SessionTimelineChart
                    timelineEvents={evidenceQuery.data!.timeline_events}
                    groupedFallEvents={evidenceQuery.data?.grouped_fall_events ?? []}
                    harAttenuationApplied={
                      // session_narrative_summary is on the inference response payload, not the
                      // admin summary endpoint; surface the warning-level signal instead, which
                      // is what the gate downgrades when it fires.
                      (alert()?.warning_level ?? 'none') !== 'high' &&
                      (alert()?.likely_fall_detected ?? false)
                    }
                    harAttenuationLabel={alert()?.top_har_label ?? null}
                  />
                </Show>
              </Gcard>
            </div>

            <Show when={(evidenceQuery.data?.grouped_fall_events ?? []).length > 0}>
              <div class="an-row" style={{ 'grid-template-columns': '1fr' }}>
                <Gcard
                  bare
                  title="Grouped fall events"
                  sub={`${evidenceQuery.data!.grouped_fall_events.length} events`}
                >
                  <table class="tbl">
                    <thead>
                      <tr>
                        <th>Event id</th>
                        <th>Start (s)</th>
                        <th>Peak p</th>
                        <th>Positive windows</th>
                      </tr>
                    </thead>
                    <tbody>
                      {evidenceQuery.data!.grouped_fall_events.map((e) => (
                        <tr>
                          <td class="id">{e.event_id.slice(0, 10)}</td>
                          <td class="mono">{e.event_start_ts.toFixed(2)}</td>
                          <td class="mono">{(e.peak_probability ?? 0).toFixed(3)}</td>
                          <td class="mono">{e.n_positive_windows}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Gcard>
              </div>
            </Show>
          </>
        )
      }}
    </Show>
  )
}
