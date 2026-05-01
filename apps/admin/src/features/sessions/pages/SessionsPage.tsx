import { createQuery, keepPreviousData } from '@tanstack/solid-query'
import { Link } from '@tanstack/solid-router'
import { For, Show, createMemo, createSignal } from 'solid-js'
import { DataState, Gcard, Metric, ProbBar, WarningTag } from '../../../components/v6'
import { listAdminSessions } from '../../../lib/api'

const PAGE_SIZE = 12

function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null) return '—'
  return `${seconds.toFixed(1)} s`
}

export function SessionsPage() {
  const [page, setPage] = createSignal(1)
  const [search, setSearch] = createSignal('')
  const [warningLevel, setWarningLevel] = createSignal<string>('')
  const [appliedSearch, setAppliedSearch] = createSignal('')

  const queryParams = createMemo(() => ({
    page: page(),
    pageSize: PAGE_SIZE,
    search: appliedSearch().trim() || undefined,
    warningLevel: warningLevel() || undefined,
    sortBy: 'created_at' as const,
    sortDir: 'desc' as const,
  }))

  const sessionsQuery = createQuery(() => ({
    queryKey: ['admin', 'sessions', queryParams()],
    queryFn: ({ signal }) => listAdminSessions(queryParams(), { signal }),
    placeholderData: keepPreviousData,
  }))

  const sessions = createMemo(() => sessionsQuery.data?.sessions ?? [])
  const total = createMemo(() => sessionsQuery.data?.total_count ?? 0)
  const totalPages = createMemo(() => Math.max(1, Math.ceil(total() / PAGE_SIZE)))
  const flaggedCount = createMemo(
    () => sessions().filter((s) => s.latest_likely_fall_detected).length,
  )
  const highCount = createMemo(
    () => sessions().filter((s) => s.latest_warning_level === 'high').length,
  )

  const handleSearchSubmit = (event: SubmitEvent) => {
    event.preventDefault()
    setPage(1)
    setAppliedSearch(search())
  }

  return (
    <>
      <div class="metrics" style={{ 'grid-template-columns': 'repeat(4, 1fr)' }}>
        <Metric label="Total sessions" value={total() || '—'} sub="matching active filters" />
        <Metric label="On this page" value={sessions().length} sub={`page ${page()} of ${totalPages()}`} />
        <Metric
          label="High warnings"
          alert={highCount() > 0}
          value={highCount()}
          sub="across visible page"
        />
        <Metric label="Likely-fall flagged" value={flaggedCount()} sub="latest inference" />
      </div>

      <Gcard
        title="Filter sessions"
        sub="search by id / subject / device · narrow by warning"
      >
        <form
          onSubmit={handleSearchSubmit}
          style={{ display: 'flex', 'flex-wrap': 'wrap', gap: '12px', 'align-items': 'flex-end' }}
        >
          <div class="form-field" style={{ flex: 1, 'min-width': '220px', margin: 0 }}>
            <label>Search</label>
            <input
              type="search"
              placeholder="se_a3f0… · subj_07 · iPhone"
              value={search()}
              onInput={(e) => setSearch(e.currentTarget.value)}
            />
          </div>
          <div class="form-field" style={{ width: '180px', margin: 0 }}>
            <label>Warning level</label>
            <select value={warningLevel()} onChange={(e) => setWarningLevel(e.currentTarget.value)}>
              <option value="">All levels</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
              <option value="none">None</option>
            </select>
          </div>
          <button class="btn-ghost" type="submit">
            Apply
          </button>
        </form>
      </Gcard>

      <div class="an-row" style={{ 'grid-template-columns': '1fr' }}>
        <Gcard
          bare
          title="Sessions"
          sub={total() ? `${total()} total · sorted by upload time` : 'awaiting first ingest'}
        >
          <Show
            when={sessions().length > 0}
            fallback={
              <div style={{ padding: '32px' }}>
                <DataState
                  title={sessionsQuery.isLoading ? 'Loading sessions' : 'No sessions yet'}
                  description={
                    sessionsQuery.isLoading
                      ? 'Querying the persisted session table for your filters.'
                      : 'Replay a runtime session or POST to /v1/infer/session to populate this list.'
                  }
                />
              </div>
            }
          >
            <div style={{ 'overflow-x': 'auto' }}>
              <table class="tbl">
                <thead>
                  <tr>
                    <th>Session</th>
                    <th>Subject</th>
                    <th>Activity</th>
                    <th>Duration</th>
                    <th>Platform</th>
                    <th>Warning</th>
                    <th>Top fall p</th>
                    <th>Events</th>
                    <th>Uploaded</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  <For each={sessions()}>
                    {(s) => {
                      const level = (s.latest_warning_level ?? 'none') as string
                      const tag =
                        level === 'medium' ? 'med' : (level as 'high' | 'low' | 'none')
                      const fallP = s.latest_top_fall_probability ?? 0
                      // Alert styling tracks the post-gate warning level, not
                      // the raw fall probability. Confident-walking/stairs
                      // sessions can produce high raw P(fall) (the meta
                      // classifier emits these on locomotion windows) but
                      // are downgraded by the HAR locomotion gate in
                      // _derive_alert_summary; the bar should follow that.
                      return (
                        <tr>
                          <td class="id">{s.session.app_session_id.slice(0, 8)}</td>
                          <td class="mono">subj_{s.session.subject_id}</td>
                          <td>{s.session.activity_label ?? '—'}</td>
                          <td class="mono">{formatDuration(s.session.duration_seconds)}</td>
                          <td class="mono">{s.session.device_platform}</td>
                          <td>
                            <WarningTag level={tag}>{level}</WarningTag>
                          </td>
                          <td>
                            <ProbBar value={fallP} alert={level === 'high'} />
                          </td>
                          <td class="mono">{s.latest_grouped_fall_event_count ?? 0}</td>
                          <td class="mono">
                            {s.session.uploaded_at
                              ? new Date(s.session.uploaded_at).toLocaleString('en-GB', {
                                  day: '2-digit',
                                  month: 'short',
                                  hour: '2-digit',
                                  minute: '2-digit',
                                })
                              : '—'}
                          </td>
                          <td>
                            <Link
                              to="/sessions/$sessionId"
                              params={{ sessionId: s.session.app_session_id }}
                              class="btn-ghost"
                            >
                              Open
                            </Link>
                          </td>
                        </tr>
                      )
                    }}
                  </For>
                </tbody>
              </table>
            </div>
          </Show>
        </Gcard>
      </div>

      <Show when={totalPages() > 1}>
        <div
          style={{
            display: 'flex',
            'align-items': 'center',
            'justify-content': 'space-between',
            padding: '18px 0',
            'font-family': 'var(--mono)',
            'font-size': '12px',
            color: 'var(--text-3)',
          }}
        >
          <span>
            Page {page()} of {totalPages()}
          </span>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              class="btn-ghost"
              disabled={page() <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
            >
              ← Previous
            </button>
            <button
              class="btn-ghost"
              disabled={page() >= totalPages()}
              onClick={() => setPage((p) => Math.min(totalPages(), p + 1))}
            >
              Next →
            </button>
          </div>
        </div>
      </Show>
    </>
  )
}
