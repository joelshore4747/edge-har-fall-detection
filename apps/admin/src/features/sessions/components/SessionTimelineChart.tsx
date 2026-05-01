import { For, Show, createMemo } from 'solid-js'
import type { GroupedFallEvent, TimelineEvent } from '../../../lib/api'

// Per-session diagnostic chart: HAR activity colour bands as the background,
// per-segment peak fall probability as a step plot, and grouped fall events
// highlighted as vertical bands. Sourced entirely from the persisted
// timeline_events + grouped_fall_events evidence sections — no live
// re-inference required, no DB schema changes.
//
// The chart is the visual companion to the walking/stairs story documented in
// fusion/vulnerability_score.py:161–186 and surfaced in the session warning
// gate at apps/api/main.py:_derive_alert_summary: a confident-walking session
// can produce high per-segment fall probability spikes that the session-level
// gate has already attenuated. The chart shows both layers at once so a
// reviewer (or viva examiner) can read the gate's behaviour directly.

const ACTIVITY_COLOURS: Record<string, { fill: string; stroke: string }> = {
  walking: { fill: 'rgba(63, 122, 131, 0.18)', stroke: 'var(--teal)' },
  running: { fill: 'rgba(63, 122, 131, 0.22)', stroke: 'var(--teal)' },
  stairs: { fill: 'rgba(184, 137, 58, 0.18)', stroke: 'var(--ochre)' },
  sitting: { fill: 'rgba(94, 122, 74, 0.18)', stroke: 'var(--moss)' },
  standing: { fill: 'rgba(94, 122, 74, 0.10)', stroke: 'var(--moss)' },
  lying: { fill: 'rgba(122, 74, 94, 0.18)', stroke: 'var(--plum)' },
  laying: { fill: 'rgba(122, 74, 94, 0.18)', stroke: 'var(--plum)' },
  static: { fill: 'rgba(120, 110, 100, 0.10)', stroke: 'var(--text-3)' },
  fall: { fill: 'rgba(192, 86, 58, 0.20)', stroke: 'var(--terracotta)' },
  unknown: { fill: 'rgba(120, 110, 100, 0.06)', stroke: 'var(--text-3)' },
}

function colourFor(label: string): { fill: string; stroke: string } {
  const key = (label ?? 'unknown').trim().toLowerCase()
  return ACTIVITY_COLOURS[key] ?? ACTIVITY_COLOURS.unknown
}

interface Props {
  timelineEvents: TimelineEvent[]
  groupedFallEvents: GroupedFallEvent[]
  // The probability above which a window is grouped into a fall event; comes
  // from RuntimeInferenceConfig.event_probability_threshold (default 0.5).
  // Drawn as a horizontal reference line.
  eventThreshold?: number
  // Optional session-level attenuation context for the legend caption.
  harAttenuationApplied?: boolean
  harAttenuationLabel?: string | null
}

export function SessionTimelineChart(props: Props) {
  // Layout — fixed pixel viewBox; SVG scales responsively via preserveAspectRatio.
  const W = 880
  const H = 280
  const PAD_L = 56
  const PAD_R = 24
  const PAD_T = 24
  const PAD_B = 48
  const plotW = W - PAD_L - PAD_R
  const plotH = H - PAD_T - PAD_B

  const sortedEvents = createMemo<TimelineEvent[]>(() => {
    const events = [...(props.timelineEvents ?? [])]
    events.sort((a, b) => a.start_ts - b.start_ts)
    return events
  })

  const timeBounds = createMemo<{ start: number; end: number }>(() => {
    const events = sortedEvents()
    if (events.length === 0) return { start: 0, end: 1 }
    const start = Math.min(events[0].start_ts, ...(props.groupedFallEvents ?? []).map((e) => e.event_start_ts))
    const end = Math.max(
      events[events.length - 1].end_ts,
      ...(props.groupedFallEvents ?? []).map((e) => e.event_end_ts),
    )
    return { start, end: end > start ? end : start + 1 }
  })

  const xOf = (ts: number) => {
    const { start, end } = timeBounds()
    return PAD_L + ((ts - start) / (end - start)) * plotW
  }
  const yOf = (p: number) => PAD_T + (1 - Math.max(0, Math.min(1, p))) * plotH

  const presentLabels = createMemo<string[]>(() => {
    const seen = new Set<string>()
    sortedEvents().forEach((e) => seen.add((e.activity_label ?? 'unknown').toLowerCase()))
    return Array.from(seen)
  })

  const threshold = () => props.eventThreshold ?? 0.5

  return (
    <Show
      when={sortedEvents().length > 0}
      fallback={
        <div
          style={{
            padding: '32px',
            'text-align': 'center',
            color: 'var(--text-3)',
            'font-size': '13px',
            'font-family': 'var(--mono)',
          }}
        >
          No timeline events for this session — nothing to chart.
        </div>
      }
    >
      <div>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="xMidYMid meet"
          style={{ width: '100%', height: 'auto', display: 'block' }}
          role="img"
          aria-label="Per-segment fall probability over the session, with HAR activity colour bands and grouped fall events"
        >
          {/* Y-axis gridlines */}
          <g stroke="rgba(40,28,20,.10)" stroke-dasharray="2 4">
            <For each={[0.25, 0.5, 0.75]}>
              {(p) => <line x1={PAD_L} x2={W - PAD_R} y1={yOf(p)} y2={yOf(p)} />}
            </For>
          </g>

          {/* HAR activity colour bands */}
          <g>
            <For each={sortedEvents()}>
              {(e) => {
                const c = colourFor(e.activity_label)
                const x = xOf(e.start_ts)
                const w = Math.max(1, xOf(e.end_ts) - x)
                return (
                  <rect
                    x={x}
                    y={PAD_T}
                    width={w}
                    height={plotH}
                    fill={c.fill}
                    stroke="none"
                  />
                )
              }}
            </For>
          </g>

          {/* Grouped fall events — vertical highlight bands */}
          <g>
            <For each={props.groupedFallEvents ?? []}>
              {(g) => {
                const x = xOf(g.event_start_ts)
                const w = Math.max(2, xOf(g.event_end_ts) - x)
                return (
                  <>
                    <rect
                      x={x}
                      y={PAD_T}
                      width={w}
                      height={plotH}
                      fill="rgba(192, 86, 58, 0.16)"
                      stroke="var(--terracotta)"
                      stroke-width="1"
                      stroke-dasharray="3 3"
                    />
                    <text
                      x={x + 4}
                      y={PAD_T + 12}
                      font-size="9"
                      font-family="var(--mono)"
                      fill="var(--terracotta)"
                    >
                      {`event · peak ${(g.peak_probability ?? 0).toFixed(2)}`}
                    </text>
                  </>
                )
              }}
            </For>
          </g>

          {/* Event grouping threshold reference line */}
          <line
            x1={PAD_L}
            x2={W - PAD_R}
            y1={yOf(threshold())}
            y2={yOf(threshold())}
            stroke="var(--terracotta)"
            stroke-width="1"
            stroke-dasharray="4 4"
          />
          <text
            x={W - PAD_R - 4}
            y={yOf(threshold()) - 4}
            text-anchor="end"
            font-size="9"
            font-family="var(--mono)"
            fill="var(--terracotta)"
          >
            {`τ = ${threshold().toFixed(2)} · grouping threshold`}
          </text>

          {/* Per-segment peak-probability step plot */}
          <g>
            <For each={sortedEvents()}>
              {(e) => {
                const p = e.fall_probability_peak ?? 0
                const x1 = xOf(e.start_ts)
                const x2 = xOf(e.end_ts)
                return (
                  <line
                    x1={x1}
                    x2={x2}
                    y1={yOf(p)}
                    y2={yOf(p)}
                    stroke={colourFor(e.activity_label).stroke}
                    stroke-width="2"
                  />
                )
              }}
            </For>
          </g>

          {/* Axes */}
          <line x1={PAD_L} y1={H - PAD_B} x2={W - PAD_R} y2={H - PAD_B} stroke="var(--text-3)" />
          <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={H - PAD_B} stroke="var(--text-3)" />

          {/* Y-axis labels */}
          <g font-size="10" font-family="var(--mono)" fill="var(--text-3)" text-anchor="end">
            <For each={[0, 0.25, 0.5, 0.75, 1.0]}>
              {(p) => (
                <text x={PAD_L - 6} y={yOf(p) + 3}>
                  {p.toFixed(2)}
                </text>
              )}
            </For>
            <text
              x={14}
              y={PAD_T + plotH / 2}
              text-anchor="middle"
              transform={`rotate(-90 14 ${PAD_T + plotH / 2})`}
              font-size="10"
              fill="var(--text-3)"
              letter-spacing="0.08em"
            >
              P(fall) · per segment
            </text>
          </g>

          {/* X-axis labels — relative seconds from session start */}
          <g font-size="10" font-family="var(--mono)" fill="var(--text-3)" text-anchor="middle">
            {(() => {
              const { start, end } = timeBounds()
              const total = end - start
              const ticks = 5
              return (
                <For each={Array.from({ length: ticks + 1 }, (_, i) => i / ticks)}>
                  {(frac) => {
                    const ts = start + frac * total
                    const label = `${(ts - start).toFixed(1)} s`
                    return (
                      <text x={xOf(ts)} y={H - PAD_B + 16}>
                        {label}
                      </text>
                    )
                  }}
                </For>
              )
            })()}
            <text
              x={PAD_L + plotW / 2}
              y={H - 8}
              font-size="10"
              fill="var(--text-3)"
              letter-spacing="0.08em"
            >
              t · session-relative seconds
            </text>
          </g>
        </svg>

        {/* Legend */}
        <div
          style={{
            display: 'flex',
            gap: '14px',
            'flex-wrap': 'wrap',
            'margin-top': '10px',
            'font-family': 'var(--mono)',
            'font-size': '11px',
            color: 'var(--text-3)',
            'align-items': 'center',
          }}
        >
          <For each={presentLabels()}>
            {(label) => {
              const c = colourFor(label)
              return (
                <span style={{ display: 'inline-flex', 'align-items': 'center', gap: '6px' }}>
                  <span
                    style={{
                      width: '12px',
                      height: '8px',
                      background: c.fill,
                      'border-bottom': `2px solid ${c.stroke}`,
                      display: 'inline-block',
                    }}
                  />
                  {label}
                </span>
              )
            }}
          </For>
          <Show when={(props.groupedFallEvents ?? []).length > 0}>
            <span style={{ display: 'inline-flex', 'align-items': 'center', gap: '6px' }}>
              <span
                style={{
                  width: '12px',
                  height: '8px',
                  background: 'rgba(192, 86, 58, 0.16)',
                  border: '1px dashed var(--terracotta)',
                  display: 'inline-block',
                }}
              />
              grouped fall event
            </span>
          </Show>
        </div>

        <Show when={props.harAttenuationApplied}>
          <p
            style={{
              'margin-top': '10px',
              'font-family': 'var(--mono)',
              'font-size': '11px',
              color: 'var(--text-3)',
              'line-height': '1.5',
            }}
          >
            Session-level HAR gate fired
            <Show when={props.harAttenuationLabel}>
              {' '}(
              <span style={{ color: 'var(--text-2)' }}>{props.harAttenuationLabel}</span>)
            </Show>
            : per-segment P(fall) spikes above are visible because the meta-classifier still
            emits high probabilities on confident locomotion windows, but the session-level
            vulnerability score and warning level have been attenuated by the gate documented in{' '}
            <code>fusion/vulnerability_score.py</code>.
          </p>
        </Show>
      </div>
    </Show>
  )
}
