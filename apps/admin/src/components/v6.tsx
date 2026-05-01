import { type JSXElement, type ParentProps, Show, For } from 'solid-js'

export function classNames(...parts: Array<string | false | null | undefined>) {
  return parts.filter(Boolean).join(' ')
}

export function Eyebrow(props: ParentProps) {
  return <p class="eyebrow">{props.children}</p>
}

export function SectionHead(props: {
  num: string
  title: string
  emphasis?: string
  kicker?: string
}) {
  return (
    <div class="section-head">
      <span class="section-num">{props.num}</span>
      <h2 class="section-title">
        {props.title}
        <Show when={props.emphasis}>
          {' '}
          <em>{props.emphasis}</em>
        </Show>
      </h2>
      <Show when={props.kicker}>
        <span class="section-kicker">{props.kicker}</span>
      </Show>
    </div>
  )
}

export function Section(
  props: ParentProps<{
    num?: string
    title?: string
    emphasis?: string
    kicker?: string
    id?: string
  }>,
) {
  return (
    <div class="section" id={props.id}>
      <Show when={props.num && props.title}>
        <SectionHead
          num={props.num as string}
          title={props.title as string}
          emphasis={props.emphasis}
          kicker={props.kicker}
        />
      </Show>
      {props.children}
    </div>
  )
}

export function Kpi(props: {
  label: string
  value: string | number
  unit?: string
  foot?: JSXElement
}) {
  return (
    <div class="kpi reveal in">
      <div class="kpi-label">{props.label}</div>
      <div class="kpi-value">
        {props.value}
        <Show when={props.unit}>
          <span class="kpi-unit">{props.unit}</span>
        </Show>
      </div>
      <Show when={props.foot}>
        <div class="kpi-foot">{props.foot}</div>
      </Show>
    </div>
  )
}

export function Kpis(props: ParentProps) {
  return <div class="kpis">{props.children}</div>
}

export function Metric(props: {
  label: string
  value: string | number
  sub?: JSXElement
  alert?: boolean
  unit?: string
}) {
  return (
    <div class={classNames('metric', props.alert && 'alert')}>
      <div class="lbl">{props.label}</div>
      <div class="val">
        {props.value}
        <Show when={props.unit}>
          <span class="kpi-unit"> {props.unit}</span>
        </Show>
      </div>
      <Show when={props.sub}>
        <div class="sub">{props.sub}</div>
      </Show>
    </div>
  )
}

export function Gcard(
  props: ParentProps<{
    title: string
    sub?: string
    pill?: JSXElement
    foot?: JSXElement
    bare?: boolean
  }>,
) {
  return (
    <div class="gcard">
      <div class="gcard-h" style={props.bare ? { padding: '22px 22px 0', 'margin-bottom': '14px' } : undefined}>
        <h4>{props.title}</h4>
        <Show when={props.sub}>
          <span class="stl">{props.sub}</span>
        </Show>
        <Show when={props.pill}>
          <span class="pill">{props.pill}</span>
        </Show>
      </div>
      {props.children}
      <Show when={props.foot}>
        <div class="gcard-foot">{props.foot}</div>
      </Show>
    </div>
  )
}

export type WarningLevel = 'high' | 'med' | 'low' | 'none' | 'live'

export function WarningTag(props: { level: WarningLevel; children: JSXElement }) {
  return <span class={`tag ${props.level}`}>{props.children}</span>
}

export function ProbBar(props: { value: number; alert?: boolean }) {
  const pct = Math.max(0, Math.min(1, props.value)) * 100
  return (
    <div style={{ display: 'flex', gap: '8px', 'align-items': 'center' }}>
      <div class="bar">
        <div class="bar-fill" style={{ width: `${pct}%` }} />
      </div>
      <span class="mono" style={props.alert ? { color: 'var(--terracotta)' } : undefined}>
        {props.value.toFixed(2)}
      </span>
    </div>
  )
}

export function FeedItem(props: {
  level: 'alert' | 'ok' | 'info' | 'mute'
  time: string
  children: JSXElement
}) {
  return (
    <div class="feed-item">
      <span class={`feed-dot ${props.level === 'mute' ? '' : props.level}`} />
      <div class="feed-msg">{props.children}</div>
      <span class="feed-time">{props.time}</span>
    </div>
  )
}

export function FeedList(props: ParentProps) {
  return <div class="feed">{props.children}</div>
}

export function PullQuote(props: ParentProps<{ cite?: string }>) {
  return (
    <div class="pullquote">
      {props.children}
      <Show when={props.cite}>
        <cite>{props.cite}</cite>
      </Show>
    </div>
  )
}

export function DatasetTable(props: {
  rows: ReadonlyArray<{
    name: string
    subjects: number
    activities: string
    sensors: string
    sampling: string
    role: string
  }>
}) {
  return (
    <table class="dataset-table">
      <thead>
        <tr>
          <th>Dataset</th>
          <th>Subjects</th>
          <th>Activities / Falls</th>
          <th>Sensors</th>
          <th>Sampling</th>
          <th>Use</th>
        </tr>
      </thead>
      <tbody>
        <For each={props.rows}>
          {(r) => (
            <tr>
              <td>{r.name}</td>
              <td class="mono">{r.subjects}</td>
              <td>{r.activities}</td>
              <td class="mono">{r.sensors}</td>
              <td class="mono">{r.sampling}</td>
              <td>{r.role}</td>
            </tr>
          )}
        </For>
      </tbody>
    </table>
  )
}

// Loading + empty placeholders, restyled with v6 vocabulary.
export function FullPageLoader(props: { title: string; subtitle?: string }) {
  return (
    <div
      style={{
        display: 'grid',
        'place-items': 'center',
        'min-height': '70vh',
        padding: '48px 24px',
        background: 'var(--bg)',
      }}
    >
      <div
        style={{
          'max-width': '420px',
          'text-align': 'center',
          padding: '48px 32px',
          background: 'var(--paper)',
          border: '1px solid var(--line)',
          'border-radius': '14px',
          'box-shadow': 'var(--shadow-md)',
        }}
      >
        <div
          style={{
            margin: '0 auto 18px',
            width: '46px',
            height: '46px',
            'border-radius': '12px',
            background: 'var(--terracotta-dim)',
            animation: 'pip 1.6s ease-in-out infinite',
          }}
        />
        <h2 style={{ 'font-family': 'var(--serif)', 'font-size': '28px', 'letter-spacing': '-.02em' }}>
          {props.title}
        </h2>
        <Show when={props.subtitle}>
          <p style={{ 'margin-top': '10px', color: 'var(--text-3)', 'font-size': '13px', 'line-height': 1.6 }}>
            {props.subtitle}
          </p>
        </Show>
      </div>
    </div>
  )
}

export function DataState(props: { title: string; description: string; action?: JSXElement }) {
  return (
    <div
      style={{
        padding: '36px 28px',
        border: '1px dashed var(--line-strong)',
        'border-radius': '14px',
        background: 'var(--paper)',
        'text-align': 'center',
      }}
    >
      <h3 style={{ 'font-family': 'var(--serif)', 'font-size': '22px', 'letter-spacing': '-.015em' }}>
        {props.title}
      </h3>
      <p style={{ 'margin-top': '8px', color: 'var(--text-3)', 'font-size': '13px', 'line-height': 1.6 }}>
        {props.description}
      </p>
      <Show when={props.action}>
        <div style={{ 'margin-top': '18px' }}>{props.action}</div>
      </Show>
    </div>
  )
}
