// RuntimeInferenceSequenceDiagram.tsx
//
// Polished UML sequence diagram for the live runtime inference path:
//   Flutter app → Nginx → FastAPI → runtime_inference → PostgreSQL
//
// ─────────────────────────────────────────────────────────────────
// EDIT POINTS — keep this list short:
//   • LIFELINES   — five vertical lanes (one per service)
//   • MESSAGES    — horizontal arrows between lanes (synchronous + response)
//   • SELF_NOTES  — annotations on the runtime_inference activation bar
//   • LATENCY     — values shown in the SLO panel; mark "representative"
//                   unless a real benchmark file lands in /results
// ─────────────────────────────────────────────────────────────────
//
// Visual style is intentionally restrained: warm off-white background,
// dark charcoal text, muted sage/teal for backend + model components,
// muted clay/red-orange ONLY on synchronous request arrows. Colours are
// hex-coded (not CSS variables) so the SVG/PNG export looks identical
// to the on-screen render.

import { For } from 'solid-js'
import { Section } from '../../../components/v6'

// ── Palette ──────────────────────────────────────────────────────
const C = {
  paper: '#faf7f1',     // warm off-white background
  bgMuted: '#ece7df',   // soft panel background (gutter, side panels)
  white: '#ffffff',     // lifeline-head fill
  text: '#1a140e',      // dark charcoal (primary text)
  textMuted: '#4a4138', // secondary text
  textDim: '#7a6f63',   // tertiary text / captions
  line: 'rgba(40,28,20,0.18)',
  lineSoft: 'rgba(40,28,20,0.10)',
  teal: '#3f7a83',      // muted sage-teal: backend / model components
  tealLight: '#dde8eb', // activation-bar fill
  moss: '#5e7a4a',      // sage variant (reserved)
  clay: '#c0563a',      // muted clay/red-orange: ONLY for sync request arrows
}

const FONT_SERIF = "'Instrument Serif', 'Iowan Old Style', Georgia, serif"
const FONT_MONO = "'JetBrains Mono', ui-monospace, monospace"

// ── LIFELINES ────────────────────────────────────────────────────
type Lifeline = {
  x: number
  label: string
  sub: string
}
const lifelines: Lifeline[] = [
  { x: 150, label: 'Flutter app', sub: 'mobile client' },
  { x: 420, label: 'Nginx', sub: 'TLS reverse proxy' },
  { x: 720, label: 'FastAPI router', sub: '/v1/infer/session' },
  { x: 1020, label: 'runtime_inference', sub: 'HAR + fall scorers (in-process)' },
  { x: 1340, label: 'PostgreSQL', sub: 'sessions · windows · inferences · timeline_events · feedback' },
]

const TOP = 200
const BOTTOM = 820

// ── MESSAGES (arrows between lifelines) ─────────────────────────
type Message = {
  t: number          // wall-clock ms since request entry
  y: number
  from: number
  to: number
  label: string
  detail?: string
  dashed?: boolean   // dashed = response / completion
}
const messages: Message[] = [
  { t: 0, y: 244, from: 0, to: 1, label: 'POST /v1/infer/session', detail: 'JWT + gzip payload' },
  { t: 2, y: 296, from: 1, to: 2, label: 'forward request' },
  { t: 5, y: 348, from: 2, to: 3, label: 'dispatch validated payload' },
  { t: 55, y: 600, from: 3, to: 2, label: 'fused result', detail: 'activity + P(fall) + alert decision', dashed: true },
  { t: 57, y: 650, from: 2, to: 4, label: 'INSERT inference · session · timeline rows' },
  { t: 60, y: 700, from: 4, to: 2, label: 'row ids', dashed: true },
  { t: 62, y: 750, from: 2, to: 1, label: '200 JSON', dashed: true },
  { t: 64, y: 790, from: 1, to: 0, label: '200 JSON · render alert', dashed: true },
]

// ── ACTIVATION BAR (in-process work on runtime_inference) ───────
const ACTIVATION_TOP = 380
const ACTIVATION_BOTTOM = 580
const SELF_NOTES = [
  { y: 408, label: 'validate payload' },
  { y: 432, label: 'preprocess + window IMU data' },
  { y: 472, label: 'HAR model → activity label' },
  { y: 500, label: 'Fall model → calibrated P(fall)' },
  { y: 552, label: 'fuse → activity + P(fall) + alert decision' },
]

// ── LATENCY / SLO ────────────────────────────────────────────────
// Representative figures from internal smoke runs. Replace with
// measured values if and when a benchmark JSON lands in /results.
const LATENCY = {
  median: 78,
  p95: 92,
  p99: 178,
  budget: 250,
  caveat: 'representative · replace once /results/latency benchmark lands',
}

// ── EXPORT HELPERS ───────────────────────────────────────────────
function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

function exportSvg(svgEl: SVGSVGElement) {
  const clone = svgEl.cloneNode(true) as SVGSVGElement
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
  const xml = new XMLSerializer().serializeToString(clone)
  const blob = new Blob([`<?xml version="1.0" encoding="UTF-8"?>\n${xml}`], { type: 'image/svg+xml' })
  downloadBlob(blob, 'runtime-inference-sequence.svg')
}

function exportPng(svgEl: SVGSVGElement, width = 2400) {
  const clone = svgEl.cloneNode(true) as SVGSVGElement
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
  const xml = new XMLSerializer().serializeToString(clone)
  const svg64 = btoa(unescape(encodeURIComponent(xml)))
  const img = new Image()
  img.onload = () => {
    const vb = svgEl.viewBox.baseVal
    const aspect = vb.height / vb.width
    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = Math.round(width * aspect)
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.fillStyle = C.paper
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
    canvas.toBlob((blob) => {
      if (blob) downloadBlob(blob, 'runtime-inference-sequence.png')
    }, 'image/png')
  }
  img.src = `data:image/svg+xml;base64,${svg64}`
}

// Mobile stacked summary — one line per step, time-ordered. Includes
// the in-process work that the desktop diagram shows on the activation
// bar so this view is also self-sufficient.
const stackedSteps: { t: string; from: string; to?: string; label: string; detail?: string }[] = [
  { t: '0', from: 'Flutter app', to: 'Nginx', label: 'POST /v1/infer/session', detail: 'JWT + gzip payload' },
  { t: '2', from: 'Nginx', to: 'FastAPI', label: 'forward request' },
  { t: '5', from: 'FastAPI', to: 'runtime_inference', label: 'dispatch validated payload' },
  { t: '8', from: 'runtime_inference', label: 'validate payload + preprocess + window IMU' },
  { t: '14 – 52', from: 'runtime_inference', label: 'parallel scoring', detail: 'HAR model → activity label · Fall model → calibrated P(fall)' },
  { t: '54', from: 'runtime_inference', label: 'fuse', detail: 'activity + P(fall) + alert decision' },
  { t: '55', from: 'runtime_inference', to: 'FastAPI', label: 'fused result' },
  { t: '57', from: 'FastAPI', to: 'PostgreSQL', label: 'INSERT inference · session · timeline rows' },
  { t: '60', from: 'PostgreSQL', to: 'FastAPI', label: 'row ids' },
  { t: '62', from: 'FastAPI', to: 'Nginx', label: '200 JSON' },
  { t: '64', from: 'Nginx', to: 'Flutter app', label: '200 JSON · render alert' },
]

// ── COMPONENT ────────────────────────────────────────────────────
export function RuntimeInferenceSequenceDiagram() {
  let svgRef: SVGSVGElement | undefined

  return (
    <Section num="03" title="Runtime inference" emphasis="sequence">
      <div class="fig figure-doc runtime-flow-figure" style={{ '--accent-c': C.teal }}>
        <div class="runtime-toolbar">
          <button type="button" class="runtime-export-btn" onClick={() => svgRef && exportSvg(svgRef)}>
            Download SVG
          </button>
          <button type="button" class="runtime-export-btn" onClick={() => svgRef && exportPng(svgRef)}>
            Download PNG
          </button>
        </div>

        <div class="runtime-svg-wrap">
          <svg
            ref={(el) => (svgRef = el)}
            class="fig-svg"
            viewBox="0 0 1500 1000"
            preserveAspectRatio="xMidYMid meet"
            role="img"
            aria-label="Runtime inference sequence diagram for live smartphone HAR and fall detection"
          >
            <defs>
              <marker id="rtisd-sync" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" orient="auto">
                <path d="M0,0 L10,5 L0,10 z" fill={C.clay} />
              </marker>
              <marker id="rtisd-async" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" orient="auto">
                <path d="M0,0 L10,5 L0,10 z" fill={C.teal} />
              </marker>
            </defs>

            {/* Background + figure title */}
            <rect width="1500" height="1000" fill={C.paper} />
            <text x="40" y="50" font-family={FONT_SERIF} font-size="24" font-weight="700" fill={C.text}>
              Runtime inference sequence for live smartphone HAR and fall detection
            </text>
            <text x="40" y="74" font-family={FONT_MONO} font-size="12" fill={C.textDim}>
              Figure 03 · time progresses downward · units = wall-clock ms since request entry
            </text>

            {/* Lifeline heads + verticals */}
            <For each={lifelines}>
              {(l, i) => {
                // Mobile client gets the clay accent because it is the originator
                // of every synchronous request edge in this figure.
                const accent = i() === 0 ? C.clay : C.teal
                return (
                  <g>
                    <rect x={l.x - 130} y={TOP - 100} width={260} height={70} rx={8} fill={C.white} stroke={accent} stroke-width="1.6" />
                    <text x={l.x} y={TOP - 70} text-anchor="middle" font-family={FONT_SERIF} font-size="18" font-weight="700" fill={C.text}>
                      {l.label}
                    </text>
                    <text x={l.x} y={TOP - 48} text-anchor="middle" font-family={FONT_MONO} font-size="11" fill={C.textDim}>
                      {l.sub}
                    </text>
                    <line x1={l.x} y1={TOP - 24} x2={l.x} y2={BOTTOM} stroke={accent} stroke-width="1.2" stroke-dasharray="4 6" stroke-opacity="0.55" />
                  </g>
                )
              }}
            </For>

            {/* Time gutter on the left */}
            <rect x="20" y={TOP - 24} width="42" height={BOTTOM - TOP + 24} fill={C.bgMuted} stroke={C.lineSoft} rx={4} />
            <text x="41" y={TOP - 34} text-anchor="middle" font-family={FONT_MONO} font-size="10.5" fill={C.textDim}>t (ms)</text>
            <For each={messages}>
              {(m) => (
                <g>
                  <line x1="20" y1={m.y} x2="62" y2={m.y} stroke={C.lineSoft} />
                  <text x="41" y={m.y + 4} text-anchor="middle" font-family={FONT_MONO} font-size="11" fill={C.text} font-weight="700">{m.t}</text>
                </g>
              )}
            </For>

            {/* Activation bar on runtime_inference lifeline */}
            <rect
              x={lifelines[3].x - 9}
              y={ACTIVATION_TOP}
              width={18}
              height={ACTIVATION_BOTTOM - ACTIVATION_TOP}
              rx={3}
              fill={C.tealLight}
              stroke={C.teal}
              stroke-width="1.4"
            />
            <For each={SELF_NOTES}>
              {(note) => (
                <g>
                  <line x1={lifelines[3].x + 10} y1={note.y} x2={lifelines[3].x + 32} y2={note.y} stroke={C.teal} stroke-width="1.2" />
                  <text x={lifelines[3].x + 40} y={note.y + 4} font-family={FONT_MONO} font-size="11.5" fill={C.text}>
                    {note.label}
                  </text>
                </g>
              )}
            </For>

            {/* Messages */}
            <For each={messages}>
              {(m) => {
                const fromX = lifelines[m.from].x
                const toX = lifelines[m.to].x
                const dir = toX > fromX ? 1 : -1
                const x1 = fromX + dir * 6
                const x2 = toX - dir * 6
                const labelX = (fromX + toX) / 2
                const colour = m.dashed ? C.teal : C.clay
                const marker = m.dashed ? 'rtisd-async' : 'rtisd-sync'
                return (
                  <g>
                    <path
                      d={`M${x1},${m.y} L${x2},${m.y}`}
                      stroke={colour}
                      stroke-width="2"
                      stroke-dasharray={m.dashed ? '8 5' : undefined}
                      fill="none"
                      marker-end={`url(#${marker})`}
                    />
                    <text x={labelX} y={m.y - 10} text-anchor="middle" font-family={FONT_MONO} font-size="12" fill={C.text} font-weight="700">
                      {m.label}
                    </text>
                    {m.detail && (
                      <text x={labelX} y={m.y + 18} text-anchor="middle" font-family={FONT_MONO} font-size="11" fill={C.textDim} font-style="italic">
                        {m.detail}
                      </text>
                    )}
                  </g>
                )
              }}
            </For>

            {/* Legend */}
            <g transform="translate(40 860)">
              <rect width="600" height="110" rx="8" fill={C.white} stroke={C.line} />
              <text x="20" y="28" font-family={FONT_MONO} font-size="11" fill={C.textDim} font-weight="700" letter-spacing="0.06em">
                LEGEND
              </text>
              <line x1="20" y1="56" x2="76" y2="56" stroke={C.clay} stroke-width="2" marker-end="url(#rtisd-sync)" />
              <text x="92" y="60" font-family={FONT_MONO} font-size="12" fill={C.text}>
                solid arrow · synchronous request
              </text>
              <line x1="20" y1="80" x2="76" y2="80" stroke={C.teal} stroke-width="2" stroke-dasharray="8 5" marker-end="url(#rtisd-async)" />
              <text x="92" y="84" font-family={FONT_MONO} font-size="12" fill={C.text}>
                dashed arrow · response / completion
              </text>
              <rect x="372" y="50" width="14" height="14" rx="3" fill={C.tealLight} stroke={C.teal} stroke-width="1.4" />
              <text x="396" y="62" font-family={FONT_MONO} font-size="12" fill={C.text}>
                vertical bar · in-process model scoring
              </text>
            </g>

            {/* Latency / SLO panel */}
            <g transform="translate(680 860)">
              <rect width="780" height="110" rx="8" fill={C.white} stroke={C.line} />
              <text x="20" y="28" font-family={FONT_MONO} font-size="11" fill={C.textDim} font-weight="700" letter-spacing="0.06em">
                END-TO-END LATENCY
              </text>
              <text x="20" y="60" font-family={FONT_MONO} font-size="14" fill={C.text}>
                median = {LATENCY.median} ms · p95 = {LATENCY.p95} ms · p99 = {LATENCY.p99} ms
              </text>
              <text x="20" y="82" font-family={FONT_MONO} font-size="13" fill={C.textMuted}>
                SLO budget = {LATENCY.budget} ms
              </text>
              <text x="20" y="100" font-family={FONT_MONO} font-size="11" fill={C.textDim} font-style="italic">
                {LATENCY.caveat}
              </text>
            </g>
          </svg>
        </div>

        {/* Mobile stacked fallback — hidden on desktop, shown via media query */}
        <ol class="runtime-stacked-fallback">
          <For each={stackedSteps}>
            {(s) => (
              <li>
                <span class="t">t = {s.t} ms</span>
                <span class="hop">
                  {s.from}
                  {s.to && (
                    <>
                      <span class="arrow"> → </span>
                      {s.to}
                    </>
                  )}
                </span>
                <span class="label">{s.label}</span>
                {s.detail && <span class="detail"> — {s.detail}</span>}
              </li>
            )}
          </For>
          <li class="latency-foot">
            END-TO-END LATENCY · median {LATENCY.median} ms · p95 {LATENCY.p95} ms · p99 {LATENCY.p99} ms · SLO {LATENCY.budget} ms ({LATENCY.caveat})
          </li>
        </ol>
      </div>

      <div class="sci-cap">
        <strong>Figure 03.</strong> The mobile client uploads a compressed IMU session through the
        reverse proxy to the FastAPI inference endpoint. The validated payload is preprocessed,
        scored by separate HAR and fall-detection model branches, fused into an activity label and
        alert decision, persisted to PostgreSQL, and returned to the client as a JSON response.
      </div>
    </Section>
  )
}
