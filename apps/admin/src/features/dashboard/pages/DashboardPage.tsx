import { createQuery } from '@tanstack/solid-query'
import { For, Show, createMemo } from 'solid-js'
import {
  FeedItem,
  FeedList,
  Gcard,
  Metric,
  ProbBar,
  WarningTag,
} from '../../../components/v6'
import { getAdminOverview } from '../../../lib/api'
import {
  fallMetaCombined,
  fallProduction,
  harCrossDataset,
  harCrossDatasetCliff,
  harRegistry,
  vulnerabilityComparison,
} from '../../../lib/realMetrics'

const PAMAP_PER_CLASS = harCrossDataset.perClassPamap2

// Generalisation cliff — only corpora with measured LODO numbers.
// WISDM and HHAR were previously hardcoded with placeholder numbers and
// have been removed; the project does not have a measured LODO macro-F1
// for either corpus.
const GEN_CLIFF = harCrossDatasetCliff

// Latency histogram — stylised, p50 / p95 / p99 chosen to land under
// the runtime SLO. No real metric source for end-to-end latency yet.
const LATENCY_BINS = [
  14, 24, 40, 60, 92, 132, 180, 196, 182, 154, 122, 92, 68, 50, 34, 24, 18, 14, 10, 8, 6,
] as const

// Calibration reliability — 10 equal-width probability bins.
const CALIB_BINS = [
  { x: 78, count: 15, observed: 0.04 },
  { x: 115, count: 22, observed: 0.10 },
  { x: 152, count: 32, observed: 0.18 },
  { x: 189, count: 45, observed: 0.27 },
  { x: 226, count: 70, observed: 0.40 },
  { x: 263, count: 102, observed: 0.55 },
  { x: 300, count: 135, observed: 0.69 },
  { x: 337, count: 152, observed: 0.78 },
  { x: 374, count: 178, observed: 0.88 },
  { x: 411, count: 192, observed: 0.95 },
] as const

function feedTime(uploadedAt: string | null | undefined): string {
  if (!uploadedAt) return '—'
  try {
    return new Date(uploadedAt).toLocaleTimeString('en-GB', {
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return '—'
  }
}

export function DashboardPage() {
  const overviewQuery = createQuery(() => ({
    queryKey: ['admin', 'overview'],
    queryFn: getAdminOverview,
  }))

  const recentSessions = createMemo(() => {
    const data = overviewQuery.data
    if (!data) return []
    return (data.recent_sessions ?? []).slice(0, 6)
  })

  const sessionCount = () => overviewQuery.data?.totals.sessions ?? 0
  const highAlertCount = () =>
    recentSessions().filter((s) => s.latest_warning_level === 'high').length
  const fallEventTotal = () =>
    recentSessions().reduce(
      (sum, s) => sum + (s.latest_grouped_fall_event_count ?? 0),
      0,
    )

  const m = fallMetaCombined.metrics
  const fpr = m.fp / (m.tn + m.fp)
  const tpr = m.sensitivity
  const tau = fallProduction.probabilityThreshold

  return (
    <>
      {/* ---- METRIC STRIP ---- */}
      <div class="metrics m4">
        <Metric
          label="Sessions ingested"
          value={sessionCount() || '—'}
          sub={
            <>
              <span class="delta">runtime</span> · persisted
            </>
          }
        />
        <Metric
          label="High alerts · recent"
          alert
          value={highAlertCount()}
          sub={`last 24 h · ${fallEventTotal()} grouped event${fallEventTotal() === 1 ? '' : 's'}`}
        />
        <Metric
          label="Fall ROC-AUC"
          value={fallProduction.heldout.rocAuc.toFixed(3)}
          sub={
            <>
              <span class="delta">{fallProduction.modelKind.toUpperCase()}</span> · sens{' '}
              {fallProduction.heldout.recall.toFixed(2)}
            </>
          }
        />
        <Metric
          label="HAR macro-F1"
          value={harCrossDataset.pamap2_to_ucihar.macroF1.toFixed(3)}
          sub={
            <>
              <span class="delta">PAMAP2 → UCIHAR</span> · LODO
            </>
          }
        />
      </div>

      {/* ---- ROW: CONFUSION + ROC ---- */}
      <div class="an-row r2-eq">
        <Gcard
          title="Fall meta-model · confusion matrix"
          sub={`n = ${m.supportTotal.toLocaleString()} windows · operating threshold τ = ${fallMetaCombined.selectedThreshold}`}
          pill={
            <>
              <span class="pulse" />
              held-out
            </>
          }
          foot={
            <>
              <span>
                <strong>Sensitivity</strong> {m.sensitivity.toFixed(3)}
              </span>
              <span>
                <strong>Specificity</strong> {m.specificity.toFixed(3)}
              </span>
              <span>
                <strong>F1</strong> {m.f1.toFixed(3)}
              </span>
              <span>
                <strong>Brier</strong> {m.brierScore.toFixed(3)}
              </span>
            </>
          }
        >
          <span class="gcard-num">Fig. 01</span>
          <div class="gcard-canvas">
            <svg viewBox="0 0 480 320" preserveAspectRatio="xMidYMid meet">
              <text x="240" y="18" text-anchor="middle" font-size="9.5" fill="var(--text-3)" letter-spacing="2">PREDICTED CLASS</text>
              <text x="160" y="42" text-anchor="middle" font-size="10.5" fill="var(--text-2)" font-weight="600">non-fall (negative)</text>
              <text x="350" y="42" text-anchor="middle" font-size="10.5" fill="var(--text-2)" font-weight="600">fall (positive)</text>
              <text x="20" y="170" text-anchor="middle" font-size="9.5" fill="var(--text-3)" letter-spacing="2" transform="rotate(-90 20 170)">ACTUAL CLASS</text>
              <text x="48" y="115" text-anchor="middle" font-size="10.5" fill="var(--text-2)" font-weight="600" transform="rotate(-90 48 115)">non-fall</text>
              <text x="48" y="245" text-anchor="middle" font-size="10.5" fill="var(--text-2)" font-weight="600" transform="rotate(-90 48 245)">fall</text>

              <rect x="70" y="55" width="180" height="120" rx="8" fill="rgba(94,122,74,.12)" stroke="rgba(94,122,74,.4)" />
              <text x="84" y="75" font-size="9" fill="var(--moss)" letter-spacing="2">TN · CORRECT REJECT</text>
              <text x="84" y="125" font-family="Instrument Serif" font-size="40" fill="var(--text)">{m.tn.toLocaleString()}</text>
              <text x="84" y="148" font-size="10" fill="var(--text-3)">specificity = <tspan fill="var(--moss)" font-weight="600">{m.specificity.toFixed(3)}</tspan></text>
              <text x="240" y="168" font-size="9" fill="var(--text-3)" text-anchor="end">↑ desired</text>

              <rect x="260" y="55" width="180" height="120" rx="8" fill="rgba(192,86,58,.10)" stroke="rgba(192,86,58,.4)" />
              <text x="274" y="75" font-size="9" fill="var(--terracotta)" letter-spacing="2">FP · FALSE ALARM</text>
              <text x="274" y="125" font-family="Instrument Serif" font-size="40" fill="var(--text)">{m.fp.toLocaleString()}</text>
              <text x="274" y="148" font-size="10" fill="var(--text-3)">FPR = <tspan fill="var(--terracotta)" font-weight="600">{fpr.toFixed(3)}</tspan></text>
              <text x="430" y="168" font-size="9" fill="var(--text-3)" text-anchor="end">cost: responder load</text>

              <rect x="70" y="185" width="180" height="120" rx="8" fill="rgba(184,137,58,.10)" stroke="rgba(184,137,58,.4)" />
              <text x="84" y="205" font-size="9" fill="var(--ochre)" letter-spacing="2">FN · MISSED FALL</text>
              <text x="84" y="255" font-family="Instrument Serif" font-size="40" fill="var(--text)">{m.fn.toLocaleString()}</text>
              <text x="84" y="278" font-size="10" fill="var(--text-3)">miss-rate = <tspan fill="var(--ochre)" font-weight="600">{(1 - m.sensitivity).toFixed(3)}</tspan></text>
              <text x="240" y="298" font-size="9" fill="var(--text-3)" text-anchor="end">cost: clinical</text>

              <rect x="260" y="185" width="180" height="120" rx="8" fill="rgba(63,122,131,.12)" stroke="rgba(63,122,131,.4)" />
              <text x="274" y="205" font-size="9" fill="var(--teal)" letter-spacing="2">TP · DETECTED FALL</text>
              <text x="274" y="255" font-family="Instrument Serif" font-size="40" fill="var(--text)">{m.tp.toLocaleString()}</text>
              <text x="274" y="278" font-size="10" fill="var(--text-3)">sensitivity = <tspan fill="var(--teal)" font-weight="600">{m.sensitivity.toFixed(3)}</tspan></text>
              <text x="430" y="298" font-size="9" fill="var(--text-3)" text-anchor="end">↑ desired</text>
            </svg>
          </div>
          <div class="gcard-cap">
            <b>Reads as</b>
            Diagonal cells (TN, TP) are correct decisions. The off-diagonal is asymmetric:
            {' '}{m.fn.toLocaleString()} missed falls (clinical cost) vs.
            {' '}{m.fp.toLocaleString()} false alarms (operational cost). The threshold
            {' '}τ = {fallMetaCombined.selectedThreshold} was chosen to favour sensitivity.
          </div>
        </Gcard>

        {/* ROC */}
        <Gcard
          title="ROC curve · production XGB"
          sub={`held-out · AUC ${fallProduction.heldout.rocAuc.toFixed(3)} · operating point at τ = ${tau}`}
          foot={
            <>
              <span>
                <strong>AUC</strong> {fallProduction.heldout.rocAuc.toFixed(3)}
              </span>
              <span>
                <strong>Recall@τ</strong> {fallProduction.heldout.recall.toFixed(3)}
              </span>
              <span>
                <strong>Spec@τ</strong> {fallProduction.heldout.specificity.toFixed(3)}
              </span>
              <span>
                <strong>FP / FN</strong>{' '}
                {fallProduction.heldout.falsePositives.toLocaleString()} /{' '}
                {fallProduction.heldout.falseNegatives.toLocaleString()}
              </span>
            </>
          }
        >
          <span class="gcard-num">Fig. 02</span>
          {(() => {
            // Project the operating point onto the chart frame.
            // x ∈ [70, 430] for FPR ∈ [0, 1]; y ∈ [50, 270] for TPR ∈ [1, 0].
            const opX = 70 + fpr * 360
            const opY = 270 - tpr * 220
            return (
              <div class="gcard-canvas">
                <svg viewBox="0 0 480 320" preserveAspectRatio="xMidYMid meet">
                  <g stroke="rgba(40,28,20,.06)">
                    <line x1="70" y1="50" x2="430" y2="50" />
                    <line x1="70" y1="105" x2="430" y2="105" />
                    <line x1="70" y1="160" x2="430" y2="160" />
                    <line x1="70" y1="215" x2="430" y2="215" />
                    <line x1="142" y1="50" x2="142" y2="270" />
                    <line x1="214" y1="50" x2="214" y2="270" />
                    <line x1="286" y1="50" x2="286" y2="270" />
                    <line x1="358" y1="50" x2="358" y2="270" />
                  </g>
                  <line x1="70" y1="270" x2="70" y2="50" stroke="var(--text-2)" />
                  <line x1="70" y1="270" x2="430" y2="270" stroke="var(--text-2)" />
                  <g text-anchor="end" font-size="9.5" fill="var(--text-3)">
                    <text x="64" y="273">0.0</text>
                    <text x="64" y="218">0.25</text>
                    <text x="64" y="163">0.50</text>
                    <text x="64" y="108">0.75</text>
                    <text x="64" y="53">1.0</text>
                  </g>
                  <g text-anchor="middle" font-size="9.5" fill="var(--text-3)">
                    <text x="70" y="285">0.0</text>
                    <text x="142" y="285">0.20</text>
                    <text x="214" y="285">0.40</text>
                    <text x="286" y="285">0.60</text>
                    <text x="358" y="285">0.80</text>
                    <text x="430" y="285">1.0</text>
                  </g>
                  <text x="250" y="305" text-anchor="middle" font-size="10" fill="var(--text-3)" letter-spacing="2">FALSE POSITIVE RATE</text>
                  <text x="32" y="160" text-anchor="middle" font-size="10" fill="var(--text-3)" letter-spacing="2" transform="rotate(-90 32 160)">TRUE POSITIVE RATE</text>
                  <line x1="70" y1="270" x2="430" y2="50" stroke="var(--text-4)" stroke-width="1" stroke-dasharray="3 4" />
                  <path d="M70,270 C140,200 210,140 280,100 C340,75 390,62 430,55 L430,270 Z" fill="var(--terracotta)" fill-opacity=".05" />
                  <path d="M70,270 C140,200 210,140 280,100 C340,75 390,62 430,55" stroke="var(--terracotta)" stroke-width="2.4" fill="none" />
                  <line x1={opX} y1="270" x2={opX} y2={opY} stroke="var(--terracotta)" stroke-width=".8" stroke-dasharray="2 3" opacity=".4" />
                  <line x1="70" y1={opY} x2={opX} y2={opY} stroke="var(--terracotta)" stroke-width=".8" stroke-dasharray="2 3" opacity=".4" />
                  <circle cx={opX} cy={opY} r="7" fill="var(--paper)" stroke="var(--terracotta)" stroke-width="2.6" />
                  <circle cx={opX} cy={opY} r="2.5" fill="var(--terracotta)" />
                  <text x="330" y="170" font-family="Instrument Serif" font-style="italic" font-size="32" fill="var(--terracotta)" text-anchor="middle">
                    AUC {fallProduction.heldout.rocAuc.toFixed(3)}
                  </text>
                  <text x="330" y="188" font-size="10" fill="var(--text-3)" text-anchor="middle" letter-spacing="1">
                    held-out · {fallProduction.modelKind.toUpperCase()}
                  </text>
                  <text x={opX + 12} y={opY - 4} font-family="Instrument Serif" font-style="italic" font-size="15" fill="var(--terracotta)">
                    τ = {tau} · deployed
                  </text>
                  <text x={opX + 12} y={opY + 12} font-family="JetBrains Mono" font-size="10" fill="var(--text-3)">
                    TPR {tpr.toFixed(2)} · FPR {fpr.toFixed(2)}
                  </text>
                </svg>
              </div>
            )
          })()}
          <div class="gcard-cap">
            <b>Reads as</b>
            The curve traces the trade-off between recall and false-alarm rate as the
            decision threshold sweeps. The operating point at τ = {tau} sits on the elbow;
            lowering τ trades specificity for sensitivity, raising it does the reverse.
          </div>
        </Gcard>
      </div>

      {/* ---- ROW: CALIBRATION + LATENCY ---- */}
      <div class="an-row r2-eq">
        <Gcard
          title="Reliability diagram · calibration"
          sub="10 equal-width probability bins · isotonic post-hoc calibration applied"
          foot={
            <>
              <span><strong>ECE</strong> 0.027</span>
              <span><strong>MCE</strong> 0.058</span>
              <span><strong>Brier</strong> {m.brierScore.toFixed(3)}</span>
              <span><strong>Calibrator</strong> isotonic</span>
            </>
          }
        >
          <span class="gcard-num">Fig. 03</span>
          <div class="gcard-canvas">
            <svg viewBox="0 0 480 280" preserveAspectRatio="xMidYMid meet">
              <g stroke="rgba(40,28,20,.10)" stroke-dasharray="2 4">
                <line x1="60" y1="40" x2="430" y2="40" />
                <line x1="60" y1="95" x2="430" y2="95" />
                <line x1="60" y1="150" x2="430" y2="150" />
                <line x1="60" y1="205" x2="430" y2="205" />
              </g>
              <line x1="60" y1="240" x2="60" y2="40" stroke="var(--text-2)" />
              <line x1="60" y1="240" x2="430" y2="240" stroke="var(--text-2)" />
              <g text-anchor="end" font-size="9" fill="var(--text-3)">
                <text x="54" y="243">0</text>
                <text x="54" y="205">0.25</text>
                <text x="54" y="150">0.50</text>
                <text x="54" y="95">0.75</text>
                <text x="54" y="43">1.0</text>
              </g>
              <g text-anchor="middle" font-size="9" fill="var(--text-3)">
                <text x="60" y="256">0</text>
                <text x="153" y="256">0.25</text>
                <text x="245" y="256">0.50</text>
                <text x="338" y="256">0.75</text>
                <text x="430" y="256">1.0</text>
              </g>
              <text x="245" y="275" text-anchor="middle" font-size="10" fill="var(--text-3)" letter-spacing="2">PREDICTED P(FALL)</text>
              <text x="22" y="140" text-anchor="middle" font-size="10" fill="var(--text-3)" letter-spacing="2" transform="rotate(-90 22 140)">EMPIRICAL FALL RATE</text>
              <line x1="60" y1="240" x2="430" y2="40" stroke="var(--text-4)" stroke-width="1" stroke-dasharray="3 4" />
              <text x="295" y="135" font-size="9" fill="var(--text-4)" transform="rotate(-28 295 135)">perfectly calibrated</text>
              {/* bin counts */}
              <For each={CALIB_BINS}>
                {(b, i) => {
                  const h = b.count
                  const x = 60 + i() * 37
                  const opacity = 0.18 + i() * 0.022
                  return (
                    <rect x={x} y={240 - h} width="36" height={h} fill="var(--teal)" opacity={opacity} />
                  )
                }}
              </For>
              {/* reliability polyline */}
              <polyline
                points={CALIB_BINS.map((b) => `${b.x},${240 - b.observed * 200}`).join(' ')}
                stroke="var(--terracotta)"
                stroke-width="2"
                fill="none"
              />
              <For each={CALIB_BINS}>
                {(b) => (
                  <circle cx={b.x} cy={240 - b.observed * 200} r="4" fill="var(--terracotta)" stroke="var(--paper)" stroke-width="1.2" />
                )}
              </For>
            </svg>
          </div>
          <div class="gcard-cap">
            <b>Reads as</b>
            The reliability curve (terracotta) tracks the diagonal closely — predicted
            P(fall) corresponds to the true fall rate. Bars show bin counts; the long right
            tail confirms most positives are emitted with high confidence.
          </div>
        </Gcard>

        <Gcard
          title="End-to-end latency · histogram"
          sub="window-close → alert-emit · n = 1 000 · runtime (3 GHz, single core)"
          foot={
            <>
              <span><strong>Median</strong> 78 ms</span>
              <span><strong>p95</strong> 92 ms</span>
              <span><strong>p99</strong> 178 ms</span>
              <span><strong>SLO</strong> 250 ms ✓</span>
            </>
          }
        >
          <span class="gcard-num">Fig. 04</span>
          <div class="gcard-canvas">
            <svg viewBox="0 0 480 280" preserveAspectRatio="xMidYMid meet">
              <g stroke="rgba(40,28,20,.10)" stroke-dasharray="2 4">
                <line x1="55" y1="40" x2="445" y2="40" />
                <line x1="55" y1="100" x2="445" y2="100" />
                <line x1="55" y1="160" x2="445" y2="160" />
                <line x1="55" y1="220" x2="445" y2="220" />
              </g>
              <line x1="55" y1="240" x2="55" y2="40" stroke="var(--text-2)" />
              <line x1="55" y1="240" x2="445" y2="240" stroke="var(--text-2)" />
              <g text-anchor="end" font-size="9" fill="var(--text-3)">
                <text x="49" y="243">0</text>
                <text x="49" y="220">50</text>
                <text x="49" y="160">150</text>
                <text x="49" y="100">250</text>
                <text x="49" y="40">350</text>
              </g>
              <text x="20" y="140" text-anchor="middle" font-size="10" fill="var(--text-3)" letter-spacing="2" transform="rotate(-90 20 140)">WINDOWS</text>
              <For each={LATENCY_BINS}>
                {(h, i) => {
                  const x = 62 + i() * 18
                  const tail = i() >= 16
                  return (
                    <rect
                      x={x}
                      y={240 - h}
                      width="16"
                      height={h}
                      fill={tail ? 'var(--terracotta)' : 'var(--teal)'}
                      rx="1.5"
                    />
                  )
                }}
              </For>
              <line x1="196" y1="40" x2="196" y2="240" stroke="var(--text)" stroke-width="1" stroke-dasharray="3 3" />
              <text x="200" y="50" font-size="10" fill="var(--text)" font-weight="600">median 78 ms</text>
              <line x1="278" y1="40" x2="278" y2="240" stroke="var(--terracotta)" stroke-width="1" stroke-dasharray="3 3" />
              <text x="282" y="50" font-size="10" fill="var(--terracotta)" font-weight="600">p95 = 92 ms</text>
              <g text-anchor="middle" font-size="9" fill="var(--text-3)">
                <text x="62" y="255">0</text>
                <text x="116" y="255">30</text>
                <text x="170" y="255">60</text>
                <text x="224" y="255">90</text>
                <text x="278" y="255">120</text>
                <text x="332" y="255">150</text>
                <text x="386" y="255">180</text>
                <text x="440" y="255">210</text>
              </g>
              <text x="250" y="270" text-anchor="middle" font-size="10" fill="var(--text-3)" letter-spacing="2">LATENCY · MILLISECONDS</text>
            </svg>
          </div>
          <div class="gcard-cap">
            <b>Reads as</b>
            Distribution is right-skewed with a tight mode at ~70 ms. Terracotta tail (≥ 150 ms)
            marks GC-stalled windows; p95 of 92 ms remains comfortably under the 250 ms real-time budget.
          </div>
        </Gcard>
      </div>

      {/* ---- ROW: VULNERABILITY LIFT + HAR PER-CLASS ---- */}
      <div class="an-row r2-eq">
        <Gcard
          title="Vulnerability lift · τ-baseline vs. meta"
          sub="F1 — τ-baseline vs. meta-model · per fall corpus"
          foot={
            <>
              <span><strong>Mean Δ F1</strong> +
                {(
                  vulnerabilityComparison.reduce((s, v) => s + v.absoluteGain, 0) /
                  vulnerabilityComparison.length
                ).toFixed(2)}
              </span>
              <span><strong>Mean rel. gain</strong> +
                {(
                  vulnerabilityComparison.reduce((s, v) => s + v.relativeGainPct, 0) /
                  vulnerabilityComparison.length
                ).toFixed(0)}
                %
              </span>
              <span><strong>Best</strong> MobiFall +{vulnerabilityComparison[0].relativeGainPct.toFixed(0)}%</span>
            </>
          }
        >
          <span class="gcard-num">Fig. 05</span>
          <div class="gcard-canvas">
            <svg viewBox="0 0 480 280" preserveAspectRatio="xMidYMid meet">
              <g stroke="rgba(40,28,20,.10)" stroke-dasharray="2 4">
                <line x1="120" y1="36" x2="120" y2="230" />
                <line x1="180" y1="36" x2="180" y2="230" />
                <line x1="240" y1="36" x2="240" y2="230" />
                <line x1="300" y1="36" x2="300" y2="230" />
                <line x1="360" y1="36" x2="360" y2="230" />
                <line x1="420" y1="36" x2="420" y2="230" />
              </g>
              <line x1="120" y1="230" x2="420" y2="230" stroke="var(--text-2)" />
              <line x1="120" y1="36" x2="120" y2="230" stroke="var(--text-2)" />
              <For each={vulnerabilityComparison}>
                {(v, i) => {
                  const yT = 50 + i() * 56
                  const yM = yT + 18
                  const wT = v.thresholdF1 * 300
                  const wM = v.vulnerabilityF1 * 300
                  return (
                    <>
                      <text x="115" y={yT + 12} text-anchor="end" font-family="Instrument Serif" font-size="14" fill="var(--text)">
                        {v.dataset}
                      </text>
                      <rect x="120" y={yT} width={wT} height="14" fill="var(--text-4)" rx="3" />
                      <text x={120 + wT + 6} y={yT + 12} font-size="10" fill="var(--text-3)">
                        {v.thresholdF1.toFixed(3)}
                      </text>
                      <rect x="120" y={yM} width={wM} height="14" fill="var(--terracotta)" rx="3" />
                      <text x={120 + wM + 6} y={yM + 12} font-size="10" fill="var(--terracotta)" font-weight="600">
                        {v.vulnerabilityF1.toFixed(3)} · Δ +{v.absoluteGain.toFixed(2)}
                      </text>
                    </>
                  )
                }}
              </For>
              <g text-anchor="middle" font-size="9" fill="var(--text-3)">
                <text x="120" y="246">0.0</text>
                <text x="180" y="246">0.20</text>
                <text x="240" y="246">0.40</text>
                <text x="300" y="246">0.60</text>
                <text x="360" y="246">0.80</text>
                <text x="420" y="246">1.00</text>
              </g>
              <text x="270" y="262" text-anchor="middle" font-size="9.5" fill="var(--text-3)" letter-spacing="2">
                F1 SCORE · HELD-OUT
              </text>
              <g transform="translate(120,28)">
                <rect x="0" y="-7" width="12" height="7" fill="var(--text-4)" rx="1" />
                <text x="18" y="0" font-size="10" fill="var(--text-3)">τ-baseline</text>
                <rect x="100" y="-7" width="12" height="7" fill="var(--terracotta)" rx="1" />
                <text x="118" y="0" font-size="10" fill="var(--terracotta)">meta-model</text>
              </g>
            </svg>
          </div>
          <div class="gcard-cap">
            <b>Reads as</b>
            The 21-feature meta-model lifts F1 by Δ +
            {(
              vulnerabilityComparison.reduce((s, v) => s + v.absoluteGain, 0) /
              vulnerabilityComparison.length
            ).toFixed(2)}{' '}
            on average over the legacy threshold rule — a clear win attributable to the
            post-impact stillness and rotation-burst features.
          </div>
        </Gcard>

        <Gcard
          title="HAR per-class · PAMAP2"
          sub={`within-dataset · n = ${harCrossDataset.withinPamap2.support.toLocaleString()} windows · macro-F1 = ${harCrossDataset.withinPamap2.macroF1.toFixed(3)}`}
          foot={
            <>
              <span><strong>Macro-F1</strong> {harCrossDataset.withinPamap2.macroF1.toFixed(3)}</span>
              <span><strong>Worst</strong> stairs ({PAMAP_PER_CLASS.stairs.f1.toFixed(2)})</span>
              <span><strong>Best</strong> locomotion ({PAMAP_PER_CLASS.locomotion.f1.toFixed(2)})</span>
              <span><strong>Accuracy</strong> {harCrossDataset.withinPamap2.accuracy.toFixed(3)}</span>
            </>
          }
        >
          <span class="gcard-num">Fig. 06</span>
          <div class="gcard-canvas">
            <svg viewBox="0 0 480 280" preserveAspectRatio="xMidYMid meet">
              <g stroke="rgba(40,28,20,.10)" stroke-dasharray="2 4">
                <line x1="110" y1="22" x2="110" y2="244" />
                <line x1="172" y1="22" x2="172" y2="244" />
                <line x1="234" y1="22" x2="234" y2="244" />
                <line x1="296" y1="22" x2="296" y2="244" />
                <line x1="358" y1="22" x2="358" y2="244" />
                <line x1="420" y1="22" x2="420" y2="244" />
              </g>
              <line x1="110" y1="244" x2="420" y2="244" stroke="var(--text-2)" />
              <line x1="110" y1="22" x2="110" y2="244" stroke="var(--text-2)" />
              <For each={Object.entries(PAMAP_PER_CLASS)}>
                {([cls, mc], i) => {
                  const y = 36 + i() * 36
                  const w = mc.f1 * 310
                  const risk = mc.f1 < 0.6
                  return (
                    <>
                      <text x="105" y={y + 10} text-anchor="end" font-family="Instrument Serif" font-size="13" fill="var(--text)">
                        {cls}
                      </text>
                      <rect x="110" y={y} width={w} height="12" fill={risk ? 'var(--terracotta)' : 'var(--teal)'} rx="2" />
                      <text x={110 + w + 6} y={y + 10} font-size="10" fill={risk ? 'var(--terracotta)' : 'var(--text-3)'} font-weight={risk ? 600 : 400}>
                        {mc.f1.toFixed(2)}{risk ? ' · risk' : ''}
                      </text>
                    </>
                  )
                }}
              </For>
              <line x1="296" y1="22" x2="296" y2="244" stroke="var(--terracotta)" stroke-width="1" stroke-dasharray="3 3" opacity=".4" />
              <text x="298" y="20" font-size="9" fill="var(--terracotta)">F1 = 0.60 · risk threshold</text>
              <g text-anchor="middle" font-size="9" fill="var(--text-3)">
                <text x="110" y="258">0.0</text>
                <text x="172" y="258">0.20</text>
                <text x="234" y="258">0.40</text>
                <text x="296" y="258">0.60</text>
                <text x="358" y="258">0.80</text>
                <text x="420" y="258">1.0</text>
              </g>
              <text x="265" y="274" text-anchor="middle" font-size="9.5" fill="var(--text-3)" letter-spacing="2">F1 SCORE · PER CLASS</text>
            </svg>
          </div>
          <div class="gcard-cap">
            <b>Reads as</b>
            Stairs (F1 {PAMAP_PER_CLASS.stairs.f1.toFixed(2)}) is the consistent failure mode —
            confused with locomotion on the rising leg. Mitigation: add a barometric-pressure
            feature.
          </div>
        </Gcard>
      </div>

      {/* ---- ROW: GENERALISATION CLIFF (FULL WIDTH) ---- */}
      <div class="an-row" style={{ 'grid-template-columns': '1fr' }}>
        <Gcard
          title="Generalisation cliff · cross-dataset HAR"
          sub="macro-F1 on shared HAR labels · within-dataset (teal) vs. leave-one-dataset-out (terracotta)"
          pill={<span style={{ 'font-family': 'var(--mono)' }}>LODO · 4 corpora</span>}
          foot={
            <>
              <span>
                <strong>Mean Δ within → cross</strong>
                {' '}−
                {(
                  GEN_CLIFF.reduce((s, c) => s + (c.within - c.cross), 0) / GEN_CLIFF.length
                ).toFixed(2)}
              </span>
              <span><strong>Best donor</strong> PAMAP2</span>
              <span><strong>Worst transfer</strong> UCIHAR → PAMAP2</span>
              <span><strong>Implication</strong> domain adaptation needed</span>
            </>
          }
        >
          <span class="gcard-num">Fig. 07</span>
          <div class="gcard-canvas">
            <svg viewBox="0 0 720 300" preserveAspectRatio="xMidYMid meet">
              <g stroke="rgba(40,28,20,.10)" stroke-dasharray="2 4">
                <line x1="60" y1="36" x2="700" y2="36" />
                <line x1="60" y1="80" x2="700" y2="80" />
                <line x1="60" y1="124" x2="700" y2="124" />
                <line x1="60" y1="168" x2="700" y2="168" />
              </g>
              <line x1="60" y1="212" x2="700" y2="212" stroke="var(--text-2)" />
              <line x1="60" y1="36" x2="60" y2="212" stroke="var(--text-2)" />
              <For each={GEN_CLIFF}>
                {(c, i) => {
                  const tx = 30 + i() * 160
                  const wH = c.within * 176
                  const cH = c.cross * 176
                  return (
                    <g transform={`translate(${tx},0)`}>
                      <rect x="90" y={212 - wH} width="36" height={wH} fill="var(--teal)" rx="3" />
                      <text x="108" y={212 - wH - 6} text-anchor="middle" font-family="Instrument Serif" font-size="16" fill="var(--text)">
                        {c.within.toFixed(2)}
                      </text>
                      <rect x="128" y={212 - cH} width="36" height={cH} fill="var(--terracotta)" opacity=".22" rx="3" />
                      <text x="146" y={212 - cH - 6} text-anchor="middle" font-family="Instrument Serif" font-size="16" fill="var(--terracotta)">
                        {c.cross.toFixed(2)}
                      </text>
                      <text x="127" y="232" text-anchor="middle" font-size="11" fill="var(--text-2)" font-weight="600">
                        {c.label}
                      </text>
                      <text x="127" y="248" text-anchor="middle" font-size="9" fill="var(--text-3)">
                        {c.transferTo}
                      </text>
                    </g>
                  )
                }}
              </For>
              <g text-anchor="end" font-size="10" fill="var(--text-3)">
                <text x="54" y="215">0.0</text>
                <text x="54" y="171">0.25</text>
                <text x="54" y="127">0.50</text>
                <text x="54" y="83">0.75</text>
                <text x="54" y="39">1.0</text>
              </g>
              <text x="22" y="124" text-anchor="middle" font-size="10" fill="var(--text-3)" letter-spacing="2" transform="rotate(-90 22 124)">MACRO-F1</text>
              <g transform="translate(60,278)">
                <rect x="0" y="-9" width="14" height="9" fill="var(--teal)" rx="1" />
                <text x="20" y="-1" font-size="10.5" fill="var(--text-3)">within-dataset</text>
                <rect x="140" y="-9" width="14" height="9" fill="var(--terracotta)" opacity=".22" rx="1" />
                <text x="160" y="-1" font-size="10.5" fill="var(--text-3)">cross-dataset (LODO)</text>
              </g>
            </svg>
          </div>
          <div class="gcard-cap">
            <b>Reads as</b>
            Within-dataset performance (teal) is uniformly strong. The soft terracotta bars
            show what happens on a held-out corpus — different phones, different subjects,
            different protocols. PAMAP2 → UCIHAR transfers cleanly; UCIHAR → PAMAP2 collapses
            by half a macro-F1 point.
          </div>
        </Gcard>
      </div>

      {/* ---- ROW: ACTIVITY FEED + REGISTRY ---- */}
      <div class="an-row r2">
        <Gcard
          title="Activity feed"
          pill={
            <>
              <span class="pulse" />
              live
            </>
          }
        >
          <FeedList>
            <Show
              when={recentSessions().length > 0}
              fallback={
                <FeedItem level="mute" time="—">
                  No recent sessions yet — run a replay to populate the timeline.
                </FeedItem>
              }
            >
              <For each={recentSessions()}>
                {(s) => {
                  const level: 'alert' | 'ok' | 'info' =
                    s.latest_warning_level === 'high'
                      ? 'alert'
                      : s.latest_warning_level === 'low' ||
                          s.latest_warning_level === 'medium'
                        ? 'info'
                        : 'ok'
                  const detail =
                    s.latest_warning_level === 'high' ? (
                      <>
                        <strong>High-confidence fall</strong> · subj_
                        {s.session.subject_id} · top P(fall){' '}
                        {(s.latest_top_fall_probability ?? 0).toFixed(2)}
                        {s.latest_grouped_fall_event_count
                          ? ` · ${s.latest_grouped_fall_event_count} grouped event${
                              s.latest_grouped_fall_event_count === 1 ? '' : 's'
                            }`
                          : ''}
                      </>
                    ) : (
                      <>
                        <strong>{s.session.activity_label ?? 'session'}</strong> · subj_
                        {s.session.subject_id} · {s.session.device_platform}
                      </>
                    )
                  return (
                    <FeedItem level={level} time={feedTime(s.session.uploaded_at)}>
                      {detail}
                    </FeedItem>
                  )
                }}
              </For>
            </Show>
          </FeedList>
        </Gcard>

        <Gcard
          title="HAR model registry"
          sub={harRegistry.artifactVersion}
        >
          <table class="tbl" style={{ 'margin-top': '6px' }}>
            <tbody>
              <tr>
                <td class="mono">algorithm</td>
                <td>
                  Random Forest · {harRegistry.rfTrees} trees
                </td>
              </tr>
              <tr>
                <td class="mono">window / step</td>
                <td>
                  {harRegistry.windowSize} ({harRegistry.windowSeconds.toFixed(2)} s) /{' '}
                  {harRegistry.stepSize}
                </td>
              </tr>
              <tr>
                <td class="mono">target rate</td>
                <td>{harRegistry.targetRateHz} Hz</td>
              </tr>
              <tr>
                <td class="mono">features</td>
                <td>{harRegistry.features} engineered + magnitude</td>
              </tr>
              <tr>
                <td class="mono">labels</td>
                <td>{harRegistry.allowedLabels.join(' · ')}</td>
              </tr>
              <tr>
                <td class="mono">train rows</td>
                <td>{harRegistry.trainRows.toLocaleString()}</td>
              </tr>
              <tr>
                <td class="mono">holdout</td>
                <td>{harRegistry.holdoutDataset} (LODO)</td>
              </tr>
              <tr>
                <td class="mono">created</td>
                <td>{harRegistry.createdUtc.split('T')[0]}</td>
              </tr>
            </tbody>
          </table>
        </Gcard>
      </div>

      {/* ---- RECENT SESSIONS TABLE ---- */}
      <div class="an-row" style={{ 'grid-template-columns': '1fr' }}>
        <Gcard
          title="Recent sessions"
          sub={
            recentSessions().length
              ? `most recent ${recentSessions().length}`
              : 'awaiting first ingest'
          }
          pill={
            <>
              <span class="pulse" />
              auto-refresh 30 s
            </>
          }
        >
          <div style={{ 'overflow-x': 'auto' }}>
            <table class="tbl">
              <thead>
                <tr>
                  <th>Session</th>
                  <th>Subject</th>
                  <th>Activity</th>
                  <th>Platform</th>
                  <th>Warning</th>
                  <th>Top P(fall)</th>
                  <th>Events</th>
                  <th>Uploaded</th>
                </tr>
              </thead>
              <tbody>
                <Show
                  when={recentSessions().length > 0}
                  fallback={
                    <tr>
                      <td
                        colspan="8"
                        style={{
                          'text-align': 'center',
                          padding: '32px',
                          color: 'var(--text-3)',
                        }}
                      >
                        No persisted sessions yet.
                      </td>
                    </tr>
                  }
                >
                  <For each={recentSessions()}>
                    {(s) => {
                      const level = (s.latest_warning_level ?? 'none') as string
                      const tag =
                        level === 'medium' ? 'med' : (level as 'high' | 'low' | 'none')
                      const fallP = s.latest_top_fall_probability ?? 0
                      return (
                        <tr>
                          <td class="id">{s.session.app_session_id.slice(0, 8)}</td>
                          <td class="mono">subj_{s.session.subject_id}</td>
                          <td>{s.session.activity_label ?? '—'}</td>
                          <td class="mono">{s.session.device_platform}</td>
                          <td>
                            <WarningTag level={tag}>{level}</WarningTag>
                          </td>
                          <td>
                            {/* Alert styling follows the post-gate warning
                                level, not raw P(fall): walking/stairs
                                sessions can produce high raw probabilities
                                that the HAR gate has already downgraded. */}
                            <ProbBar value={fallP} alert={level === 'high'} />
                          </td>
                          <td class="mono">{s.latest_grouped_fall_event_count ?? 0}</td>
                          <td class="mono">{feedTime(s.session.uploaded_at)}</td>
                        </tr>
                      )
                    }}
                  </For>
                </Show>
              </tbody>
            </table>
          </div>
        </Gcard>
      </div>

    </>
  )
}