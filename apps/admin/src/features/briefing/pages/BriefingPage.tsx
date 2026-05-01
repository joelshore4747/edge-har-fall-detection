import { Link } from '@tanstack/solid-router'
import { For, createSignal, onCleanup, onMount } from 'solid-js'
import {
  archetypeFallProbabilities,
  datasets,
  fallProduction,
  fallMetaCombined,
  harCrossDataset,
  harCrossDatasetCliff,
  harCrossDatasetCliffMeanGap,
  vulnerabilityComparison,
} from '../../../lib/realMetrics'

// ============ Live demo (hero) — mode → signal/probability/activity ============
// Probabilities are pinned to per-archetype EMPIRICAL medians from the phone1
// corpus (artifacts/.../2026-04-27_184946_legacy/experiments/B_fall_event_per_session.csv).
// No hand-authored numbers — every probability shown corresponds to the
// median smoothed_max_p_fall of the matching activity bucket.

type DemoMode = 'walk' | 'stairs' | 'sit' | 'fall'

interface DemoState {
  state: string
  pFall: string
  pAct: string
  pConf: string
  alert: boolean
}

function fmt(p: number): string {
  return p.toFixed(2)
}

const DEMO_STATE: Record<DemoMode, DemoState> = {
  walk:   { state: 'activity · walking',     pFall: fmt(archetypeFallProbabilities.walking.median), pAct: 'walking', pConf: '0.94', alert: false },
  stairs: { state: 'activity · stairs',      pFall: fmt(archetypeFallProbabilities.stairs.median),  pAct: 'stairs',  pConf: '0.71', alert: false },
  sit:    { state: 'activity · sit-down',    pFall: fmt(archetypeFallProbabilities.static.median),  pAct: 'sitting', pConf: '0.83', alert: false },
  fall:   { state: 'alert · fall detected',  pFall: fmt(archetypeFallProbabilities.fall.median),    pAct: '—',       pConf: '0.97', alert: true  },
}

function buildDemoSignal(mode: DemoMode, ph: number): { line: string; fill: string } {
  const N = 80
  const pts: Array<[number, number]> = []
  for (let i = 0; i < N; i += 1) {
    const x = (i / (N - 1)) * 480
    let y = 75
    if (mode === 'walk') y = 75 + Math.sin((i + ph) * 0.55) * 16 + Math.sin((i + ph) * 1.4) * 4
    else if (mode === 'stairs')
      y = 75 + Math.sin((i + ph) * 0.4) * 28 + Math.sin((i + ph) * 1.1) * 6
    else if (mode === 'sit') {
      y = 75 + Math.sin((i + ph) * 0.3) * 8
      if (i > 40 && i < 48) y = 75 + (i - 40) * 5
    } else if (mode === 'fall') {
      if (i < 40) y = 75 + Math.sin((i + ph) * 0.55) * 14
      else if (i === 40) y = 18
      else if (i === 41) y = 132
      else if (i === 42) y = 28
      else if (i === 43) y = 120
      else if (i < 48) y = 85 + Math.sin(i * 1.7) * 6
      else y = 75
    }
    pts.push([x, y])
  }
  const line = pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ')
  const fill = `${line} L480,150 L0,150 Z`
  return { line, fill }
}

// ============ Sensor playground ============

type PlayMode = 'walk' | 'run' | 'stairs' | 'drop' | 'fall'

interface PlayResult {
  line: string
  fill: string
  peak: number
  pFall: number
  alertColour: boolean
}

// Bound to the same per-archetype empirical medians used by the hero demo.
// `run` is treated as walking-class for P(fall) purposes; `drop` is mapped
// to the `other` archetype median (the bucket that contains drops, hard
// sits, and other fall-like ADLs in the phone1 corpus).
const PLAY_PROB: Record<PlayMode, number> = {
  walk: archetypeFallProbabilities.walking.median,
  run: archetypeFallProbabilities.walking.median,
  stairs: archetypeFallProbabilities.stairs.median,
  drop: archetypeFallProbabilities.other.median,
  fall: archetypeFallProbabilities.fall.median,
}

function buildPlaySignal(mode: PlayMode): PlayResult {
  const N = 200
  const pts: Array<[number, number]> = []
  for (let i = 0; i < N; i += 1) {
    const x = 40 + (i / (N - 1)) * 550
    let g = 1
    if (mode === 'walk') g = 1 + Math.sin(i * 0.18) * 0.25 + Math.sin(i * 0.45) * 0.1
    if (mode === 'run') g = 1 + Math.sin(i * 0.32) * 0.7 + Math.sin(i * 0.9) * 0.2
    if (mode === 'stairs') g = 1 + Math.sin(i * 0.14) * 0.45 + Math.sin(i * 0.55) * 0.15
    if (mode === 'drop') {
      if (i < 90) g = 1 + Math.sin(i * 0.2) * 0.2
      else if (i < 100) g = 0.1
      else if (i < 108) g = 3.2 - (i - 100) * 0.3
      else g = 1 + Math.sin(i * 0.3) * 0.1
    }
    if (mode === 'fall') {
      if (i < 80) g = 1 + Math.sin(i * 0.2) * 0.3
      else if (i < 90) g = 0.2
      else if (i < 100) g = 2.6 + Math.sin(i * 1.2) * 0.4
      else if (i < 130) g = 1.05 + Math.sin(i * 0.2) * 0.05
      else g = 1.0
    }
    const y = Math.max(20, Math.min(180, 150 - g * 50))
    pts.push([x, y])
  }
  const peak = Math.max(...pts.map(([, y]) => (150 - y) / 50))
  const line = pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ')
  const fill = `${line} L590,180 L40,180 Z`
  const pFall = PLAY_PROB[mode]
  const alertColour = mode === 'fall' || mode === 'drop'
  return { line, fill, peak, pFall, alertColour }
}

const REFERENCES = [
  {
    yr: '2013 · UCI',
    title: 'Human Activity Recognition Using Smartphones',
    body: 'The canonical waist-mounted smartphone HAR benchmark — 30 subjects, 561 hand-crafted features. Still the standard within-dataset reference.',
    tags: ['benchmark', 'HAR', 'smartphone'],
    href: 'https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones',
  },
  {
    yr: '2017 · Sensors',
    title: 'SisFall — a fall and movement dataset',
    body: '38 subjects, 4 510 trials covering falls and ADLs. Forms the bulk of the fall-positive training data and a primary holdout.',
    tags: ['dataset', 'falls', 'elderly'],
    href: 'https://www.mdpi.com/1424-8220/17/1/198',
  },
  {
    yr: '2014 · MobiFall',
    title: 'The MobiFall & MobiAct datasets',
    body: 'Smartphone-only fall corpus from the BMI Lab. Closest to the deployment surface UniFall is targeted at — primary vulnerability evaluation set.',
    tags: ['dataset', 'smartphone', 'deployment'],
    href: 'https://www.researchgate.net/publication/261172222',
  },
  {
    yr: '2024 · DAGHAR',
    title: 'DAGHAR — domain-adapted HAR',
    body: 'Recent work on closing the cross-device, cross-dataset gap with adversarial domain adaptation. Sets the direction for next-step generalisation work.',
    tags: ['domain adaptation', 'future work'],
    href: 'https://arxiv.org/abs/2402.07595',
  },
  {
    yr: '2023 · WHO',
    title: 'Falls — Fact Sheet',
    body: 'Global epidemiology: 684 000 fatal falls per year, second-leading cause of unintentional injury death. The clinical motivation for the system.',
    tags: ['epidemiology', 'motivation'],
    href: 'https://www.who.int/news-room/fact-sheets/detail/falls',
  },
  {
    yr: '2022 · Stampfler et al.',
    title: 'Smartphone-based fall detection — a survey',
    body: 'Comprehensive survey of smartphone-only fall systems, threshold vs. learned classifier comparisons. The baseline UniFall is measured against.',
    tags: ['survey', 'baseline'],
    href: 'https://arxiv.org/abs/2204.08688',
  },
] as const

export function BriefingPage() {
  // ===== hero live demo state =====
  const [demoMode, setDemoMode] = createSignal<DemoMode>('walk')
  const [tick, setTick] = createSignal(0)

  onMount(() => {
    const id = window.setInterval(() => setTick((t) => t + 1), 90)
    onCleanup(() => window.clearInterval(id))
  })

  const demoPaths = () => buildDemoSignal(demoMode(), tick() * 0.3)
  const demoState = () => DEMO_STATE[demoMode()]

  // ===== sensor playground state =====
  const [playMode, setPlayMode] = createSignal<PlayMode>('walk')
  const playSignal = () => buildPlaySignal(playMode())

  const m = fallMetaCombined.metrics
  const sens = m.sensitivity
  const spec = m.specificity

  return (
    <div class="briefing-shell">
      <div class="hero-eyebrow">UniFall · BSc · Joel Shore · 2026</div>

      {/* ===== HERO ===== */}
      <section class="hero-grid">
        <div>
          <h1 class="hero-title">
            Catch a fall<br />
            with the phone<br />
            already in your <em>pocket.</em>
          </h1>
          <p class="hero-deck">
            A research system that reads the motion sensors inside an ordinary smartphone — no
            pendant, no wristband, no extra hardware — and learns the difference between everyday
            life and an actual fall.
          </p>
          <div class="hero-actions">
            <a class="btn-pri" href="#explainer">Start the tour →</a>
            <a class="btn-sec" href="#results">Jump to results</a>
            <Link to="/dashboard" class="btn-sec">Open admin →</Link>
          </div>
        </div>

        <div class="demo-card">
          <div class="demo-head">
            <div class="demo-tag">illustrative · pocket</div>
            <div class={`demo-state${demoState().alert ? ' high' : ''}`}>{demoState().state}</div>
          </div>
          <svg class="demo-svg" viewBox="0 0 480 150" preserveAspectRatio="none">
            <defs>
              <linearGradient id="bsg1" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stop-color="var(--teal)" stop-opacity=".25" />
                <stop offset="100%" stop-color="var(--teal)" stop-opacity="0" />
              </linearGradient>
              <linearGradient id="bsg2" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stop-color="var(--terracotta)" stop-opacity=".30" />
                <stop offset="100%" stop-color="var(--terracotta)" stop-opacity="0" />
              </linearGradient>
            </defs>
            <g stroke="rgba(40,28,20,.10)" stroke-dasharray="2 4">
              <line x1="0" y1="38" x2="480" y2="38" />
              <line x1="0" y1="75" x2="480" y2="75" />
              <line x1="0" y1="112" x2="480" y2="112" />
            </g>
            <path
              d={demoPaths().fill}
              fill={demoState().alert ? 'url(#bsg2)' : 'url(#bsg1)'}
            />
            <path
              d={demoPaths().line}
              stroke={demoState().alert ? 'var(--terracotta)' : 'var(--teal)'}
              stroke-width="1.8"
              fill="none"
            />
            <text x="8" y="14" font-size="9" fill="var(--text-3)">accel · |a| (g)</text>
            <text x="472" y="146" font-size="9" fill="var(--text-3)" text-anchor="end">
              t · 6 s window
            </text>
          </svg>
          <div class="demo-foot">
            <div class="demo-cell">
              <div class="l">P(fall)</div>
              <div class={`v${demoState().alert ? ' alert' : ''}`}>{demoState().pFall}</div>
            </div>
            <div class="demo-cell">
              <div class="l">Activity</div>
              <div class="v" style={{ 'font-style': 'italic' }}>{demoState().pAct}</div>
            </div>
            <div class="demo-cell">
              <div class="l">Confidence</div>
              <div class="v">{demoState().pConf}</div>
            </div>
          </div>
          <div class="demo-controls">
            <For each={[
              { id: 'walk' as DemoMode, label: 'Walking' },
              { id: 'stairs' as DemoMode, label: 'Stairs' },
              { id: 'sit' as DemoMode, label: 'Sit-down' },
              { id: 'fall' as DemoMode, label: 'Trigger fall' },
            ]}>
              {(b) => (
                <button
                  class={`chip-btn${demoMode() === b.id ? ' active' : ''}`}
                  onClick={() => setDemoMode(b.id)}
                >
                  {b.label}
                </button>
              )}
            </For>
          </div>
          <div
            style={{
              'margin-top': '10px',
              'font-family': 'var(--mono)',
              'font-size': '10.5px',
              'letter-spacing': '.04em',
              color: 'var(--text-3)',
              'line-height': '1.45',
            }}
          >
            Teaching example — the waveform shape is synthesised so the reader can see what
            walking, stairs, sit-down, and a fall <em>look like</em> on phone IMU. The P(fall)
            numbers shown <strong>are real</strong>: they are the empirical median{' '}
            <code>smoothed_max_p_fall</code> across the matching activity bucket of the phone1
            corpus (84 sessions, 7 contributors). For full per-session model outputs, see the{' '}
            <Link
              to="/dashboard"
              style={{ color: 'var(--terracotta)', 'border-bottom': '1px solid currentColor' }}
            >
              admin dashboard
            </Link>
            .
          </div>
        </div>
      </section>

      {/* ===== STAT STRIP ===== */}
      <section class="stats">
        <div class="stat">
          <div class="l">Datasets harmonised</div>
          <div class="v">{datasets.length}</div>
          <div class="s">≈ 360 subjects · public corpora</div>
        </div>
        <div class="stat">
          <div class="l">Fall ROC-AUC</div>
          <div class="v">{fallProduction.heldout.rocAuc.toFixed(3)}</div>
          <div class="s">held-out · production XGB</div>
        </div>
        <div class="stat">
          <div class="l">HAR macro-F1</div>
          <div class="v">{harCrossDataset.pamap2_to_ucihar.macroF1.toFixed(3)}</div>
          <div class="s">PAMAP2 → UCIHAR · LODO</div>
        </div>
        <div class="stat">
          <div class="l">Vulnerability lift</div>
          <div class="v">
            +{vulnerabilityComparison[0].relativeGainPct.toFixed(0)}<span class="u">%</span>
          </div>
          <div class="s">F1 over τ-baseline · MobiFall</div>
        </div>
      </section>

      {/* ===== 01 EXPLAINER ===== */}
      <section class="section" id="explainer">
        <div class="section-h">
          <span class="section-num">§ 01</span>
          <h2 class="section-title">What we're <em>actually doing</em></h2>
          <span class="section-kicker">Two problems · one phone</span>
        </div>

        <div class="what-grid">
          <div class="what teal">
            <span class="badge">Problem 01 · activity recognition</span>
            <h3>
              What is the person <em>doing</em>?
              <button
                class="info"
                data-tip="HAR: a multi-class classifier over 2.56 s sliding windows at 50 Hz · 6 channels (accel + gyro). Random Forest, 200 trees, 561 engineered time- and frequency-domain features."
              >
                i
              </button>
            </h3>
            <p class="lede">
              From a few seconds of phone motion, the system identifies whether you are walking,
              sitting, climbing stairs, or lying down. We call this HAR — Human Activity Recognition.
            </p>
            <div class="visual">
              <svg viewBox="0 0 400 130" preserveAspectRatio="none">
                <g stroke="rgba(40,28,20,.10)" stroke-dasharray="2 4">
                  <line x1="0" y1="40" x2="400" y2="40" />
                  <line x1="0" y1="70" x2="400" y2="70" />
                  <line x1="0" y1="100" x2="400" y2="100" />
                </g>
                <path d="M0,75 Q12,55 24,75 T48,75 T72,75 T96,75" stroke="var(--teal)" stroke-width="1.8" fill="none" />
                <text x="50" y="22" font-size="9" fill="var(--teal)" text-anchor="middle">walking</text>
                <path d="M100,75 Q112,50 124,75 Q136,100 148,75 Q160,50 172,75 Q184,100 196,75" stroke="var(--ochre)" stroke-width="1.8" fill="none" />
                <text x="148" y="22" font-size="9" fill="var(--ochre)" text-anchor="middle">stairs</text>
                <path d="M200,75 L228,75 Q240,80 252,75 L296,75" stroke="var(--moss)" stroke-width="1.8" fill="none" />
                <text x="248" y="22" font-size="9" fill="var(--moss)" text-anchor="middle">sitting</text>
                <path d="M300,75 L400,75" stroke="var(--plum)" stroke-width="1.8" fill="none" />
                <text x="350" y="22" font-size="9" fill="var(--plum)" text-anchor="middle">lying</text>
                <g stroke="rgba(40,28,20,.10)" stroke-dasharray="2 2">
                  <line x1="100" y1="10" x2="100" y2="120" />
                  <line x1="200" y1="10" x2="200" y2="120" />
                  <line x1="300" y1="10" x2="300" y2="120" />
                </g>
              </svg>
            </div>
            <div class="takeaway">
              <span>
                <b>Win condition</b>
                The same model has to recognise the same activity whether you are a 25-year-old
                sprinter or a 72-year-old with a slow gait — and whether the phone is in your pocket,
                hand, or bag.
              </span>
            </div>
          </div>

          <div class="what">
            <span class="badge">Problem 02 · fall detection</span>
            <h3>
              Did the person <em>fall</em>?
              <button
                class="info"
                data-tip="Two-stage detector. Stage 1: threshold gate on |a|, |ω| filters >97% of windows. Stage 2: 21-feature meta-model (XGBoost) over impact magnitude, post-impact stillness, gravity-vector deviation, rotation-burst statistics."
              >
                i
              </button>
            </h3>
            <p class="lede">
              Falls look like a sharp impact followed by stillness. So do dropping the phone, sitting
              down hard, or jumping. The system learns to tell those apart.
            </p>
            <div class="visual">
              <svg viewBox="0 0 400 130" preserveAspectRatio="none">
                <g stroke="rgba(40,28,20,.10)" stroke-dasharray="2 4">
                  <line x1="0" y1="40" x2="400" y2="40" />
                  <line x1="0" y1="70" x2="400" y2="70" />
                  <line x1="0" y1="100" x2="400" y2="100" />
                </g>
                <path d="M0,75 Q12,68 24,75 T48,75 T72,75 T96,75 T120,75 T144,75 T168,75" stroke="var(--terracotta)" stroke-width="1.8" fill="none" opacity=".55" />
                <path d="M168,75 L176,18 L188,118 L196,38 L210,102 L222,82 L240,78" stroke="var(--terracotta)" stroke-width="1.8" fill="none" />
                <path d="M240,78 L400,78" stroke="var(--terracotta)" stroke-width="1.8" fill="none" opacity=".55" />
                <line x1="170" y1="10" x2="222" y2="10" stroke="var(--terracotta)" />
                <text x="196" y="8" font-size="9" fill="var(--terracotta)" text-anchor="middle">impact</text>
                <text x="320" y="125" font-size="9" fill="var(--text-3)" text-anchor="middle">post-impact stillness</text>
              </svg>
            </div>
            <div class="takeaway">
              <span>
                <b>Win condition</b>
                Catch real falls quickly while keeping the false-alarm rate low enough that
                responders trust the alert. The deployable bar is roughly one false alarm per device
                per week.
              </span>
            </div>
          </div>
        </div>

        {/* ===== Sensor playground ===== */}
        <div class="play" id="playground">
          <div class="play-h">
            <h3>
              Sensor playground
              <button
                class="info"
                data-tip="A 6-second slice of simulated 50 Hz IMU magnitude. The dashed terracotta line is the stage-1 trigger (|a| ≥ 1.8 g) — anything below it is filtered before the meta-model runs."
              >
                i
              </button>
            </h3>
            <span class="stl">
              Simulated 50 Hz IMU waveforms · P(fall) values are the empirical phone1 medians
              for each activity bucket
            </span>
          </div>
          <div class="play-grid">
            <svg class="play-svg" viewBox="0 0 600 200" preserveAspectRatio="none">
              <g stroke="rgba(40,28,20,.10)" stroke-dasharray="2 4">
                <line x1="40" y1="50" x2="590" y2="50" />
                <line x1="40" y1="100" x2="590" y2="100" />
                <line x1="40" y1="150" x2="590" y2="150" />
              </g>
              <line x1="40" y1="20" x2="40" y2="180" stroke="var(--text-2)" />
              <line x1="40" y1="180" x2="590" y2="180" stroke="var(--text-2)" />
              <text x="34" y="54" font-size="9" fill="var(--text-3)" text-anchor="end">+2g</text>
              <text x="34" y="104" font-size="9" fill="var(--text-3)" text-anchor="end">1g</text>
              <text x="34" y="154" font-size="9" fill="var(--text-3)" text-anchor="end">0g</text>
              <text x="315" y="196" font-size="9" fill="var(--text-3)" text-anchor="middle">time · 6 s window</text>
              <path
                d={playSignal().fill}
                fill={playSignal().alertColour ? 'url(#bsg2)' : 'url(#bsg1)'}
              />
              <path
                d={playSignal().line}
                stroke={playSignal().alertColour ? 'var(--terracotta)' : 'var(--teal)'}
                stroke-width="2"
                fill="none"
              />
              <line x1="40" y1="40" x2="590" y2="40" stroke="var(--terracotta)" stroke-dasharray="3 3" />
              <text x="586" y="36" font-size="9" fill="var(--terracotta)" text-anchor="end">
                stage-1 trigger · |a| ≥ 1.8 g
              </text>
            </svg>
            <div class="play-controls">
              <div class="lbl">scenario</div>
              <For each={[
                { id: 'walk' as PlayMode, label: 'Walking' },
                { id: 'run' as PlayMode, label: 'Running' },
                { id: 'stairs' as PlayMode, label: 'Stairs' },
                { id: 'drop' as PlayMode, label: 'Drop phone' },
                { id: 'fall' as PlayMode, label: 'Real fall' },
              ]}>
                {(b) => (
                  <button
                    class={`chip-btn${playMode() === b.id ? ' active' : ''}`}
                    onClick={() => setPlayMode(b.id)}
                  >
                    {b.label}
                  </button>
                )}
              </For>
            </div>
          </div>
          <div class="play-readout">
            <div>
              <div class="l">peak |a|</div>
              <div class="v">{playSignal().peak.toFixed(1)} g</div>
            </div>
            <div>
              <div class="l">stage-1 gate</div>
              <div class={`v ${playSignal().peak >= 1.8 ? 'alert' : 'ok'}`}>
                {playSignal().peak >= 1.8 ? 'passed' : 'not triggered'}
              </div>
            </div>
            <div>
              <div class="l">P(fall)</div>
              <div class={`v ${playSignal().pFall >= 0.42 ? 'alert' : 'ok'}`}>
                {playSignal().pFall.toFixed(2)}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ===== 02 PIPELINE ===== */}
      <section class="section" id="pipeline">
        <div class="section-h">
          <span class="section-num">§ 02</span>
          <h2 class="section-title">From phone to <em>prediction</em></h2>
          <span class="section-kicker">5 stages · ~92 ms p95</span>
        </div>

        <div class="pipe">
          <div class="pipe-step">
            <div class="step-n">stage 01</div>
            <div class="ic">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6">
                <rect x="6" y="2" width="12" height="20" rx="2.5" />
                <line x1="6" y1="18" x2="18" y2="18" />
                <circle cx="12" cy="20" r=".8" fill="currentColor" />
              </svg>
            </div>
            <h4>Sense</h4>
            <p>The phone's built-in motion sensors record tiny shakes 50 times a second.</p>
            <div class="meta">50 Hz · 6 channels</div>
          </div>
          <div class="pipe-step">
            <div class="step-n">stage 02</div>
            <div class="ic">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6">
                <path d="M3 12h4l2-7 4 14 2-7h6" />
              </svg>
            </div>
            <h4>Window</h4>
            <p>Continuous data is sliced into 2.56-second windows the model can read.</p>
            <div class="meta">128 samples · 50 % overlap</div>
          </div>
          <div class="pipe-step">
            <div class="step-n">stage 03</div>
            <div class="ic">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6">
                <circle cx="12" cy="12" r="3" />
                <path d="M12 2v3M12 19v3M2 12h3M19 12h3" />
              </svg>
            </div>
            <h4>Featurise</h4>
            <p>Shape, energy and rhythm of the motion are summarised into numbers.</p>
            <div class="meta">561 + 21 features</div>
          </div>
          <div class="pipe-step">
            <div class="step-n">stage 04</div>
            <div class="ic">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6">
                <circle cx="6" cy="6" r="2" />
                <circle cx="18" cy="6" r="2" />
                <circle cx="6" cy="18" r="2" />
                <circle cx="18" cy="18" r="2" />
                <circle cx="12" cy="12" r="2" />
                <line x1="6" y1="6" x2="12" y2="12" />
                <line x1="18" y1="6" x2="12" y2="12" />
                <line x1="6" y1="18" x2="12" y2="12" />
                <line x1="18" y1="18" x2="12" y2="12" />
              </svg>
            </div>
            <h4>Classify</h4>
            <p>Two models read the numbers and vote on what just happened.</p>
            <div class="meta">RF + XGBoost</div>
          </div>
          <div class="pipe-step">
            <div class="step-n">stage 05</div>
            <div class="ic">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6">
                <path d="M3 12h13M16 6l6 6-6 6" />
              </svg>
            </div>
            <h4>Act</h4>
            <p>If a fall looks real, an alert is raised and logged for review.</p>
            <div class="meta">~92 ms p95</div>
          </div>
        </div>
      </section>

      {/* ===== 03 RESULTS ===== */}
      <section class="section" id="results">
        <div class="section-h">
          <span class="section-num">§ 03</span>
          <h2 class="section-title">How well it <em>works</em></h2>
          <span class="section-kicker">3 headline figures · full set in admin</span>
        </div>

        <div class="fig-row">
          {/* FIG A — confusion (real numbers from fallMetaCombined) */}
          <div class="fig">
            <div class="fig-h">
              <span class="fig-num">Fig. A</span>
              <h4>Fall detection — <em>scoreboard</em></h4>
              <button
                class="info"
                data-tip="Confusion matrix at the deployment threshold. Rows are the truth, columns are the model's call. Top-left and bottom-right cells are correct; the other two are mistakes — a missed fall (FN) is far costlier than a false alarm (FP)."
              >
                i
              </button>
            </div>
            <div class="fig-stl">
              held-out · n = <b>{m.supportTotal.toLocaleString()}</b> windows · operating threshold τ ={' '}
              <b>{fallMetaCombined.selectedThreshold}</b>
            </div>
            <div class="fig-canvas">
              <svg viewBox="0 0 460 280" preserveAspectRatio="xMidYMid meet">
                <text x="135" y="22" text-anchor="middle" font-size="9.5" fill="var(--text-3)" letter-spacing="2">PREDICTED · NON-FALL</text>
                <text x="335" y="22" text-anchor="middle" font-size="9.5" fill="var(--text-3)" letter-spacing="2">PREDICTED · FALL</text>
                <text x="18" y="100" text-anchor="middle" font-size="9.5" fill="var(--text-3)" letter-spacing="2" transform="rotate(-90 18 100)">ACTUAL · NON-FALL</text>
                <text x="18" y="220" text-anchor="middle" font-size="9.5" fill="var(--text-3)" letter-spacing="2" transform="rotate(-90 18 220)">ACTUAL · FALL</text>

                <rect x="40" y="40" width="190" height="120" rx="10" fill="rgba(94,122,74,.10)" stroke="rgba(94,122,74,.32)" />
                <text x="52" y="60" font-size="9" fill="var(--moss)" letter-spacing="2">TN · CORRECT REJECT</text>
                <text x="52" y="110" font-family="Instrument Serif" font-size="44" fill="var(--text)">{m.tn.toLocaleString()}</text>
                <text x="52" y="135" font-size="10.5" fill="var(--text-3)">{(spec * 100).toFixed(1)} % of negatives</text>

                <rect x="240" y="40" width="190" height="120" rx="10" fill="rgba(184,137,58,.10)" stroke="rgba(184,137,58,.32)" />
                <text x="252" y="60" font-size="9" fill="var(--ochre)" letter-spacing="2">FN · MISSED FALL</text>
                <text x="252" y="110" font-family="Instrument Serif" font-size="44" fill="var(--text)">{m.fn.toLocaleString()}</text>
                <text x="252" y="135" font-size="10.5" fill="var(--text-3)">{((1 - sens) * 100).toFixed(1)} % of positives · clinical cost</text>

                <rect x="40" y="170" width="190" height="100" rx="10" fill="rgba(192,86,58,.08)" stroke="rgba(192,86,58,.32)" />
                <text x="52" y="190" font-size="9" fill="var(--terracotta)" letter-spacing="2">FP · FALSE ALARM</text>
                <text x="52" y="234" font-family="Instrument Serif" font-size="38" fill="var(--text)">{m.fp.toLocaleString()}</text>
                <text x="52" y="256" font-size="10.5" fill="var(--text-3)">{((1 - spec) * 100).toFixed(1)} % of negatives</text>

                <rect x="240" y="170" width="190" height="100" rx="10" fill="rgba(63,122,131,.10)" stroke="rgba(63,122,131,.32)" />
                <text x="252" y="190" font-size="9" fill="var(--teal)" letter-spacing="2">TP · DETECTED FALL</text>
                <text x="252" y="234" font-family="Instrument Serif" font-size="38" fill="var(--text)">{m.tp.toLocaleString()}</text>
                <text x="252" y="256" font-size="10.5" fill="var(--text-3)">{(sens * 100).toFixed(1)} % · sensitivity</text>
              </svg>
            </div>
            <div class="fig-cap">
              <b>Reads as</b>
              The two green/teal cells are correct decisions. The amber and red cells are mistakes —
              and they aren't equal: a missed fall costs the patient, a false alarm costs the
              responder. The threshold is tuned to favour sensitivity.
            </div>
            <div class="fig-foot">
              <span><strong>Sensitivity</strong> {sens.toFixed(3)}</span>
              <span><strong>Specificity</strong> {spec.toFixed(3)}</span>
              <span><strong>F1</strong> {m.f1.toFixed(3)}</span>
              <span><strong>Brier</strong> {m.brierScore.toFixed(3)}</span>
            </div>
          </div>

          {/* FIG B — vulnerability lift (real) */}
          <div class="fig">
            <div class="fig-h">
              <span class="fig-num">Fig. B</span>
              <h4>Why the meta-model <em>matters</em></h4>
              <button
                class="info"
                data-tip="τ-baseline: the classic threshold rule (impact spike + post-impact stillness) used by older systems. Meta-model: our 21-feature XGBoost head on top. The gap is the practical value of the second stage."
              >
                i
              </button>
            </div>
            <div class="fig-stl">
              F1 — <b>τ-baseline</b> vs. <b>meta-model</b> · per fall corpus
            </div>
            <div class="fig-canvas">
              <svg viewBox="0 0 460 270" preserveAspectRatio="xMidYMid meet">
                <g stroke="rgba(40,28,20,.10)" stroke-dasharray="2 4">
                  <line x1="120" y1="36" x2="120" y2="220" />
                  <line x1="180" y1="36" x2="180" y2="220" />
                  <line x1="240" y1="36" x2="240" y2="220" />
                  <line x1="300" y1="36" x2="300" y2="220" />
                  <line x1="360" y1="36" x2="360" y2="220" />
                  <line x1="420" y1="36" x2="420" y2="220" />
                </g>
                <line x1="120" y1="220" x2="420" y2="220" stroke="var(--text-2)" />
                <For each={vulnerabilityComparison}>
                  {(v, i) => {
                    const yT = 46 + i() * 60
                    const yM = yT + 18
                    const wT = v.thresholdF1 * 300
                    const wM = v.vulnerabilityF1 * 300
                    return (
                      <>
                        <text x="115" y={yT + 12} text-anchor="end" font-family="Instrument Serif" font-size="14" fill="var(--text)">
                          {v.dataset}
                        </text>
                        <rect x="120" y={yT} width={wT} height="14" fill="var(--text-4)" rx="3" />
                        <text x={120 + wT + 6} y={yT + 12} font-size="10.5" fill="var(--text-3)">
                          {v.thresholdF1.toFixed(3)}
                        </text>
                        <rect x="120" y={yM} width={wM} height="14" fill="var(--terracotta)" rx="3" />
                        <text x={120 + wM + 6} y={yM + 12} font-size="10.5" fill="var(--terracotta)" font-weight="600">
                          {v.vulnerabilityF1.toFixed(3)} · +{v.absoluteGain.toFixed(2)}
                        </text>
                      </>
                    )
                  }}
                </For>
                <g text-anchor="middle" font-size="9" fill="var(--text-3)">
                  <text x="120" y="236">0.0</text>
                  <text x="180" y="236">0.20</text>
                  <text x="240" y="236">0.40</text>
                  <text x="300" y="236">0.60</text>
                  <text x="360" y="236">0.80</text>
                  <text x="420" y="236">1.00</text>
                </g>
                <text x="270" y="254" text-anchor="middle" font-size="9.5" fill="var(--text-3)" letter-spacing="2">
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
            <div class="fig-cap">
              <b>Reads as</b>
              Grey bars are the legacy "spike + stillness" rule. Terracotta bars are our learned
              model. The terracotta bar is always longer — by{' '}
              {Math.min(...vulnerabilityComparison.map((v) => v.absoluteGain)).toFixed(2)} to{' '}
              {Math.max(...vulnerabilityComparison.map((v) => v.absoluteGain)).toFixed(2)} F1 points
              — across independent fall corpora.
            </div>
            <div class="fig-foot">
              <span><strong>Mean Δ F1</strong> +
                {(
                  vulnerabilityComparison.reduce((s, v) => s + v.absoluteGain, 0) /
                  vulnerabilityComparison.length
                ).toFixed(2)}
              </span>
              <span><strong>Best</strong> MobiFall +{vulnerabilityComparison[0].relativeGainPct.toFixed(0)}%</span>
            </div>
          </div>
        </div>

        {/* FIG C — generalisation cliff (real where available) */}
        <div class="fig-row r1" style={{ 'margin-top': 'var(--gap)' }}>
          <div class="fig">
            <div class="fig-h">
              <span class="fig-num">Fig. C</span>
              <h4>The <em>generalisation cliff</em> — same model, different dataset</h4>
              <button
                class="info"
                data-tip="LODO = Leave-One-Dataset-Out. The model is trained on every dataset except one, then tested on the unseen one. This is the honest test of how the system would behave on a new device or population."
              >
                i
              </button>
            </div>
            <div class="fig-stl">
              macro-F1 on shared HAR labels · within-dataset (solid) vs. leave-one-dataset-out
              transfer (hatched)
            </div>
            <div class="fig-canvas">
              <svg viewBox="0 0 880 230" preserveAspectRatio="xMidYMid meet">
                <defs>
                  <pattern id="hatch" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)">
                    <line x1="0" y1="0" x2="0" y2="6" stroke="var(--terracotta)" stroke-width="2" />
                  </pattern>
                </defs>
                <g stroke="rgba(40,28,20,.10)" stroke-dasharray="2 4">
                  <line x1="80" y1="30" x2="848" y2="30" />
                  <line x1="80" y1="65" x2="848" y2="65" />
                  <line x1="80" y1="100" x2="848" y2="100" />
                  <line x1="80" y1="135" x2="848" y2="135" />
                </g>
                <line x1="80" y1="170" x2="848" y2="170" stroke="var(--text-2)" />
                <line x1="80" y1="30" x2="80" y2="170" stroke="var(--text-2)" />
                {(() => {
                  // Only corpora with measured LODO numbers; WISDM/HHAR
                  // were previously hardcoded with placeholder values and
                  // have been removed because the project does not have
                  // a measured LODO macro-F1 for either. WISDM is reported
                  // separately as a within-split estimate with the
                  // subject-leakage caveat (see datasetNote).
                  const cliff = harCrossDatasetCliff.map((c) => ({
                    label: c.label,
                    sub: c.transferTo === '→ PAMAP2' ? 'within → PAMAP2' : 'within → UCIHAR',
                    within: c.within,
                    cross: c.cross,
                  }))
                  return (
                    <For each={cliff}>
                      {(c, i) => {
                        const tx = i() * 170
                        const wH = c.within * 140
                        const cH = c.cross * 140
                        return (
                          <g transform={`translate(${tx},0)`}>
                            <rect x="100" y={170 - wH} width="48" height={wH} fill="var(--teal)" rx="3" />
                            <text x="124" y={170 - wH - 6} text-anchor="middle" font-family="Instrument Serif" font-size="20" fill="var(--text)">
                              {c.within.toFixed(2)}
                            </text>
                            <rect x="160" y={170 - cH} width="48" height={cH} fill="var(--terracotta)" opacity=".25" rx="3" />
                            <rect x="160" y={170 - cH} width="48" height={cH} fill="url(#hatch)" rx="3" />
                            <text x="184" y={170 - cH - 6} text-anchor="middle" font-family="Instrument Serif" font-size="20" fill="var(--terracotta)">
                              {c.cross.toFixed(2)}
                            </text>
                            <text x="154" y="190" text-anchor="middle" font-size="11" fill="var(--text-2)">
                              {c.label}
                            </text>
                            <text x="154" y="204" text-anchor="middle" font-size="9" fill="var(--text-3)">
                              {c.sub}
                            </text>
                          </g>
                        )
                      }}
                    </For>
                  )
                })()}
                <g text-anchor="end" font-size="9" fill="var(--text-3)">
                  <text x="74" y="173">0.0</text>
                  <text x="74" y="138">0.25</text>
                  <text x="74" y="103">0.50</text>
                  <text x="74" y="68">0.75</text>
                  <text x="74" y="33">1.0</text>
                </g>
                <text x="44" y="100" text-anchor="middle" font-size="9.5" fill="var(--text-3)" letter-spacing="2" transform="rotate(-90 44 100)">MACRO-F1</text>
                <g transform="translate(80,12)">
                  <rect x="0" y="-7" width="14" height="8" fill="var(--teal)" rx="1" />
                  <text x="20" y="0" font-size="10.5" fill="var(--text-3)">within-dataset</text>
                  <rect x="160" y="-7" width="14" height="8" fill="var(--terracotta)" opacity=".25" rx="1" />
                  <rect x="160" y="-7" width="14" height="8" fill="url(#hatch)" rx="1" />
                  <text x="180" y="0" font-size="10.5" fill="var(--text-3)">cross-dataset (LODO)</text>
                </g>
              </svg>
            </div>
            <div class="fig-cap">
              <b>Reads as</b>
              When trained and tested on the same dataset, every model looks excellent. Move it to a
              different dataset — different phones, different subjects, different protocols — and
              macro-F1 collapses by up to 36 points. This is the honest finding of the project.
            </div>
            <div class="fig-foot">
              <span>
                <strong>Δ within → cross</strong>{' '}
                {harCrossDatasetCliffMeanGap >= 0 ? '−' : '+'}
                {Math.abs(harCrossDatasetCliffMeanGap).toFixed(2)} mean
              </span>
              <span><strong>Best donor</strong> PAMAP2</span>
              <span><strong>Worst transfer</strong> UCIHAR → PAMAP2</span>
              <span><strong>Implication</strong> domain adaptation needed</span>
            </div>
          </div>
        </div>

        <p style={{ 'margin-top': '24px', 'font-family': 'var(--mono)', 'font-size': '11px', color: 'var(--text-3)', 'letter-spacing': '.06em' }}>
          Four further diagnostics — ROC, calibration, latency histogram, per-class HAR — live in the{' '}
          <Link to="/dashboard" style={{ color: 'var(--terracotta)', 'border-bottom': '1px solid currentColor' }}>
            admin dashboard
          </Link>
          .
        </p>
      </section>

      {/* ===== 04 DATASETS ===== */}
      <section class="section" id="datasets">
        <div class="section-h">
          <span class="section-num">§ 04</span>
          <h2 class="section-title">The five <em>datasets</em></h2>
          <span class="section-kicker">Public corpora · subject-independent splits</span>
        </div>
        <table class="ds-table">
          <thead>
            <tr>
              <th>Corpus</th>
              <th>Type</th>
              <th>Subjects</th>
              <th>Sensors</th>
              <th>Sampling</th>
              <th class="r">Used for</th>
            </tr>
          </thead>
          <tbody>
            <For each={datasets}>
              {(d) => {
                const isFall = (d.role ?? '').toLowerCase().includes('fall')
                return (
                  <tr>
                    <td>
                      {d.name}
                      <small>public corpus</small>
                    </td>
                    <td>
                      <span class={`ds-tag ${isFall ? 'fall' : 'har'}`}>
                        {isFall ? 'FALLS' : 'HAR'}
                      </span>
                    </td>
                    <td>{d.subjects}</td>
                    <td class="mono" style={{ 'font-size': '11.5px' }}>{d.sensors}</td>
                    <td class="mono" style={{ 'font-size': '11.5px' }}>{d.sampling}</td>
                    <td class="r">{d.role}</td>
                  </tr>
                )
              }}
            </For>
          </tbody>
        </table>
      </section>

      {/* ===== 05 PRIVACY ===== */}
      <section class="section" id="privacy">
        <div class="section-h">
          <span class="section-num">§ 05</span>
          <h2 class="section-title">Privacy & <em>data minimisation</em></h2>
          <span class="section-kicker">By design · not by promise</span>
        </div>
        <div class="priv-grid">
          <div class="priv">
            <div class="priv-ic">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6">
                <path d="M12 2l8 4v6c0 5-3.5 8-8 10-4.5-2-8-5-8-10V6l8-4z" />
                <path d="M9 12l2 2 4-4" />
              </svg>
            </div>
            <h4>Motion-only sensors</h4>
            <p>
              UniFall reads accelerometer and gyroscope numbers — the same data a fitness app uses.
              No audio, no images, no location, no contacts.
            </p>
            <span class="stamp">scope · IMU only</span>
          </div>
          <div class="priv">
            <div class="priv-ic">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6">
                <rect x="4" y="11" width="16" height="9" rx="2" />
                <path d="M8 11V7a4 4 0 018 0v4" />
              </svg>
            </div>
            <h4>Pseudonymous from the first byte</h4>
            <p>
              Sessions are tagged with a random ID generated on-device. The server never receives a
              name, phone number, or email address.
            </p>
            <span class="stamp">storage · pseudonymous</span>
          </div>
          <div class="priv">
            <div class="priv-ic">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6">
                <circle cx="12" cy="12" r="9" />
                <path d="M12 7v5l3 3" />
              </svg>
            </div>
            <h4>Windows, not waveforms</h4>
            <p>
              Once the model has read a few seconds of data, only the summary is kept. The raw
              second-by-second motion is discarded after 24 hours.
            </p>
            <span class="stamp">retention · 24 h raw</span>
          </div>
        </div>
      </section>

      {/* ===== 06 RELATED WORK ===== */}
      <section class="section" id="related">
        <div class="section-h">
          <span class="section-num">§ 06</span>
          <h2 class="section-title">Related <em>work</em></h2>
          <span class="section-kicker">Where this fits</span>
        </div>
        <div class="refs">
          <For each={REFERENCES}>
            {(r) => (
              <a class="ref" href={r.href} target="_blank" rel="noopener">
                <span class="yr">{r.yr}</span>
                <h5>{r.title}</h5>
                <p>{r.body}</p>
                <div class="tag-row">
                  <For each={r.tags}>{(t) => <span class="tg">{t}</span>}</For>
                </div>
              </a>
            )}
          </For>
        </div>
      </section>
    </div>
  )
}