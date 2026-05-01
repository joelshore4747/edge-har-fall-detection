import { For, createSignal, createMemo } from 'solid-js'
import { Eyebrow, Gcard, Section } from '../../../components/v6'
import { harCrossDataset } from '../../../lib/realMetrics'

type Slice = {
  key: string
  label: string
  blurb: string
  rows: ReadonlyArray<{ metric: string; value: number; color?: string }>
}

const SLICES: Slice[] = [
  {
    key: 'within_uci',
    label: 'Within UCI HAR',
    blurb:
      "Subject-grouped split inside UCI HAR. The strongest within-dataset baseline — the model is essentially learning the dataset.",
    rows: [
      { metric: 'macro-F1', value: harCrossDataset.withinUcihar.macroF1, color: 'var(--moss)' },
      { metric: 'accuracy', value: harCrossDataset.withinUcihar.accuracy, color: 'var(--moss)' },
      { metric: 'static · F1', value: harCrossDataset.perClassUcihar.static.f1 },
      { metric: 'locomotion · F1', value: harCrossDataset.perClassUcihar.locomotion.f1 },
      { metric: 'stairs · F1', value: harCrossDataset.perClassUcihar.stairs.f1 },
    ],
  },
  {
    key: 'within_pamap',
    label: 'Within PAMAP2',
    blurb:
      "Subject-grouped split inside PAMAP2 with the full four-class label space. 'Other' and 'stairs' are the hard classes.",
    rows: [
      { metric: 'macro-F1', value: harCrossDataset.withinPamap2.macroF1, color: 'var(--ochre)' },
      { metric: 'accuracy', value: harCrossDataset.withinPamap2.accuracy, color: 'var(--ochre)' },
      { metric: 'static · F1', value: harCrossDataset.perClassPamap2.static.f1 },
      { metric: 'locomotion · F1', value: harCrossDataset.perClassPamap2.locomotion.f1 },
      { metric: 'stairs · F1', value: harCrossDataset.perClassPamap2.stairs.f1 },
      { metric: 'other · F1', value: harCrossDataset.perClassPamap2.other.f1 },
    ],
  },
  {
    key: 'uci_to_pamap',
    label: 'UCIHAR → PAMAP2',
    blurb:
      'Train on UCI HAR, test on PAMAP2 — the harder of the two transfers. PAMAP2 is body-worn, UCI HAR is waist-mounted; the gait band shifts.',
    rows: [
      { metric: 'macro-F1', value: harCrossDataset.ucihar_to_pamap2.macroF1, color: 'var(--terracotta)' },
      { metric: 'accuracy', value: harCrossDataset.ucihar_to_pamap2.accuracy, color: 'var(--terracotta)' },
    ],
  },
  {
    key: 'pamap_to_uci',
    label: 'PAMAP2 → UCIHAR',
    blurb:
      'Train on PAMAP2, test on UCI HAR. Easier transfer — PAMAP2 is the higher-rate, more-sensor dataset, so the source distribution is broader.',
    rows: [
      { metric: 'macro-F1', value: harCrossDataset.pamap2_to_ucihar.macroF1, color: 'var(--teal)' },
      { metric: 'accuracy', value: harCrossDataset.pamap2_to_ucihar.accuracy, color: 'var(--teal)' },
    ],
  },
]

export function ExplorePage() {
  const [activeKey, setActiveKey] = createSignal<string>('pamap_to_uci')
  const slice = createMemo(() => SLICES.find((s) => s.key === activeKey())!)

  return (
    <>
      <Eyebrow>Explore · cohort slices</Eyebrow>

      <h1
        style={{
          'font-family': 'var(--serif)',
          'font-size': 'clamp(48px, 6vw, 80px)',
          'letter-spacing': '-.03em',
          'line-height': 0.95,
          'margin-bottom': '18px',
        }}
      >
        Pick a slice.<br />
        See where the <em style={{ 'font-style': 'italic', color: 'var(--terracotta)' }}>model breaks.</em>
      </h1>

      <div class="prose">
        <p>
          The same harmonised feature space is evaluated under four regimes — two within-dataset
          subject splits and two cross-dataset transfers. Click a slice to read its per-class break-down;
          the steep drop on cross-dataset transfer is the project's headline limitation.
        </p>
      </div>

      <Section num="01" title="Choose a" emphasis="slice">
        <div style={{ display: 'flex', 'flex-wrap': 'wrap', gap: '8px', 'margin-bottom': '18px' }}>
          <For each={SLICES}>
            {(s) => (
              <button
                class={`world-btn${activeKey() === s.key ? ' active is-editorial' : ''}`}
                onClick={() => setActiveKey(s.key)}
              >
                <span class="dot" />
                {s.label}
              </button>
            )}
          </For>
        </div>

        <Gcard title={slice().label} sub="random forest · shared labels">
          <div class="prose" style={{ 'margin-bottom': '18px' }}>
            <p>{slice().blurb}</p>
          </div>
          <div class="ds-rows">
            <For each={slice().rows}>
              {(r) => (
                <div class="ds-row">
                  <span class="lbl">{r.metric}</span>
                  <div class="ds-bar">
                    <span style={{ width: `${r.value * 100}%`, background: r.color ?? 'var(--teal)' }} />
                  </div>
                  <span class="v">{r.value.toFixed(3)}</span>
                </div>
              )}
            </For>
          </div>
        </Gcard>
      </Section>

      <Section num="02" title="The" emphasis="cliff">
        <div class="prose">
          <p>
            The drop from <strong>{harCrossDataset.withinUcihar.macroF1.toFixed(2)}</strong> macro-F1
            (within UCI HAR) to <strong>{harCrossDataset.ucihar_to_pamap2.macroF1.toFixed(2)}</strong>{' '}
            (UCIHAR → PAMAP2) is roughly{' '}
            {(
              ((harCrossDataset.withinUcihar.macroF1 - harCrossDataset.ucihar_to_pamap2.macroF1) /
                harCrossDataset.withinUcihar.macroF1) *
              100
            ).toFixed(0)}
            % relative — and it's a known, well-documented effect in the HAR literature. The PAMAP2 →
            UCIHAR direction is gentler ({harCrossDataset.pamap2_to_ucihar.macroF1.toFixed(2)})
            because PAMAP2 is the higher-rate, more-diverse source distribution.
          </p>
        </div>
      </Section>
    </>
  )
}
