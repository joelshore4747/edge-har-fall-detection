import { For } from 'solid-js'
import { Eyebrow, Section } from '../../../components/v6'
import { fallProduction, harRegistry } from '../../../lib/realMetrics'

const METHODS = [
  {
    num: '01',
    title: 'Harmonisation',
    body: 'Each source dataset is resampled to 50 Hz, aligned on a common 6-channel layout (ax, ay, az, gx, gy, gz), and tagged with a unified label drawn from {static, locomotion, stairs, other}. The harmonisation step is the precondition for cross-dataset evaluation.',
    code: 'pipeline/harmonise.py',
  },
  {
    num: '02',
    title: 'Windowing',
    body: `Each harmonised stream is sliced into ${harRegistry.windowSize}-sample windows (${harRegistry.windowSeconds.toFixed(2)} s @ ${harRegistry.targetRateHz} Hz) with a ${harRegistry.stepSize}-sample stride — i.e. 50% overlap. Windows that span large gaps or fail acceptance checks are dropped before feature extraction.`,
    code: 'pipeline/window.py',
  },
  {
    num: '03',
    title: 'Feature extraction',
    body: `The HAR head consumes ${harRegistry.features} engineered features per window: per-axis magnitude/IQR/std/RMS plus accel/gyro magnitude band-power on three frequency bands and dominant-frequency descriptors. Features are cheap on-device.`,
    code: 'pipeline/features',
  },
  {
    num: '04',
    title: 'HAR head training',
    body: `A ${harRegistry.algorithm} (${harRegistry.rfTrees} trees, class-balanced) is fitted on the LODO-${harRegistry.holdoutDataset} train union (${harRegistry.trainDatasets.join(' + ')}, ${harRegistry.trainRows.toLocaleString()} windows). Subject-grouped splits are used wherever subject IDs are available.`,
    code: 'pipeline/har/train.py',
  },
  {
    num: '05',
    title: 'Fall head: vulnerability features',
    body: 'The fall head does not see raw IMU. It sees 21 engineered fall-specific features — peak / post-impact dynamics, jerk, gyro peak, and three stage-pass flags from the threshold pipeline. The XGB classifier learns the right weighting from the (MobiFall + SisFall) labelled set.',
    code: 'pipeline/fall/train.py',
  },
  {
    num: '06',
    title: 'Fall head deployment',
    body: `Promoted artifact: ${fallProduction.artifactId} (${fallProduction.modelKind}). Operating threshold τ = ${fallProduction.probabilityThreshold} chosen on the validation set; held-out F1 ${fallProduction.heldout.f1.toFixed(2)}, AUC ${fallProduction.heldout.rocAuc.toFixed(2)}. The artifact lives at /artifacts/fall/current/model.joblib and is loaded once per process.`,
    code: 'apps/api/runtime_inference.py',
  },
] as const

export function DocsPage() {
  return (
    <>
      <Eyebrow>Methods · how the pipeline trains</Eyebrow>

      <h1
        style={{
          'font-family': 'var(--serif)',
          'font-size': 'clamp(56px, 7vw, 96px)',
          'letter-spacing': '-.03em',
          'line-height': 0.95,
          'margin-bottom': '24px',
        }}
      >
        Six steps from raw IMU<br />
        to a <em style={{ 'font-style': 'italic', color: 'var(--terracotta)' }}>shipped artifact.</em>
      </h1>

      <div class="prose">
        <p>
          The training pipeline is intentionally linear and reproducible: each stage writes a
          checkpointed artifact under <code>artifacts/</code> and a JSON metric report under{' '}
          <code>results/</code>. Runs are pinned by hash and the registry only promotes when both
          validation and held-out metrics improve.
        </p>
      </div>

      <Section num="01" title="Pipeline" emphasis="stages">
        <div class="eds-list">
          <For each={METHODS}>
            {(m) => (
              <div class="eds-row">
                <span class="eds-num">M—{m.num}</span>
                <div>
                  <div class="eds-title">{m.title}</div>
                  <p>{m.body}</p>
                  <p>
                    <code>{m.code}</code>
                  </p>
                </div>
                <span class="eds-arrow">→</span>
              </div>
            )}
          </For>
        </div>
      </Section>

      <Section num="02" title="Reproducing" emphasis="the runs">
        <div class="prose">
          <p>
            Every result on this site can be regenerated locally from the raw datasets in{' '}
            <code>data/raw/</code>. The end-to-end recipe is:
          </p>
        </div>
        <div class="fig" style={{ '--accent-c': 'var(--teal)', 'margin-top': '18px' }}>
          <pre
            style={{
              'font-family': 'var(--mono)',
              'font-size': '12.5px',
              'line-height': 1.7,
              color: 'var(--text-2)',
              'white-space': 'pre-wrap',
              margin: 0,
            }}
          >
            {`# 1. Harmonise the five public corpora to 50 Hz / 6 channels
python -m pipeline.harmonise --target-rate 50 --window 128 --step 64

# 2. Train the HAR head (LODO-PAMAP2 by default)
python scripts/train_unifallmonitor_har.py \\
    --train UCIHAR --train WISDM --holdout PAMAP2 \\
    --rf-trees 400 --balanced

# 3. Train the production fall head
python -m pipeline.fall.train_xgb \\
    --datasets MOBIFALL SISFALL --threshold 0.45

# 4. Run the cross-dataset evaluator
python -m eval.cross_dataset \\
    --output results/validation/har_cross_dataset_eval.json

# 5. Compare threshold-only vs vulnerability-aware
python -m eval.primary_comparison \\
    --output results/reports/primary_comparison.json
`}
          </pre>
        </div>
        <div class="sci-cap">
          <strong>Note</strong>The dataset paths and per-subject splits are pinned in{' '}
          <code>configs/datasets.yaml</code>; the random_state is 42 throughout.
        </div>
      </Section>
    </>
  )
}
