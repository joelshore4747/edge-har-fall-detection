import { For } from 'solid-js'
import { Eyebrow, Gcard, Section } from '../../../components/v6'
import {
  datasets,
  fallMetaCombined,
  fallProduction,
  harRegistry,
} from '../../../lib/realMetrics'

const ARTIFACTS = [
  {
    name: 'fall_xgb_candidate',
    kind: 'XGBoost · binary',
    path: 'artifacts/fall/current/model.joblib',
    promoted: fallProduction.promotedUtc,
    metric: `AUC ${fallProduction.heldout.rocAuc.toFixed(3)} · F1 ${fallProduction.heldout.f1.toFixed(2)}`,
    role: 'production fall head',
  },
  {
    name: 'fall_meta_combined',
    kind: 'Logistic regression · 21 feats',
    path: 'artifacts/fall/fall_meta_combined/',
    promoted: 'staging',
    metric: `F1 ${fallMetaCombined.metrics.f1.toFixed(3)} · AUC ${fallMetaCombined.metrics.rocAuc.toFixed(3)}`,
    role: 'reference baseline',
  },
  {
    name: harRegistry.artifactVersion,
    kind: `${harRegistry.algorithm} · ${harRegistry.rfTrees} trees`,
    path: 'artifacts/har/current/',
    promoted: harRegistry.createdUtc,
    metric: `${harRegistry.trainRows.toLocaleString()} windows · ${harRegistry.features} features`,
    role: 'production HAR head',
  },
]

const REPORTS = [
  {
    name: 'primary_comparison.json',
    path: 'results/reports/primary_comparison.json',
    purpose: 'Threshold vs vulnerability-aware F1 on MobiFall + SisFall held-outs.',
  },
  {
    name: 'har_cross_dataset_eval.json',
    path: 'results/validation/har_cross_dataset_eval.json',
    purpose: 'Within- and cross-dataset macro-F1 between UCI HAR and PAMAP2 on shared labels.',
  },
  {
    name: 'fall_artifact_eval_xgb.json',
    path: 'results/validation/fall/candidates/xgb.json',
    purpose: 'Held-out evaluation of the production XGB fall head on subject-grouped splits.',
  },
  {
    name: 'evaluation_contract.md',
    path: 'results/reports/evaluation_contract.md',
    purpose: 'Written contract for what every evaluation run must record (split, metrics, hash).',
  },
]

export function LibraryPage() {
  return (
    <>
      <Eyebrow>Library · artifacts and reports</Eyebrow>

      <h1
        style={{
          'font-family': 'var(--serif)',
          'font-size': 'clamp(48px, 6vw, 80px)',
          'letter-spacing': '-.03em',
          'line-height': 0.95,
          'margin-bottom': '18px',
        }}
      >
        Every model and every report,<br />
        <em style={{ 'font-style': 'italic', color: 'var(--terracotta)' }}>pinned by hash.</em>
      </h1>

      <div class="prose">
        <p>
          The artifact registry is the canonical store for trained models; the report directory is
          the canonical store for the JSON metric files those models produce. This page lists what
          is currently promoted to <code>current/</code>, alongside the supporting reports.
        </p>
      </div>

      <Section num="01" title="Promoted" emphasis="artifacts">
        <Gcard title="Model registry" sub="artifacts/{har,fall}/current/">
          <table class="tbl">
            <thead>
              <tr>
                <th>Artifact</th>
                <th>Kind</th>
                <th>Path</th>
                <th>Headline metric</th>
                <th>Role</th>
              </tr>
            </thead>
            <tbody>
              <For each={ARTIFACTS}>
                {(a) => (
                  <tr>
                    <td class="id">{a.name}</td>
                    <td>{a.kind}</td>
                    <td class="mono">{a.path}</td>
                    <td class="mono">{a.metric}</td>
                    <td>{a.role}</td>
                  </tr>
                )}
              </For>
            </tbody>
          </table>
        </Gcard>
      </Section>

      <Section num="02" title="Validation" emphasis="reports">
        <Gcard title="JSON reports" sub="results/">
          <table class="tbl">
            <thead>
              <tr>
                <th>File</th>
                <th>Path</th>
                <th>Purpose</th>
              </tr>
            </thead>
            <tbody>
              <For each={REPORTS}>
                {(r) => (
                  <tr>
                    <td class="id">{r.name}</td>
                    <td class="mono">{r.path}</td>
                    <td>{r.purpose}</td>
                  </tr>
                )}
              </For>
            </tbody>
          </table>
        </Gcard>
      </Section>

      <Section num="03" title="Datasets" emphasis="harmonised">
        <table class="dataset-table">
          <thead>
            <tr>
              <th>Dataset</th>
              <th>Subjects</th>
              <th>Activities / Falls</th>
              <th>Sensors</th>
              <th>Sampling</th>
              <th>Windows</th>
            </tr>
          </thead>
          <tbody>
            <For each={datasets}>
              {(d) => (
                <tr>
                  <td>{d.name}</td>
                  <td class="mono">{d.subjects}</td>
                  <td>{d.activities}</td>
                  <td class="mono">{d.sensors}</td>
                  <td class="mono">{d.sampling}</td>
                  <td class="mono">{d.windows ? d.windows.toLocaleString() : '—'}</td>
                </tr>
              )}
            </For>
          </tbody>
        </table>
      </Section>
    </>
  )
}
