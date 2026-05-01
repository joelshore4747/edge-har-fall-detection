import { For } from 'solid-js'
import { Eyebrow, Kpi, Kpis, Section } from '../../../components/v6'
import {
  fallMetaCombined,
  fallProduction,
  harCrossDataset,
  vulnerabilityComparison,
} from '../../../lib/realMetrics'

const FALL_TABLE = [
  {
    head: 'Production XGB · validation',
    rows: [
      ['F1', fallProduction.validation.f1.toFixed(3)],
      ['F2', fallProduction.validation.f2.toFixed(3)],
      ['Recall', fallProduction.validation.recall.toFixed(3)],
      ['Specificity', fallProduction.validation.specificity.toFixed(3)],
      ['Precision', fallProduction.validation.precision.toFixed(3)],
      ['ROC-AUC', fallProduction.validation.rocAuc.toFixed(3)],
    ],
  },
  {
    head: 'Production XGB · held-out',
    rows: [
      ['F1', fallProduction.heldout.f1.toFixed(3)],
      ['F2', fallProduction.heldout.f2.toFixed(3)],
      ['Recall', fallProduction.heldout.recall.toFixed(3)],
      ['Specificity', fallProduction.heldout.specificity.toFixed(3)],
      ['Precision', fallProduction.heldout.precision.toFixed(3)],
      ['ROC-AUC', fallProduction.heldout.rocAuc.toFixed(3)],
    ],
  },
  {
    head: 'Combined meta · held-out',
    rows: [
      ['F1', fallMetaCombined.metrics.f1.toFixed(3)],
      ['Sensitivity', fallMetaCombined.metrics.sensitivity.toFixed(3)],
      ['Specificity', fallMetaCombined.metrics.specificity.toFixed(3)],
      ['Precision', fallMetaCombined.metrics.precision.toFixed(3)],
      ['ROC-AUC', fallMetaCombined.metrics.rocAuc.toFixed(3)],
      ['Brier', fallMetaCombined.metrics.brierScore.toFixed(3)],
    ],
  },
] as const

export function ResultsPage() {
  return (
    <>
      <Eyebrow>Results · evaluation evidence</Eyebrow>

      <h1
        style={{
          'font-family': 'var(--serif)',
          'font-size': 'clamp(56px, 7vw, 96px)',
          'letter-spacing': '-.03em',
          'line-height': 0.95,
          'margin-bottom': '24px',
        }}
      >
        What the <em style={{ 'font-style': 'italic', color: 'var(--terracotta)' }}>numbers say.</em>
      </h1>

      <div class="prose">
        <p>
          Three pieces of evidence, each from a different evaluation regime: the production XGB fall
          head measured on its training validation split and on a strict held-out set, the combined
          logistic meta-model on the largest held-out subject group, and the relative-gain comparison
          between the threshold-only baseline and the vulnerability-aware fusion.
        </p>
      </div>

      <Section num="01" title="Fall head" emphasis="evaluation" kicker="three regimes">
        <Kpis>
          <Kpi
            label="ROC-AUC · heldout"
            value={fallProduction.heldout.rocAuc.toFixed(3)}
            foot={<><span class="delta">production XGB</span> · 19 subjects</>}
          />
          <Kpi
            label="F1 · heldout"
            value={fallProduction.heldout.f1.toFixed(3)}
            foot={<>τ = {fallProduction.probabilityThreshold} · {fallProduction.modelKind.toUpperCase()}</>}
          />
          <Kpi
            label="Sensitivity"
            value={fallProduction.heldout.recall.toFixed(3)}
            foot={<>{fallProduction.heldout.falseNegatives.toLocaleString()} false negatives</>}
          />
          <Kpi
            label="Specificity"
            value={fallProduction.heldout.specificity.toFixed(3)}
            foot={<>{fallProduction.heldout.falsePositives.toLocaleString()} false positives</>}
          />
        </Kpis>

        <div class="an-row r3" style={{ 'margin-top': '24px' }}>
          <For each={FALL_TABLE}>
            {(t) => (
              <div class="gcard">
                <div class="gcard-h">
                  <h4>{t.head}</h4>
                </div>
                <table class="tbl">
                  <tbody>
                    <For each={t.rows}>
                      {([label, value]) => (
                        <tr>
                          <td class="mono">{label}</td>
                          <td class="mono" style={{ 'text-align': 'right' }}>
                            {value}
                          </td>
                        </tr>
                      )}
                    </For>
                  </tbody>
                </table>
              </div>
            )}
          </For>
        </div>
      </Section>

      <Section num="02" title="Vulnerability-aware" emphasis="lift" kicker="threshold → meta">
        <div class="prose">
          <p>
            The headline contribution of the dissertation is the F1 lift you get when the
            engineered post-impact features are fused into a calibrated probability — instead of
            relying on a hand-tuned threshold over peak acceleration alone. MobiFall and SisFall are
            the two purpose-built fall datasets in the corpus; the table below is computed directly
            from{' '}
            <code>results/reports/primary_comparison.json</code>.
          </p>
        </div>

        <table class="dataset-table" style={{ 'margin-top': '18px' }}>
          <thead>
            <tr>
              <th>Dataset</th>
              <th>Threshold F1</th>
              <th>Meta F1</th>
              <th>Δ F1</th>
              <th>Δ %</th>
              <th>Sensitivity (meta)</th>
              <th>Specificity (meta)</th>
            </tr>
          </thead>
          <tbody>
            <For each={vulnerabilityComparison}>
              {(v) => (
                <tr>
                  <td>{v.dataset}</td>
                  <td class="mono">{v.thresholdF1.toFixed(3)}</td>
                  <td class="mono" style={{ color: 'var(--terracotta)', 'font-weight': 600 }}>
                    {v.vulnerabilityF1.toFixed(3)}
                  </td>
                  <td class="mono">+{v.absoluteGain.toFixed(3)}</td>
                  <td class="mono" style={{ color: 'var(--terracotta)' }}>
                    +{v.relativeGainPct.toFixed(1)}%
                  </td>
                  <td class="mono">{v.vulnerabilitySensitivity.toFixed(3)}</td>
                  <td class="mono">{v.vulnerabilitySpecificity.toFixed(3)}</td>
                </tr>
              )}
            </For>
          </tbody>
        </table>
      </Section>

      <Section num="03" title="HAR cross-dataset" emphasis="transfer" kicker="random forest baseline">
        <div class="prose">
          <p>
            The HAR head is a random forest trained on harmonised band-power features. Within-dataset
            scores are strong; the interesting result is the cliff under leave-one-dataset-out
            evaluation, where the same model loses 25–40 macro-F1 points on cross-dataset transfer.
            This is the literature's known generalisation gap, reproduced here on the harmonised
            feature space.
          </p>
        </div>

        <div class="an-row r2-eq" style={{ 'margin-top': '24px' }}>
          <div class="gcard">
            <div class="gcard-h">
              <h4>Within-dataset</h4>
              <span class="stl">subject-independent split</span>
            </div>
            <table class="tbl">
              <tbody>
                <tr>
                  <td>UCI HAR · macro-F1</td>
                  <td class="mono">{harCrossDataset.withinUcihar.macroF1.toFixed(3)}</td>
                </tr>
                <tr>
                  <td>UCI HAR · accuracy</td>
                  <td class="mono">{harCrossDataset.withinUcihar.accuracy.toFixed(3)}</td>
                </tr>
                <tr>
                  <td>PAMAP2 · macro-F1</td>
                  <td class="mono">{harCrossDataset.withinPamap2.macroF1.toFixed(3)}</td>
                </tr>
                <tr>
                  <td>PAMAP2 · accuracy</td>
                  <td class="mono">{harCrossDataset.withinPamap2.accuracy.toFixed(3)}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div class="gcard">
            <div class="gcard-h">
              <h4>Cross-dataset</h4>
              <span class="stl">shared labels only</span>
            </div>
            <table class="tbl">
              <tbody>
                <tr>
                  <td>UCIHAR → PAMAP2</td>
                  <td class="mono">{harCrossDataset.ucihar_to_pamap2.macroF1.toFixed(3)}</td>
                </tr>
                <tr>
                  <td>PAMAP2 → UCIHAR</td>
                  <td class="mono">{harCrossDataset.pamap2_to_ucihar.macroF1.toFixed(3)}</td>
                </tr>
                <tr>
                  <td>Mean macro-F1</td>
                  <td class="mono" style={{ 'font-weight': 600 }}>
                    {harCrossDataset.meanMacroF1.toFixed(3)}
                  </td>
                </tr>
                <tr>
                  <td>Δ within → cross</td>
                  <td class="mono" style={{ color: 'var(--terracotta)' }}>
                    −
                    {(
                      harCrossDataset.withinUcihar.macroF1 -
                      harCrossDataset.ucihar_to_pamap2.macroF1
                    ).toFixed(2)}{' '}
                    F1
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </Section>

      <Section
        num="04"
        title="Top RF feature"
        emphasis="importances"
        kicker="UCIHAR → PAMAP2"
      >
        <div class="prose">
          <p>
            Bandpower and inter-quartile-range features dominate. This is consistent with the
            physical intuition that locomotion / static / stairs separate cleanly on energy in the
            1–3&nbsp;Hz band — the gait fundamental.
          </p>
        </div>
        <table class="dataset-table" style={{ 'margin-top': '18px' }}>
          <thead>
            <tr>
              <th>Feature</th>
              <th>Importance</th>
            </tr>
          </thead>
          <tbody>
            <For each={harCrossDataset.topFeatures}>
              {(f) => (
                <tr>
                  <td class="mono" style={{ 'font-family': 'var(--mono)', 'font-size': '13px' }}>
                    {f.feature}
                  </td>
                  <td class="mono">{f.importance.toFixed(4)}</td>
                </tr>
              )}
            </For>
          </tbody>
        </table>
      </Section>
    </>
  )
}
