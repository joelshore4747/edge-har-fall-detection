import { DiagramFigure } from './DiagramFigure'

export function D07FallFusion() {
  return (
    <DiagramFigure
      num="07"
      title="Fall fusion"
      emphasis="decision graph"
      accent="var(--terracotta)"
      src="/figures/D07_fall_fusion.svg"
      alt="Fall event fusion and vulnerability scoring decision graph"
      caption={
        <>
          <strong>Figure 07.</strong> Two-stage fall fusion. Stage 1 produces a calibrated fall
          probability (XGBoost) and an activity label (Random Forest) per IMU window. Stage 2
          combines the meta-probability with interpretable threshold evidence into a 4-state{' '}
          <code>FallEventState</code> machine (0.20 / 0.50 / 0.75 cuts). Stage 3 fuses the event
          signal with HAR posture, inactivity, impact, and heart-rate evidence into a 3-level{' '}
          <code>VulnerabilityLevel</code> alert ladder (0.32 / 0.60 cuts). All weights and
          thresholds are pulled from <code>fusion/fall_event.py</code> and{' '}
          <code>fusion/vulnerability_score.py</code>.
        </>
      }
    />
  )
}
