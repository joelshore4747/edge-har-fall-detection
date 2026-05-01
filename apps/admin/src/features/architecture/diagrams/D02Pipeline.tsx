import { DiagramFigure } from './DiagramFigure'

export function D02Pipeline() {
  return (
    <DiagramFigure
      num="02"
      title="ML training"
      emphasis="pipeline"
      accent="var(--terracotta)"
      src="/figures/D02_ml_training_pipeline.svg"
      alt="Directed acyclic graph of the UniFall ML training pipeline"
      caption={
        <>
          <strong>Figure 02.</strong> Offline training pipeline for the deployed HAR (Random
          Forest) and Fall (XGBoost) heads. Each branch ends in a checkpointed artifact whose
          generalisation metric is shown next to the checkpoint marker — HAR reports cross-dataset
          macro-F1, Fall reports held-out ROC-AUC and F1. Both artifacts feed a single in-process
          runtime scorer.
        </>
      }
    />
  )
}
