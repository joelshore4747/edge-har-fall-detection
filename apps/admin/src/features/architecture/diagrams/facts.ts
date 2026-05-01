import { fallProduction, harCrossDataset, harRegistry } from '../../../lib/realMetrics'

export const architectureFacts = {
  har: {
    artifactId: harRegistry.artifactVersion,
    legacyLodoArtifactId: harRegistry.artifactVersion,
    modelKind: 'Random Forest',
    trees: harRegistry.rfTrees,
    features: harRegistry.features,
    targetRateHz: harRegistry.targetRateHz,
    windowSize: harRegistry.windowSize,
    stepSize: harRegistry.stepSize,
    windowSeconds: harRegistry.windowSeconds,
    trainDatasets: harRegistry.trainDatasets.join(' + '),
    lodoTrainDatasets: harRegistry.trainDatasets.join(' + '),
    lodoHoldout: harRegistry.holdoutDataset,
    labels: harRegistry.allowedLabels.join(' · '),
    // Symmetric metrics so D02 can show HAR alongside Fall AUC/F1.
    macroF1Within: harCrossDataset.withinUcihar.macroF1,
    accuracyWithin: harCrossDataset.withinUcihar.accuracy,
    macroF1Cross: harCrossDataset.pamap2_to_ucihar.macroF1,
    accuracyCross: harCrossDataset.pamap2_to_ucihar.accuracy,
  },
  fall: {
    artifactId: fallProduction.artifactId,
    modelKind: 'XGBoost',
    features: 21,
    probabilityThreshold: fallProduction.probabilityThreshold,
    heldoutAuc: fallProduction.heldout.rocAuc,
    heldoutF1: fallProduction.heldout.f1,
    recall: fallProduction.heldout.recall,
    specificity: fallProduction.heldout.specificity,
    calibrator: 'isotonic',
    ece: 0.027,
  },
  // Canonical thresholds and weights from fusion/fall_event.py and
  // fusion/vulnerability_score.py. Pinned here so D07 stays in lockstep with
  // the algorithm definition rather than drifting visually.
  fallEvent: {
    weights: {
      meta: 0.5,
      impact: 0.22,
      motion: 0.12,
      variance: 0.08,
      confirm: 0.08,
      predictedBonus: 0.05,
      recoveryPenalty: 0.3,
    },
    thresholds: {
      probable: 0.75,
      possible: 0.5,
      impactOnly: 0.2,
    },
  },
  vulnerability: {
    weights: {
      fall: 0.42,
      eventConfidence: 0.18,
      inactivity: 0.18,
      posture: 0.1,
      impact: 0.08,
      hr: 0.04,
      recoveryPenalty: 0.22,
    },
    thresholds: {
      high: 0.6,
      medium: 0.32,
    },
    stateBonus: {
      impactOnly: 0.04,
      possible: 0.1,
      probable: 0.18,
    },
  },
  db: {
    tableCount: 11,
    coreTables:
      'app_users · app_sessions · app_session_inferences · app_grouped_fall_events · app_timeline_events',
  },
  latency: {
    medianMs: 78,
    p95Ms: 92,
    p99Ms: 178,
    budgetMs: 250,
  },
} as const
