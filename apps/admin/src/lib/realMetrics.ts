// Real numbers extracted from the project's results/ and artifacts/ directories.
// Single source of truth for the v6 admin reskin. Update when retraining shifts
// the underlying JSON files; values below are pinned to the hashes recorded in
// the metadata files at the time of writing.
//
// Sources:
//   artifacts/fall/current/metadata.json                 (production XGB fall head)
//   artifacts/fall/fall_meta_combined/metrics.json       (combined meta-model)
//   artifacts/fall/fall_meta_combined/run_summary.json
//   artifacts/har/metadata.json                          (HAR registry)
//   results/validation/har_cross_dataset_eval.json
//   results/reports/primary_comparison.json              (vulnerability gain)

export type FallMetricsBlock = {
  f1: number
  f2: number
  recall: number
  specificity: number
  precision: number
  rocAuc: number
}

// ---- Fall production model (XGB, current artifact) ----------------------
export const fallProduction = {
  artifactId: 'fall_xgb_candidate',
  modelKind: 'xgb' as const,
  promotedUtc: '2026-04-22T20:50:04Z',
  probabilityThreshold: 0.45,
  validation: {
    f1: 0.6530,
    f2: 0.7207,
    recall: 0.7743,
    specificity: 0.7180,
    precision: 0.5645,
    rocAuc: 0.8262,
  } satisfies FallMetricsBlock,
  heldout: {
    f1: 0.5920,
    f2: 0.6985,
    recall: 0.7936,
    specificity: 0.6524,
    precision: 0.4721,
    rocAuc: 0.8199,
    falsePositives: 6752,
    falseNegatives: 1570,
  },
}

// ---- Fall combined meta-model (logistic regression on engineered features)
export const fallMetaCombined = {
  selectedThreshold: 0.4,
  trainRows: 105_531,
  testRows: 29_134,
  trainSubjects: 43,
  testSubjects: 19,
  metrics: {
    tn: 7429,
    fp: 13_761,
    fn: 949,
    tp: 6995,
    accuracy: 0.4951,
    sensitivity: 0.8805,
    specificity: 0.3506,
    precision: 0.3370,
    f1: 0.4875,
    supportTotal: 29_134,
    supportPositive: 7944,
    supportNegative: 21_190,
    brierScore: 0.2182,
    rocAuc: 0.7109,
    averagePrecision: 0.4916,
  },
  features: 21,
}

// ---- Vulnerability-aware fall improvement (primary comparison) ----------
export const vulnerabilityComparison = [
  {
    dataset: 'MobiFall',
    thresholdF1: 0.276,
    vulnerabilityF1: 0.857,
    absoluteGain: 0.581,
    relativeGainPct: 210.5,
    thresholdSensitivity: 0.335,
    vulnerabilitySensitivity: 0.898,
    thresholdSpecificity: 0.709,
    vulnerabilitySpecificity: 0.947,
  },
  {
    dataset: 'SisFall',
    thresholdF1: 0.256,
    vulnerabilityF1: 0.551,
    absoluteGain: 0.295,
    relativeGainPct: 115.0,
    thresholdSensitivity: 0.250,
    vulnerabilitySensitivity: 0.813,
    thresholdSpecificity: 0.717,
    vulnerabilitySpecificity: 0.543,
  },
] as const

// ---- HAR registry (movement_v2 / pamap2_shared_lodo_rf_balanced) --------
export const harRegistry = {
  artifactVersion: 'pamap2_shared_lodo_rf_balanced',
  createdUtc: '2026-04-21T21:19:09Z',
  taskType: 'har',
  algorithm: 'random_forest',
  rfTrees: 400,
  targetRateHz: 50.0,
  windowSize: 128,
  stepSize: 64,
  windowSeconds: 2.56,
  trainRows: 31_733,
  trainDatasets: ['UCIHAR', 'WISDM'] as const,
  holdoutDataset: 'PAMAP2',
  features: 87,
  allowedLabels: ['static', 'locomotion', 'stairs', 'other'] as const,
  sharedLabels: ['static', 'locomotion', 'stairs'] as const,
}

// ---- HAR cross-dataset evaluation (UCI HAR ↔ PAMAP2) --------------------
// Random-forest baseline trained on shared-label subset, evaluated on
// the alternate dataset's shared-label subset.
export const harCrossDataset = {
  withinUcihar: { accuracy: 0.9677, macroF1: 0.9489, support: 3156 },
  withinPamap2: { accuracy: 0.7616, macroF1: 0.6833, support: 9418 },
  ucihar_to_pamap2: { accuracy: 0.5054, macroF1: 0.4516, support: 11_454 },
  pamap2_to_ucihar: { accuracy: 0.8070, macroF1: 0.7065, support: 10_299 },
  // Mean of within + cross macro-F1 (used as headline number).
  meanMacroF1: (0.9489 + 0.6833 + 0.4516 + 0.7065) / 4,
  perClassUcihar: {
    static: { precision: 0.998, recall: 1.0, f1: 0.999 },
    locomotion: { precision: 0.954, recall: 0.851, f1: 0.900 },
    stairs: { precision: 0.923, recall: 0.975, f1: 0.948 },
  },
  perClassPamap2: {
    static: { precision: 0.675, recall: 0.662, f1: 0.668 },
    locomotion: { precision: 0.861, recall: 0.825, f1: 0.842 },
    stairs: { precision: 0.536, recall: 0.364, f1: 0.433 },
    other: { precision: 0.765, recall: 0.815, f1: 0.789 },
  },
  // Top RF feature importances from the UCI→PAMAP2 transfer.
  topFeatures: [
    { feature: 'acc_magnitude_bandpower_1_3hz', importance: 0.0802 },
    { feature: 'acc_magnitude_iqr', importance: 0.0626 },
    { feature: 'acc_magnitude_std', importance: 0.0541 },
    { feature: 'acc_magnitude_cv', importance: 0.0521 },
    { feature: 'acc_magnitude_max', importance: 0.0479 },
    { feature: 'acc_magnitude_spectral_energy', importance: 0.0461 },
    { feature: 'ax_max', importance: 0.0460 },
  ],
}

// ---- Datasets summary table ---------------------------------------------
export const datasets = [
  {
    name: 'UCI HAR',
    subjects: 30,
    activities: '6 ADLs',
    sensors: 'accel + gyro',
    sampling: '50 Hz',
    role: 'HAR baseline',
    windows: 10_299,
  },
  {
    name: 'PAMAP2',
    subjects: 9,
    activities: '18 activities',
    sensors: '3× IMU + HR',
    sampling: '100 Hz',
    role: 'HAR cross-eval (LODO holdout)',
    windows: 22_217,
  },
  {
    name: 'WISDM',
    subjects: 36,
    activities: '18 activities',
    sensors: 'accel + gyro',
    sampling: '20 Hz',
    role: 'HAR cross-eval *',
    windows: 21_434,
  },
  {
    name: 'MobiFall',
    subjects: 24,
    activities: '4 falls + 9 ADLs',
    sensors: 'accel + gyro + ori',
    sampling: '≈87 Hz',
    role: 'Fall positives',
    windows: undefined as number | undefined,
  },
  {
    name: 'SisFall',
    subjects: 38,
    activities: '15 falls + 19 ADLs',
    sensors: '2× accel + gyro',
    sampling: '200 Hz',
    role: 'Fall positives',
    windows: undefined as number | undefined,
  },
] as const

export const datasetNote =
  '* WISDM lacks clean per-user IDs in the checked-in export, so its within-dataset result is a provided train/test split rather than a strict subject-independent split.'

// ---- Per-archetype empirical fall probabilities (phone-session corpus) -
// Median of `smoothed_max_p_fall` per archetype across 95 sessions.
// Source: artifacts/unifallmonitor/runs/2026-04-30_081837_8c6a3fb/experiments/B_fall_event_per_session.csv
// These are real numbers, not illustrative — used by the Briefing demo and
// the sensor playground so no chart in the admin shows hand-authored
// probabilities.
export const archetypeFallProbabilities = {
  walking: { median: 0.072, mean: 0.138, max: 0.454, n: 14 },
  stairs:  { median: 0.201, mean: 0.183, max: 0.368, n: 16 },
  static:  { median: 0.040, mean: 0.076, max: 0.362, n: 17 },
  fall:    { median: 0.698, mean: 0.651, max: 0.915, n: 39 },
  other:   { median: 0.180, mean: 0.261, max: 0.940, n: 9 },
} as const

// ---- Cross-dataset cliff entries (real numbers only) -------------------
// Each entry is sourced from realMetrics::harCrossDataset; nothing is
// hand-authored. WISDM and HHAR were previously hardcoded with placeholder
// numbers — they have been removed because the project does not have a
// measured LODO macro-F1 for either corpus. WISDM is reported separately
// in the briefing as a within-split estimate with the subject-leakage
// caveat documented in datasets[].
export const harCrossDatasetCliff = [
  {
    label: 'UCIHAR',
    transferTo: '→ PAMAP2',
    within: 0.9489,
    cross: 0.4516,
  },
  {
    label: 'PAMAP2',
    transferTo: '→ UCIHAR',
    within: 0.6833,
    cross: 0.7065,
  },
] as const

export const harCrossDatasetCliffMeanGap =
  harCrossDatasetCliff.reduce((s, c) => s + (c.within - c.cross), 0) /
  harCrossDatasetCliff.length

// ---- Headline numbers used in mastheads / KPIs --------------------------
export const headline = {
  totalSubjects: datasets.reduce((s, d) => s + d.subjects, 0),
  fallTestRows: fallMetaCombined.testRows,
  fallTrainRows: fallMetaCombined.trainRows,
  harWindows: harRegistry.trainRows,
  fallAuc: fallProduction.heldout.rocAuc,
  fallSensitivity: fallProduction.heldout.recall,
  fallSpecificity: fallProduction.heldout.specificity,
  fallF1Heldout: fallProduction.heldout.f1,
  fallThreshold: fallProduction.probabilityThreshold,
  harMacroF1Within: harCrossDataset.withinUcihar.macroF1,
  harMacroF1Cross: harCrossDataset.pamap2_to_ucihar.macroF1,
  vulnerabilityF1Mobifall: vulnerabilityComparison[0].vulnerabilityF1,
  vulnerabilityGainMobifall: vulnerabilityComparison[0].relativeGainPct,
}

// ---- Research-background card content (literature pillars) --------------
export const researchPillars = [
  {
    num: '01',
    accent: 'terracotta' as const,
    title: 'Smartphone-based fall detection',
    body: 'Researchers have explored smartphones as fall-detection devices because phones already contain accelerometers and gyroscopes. These systems aim to detect sudden fall-like motion and trigger alerts without specialist hardware.',
    ref: 'Stampfler et al. — smartphone accelerometer-based fall-detection apps',
    url: 'https://pmc.ncbi.nlm.nih.gov/articles/PMC9618891/',
    relates:
      'My system is built around exactly this premise — a live-capable smartphone runtime, not a dedicated wearable.',
  },
  {
    num: '02',
    accent: 'teal' as const,
    title: 'Deep learning on wearable sensors',
    body: 'Recent fall-detection research uses deep learning to distinguish real falls from fall-like ADLs — sitting quickly, lying down, jumping, or dropping the device. False positives remain the central challenge.',
    ref: 'Zhang et al. (2024) — deep-learning framework on accel + gyro',
    url: 'https://www.jmir.org/2024/1/e56750/',
    relates:
      'The vulnerability-aware fusion head is trained explicitly to suppress fall-like ADLs through engineered post-impact features.',
  },
  {
    num: '03',
    accent: 'plum' as const,
    title: 'Wearable sensor systems',
    body: 'Body-worn IMUs allow continuous monitoring while preserving more privacy than camera-based systems. Most studies use accelerometers and gyroscopes; some add magnetometers or heart-rate sensors.',
    ref: 'Li et al. (2025) — decade-long review of wearable fall detection',
    url: 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11991334/',
    relates:
      "UniFall reuses the smartphone's IMU as a wearable proxy — same channels, same physics, different form factor.",
  },
  {
    num: '04',
    accent: 'moss' as const,
    title: 'Smartphone-based HAR',
    body: 'HAR research uses motion signals from accelerometers and gyroscopes to classify walking, standing, sitting, stairs, running. HAR is the foundation that fall detection sits on top of.',
    ref: 'Dentamaro et al. (2024) — smartphone HAR practical survey',
    url: 'https://www.sciencedirect.com/science/article/pii/S0957417424000083',
    relates:
      "The four-class HAR target space (static / locomotion / stairs / other) is drawn directly from this literature.",
  },
  {
    num: '05',
    accent: 'ochre' as const,
    title: 'Random forests for cross-dataset HAR',
    body: 'Recent HAR research tests transformers because they capture long-range time-series patterns, but evidence shows that simpler tree-based models often beat them on small wearable datasets.',
    ref: 'Leite et al. (2024) — transformers vs classical for HAR',
    url: 'https://arxiv.org/abs/2410.13605',
    relates:
      'My HAR head is a 400-tree random forest trained on harmonised band-power features — chosen because it transferred better than transformers on the available data.',
  },
  {
    num: '06',
    accent: 'indigo' as const,
    title: 'Sensor fusion',
    body: 'Multiple sensors capture different parts of movement: acceleration, rotation, orientation. Fusing them tends to improve recognition accuracy — at the cost of complexity and battery.',
    ref: 'San Buenaventura et al. — lightweight activity recognition fusion',
    url: 'https://dl.ifip.org/db/conf/im/im2017-ws5-papele/203.pdf',
    relates:
      'UniFall uses 6-channel accel+gyro fusion; magnetometer was tested but excluded for cross-device portability.',
  },
  {
    num: '07',
    accent: 'terracotta' as const,
    title: 'Cross-dataset generalisation',
    body: 'A major weakness in HAR / fall research: models trained on one dataset often fail on another. Differences in placement, sampling rate, participants, and labelling protocol all reduce real-world reliability.',
    ref: 'Napoli et al. (2024) — DAGHAR domain adaptation in HAR',
    url: 'https://www.nature.com/articles/s41597-024-03951-4',
    relates:
      'This is precisely why the headline HAR result is the leave-one-dataset-out PAMAP2→UCIHAR transfer, not a within-dataset best case.',
  },
  {
    num: '08',
    accent: 'teal' as const,
    title: 'Real-time mobile deployment',
    body: 'Some research targets phones, wearables, and IoT devices where memory, battery, and compute are limited. A real system has to run efficiently in practice — not just on a workstation.',
    ref: 'Wang et al. (2024) — real-time smartphone fall detection',
    url: 'https://arxiv.org/pdf/2412.09980',
    relates:
      'The runtime path is FastAPI + scikit-learn joblib in a single container — the entire production fall head is 12 MB.',
  },
] as const

// ---- Editorial diagram index --------------------------------------------
export const diagramIndex = [
  {
    n: 1,
    title: 'Full system architecture',
    deck: 'Flutter client through Nginx, Docker Compose, FastAPI, PostgreSQL, and the SolidJS admin you are reading right now.',
  },
  {
    n: 2,
    title: 'ML training pipeline',
    deck: 'From raw IMU through harmonisation and windowing to HAR + Fall checkpoints that ship to the API.',
  },
  {
    n: 3,
    title: 'Runtime inference flow',
    deck: 'What happens when a phone POSTs a session — auth, sliding windows, dual-model scoring, event grouping, persistence.',
  },
  {
    n: 4,
    title: 'Database entity-relationships',
    deck: 'Eight core tables — users, sessions, inferences, fall events, timeline, transitions, feedback, auth — with their joins.',
  },
  {
    n: 5,
    title: 'Data schema & label mapping',
    deck: 'The harmonised feature columns and the unified label space across UCI HAR, PAMAP2, WISDM, MobiFall, SisFall.',
  },
  {
    n: 6,
    title: 'Admin panel navigation & IA',
    deck: 'The two-shell information architecture: Editorial routes for narrative, Analytics for operators.',
  },
  {
    n: 7,
    title: 'Vulnerability-aware fall fusion',
    deck: 'How impact, post-impact dynamics, and stage-pass flags are combined into the calibrated fall probability.',
  },
] as const
