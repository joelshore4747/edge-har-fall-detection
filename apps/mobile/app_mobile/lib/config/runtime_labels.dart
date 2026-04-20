const List<String> runtimeSaveCategories = <String>[
  'fall',
  'stairs',
  'walking',
  'standing',
  'sitting',
  'lying',
  'transition',
  'other',
];

const List<String> runtimeAnnotationActivityOptions = <String>[
  'unknown',
  ...runtimeSaveCategories,
];

String normaliseRuntimeActivityLabel(
  String? value, {
  String fallback = 'other',
}) {
  final normalised = (value ?? '').trim().toLowerCase();
  if (normalised.isEmpty) {
    return fallback;
  }

  switch (normalised) {
    case 'laying':
      return 'lying';
    case 'stairs_up':
    case 'stairs_down':
      return 'stairs';
    default:
      if (runtimeAnnotationActivityOptions.contains(normalised)) {
        return normalised;
      }
      return fallback;
  }
}
