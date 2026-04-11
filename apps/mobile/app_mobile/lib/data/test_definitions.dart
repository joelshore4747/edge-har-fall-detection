class TestDefinition {
  final String id;
  final String title;
  final String instructions;
  final int suggestedDurationSeconds;
  final bool isFallRelated;
  final String safetyNote;

  const TestDefinition({
    required this.id,
    required this.title,
    required this.instructions,
    required this.suggestedDurationSeconds,
    required this.isFallRelated,
    required this.safetyNote,
  });
}

const List<TestDefinition> kTestDefinitions = [
  TestDefinition(
    id: 'baseline_still',
    title: '1. Stand Still',
    instructions:
        'Stand upright and stay as still as possible with the phone in your usual test position.',
    suggestedDurationSeconds: 15,
    isFallRelated: false,
    safetyNote: 'Normal safe test.',
  ),
  TestDefinition(
    id: 'slow_walk',
    title: '2. Slow Walk',
    instructions:
        'Walk slowly in a straight line for the full test duration.',
    suggestedDurationSeconds: 20,
    isFallRelated: false,
    safetyNote: 'Normal safe test.',
  ),
  TestDefinition(
    id: 'normal_walk',
    title: '3. Normal Walk',
    instructions:
        'Walk at a natural pace as you normally would indoors or on flat ground.',
    suggestedDurationSeconds: 20,
    isFallRelated: false,
    safetyNote: 'Normal safe test.',
  ),
  TestDefinition(
    id: 'fast_walk',
    title: '4. Fast Walk',
    instructions:
        'Walk quickly without running. Keep the phone in the same position throughout.',
    suggestedDurationSeconds: 20,
    isFallRelated: false,
    safetyNote: 'Normal safe test.',
  ),
  TestDefinition(
    id: 'sit_down_stand_up',
    title: '5. Sit Down and Stand Up',
    instructions:
        'Start standing, sit down in a controlled way, pause briefly, then stand back up.',
    suggestedDurationSeconds: 15,
    isFallRelated: false,
    safetyNote: 'Use a stable chair.',
  ),
  TestDefinition(
    id: 'lie_down_and_get_up',
    title: '6. Lie Down and Get Up',
    instructions:
        'Lower yourself carefully to a lying position, pause, then get back up.',
    suggestedDurationSeconds: 20,
    isFallRelated: false,
    safetyNote: 'Use a safe soft surface if needed.',
  ),
  TestDefinition(
    id: 'stairs_up',
    title: '7. Walk Up Stairs',
    instructions:
        'Walk up a short flight of stairs at a normal pace.',
    suggestedDurationSeconds: 15,
    isFallRelated: false,
    safetyNote: 'Hold the handrail if needed.',
  ),
  TestDefinition(
    id: 'stairs_down',
    title: '8. Walk Down Stairs',
    instructions:
        'Walk down a short flight of stairs at a normal pace.',
    suggestedDurationSeconds: 15,
    isFallRelated: false,
    safetyNote: 'Use extra caution and hold the handrail.',
  ),
  TestDefinition(
    id: 'stumble_recovery',
    title: '9. Stumble Recovery',
    instructions:
        'Perform a mild, controlled stumble or balance-loss motion and recover without falling.',
    suggestedDurationSeconds: 10,
    isFallRelated: true,
    safetyNote: 'Keep this controlled and do not force a real fall.',
  ),
  TestDefinition(
    id: 'controlled_fall_simulation',
    title: '10. Controlled Fall Simulation',
    instructions:
        'Only if safe: perform a carefully controlled fall-like motion onto a soft protected surface.',
    suggestedDurationSeconds: 10,
    isFallRelated: true,
    safetyNote: 'Only do this with padding, supervision, and no real injury risk.',
  ),
];