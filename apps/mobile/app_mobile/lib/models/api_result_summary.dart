import 'dart:convert';

class ApiResultSummary {
  ApiResultSummary({
    required this.sessionId,
    required this.alertSummary,
    required this.placementSummary,
    required this.harSummary,
    required this.fallSummary,
    required this.timelineEvents,
    required this.transitionEvents,
    required this.narrativeSummary,
    required this.pointTimeline,
    required this.groupedFallEvents,
    required this.rawJson,
    this.requestId,
    this.persistedUserId,
    this.persistedSessionId,
    this.persistedInferenceId,
  });

  final String? requestId;
  final String sessionId;
  final String? persistedUserId;
  final String? persistedSessionId;
  final String? persistedInferenceId;
  final RuntimeAlertSummaryModel alertSummary;
  final PlacementSummaryModel placementSummary;
  final HarSummaryModel harSummary;
  final FallSummaryModel fallSummary;
  final List<TimelineEventModel> timelineEvents;
  final List<TransitionEventModel> transitionEvents;
  final SessionNarrativeSummaryModel? narrativeSummary;
  final List<PointTimelinePointModel> pointTimeline;
  final List<GroupedFallEventModel> groupedFallEvents;
  final Map<String, dynamic> rawJson;

  String get warningLevel => alertSummary.warningLevel;
  bool get likelyFallDetected => alertSummary.likelyFallDetected;
  String? get topHarLabel => alertSummary.topHarLabel;
  double? get topHarFraction => alertSummary.topHarFraction;
  int get groupedFallEventCount => alertSummary.groupedFallEventCount;
  double? get topFallProbability => alertSummary.topFallProbability;
  String get recommendedMessage => alertSummary.recommendedMessage;

  factory ApiResultSummary.fromJson(Map<String, dynamic> json) {
    return ApiResultSummary(
      requestId: _asString(json['request_id']),
      sessionId: _asString(json['session_id']) ?? 'unknown_session',
      persistedUserId: _asString(json['persisted_user_id']),
      persistedSessionId: _asString(json['persisted_session_id']),
      persistedInferenceId: _asString(json['persisted_inference_id']),
      alertSummary: RuntimeAlertSummaryModel.fromJson(
        _asMap(json['alert_summary']),
      ),
      placementSummary: PlacementSummaryModel.fromJson(
        _asMap(json['placement_summary']),
      ),
      harSummary: HarSummaryModel.fromJson(_asMap(json['har_summary'])),
      fallSummary: FallSummaryModel.fromJson(_asMap(json['fall_summary'])),
      timelineEvents: _asList(json['timeline_events'])
          .map((item) => TimelineEventModel.fromJson(_asMap(item)))
          .toList(growable: false),
      transitionEvents: _asList(json['transition_events'])
          .map((item) => TransitionEventModel.fromJson(_asMap(item)))
          .toList(growable: false),
      narrativeSummary: json['session_narrative_summary'] == null
          ? null
          : SessionNarrativeSummaryModel.fromJson(
              _asMap(json['session_narrative_summary']),
            ),
      pointTimeline: _asList(json['point_timeline'])
          .map((item) => PointTimelinePointModel.fromJson(_asMap(item)))
          .toList(growable: false),
      groupedFallEvents: _asList(json['grouped_fall_events'])
          .map((item) => GroupedFallEventModel.fromJson(_asMap(item)))
          .toList(growable: false),
      rawJson: Map<String, dynamic>.from(json),
    );
  }

  factory ApiResultSummary.fromRawJson(String raw) {
    final decoded = jsonDecode(raw) as Map<String, dynamic>;
    return ApiResultSummary.fromJson(decoded);
  }
}

class RuntimeAlertSummaryModel {
  RuntimeAlertSummaryModel({
    required this.warningLevel,
    required this.likelyFallDetected,
    required this.groupedFallEventCount,
    required this.recommendedMessage,
    this.topHarLabel,
    this.topHarFraction,
    this.topFallProbability,
  });

  final String warningLevel;
  final bool likelyFallDetected;
  final String? topHarLabel;
  final double? topHarFraction;
  final int groupedFallEventCount;
  final double? topFallProbability;
  final String recommendedMessage;

  factory RuntimeAlertSummaryModel.fromJson(Map<String, dynamic> json) {
    return RuntimeAlertSummaryModel(
      warningLevel: _asString(json['warning_level']) ?? 'none',
      likelyFallDetected: _asBool(json['likely_fall_detected']) ?? false,
      topHarLabel: _asString(json['top_har_label']),
      topHarFraction: _asDouble(json['top_har_fraction']),
      groupedFallEventCount: _asInt(json['grouped_fall_event_count']) ?? 0,
      topFallProbability: _asDouble(json['top_fall_probability']),
      recommendedMessage:
          _asString(json['recommended_message']) ??
          'No recommendation available.',
    );
  }
}

class PlacementSummaryModel {
  PlacementSummaryModel({
    required this.placementState,
    required this.stateCounts,
    this.placementConfidence,
    this.stateFraction,
  });

  final String placementState;
  final double? placementConfidence;
  final double? stateFraction;
  final Map<String, int> stateCounts;

  factory PlacementSummaryModel.fromJson(Map<String, dynamic> json) {
    final counts = <String, int>{};
    final rawCounts = json['state_counts'];
    if (rawCounts is Map) {
      rawCounts.forEach((key, value) {
        final parsed = _asInt(value);
        counts[key.toString()] = parsed ?? 0;
      });
    }

    return PlacementSummaryModel(
      placementState: _asString(json['placement_state']) ?? 'unknown',
      placementConfidence: _asDouble(json['placement_confidence']),
      stateFraction: _asDouble(json['state_fraction']),
      stateCounts: counts,
    );
  }
}

class HarSummaryModel {
  HarSummaryModel({
    required this.labelCounts,
    required this.totalWindows,
    this.topLabel,
    this.topLabelFraction,
  });

  final String? topLabel;
  final double? topLabelFraction;
  final Map<String, int> labelCounts;
  final int totalWindows;

  factory HarSummaryModel.fromJson(Map<String, dynamic> json) {
    final counts = <String, int>{};
    final rawCounts = json['label_counts'];
    if (rawCounts is Map) {
      rawCounts.forEach((key, value) {
        counts[key.toString()] = _asInt(value) ?? 0;
      });
    }

    return HarSummaryModel(
      topLabel: _asString(json['top_label']),
      topLabelFraction: _asDouble(json['top_label_fraction']),
      labelCounts: counts,
      totalWindows: _asInt(json['total_windows']) ?? 0,
    );
  }
}

class FallSummaryModel {
  FallSummaryModel({
    required this.likelyFallDetected,
    required this.positiveWindowCount,
    required this.groupedEventCount,
    this.topFallProbability,
    this.meanFallProbability,
  });

  final bool likelyFallDetected;
  final int positiveWindowCount;
  final int groupedEventCount;
  final double? topFallProbability;
  final double? meanFallProbability;

  factory FallSummaryModel.fromJson(Map<String, dynamic> json) {
    return FallSummaryModel(
      likelyFallDetected: _asBool(json['likely_fall_detected']) ?? false,
      positiveWindowCount: _asInt(json['positive_window_count']) ?? 0,
      groupedEventCount: _asInt(json['grouped_event_count']) ?? 0,
      topFallProbability: _asDouble(json['top_fall_probability']),
      meanFallProbability: _asDouble(json['mean_fall_probability']),
    );
  }
}

class TimelineEventModel {
  TimelineEventModel({
    required this.eventId,
    required this.startTs,
    required this.endTs,
    required this.durationSeconds,
    required this.pointCount,
    required this.activityLabel,
    required this.placementLabel,
    required this.likelyFall,
    required this.eventKind,
    required this.relatedGroupedFallEventIds,
    required this.description,
    this.midpointTs,
    this.activityConfidenceMean,
    this.placementConfidenceMean,
    this.fallProbabilityPeak,
    this.fallProbabilityMean,
  });

  final String eventId;
  final double startTs;
  final double endTs;
  final double durationSeconds;
  final double? midpointTs;
  final int pointCount;
  final String activityLabel;
  final String placementLabel;
  final double? activityConfidenceMean;
  final double? placementConfidenceMean;
  final double? fallProbabilityPeak;
  final double? fallProbabilityMean;
  final bool likelyFall;
  final String eventKind;
  final List<String> relatedGroupedFallEventIds;
  final String description;

  factory TimelineEventModel.fromJson(Map<String, dynamic> json) {
    return TimelineEventModel(
      eventId: _asString(json['event_id']) ?? 'unknown_event',
      startTs: _asDouble(json['start_ts']) ?? 0.0,
      endTs: _asDouble(json['end_ts']) ?? 0.0,
      durationSeconds: _asDouble(json['duration_seconds']) ?? 0.0,
      midpointTs: _asDouble(json['midpoint_ts']),
      pointCount: _asInt(json['point_count']) ?? 0,
      activityLabel: _asString(json['activity_label']) ?? 'unknown',
      placementLabel: _asString(json['placement_label']) ?? 'unknown',
      activityConfidenceMean: _asDouble(json['activity_confidence_mean']),
      placementConfidenceMean: _asDouble(json['placement_confidence_mean']),
      fallProbabilityPeak: _asDouble(json['fall_probability_peak']),
      fallProbabilityMean: _asDouble(json['fall_probability_mean']),
      likelyFall: _asBool(json['likely_fall']) ?? false,
      eventKind: _asString(json['event_kind']) ?? 'activity',
      relatedGroupedFallEventIds: _asList(
        json['related_grouped_fall_event_ids'],
      ).map((item) => item.toString()).toList(growable: false),
      description: _asString(json['description']) ?? '',
    );
  }

  String get humanActivityLabel => _titleise(activityLabel);
  String get humanPlacementLabel => _titleise(placementLabel);
}

class TransitionEventModel {
  TransitionEventModel({
    required this.transitionId,
    required this.transitionTs,
    required this.fromEventId,
    required this.toEventId,
    required this.transitionKind,
    required this.description,
    this.fromActivityLabel,
    this.toActivityLabel,
    this.fromPlacementLabel,
    this.toPlacementLabel,
  });

  final String transitionId;
  final double transitionTs;
  final String fromEventId;
  final String toEventId;
  final String transitionKind;
  final String? fromActivityLabel;
  final String? toActivityLabel;
  final String? fromPlacementLabel;
  final String? toPlacementLabel;
  final String description;

  factory TransitionEventModel.fromJson(Map<String, dynamic> json) {
    return TransitionEventModel(
      transitionId: _asString(json['transition_id']) ?? 'unknown_transition',
      transitionTs: _asDouble(json['transition_ts']) ?? 0.0,
      fromEventId: _asString(json['from_event_id']) ?? '',
      toEventId: _asString(json['to_event_id']) ?? '',
      transitionKind: _asString(json['transition_kind']) ?? 'transition',
      fromActivityLabel: _asString(json['from_activity_label']),
      toActivityLabel: _asString(json['to_activity_label']),
      fromPlacementLabel: _asString(json['from_placement_label']),
      toPlacementLabel: _asString(json['to_placement_label']),
      description: _asString(json['description']) ?? '',
    );
  }
}

class SessionNarrativeSummaryModel {
  SessionNarrativeSummaryModel({
    required this.sessionId,
    required this.datasetName,
    required this.subjectId,
    required this.totalDurationSeconds,
    required this.eventCount,
    required this.transitionCount,
    required this.fallEventCount,
    required this.dominantActivityLabel,
    required this.dominantPlacementLabel,
    required this.summaryText,
    this.highestFallProbability,
  });

  final String sessionId;
  final String datasetName;
  final String subjectId;
  final double totalDurationSeconds;
  final int eventCount;
  final int transitionCount;
  final int fallEventCount;
  final String dominantActivityLabel;
  final String dominantPlacementLabel;
  final double? highestFallProbability;
  final String summaryText;

  factory SessionNarrativeSummaryModel.fromJson(Map<String, dynamic> json) {
    return SessionNarrativeSummaryModel(
      sessionId: _asString(json['session_id']) ?? 'unknown_session',
      datasetName: _asString(json['dataset_name']) ?? 'unknown_dataset',
      subjectId: _asString(json['subject_id']) ?? 'unknown_subject',
      totalDurationSeconds: _asDouble(json['total_duration_seconds']) ?? 0.0,
      eventCount: _asInt(json['event_count']) ?? 0,
      transitionCount: _asInt(json['transition_count']) ?? 0,
      fallEventCount: _asInt(json['fall_event_count']) ?? 0,
      dominantActivityLabel:
          _asString(json['dominant_activity_label']) ?? 'unknown',
      dominantPlacementLabel:
          _asString(json['dominant_placement_label']) ?? 'unknown',
      highestFallProbability: _asDouble(json['highest_fall_probability']),
      summaryText: _asString(json['summary_text']) ?? '',
    );
  }
}

class PointTimelinePointModel {
  PointTimelinePointModel({
    required this.midpointTs,
    this.activityLabel,
    this.placementLabel,
    this.activityConfidence,
    this.placementConfidence,
    this.fallProbability,
    this.elevatedFall,
  });

  final double midpointTs;
  final String? activityLabel;
  final String? placementLabel;
  final double? activityConfidence;
  final double? placementConfidence;
  final double? fallProbability;
  final bool? elevatedFall;

  factory PointTimelinePointModel.fromJson(Map<String, dynamic> json) {
    return PointTimelinePointModel(
      midpointTs: _asDouble(json['midpoint_ts']) ?? 0.0,
      activityLabel: _asString(json['activity_label']),
      placementLabel: _asString(json['placement_label']),
      activityConfidence: _asDouble(json['activity_confidence']),
      placementConfidence: _asDouble(json['placement_confidence']),
      fallProbability: _asDouble(json['fall_probability']),
      elevatedFall: _asBool(json['elevated_fall']),
    );
  }
}

class GroupedFallEventModel {
  GroupedFallEventModel({
    required this.eventId,
    required this.eventStartTs,
    required this.eventEndTs,
    required this.eventDurationSeconds,
    required this.positiveWindowCount,
    this.peakProbability,
    this.meanProbability,
    this.medianProbability,
  });

  final String eventId;
  final double eventStartTs;
  final double eventEndTs;
  final double eventDurationSeconds;
  final int positiveWindowCount;
  final double? peakProbability;
  final double? meanProbability;
  final double? medianProbability;

  factory GroupedFallEventModel.fromJson(Map<String, dynamic> json) {
    return GroupedFallEventModel(
      eventId: _asString(json['event_id']) ?? 'unknown_event',
      eventStartTs: _asDouble(json['event_start_ts']) ?? 0.0,
      eventEndTs: _asDouble(json['event_end_ts']) ?? 0.0,
      eventDurationSeconds: _asDouble(json['event_duration_seconds']) ?? 0.0,
      positiveWindowCount: _asInt(json['n_positive_windows']) ?? 0,
      peakProbability: _asDouble(json['peak_probability']),
      meanProbability: _asDouble(json['mean_probability']),
      medianProbability: _asDouble(json['median_probability']),
    );
  }
}

Map<String, dynamic> _asMap(dynamic value) {
  if (value is Map<String, dynamic>) {
    return value;
  }
  if (value is Map) {
    return value.map((key, item) => MapEntry(key.toString(), item));
  }
  return <String, dynamic>{};
}

List<dynamic> _asList(dynamic value) {
  if (value is List) {
    return value;
  }
  return const <dynamic>[];
}

String? _asString(dynamic value) {
  if (value == null) {
    return null;
  }
  final text = value.toString();
  return text.isEmpty ? null : text;
}

int? _asInt(dynamic value) {
  if (value == null) {
    return null;
  }
  if (value is int) {
    return value;
  }
  if (value is num) {
    return value.toInt();
  }
  return int.tryParse(value.toString());
}

double? _asDouble(dynamic value) {
  if (value == null) {
    return null;
  }
  if (value is double) {
    return value;
  }
  if (value is num) {
    return value.toDouble();
  }
  return double.tryParse(value.toString());
}

bool? _asBool(dynamic value) {
  if (value == null) {
    return null;
  }
  if (value is bool) {
    return value;
  }
  final text = value.toString().trim().toLowerCase();
  if (text == 'true' || text == '1' || text == 'yes') {
    return true;
  }
  if (text == 'false' || text == '0' || text == 'no') {
    return false;
  }
  return null;
}

String _titleise(String value) {
  final cleaned = value.replaceAll('_', ' ').trim();
  if (cleaned.isEmpty) {
    return value;
  }

  return cleaned
      .split(' ')
      .map((part) {
        if (part.isEmpty) {
          return part;
        }
        return '${part[0].toUpperCase()}${part.substring(1)}';
      })
      .join(' ');
}
