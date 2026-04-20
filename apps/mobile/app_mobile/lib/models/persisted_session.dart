import 'api_result_summary.dart';

class PersistedSessionRecord {
  PersistedSessionRecord({
    required this.appSessionId,
    required this.userId,
    required this.subjectId,
    required this.clientSessionId,
    required this.datasetName,
    required this.sourceType,
    required this.taskType,
    required this.placementDeclared,
    required this.devicePlatform,
    required this.recordingMode,
    required this.runtimeMode,
    required this.uploadedAt,
    required this.sampleCount,
    required this.hasGyro,
    required this.createdAt,
    required this.updatedAt,
    this.requestId,
    this.traceId,
    this.clientVersion,
    this.deviceModel,
    this.appVersion,
    this.appBuild,
    this.recordingStartedAt,
    this.recordingEndedAt,
    this.samplingRateHz,
    this.durationSeconds,
    this.notes,
    this.rawStorageUri,
    this.rawStorageFormat,
    this.rawPayloadSha256,
    this.rawPayloadBytes,
  });

  final String appSessionId;
  final String userId;
  final String subjectId;
  final String clientSessionId;
  final String? requestId;
  final String? traceId;
  final String? clientVersion;
  final String datasetName;
  final String sourceType;
  final String taskType;
  final String placementDeclared;
  final String devicePlatform;
  final String? deviceModel;
  final String? appVersion;
  final String? appBuild;
  final String recordingMode;
  final String runtimeMode;
  final DateTime? recordingStartedAt;
  final DateTime? recordingEndedAt;
  final DateTime uploadedAt;
  final double? samplingRateHz;
  final int sampleCount;
  final bool hasGyro;
  final double? durationSeconds;
  final String? notes;
  final String? rawStorageUri;
  final String? rawStorageFormat;
  final String? rawPayloadSha256;
  final int? rawPayloadBytes;
  final DateTime createdAt;
  final DateTime updatedAt;

  factory PersistedSessionRecord.fromJson(Map<String, dynamic> json) {
    return PersistedSessionRecord(
      appSessionId: _asString(json['app_session_id']) ?? '',
      userId: _asString(json['user_id']) ?? '',
      subjectId: _asString(json['subject_id']) ?? 'unknown',
      clientSessionId: _asString(json['client_session_id']) ?? '',
      requestId: _asString(json['request_id']),
      traceId: _asString(json['trace_id']),
      clientVersion: _asString(json['client_version']),
      datasetName: _asString(json['dataset_name']) ?? 'unknown',
      sourceType: _asString(json['source_type']) ?? 'unknown',
      taskType: _asString(json['task_type']) ?? 'unknown',
      placementDeclared: _asString(json['placement_declared']) ?? 'unknown',
      devicePlatform: _asString(json['device_platform']) ?? 'unknown',
      deviceModel: _asString(json['device_model']),
      appVersion: _asString(json['app_version']),
      appBuild: _asString(json['app_build']),
      recordingMode: _asString(json['recording_mode']) ?? 'unknown',
      runtimeMode: _asString(json['runtime_mode']) ?? 'unknown',
      recordingStartedAt: _tryParseDateTime(json['recording_started_at']),
      recordingEndedAt: _tryParseDateTime(json['recording_ended_at']),
      uploadedAt:
          _tryParseDateTime(json['uploaded_at']) ?? DateTime.now().toUtc(),
      samplingRateHz: _asDouble(json['sampling_rate_hz']),
      sampleCount: _asInt(json['sample_count']) ?? 0,
      hasGyro: _asBool(json['has_gyro']) ?? false,
      durationSeconds: _asDouble(json['duration_seconds']),
      notes: _asString(json['notes']),
      rawStorageUri: _asString(json['raw_storage_uri']),
      rawStorageFormat: _asString(json['raw_storage_format']),
      rawPayloadSha256: _asString(json['raw_payload_sha256']),
      rawPayloadBytes: _asInt(json['raw_payload_bytes']),
      createdAt:
          _tryParseDateTime(json['created_at']) ?? DateTime.now().toUtc(),
      updatedAt:
          _tryParseDateTime(json['updated_at']) ?? DateTime.now().toUtc(),
    );
  }
}

class PersistedSessionSummary {
  PersistedSessionSummary({
    required this.session,
    this.latestInferenceId,
    this.latestInferenceRequestId,
    this.latestInferenceCreatedAt,
    this.latestStatus,
    this.latestWarningLevel,
    this.latestLikelyFallDetected,
    this.latestTopHarLabel,
    this.latestTopFallProbability,
    this.latestGroupedFallEventCount,
  });

  final PersistedSessionRecord session;
  final String? latestInferenceId;
  final String? latestInferenceRequestId;
  final DateTime? latestInferenceCreatedAt;
  final String? latestStatus;
  final String? latestWarningLevel;
  final bool? latestLikelyFallDetected;
  final String? latestTopHarLabel;
  final double? latestTopFallProbability;
  final int? latestGroupedFallEventCount;

  DateTime get sortTimestamp => latestInferenceCreatedAt ?? session.uploadedAt;

  factory PersistedSessionSummary.fromJson(Map<String, dynamic> json) {
    return PersistedSessionSummary(
      session: PersistedSessionRecord.fromJson(_asMap(json['session'])),
      latestInferenceId: _asString(json['latest_inference_id']),
      latestInferenceRequestId: _asString(json['latest_inference_request_id']),
      latestInferenceCreatedAt: _tryParseDateTime(
        json['latest_inference_created_at'],
      ),
      latestStatus: _asString(json['latest_status']),
      latestWarningLevel: _asString(json['latest_warning_level']),
      latestLikelyFallDetected: _asBool(json['latest_likely_fall_detected']),
      latestTopHarLabel: _asString(json['latest_top_har_label']),
      latestTopFallProbability: _asDouble(json['latest_top_fall_probability']),
      latestGroupedFallEventCount: _asInt(
        json['latest_grouped_fall_event_count'],
      ),
    );
  }
}

class PersistedInferenceDetail {
  PersistedInferenceDetail({
    required this.inferenceId,
    required this.status,
    required this.startedAt,
    required this.createdAt,
    required this.response,
    this.requestId,
    this.errorMessage,
    this.completedAt,
  });

  final String inferenceId;
  final String? requestId;
  final String status;
  final String? errorMessage;
  final DateTime startedAt;
  final DateTime? completedAt;
  final DateTime createdAt;
  final ApiResultSummary response;

  factory PersistedInferenceDetail.fromJson(Map<String, dynamic> json) {
    return PersistedInferenceDetail(
      inferenceId: _asString(json['inference_id']) ?? '',
      requestId: _asString(json['request_id']),
      status: _asString(json['status']) ?? 'unknown',
      errorMessage: _asString(json['error_message']),
      startedAt:
          _tryParseDateTime(json['started_at']) ?? DateTime.now().toUtc(),
      completedAt: _tryParseDateTime(json['completed_at']),
      createdAt:
          _tryParseDateTime(json['created_at']) ?? DateTime.now().toUtc(),
      response: ApiResultSummary.fromJson(_asMap(json['response'])),
    );
  }
}

class PersistedFeedbackRecord {
  PersistedFeedbackRecord({
    required this.feedbackId,
    required this.appSessionId,
    required this.targetType,
    required this.feedbackType,
    required this.recordedAt,
    this.inferenceId,
    this.targetEventKey,
    this.windowId,
    this.correctedLabel,
    this.reviewerIdentifier,
    this.subjectKey,
    this.notes,
    this.requestId,
  });

  final String feedbackId;
  final String appSessionId;
  final String? inferenceId;
  final String targetType;
  final String? targetEventKey;
  final String? windowId;
  final String feedbackType;
  final String? correctedLabel;
  final String? reviewerIdentifier;
  final String? subjectKey;
  final String? notes;
  final String? requestId;
  final DateTime recordedAt;

  factory PersistedFeedbackRecord.fromJson(Map<String, dynamic> json) {
    return PersistedFeedbackRecord(
      feedbackId: _asString(json['feedback_id']) ?? '',
      appSessionId: _asString(json['app_session_id']) ?? '',
      inferenceId: _asString(json['inference_id']),
      targetType: _asString(json['target_type']) ?? 'session',
      targetEventKey: _asString(json['target_event_key']),
      windowId: _asString(json['window_id']),
      feedbackType: _asString(json['feedback_type']) ?? 'unknown',
      correctedLabel: _asString(json['corrected_label']),
      reviewerIdentifier: _asString(json['reviewer_identifier']),
      subjectKey: _asString(json['subject_key']),
      notes: _asString(json['notes']),
      requestId: _asString(json['request_id']),
      recordedAt:
          _tryParseDateTime(json['recorded_at']) ?? DateTime.now().toUtc(),
    );
  }

  Map<String, dynamic> toJson() {
    return <String, dynamic>{
      'request_id': requestId,
      'recorded_at': recordedAt.toUtc().toIso8601String(),
      'status': 'accepted',
      'session_id': null,
      'persisted_session_id': appSessionId,
      'persisted_inference_id': inferenceId,
      'persisted_feedback_id': feedbackId,
      'target_type': targetType,
      'event_id': targetEventKey,
      'window_id': windowId,
      'user_feedback': feedbackType,
      'message': 'Loaded from server',
      'notes': notes,
    };
  }
}

class PersistedSessionDetail {
  PersistedSessionDetail({
    required this.session,
    required this.feedback,
    this.latestInference,
  });

  final PersistedSessionRecord session;
  final PersistedInferenceDetail? latestInference;
  final List<PersistedFeedbackRecord> feedback;

  factory PersistedSessionDetail.fromJson(Map<String, dynamic> json) {
    return PersistedSessionDetail(
      session: PersistedSessionRecord.fromJson(_asMap(json['session'])),
      latestInference: json['latest_inference'] == null
          ? null
          : PersistedInferenceDetail.fromJson(_asMap(json['latest_inference'])),
      feedback: _asList(json['feedback'])
          .map((item) => PersistedFeedbackRecord.fromJson(_asMap(item)))
          .toList(growable: false),
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
  final text = value.toString().trim();
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
  final raw = value.toString().trim().toLowerCase();
  if (raw.isEmpty) {
    return null;
  }
  if (raw == 'true' || raw == '1' || raw == 'yes') {
    return true;
  }
  if (raw == 'false' || raw == '0' || raw == 'no') {
    return false;
  }
  return null;
}

DateTime? _tryParseDateTime(dynamic value) {
  final raw = _asString(value);
  if (raw == null) {
    return null;
  }
  return DateTime.tryParse(raw);
}
