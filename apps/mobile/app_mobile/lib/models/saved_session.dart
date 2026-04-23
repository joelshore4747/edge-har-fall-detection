import 'package:flutter/foundation.dart';

import 'persisted_session.dart';

@immutable
class SavedSession {
  const SavedSession({
    this.filePath,
    required this.fileName,
    required this.subjectId,
    required this.placement,
    required this.sampleCount,
    required this.savedAt,
    this.activityLabel,
    this.placementLabel,
    this.notes,
    this.persistedUserId,
    this.persistedSessionId,
    this.persistedInferenceId,
    this.isRemote = false,
  });

  final String? filePath;
  final String fileName;
  final String subjectId;
  final String placement;
  final int sampleCount;
  final DateTime savedAt;
  final String? activityLabel;
  final String? placementLabel;
  final String? notes;
  final String? persistedUserId;
  final String? persistedSessionId;
  final String? persistedInferenceId;
  final bool isRemote;

  bool get hasActivityLabel =>
      activityLabel != null && activityLabel!.trim().isNotEmpty;

  bool get hasPlacementLabel =>
      placementLabel != null && placementLabel!.trim().isNotEmpty;

  bool get hasNotes => notes != null && notes!.trim().isNotEmpty;

  bool get isLabelled => hasActivityLabel || hasPlacementLabel;

  bool get hasLocalFile => filePath != null && filePath!.trim().isNotEmpty;

  bool get hasPersistedSession =>
      persistedSessionId != null && persistedSessionId!.trim().isNotEmpty;

  bool get isRemoteOnly => isRemote && !hasLocalFile;

  factory SavedSession.fromPersistedSummary(PersistedSessionSummary summary) {
    return SavedSession(
      fileName: summary.session.sessionName ?? summary.session.clientSessionId,
      subjectId: summary.session.subjectId,
      placement: summary.session.placementDeclared,
      sampleCount: summary.session.sampleCount,
      savedAt: summary.sortTimestamp.toUtc(),
      activityLabel: summary.session.activityLabel,
      notes: summary.session.notes,
      persistedUserId: summary.session.userId,
      persistedSessionId: summary.session.appSessionId,
      persistedInferenceId: summary.latestInferenceId,
      isRemote: true,
    );
  }

  SavedSession copyWith({
    String? filePath,
    String? fileName,
    String? subjectId,
    String? placement,
    int? sampleCount,
    DateTime? savedAt,
    String? activityLabel,
    String? placementLabel,
    String? notes,
    String? persistedUserId,
    String? persistedSessionId,
    String? persistedInferenceId,
    bool? isRemote,
    bool clearActivityLabel = false,
    bool clearPlacementLabel = false,
    bool clearNotes = false,
    bool clearFilePath = false,
    bool clearPersistedUserId = false,
    bool clearPersistedSessionId = false,
    bool clearPersistedInferenceId = false,
  }) {
    return SavedSession(
      filePath: clearFilePath ? null : (filePath ?? this.filePath),
      fileName: fileName ?? this.fileName,
      subjectId: subjectId ?? this.subjectId,
      placement: placement ?? this.placement,
      sampleCount: sampleCount ?? this.sampleCount,
      savedAt: savedAt ?? this.savedAt,
      activityLabel: clearActivityLabel
          ? null
          : (activityLabel ?? this.activityLabel),
      placementLabel: clearPlacementLabel
          ? null
          : (placementLabel ?? this.placementLabel),
      notes: clearNotes ? null : (notes ?? this.notes),
      persistedUserId: clearPersistedUserId
          ? null
          : (persistedUserId ?? this.persistedUserId),
      persistedSessionId: clearPersistedSessionId
          ? null
          : (persistedSessionId ?? this.persistedSessionId),
      persistedInferenceId: clearPersistedInferenceId
          ? null
          : (persistedInferenceId ?? this.persistedInferenceId),
      isRemote: isRemote ?? this.isRemote,
    );
  }

  Map<String, dynamic> toJson() {
    return <String, dynamic>{
      'file_path': filePath,
      'file_name': fileName,
      'subject_id': subjectId,
      'placement': placement,
      'sample_count': sampleCount,
      'saved_at': savedAt.toUtc().toIso8601String(),
      'activity_label': activityLabel,
      'placement_label': placementLabel,
      'notes': notes,
      'persisted_user_id': persistedUserId,
      'persisted_session_id': persistedSessionId,
      'persisted_inference_id': persistedInferenceId,
      'is_remote': isRemote,
    };
  }

  factory SavedSession.fromJson(Map<String, dynamic> json) {
    return SavedSession(
      filePath: _asNullableString(json['file_path']),
      fileName: (json['file_name'] ?? '').toString(),
      subjectId: (json['subject_id'] ?? 'unknown').toString(),
      placement: (json['placement'] ?? 'unknown').toString(),
      sampleCount: _asInt(json['sample_count']) ?? 0,
      savedAt: _asDateTime(json['saved_at']) ?? DateTime.now().toUtc(),
      activityLabel: _asNullableString(json['activity_label']),
      placementLabel: _asNullableString(json['placement_label']),
      notes: _asNullableString(json['notes']),
      persistedUserId: _asNullableString(json['persisted_user_id']),
      persistedSessionId: _asNullableString(json['persisted_session_id']),
      persistedInferenceId: _asNullableString(json['persisted_inference_id']),
      isRemote: json['is_remote'] == true,
    );
  }

  static int? _asInt(dynamic value) {
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

  static DateTime? _asDateTime(dynamic value) {
    if (value == null) {
      return null;
    }
    if (value is DateTime) {
      return value;
    }
    return DateTime.tryParse(value.toString());
  }

  static String? _asNullableString(dynamic value) {
    if (value == null) {
      return null;
    }
    final text = value.toString().trim();
    return text.isEmpty ? null : text;
  }

  @override
  String toString() {
    return 'SavedSession('
        'filePath: $filePath, '
        'fileName: $fileName, '
        'subjectId: $subjectId, '
        'placement: $placement, '
        'sampleCount: $sampleCount, '
        'savedAt: $savedAt, '
        'activityLabel: $activityLabel, '
        'placementLabel: $placementLabel, '
        'notes: $notes, '
        'persistedUserId: $persistedUserId, '
        'persistedSessionId: $persistedSessionId, '
        'persistedInferenceId: $persistedInferenceId, '
        'isRemote: $isRemote'
        ')';
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;

    return other is SavedSession &&
        other.filePath == filePath &&
        other.fileName == fileName &&
        other.subjectId == subjectId &&
        other.placement == placement &&
        other.sampleCount == sampleCount &&
        other.savedAt == savedAt &&
        other.activityLabel == activityLabel &&
        other.placementLabel == placementLabel &&
        other.notes == notes &&
        other.persistedUserId == persistedUserId &&
        other.persistedSessionId == persistedSessionId &&
        other.persistedInferenceId == persistedInferenceId &&
        other.isRemote == isRemote;
  }

  @override
  int get hashCode => Object.hash(
    filePath,
    fileName,
    subjectId,
    placement,
    sampleCount,
    savedAt,
    activityLabel,
    placementLabel,
    notes,
    persistedUserId,
    persistedSessionId,
    persistedInferenceId,
    isRemote,
  );
}
