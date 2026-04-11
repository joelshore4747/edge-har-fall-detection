import 'package:flutter/foundation.dart';

@immutable
class SavedSession {
  const SavedSession({
    required this.filePath,
    required this.fileName,
    required this.subjectId,
    required this.placement,
    required this.sampleCount,
    required this.savedAt,
    this.activityLabel,
    this.placementLabel,
    this.testId,
    this.testTitle,
    this.notes,
  });

  final String filePath;
  final String fileName;
  final String subjectId;
  final String placement;
  final int sampleCount;
  final DateTime savedAt;
  final String? activityLabel;
  final String? placementLabel;
  final String? testId;
  final String? testTitle;
  final String? notes;

  bool get hasActivityLabel =>
      activityLabel != null && activityLabel!.trim().isNotEmpty;

  bool get hasPlacementLabel =>
      placementLabel != null && placementLabel!.trim().isNotEmpty;

  bool get hasTestId => testId != null && testId!.trim().isNotEmpty;

  bool get hasTestTitle => testTitle != null && testTitle!.trim().isNotEmpty;

  bool get hasNotes => notes != null && notes!.trim().isNotEmpty;

  bool get isLabelled => hasActivityLabel || hasPlacementLabel;

  SavedSession copyWith({
    String? filePath,
    String? fileName,
    String? subjectId,
    String? placement,
    int? sampleCount,
    DateTime? savedAt,
    String? activityLabel,
    String? placementLabel,
    String? testId,
    String? testTitle,
    String? notes,
    bool clearActivityLabel = false,
    bool clearPlacementLabel = false,
    bool clearTestId = false,
    bool clearTestTitle = false,
    bool clearNotes = false,
  }) {
    return SavedSession(
      filePath: filePath ?? this.filePath,
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
      testId: clearTestId ? null : (testId ?? this.testId),
      testTitle: clearTestTitle ? null : (testTitle ?? this.testTitle),
      notes: clearNotes ? null : (notes ?? this.notes),
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
      'test_id': testId,
      'test_title': testTitle,
      'notes': notes,
    };
  }

  factory SavedSession.fromJson(Map<String, dynamic> json) {
    final notes = _asNullableString(json['notes']);

    return SavedSession(
      filePath: (json['file_path'] ?? '').toString(),
      fileName: (json['file_name'] ?? '').toString(),
      subjectId: (json['subject_id'] ?? 'unknown').toString(),
      placement: (json['placement'] ?? 'unknown').toString(),
      sampleCount: _asInt(json['sample_count']) ?? 0,
      savedAt: _asDateTime(json['saved_at']) ?? DateTime.now().toUtc(),
      activityLabel: _asNullableString(json['activity_label']),
      placementLabel: _asNullableString(json['placement_label']),
      testId:
          _asNullableString(json['test_id']) ??
          _metadataValueFromNotes(notes, 'test_id'),
      testTitle:
          _asNullableString(json['test_title']) ??
          _metadataValueFromNotes(notes, 'test_title'),
      notes: notes,
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

  static String? _metadataValueFromNotes(String? notes, String fieldName) {
    final trimmedNotes = notes?.trim();
    if (trimmedNotes == null || trimmedNotes.isEmpty) {
      return null;
    }

    for (final part in trimmedNotes.split(' | ')) {
      if (part.startsWith('$fieldName=')) {
        final value = part.substring(fieldName.length + 1).trim();
        if (value.isNotEmpty) {
          return value;
        }
      }
    }

    return null;
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
        'testId: $testId, '
        'testTitle: $testTitle, '
        'notes: $notes'
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
        other.testId == testId &&
        other.testTitle == testTitle &&
        other.notes == notes;
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
    testId,
    testTitle,
    notes,
  );
}
