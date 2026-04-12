import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/foundation.dart' show debugPrint, kDebugMode;
import 'package:path_provider/path_provider.dart';

import '../models/saved_session.dart';
import '../models/sensor_sample.dart';

class SessionStorageService {
  static const String _sessionsFolderName = 'saved_sessions';
  static const int _schemaVersion = 3;

  void _log(String message) {
    if (kDebugMode) {
      debugPrint('[SessionStorage] $message');
    }
  }

  Future<Directory> _getSessionsDirectory() async {
    final docs = await getApplicationDocumentsDirectory();
    final dir = Directory('${docs.path}/$_sessionsFolderName');
    if (!await dir.exists()) {
      await dir.create(recursive: true);
      _log('Created sessions directory at ${dir.path}');
    } else {
      _log('Using sessions directory at ${dir.path}');
    }
    return dir;
  }

  Future<String> getSavedSessionsDirectoryPath() async {
    final dir = await _getSessionsDirectory();
    return dir.path;
  }

  Future<String?> saveSession({
    required String sessionId,
    required String subjectId,
    required String placement,
    required String datasetName,
    required String sourceType,
    required String devicePlatform,
    required List<Map<String, dynamic>> samples,
    String? deviceModel,
    String? appVersion,
    String? appBuild,
    String? recordingMode,
    String? runtimeMode,
    double? samplingRateHz,
    String? testId,
    String? testTitle,
    String? notes,
  }) async {
    if (samples.isEmpty) {
      _log('Skipped saving session $sessionId because there are no samples.');
      return null;
    }

    final dir = await _getSessionsDirectory();
    final now = DateTime.now();
    final resolvedTestId =
        _cleanOptionalValue(testId) ??
        _metadataValueFromNotes(notes, 'test_id');
    final resolvedTestTitle =
        _cleanOptionalValue(testTitle) ??
        _metadataValueFromNotes(notes, 'test_title');
    final fileName = _buildReadableFileName(
      timestamp: now,
      testId: resolvedTestId,
      testTitle: resolvedTestTitle,
    );
    final file = await _uniqueFile(dir, fileName);
    _log('Saving session file to ${file.path}');

    final payload = <String, dynamic>{
      'schema_version': _schemaVersion,
      'session_id': sessionId,
      'subject_id': subjectId.trim().isEmpty
          ? 'anonymous_user'
          : subjectId.trim(),
      'placement': placement.trim().isEmpty ? 'unknown' : placement.trim(),
      'saved_at': now.toIso8601String(),
      'updated_at': now.toIso8601String(),
      'sample_count': samples.length,
      'dataset_name': datasetName,
      'source_type': sourceType,
      'device_platform': devicePlatform,
      'device_model': deviceModel,
      'app_version': appVersion,
      'app_build': appBuild,
      'recording_mode': recordingMode,
      'runtime_mode': runtimeMode,
      'sampling_rate_hz': samplingRateHz,
      'activity_label': null,
      'placement_label': null,
      'test_id': resolvedTestId,
      'test_title': resolvedTestTitle,
      'notes': notes ?? '',
      'samples': samples,
      'feedback': <Map<String, dynamic>>[],
      'inference_result': null,
    };

    try {
      await file.writeAsString(
        const JsonEncoder.withIndent('  ').convert(_stripNulls(payload)),
      );
    } catch (error, stackTrace) {
      _log('Failed to write session file ${file.path}');
      _log('Error: $error');
      if (kDebugMode) {
        debugPrint(stackTrace.toString());
      }
      rethrow;
    }

    _log('Session saved successfully: ${file.path}');
    return file.path;
  }

  Future<List<SavedSession>> listSessions() async {
    final dir = await _getSessionsDirectory();
    final entities = await dir.list().toList();

    final files = entities
        .whereType<File>()
        .where((file) => file.path.toLowerCase().endsWith('.json'))
        .toList(growable: false);

    _log('Scanning ${files.length} saved session file(s) from ${dir.path}');
    final sessions = <SavedSession>[];

    for (final file in files) {
      try {
        _log('Reading saved session manifest ${file.path}');
        final raw = await file.readAsString();
        final payload = _decodeJsonObject(raw, file.path);
        final stat = await file.stat();

        sessions.add(
          SavedSession(
            filePath: file.path,
            fileName: file.uri.pathSegments.last,
            subjectId: _asString(payload['subject_id']) ?? 'unknown',
            placement: _asString(payload['placement']) ?? 'unknown',
            sampleCount:
                _asInt(payload['sample_count']) ??
                ((_asList(payload['samples'])).length),
            savedAt:
                DateTime.tryParse(_asString(payload['saved_at']) ?? '') ??
                stat.modified,
            activityLabel: _asString(payload['activity_label']),
            placementLabel: _asString(payload['placement_label']),
            testId: _resolvedSessionMetadata(payload, 'test_id'),
            testTitle: _resolvedSessionMetadata(payload, 'test_title'),
            notes: _asString(payload['notes']),
          ),
        );
      } catch (error, stackTrace) {
        _log('Skipping malformed session file ${file.path}: $error');
        if (kDebugMode) {
          debugPrint(stackTrace.toString());
        }
      }
    }

    sessions.sort((a, b) => b.savedAt.compareTo(a.savedAt));
    _log('Loaded ${sessions.length} saved session manifest(s)');
    return sessions;
  }

  Future<Map<String, dynamic>> loadSessionPayload(String filePath) async {
    final file = File(filePath);
    _log('Loading session payload from $filePath');
    if (!await file.exists()) {
      throw FileSystemException('Saved session file not found.', filePath);
    }

    final raw = await file.readAsString();
    _log('Read ${raw.length} characters from $filePath');
    final payload = _decodeJsonObject(raw, filePath);
    final normalised = _normaliseLoadedPayload(payload, filePath: filePath);
    _log(
      'Loaded session payload from $filePath with '
      '${_asList(normalised['samples']).length} valid sample(s)',
    );
    return normalised;
  }

  Future<File> exportSessionJsonFile(String filePath) async {
    final payload = await loadSessionPayload(filePath);
    final exportDir = await _getExportsDirectory();
    final baseName = _buildExportBaseName(filePath, payload);

    final file = await _uniqueExportFile(exportDir, baseName, 'json');
    await file.writeAsString(
      const JsonEncoder.withIndent('  ').convert(_stripNulls(payload)),
    );

    _log('Exported JSON session to ${file.path}');
    return file;
  }

  Future<File> exportSessionCsvFile(String filePath) async {
    final payload = await loadSessionPayload(filePath);
    final exportDir = await _getExportsDirectory();
    final baseName = _buildExportBaseName(filePath, payload);

    final csv = _buildSessionCsv(payload);
    final file = await _uniqueExportFile(exportDir, baseName, 'csv');
    await file.writeAsString(csv);

    _log('Exported CSV session to ${file.path}');
    return file;
  }

  Future<Directory> _getExportsDirectory() async {
    final temp = await getTemporaryDirectory();
    final dir = Directory('${temp.path}/session_exports');
    if (!await dir.exists()) {
      await dir.create(recursive: true);
      _log('Created exports directory at ${dir.path}');
    }
    return dir;
  }

  Future<File> _uniqueExportFile(
    Directory dir,
    String baseName,
    String extension,
  ) async {
    var candidate = File('${dir.path}/$baseName.$extension');
    if (!await candidate.exists()) {
      return candidate;
    }

    var index = 2;
    while (await candidate.exists()) {
      candidate = File('${dir.path}/${baseName}_$index.$extension');
      index += 1;
    }
    return candidate;
  }

  String _buildExportBaseName(String filePath, Map<String, dynamic> payload) {
    final testId = _resolvedSessionMetadata(payload, 'test_id');
    final testTitle = _resolvedSessionMetadata(payload, 'test_title');
    final slug = _testSlug(testTitle: testTitle, testId: testId);

    final sessionId =
        _cleanOptionalValue(_asString(payload['session_id'])) ??
        File(filePath).uri.pathSegments.last.replaceAll('.json', '');

    return 'session_export_${_slug(slug)}_${_slug(sessionId)}';
  }

  String _buildSessionCsv(Map<String, dynamic> payload) {
    final samples = _asList(payload['samples']);

    final sessionId = _asString(payload['session_id']) ?? '';
    final subjectId = _asString(payload['subject_id']) ?? '';
    final placement = _asString(payload['placement']) ?? '';
    final savedAt = _asString(payload['saved_at']) ?? '';
    final activityLabel = _asString(payload['activity_label']) ?? '';
    final placementLabel = _asString(payload['placement_label']) ?? '';
    final testId = _resolvedSessionMetadata(payload, 'test_id') ?? '';
    final testTitle = _resolvedSessionMetadata(payload, 'test_title') ?? '';
    final notes = _asString(payload['notes']) ?? '';

    final header = <String>[
      'sample_index',
      'session_id',
      'subject_id',
      'placement',
      'saved_at',
      'activity_label',
      'placement_label',
      'test_id',
      'test_title',
      'timestamp',
      'ax',
      'ay',
      'az',
      'gx',
      'gy',
      'gz',
      'magnitude',
      'notes',
    ];

    final buffer = StringBuffer()..writeln(header.map(_csvCell).join(','));

    for (var i = 0; i < samples.length; i++) {
      final item = samples[i];
      if (item is! Map) {
        continue;
      }

      final sample = item.map((key, value) => MapEntry(key.toString(), value));

      final timestamp = _asDouble(sample['timestamp']);
      final ax = _asDouble(sample['ax']);
      final ay = _asDouble(sample['ay']);
      final az = _asDouble(sample['az']);
      final gx = _asDouble(sample['gx']);
      final gy = _asDouble(sample['gy']);
      final gz = _asDouble(sample['gz']);

      final magnitude = (ax != null && ay != null && az != null)
          ? math.sqrt((ax * ax) + (ay * ay) + (az * az))
          : null;

      final row = <dynamic>[
        i,
        sessionId,
        subjectId,
        placement,
        savedAt,
        activityLabel,
        placementLabel,
        testId,
        testTitle,
        timestamp,
        ax,
        ay,
        az,
        gx,
        gy,
        gz,
        magnitude,
        notes,
      ];

      buffer.writeln(row.map(_csvCell).join(','));
    }

    return buffer.toString();
  }

  Future<void> updateSessionLabels({
    required String filePath,
    required String activityLabel,
    required String placementLabel,
    required String notes,
  }) async {
    final payload = await loadSessionPayload(filePath);
    final existingNotes = _asString(payload['notes']);
    final resolvedTestId =
        _cleanOptionalValue(_asString(payload['test_id'])) ??
        _metadataValueFromNotes(notes, 'test_id') ??
        _metadataValueFromNotes(existingNotes, 'test_id');
    final resolvedTestTitle =
        _cleanOptionalValue(_asString(payload['test_title'])) ??
        _metadataValueFromNotes(notes, 'test_title') ??
        _metadataValueFromNotes(existingNotes, 'test_title');

    payload['activity_label'] = activityLabel;
    payload['placement_label'] = placementLabel;
    payload['test_id'] = resolvedTestId;
    payload['test_title'] = resolvedTestTitle;
    payload['notes'] = notes;
    payload['updated_at'] = DateTime.now().toIso8601String();

    final file = File(filePath);
    await file.writeAsString(
      const JsonEncoder.withIndent('  ').convert(_stripNulls(payload)),
    );
  }

  Future<void> saveInferenceResult({
    required String filePath,
    required Map<String, dynamic> inferenceResult,
  }) async {
    final payload = await loadSessionPayload(filePath);

    payload['inference_result'] = inferenceResult;
    payload['inference_saved_at'] = DateTime.now().toIso8601String();
    payload['updated_at'] = DateTime.now().toIso8601String();

    final file = File(filePath);
    await file.writeAsString(
      const JsonEncoder.withIndent('  ').convert(_stripNulls(payload)),
    );
  }

  Future<void> appendFeedback({
    required String filePath,
    required Map<String, dynamic> feedbackEntry,
  }) async {
    final payload = await loadSessionPayload(filePath);

    final feedback = _asList(payload['feedback'])
        .map(
          (item) => item is Map
              ? item.map((key, value) => MapEntry(key.toString(), value))
              : <String, dynamic>{},
        )
        .toList(growable: true);

    feedback.add(feedbackEntry);
    payload['feedback'] = feedback;
    payload['updated_at'] = DateTime.now().toIso8601String();

    final file = File(filePath);
    await file.writeAsString(
      const JsonEncoder.withIndent('  ').convert(_stripNulls(payload)),
    );
  }

  Future<Map<String, dynamic>> buildInferencePayloadFromFile({
    required String filePath,
    bool includeHarWindows = false,
    bool includeFallWindows = false,
    bool includeCombinedTimeline = true,
    bool includeGroupedFallEvents = true,
  }) async {
    final payload = await loadSessionPayload(filePath);
    final samples = _asList(payload['samples']);

    if (samples.isEmpty) {
      throw StateError('Saved session contains no valid samples.');
    }

    _log(
      'Building inference payload from $filePath with ${samples.length} sample(s)',
    );

    final sessionId =
        _asString(payload['session_id']) ??
        File(filePath).uri.pathSegments.last.replaceAll('.json', '');

    final subjectId = _asString(payload['subject_id']) ?? 'anonymous_user';
    final placement = _normalisePlacement(
      _asString(payload['placement']) ?? 'unknown',
    );
    final testId = _resolvedSessionMetadata(payload, 'test_id');
    final testTitle = _resolvedSessionMetadata(payload, 'test_title');

    final metadata = <String, dynamic>{
      'session_id': sessionId,
      'subject_id': subjectId,
      'placement': placement,
      'task_type': _asString(payload['task_type']) ?? 'runtime',
      'dataset_name': _asString(payload['dataset_name']) ?? 'APP_RUNTIME_SAVED',
      'source_type': _asString(payload['source_type']) ?? 'mobile_app',
      'device_platform': _asString(payload['device_platform']) ?? 'unknown',
      'device_model': _asString(payload['device_model']),
      'sampling_rate_hz': _asDouble(payload['sampling_rate_hz']),
      'notes': _asString(payload['notes']),
      'app_version': _asString(payload['app_version']),
      'app_build': _asString(payload['app_build']),
      'recording_mode':
          _asString(payload['recording_mode']) ?? 'import_session',
      'runtime_mode': _asString(payload['runtime_mode']) ?? 'session_replay',
      'test_id': testId,
      'test_title': testTitle,
    };

    return <String, dynamic>{
      'metadata': _stripNulls(metadata),
      'samples': samples,
      'include_har_windows': includeHarWindows,
      'include_fall_windows': includeFallWindows,
      'include_combined_timeline': includeCombinedTimeline,
      'include_grouped_fall_events': includeGroupedFallEvents,
      'include_point_timeline': true,
      'include_timeline_events': true,
      'include_transition_events': true,
    };
  }

  Future<void> deleteSession(String filePath) async {
    final file = File(filePath);
    if (await file.exists()) {
      await file.delete();
    }
  }

  String _buildReadableFileName({
    required DateTime timestamp,
    String? testId,
    String? testTitle,
  }) {
    final slug = _testSlug(testTitle: testTitle, testId: testId);
    return 'session_${slug}_${timestamp.millisecondsSinceEpoch}.json';
  }

  Future<File> _uniqueFile(Directory dir, String fileName) async {
    var candidate = File('${dir.path}/$fileName');
    if (!await candidate.exists()) {
      return candidate;
    }

    final stem = fileName.toLowerCase().endsWith('.json')
        ? fileName.substring(0, fileName.length - 5)
        : fileName;

    var index = 2;
    while (await candidate.exists()) {
      candidate = File('${dir.path}/${stem}_$index.json');
      index += 1;
    }
    return candidate;
  }

  String _testSlug({String? testTitle, String? testId}) {
    final cleanedTitle = _cleanOptionalValue(testTitle);
    if (cleanedTitle != null) {
      final withoutPrefix = cleanedTitle.replaceFirst(
        RegExp(r'^\d+\s*[.):_/-]*\s*'),
        '',
      );
      final withoutSuffix = withoutPrefix.replaceFirst(
        RegExp(r'\s+test$', caseSensitive: false),
        '',
      );
      final slug = _slug(withoutSuffix);
      if (slug != 'unknown') {
        return slug;
      }
    }

    final cleanedId = _cleanOptionalValue(testId);
    if (cleanedId != null) {
      final slug = _slug(cleanedId);
      if (slug != 'unknown') {
        return slug;
      }
    }

    return 'unspecified';
  }

  String? _resolvedSessionMetadata(
    Map<String, dynamic> payload,
    String fieldName,
  ) {
    final directValue = _cleanOptionalValue(_asString(payload[fieldName]));
    if (directValue != null) {
      return directValue;
    }

    return _metadataValueFromNotes(_asString(payload['notes']), fieldName);
  }

  String? _metadataValueFromNotes(String? notes, String fieldName) {
    final trimmedNotes = _cleanOptionalValue(notes);
    if (trimmedNotes == null) {
      return null;
    }

    for (final part in trimmedNotes.split(' | ')) {
      if (part.startsWith('$fieldName=')) {
        return _cleanOptionalValue(part.substring(fieldName.length + 1));
      }
    }

    return null;
  }

  String? _cleanOptionalValue(String? value) {
    final trimmed = value?.trim();
    if (trimmed == null || trimmed.isEmpty) {
      return null;
    }
    return trimmed;
  }

  String _slug(String value) {
    final normalised = value
        .trim()
        .toLowerCase()
        .replaceAll(RegExp(r'[^a-z0-9]+'), '_')
        .replaceAll(RegExp(r'_+'), '_')
        .replaceAll(RegExp(r'^_+|_+$'), '');

    return normalised.isEmpty ? 'unknown' : normalised;
  }

  String _normalisePlacement(String value) {
    switch (value.trim().toLowerCase()) {
      case 'pocket':
        return 'pocket';
      case 'hand':
      case 'arm_mounted':
        return 'hand';
      case 'desk':
      case 'on_surface':
        return 'desk';
      case 'bag':
        return 'bag';
      default:
        return 'unknown';
    }
  }

  Map<String, dynamic> _decodeJsonObject(String raw, String filePath) {
    final dynamic decoded;
    try {
      decoded = jsonDecode(raw);
    } on FormatException catch (error) {
      throw FormatException(
        'Invalid JSON in saved session "$filePath": ${error.message}',
      );
    }

    if (decoded is Map<String, dynamic>) {
      return decoded;
    }
    if (decoded is Map) {
      return decoded.map((key, value) => MapEntry(key.toString(), value));
    }

    throw FormatException(
      'Saved session payload is not a JSON object: $filePath',
    );
  }

  Map<String, dynamic> _normaliseLoadedPayload(
    Map<String, dynamic> payload, {
    required String filePath,
  }) {
    final normalised = Map<String, dynamic>.from(payload);
    final samples = _sanitiseSampleMaps(
      normalised['samples'],
      filePath: filePath,
    );
    final feedback = _sanitiseMapList(
      normalised['feedback'],
      fieldName: 'feedback',
      filePath: filePath,
    );

    final declaredSampleCount = _asInt(normalised['sample_count']);
    if (declaredSampleCount == null || declaredSampleCount != samples.length) {
      _log(
        'Normalised sample_count for $filePath from '
        '${declaredSampleCount ?? 'null'} to ${samples.length}',
      );
    }

    normalised['subject_id'] = _asString(normalised['subject_id']) ?? 'unknown';
    normalised['placement'] = _asString(normalised['placement']) ?? 'unknown';
    normalised['samples'] = samples;
    normalised['sample_count'] = samples.length;
    normalised['feedback'] = feedback;

    final inferenceResult = normalised['inference_result'];
    if (inferenceResult == null) {
      return normalised;
    }

    if (inferenceResult is Map<String, dynamic>) {
      return normalised;
    }
    if (inferenceResult is Map) {
      normalised['inference_result'] = inferenceResult.map(
        (key, value) => MapEntry(key.toString(), value),
      );
      return normalised;
    }

    _log(
      'Dropping malformed inference_result from $filePath because it is not '
      'a JSON object',
    );
    normalised.remove('inference_result');
    return normalised;
  }

  List<Map<String, dynamic>> _sanitiseSampleMaps(
    dynamic value, {
    required String filePath,
  }) {
    if (value == null) {
      _log('Saved session $filePath is missing a samples list');
      return const <Map<String, dynamic>>[];
    }
    if (value is! List) {
      _log('Saved session $filePath has a non-list samples field');
      return const <Map<String, dynamic>>[];
    }

    final samples = <Map<String, dynamic>>[];
    for (var i = 0; i < value.length; i++) {
      final sample = SensorSample.tryFromJson(
        value[i],
        onInvalid: (message) {
          _log('Skipping invalid sample #$i in $filePath: $message');
        },
      );
      if (sample != null) {
        samples.add(sample.toJson());
      }
    }

    if (samples.length != value.length) {
      _log(
        'Kept ${samples.length}/${value.length} valid samples from $filePath',
      );
    }

    return samples;
  }

  List<Map<String, dynamic>> _sanitiseMapList(
    dynamic value, {
    required String fieldName,
    required String filePath,
  }) {
    if (value == null) {
      return const <Map<String, dynamic>>[];
    }
    if (value is! List) {
      _log('Saved session $filePath has a non-list $fieldName field');
      return const <Map<String, dynamic>>[];
    }

    final items = <Map<String, dynamic>>[];
    for (var i = 0; i < value.length; i++) {
      final item = value[i];
      if (item is! Map) {
        _log('Skipping invalid $fieldName entry #$i in $filePath');
        continue;
      }

      items.add(item.map((key, value) => MapEntry(key.toString(), value)));
    }

    return items;
  }
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
  if (value is num) {
    return value.toDouble();
  }
  return double.tryParse(value.toString());
}

dynamic _stripNulls(dynamic value) {
  if (value is Map) {
    final result = <String, dynamic>{};
    value.forEach((key, item) {
      if (item == null) {
        return;
      }
      final cleaned = _stripNulls(item);
      if (cleaned != null) {
        result[key.toString()] = cleaned;
      }
    });
    return result;
  }

  if (value is List) {
    return value
        .map(_stripNulls)
        .where((item) => item != null)
        .toList(growable: false);
  }

  return value;
}

String _csvCell(dynamic value) {
  if (value == null) {
    return '';
  }

  final text = value.toString();
  final escaped = text.replaceAll('"', '""');

  if (escaped.contains(',') ||
      escaped.contains('"') ||
      escaped.contains('\n') ||
      escaped.contains('\r')) {
    return '"$escaped"';
  }

  return escaped;
}
