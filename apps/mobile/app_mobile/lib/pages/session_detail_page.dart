import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../config/runtime_config.dart';
import '../config/runtime_labels.dart';
import '../models/api_result_summary.dart';
import '../models/persisted_session.dart';
import '../models/saved_session.dart';
import '../models/sensor_sample.dart';
import '../services/runtime_api_service.dart';
import '../services/runtime_identity_service.dart';
import '../services/session_storage_service.dart';
import 'saved_sessions_page.dart';

class SessionDetailPage extends StatefulWidget {
  const SessionDetailPage({super.key, required this.session});

  final SavedSession session;

  @override
  State<SessionDetailPage> createState() => _SessionDetailPageState();
}

class _SessionDetailPageState extends State<SessionDetailPage> {
  static const Color _accent = Color(0xFF2C8A66);
  static const Color _pageBackground = Color(0xFFF4F1E9);
  static const Color _cardBackground = Color(0xFFFFFFFF);
  static const Color _softBackground = Color(0xFFEDE9DF);
  static const Color _border = Color(0xFFE5E1D4);
  static const Color _textPrimary = Color(0xFF141713);
  static const Color _textSecondary = Color(0xFF5A5E58);
  static const Color _textTertiary = Color(0xFF8E918A);
  static const Color _sageDeep = Color(0xFF1A5A44);
  static const Color _sageSoft = Color(0xFFDCEBE3);
  static const Color _danger = Color(0xFFC14C41);
  static const Color _warning = Color(0xFFC29A20);
  static const Color _sky = Color(0xFF6F9BB8);

  final SessionStorageService _storage = SessionStorageService();
  RuntimeApiService? _api;

  static const List<String> _activityOptions = runtimeAnnotationActivityOptions;

  static const List<String> _placementOptions = [
    'unknown',
    'pocket',
    'hand',
    'desk',
    'bag',
    'arm_mounted',
    'on_surface',
    'repositioning',
  ];

  bool _loading = true;
  bool _saving = false;
  bool _sending = false;
  bool _submittingFeedback = false;
  Object? _bootstrapError;

  Map<String, dynamic>? _payload;
  ApiResultSummary? _result;
  String _status = 'Idle';

  late String _activityLabel;
  late String _placementLabel;
  late TextEditingController _notesController;

  @override
  void initState() {
    super.initState();
    _activityLabel = normaliseRuntimeActivityLabel(
      widget.session.activityLabel,
      fallback: 'unknown',
    );
    _placementLabel = widget.session.placementLabel ?? 'unknown';
    _notesController = TextEditingController(text: widget.session.notes ?? '');
    _bootstrapAndLoad();
  }

  @override
  void dispose() {
    _api?.dispose();
    _notesController.dispose();
    super.dispose();
  }

  Future<void> _bootstrapAndLoad() async {
    try {
      final identity = await RuntimeIdentityService.instance.ensureIdentity();
      _api?.dispose();
      _api = RuntimeApiService(
        baseUrl: runtimeApiBaseUrl,
        basicAuthUsername: identity.username,
        basicAuthPassword: identity.password,
      );
      _bootstrapError = null;
    } catch (e) {
      _bootstrapError = e;
    }

    await _load();
  }

  Future<void> _load() async {
    final api = _api;
    Map<String, dynamic>? localPayload;
    ApiResultSummary? localResult;
    Object? localError;

    if (widget.session.hasLocalFile) {
      try {
        localPayload = await _storage.loadSessionPayload(
          widget.session.filePath!,
        );

        final savedInference = localPayload['inference_result'];
        if (savedInference is Map) {
          localResult = ApiResultSummary.fromJson(
            savedInference.map((key, value) => MapEntry(key.toString(), value)),
          );
        }
      } catch (e) {
        localError = e;
      }
    }

    PersistedSessionDetail? persistedDetail;
    Object? remoteError;
    final persistedSessionId = _currentPersistedSessionId(
      localResult: localResult,
    );
    if (persistedSessionId != null && api != null) {
      try {
        persistedDetail = await api.fetchPersistedSessionDetail(
          persistedSessionId,
        );
      } catch (e) {
        remoteError = e;
      }
    } else if (persistedSessionId != null) {
      remoteError =
          _bootstrapError ?? StateError('Your account is not ready yet.');
    }

    final mergedPayload = _mergePayloadWithPersistedDetail(
      localPayload: localPayload,
      persistedDetail: persistedDetail,
      fallbackResult: localResult,
    );
    final mergedResult =
        persistedDetail?.latestInference?.response ?? localResult;

    final localActivityLabel = _stringOrNull(mergedPayload?['activity_label']);
    final localPlacementLabel = _stringOrNull(
      mergedPayload?['placement_label'],
    );
    final localNotes = _stringOrNull(mergedPayload?['notes']);

    if (!mounted) return;
    setState(() {
      _payload = mergedPayload;
      _result = mergedResult;
      if (localActivityLabel != null && localActivityLabel.isNotEmpty) {
        _activityLabel = normaliseRuntimeActivityLabel(
          localActivityLabel,
          fallback: 'unknown',
        );
      }
      if (localPlacementLabel != null && localPlacementLabel.isNotEmpty) {
        _placementLabel = localPlacementLabel;
      }
      if (localNotes != null) {
        _notesController.text = localNotes;
      }
      _loading = false;
      _status = _statusForLoadedSession(
        localPayload: localPayload,
        persistedDetail: persistedDetail,
        localError: localError,
        remoteError: remoteError,
        mergedResult: mergedResult,
      );
    });
  }

  String? _currentPersistedSessionId({ApiResultSummary? localResult}) {
    final fromResult = _result?.persistedSessionId;
    if (fromResult != null && fromResult.trim().isNotEmpty) {
      return fromResult;
    }

    final fromLocalResult = localResult?.persistedSessionId;
    if (fromLocalResult != null && fromLocalResult.trim().isNotEmpty) {
      return fromLocalResult;
    }

    final fromSession = widget.session.persistedSessionId;
    if (fromSession != null && fromSession.trim().isNotEmpty) {
      return fromSession;
    }

    return null;
  }

  Map<String, dynamic>? _mergePayloadWithPersistedDetail({
    required Map<String, dynamic>? localPayload,
    required PersistedSessionDetail? persistedDetail,
    required ApiResultSummary? fallbackResult,
  }) {
    if (localPayload == null && persistedDetail == null) {
      return null;
    }

    final merged = <String, dynamic>{...(localPayload ?? <String, dynamic>{})};
    if (persistedDetail == null) {
      return merged;
    }

    merged['session_id'] = persistedDetail.session.clientSessionId;
    merged['subject_id'] = persistedDetail.session.subjectId;
    merged['placement'] = persistedDetail.session.placementDeclared;
    merged['session_name'] =
        persistedDetail.session.sessionName ?? widget.session.fileName;
    if (_stringOrNull(merged['activity_label']) == null &&
        persistedDetail.session.activityLabel != null) {
      merged['activity_label'] = persistedDetail.session.activityLabel;
    }
    merged['sample_count'] = persistedDetail.session.sampleCount;
    merged['saved_at'] = persistedDetail.session.uploadedAt
        .toUtc()
        .toIso8601String();
    merged['updated_at'] = persistedDetail.session.updatedAt
        .toUtc()
        .toIso8601String();
    merged['persisted_session_id'] = persistedDetail.session.appSessionId;

    final remoteFeedback = persistedDetail.feedback
        .map((entry) => entry.toJson())
        .toList(growable: false);
    if (remoteFeedback.isNotEmpty) {
      merged['feedback'] = remoteFeedback;
    } else if (!merged.containsKey('feedback')) {
      merged['feedback'] = const <Map<String, dynamic>>[];
    }

    final remoteResult = persistedDetail.latestInference?.response;
    if (remoteResult != null) {
      merged['inference_result'] = remoteResult.rawJson;
      merged['inference_saved_at'] = DateTime.now().toUtc().toIso8601String();
    } else if (fallbackResult != null &&
        !merged.containsKey('inference_result')) {
      merged['inference_result'] = fallbackResult.rawJson;
    }

    return merged;
  }

  String _statusForLoadedSession({
    required Map<String, dynamic>? localPayload,
    required PersistedSessionDetail? persistedDetail,
    required Object? localError,
    required Object? remoteError,
    required ApiResultSummary? mergedResult,
  }) {
    if (localError != null && remoteError != null) {
      return 'Failed to load local session ($localError) and remote session detail ($remoteError).';
    }
    if (localError != null) {
      return 'Local session file unavailable: $localError';
    }
    if (remoteError != null && mergedResult != null) {
      return 'Loaded local session, but remote refresh failed: $remoteError';
    }
    if (remoteError != null) {
      return 'Failed to load remote session detail: $remoteError';
    }
    if (persistedDetail != null && widget.session.hasLocalFile) {
      return 'Session loaded from local file and synced server detail.';
    }
    if (persistedDetail != null) {
      return 'Session loaded from server.';
    }
    if (localPayload != null && mergedResult != null) {
      return 'Session loaded with saved inference result.';
    }
    if (localPayload != null) {
      return 'Session loaded.';
    }
    return 'No session data available.';
  }

  String? _stringOrNull(dynamic value) {
    if (value == null) {
      return null;
    }
    final text = value.toString().trim();
    return text.isEmpty ? null : text;
  }

  Future<void> _saveLabels() async {
    if (!widget.session.hasLocalFile) {
      setState(() {
        _status =
            'This session has no local file, so labels cannot be saved on-device.';
      });
      return;
    }

    setState(() {
      _saving = true;
      _status = 'Saving labels...';
    });

    try {
      await _storage.updateSessionLabels(
        filePath: widget.session.filePath!,
        activityLabel: _activityLabel,
        placementLabel: _placementLabel,
        notes: _notesController.text.trim(),
      );

      if (!mounted) return;
      setState(() {
        _saving = false;
        _status = 'Labels saved';
      });

      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('Session labels saved')));
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _saving = false;
        _status = 'Failed to save labels: $e';
      });
    }
  }

  Future<void> _sendSavedSessionToServer() async {
    final api = _api;
    if (api == null) {
      setState(() {
        _status =
            'Your account is still being prepared. Try again in a moment.';
      });
      return;
    }

    if (!widget.session.hasLocalFile) {
      final persistedSessionId = _currentPersistedSessionId();
      if (persistedSessionId == null) {
        setState(() {
          _status = 'This session has no local payload to replay.';
        });
        return;
      }

      setState(() {
        _sending = true;
        _status = 'Refreshing session from server...';
      });

      try {
        final persistedDetail = await api.fetchPersistedSessionDetail(
          persistedSessionId,
        );
        if (!mounted) return;
        setState(() {
          _payload = _mergePayloadWithPersistedDetail(
            localPayload: _payload,
            persistedDetail: persistedDetail,
            fallbackResult: _result,
          );
          _result = persistedDetail.latestInference?.response ?? _result;
          _status = 'Session refreshed from server.';
        });
      } catch (e) {
        if (!mounted) return;
        setState(() {
          _status = 'Failed to refresh session from server: $e';
        });
      } finally {
        if (mounted) {
          setState(() {
            _sending = false;
          });
        }
      }
      return;
    }

    if (_payload == null) {
      setState(() {
        _status = 'No session payload loaded.';
      });
      return;
    }

    final samplesRaw = _payload!['samples'] as List?;
    if (samplesRaw == null || samplesRaw.isEmpty) {
      setState(() {
        _status = 'This saved session has no samples.';
      });
      return;
    }

    setState(() {
      _sending = true;
      _status = 'Sending saved session to server...';
    });

    try {
      final runtimePayload = await _storage.buildInferencePayloadFromFile(
        filePath: widget.session.filePath!,
        includeHarWindows: false,
        includeFallWindows: false,
        includeCombinedTimeline: true,
        includeGroupedFallEvents: true,
      );

      final metadata =
          (runtimePayload['metadata'] as Map?)?.cast<String, dynamic>() ??
          <String, dynamic>{};

      final rawSamples =
          runtimePayload['samples'] as List<dynamic>? ?? <dynamic>[];
      final samples = rawSamples
          .whereType<Map>()
          .map((item) => item.cast<String, dynamic>())
          .map(SensorSample.fromJson)
          .toList(growable: false);

      if (samples.length < 32) {
        throw StateError('Saved session contains fewer than 32 samples.');
      }

      final summary = await api.submitSession(
        sessionId:
            (metadata['session_id'] as String?) ??
            widget.session.fileName.replaceAll('.json', ''),
        sessionName:
            (metadata['session_name'] as String?) ??
            (metadata['file_name'] as String?) ??
            widget.session.fileName,
        activityLabel: _activityLabel != 'unknown'
            ? _activityLabel
            : (metadata['activity_label'] as String?),
        subjectId:
            (metadata['subject_id'] as String?) ?? widget.session.subjectId,
        placement: _normalisePlacementForRuntime(
          (_placementLabel != 'unknown'
                  ? _placementLabel
                  : (metadata['placement'] as String?) ??
                        widget.session.placement)
              .toString(),
        ),
        taskType: (metadata['task_type'] as String?) ?? 'runtime',
        datasetName:
            (metadata['dataset_name'] as String?) ?? 'APP_RUNTIME_SAVED',
        sourceType: (metadata['source_type'] as String?) ?? 'mobile_app',
        devicePlatform: (metadata['device_platform'] as String?) ?? 'unknown',
        deviceModel: metadata['device_model'] as String?,
        appVersion: metadata['app_version'] as String?,
        appBuild: metadata['app_build'] as String?,
        recordingMode: (metadata['recording_mode'] as String?) ?? 'import',
        runtimeMode: (metadata['runtime_mode'] as String?) ?? 'mobile_live',
        samplingRateHz: _asDouble(metadata['sampling_rate_hz']),
        notes: _notesController.text.trim().isEmpty
            ? (metadata['notes'] as String?)
            : _notesController.text.trim(),
        samples: samples,
        includeHarWindows:
            runtimePayload['include_har_windows'] as bool? ?? false,
        includeFallWindows:
            runtimePayload['include_fall_windows'] as bool? ?? false,
        includeCombinedTimeline:
            runtimePayload['include_combined_timeline'] as bool? ?? true,
        includeGroupedFallEvents:
            runtimePayload['include_grouped_fall_events'] as bool? ?? true,
        includePointTimeline: true,
        includeTimelineEvents: true,
        includeTransitionEvents: true,
      );

      await _storage.saveInferenceResult(
        filePath: widget.session.filePath!,
        inferenceResult: summary.rawJson,
      );

      if (!mounted) return;
      setState(() {
        _result = summary;
        _payload = <String, dynamic>{
          ...?_payload,
          'persisted_session_id': summary.persistedSessionId,
          'inference_result': summary.rawJson,
        };
        _status = 'Saved session inference complete.';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _status = 'Saved session inference failed: $e';
      });
    } finally {
      if (mounted) {
        setState(() {
          _sending = false;
        });
      }
    }
  }

  Future<void> _submitFeedback(String feedback) async {
    final api = _api;
    if (api == null) {
      setState(() {
        _status =
            'Your account is still being prepared. Try again in a moment.';
      });
      return;
    }
    final result = _result;
    if (result == null) {
      setState(() {
        _status = 'No inference result available yet.';
      });
      return;
    }

    setState(() {
      _submittingFeedback = true;
      _status = 'Submitting feedback...';
    });

    try {
      final ack = await api.submitFeedback(
        sessionId: result.sessionId,
        subjectId: widget.session.subjectId,
        persistedSessionId: result.persistedSessionId,
        persistedInferenceId: result.persistedInferenceId,
        targetType: 'session',
        userFeedback: feedback,
        notes: _notesController.text.trim().isEmpty
            ? null
            : _notesController.text.trim(),
      );

      final feedbackEntry = <String, dynamic>{
        'request_id': ack.requestId,
        'recorded_at': ack.recordedAt?.toUtc().toIso8601String(),
        'status': ack.status,
        'session_id': ack.sessionId,
        'persisted_session_id': ack.persistedSessionId,
        'persisted_inference_id': ack.persistedInferenceId,
        'persisted_feedback_id': ack.persistedFeedbackId,
        'target_type': ack.targetType,
        'event_id': ack.eventId,
        'window_id': ack.windowId,
        'user_feedback': ack.userFeedback,
        'message': ack.message,
      };

      if (widget.session.hasLocalFile) {
        await _storage.appendFeedback(
          filePath: widget.session.filePath!,
          feedbackEntry: feedbackEntry,
        );
      }

      PersistedSessionDetail? persistedDetail;
      final persistedSessionId =
          ack.persistedSessionId ??
          result.persistedSessionId ??
          _currentPersistedSessionId();
      if (persistedSessionId != null && persistedSessionId.trim().isNotEmpty) {
        try {
          persistedDetail = await api.fetchPersistedSessionDetail(
            persistedSessionId,
          );
        } catch (_) {
          persistedDetail = null;
        }
      }

      if (!mounted) return;
      setState(() {
        final feedback = _feedbackEntries().toList(growable: true);
        feedback.add(feedbackEntry);
        _payload = _mergePayloadWithPersistedDetail(
          localPayload: <String, dynamic>{...?_payload, 'feedback': feedback},
          persistedDetail: persistedDetail,
          fallbackResult: _result,
        );
        _result = persistedDetail?.latestInference?.response ?? _result;
        _status = ack.message;
      });

      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text(ack.message)));
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _status = 'Feedback submission failed: $e';
      });
    } finally {
      if (mounted) {
        setState(() {
          _submittingFeedback = false;
        });
      }
    }
  }

  Future<void> _openSavedSessions() async {
    if (!mounted) return;

    await Navigator.of(context).pushAndRemoveUntil(
      MaterialPageRoute(
        builder: (_) =>
            SavedSessionsPage(initialSubjectId: widget.session.subjectId),
      ),
      (route) => route.isFirst,
    );
  }

  String _normalisePlacementForRuntime(String value) {
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
      case 'repositioning':
      case 'unknown':
      default:
        return 'unknown';
    }
  }

  int _effectiveFirstSampleIndex() {
    final samples = _payload?['samples'] as List?;
    if (samples == null || samples.length < 3) return 0;
    final s0 = samples[0];
    final s1 = samples[1];
    final s2 = samples[2];
    if (s0 is! Map || s1 is! Map || s2 is! Map) return 0;
    final t0 = _asDouble(s0['timestamp']);
    final t1 = _asDouble(s1['timestamp']);
    final t2 = _asDouble(s2['timestamp']);
    if (t0 == null || t1 == null || t2 == null) return 0;
    if (t0 >= 0.05) return 0;
    if (t1 - t0 <= 1.0) return 0;
    if (t2 - t1 >= 0.5) return 0;
    return 1;
  }

  double? _firstTimestamp() {
    final samples = _payload?['samples'] as List?;
    if (samples == null || samples.isEmpty) return null;
    final idx = _effectiveFirstSampleIndex();
    if (idx >= samples.length) return null;
    final first = samples[idx];
    if (first is! Map) return null;
    return _asDouble(first['timestamp']);
  }

  double? _lastTimestamp() {
    final samples = _payload?['samples'] as List?;
    if (samples == null || samples.isEmpty) return null;
    final last = samples.last;
    if (last is! Map) return null;
    return _asDouble(last['timestamp']);
  }

  double? _precisionFromFeedback(List<Map<String, dynamic>> entries) {
    var confirmed = 0;
    var falseAlarm = 0;
    for (final entry in entries) {
      final type = (entry['user_feedback'] ?? '').toString();
      if (type == 'confirmed_fall') confirmed += 1;
      if (type == 'false_alarm') falseAlarm += 1;
    }
    final denom = confirmed + falseAlarm;
    if (denom == 0) return null;
    return confirmed / denom;
  }

  Widget _buildFeedbackTile(Map<String, dynamic> entry) {
    final type = (entry['user_feedback'] ?? 'unknown').toString();
    final recordedAt = entry['recorded_at']?.toString();
    DateTime? when;
    if (recordedAt != null) {
      when = DateTime.tryParse(recordedAt);
    }
    final whenText = when == null
        ? '—'
        : '${when.toLocal().year}-${when.toLocal().month.toString().padLeft(2, '0')}-${when.toLocal().day.toString().padLeft(2, '0')} '
            '${when.toLocal().hour.toString().padLeft(2, '0')}:${when.toLocal().minute.toString().padLeft(2, '0')}';
    final color = type == 'confirmed_fall'
        ? _danger
        : type == 'false_alarm'
            ? _accent
            : _warning;
    final icon = type == 'confirmed_fall'
        ? Icons.priority_high_rounded
        : type == 'false_alarm'
            ? Icons.cancel_outlined
            : Icons.help_outline_rounded;

    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        children: [
          Container(
            width: 28,
            height: 28,
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.12),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Icon(icon, size: 16, color: color),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              _titleCase(type),
              style: GoogleFonts.interTight(
                fontSize: 13,
                fontWeight: FontWeight.w600,
                color: _textPrimary,
              ),
            ),
          ),
          Text(
            whenText,
            style: GoogleFonts.jetBrainsMono(
              fontSize: 11,
              color: _textSecondary,
            ),
          ),
        ],
      ),
    );
  }

  List<Map<String, dynamic>> _feedbackEntries() {
    final payload = _payload;
    if (payload == null) {
      return const <Map<String, dynamic>>[];
    }

    final raw = payload['feedback'];
    if (raw is! List) {
      return const <Map<String, dynamic>>[];
    }

    return raw
        .whereType<Map>()
        .map(
          (item) => item.map((key, value) => MapEntry(key.toString(), value)),
        )
        .toList(growable: false);
  }

  String _fmt(DateTime dt) {
    return '${dt.year}-${dt.month.toString().padLeft(2, '0')}-${dt.day.toString().padLeft(2, '0')} '
        '${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}';
  }

  String _formatSeconds(double seconds) {
    final total = seconds.round();
    final mins = total ~/ 60;
    final secs = total % 60;
    if (mins <= 0) {
      return '${secs}s';
    }
    return '${mins}m ${secs.toString().padLeft(2, '0')}s';
  }

  String _dateLabel(DateTime value) {
    const months = [
      'Jan',
      'Feb',
      'Mar',
      'Apr',
      'May',
      'Jun',
      'Jul',
      'Aug',
      'Sep',
      'Oct',
      'Nov',
      'Dec',
    ];
    final local = value.toLocal();
    final month = months[local.month - 1];
    final hour = local.hour.toString().padLeft(2, '0');
    final minute = local.minute.toString().padLeft(2, '0');
    return '$month ${local.day}, ${local.year} · $hour:$minute';
  }

  String _titleCase(String value) {
    return value
        .trim()
        .replaceAll('_', ' ')
        .split(RegExp(r'\s+'))
        .where((part) => part.isNotEmpty)
        .map((part) => '${part[0].toUpperCase()}${part.substring(1)}')
        .join(' ');
  }

  List<SensorSample> _sessionSamples() {
    final raw = _payload?['samples'];
    if (raw is! List) {
      return const <SensorSample>[];
    }

    final samples = <SensorSample>[];
    for (final item in raw) {
      if (item is! Map) {
        continue;
      }
      try {
        samples.add(
          SensorSample.fromJson(
            item.map((key, value) => MapEntry(key.toString(), value)),
          ),
        );
      } catch (_) {
        // Ignore malformed samples in the detail summary.
      }
    }
    return samples;
  }

  double _sessionDurationSeconds() {
    final narrativeDuration = _result?.narrativeSummary?.totalDurationSeconds;
    if (narrativeDuration != null && narrativeDuration > 0) {
      return narrativeDuration;
    }

    final fromPayload = _asDouble(_payload?['duration_seconds']);
    if (fromPayload != null && fromPayload > 0) {
      return fromPayload;
    }

    final sampleCount = _asDouble(_payload?['sample_count']);
    final samplingRate = _samplingRateHz() ?? 50.0;
    if (sampleCount != null && sampleCount > 0 && samplingRate > 0) {
      return sampleCount / samplingRate;
    }

    final firstTs = _firstTimestamp();
    final lastTs = _lastTimestamp();
    if (firstTs != null && lastTs != null && lastTs > firstTs) {
      return lastTs - firstTs;
    }

    return 60;
  }

  double? _samplingRateHz() {
    final fromPayload = _asDouble(_payload?['sampling_rate_hz']);
    if (fromPayload != null && fromPayload > 0) {
      return fromPayload;
    }

    final samples = _sessionSamples();
    final startIdx = _effectiveFirstSampleIndex();
    if (samples.length - startIdx < 2) {
      return null;
    }

    final duration = samples.last.timestamp - samples[startIdx].timestamp;
    if (duration <= 0) {
      return null;
    }
    return (samples.length - startIdx - 1) / duration;
  }

  double? _gMax() {
    final samples = _sessionSamples();
    if (samples.isEmpty) {
      return null;
    }

    var maxMagnitude = 0.0;
    for (final sample in samples) {
      final magnitude = math.sqrt(
        sample.ax * sample.ax + sample.ay * sample.ay + sample.az * sample.az,
      );
      maxMagnitude = math.max(maxMagnitude, magnitude);
    }

    // Mobile accelerometer samples are commonly m/s^2. Convert to g for display.
    return maxMagnitude / 9.80665;
  }

  int _gapCount() {
    final samples = _sessionSamples();
    if (samples.length < 2) {
      return 0;
    }

    final startIdx = _effectiveFirstSampleIndex();
    var gaps = 0;
    for (var i = startIdx + 1; i < samples.length; i++) {
      final delta = samples[i].timestamp - samples[i - 1].timestamp;
      if (delta > 0.25) {
        gaps++;
      }
    }
    return gaps;
  }

  String _narrativeHeadline() {
    final result = _result;
    if (result?.likelyFallDetected == true) {
      return 'A moment worth reviewing.';
    }
    return 'A steady session.';
  }

  String _narrativeSubline() {
    final summary = _result?.narrativeSummary;
    if (summary != null && summary.summaryText.trim().isNotEmpty) {
      return summary.summaryText.trim();
    }

    final activity = _titleCase(
      _result?.topHarLabel ?? widget.session.activityLabel ?? _activityLabel,
    );
    final placement = _titleCase(widget.session.placement);
    return '$activity recorded from $placement.';
  }

  List<TimelineEventModel> _keyEvents() {
    final events = _result?.timelineEvents ?? const <TimelineEventModel>[];
    if (events.isEmpty) {
      return const <TimelineEventModel>[];
    }

    final sorted = [...events]
      ..sort((a, b) {
        if (a.likelyFall != b.likelyFall) {
          return a.likelyFall ? -1 : 1;
        }
        return a.startTs.compareTo(b.startTs);
      });
    return sorted.take(4).toList(growable: false);
  }

  List<_TimelineSegment> _timelineSegments() {
    final result = _result;
    final duration = _sessionDurationSeconds();
    if (result == null || result.timelineEvents.isEmpty) {
      return [
        _TimelineSegment(
          startFraction: 0,
          endFraction: 1,
          label: _titleCase(_activityLabel),
          color: _accent,
        ),
      ];
    }

    final colors = <Color>[_accent, _warning, _sky, _border];
    final colorByLabel = <String, Color>{};
    var colorIndex = 0;

    return result.timelineEvents
        .map((event) {
          final label = _titleCase(event.activityLabel);
          colorByLabel.putIfAbsent(label, () {
            final color = event.likelyFall
                ? _danger
                : colors[colorIndex++ % colors.length];
            return color;
          });

          return _TimelineSegment(
            startFraction: (event.startTs / duration).clamp(0.0, 1.0),
            endFraction: (event.endTs / duration).clamp(0.0, 1.0),
            label: label,
            color: colorByLabel[label]!,
          );
        })
        .toList(growable: false);
  }

  List<_TimelineLegendItem> _timelineLegend() {
    final seen = <String>{};
    final items = <_TimelineLegendItem>[];
    for (final segment in _timelineSegments()) {
      if (seen.add(segment.label)) {
        items.add(
          _TimelineLegendItem(label: segment.label, color: segment.color),
        );
      }
      if (items.length == 4) {
        break;
      }
    }
    return items;
  }

  Widget _card({
    required Widget child,
    EdgeInsets padding = const EdgeInsets.all(20),
  }) {
    return Container(
      decoration: BoxDecoration(
        color: _cardBackground,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: _border),
        boxShadow: const [
          BoxShadow(
            color: Color(0x0A141713),
            blurRadius: 2,
            offset: Offset(0, 1),
          ),
          BoxShadow(
            color: Color(0x0A141713),
            blurRadius: 16,
            offset: Offset(0, 4),
          ),
        ],
      ),
      child: Padding(padding: padding, child: child),
    );
  }

  Widget _sectionTitle(String title, String subtitle) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: GoogleFonts.instrumentSerif(
              fontSize: 28,
              height: 1.08,
              fontWeight: FontWeight.w400,
              letterSpacing: -0.6,
              color: _textPrimary,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            subtitle,
            style: GoogleFonts.interTight(
              fontSize: 15,
              color: _textSecondary,
              height: 1.35,
              fontWeight: FontWeight.w400,
            ),
          ),
        ],
      ),
    );
  }

  Widget _metricBox({
    required String label,
    required String value,
    required IconData icon,
  }) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: _softBackground,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: _border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 18, color: _textSecondary),
          const SizedBox(height: 14),
          Text(
            label.toUpperCase(),
            style: GoogleFonts.interTight(
              fontSize: 10.5,
              height: 1.2,
              fontWeight: FontWeight.w600,
              letterSpacing: 0.9,
              color: _textTertiary,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            value,
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
            style: GoogleFonts.jetBrainsMono(
              fontSize: 22,
              height: 1.0,
              fontWeight: FontWeight.w600,
              letterSpacing: -0.6,
              color: _textPrimary,
            ),
          ),
        ],
      ),
    );
  }

  Widget _uniformActionButton({
    required String label,
    required IconData icon,
    required VoidCallback? onPressed,
    bool primary = false,
    bool loading = false,
  }) {
    final child = loading
        ? const SizedBox(
            width: 18,
            height: 18,
            child: CircularProgressIndicator(
              strokeWidth: 2,
              color: Colors.white,
            ),
          )
        : Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                icon,
                size: 18,
                color: primary ? Colors.white : _textPrimary,
              ),
              const SizedBox(width: 8),
              Flexible(child: Text(label, overflow: TextOverflow.ellipsis)),
            ],
          );

    final style = primary
        ? ElevatedButton.styleFrom(
            backgroundColor: _accent,
            foregroundColor: Colors.white,
            disabledBackgroundColor: const Color(0xFFD8DCE4),
            disabledForegroundColor: const Color(0xFF98A2B3),
            minimumSize: const Size.fromHeight(52),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
            elevation: 0,
            textStyle: const TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.w800,
              letterSpacing: -0.2,
              fontFamilyFallback: [
                'SF Pro Display',
                'Inter',
                'Segoe UI',
                'Roboto',
              ],
            ),
          )
        : OutlinedButton.styleFrom(
            foregroundColor: _textPrimary,
            backgroundColor: Colors.white,
            side: const BorderSide(color: _border),
            minimumSize: const Size.fromHeight(52),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
            textStyle: const TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.w700,
              letterSpacing: -0.2,
              fontFamilyFallback: [
                'SF Pro Display',
                'Inter',
                'Segoe UI',
                'Roboto',
              ],
            ),
          );

    return primary
        ? ElevatedButton(onPressed: onPressed, style: style, child: child)
        : OutlinedButton(onPressed: onPressed, style: style, child: child);
  }

  Widget _buildHeaderBanner() {
    final isSafe = _result?.likelyFallDetected != true;

    return Row(
      children: [
        IconButton(
          onPressed: () => Navigator.of(context).maybePop(),
          style: IconButton.styleFrom(
            backgroundColor: Colors.white,
            foregroundColor: _textPrimary,
            side: const BorderSide(color: _border),
          ),
          icon: const Icon(Icons.arrow_back_rounded),
          tooltip: 'Back',
        ),
        const SizedBox(width: 10),
        Expanded(
          child: Text(
            _dateLabel(widget.session.savedAt),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: GoogleFonts.jetBrainsMono(
              fontSize: 12,
              fontWeight: FontWeight.w500,
              color: _textSecondary,
            ),
          ),
        ),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          decoration: BoxDecoration(
            color: isSafe ? _sageSoft : const Color(0xFFF6DDD8),
            borderRadius: BorderRadius.circular(999),
          ),
          child: Text(
            isSafe ? 'Safe' : 'Review',
            style: GoogleFonts.interTight(
              fontSize: 12,
              fontWeight: FontWeight.w700,
              color: isSafe ? _sageDeep : _danger,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildSessionInfoCard() {
    final duration = _sessionDurationSeconds();

    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionTitle(
            'Session Details',
            'Basic context and timing for this saved recording.',
          ),
          LayoutBuilder(
            builder: (context, constraints) {
              const gap = 12.0;
              final cols = constraints.maxWidth >= 760
                  ? 4
                  : constraints.maxWidth >= 420
                  ? 2
                  : 1;
              final width = (constraints.maxWidth - gap * (cols - 1)) / cols;

              return Wrap(
                spacing: gap,
                runSpacing: gap,
                children: [
                  SizedBox(
                    width: width,
                    child: _metricBox(
                      label: 'Saved At',
                      value: _fmt(widget.session.savedAt),
                      icon: Icons.schedule_rounded,
                    ),
                  ),
                  SizedBox(
                    width: width,
                    child: _metricBox(
                      label: 'Time Range',
                      value: '0.00s → ${duration.toStringAsFixed(2)}s',
                      icon: Icons.timelapse_rounded,
                    ),
                  ),
                  SizedBox(
                    width: width,
                    child: _metricBox(
                      label: 'Activity Label',
                      value: _activityLabel,
                      icon: Icons.directions_walk_rounded,
                    ),
                  ),
                  SizedBox(
                    width: width,
                    child: _metricBox(
                      label: 'Placement Label',
                      value: _placementLabel,
                      icon: Icons.phone_android_outlined,
                    ),
                  ),
                ],
              );
            },
          ),
        ],
      ),
    );
  }

  Widget _buildAnnotationCard() {
    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionTitle(
            'Ground Truth Annotation',
            'Label this session for evaluation, review, and later retraining.',
          ),
          DropdownButtonFormField<String>(
            initialValue: _activityLabel,
            decoration: const InputDecoration(
              labelText: 'Activity Label',
              prefixIcon: Icon(Icons.directions_walk_rounded),
            ),
            items: _activityOptions
                .map(
                  (label) => DropdownMenuItem(
                    value: label,
                    child: Text(
                      label,
                      style: const TextStyle(fontWeight: FontWeight.w600),
                    ),
                  ),
                )
                .toList(),
            onChanged: (value) {
              if (value == null) return;
              setState(() {
                _activityLabel = value;
              });
            },
          ),
          const SizedBox(height: 14),
          DropdownButtonFormField<String>(
            initialValue: _placementLabel,
            decoration: const InputDecoration(
              labelText: 'Placement Label',
              prefixIcon: Icon(Icons.phone_android_outlined),
            ),
            items: _placementOptions
                .map(
                  (label) => DropdownMenuItem(
                    value: label,
                    child: Text(
                      label,
                      style: const TextStyle(fontWeight: FontWeight.w600),
                    ),
                  ),
                )
                .toList(),
            onChanged: (value) {
              if (value == null) return;
              setState(() {
                _placementLabel = value;
              });
            },
          ),
          const SizedBox(height: 14),
          TextField(
            controller: _notesController,
            maxLines: 5,
            decoration: const InputDecoration(
              labelText: 'Notes',
              alignLabelWithHint: true,
              hintText: 'Add observations, corrections, or review comments...',
              prefixIcon: Icon(Icons.notes_rounded),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildNarrativeCard() {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 16),
      child: Text.rich(
        TextSpan(
          children: [
            TextSpan(
              text: '${_narrativeHeadline()}\n',
              style: GoogleFonts.instrumentSerif(
                fontSize: 34,
                height: 1.06,
                letterSpacing: -0.7,
                color: _textPrimary,
              ),
            ),
            TextSpan(
              text: _narrativeSubline(),
              style: GoogleFonts.instrumentSerif(
                fontSize: 30,
                height: 1.12,
                letterSpacing: -0.5,
                color: _textTertiary,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildInferenceResultCard() {
    final events = _keyEvents();

    return _card(
      padding: EdgeInsets.zero,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(18, 18, 18, 8),
            child: _sectionTitle(
              'Key events',
              'The most relevant moments from this session.',
            ),
          ),
          if (events.isEmpty)
            _buildFallbackEventTile()
          else
            ...events.map(_buildTimelineEventTile),
          const SizedBox(height: 4),
        ],
      ),
    );
  }

  Widget _buildSignalQualityCard() {
    final gMax = _gMax();
    final hz = _samplingRateHz();
    final gaps = _gapCount();

    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionTitle(
            'Signal quality',
            'Sampling health for this recording.',
          ),
          LayoutBuilder(
            builder: (context, constraints) {
              const gap = 10.0;
              final columns = constraints.maxWidth >= 620 ? 3 : 1;
              final width =
                  (constraints.maxWidth - gap * (columns - 1)) / columns;

              return Wrap(
                spacing: gap,
                runSpacing: gap,
                children: [
                  SizedBox(
                    width: width,
                    child: _metricBox(
                      label: 'g-max',
                      value: gMax == null ? '-' : '${gMax.toStringAsFixed(2)}g',
                      icon: Icons.show_chart_rounded,
                    ),
                  ),
                  SizedBox(
                    width: width,
                    child: _metricBox(
                      label: 'Hz',
                      value: hz == null ? '-' : hz.toStringAsFixed(1),
                      icon: Icons.speed_rounded,
                    ),
                  ),
                  SizedBox(
                    width: width,
                    child: _metricBox(
                      label: 'Gaps',
                      value: gaps.toString(),
                      icon: Icons.more_horiz_rounded,
                    ),
                  ),
                ],
              );
            },
          ),
        ],
      ),
    );
  }

  Widget _buildReviewActionsCard() {
    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionTitle('Review tools', _status),
          LayoutBuilder(
            builder: (context, constraints) {
              const gap = 10.0;
              final columns = constraints.maxWidth >= 620 ? 2 : 1;
              final width =
                  (constraints.maxWidth - gap * (columns - 1)) / columns;
              return Wrap(
                spacing: gap,
                runSpacing: gap,
                children: [
                  SizedBox(
                    width: width,
                    child: _uniformActionButton(
                      label: _saving ? 'Saving...' : 'Save Labels',
                      icon: Icons.save_outlined,
                      onPressed: (_saving || !widget.session.hasLocalFile)
                          ? null
                          : _saveLabels,
                      primary: true,
                      loading: _saving,
                    ),
                  ),
                  SizedBox(
                    width: width,
                    child: _uniformActionButton(
                      label: _sending
                          ? (widget.session.hasLocalFile
                                ? 'Replaying...'
                                : 'Refreshing...')
                          : (widget.session.hasLocalFile
                                ? 'Replay Through Server'
                                : 'Refresh From Server'),
                      icon: Icons.refresh_rounded,
                      onPressed: _sending ? null : _sendSavedSessionToServer,
                      loading: _sending,
                    ),
                  ),
                ],
              );
            },
          ),
        ],
      ),
    );
  }

  Widget _buildTransitionCard() {
    final result = _result;
    if (result == null || result.transitionEvents.isEmpty) {
      return const SizedBox.shrink();
    }

    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionTitle('Transitions', 'Changes detected between key events.'),
          ...result.transitionEvents
              .take(3)
              .map(
                (transition) => Padding(
                  padding: const EdgeInsets.only(bottom: 10),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Icon(
                        Icons.compare_arrows_rounded,
                        color: _textSecondary,
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Text(
                          transition.description,
                          style: GoogleFonts.interTight(
                            fontSize: 14,
                            height: 1.35,
                            fontWeight: FontWeight.w600,
                            color: _textPrimary,
                          ),
                        ),
                      ),
                      Text(
                        _formatSeconds(transition.transitionTs),
                        style: GoogleFonts.jetBrainsMono(
                          fontSize: 12,
                          color: _textSecondary,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
        ],
      ),
    );
  }

  Widget _buildTimelineEventTile(TimelineEventModel event) {
    final eventColor = event.likelyFall
        ? _danger
        : event.eventKind == 'placement_change'
        ? _warning
        : _accent;
    final icon = event.likelyFall
        ? Icons.priority_high_rounded
        : event.eventKind == 'placement_change'
        ? Icons.phone_android_outlined
        : Icons.directions_walk_rounded;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      decoration: const BoxDecoration(
        border: Border(bottom: BorderSide(color: Color(0xFFEEEAE0))),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Container(
            width: 36,
            height: 36,
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: _border),
            ),
            child: Icon(icon, size: 19, color: eventColor),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  event.description.isEmpty
                      ? event.humanActivityLabel
                      : event.description,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.interTight(
                    fontSize: 14,
                    height: 1.25,
                    fontWeight: FontWeight.w600,
                    color: _textPrimary,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  '${event.humanActivityLabel} · ${_formatSeconds(event.durationSeconds)}',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.interTight(
                    fontSize: 12,
                    fontWeight: FontWeight.w500,
                    color: _textTertiary,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 12),
          Text(
            _formatSeconds(event.startTs),
            style: GoogleFonts.jetBrainsMono(
              fontSize: 12,
              fontWeight: FontWeight.w500,
              color: _textSecondary,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFallbackEventTile() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      child: Row(
        children: [
          Container(
            width: 36,
            height: 36,
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: _border),
            ),
            child: const Icon(Icons.check_rounded, size: 19, color: _sageDeep),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              _result == null
                  ? 'Replay this session to generate key events.'
                  : 'No key events needed review.',
              style: GoogleFonts.interTight(
                fontSize: 14,
                fontWeight: FontWeight.w600,
                color: _textPrimary,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTimelineLegendRow() {
    final items = _timelineLegend();
    return Wrap(
      spacing: 12,
      runSpacing: 8,
      children: [
        for (final item in items)
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 9,
                height: 9,
                decoration: BoxDecoration(
                  color: item.color,
                  borderRadius: BorderRadius.circular(3),
                ),
              ),
              const SizedBox(width: 6),
              Text(
                item.label,
                style: GoogleFonts.interTight(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: _textSecondary,
                ),
              ),
            ],
          ),
      ],
    );
  }

  Widget _buildTimelineCard() {
    final fallPoints = _fallProbabilityPoints();
    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionTitle(
            'Timeline',
            'A compact view of activity across the session.',
          ),
          Container(
            height: 40,
            decoration: BoxDecoration(
              color: _softBackground,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: _border),
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(11),
              child: CustomPaint(
                painter: _TimelineStripPainter(segments: _timelineSegments()),
                child: const SizedBox.expand(),
              ),
            ),
          ),
          if (fallPoints.isNotEmpty) ...[
            const SizedBox(height: 8),
            Container(
              height: 36,
              decoration: BoxDecoration(
                color: _softBackground,
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: _border),
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(9),
                child: CustomPaint(
                  painter: _FallProbabilityPainter(
                    points: fallPoints,
                    threshold: 0.75,
                    lineColor: _danger,
                    thresholdColor: _danger.withValues(alpha: 0.5),
                  ),
                  child: const SizedBox.expand(),
                ),
              ),
            ),
            const SizedBox(height: 4),
            Text(
              'Fall probability · dashed line at 0.75 (PROBABLE_FALL)',
              style: GoogleFonts.jetBrainsMono(
                fontSize: 10.5,
                color: _textTertiary,
              ),
            ),
          ],
          const SizedBox(height: 10),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              for (final marker in const ['0s', '15s', '30s', '45s', '60s'])
                Text(
                  marker,
                  style: GoogleFonts.jetBrainsMono(
                    fontSize: 11,
                    fontWeight: FontWeight.w500,
                    color: _textTertiary,
                  ),
                ),
            ],
          ),
          const SizedBox(height: 14),
          _buildTimelineLegendRow(),
        ],
      ),
    );
  }

  List<Offset> _fallProbabilityPoints() {
    final result = _result;
    if (result == null) return const <Offset>[];
    final timeline = result.pointTimeline;
    if (timeline.isEmpty) return const <Offset>[];

    final duration = _sessionDurationSeconds();
    if (duration <= 0) return const <Offset>[];

    final points = <Offset>[];
    for (final p in timeline) {
      final prob = p.fallProbability;
      if (prob == null) continue;
      final x = (p.midpointTs / duration).clamp(0.0, 1.0);
      final y = prob.clamp(0.0, 1.0);
      points.add(Offset(x, y));
    }
    return points;
  }

  Widget _buildFeedbackCard() {
    final feedbackEntries = _feedbackEntries();
    final sorted = [...feedbackEntries]
      ..sort(
        (a, b) => (b['recorded_at']?.toString() ?? '').compareTo(
          a['recorded_at']?.toString() ?? '',
        ),
      );
    final precision = _precisionFromFeedback(feedbackEntries);

    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionTitle(
            'Feedback',
            'Record whether the alert was correct and keep an audit trail.',
          ),
          if (sorted.isEmpty)
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: _softBackground,
                borderRadius: BorderRadius.circular(18),
                border: Border.all(color: _border),
              ),
              child: Text(
                'No feedback recorded for this session yet.',
                style: GoogleFonts.interTight(
                  fontSize: 13,
                  color: _textSecondary,
                ),
              ),
            )
          else
            Column(
              children: [
                for (final entry in sorted) _buildFeedbackTile(entry),
                if (precision != null) ...[
                  const SizedBox(height: 8),
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: _sageSoft,
                      borderRadius: BorderRadius.circular(14),
                    ),
                    child: Row(
                      children: [
                        const Icon(
                          Icons.equalizer_rounded,
                          size: 16,
                          color: _sageDeep,
                        ),
                        const SizedBox(width: 8),
                        Text(
                          'Precision after feedback (this session): '
                          '${(precision * 100).round()}%',
                          style: GoogleFonts.interTight(
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                            color: _sageDeep,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ],
            ),
          const SizedBox(height: 16),
          LayoutBuilder(
            builder: (context, constraints) {
              const gap = 12.0;
              final cols = constraints.maxWidth >= 760
                  ? 4
                  : constraints.maxWidth >= 520
                  ? 2
                  : 1;
              final width = (constraints.maxWidth - gap * (cols - 1)) / cols;

              return Wrap(
                spacing: gap,
                runSpacing: gap,
                children: [
                  SizedBox(
                    width: width,
                    child: _uniformActionButton(
                      label: 'Confirm Fall',
                      icon: Icons.check_circle_outline,
                      onPressed: (_submittingFeedback || _sending)
                          ? null
                          : () => _submitFeedback('confirmed_fall'),
                    ),
                  ),
                  SizedBox(
                    width: width,
                    child: _uniformActionButton(
                      label: 'False Alarm',
                      icon: Icons.close_rounded,
                      onPressed: (_submittingFeedback || _sending)
                          ? null
                          : () => _submitFeedback('false_alarm'),
                    ),
                  ),
                  SizedBox(
                    width: width,
                    child: _uniformActionButton(
                      label: 'Uncertain',
                      icon: Icons.help_outline_rounded,
                      onPressed: (_submittingFeedback || _sending)
                          ? null
                          : () => _submitFeedback('uncertain'),
                    ),
                  ),
                  SizedBox(
                    width: width,
                    child: _uniformActionButton(
                      label: 'Saved Sessions',
                      icon: Icons.folder_open_rounded,
                      onPressed: _openSavedSessions,
                    ),
                  ),
                ],
              );
            },
          ),
        ],
      ),
    );
  }

  Widget _buildContentLayout() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _buildHeaderBanner(),
        _buildNarrativeCard(),
        _buildTimelineCard(),
        const SizedBox(height: 16),
        _buildInferenceResultCard(),
        const SizedBox(height: 16),
        _buildSignalQualityCard(),
        const SizedBox(height: 16),
        _buildTransitionCard(),
        const SizedBox(height: 16),
        _buildSessionInfoCard(),
        const SizedBox(height: 16),
        _buildReviewActionsCard(),
        const SizedBox(height: 16),
        _buildAnnotationCard(),
        const SizedBox(height: 16),
        _buildFeedbackCard(),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _pageBackground,
      body: SafeArea(
        child: _loading
            ? const Center(child: CircularProgressIndicator(color: _accent))
            : Center(
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 760),
                  child: ListView(
                    padding: const EdgeInsets.fromLTRB(20, 12, 20, 28),
                    children: [_buildContentLayout()],
                  ),
                ),
              ),
      ),
    );
  }
}

class _TimelineSegment {
  const _TimelineSegment({
    required this.startFraction,
    required this.endFraction,
    required this.label,
    required this.color,
  });

  final double startFraction;
  final double endFraction;
  final String label;
  final Color color;
}

class _TimelineLegendItem {
  const _TimelineLegendItem({required this.label, required this.color});

  final String label;
  final Color color;
}

class _TimelineStripPainter extends CustomPainter {
  const _TimelineStripPainter({required this.segments});

  final List<_TimelineSegment> segments;

  @override
  void paint(Canvas canvas, Size size) {
    if (segments.isEmpty) {
      final paint = Paint()..color = const Color(0xFFE5E1D4);
      canvas.drawRect(Offset.zero & size, paint);
      return;
    }

    for (final segment in segments) {
      final start = (segment.startFraction * size.width).clamp(0.0, size.width);
      final end = (segment.endFraction * size.width).clamp(0.0, size.width);
      final width = math.max(2.0, end - start);
      final rect = Rect.fromLTWH(start, 0, width, size.height);
      final paint = Paint()..color = segment.color;
      canvas.drawRect(rect, paint);
    }

    final markerPaint = Paint()
      ..color = Colors.white.withValues(alpha: 0.62)
      ..strokeWidth = 1;
    for (var i = 1; i < 4; i++) {
      final x = size.width * i / 4;
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), markerPaint);
    }
  }

  @override
  bool shouldRepaint(covariant _TimelineStripPainter oldDelegate) {
    return oldDelegate.segments != segments;
  }
}

class _FallProbabilityPainter extends CustomPainter {
  const _FallProbabilityPainter({
    required this.points,
    required this.threshold,
    required this.lineColor,
    required this.thresholdColor,
  });

  final List<Offset> points;
  final double threshold;
  final Color lineColor;
  final Color thresholdColor;

  @override
  void paint(Canvas canvas, Size size) {
    final thresholdY = size.height * (1.0 - threshold);
    final dashPaint = Paint()
      ..color = thresholdColor
      ..strokeWidth = 1.2;
    const dashWidth = 4.0;
    const gapWidth = 4.0;
    var x = 0.0;
    while (x < size.width) {
      final next = math.min(x + dashWidth, size.width);
      canvas.drawLine(Offset(x, thresholdY), Offset(next, thresholdY), dashPaint);
      x += dashWidth + gapWidth;
    }

    if (points.isEmpty) return;

    final path = Path();
    for (var i = 0; i < points.length; i++) {
      final px = points[i].dx * size.width;
      final py = (1.0 - points[i].dy) * size.height;
      if (i == 0) {
        path.moveTo(px, py);
      } else {
        path.lineTo(px, py);
      }
    }

    final linePaint = Paint()
      ..color = lineColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.6
      ..strokeJoin = StrokeJoin.round
      ..strokeCap = StrokeCap.round;
    canvas.drawPath(path, linePaint);
  }

  @override
  bool shouldRepaint(covariant _FallProbabilityPainter oldDelegate) {
    return oldDelegate.points != points ||
        oldDelegate.threshold != threshold ||
        oldDelegate.lineColor != lineColor;
  }
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
