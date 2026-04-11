import 'package:flutter/material.dart';

import '../models/api_result_summary.dart';
import '../models/saved_session.dart';
import '../models/sensor_sample.dart';
import '../services/runtime_api_service.dart';
import '../services/session_storage_service.dart';
import 'saved_sessions_page.dart';

class SessionDetailPage extends StatefulWidget {
  const SessionDetailPage({super.key, required this.session});

  final SavedSession session;

  @override
  State<SessionDetailPage> createState() => _SessionDetailPageState();
}

class _SessionDetailPageState extends State<SessionDetailPage> {
  static const Color _accent = Color(0xFFC14953);
  static const Color _pageBackground = Color(0xFF848FA5);
  static const Color _cardBackground = Color(0xFFF9FAFC);
  static const Color _softBackground = Color(0xFFF1F3F7);
  static const Color _border = Color(0xFFD8DEE8);
  static const Color _textPrimary = Color(0xFF17202D);
  static const Color _textSecondary = Color(0xFF5F6878);
  static const Color _success = Color(0xFF2FA36B);
  static const Color _danger = Color(0xFFD64545);
  static const Color _warning = Color(0xFFE79A1F);

  final SessionStorageService _storage = SessionStorageService();
  final RuntimeApiService _api = RuntimeApiService(
    baseUrl: 'http://192.168.1.50:8000',
  );

  static const List<String> _activityOptions = [
    'unknown',
    'walking',
    'stairs',
    'sitting',
    'standing',
    'laying',
    'phone_handling',
    'fall',
    'other',
  ];

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

  Map<String, dynamic>? _payload;
  ApiResultSummary? _result;
  String _status = 'Idle';

  late String _activityLabel;
  late String _placementLabel;
  late TextEditingController _notesController;

  @override
  void initState() {
    super.initState();
    _activityLabel = widget.session.activityLabel ?? 'unknown';
    _placementLabel = widget.session.placementLabel ?? 'unknown';
    _notesController = TextEditingController(text: widget.session.notes ?? '');
    _load();
  }

  @override
  void dispose() {
    _api.dispose();
    _notesController.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    try {
      final payload = await _storage.loadSessionPayload(
        widget.session.filePath,
      );

      ApiResultSummary? restoredResult;
      final savedInference = payload['inference_result'];
      if (savedInference is Map) {
        restoredResult = ApiResultSummary.fromJson(
          savedInference.map((key, value) => MapEntry(key.toString(), value)),
        );
      }

      if (!mounted) return;
      setState(() {
        _payload = payload;
        _result = restoredResult;
        _loading = false;
        _status = restoredResult == null
            ? 'Session loaded'
            : 'Session loaded with saved inference result';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _status = 'Failed to load session: $e';
      });
    }
  }

  Future<void> _saveLabels() async {
    setState(() {
      _saving = true;
      _status = 'Saving labels...';
    });

    try {
      await _storage.updateSessionLabels(
        filePath: widget.session.filePath,
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
        filePath: widget.session.filePath,
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

      final summary = await _api.submitSession(
        sessionId:
            (metadata['session_id'] as String?) ??
            widget.session.fileName.replaceAll('.json', ''),
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
        filePath: widget.session.filePath,
        inferenceResult: summary.rawJson,
      );

      if (!mounted) return;
      setState(() {
        _result = summary;
        _payload = <String, dynamic>{
          ...?_payload,
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
      final ack = await _api.submitFeedback(
        sessionId: result.sessionId,
        userFeedback: feedback,
        notes: _notesController.text.trim().isEmpty
            ? null
            : _notesController.text.trim(),
      );

      await _storage.appendFeedback(
        filePath: widget.session.filePath,
        feedbackEntry: <String, dynamic>{
          'request_id': ack.requestId,
          'recorded_at': ack.recordedAt?.toUtc().toIso8601String(),
          'status': ack.status,
          'session_id': ack.sessionId,
          'event_id': ack.eventId,
          'window_id': ack.windowId,
          'user_feedback': ack.userFeedback,
          'message': ack.message,
        },
      );

      if (!mounted) return;
      setState(() {
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
      MaterialPageRoute(builder: (_) => const SavedSessionsPage()),
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

  double? _firstTimestamp() {
    final samples = _payload?['samples'] as List?;
    if (samples == null || samples.isEmpty) return null;
    final first = samples.first;
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

  String? _preferredTestTitle() {
    final savedTitle =
        _trimmedText(_payload?['test_title']) ?? widget.session.testTitle;
    if (savedTitle != null && savedTitle.isNotEmpty) {
      return savedTitle;
    }

    return _testTitleFromNotes();
  }

  String? _testTitleFromNotes() {
    final notes = _notesController.text.trim();
    if (notes.isEmpty) {
      return null;
    }

    for (final part in notes.split(' | ')) {
      if (part.startsWith('test_title=')) {
        final value = part.substring('test_title='.length).trim();
        if (value.isNotEmpty) {
          return value;
        }
      }
    }

    return null;
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

  Color _levelColor(String level) {
    switch (level.toLowerCase()) {
      case 'high':
        return _danger;
      case 'medium':
        return _warning;
      case 'low':
        return const Color(0xFFD4A72C);
      default:
        return _success;
    }
  }

  Widget _card({
    required Widget child,
    EdgeInsets padding = const EdgeInsets.all(20),
  }) {
    return Container(
      decoration: BoxDecoration(
        color: _cardBackground,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: Colors.white.withValues(alpha: 0.22)),
        boxShadow: const [
          BoxShadow(
            color: Color(0x220F172A),
            blurRadius: 24,
            offset: Offset(0, 12),
          ),
        ],
      ),
      child: Padding(padding: padding, child: child),
    );
  }

  Widget _chip({
    required String label,
    required Color textColor,
    IconData? icon,
    Color? background,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 13, vertical: 10),
      decoration: BoxDecoration(
        color: background ?? textColor.withValues(alpha: 0.10),
        borderRadius: BorderRadius.circular(999),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (icon != null) ...[
            Icon(icon, size: 15, color: textColor),
            const SizedBox(width: 7),
          ],
          Text(
            label,
            style: TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w700,
              color: textColor,
            ),
          ),
        ],
      ),
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
            style: const TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.w800,
              letterSpacing: -0.8,
              color: _textPrimary,
              fontFamilyFallback: [
                'SF Pro Display',
                'Inter',
                'Segoe UI',
                'Roboto',
              ],
            ),
          ),
          const SizedBox(height: 4),
          Text(
            subtitle,
            style: const TextStyle(
              fontSize: 15,
              color: _textSecondary,
              height: 1.35,
              fontWeight: FontWeight.w500,
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
            label,
            style: const TextStyle(
              fontSize: 14,
              color: _textSecondary,
              fontWeight: FontWeight.w500,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            value,
            style: const TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.w800,
              letterSpacing: -0.6,
              color: _textPrimary,
              fontFamilyFallback: [
                'SF Pro Display',
                'Inter',
                'Segoe UI',
                'Roboto',
              ],
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
    return Container(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(28),
        color: Colors.white.withValues(alpha: 0.16),
        border: Border.all(color: Colors.white.withValues(alpha: 0.18)),
        boxShadow: const [
          BoxShadow(
            color: Color(0x220F172A),
            blurRadius: 28,
            offset: Offset(0, 14),
          ),
        ],
      ),
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: LayoutBuilder(
          builder: (context, constraints) {
            final wide = constraints.maxWidth >= 720;
            final firstTs = _firstTimestamp();
            final lastTs = _lastTimestamp();

            final left = Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'SESSION REVIEW',
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 0.9,
                    color: Color(0xFFF3F4F8),
                  ),
                ),
                const SizedBox(height: 12),
                Text(
                  widget.session.fileName,
                  style: const TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.w800,
                    letterSpacing: -0.9,
                    color: Colors.white,
                    fontFamilyFallback: [
                      'SF Pro Display',
                      'Inter',
                      'Segoe UI',
                      'Roboto',
                    ],
                  ),
                ),
                const SizedBox(height: 10),
                Text(
                  _status,
                  style: const TextStyle(
                    fontSize: 15,
                    height: 1.45,
                    color: Color(0xFFF4F6FA),
                    fontWeight: FontWeight.w500,
                  ),
                ),
                const SizedBox(height: 18),
                Wrap(
                  spacing: 10,
                  runSpacing: 10,
                  children: [
                    _chip(
                      label: widget.session.subjectId,
                      textColor: Colors.white,
                      icon: Icons.person_outline,
                      background: Colors.white.withValues(alpha: 0.16),
                    ),
                    _chip(
                      label: widget.session.placement,
                      textColor: Colors.white,
                      icon: Icons.phone_android_outlined,
                      background: Colors.white.withValues(alpha: 0.16),
                    ),
                    _chip(
                      label:
                          '${firstTs?.toStringAsFixed(1) ?? '-'}s → ${lastTs?.toStringAsFixed(1) ?? '-'}s',
                      textColor: Colors.white,
                      icon: Icons.timelapse_rounded,
                      background: Colors.white.withValues(alpha: 0.16),
                    ),
                  ],
                ),
              ],
            );

            final right = Wrap(
              runSpacing: 10,
              spacing: 10,
              children: [
                SizedBox(
                  width: wide ? 220 : double.infinity,
                  child: _uniformActionButton(
                    label: _saving ? 'Saving...' : 'Save Labels',
                    icon: Icons.save_outlined,
                    onPressed: _saving ? null : _saveLabels,
                    primary: true,
                    loading: _saving,
                  ),
                ),
                SizedBox(
                  width: wide ? 220 : double.infinity,
                  child: _uniformActionButton(
                    label: _sending ? 'Replaying...' : 'Replay Through Server',
                    icon: Icons.refresh_rounded,
                    onPressed: _sending ? null : _sendSavedSessionToServer,
                    loading: _sending,
                  ),
                ),
              ],
            );

            if (!wide) {
              return Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [left, const SizedBox(height: 20), right],
              );
            }

            return Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(flex: 7, child: left),
                const SizedBox(width: 24),
                Expanded(flex: 3, child: right),
              ],
            );
          },
        ),
      ),
    );
  }

  Widget _buildSessionInfoCard() {
    final firstTs = _firstTimestamp();
    final lastTs = _lastTimestamp();
    final testTitle = _preferredTestTitle();

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
                      value:
                          '${firstTs?.toStringAsFixed(2) ?? '-'}s → ${lastTs?.toStringAsFixed(2) ?? '-'}s',
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
                  if (testTitle != null)
                    SizedBox(
                      width: width,
                      child: _metricBox(
                        label: 'Test',
                        value: testTitle,
                        icon: Icons.assignment_outlined,
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
    final summary = _result?.narrativeSummary;
    if (summary == null) {
      return const SizedBox.shrink();
    }

    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionTitle(
            'Session Narrative',
            'High-level interpretation of the full recording.',
          ),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
              color: _softBackground,
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: _border),
            ),
            child: Text(
              summary.summaryText,
              style: const TextStyle(
                fontSize: 15,
                height: 1.45,
                fontWeight: FontWeight.w700,
                color: _textPrimary,
              ),
            ),
          ),
          const SizedBox(height: 16),
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
                      label: 'Dominant Activity',
                      value: summary.dominantActivityLabel,
                      icon: Icons.directions_walk_rounded,
                    ),
                  ),
                  SizedBox(
                    width: width,
                    child: _metricBox(
                      label: 'Dominant Placement',
                      value: summary.dominantPlacementLabel,
                      icon: Icons.phone_android_outlined,
                    ),
                  ),
                  SizedBox(
                    width: width,
                    child: _metricBox(
                      label: 'Events',
                      value: summary.eventCount.toString(),
                      icon: Icons.timeline_rounded,
                    ),
                  ),
                  SizedBox(
                    width: width,
                    child: _metricBox(
                      label: 'Transitions',
                      value: summary.transitionCount.toString(),
                      icon: Icons.compare_arrows_rounded,
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

  Widget _buildInferenceResultCard() {
    final result = _result;
    if (result == null) {
      return _card(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _sectionTitle(
              'Inference Result',
              'Replay the session through the backend to populate this section.',
            ),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(26),
              decoration: BoxDecoration(
                color: _softBackground,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: _border),
              ),
              child: const Column(
                children: [
                  Icon(
                    Icons.insights_outlined,
                    size: 42,
                    color: _textSecondary,
                  ),
                  SizedBox(height: 10),
                  Text(
                    'No saved inference result yet',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w800,
                      color: _textPrimary,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      );
    }

    final levelColor = _levelColor(result.warningLevel);
    final stateCounts = result.placementSummary.stateCounts;

    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionTitle(
            'Inference Result',
            'Latest saved or replayed model output for this session.',
          ),
          Wrap(
            spacing: 10,
            runSpacing: 10,
            children: [
              _chip(
                label: result.warningLevel.toUpperCase(),
                textColor: levelColor,
                icon: Icons.warning_amber_rounded,
              ),
              _chip(
                label: result.likelyFallDetected
                    ? 'Likely Fall Detected'
                    : 'No Strong Fall Detected',
                textColor: result.likelyFallDetected ? _danger : _success,
                icon: result.likelyFallDetected
                    ? Icons.report_problem_outlined
                    : Icons.verified_outlined,
              ),
            ],
          ),
          const SizedBox(height: 16),
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
                      label: 'Top Activity',
                      value: result.topHarLabel ?? '-',
                      icon: Icons.directions_walk_rounded,
                    ),
                  ),
                  SizedBox(
                    width: width,
                    child: _metricBox(
                      label: 'HAR Fraction',
                      value: result.topHarFraction == null
                          ? '-'
                          : result.topHarFraction!.toStringAsFixed(3),
                      icon: Icons.pie_chart_outline_rounded,
                    ),
                  ),
                  SizedBox(
                    width: width,
                    child: _metricBox(
                      label: 'Grouped Events',
                      value: result.groupedFallEventCount.toString(),
                      icon: Icons.timeline_rounded,
                    ),
                  ),
                  SizedBox(
                    width: width,
                    child: _metricBox(
                      label: 'Fall Probability',
                      value: result.topFallProbability == null
                          ? '-'
                          : result.topFallProbability!.toStringAsFixed(4),
                      icon: Icons.show_chart_rounded,
                    ),
                  ),
                ],
              );
            },
          ),
          const SizedBox(height: 16),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
              color: _softBackground,
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: _border),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Placement Summary',
                  style: TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w800,
                    color: _textPrimary,
                  ),
                ),
                const SizedBox(height: 12),
                Text('State: ${result.placementSummary.placementState}'),
                const SizedBox(height: 6),
                Text(
                  'Confidence: ${result.placementSummary.placementConfidence?.toStringAsFixed(3) ?? '-'}',
                ),
                const SizedBox(height: 6),
                Text(
                  'State Fraction: ${result.placementSummary.stateFraction?.toStringAsFixed(3) ?? '-'}',
                ),
                const SizedBox(height: 10),
                Text(
                  'State Counts: ${stateCounts.entries.map((e) => '${e.key}:${e.value}').join(', ')}',
                  style: const TextStyle(
                    fontSize: 14,
                    color: _textSecondary,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          Text(
            result.recommendedMessage,
            style: const TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.w700,
              color: _textPrimary,
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

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: _softBackground,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: _border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Wrap(
            spacing: 10,
            runSpacing: 10,
            crossAxisAlignment: WrapCrossAlignment.center,
            children: [
              _chip(
                label: _formatSeconds(event.startTs),
                textColor: _textPrimary,
                icon: Icons.schedule_rounded,
                background: Colors.white,
              ),
              _chip(
                label: event.eventKind.replaceAll('_', ' '),
                textColor: eventColor,
                background: eventColor.withValues(alpha: 0.10),
              ),
              if (event.likelyFall)
                _chip(
                  label: 'fall-like',
                  textColor: _danger,
                  background: _danger.withValues(alpha: 0.10),
                ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            event.description,
            style: const TextStyle(
              fontSize: 16,
              height: 1.4,
              fontWeight: FontWeight.w700,
              color: _textPrimary,
            ),
          ),
          const SizedBox(height: 10),
          Wrap(
            spacing: 16,
            runSpacing: 8,
            children: [
              Text(
                'Activity: ${event.humanActivityLabel}',
                style: const TextStyle(
                  fontSize: 14,
                  color: _textSecondary,
                  fontWeight: FontWeight.w600,
                ),
              ),
              Text(
                'Placement: ${event.humanPlacementLabel}',
                style: const TextStyle(
                  fontSize: 14,
                  color: _textSecondary,
                  fontWeight: FontWeight.w600,
                ),
              ),
              Text(
                'Duration: ${_formatSeconds(event.durationSeconds)}',
                style: const TextStyle(
                  fontSize: 14,
                  color: _textSecondary,
                  fontWeight: FontWeight.w600,
                ),
              ),
              Text(
                'Peak fall prob: ${event.fallProbabilityPeak?.toStringAsFixed(3) ?? '-'}',
                style: const TextStyle(
                  fontSize: 14,
                  color: _textSecondary,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildTimelineCard() {
    final result = _result;
    if (result == null || result.timelineEvents.isEmpty) {
      return const SizedBox.shrink();
    }

    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionTitle(
            'Timeline Events',
            'Interpreted session events built from activity, placement, and fall signals.',
          ),
          ...result.timelineEvents.map(_buildTimelineEventTile),
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
          _sectionTitle(
            'Transitions',
            'Detected changes between timeline events.',
          ),
          ...result.transitionEvents.map(
            (transition) => Container(
              margin: const EdgeInsets.only(bottom: 12),
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: _softBackground,
                borderRadius: BorderRadius.circular(18),
                border: Border.all(color: _border),
              ),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Icon(
                    Icons.compare_arrows_rounded,
                    color: _textSecondary,
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          transition.description,
                          style: const TextStyle(
                            fontSize: 15,
                            fontWeight: FontWeight.w700,
                            color: _textPrimary,
                            height: 1.35,
                          ),
                        ),
                        const SizedBox(height: 6),
                        Text(
                          'At ${_formatSeconds(transition.transitionTs)}',
                          style: const TextStyle(
                            fontSize: 13,
                            color: _textSecondary,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
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

  Widget _buildFeedbackCard() {
    final feedbackEntries = _feedbackEntries();
    final latest = feedbackEntries.isEmpty ? null : feedbackEntries.last;

    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionTitle(
            'Feedback',
            'Record whether the alert was correct and keep an audit trail.',
          ),
          if (latest != null)
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: _softBackground,
                borderRadius: BorderRadius.circular(18),
                border: Border.all(color: _border),
              ),
              child: Text(
                'Latest feedback: ${(latest['user_feedback'] ?? 'unknown').toString()}',
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w700,
                  color: _textPrimary,
                ),
              ),
            ),
          if (latest != null) const SizedBox(height: 16),
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
    return LayoutBuilder(
      builder: (context, constraints) {
        final wide = constraints.maxWidth >= 1080;

        final main = Column(
          children: [
            _buildHeaderBanner(),
            const SizedBox(height: 16),
            _buildInferenceResultCard(),
            const SizedBox(height: 16),
            _buildNarrativeCard(),
            const SizedBox(height: 16),
            _buildTimelineCard(),
            const SizedBox(height: 16),
            _buildTransitionCard(),
          ],
        );

        final side = Column(
          children: [
            _buildSessionInfoCard(),
            const SizedBox(height: 16),
            _buildAnnotationCard(),
            const SizedBox(height: 16),
            _buildFeedbackCard(),
          ],
        );

        if (!wide) {
          return Column(children: [main, const SizedBox(height: 16), side]);
        }

        return Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Expanded(flex: 6, child: main),
            const SizedBox(width: 16),
            Expanded(flex: 4, child: side),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _pageBackground,
      appBar: AppBar(
        backgroundColor: _pageBackground,
        foregroundColor: Colors.white,
        title: const Text(
          'Session Review',
          style: TextStyle(
            fontWeight: FontWeight.w800,
            letterSpacing: -0.8,
            color: Colors.white,
            fontFamilyFallback: [
              'SF Pro Display',
              'Inter',
              'Segoe UI',
              'Roboto',
            ],
          ),
        ),
      ),
      body: SafeArea(
        child: _loading
            ? const Center(child: CircularProgressIndicator())
            : Center(
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 1220),
                  child: ListView(
                    padding: const EdgeInsets.fromLTRB(16, 10, 16, 24),
                    children: [_buildContentLayout()],
                  ),
                ),
              ),
      ),
    );
  }
}

String? _trimmedText(dynamic value) {
  if (value == null) {
    return null;
  }

  final text = value.toString().trim();
  if (text.isEmpty) {
    return null;
  }

  return text;
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
