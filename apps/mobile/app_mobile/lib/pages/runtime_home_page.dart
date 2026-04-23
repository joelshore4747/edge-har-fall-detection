import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;
import 'dart:ui' show ImageFilter;

import 'package:flutter/foundation.dart'
    show TargetPlatform, defaultTargetPlatform, kIsWeb;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:google_fonts/google_fonts.dart';

import '../config/runtime_config.dart';
import '../models/runtime_identity.dart';
import '../models/saved_session.dart';
import '../models/sensor_sample.dart';
import '../services/runtime_api_service.dart';
import '../services/sensor_recorder.dart';
import '../services/session_storage_service.dart';
import '../widgets/session_save_sheet.dart';
import 'saved_sessions_page.dart';
import 'session_detail_page.dart';

import '../models/api_result_summary.dart';

class RuntimeHomePage extends StatefulWidget {
  const RuntimeHomePage({
    super.key,
    required this.initialIdentity,
    this.onSignOut,
  });

  final RuntimeIdentity initialIdentity;
  final Future<void> Function()? onSignOut;

  @override
  State<RuntimeHomePage> createState() => _RuntimeHomePageState();
}

class _RuntimeHomePageState extends State<RuntimeHomePage> {
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
  static const Color _sageMist = Color(0xFFECF3EE);
  static const Color _success = Color(0xFF2C8A66);
  static const Color _danger = Color(0xFFC14C41);
  static const Color _warning = Color(0xFFC29A20);
  static const Color _darkInk = Color(0xFFF2EFE8);
  static const Color _darkInk2 = Color(0xFFC8C6BE);
  static const Color _darkInk3 = Color(0xFF8C8E86);
  static const Color _darkBg = Color(0xFF0F1210);
  static const Color _darkCard = Color(0xFF171A18);
  static const Color _darkBorder = Color(0xFF1F2420);
  static const Color _waveCyan = Color(0xFF6ED0A8);
  static const Color _waveAmber = Color(0xFFE6C968);
  static const Color _waveBlue = Color(0xFF8AB9E0);
  static const Color _recordRed = Color(0xFFFF5757);
  static const Color _alertBg = Color(0xFF1F0A08);
  static const Color _alertRed = Color(0xFFFF8378);
  static const Duration _fallAlertDuration = Duration(seconds: 30);

  static const List<String> _placementOptions = <String>[
    'pocket',
    'hand',
    'desk',
    'bag',
    'unknown',
  ];

  RuntimeApiService? _api;
  final SensorRecorderService _recorder = SensorRecorderService();
  final SessionStorageService _storage = SessionStorageService();
  Timer? _recordingUiTimer;
  Timer? _fallAlertTimer;

  final TextEditingController _subjectController = TextEditingController();

  String _selectedPlacement = 'pocket';
  int _selectedSection = 0; // 0 home, 1 care, 2 setup

  String? _savedSessionsDirPath;
  String? _savedSessionsDirError;
  String? _lastSavedSessionPath;
  String _lastSaveStatus = 'No local save attempted yet.';
  _SaveOutcome _lastSaveOutcome = _SaveOutcome.none;
  DateTime? _lastSaveTimestamp;

  bool _isSending = false;
  bool _isCheckingHealth = false;
  bool _serverHealthy = false;
  bool _showResultPage = false;
  bool _showFallAlert = false;

  String _status = 'Idle';
  String? _activeSessionId;
  DateTime? _fallAlertStartedAt;
  RuntimeHealthResponse? _health;
  ApiResultSummary? _result;

  @override
  void initState() {
    super.initState();
    _bootstrapIdentityAndHealth();
    _refreshSavedSessionsPath();
  }

  @override
  void dispose() {
    _recordingUiTimer?.cancel();
    _fallAlertTimer?.cancel();
    _api?.dispose();
    _recorder.dispose();
    _subjectController.dispose();
    super.dispose();
  }

  String _newSessionId() => 'session_${DateTime.now().millisecondsSinceEpoch}';

  String _devicePlatformValue() {
    if (kIsWeb) {
      return 'unknown';
    }

    switch (defaultTargetPlatform) {
      case TargetPlatform.iOS:
        return 'ios';
      case TargetPlatform.android:
        return 'android';
      default:
        return 'unknown';
    }
  }

  String _normalisedSubjectId() {
    final value = _subjectController.text.trim();
    return value.isEmpty ? 'anonymous_user' : value;
  }

  String _normalisedPlacement() => _selectedPlacement;

  bool _hasPendingRecording() =>
      !_recorder.isRecording && _recorder.samples.isNotEmpty;

  void _startRecordingUiTimer() {
    _recordingUiTimer?.cancel();
    _recordingUiTimer = Timer.periodic(const Duration(milliseconds: 120), (_) {
      if (!mounted || !_recorder.isRecording) {
        _recordingUiTimer?.cancel();
        _recordingUiTimer = null;
        return;
      }
      setState(() {});
    });
  }

  void _stopRecordingUiTimer() {
    _recordingUiTimer?.cancel();
    _recordingUiTimer = null;
  }

  void _stopFallAlertTimer() {
    _fallAlertTimer?.cancel();
    _fallAlertTimer = null;
  }

  void _startFallAlert() {
    _stopFallAlertTimer();
    if (!mounted) return;

    setState(() {
      _fallAlertStartedAt = DateTime.now().toUtc();
      _showFallAlert = true;
      _showResultPage = false;
      _status = 'Fall detected. Awaiting response.';
    });

    _fallAlertTimer = Timer.periodic(const Duration(milliseconds: 250), (_) {
      if (!mounted || !_showFallAlert) {
        _stopFallAlertTimer();
        return;
      }

      if (_fallAlertRemaining().inMilliseconds <= 0) {
        _stopFallAlertTimer();
      }

      setState(() {});
    });
  }

  void _setResultVisibility(ApiResultSummary? summary) {
    _result = summary;
    _showFallAlert = false;
    _showResultPage = summary != null && !summary.likelyFallDetected;
    if (summary == null) {
      _fallAlertStartedAt = null;
    }
  }

  void _presentFallAlertIfNeeded(ApiResultSummary? summary) {
    if (summary?.likelyFallDetected == true) {
      _startFallAlert();
    } else {
      _stopFallAlertTimer();
    }
  }

  String _defaultSessionName() {
    final now = DateTime.now();
    final date =
        '${now.year}-${now.month.toString().padLeft(2, '0')}-${now.day.toString().padLeft(2, '0')}';
    final time =
        '${now.hour.toString().padLeft(2, '0')}-${now.minute.toString().padLeft(2, '0')}';
    return 'session_${date}_$time';
  }

  Future<void> _bootstrapIdentityAndHealth() async {
    if (!mounted) return;

    setState(() {
      _isCheckingHealth = true;
      _status = 'Checking server health...';
    });

    try {
      final identity = widget.initialIdentity;
      _api?.dispose();
      _api = RuntimeApiService(
        baseUrl: runtimeApiBaseUrl,
        basicAuthUsername: identity.username,
        basicAuthPassword: identity.password,
      );

      if (!mounted) return;
      setState(() {
        _subjectController.text = identity.subjectId;
        _status =
            'Signed in as ${identity.username}. Checking server health...';
      });

      await _runHealthCheck(updateStatus: false);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _health = null;
        _serverHealthy = false;
        _status = 'Failed to prepare signed-in account: $e';
      });
    } finally {
      if (mounted) {
        setState(() {
          _isCheckingHealth = false;
        });
      }
    }
  }

  Future<void> _checkHealth() async {
    await _runHealthCheck(updateStatus: true);
  }

  Future<void> _runHealthCheck({required bool updateStatus}) async {
    final api = _api;
    if (api == null) {
      if (mounted && updateStatus) {
        setState(() {
          _health = null;
          _serverHealthy = false;
          _status = 'Your account is not ready yet.';
        });
      }
      return;
    }
    if (!mounted) return;

    if (updateStatus) {
      setState(() {
        _isCheckingHealth = true;
        _status = 'Checking server health...';
      });
    }

    try {
      final health = await api.checkHealth();

      if (!mounted) return;
      setState(() {
        _health = health;
        _serverHealthy = health.status.toLowerCase() == 'ok';
        _status = _serverHealthy
            ? 'Server is healthy.'
            : 'Server responded, but health status is not OK.';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _health = null;
        _serverHealthy = false;
        _status = 'Health check failed: $e';
      });
    } finally {
      if (mounted && updateStatus) {
        setState(() {
          _isCheckingHealth = false;
        });
      }
    }
  }

  Future<void> _refreshSavedSessionsPath({bool updateStatus = false}) async {
    try {
      final path = await _storage.getSavedSessionsDirectoryPath();
      if (!mounted) return;
      setState(() {
        _savedSessionsDirPath = path;
        _savedSessionsDirError = null;
        if (updateStatus) {
          _status = 'Saved sessions directory: $path';
        }
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _savedSessionsDirPath = null;
        _savedSessionsDirError = e.toString();
        if (updateStatus) {
          _status = 'Failed to resolve saved sessions directory: $e';
        }
      });
    }
  }

  void _recordSaveOutcome({
    required _SaveOutcome outcome,
    required String status,
    String? savedPath,
  }) {
    _lastSaveOutcome = outcome;
    _lastSaveStatus = status;
    _lastSavedSessionPath = savedPath;
    _lastSaveTimestamp = DateTime.now();
  }

  Future<void> _startRecording() async {
    if (_recorder.isRecording) {
      return;
    }

    final sessionId = _newSessionId();
    _stopFallAlertTimer();

    setState(() {
      _setResultVisibility(null);
      _activeSessionId = sessionId;
      _status = 'Starting recording...';
    });

    try {
      await _recorder.start(
        onSample: (_) {
          if (!mounted) return;
          setState(() {
            _status = 'Recording...';
          });
        },
        onError: (error, stackTrace) {
          if (!mounted) return;
          setState(() {
            _status = 'Sensor recording error: $error';
          });
        },
      );

      if (!mounted) return;
      setState(() {
        _selectedSection = 0;
        _status = 'Recording...';
      });
      _startRecordingUiTimer();
    } catch (e) {
      _stopRecordingUiTimer();
      if (!mounted) return;
      setState(() {
        _status = 'Failed to start recording: $e';
      });
    }
  }

  Future<void> _stopRecording() async {
    try {
      await _recorder.stop();
      _stopRecordingUiTimer();

      if (_recorder.samples.isEmpty) {
        if (!mounted) return;
        setState(() {
          _status = 'Recording stopped. No samples were captured.';
          _recordSaveOutcome(
            outcome: _SaveOutcome.skipped,
            status: 'No samples were available to save.',
          );
        });
        return;
      }

      if (!mounted) return;
      setState(() {
        _status = 'Recording stopped. Name and save your session.';
      });

      await _openSaveFlowForCurrentRecording(openedAfterStop: true);
    } catch (e) {
      _stopRecordingUiTimer();
      if (!mounted) return;
      setState(() {
        _status = 'Failed to stop recording: $e';
        _recordSaveOutcome(
          outcome: _SaveOutcome.failed,
          status: 'Save failed: $e',
        );
      });
    }
  }

  Future<void> _openSaveFlowForCurrentRecording({
    bool openedAfterStop = false,
  }) async {
    if (_recorder.samples.isEmpty) {
      if (!mounted) return;
      setState(() {
        _status = 'No recorded samples are waiting to be saved.';
      });
      return;
    }

    final saveRequest = await showModalBottomSheet<SessionSaveRequest>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.white,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
      ),
      builder: (context) {
        return SessionSaveSheet(
          initialFileName: _defaultSessionName(),
          initialCategory: 'other',
          sampleCount: _recorder.sampleCount,
          allowUpload: _serverHealthy && _api != null,
        );
      },
    );

    if (saveRequest == null) {
      if (!mounted) return;
      setState(() {
        _status = openedAfterStop
            ? 'Save cancelled. This recording is still available to save.'
            : 'Save cancelled.';
        _recordSaveOutcome(
          outcome: _SaveOutcome.skipped,
          status: 'Save dialog dismissed. Recording is still pending.',
        );
      });
      return;
    }

    await _saveCurrentRecording(saveRequest);
  }

  Future<void> _saveCurrentRecording(SessionSaveRequest saveRequest) async {
    final sessionId = _activeSessionId ?? _newSessionId();
    final shouldUpload =
        saveRequest.destination == SessionSaveDestination.localAndUpload;

    setState(() {
      _isSending = true;
      _status = shouldUpload
          ? 'Saving locally and uploading to server...'
          : 'Saving session locally...';
    });

    String? savedPath;
    ApiResultSummary? summary;

    try {
      savedPath = await _storage.saveSession(
        sessionId: sessionId,
        fileName: saveRequest.fileName,
        subjectId: _normalisedSubjectId(),
        placement: _normalisedPlacement(),
        datasetName: 'APP_RUNTIME',
        sourceType: _recorder.supportsLiveSensors ? 'mobile_app' : 'debug',
        devicePlatform: _devicePlatformValue(),
        deviceModel: kIsWeb ? 'web_browser' : defaultTargetPlatform.name,
        recordingMode: _recorder.supportsLiveSensors ? 'live_capture' : 'demo',
        runtimeMode: _recorder.supportsLiveSensors
            ? 'mobile_live'
            : 'desktop_demo',
        activityLabel: saveRequest.category,
        samples: _recorder.samples
            .map((sample) => sample.toJson())
            .toList(growable: false),
      );

      await _refreshSavedSessionsPath();

      if (savedPath == null) {
        throw StateError('The session did not produce a local save file.');
      }

      if (shouldUpload) {
        final api = _api;
        if (api == null) {
          throw StateError('Your account is not ready for upload yet.');
        }

        summary = await api.submitSession(
          sessionId: sessionId,
          sessionName: saveRequest.fileName,
          activityLabel: saveRequest.category,
          subjectId: _normalisedSubjectId(),
          placement: _normalisedPlacement(),
          samples: _recorder.samples,
          devicePlatform: _devicePlatformValue(),
          deviceModel: kIsWeb ? 'web_browser' : defaultTargetPlatform.name,
          includeHarWindows: false,
          includeFallWindows: false,
          includeCombinedTimeline: true,
          includeGroupedFallEvents: true,
        );

        await _storage.saveInferenceResult(
          filePath: savedPath,
          inferenceResult: summary.rawJson,
        );
      }

      _recorder.clear();
      if (!mounted) return;
      setState(() {
        _activeSessionId = null;
        _setResultVisibility(summary);
        _selectedSection = 0;
        _status = shouldUpload
            ? 'Saved and uploaded "${saveRequest.fileName}".'
            : 'Saved "${saveRequest.fileName}" locally.';
        _recordSaveOutcome(
          outcome: _SaveOutcome.success,
          status: shouldUpload
              ? 'Saved locally and uploaded to the server.'
              : 'Saved locally only.',
          savedPath: savedPath,
        );
      });
      _presentFallAlertIfNeeded(summary);

      if (!mounted) return;
      if (summary?.likelyFallDetected != true) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              shouldUpload
                  ? 'Saved and uploaded ${saveRequest.fileName}'
                  : 'Saved ${saveRequest.fileName} locally',
            ),
          ),
        );
      }
    } catch (e) {
      final localSaveSucceeded = savedPath != null;
      if (localSaveSucceeded) {
        _recorder.clear();
      }
      if (!mounted) return;
      setState(() {
        _activeSessionId = localSaveSucceeded ? null : _activeSessionId;
        _setResultVisibility(summary);
        _status = localSaveSucceeded
            ? 'Saved locally as "${saveRequest.fileName}", but upload failed: $e'
            : 'Failed to save session: $e';
        _recordSaveOutcome(
          outcome: localSaveSucceeded
              ? _SaveOutcome.partial
              : _SaveOutcome.failed,
          status: localSaveSucceeded
              ? 'Local save succeeded, but upload failed.'
              : 'Local save failed.',
          savedPath: savedPath,
        );
      });
      _presentFallAlertIfNeeded(summary);
    } finally {
      if (mounted) {
        setState(() {
          _isSending = false;
        });
      }
    }
  }

  Future<void> _sendBundledDemoSession() async {
    final api = _api;
    if (api == null) {
      setState(() {
        _status =
            'Your account is still being prepared. Try again in a moment.';
      });
      return;
    }
    _stopFallAlertTimer();
    setState(() {
      _isSending = true;
      _status = 'Sending bundled demo session...';
      _setResultVisibility(null);
    });

    try {
      final raw = await rootBundle.loadString(
        'assets/demo_session_phone1.json',
      );
      final payload = jsonDecode(raw) as Map<String, dynamic>;

      final metadata =
          (payload['metadata'] as Map?)?.cast<String, dynamic>() ??
          <String, dynamic>{};
      final rawSamples = payload['samples'] as List<dynamic>? ?? <dynamic>[];

      final samples = rawSamples
          .whereType<Map>()
          .map((item) => item.cast<String, dynamic>())
          .map(SensorSample.fromJson)
          .toList(growable: false);

      if (samples.length < 32) {
        throw StateError('Demo session contains fewer than 32 samples.');
      }

      final sessionId = (metadata['session_id'] as String?) ?? _newSessionId();

      final summary = await api.submitSession(
        sessionId: sessionId,
        sessionName:
            (metadata['session_name'] as String?) ??
            (metadata['file_name'] as String?) ??
            sessionId,
        activityLabel: metadata['activity_label'] as String?,
        subjectId: (metadata['subject_id'] as String?) ?? 'demo_user',
        placement: ((metadata['placement'] as String?) ?? 'pocket')
            .toLowerCase(),
        taskType: (metadata['task_type'] as String?) ?? 'runtime',
        datasetName: (metadata['dataset_name'] as String?) ?? 'APP_RUNTIME',
        sourceType: (metadata['source_type'] as String?) ?? 'mobile_app',
        devicePlatform:
            (metadata['device_platform'] as String?) ?? _devicePlatformValue(),
        deviceModel: metadata['device_model'] as String?,
        appVersion: metadata['app_version'] as String?,
        appBuild: metadata['app_build'] as String?,
        recordingMode:
            (metadata['recording_mode'] as String?) ?? 'live_capture',
        runtimeMode: (metadata['runtime_mode'] as String?) ?? 'mobile_live',
        samplingRateHz: _asDouble(metadata['sampling_rate_hz']),
        notes: metadata['notes'] as String?,
        samples: samples,
        includeHarWindows: false,
        includeFallWindows: false,
        includeCombinedTimeline: true,
        includeGroupedFallEvents: true,
      );

      final savedPath = await _storage.saveSession(
        sessionId: sessionId,
        fileName:
            (metadata['session_name'] as String?) ??
            (metadata['file_name'] as String?) ??
            sessionId,
        subjectId: (metadata['subject_id'] as String?) ?? 'demo_user',
        placement: (metadata['placement'] as String?) ?? 'pocket',
        datasetName: (metadata['dataset_name'] as String?) ?? 'APP_RUNTIME',
        sourceType: (metadata['source_type'] as String?) ?? 'mobile_app',
        devicePlatform:
            (metadata['device_platform'] as String?) ?? _devicePlatformValue(),
        deviceModel: metadata['device_model'] as String?,
        appVersion: metadata['app_version'] as String?,
        appBuild: metadata['app_build'] as String?,
        recordingMode:
            (metadata['recording_mode'] as String?) ?? 'live_capture',
        runtimeMode: (metadata['runtime_mode'] as String?) ?? 'mobile_live',
        samplingRateHz: _asDouble(metadata['sampling_rate_hz']),
        activityLabel: metadata['activity_label'] as String?,
        notes: (metadata['notes'] as String?) ?? '',
        samples: samples.map((s) => s.toJson()).toList(growable: false),
      );

      await _refreshSavedSessionsPath();

      if (savedPath != null) {
        await _storage.saveInferenceResult(
          filePath: savedPath,
          inferenceResult: summary.rawJson,
        );
      }

      if (!mounted) return;
      setState(() {
        _setResultVisibility(summary);
        _activeSessionId = summary.sessionId;
        _status = savedPath == null
            ? 'Bundled demo inference complete. No samples saved.'
            : 'Bundled demo inference complete. Demo session saved locally at $savedPath.';
        _selectedSection = 0;
        _recordSaveOutcome(
          outcome: savedPath == null
              ? _SaveOutcome.skipped
              : _SaveOutcome.success,
          status: savedPath == null
              ? 'Demo session did not produce a local save.'
              : 'Demo session saved to disk.',
          savedPath: savedPath,
        );
      });
      _presentFallAlertIfNeeded(summary);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _status = 'Bundled demo inference failed: $e';
        _recordSaveOutcome(
          outcome: _SaveOutcome.failed,
          status: 'Local save not completed: $e',
        );
      });
    } finally {
      if (mounted) {
        setState(() {
          _isSending = false;
        });
      }
    }
  }

  Future<bool> _submitFeedback(String feedback) async {
    final api = _api;
    if (api == null) {
      setState(() {
        _status =
            'Your account is still being prepared. Try again in a moment.';
      });
      return false;
    }
    final result = _result;
    if (result == null) {
      return false;
    }

    setState(() {
      _isSending = true;
      _status = 'Submitting feedback...';
    });

    try {
      final ack = await api.submitFeedback(
        sessionId: result.sessionId,
        subjectId: _normalisedSubjectId(),
        persistedSessionId: result.persistedSessionId,
        persistedInferenceId: result.persistedInferenceId,
        targetType: 'session',
        userFeedback: feedback,
      );

      if (!mounted) return false;
      setState(() {
        _status = ack.message;
      });

      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text(ack.message)));
      return true;
    } catch (e) {
      if (!mounted) return false;
      setState(() {
        _status = 'Feedback submission failed: $e';
      });
      return false;
    } finally {
      if (mounted) {
        setState(() {
          _isSending = false;
        });
      }
    }
  }

  Future<void> _openSavedSessions() async {
    await Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) =>
            SavedSessionsPage(initialSubjectId: _normalisedSubjectId()),
      ),
    );
  }

  Future<void> _openLatestResultDetails() async {
    final result = _result;
    if (result == null) {
      return;
    }

    final persistedSessionId = result.persistedSessionId?.trim();
    final savedPath = _lastSavedSessionPath?.trim();
    if ((savedPath == null || savedPath.isEmpty) &&
        (persistedSessionId == null || persistedSessionId.isEmpty)) {
      await _openSavedSessions();
      return;
    }

    final session = SavedSession(
      filePath: savedPath == null || savedPath.isEmpty ? null : savedPath,
      fileName: result.sessionId,
      subjectId: _normalisedSubjectId(),
      placement: _normalisedPlacement(),
      sampleCount: 0,
      savedAt: _lastSaveTimestamp ?? DateTime.now().toUtc(),
      activityLabel: result.topHarLabel,
      persistedUserId: result.persistedUserId,
      persistedSessionId:
          persistedSessionId == null || persistedSessionId.isEmpty
          ? null
          : persistedSessionId,
      persistedInferenceId: result.persistedInferenceId,
      isRemote: savedPath == null || savedPath.isEmpty,
    );

    await Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => SessionDetailPage(session: session)),
    );
  }

  bool _hasSavedResult() {
    final savedPath = _lastSavedSessionPath?.trim();
    final persistedSessionId = _result?.persistedSessionId?.trim();
    return (savedPath != null && savedPath.isNotEmpty) ||
        (persistedSessionId != null && persistedSessionId.isNotEmpty);
  }

  Future<void> _handleResultSave() async {
    if (_recorder.samples.isNotEmpty) {
      await _openSaveFlowForCurrentRecording();
      return;
    }

    if (_hasSavedResult()) {
      await _openLatestResultDetails();
      return;
    }

    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('No unsaved recording is waiting.')),
    );
  }

  String _healthText() {
    if (_health == null) {
      return 'Backend not checked';
    }
    return '${_health!.serviceName} · ${_health!.version}';
  }

  String _displayName() {
    final displayName = widget.initialIdentity.displayName?.trim();
    if (displayName != null && displayName.isNotEmpty) {
      return displayName;
    }
    return widget.initialIdentity.username.trim();
  }

  String _preferredName() {
    final rawName = _displayName();
    final withoutEmail = rawName.split('@').first;
    final firstToken = withoutEmail
        .split(RegExp(r'[\s._-]+'))
        .where((part) => part.trim().isNotEmpty)
        .firstOrNull;
    return _titleCase(firstToken ?? withoutEmail);
  }

  String _avatarInitials() {
    final parts = _displayName()
        .split(RegExp(r'[\s._@-]+'))
        .where((part) => part.trim().isNotEmpty)
        .toList(growable: false);
    if (parts.isEmpty) {
      return 'FM';
    }
    final initials = parts.take(2).map((part) => part[0].toUpperCase()).join();
    return initials.isEmpty ? 'FM' : initials;
  }

  String _greeting() {
    final hour = DateTime.now().hour;
    if (hour < 12) {
      return 'Good morning';
    }
    if (hour < 18) {
      return 'Good afternoon';
    }
    return 'Good evening';
  }

  String _titleCase(String value) {
    final trimmed = value.trim();
    if (trimmed.isEmpty) {
      return trimmed;
    }
    return '${trimmed[0].toUpperCase()}${trimmed.substring(1)}';
  }

  String _placementTitle(String placement) {
    return _titleCase(placement.replaceAll('_', ' '));
  }

  IconData _placementIcon(String placement) {
    switch (placement) {
      case 'pocket':
        return Icons.stay_current_portrait_rounded;
      case 'hand':
        return Icons.back_hand_outlined;
      case 'desk':
        return Icons.table_restaurant_outlined;
      case 'bag':
        return Icons.shopping_bag_outlined;
      default:
        return Icons.help_outline_rounded;
    }
  }

  List<SensorSample> _recentSamples({int limit = 180}) {
    final samples = _recorder.samples;
    if (samples.length <= limit) {
      return samples;
    }
    return samples.sublist(samples.length - limit);
  }

  Duration _recordingElapsed() {
    final startedAt = _recorder.recordingStartedAtUtc;
    if (startedAt != null) {
      final elapsed = DateTime.now().toUtc().difference(startedAt);
      return elapsed.isNegative ? Duration.zero : elapsed;
    }

    final durationSeconds = _recorder.durationSeconds;
    if (durationSeconds == null) {
      return Duration.zero;
    }
    return Duration(milliseconds: (durationSeconds * 1000).round());
  }

  String _timerMainText(Duration elapsed) {
    final minutes = elapsed.inMinutes.remainder(100).toString().padLeft(2, '0');
    final seconds = elapsed.inSeconds.remainder(60).toString().padLeft(2, '0');
    return '$minutes:$seconds';
  }

  String _timerFractionText(Duration elapsed) {
    final centiseconds = (elapsed.inMilliseconds ~/ 10).remainder(100);
    return '.${centiseconds.toString().padLeft(2, '0')}';
  }

  Duration _fallAlertElapsed() {
    final startedAt = _fallAlertStartedAt;
    if (startedAt == null) {
      return Duration.zero;
    }
    final elapsed = DateTime.now().toUtc().difference(startedAt);
    return elapsed.isNegative ? Duration.zero : elapsed;
  }

  Duration _fallAlertRemaining() {
    final remaining = _fallAlertDuration - _fallAlertElapsed();
    return remaining.isNegative ? Duration.zero : remaining;
  }

  int _fallCountdownSeconds() {
    final remaining = _fallAlertRemaining().inMilliseconds;
    if (remaining <= 0) {
      return 0;
    }
    return (remaining / 1000).ceil();
  }

  double _fallCountdownProgress() {
    final total = _fallAlertDuration.inMilliseconds;
    if (total <= 0) {
      return 0;
    }
    return (_fallAlertRemaining().inMilliseconds / total).clamp(0.0, 1.0);
  }

  String _fallElapsedLabel() {
    final elapsed = _fallAlertElapsed();
    final minutes = elapsed.inMinutes.remainder(100).toString().padLeft(2, '0');
    final seconds = elapsed.inSeconds.remainder(60).toString().padLeft(2, '0');
    return '$minutes:$seconds';
  }

  Future<void> _cancelFallAlert() async {
    _stopFallAlertTimer();
    if (!mounted) return;

    setState(() {
      _showFallAlert = false;
      _showResultPage = _result != null;
      _status = 'Fall alert cancelled.';
    });

    final submitted = await _submitFeedback('false_alarm');
    if (submitted && mounted) {
      await _openLatestResultDetails();
    }
  }

  Future<void> _callHelpNow() async {
    _stopFallAlertTimer();
    if (!mounted) return;

    setState(() {
      _showFallAlert = false;
      _showResultPage = _result != null;
      _status = 'Emergency response recorded.';
    });

    final submitted = await _submitFeedback('confirmed_fall');
    if (submitted && mounted) {
      await _openLatestResultDetails();
    }
  }

  String _liveActivityLabel() {
    final samples = _recentSamples(limit: 40);
    if (samples.length < 10) {
      return 'Stabilizing';
    }

    var minMagnitude = double.infinity;
    var maxMagnitude = 0.0;
    for (final sample in samples) {
      final magnitude = math.sqrt(
        sample.ax * sample.ax + sample.ay * sample.ay + sample.az * sample.az,
      );
      minMagnitude = math.min(minMagnitude, magnitude);
      maxMagnitude = math.max(maxMagnitude, magnitude);
    }

    final spread = maxMagnitude - minMagnitude;
    if (spread > 6.0) {
      return 'Moving';
    }
    if (spread > 2.4) {
      return 'Walking';
    }
    return 'Steady';
  }

  double _resultConfidence(ApiResultSummary result) {
    final fallProbability = result.topFallProbability;
    if (fallProbability != null) {
      final confidence = result.likelyFallDetected
          ? fallProbability
          : 1 - fallProbability;
      return confidence.clamp(0.0, 1.0);
    }

    final harFraction = result.topHarFraction;
    if (harFraction != null) {
      return harFraction.clamp(0.0, 1.0);
    }

    final placementConfidence = result.placementSummary.placementConfidence;
    if (placementConfidence != null) {
      return placementConfidence.clamp(0.0, 1.0);
    }

    return result.likelyFallDetected ? 0.72 : 0.88;
  }

  String _formatPercent(double value) {
    return '${(value.clamp(0.0, 1.0) * 100).round()}%';
  }

  String _formatDurationLabel(double seconds) {
    if (seconds < 60) {
      return '${seconds.round()}s';
    }
    final minutes = seconds / 60;
    if (minutes < 10) {
      return '${minutes.toStringAsFixed(1)}m';
    }
    return '${minutes.round()}m';
  }

  double _resultDurationSeconds(ApiResultSummary result) {
    final narrativeDuration = result.narrativeSummary?.totalDurationSeconds;
    if (narrativeDuration != null && narrativeDuration > 0) {
      return narrativeDuration;
    }

    var maxEnd = 0.0;
    for (final event in result.timelineEvents) {
      maxEnd = math.max(maxEnd, event.endTs);
    }
    if (maxEnd > 0) {
      return maxEnd;
    }

    for (final point in result.pointTimeline) {
      maxEnd = math.max(maxEnd, point.midpointTs);
    }
    return maxEnd > 0 ? maxEnd : 60.0;
  }

  List<_ActivityBreakdownItem> _activityBreakdown(ApiResultSummary result) {
    final totalWindows = result.harSummary.totalWindows;
    final totalDuration = _resultDurationSeconds(result);
    final entries =
        result.harSummary.labelCounts.entries
            .where((entry) => entry.value > 0)
            .toList(growable: false)
          ..sort((a, b) => b.value.compareTo(a.value));

    final colors = <Color>[_accent, _warning, const Color(0xFF6F9BB8), _border];

    if (entries.isEmpty) {
      final label = result.topHarLabel ?? 'unclassified';
      final fraction = result.topHarFraction ?? 1.0;
      return [
        _ActivityBreakdownItem(
          label: _titleCase(label.replaceAll('_', ' ')),
          fraction: fraction.clamp(0.0, 1.0),
          durationSeconds: totalDuration * fraction.clamp(0.0, 1.0),
          color: colors.first,
        ),
      ];
    }

    final visible = entries.take(4).toList(growable: false);
    return [
      for (var i = 0; i < visible.length; i++)
        _ActivityBreakdownItem(
          label: _titleCase(visible[i].key.replaceAll('_', ' ')),
          fraction: totalWindows <= 0 ? 0 : visible[i].value / totalWindows,
          durationSeconds: totalWindows <= 0
              ? 0
              : totalDuration * (visible[i].value / totalWindows),
          color: colors[math.min(i, colors.length - 1)],
        ),
    ];
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
        border: Border.all(color: textColor.withValues(alpha: 0.10)),
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
            style: GoogleFonts.interTight(
              fontSize: 12,
              fontWeight: FontWeight.w600,
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

  Widget _buildSessionBanner() {
    final estimatedRate = _recorder.estimatedSamplingRateHz;
    final isRecording = _recorder.isRecording;
    final tickerLabel = isRecording
        ? 'RECORDING LIVE'
        : _serverHealthy
        ? 'LIVE MONITOR READY'
        : 'MONITOR STANDBY';
    final sessionLabel = _activeSessionId ?? 'No active session';

    return ClipRRect(
      borderRadius: BorderRadius.circular(28),
      child: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment(-0.85, -1),
            end: Alignment(0.85, 1),
            colors: [Color(0xFF1A2420), Color(0xFF101512), Color(0xFF0C100E)],
            stops: [0, 0.60, 1],
          ),
        ),
        child: Stack(
          children: [
            Positioned(
              right: -86,
              top: -92,
              child: Container(
                width: 240,
                height: 240,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: RadialGradient(
                    colors: [
                      _waveCyan.withValues(alpha: 0.34),
                      _waveCyan.withValues(alpha: 0.00),
                    ],
                  ),
                ),
              ),
            ),
            Positioned.fill(
              child: CustomPaint(painter: _HomeWavePainter(color: _waveCyan)),
            ),
            Container(
              decoration: BoxDecoration(
                border: Border.all(color: _darkBorder),
                borderRadius: BorderRadius.circular(28),
              ),
              padding: const EdgeInsets.all(22),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      _PulseDot(
                        color: isRecording ? const Color(0xFFFF5757) : _accent,
                      ),
                      const SizedBox(width: 9),
                      Text(
                        tickerLabel,
                        style: GoogleFonts.jetBrainsMono(
                          fontSize: 11,
                          fontWeight: FontWeight.w600,
                          letterSpacing: 0.4,
                          color: _darkInk3,
                        ),
                      ),
                      const Spacer(),
                      _DarkStatusPill(
                        icon: _serverHealthy
                            ? Icons.cloud_done_outlined
                            : Icons.cloud_off_outlined,
                        label: _serverHealthy ? 'Server on' : 'Offline',
                      ),
                    ],
                  ),
                  const SizedBox(height: 30),
                  Text.rich(
                    TextSpan(
                      children: [
                        TextSpan(
                          text: 'Everything looks ',
                          style: GoogleFonts.instrumentSerif(
                            fontSize: 40,
                            height: 1.04,
                            letterSpacing: -0.8,
                            color: _darkInk,
                          ),
                        ),
                        TextSpan(
                          text: 'steady.',
                          style: GoogleFonts.instrumentSerif(
                            fontSize: 40,
                            height: 1.04,
                            letterSpacing: -0.8,
                            fontStyle: FontStyle.italic,
                            color: _darkInk,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 10),
                  Text(
                    sessionLabel,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: GoogleFonts.interTight(
                      fontSize: 14,
                      height: 1.45,
                      fontWeight: FontWeight.w500,
                      color: _darkInk2,
                    ),
                  ),
                  const SizedBox(height: 18),
                  Container(
                    height: 1,
                    color: Colors.white.withValues(alpha: 0.08),
                  ),
                  const SizedBox(height: 15),
                  LayoutBuilder(
                    builder: (context, constraints) {
                      const gap = 10.0;
                      final width = (constraints.maxWidth - gap * 2) / 3;
                      return Wrap(
                        spacing: gap,
                        runSpacing: gap,
                        children: [
                          SizedBox(
                            width: width,
                            child: _HeroStat(
                              label: 'Samples',
                              value: _recorder.sampleCount.toString(),
                            ),
                          ),
                          SizedBox(
                            width: width,
                            child: _HeroStat(
                              label: 'Rate',
                              value: estimatedRate == null
                                  ? '-- Hz'
                                  : '${estimatedRate.toStringAsFixed(1)} Hz',
                            ),
                          ),
                          SizedBox(
                            width: width,
                            child: _HeroStat(
                              label: 'Placement',
                              value: _placementTitle(_normalisedPlacement()),
                            ),
                          ),
                        ],
                      );
                    },
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _tabButton({
    required IconData icon,
    required String label,
    required bool selected,
    required VoidCallback onTap,
  }) {
    return Expanded(
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(999),
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 8),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              AnimatedContainer(
                duration: const Duration(milliseconds: 180),
                curve: Curves.easeOut,
                width: 38,
                height: 32,
                decoration: BoxDecoration(
                  color: selected ? _sageSoft : Colors.transparent,
                  borderRadius: BorderRadius.circular(999),
                ),
                child: Icon(
                  icon,
                  size: 20,
                  color: selected ? _sageDeep : _textTertiary,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                label,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: GoogleFonts.interTight(
                  fontSize: 10.5,
                  fontWeight: FontWeight.w600,
                  color: selected ? _sageDeep : _textTertiary,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildTabShell() {
    return AnimatedSwitcher(
      duration: const Duration(milliseconds: 180),
      switchInCurve: Curves.easeOut,
      switchOutCurve: Curves.easeIn,
      child: KeyedSubtree(
        key: ValueKey<int>(_selectedSection),
        child: _buildSelectedSection(),
      ),
    );
  }

  Widget _buildFloatingTabBar() {
    return ClipRRect(
      borderRadius: BorderRadius.circular(999),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
        child: Container(
          decoration: BoxDecoration(
            color: Colors.white.withValues(alpha: 0.82),
            borderRadius: BorderRadius.circular(999),
            border: Border.all(color: _border),
            boxShadow: const [
              BoxShadow(
                color: Color(0x0D141713),
                blurRadius: 6,
                offset: Offset(0, 2),
              ),
              BoxShadow(
                color: Color(0x0F141713),
                blurRadius: 40,
                offset: Offset(0, 16),
              ),
            ],
          ),
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
          child: Row(
            children: [
              _tabButton(
                icon: Icons.monitor_heart_outlined,
                label: 'Home',
                selected: _selectedSection == 0,
                onTap: () => setState(() => _selectedSection = 0),
              ),
              _tabButton(
                icon: Icons.favorite_border_rounded,
                label: 'Care',
                selected: _selectedSection == 1,
                onTap: () => setState(() => _selectedSection = 1),
              ),
              _tabButton(
                icon: Icons.folder_open_rounded,
                label: 'Sessions',
                selected: false,
                onTap: _openSavedSessions,
              ),
              _tabButton(
                icon: Icons.manage_accounts_outlined,
                label: 'Setup',
                selected: _selectedSection == 2,
                onTap: () => setState(() => _selectedSection = 2),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSelectedSection() {
    switch (_selectedSection) {
      case 2:
        return _buildSessionTab();
      default:
        return _buildMonitorTab();
    }
  }

  Widget _buildPlacementPicker() {
    return _card(
      padding: const EdgeInsets.all(18),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionTitle(
            'Placement',
            'Where the phone will sit for this session.',
          ),
          LayoutBuilder(
            builder: (context, constraints) {
              const gap = 10.0;
              final columns = constraints.maxWidth >= 560
                  ? 5
                  : constraints.maxWidth >= 360
                  ? 3
                  : 2;
              final tileWidth =
                  (constraints.maxWidth - gap * (columns - 1)) / columns;

              return Wrap(
                spacing: gap,
                runSpacing: gap,
                children: [
                  for (final placement in _placementOptions)
                    SizedBox(
                      width: tileWidth,
                      child: _PlacementTile(
                        icon: _placementIcon(placement),
                        label: _placementTitle(placement),
                        selected: _selectedPlacement == placement,
                        onTap: () {
                          setState(() {
                            _selectedPlacement = placement;
                          });
                        },
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

  Widget _buildStartSessionCard() {
    final liveSensorsSupported = _recorder.supportsLiveSensors;
    final isRecording = _recorder.isRecording;
    final hasPendingRecording = _hasPendingRecording();

    final VoidCallback? primaryAction = _isSending
        ? null
        : isRecording
        ? _stopRecording
        : hasPendingRecording
        ? _openSaveFlowForCurrentRecording
        : liveSensorsSupported
        ? _startRecording
        : _sendBundledDemoSession;

    final IconData primaryIcon = isRecording
        ? Icons.stop_rounded
        : hasPendingRecording
        ? Icons.save_alt_rounded
        : liveSensorsSupported
        ? Icons.fiber_manual_record_rounded
        : Icons.bolt_rounded;

    final String title = isRecording
        ? 'Recording now'
        : hasPendingRecording
        ? 'Ready to save'
        : 'Start a session';
    final String subtitle = isRecording
        ? 'Stop when you are ready to name and save this recording.'
        : hasPendingRecording
        ? 'A finished recording is waiting to be saved or uploaded.'
        : liveSensorsSupported
        ? 'Capture a quiet movement check from this phone.'
        : 'Live sensors are unavailable here. Use the bundled demo session.';

    return _card(
      padding: const EdgeInsets.all(18),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      style: GoogleFonts.instrumentSerif(
                        fontSize: 30,
                        height: 1.05,
                        letterSpacing: -0.6,
                        color: _textPrimary,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      subtitle,
                      style: GoogleFonts.interTight(
                        fontSize: 14,
                        height: 1.42,
                        color: _textSecondary,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 18),
              _RecordCircleButton(
                icon: primaryIcon,
                color: isRecording ? _danger : _accent,
                onPressed: primaryAction,
                loading: _isSending && !isRecording,
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildLatestEmptyCard() {
    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionTitle(
            'Latest result',
            'Session summaries appear here after a successful run.',
          ),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(22),
            decoration: BoxDecoration(
              color: _sageMist,
              borderRadius: BorderRadius.circular(18),
              border: Border.all(color: _border),
            ),
            child: Row(
              children: [
                Container(
                  width: 38,
                  height: 38,
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(color: _border),
                  ),
                  child: const Icon(
                    Icons.insights_outlined,
                    size: 20,
                    color: _sageDeep,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    'No inference result yet',
                    style: GoogleFonts.interTight(
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                      color: _textSecondary,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMonitorTab() {
    final result = _result;

    return Column(
      children: [
        _buildPlacementPicker(),
        const SizedBox(height: 16),
        _buildStartSessionCard(),
        const SizedBox(height: 16),
        if (result == null)
          _buildLatestEmptyCard()
        else
          _buildResultCardBody(result),
      ],
    );
  }

  Widget _buildResultCardBody(ApiResultSummary result) {
    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionTitle(
            'Latest result',
            result.likelyFallDetected
                ? 'A session needs review.'
                : 'The last session looks steady.',
          ),
          InkWell(
            onTap: () {
              setState(() {
                _showResultPage = true;
              });
            },
            borderRadius: BorderRadius.circular(18),
            child: Container(
              width: double.infinity,
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: _sageMist,
                borderRadius: BorderRadius.circular(18),
                border: Border.all(color: _border),
              ),
              child: Row(
                children: [
                  Container(
                    width: 40,
                    height: 40,
                    decoration: BoxDecoration(
                      color: _sageSoft,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Icon(
                      result.likelyFallDetected
                          ? Icons.priority_high_rounded
                          : Icons.check_rounded,
                      color: result.likelyFallDetected ? _danger : _sageDeep,
                      size: 22,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      result.likelyFallDetected
                          ? 'Possible fall detected'
                          : 'No fall detected',
                      style: GoogleFonts.interTight(
                        fontSize: 15,
                        fontWeight: FontWeight.w700,
                        color: _textPrimary,
                      ),
                    ),
                  ),
                  const Icon(Icons.arrow_forward_rounded, color: _sageDeep),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildResultHeader() {
    return Row(
      children: [
        IconButton(
          onPressed: () {
            setState(() {
              _showResultPage = false;
            });
          },
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
            'Session summary',
            style: GoogleFonts.interTight(
              fontSize: 20,
              fontWeight: FontWeight.w600,
              letterSpacing: -0.3,
              color: _textPrimary,
            ),
          ),
        ),
        PopupMenuButton<String>(
          tooltip: 'More',
          icon: const Icon(Icons.more_horiz_rounded),
          color: Colors.white,
          surfaceTintColor: Colors.white,
          onSelected: (value) {
            switch (value) {
              case 'confirmed_fall':
              case 'false_alarm':
              case 'uncertain':
                _submitFeedback(value);
                break;
              case 'saved':
                _openSavedSessions();
                break;
            }
          },
          itemBuilder: (context) => const [
            PopupMenuItem(value: 'confirmed_fall', child: Text('Confirm fall')),
            PopupMenuItem(value: 'false_alarm', child: Text('False alarm')),
            PopupMenuItem(value: 'uncertain', child: Text('Uncertain')),
            PopupMenuDivider(),
            PopupMenuItem(value: 'saved', child: Text('Saved sessions')),
          ],
        ),
      ],
    );
  }

  Widget _buildVerdictCard(ApiResultSummary result) {
    final confidence = _resultConfidence(result);
    final verdict = result.likelyFallDetected
        ? 'Possible fall detected'
        : 'No fall detected';
    final body = result.likelyFallDetected
        ? 'Review suggested with '
        : 'The session was clear with ';
    final tone = result.likelyFallDetected ? _danger : _accent;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(22, 26, 22, 24),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(26),
        border: Border.all(color: _border),
        boxShadow: const [
          BoxShadow(
            color: Color(0x0D141713),
            blurRadius: 6,
            offset: Offset(0, 2),
          ),
          BoxShadow(
            color: Color(0x0F141713),
            blurRadius: 40,
            offset: Offset(0, 16),
          ),
        ],
      ),
      child: Stack(
        alignment: Alignment.topCenter,
        children: [
          Positioned(
            top: -96,
            child: Container(
              width: 260,
              height: 190,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: RadialGradient(
                  colors: [tone.withValues(alpha: 0.18), Colors.transparent],
                ),
              ),
            ),
          ),
          Column(
            children: [
              Container(
                width: 76,
                height: 76,
                decoration: BoxDecoration(
                  color: tone,
                  shape: BoxShape.circle,
                  boxShadow: [
                    BoxShadow(
                      color: tone.withValues(alpha: 0.18),
                      spreadRadius: 8,
                      blurRadius: 0,
                    ),
                  ],
                ),
                child: Icon(
                  result.likelyFallDetected
                      ? Icons.priority_high_rounded
                      : Icons.check_rounded,
                  size: 38,
                  color: Colors.white,
                ),
              ),
              const SizedBox(height: 22),
              Text(
                verdict,
                textAlign: TextAlign.center,
                style: GoogleFonts.instrumentSerif(
                  fontSize: 36,
                  height: 1.04,
                  letterSpacing: -0.7,
                  color: _textPrimary,
                ),
              ),
              const SizedBox(height: 12),
              Text.rich(
                TextSpan(
                  children: [
                    TextSpan(text: body),
                    TextSpan(
                      text: '${_formatPercent(confidence)} confidence',
                      style: const TextStyle(fontWeight: FontWeight.w700),
                    ),
                    const TextSpan(text: ' across this recording.'),
                  ],
                ),
                textAlign: TextAlign.center,
                style: GoogleFonts.interTight(
                  fontSize: 15,
                  height: 1.45,
                  color: _textSecondary,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildActivityBreakdownCard(ApiResultSummary result) {
    final items = _activityBreakdown(result);

    return _card(
      padding: const EdgeInsets.all(18),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionTitle(
            'Activity breakdown',
            'How the session was classified over time.',
          ),
          _StackedActivityBar(items: items),
          const SizedBox(height: 16),
          LayoutBuilder(
            builder: (context, constraints) {
              const gap = 10.0;
              final width = (constraints.maxWidth - gap) / 2;
              return Wrap(
                spacing: gap,
                runSpacing: gap,
                children: [
                  for (final item in items.take(4))
                    SizedBox(
                      width: width,
                      child: _ActivityLegendTile(
                        item: item,
                        duration: _formatDurationLabel(item.durationSeconds),
                        percent: _formatPercent(item.fraction),
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

  Widget _buildRecommendedMessage(ApiResultSummary result) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: _sageSoft,
        borderRadius: BorderRadius.circular(22),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 38,
            height: 38,
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: Colors.white),
            ),
            child: const Icon(
              Icons.shield_outlined,
              size: 20,
              color: _sageDeep,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              result.recommendedMessage,
              style: GoogleFonts.interTight(
                fontSize: 14,
                height: 1.45,
                fontWeight: FontWeight.w600,
                color: _sageDeep,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildResultActions() {
    return LayoutBuilder(
      builder: (context, constraints) {
        final stacked = constraints.maxWidth < 380;
        final saveButton = FilledButton(
          onPressed: _handleResultSave,
          style: FilledButton.styleFrom(
            elevation: 0,
            backgroundColor: _textPrimary,
            foregroundColor: const Color(0xFFF5F3EE),
            padding: const EdgeInsets.symmetric(vertical: 16),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
            textStyle: GoogleFonts.interTight(
              fontSize: 15,
              fontWeight: FontWeight.w600,
            ),
          ),
          child: Text(_hasSavedResult() ? 'Open saved session' : 'Save'),
        );
        final detailsButton = OutlinedButton(
          onPressed: _openLatestResultDetails,
          style: OutlinedButton.styleFrom(
            elevation: 0,
            backgroundColor: Colors.white,
            foregroundColor: _textPrimary,
            side: const BorderSide(color: _border),
            padding: const EdgeInsets.symmetric(vertical: 16),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
            textStyle: GoogleFonts.interTight(
              fontSize: 15,
              fontWeight: FontWeight.w600,
            ),
          ),
          child: const Text('View details ->'),
        );

        if (stacked) {
          return Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [saveButton, const SizedBox(height: 10), detailsButton],
          );
        }

        return Row(
          children: [
            Expanded(child: saveButton),
            const SizedBox(width: 10),
            Expanded(child: detailsButton),
          ],
        );
      },
    );
  }

  Widget _fallAlertAction({
    required String label,
    required VoidCallback? onPressed,
    required bool primary,
  }) {
    final foreground = primary ? _alertBg : const Color(0xFFFFF3EF);
    final background = primary
        ? const Color(0xFFFFF7F2)
        : const Color(0xFF7C1E19);
    final border = primary
        ? Colors.white.withValues(alpha: 0.72)
        : _alertRed.withValues(alpha: 0.32);

    return SizedBox(
      width: double.infinity,
      child: FilledButton(
        onPressed: onPressed,
        style: ButtonStyle(
          elevation: const WidgetStatePropertyAll(0),
          backgroundColor: WidgetStateProperty.resolveWith((states) {
            if (states.contains(WidgetState.disabled)) {
              return background.withValues(alpha: 0.46);
            }
            return background;
          }),
          foregroundColor: WidgetStatePropertyAll(foreground),
          padding: const WidgetStatePropertyAll(
            EdgeInsets.symmetric(vertical: 17),
          ),
          shape: WidgetStatePropertyAll(
            RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(18),
              side: BorderSide(color: border),
            ),
          ),
          textStyle: WidgetStatePropertyAll(
            GoogleFonts.interTight(fontSize: 15.5, fontWeight: FontWeight.w700),
          ),
        ),
        child: Text(label),
      ),
    );
  }

  Widget _buildFallAlertScreen() {
    final name = _preferredName();
    final countdown = _fallCountdownSeconds().toString().padLeft(2, '0');

    return Scaffold(
      backgroundColor: _alertBg,
      body: Stack(
        children: [
          Positioned(
            top: -220,
            left: -120,
            right: -120,
            child: Container(
              height: 460,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: RadialGradient(
                  center: Alignment.topCenter,
                  radius: 0.82,
                  colors: [
                    _alertRed.withValues(alpha: 0.42),
                    const Color(0xFFFF3B30).withValues(alpha: 0.16),
                    Colors.transparent,
                  ],
                  stops: const [0, 0.46, 1],
                ),
              ),
            ),
          ),
          SafeArea(
            child: LayoutBuilder(
              builder: (context, constraints) {
                return Center(
                  child: ConstrainedBox(
                    constraints: const BoxConstraints(maxWidth: 500),
                    child: SingleChildScrollView(
                      padding: const EdgeInsets.fromLTRB(24, 28, 24, 22),
                      child: ConstrainedBox(
                        constraints: BoxConstraints(
                          minHeight: math.max(0, constraints.maxHeight - 50),
                        ),
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            Column(
                              crossAxisAlignment: CrossAxisAlignment.stretch,
                              children: [
                                Text(
                                  'Fall detected · ${_fallElapsedLabel()} ago',
                                  style: GoogleFonts.jetBrainsMono(
                                    fontSize: 12,
                                    fontWeight: FontWeight.w700,
                                    letterSpacing: 0.5,
                                    color: _alertRed.withValues(alpha: 0.86),
                                  ),
                                ),
                                const SizedBox(height: 22),
                                Text(
                                  'Are you okay,\n$name?',
                                  style: GoogleFonts.instrumentSerif(
                                    fontSize: 46,
                                    height: 1.02,
                                    fontWeight: FontWeight.w400,
                                    color: const Color(0xFFFFF3EF),
                                  ),
                                ),
                                const SizedBox(height: 42),
                                Center(
                                  child: SizedBox(
                                    width: 180,
                                    height: 180,
                                    child: CustomPaint(
                                      painter: _FallCountdownRingPainter(
                                        progress: _fallCountdownProgress(),
                                      ),
                                      child: Center(
                                        child: Text(
                                          countdown,
                                          style: GoogleFonts.jetBrainsMono(
                                            fontSize: 58,
                                            height: 1,
                                            fontWeight: FontWeight.w700,
                                            color: const Color(0xFFFFF7F2),
                                          ),
                                        ),
                                      ),
                                    ),
                                  ),
                                ),
                                const SizedBox(height: 42),
                                _fallAlertAction(
                                  label: "I'm fine — cancel",
                                  onPressed: _isSending
                                      ? null
                                      : () {
                                          _cancelFallAlert();
                                        },
                                  primary: true,
                                ),
                                const SizedBox(height: 12),
                                _fallAlertAction(
                                  label: 'Call help now',
                                  onPressed: _isSending
                                      ? null
                                      : () {
                                          _callHelpNow();
                                        },
                                  primary: false,
                                ),
                              ],
                            ),
                            Padding(
                              padding: const EdgeInsets.only(top: 30),
                              child: Text(
                                'Location shared with 2 contacts',
                                textAlign: TextAlign.center,
                                style: GoogleFonts.jetBrainsMono(
                                  fontSize: 11,
                                  fontWeight: FontWeight.w700,
                                  letterSpacing: 0.5,
                                  color: Colors.white.withValues(alpha: 0.58),
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildResultsScreen(ApiResultSummary result) {
    return Scaffold(
      backgroundColor: _pageBackground,
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 720),
            child: ListView(
              padding: const EdgeInsets.fromLTRB(20, 12, 20, 28),
              children: [
                _buildResultHeader(),
                const SizedBox(height: 18),
                _buildVerdictCard(result),
                const SizedBox(height: 16),
                _buildActivityBreakdownCard(result),
                const SizedBox(height: 16),
                _buildRecommendedMessage(result),
                const SizedBox(height: 18),
                _buildResultActions(),
              ],
            ),
          ),
        ),
      ),
    );
  }

  // ── Caregiver screen (page 08) ─────────────────────────────────────────────

  Widget _buildCaregiverScreen() {
    return Scaffold(
      backgroundColor: _pageBackground,
      body: SafeArea(
        child: Stack(
          children: [
            Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 920),
                child: ListView(
                  padding: const EdgeInsets.fromLTRB(20, 14, 20, 118),
                  children: [
                    _buildCaregiverHeader(),
                    const SizedBox(height: 20),
                    _buildCaregiverReassuranceCard(),
                    const SizedBox(height: 16),
                    Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Expanded(
                          child: _buildCareStatCard(
                            label: 'Activity today',
                            value: '6.2',
                            unit: 'hrs',
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: _buildCareStatCard(
                            label: 'Falls this month',
                            value: '0',
                            unit: '',
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),
                    _buildCaregiverRecentSessionsCard(),
                    const SizedBox(height: 16),
                    _buildCaregiverCallCard(),
                  ],
                ),
              ),
            ),
            Positioned(
              left: 16,
              right: 16,
              bottom: 14,
              child: Center(
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 560),
                  child: _buildFloatingTabBar(),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCaregiverHeader() {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Watching over',
                style: GoogleFonts.interTight(
                  fontSize: 13,
                  height: 1.15,
                  fontWeight: FontWeight.w500,
                  color: _textSecondary,
                ),
              ),
              const SizedBox(height: 2),
              Text(
                'Mom · Margaret',
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: GoogleFonts.instrumentSerif(
                  fontSize: 30,
                  height: 1.0,
                  letterSpacing: -0.6,
                  color: _textPrimary,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(width: 12),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 7),
          decoration: BoxDecoration(
            color: _sageSoft,
            borderRadius: BorderRadius.circular(999),
            border: Border.all(color: _border),
          ),
          child: Text(
            'ON',
            style: GoogleFonts.interTight(
              fontSize: 11,
              fontWeight: FontWeight.w700,
              letterSpacing: 0.9,
              color: _sageDeep,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildCaregiverReassuranceCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: _sageSoft,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: _accent.withValues(alpha: 0.12)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const _PulseDot(color: _accent),
              const SizedBox(width: 9),
              Text(
                'STEADY · 14 DAYS CLEAR',
                style: GoogleFonts.jetBrainsMono(
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 0.4,
                  color: _sageDeep,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Text.rich(
            TextSpan(
              children: [
                TextSpan(
                  text: 'Margaret is doing ',
                  style: GoogleFonts.instrumentSerif(
                    fontSize: 28,
                    height: 1.08,
                    letterSpacing: -0.5,
                    color: _sageDeep,
                  ),
                ),
                TextSpan(
                  text: 'well.',
                  style: GoogleFonts.instrumentSerif(
                    fontSize: 28,
                    height: 1.08,
                    letterSpacing: -0.5,
                    fontStyle: FontStyle.italic,
                    color: _sageDeep,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'No incidents in the last 2 weeks. Last activity was this morning.',
            style: GoogleFonts.interTight(
              fontSize: 14,
              height: 1.45,
              fontWeight: FontWeight.w400,
              color: _sageDeep.withValues(alpha: 0.72),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCareStatCard({
    required String label,
    required String value,
    required String unit,
  }) {
    return Container(
      padding: const EdgeInsets.all(16),
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
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label.toUpperCase(),
            style: GoogleFonts.interTight(
              fontSize: 10.5,
              fontWeight: FontWeight.w600,
              letterSpacing: 0.9,
              color: _textTertiary,
            ),
          ),
          const SizedBox(height: 10),
          Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Text(
                value,
                style: GoogleFonts.instrumentSerif(
                  fontSize: 42,
                  height: 1.0,
                  letterSpacing: -1.0,
                  color: _textPrimary,
                ),
              ),
              if (unit.isNotEmpty) ...[
                const SizedBox(width: 4),
                Padding(
                  padding: const EdgeInsets.only(bottom: 6),
                  child: Text(
                    unit,
                    style: GoogleFonts.interTight(
                      fontSize: 14,
                      fontWeight: FontWeight.w500,
                      color: _textSecondary,
                    ),
                  ),
                ),
              ],
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildCaregiverRecentSessionsCard() {
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
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(18, 18, 18, 12),
            child: Text(
              'Recent sessions',
              style: GoogleFonts.instrumentSerif(
                fontSize: 22,
                height: 1.08,
                letterSpacing: -0.4,
                color: _textPrimary,
              ),
            ),
          ),
          _buildCareSessionRow(
            icon: Icons.directions_walk,
            label: 'Morning walk',
            time: '9:14 AM today',
            tone: _accent,
            hasDividerAbove: false,
          ),
          _buildCareSessionRow(
            icon: Icons.hotel,
            label: 'Evening rest',
            time: 'Yesterday · 6:30 PM',
            tone: _textSecondary,
            hasDividerAbove: true,
          ),
          _buildCareSessionRow(
            icon: Icons.home,
            label: 'Indoor activity',
            time: '2 days ago · 11:00 AM',
            tone: const Color(0xFF6F9BB8),
            hasDividerAbove: true,
          ),
          const SizedBox(height: 4),
        ],
      ),
    );
  }

  Widget _buildCareSessionRow({
    required IconData icon,
    required String label,
    required String time,
    required Color tone,
    required bool hasDividerAbove,
  }) {
    return Column(
      children: [
        if (hasDividerAbove)
          Container(
            height: 1,
            margin: const EdgeInsets.symmetric(horizontal: 16),
            color: const Color(0xFFEEEAE0),
          ),
        Padding(
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
                child: Icon(icon, size: 18, color: tone),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      label,
                      style: GoogleFonts.interTight(
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                        color: _textPrimary,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      time,
                      style: GoogleFonts.jetBrainsMono(
                        fontSize: 11,
                        color: _textSecondary,
                      ),
                    ),
                  ],
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 10,
                  vertical: 5,
                ),
                decoration: BoxDecoration(
                  color: _sageSoft,
                  borderRadius: BorderRadius.circular(999),
                ),
                child: Text(
                  'Safe',
                  style: GoogleFonts.interTight(
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                    color: _sageDeep,
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildCaregiverCallCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: _textPrimary,
        borderRadius: BorderRadius.circular(22),
        boxShadow: const [
          BoxShadow(
            color: Color(0x1A141713),
            blurRadius: 6,
            offset: Offset(0, 2),
          ),
          BoxShadow(
            color: Color(0x1F141713),
            blurRadius: 40,
            offset: Offset(0, 16),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            width: 48,
            height: 48,
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.12),
              borderRadius: BorderRadius.circular(999),
            ),
            child: const Icon(
              Icons.phone_rounded,
              color: Colors.white,
              size: 22,
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Call Mom',
                  style: GoogleFonts.instrumentSerif(
                    fontSize: 22,
                    height: 1.08,
                    color: _darkInk,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  'Check in with Margaret',
                  style: GoogleFonts.interTight(
                    fontSize: 13,
                    fontWeight: FontWeight.w500,
                    color: _darkInk2,
                  ),
                ),
              ],
            ),
          ),
          Icon(Icons.arrow_forward_rounded, color: _darkInk2, size: 20),
        ],
      ),
    );
  }

  // ── Setup tab ──────────────────────────────────────────────────────────────

  Widget _buildSessionTab() {
    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionTitle(
            'Session Setup',
            'Account and placement used for uploaded recordings.',
          ),
          TextField(
            controller: _subjectController,
            readOnly: true,
            decoration: const InputDecoration(
              labelText: 'Subject ID',
              prefixIcon: Icon(Icons.person_outline),
              helperText: 'Linked to the signed-in account.',
            ),
          ),
          if (widget.onSignOut != null) ...[
            const SizedBox(height: 12),
            Align(
              alignment: Alignment.centerLeft,
              child: OutlinedButton.icon(
                onPressed: _isSending || _isCheckingHealth
                    ? null
                    : () async {
                        await widget.onSignOut?.call();
                      },
                icon: const Icon(Icons.logout_rounded),
                label: const Text('Sign out'),
              ),
            ),
          ],
          const SizedBox(height: 14),
          DropdownButtonFormField<String>(
            initialValue: _selectedPlacement,
            decoration: const InputDecoration(
              labelText: 'Declared Placement',
              prefixIcon: Icon(Icons.phone_android_outlined),
            ),
            items: _placementOptions
                .map(
                  (placement) => DropdownMenuItem<String>(
                    value: placement,
                    child: Text(
                      placement,
                      style: const TextStyle(fontWeight: FontWeight.w600),
                    ),
                  ),
                )
                .toList(),
            onChanged: (value) {
              if (value == null) return;
              setState(() {
                _selectedPlacement = value;
              });
            },
          ),
          const SizedBox(height: 16),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: _softBackground,
              borderRadius: BorderRadius.circular(18),
              border: Border.all(color: _border),
            ),
            child: Text(
              'Current session: ${_activeSessionId ?? '-'}',
              style: const TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w700,
                color: _textPrimary,
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ignore: unused_element
  Widget _buildChecksTab() {
    final estimatedRate = _recorder.estimatedSamplingRateHz;

    return Column(
      children: [
        _card(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _sectionTitle(
                'Backend',
                'Connection, health, and version details.',
              ),
              Wrap(
                spacing: 10,
                runSpacing: 10,
                children: [
                  _chip(
                    label: _serverHealthy ? 'Healthy' : 'Unavailable',
                    textColor: _serverHealthy ? _success : _danger,
                    icon: _serverHealthy ? Icons.cloud_done : Icons.cloud_off,
                  ),
                  _chip(
                    label: _healthText(),
                    textColor: _textPrimary,
                    background: _softBackground,
                  ),
                ],
              ),
              const SizedBox(height: 16),
              SizedBox(
                width: 220,
                child: _uniformActionButton(
                  label: _isCheckingHealth ? 'Checking...' : 'Check Backend',
                  icon: Icons.health_and_safety_outlined,
                  onPressed: _isCheckingHealth ? null : _checkHealth,
                  primary: true,
                  loading: _isCheckingHealth,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 16),
        _card(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _sectionTitle(
                'Runtime State',
                'Recorder, sample count, and current status.',
              ),
              LayoutBuilder(
                builder: (context, constraints) {
                  const gap = 12.0;
                  final cols = constraints.maxWidth >= 760
                      ? 4
                      : constraints.maxWidth >= 420
                      ? 2
                      : 1;
                  final width =
                      (constraints.maxWidth - gap * (cols - 1)) / cols;

                  return Wrap(
                    spacing: gap,
                    runSpacing: gap,
                    children: [
                      SizedBox(
                        width: width,
                        child: _metricBox(
                          label: 'Recorder',
                          value: _recorder.state.name,
                          icon: Icons.graphic_eq_rounded,
                        ),
                      ),
                      SizedBox(
                        width: width,
                        child: _metricBox(
                          label: 'Samples',
                          value: _recorder.sampleCount.toString(),
                          icon: Icons.data_usage_rounded,
                        ),
                      ),
                      SizedBox(
                        width: width,
                        child: _metricBox(
                          label: 'Rate',
                          value: estimatedRate == null
                              ? '-'
                              : '${estimatedRate.toStringAsFixed(1)} Hz',
                          icon: Icons.speed_rounded,
                        ),
                      ),
                      SizedBox(
                        width: width,
                        child: _metricBox(
                          label: 'Session',
                          value: _activeSessionId ?? '-',
                          icon: Icons.tag_outlined,
                        ),
                      ),
                    ],
                  );
                },
              ),
              const SizedBox(height: 16),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: _softBackground,
                  borderRadius: BorderRadius.circular(18),
                  border: Border.all(color: _border),
                ),
                child: Row(
                  children: [
                    const Icon(
                      Icons.info_outline_rounded,
                      size: 18,
                      color: _textSecondary,
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        _status,
                        style: const TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w700,
                          color: _textPrimary,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 16),
        _buildStorageDebugCard(),
      ],
    );
  }

  Color _saveOutcomeColor() {
    switch (_lastSaveOutcome) {
      case _SaveOutcome.success:
        return _success;
      case _SaveOutcome.partial:
        return _warning;
      case _SaveOutcome.failed:
        return _danger;
      case _SaveOutcome.skipped:
        return _warning;
      case _SaveOutcome.none:
        return _textSecondary;
    }
  }

  String _formatTimestamp(DateTime? value) {
    if (value == null) {
      return '-';
    }
    return value.toLocal().toIso8601String();
  }

  String _savedSessionsDirDisplay() {
    if (_savedSessionsDirError != null) {
      return 'Error: $_savedSessionsDirError';
    }
    return _savedSessionsDirPath ?? 'Resolving...';
  }

  Widget _storageRow({
    required String label,
    required String value,
    Color? valueColor,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(
            fontSize: 13,
            fontWeight: FontWeight.w700,
            color: _textSecondary,
          ),
        ),
        const SizedBox(height: 6),
        SelectableText(
          value,
          style: TextStyle(
            fontSize: 13,
            height: 1.35,
            color: valueColor ?? _textPrimary,
            fontWeight: FontWeight.w600,
            fontFamilyFallback: const ['SF Mono', 'Menlo', 'Consolas'],
          ),
        ),
      ],
    );
  }

  Widget _buildStorageDebugCard() {
    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionTitle(
            'Local Storage (Debug)',
            'Actual path_provider save location and most recent save result.',
          ),
          _storageRow(
            label: 'Saved sessions directory',
            value: _savedSessionsDirDisplay(),
          ),
          const SizedBox(height: 12),
          _storageRow(
            label: 'Last saved file',
            value: _lastSavedSessionPath ?? '-',
          ),
          const SizedBox(height: 12),
          _storageRow(
            label: 'Last save status',
            value: _lastSaveStatus,
            valueColor: _saveOutcomeColor(),
          ),
          const SizedBox(height: 12),
          _storageRow(
            label: 'Last save time',
            value: _formatTimestamp(_lastSaveTimestamp),
          ),
          const SizedBox(height: 16),
          SizedBox(
            width: 220,
            child: _uniformActionButton(
              label: 'Refresh Save Path',
              icon: Icons.refresh_rounded,
              onPressed: () => _refreshSavedSessionsPath(updateStatus: true),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHomeHeader() {
    return Row(
      children: [
        Container(
          width: 44,
          height: 44,
          decoration: BoxDecoration(
            color: _sageSoft,
            borderRadius: BorderRadius.circular(999),
            border: Border.all(color: _border),
          ),
          child: Center(
            child: Text(
              _avatarInitials(),
              style: GoogleFonts.jetBrainsMono(
                fontSize: 13,
                fontWeight: FontWeight.w600,
                color: _sageDeep,
              ),
            ),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                _greeting(),
                style: GoogleFonts.interTight(
                  fontSize: 13,
                  height: 1.15,
                  fontWeight: FontWeight.w500,
                  color: _textSecondary,
                ),
              ),
              const SizedBox(height: 2),
              Text(
                _preferredName(),
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: GoogleFonts.instrumentSerif(
                  fontSize: 30,
                  height: 1.0,
                  letterSpacing: -0.6,
                  color: _textPrimary,
                ),
              ),
            ],
          ),
        ),
        IconButton.filledTonal(
          onPressed: _openSavedSessions,
          style: IconButton.styleFrom(
            backgroundColor: Colors.white.withValues(alpha: 0.72),
            foregroundColor: _textPrimary,
            side: const BorderSide(color: _border),
          ),
          icon: const Icon(Icons.folder_open_rounded),
          tooltip: 'Saved Sessions',
        ),
      ],
    );
  }

  Widget _buildRecordingStatusBar() {
    final estimatedRate = _recorder.estimatedSamplingRateHz;
    final rateText = estimatedRate == null
        ? '-- Hz'
        : '${estimatedRate.toStringAsFixed(1)} Hz';

    return Row(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        _PulseDot(color: _recordRed),
        const SizedBox(width: 10),
        Expanded(
          child: Text(
            (_activeSessionId ?? 'session pending').toUpperCase(),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: GoogleFonts.jetBrainsMono(
              fontSize: 12,
              height: 1.2,
              fontWeight: FontWeight.w600,
              letterSpacing: 0.2,
              color: _darkInk2,
            ),
          ),
        ),
        const SizedBox(width: 12),
        Text(
          '$rateText / ${_placementTitle(_normalisedPlacement())}',
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          style: GoogleFonts.jetBrainsMono(
            fontSize: 12,
            height: 1.2,
            fontWeight: FontWeight.w500,
            color: _darkInk3,
          ),
        ),
      ],
    );
  }

  Widget _buildRecordingTimer() {
    final elapsed = _recordingElapsed();
    return FittedBox(
      fit: BoxFit.scaleDown,
      child: Text.rich(
        TextSpan(
          children: [
            TextSpan(
              text: _timerMainText(elapsed),
              style: GoogleFonts.instrumentSerif(
                fontSize: 96,
                height: 1,
                letterSpacing: -3.8,
                color: _darkInk,
              ),
            ),
            TextSpan(
              text: _timerFractionText(elapsed),
              style: GoogleFonts.instrumentSerif(
                fontSize: 44,
                height: 1,
                letterSpacing: -1.2,
                fontStyle: FontStyle.italic,
                color: _darkInk3,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildRecordingWaveformCard() {
    final samples = _recentSamples(limit: 180);
    return Container(
      height: 236,
      decoration: BoxDecoration(
        color: _darkCard,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: _darkBorder),
      ),
      child: Stack(
        children: [
          Positioned.fill(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(12, 38, 12, 14),
              child: CustomPaint(
                painter: _RecordingWaveformPainter(samples: samples),
              ),
            ),
          ),
          Positioned(
            left: 18,
            top: 15,
            right: 18,
            child: Row(
              children: [
                Text(
                  'MOTION WAVE',
                  style: GoogleFonts.jetBrainsMono(
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                    letterSpacing: 0.5,
                    color: _darkInk3,
                  ),
                ),
                const Spacer(),
                const _WaveLegend(label: 'x', color: _waveCyan),
                const SizedBox(width: 10),
                const _WaveLegend(label: 'y', color: _waveAmber),
                const SizedBox(width: 10),
                const _WaveLegend(label: 'z', color: _waveBlue),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRecordingActivityChips() {
    final activeLabel = _liveActivityLabel();
    final labels = <String>['Steady', 'Walking', 'Moving', 'Stabilizing'];
    return Wrap(
      alignment: WrapAlignment.center,
      spacing: 8,
      runSpacing: 8,
      children: [
        for (final label in labels)
          _RecordingActivityChip(label: label, active: label == activeLabel),
      ],
    );
  }

  Widget _buildRecordingScreen() {
    return Scaffold(
      backgroundColor: _darkBg,
      body: SafeArea(
        child: Stack(
          children: [
            Positioned.fill(
              child: ListView(
                padding: const EdgeInsets.fromLTRB(20, 16, 20, 138),
                children: [
                  _buildRecordingStatusBar(),
                  const SizedBox(height: 48),
                  Center(child: _buildRecordingTimer()),
                  const SizedBox(height: 42),
                  _buildRecordingWaveformCard(),
                  const SizedBox(height: 18),
                  _buildRecordingActivityChips(),
                  const SizedBox(height: 26),
                  Text(
                    'Keep the phone ${_normalisedPlacement()} and stay nearby.',
                    textAlign: TextAlign.center,
                    style: GoogleFonts.interTight(
                      fontSize: 14,
                      height: 1.45,
                      fontWeight: FontWeight.w500,
                      color: _darkInk3,
                    ),
                  ),
                ],
              ),
            ),
            Positioned(
              left: 0,
              right: 0,
              bottom: 24,
              child: Center(
                child: _RecordingStopButton(onPressed: _stopRecording),
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_recorder.isRecording) {
      return _buildRecordingScreen();
    }

    final result = _result;
    if (_showFallAlert && result?.likelyFallDetected == true) {
      return _buildFallAlertScreen();
    }

    if (_showResultPage && result != null) {
      return _buildResultsScreen(result);
    }

    if (_selectedSection == 1) {
      return _buildCaregiverScreen();
    }

    return Scaffold(
      backgroundColor: _pageBackground,
      body: SafeArea(
        child: Stack(
          children: [
            Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 920),
                child: ListView(
                  padding: const EdgeInsets.fromLTRB(20, 14, 20, 118),
                  children: [
                    _buildHomeHeader(),
                    const SizedBox(height: 18),
                    _buildSessionBanner(),
                    const SizedBox(height: 16),
                    _buildTabShell(),
                  ],
                ),
              ),
            ),
            Positioned(
              left: 16,
              right: 16,
              bottom: 14,
              child: Center(
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 560),
                  child: _buildFloatingTabBar(),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _PulseDot extends StatefulWidget {
  const _PulseDot({required this.color});

  final Color color;

  @override
  State<_PulseDot> createState() => _PulseDotState();
}

class _PulseDotState extends State<_PulseDot>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 1400),
  )..repeat(reverse: true);

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        final scale = 1 + (_controller.value * 0.15);
        return Transform.scale(
          scale: scale,
          child: Container(
            width: 9,
            height: 9,
            decoration: BoxDecoration(
              color: widget.color,
              shape: BoxShape.circle,
              boxShadow: [
                BoxShadow(
                  color: widget.color.withValues(alpha: 0.45),
                  blurRadius: 12,
                  spreadRadius: 1,
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

class _DarkStatusPill extends StatelessWidget {
  const _DarkStatusPill({required this.icon, required this.label});

  final IconData icon;
  final String label;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: const Color(0xFFC8C6BE)),
          const SizedBox(width: 6),
          Text(
            label,
            style: GoogleFonts.interTight(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: const Color(0xFFC8C6BE),
            ),
          ),
        ],
      ),
    );
  }
}

class _HeroStat extends StatelessWidget {
  const _HeroStat({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label.toUpperCase(),
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          style: GoogleFonts.interTight(
            fontSize: 10.5,
            height: 1.2,
            fontWeight: FontWeight.w600,
            letterSpacing: 0.9,
            color: const Color(0xFF8C8E86),
          ),
        ),
        const SizedBox(height: 6),
        Text(
          value,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          style: GoogleFonts.jetBrainsMono(
            fontSize: 17,
            height: 1,
            fontWeight: FontWeight.w600,
            letterSpacing: -0.3,
            color: const Color(0xFFF2EFE8),
          ),
        ),
      ],
    );
  }
}

class _PlacementTile extends StatelessWidget {
  const _PlacementTile({
    required this.icon,
    required this.label,
    required this.selected,
    required this.onTap,
  });

  final IconData icon;
  final String label;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(18),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 160),
        curve: Curves.easeOut,
        padding: const EdgeInsets.fromLTRB(6, 14, 6, 12),
        decoration: BoxDecoration(
          color: selected ? const Color(0xFF2C8A66) : Colors.white,
          borderRadius: BorderRadius.circular(18),
          border: Border.all(
            color: selected ? const Color(0xFF2C8A66) : const Color(0xFFE5E1D4),
          ),
          boxShadow: selected
              ? const [
                  BoxShadow(
                    color: Color(0x472C8A66),
                    blurRadius: 14,
                    offset: Offset(0, 4),
                  ),
                ]
              : null,
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              icon,
              size: 22,
              color: selected ? Colors.white : const Color(0xFF141713),
            ),
            const SizedBox(height: 9),
            Text(
              label,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: GoogleFonts.interTight(
                fontSize: 11.5,
                fontWeight: FontWeight.w600,
                color: selected ? Colors.white : const Color(0xFF141713),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _RecordCircleButton extends StatelessWidget {
  const _RecordCircleButton({
    required this.icon,
    required this.color,
    required this.onPressed,
    required this.loading,
  });

  final IconData icon;
  final Color color;
  final VoidCallback? onPressed;
  final bool loading;

  @override
  Widget build(BuildContext context) {
    final enabled = onPressed != null;
    return AnimatedOpacity(
      duration: const Duration(milliseconds: 120),
      opacity: enabled ? 1 : 0.48,
      child: Container(
        width: 64,
        height: 64,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          boxShadow: [
            BoxShadow(
              color: color.withValues(alpha: 0.28),
              blurRadius: 14,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: Material(
          color: color,
          shape: const CircleBorder(),
          child: InkWell(
            onTap: onPressed,
            customBorder: const CircleBorder(),
            child: Center(
              child: loading
                  ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: Colors.white,
                      ),
                    )
                  : Icon(icon, size: 28, color: Colors.white),
            ),
          ),
        ),
      ),
    );
  }
}

class _HomeWavePainter extends CustomPainter {
  const _HomeWavePainter({required this.color});

  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color.withValues(alpha: 0.22)
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeWidth = 2.4;

    for (var i = 0; i < 3; i++) {
      final y = size.height * (0.62 + i * 0.09);
      final path = Path()
        ..moveTo(-20, y)
        ..cubicTo(
          size.width * 0.18,
          y - 34,
          size.width * 0.30,
          y + 30,
          size.width * 0.48,
          y - 4,
        )
        ..cubicTo(
          size.width * 0.66,
          y - 38,
          size.width * 0.78,
          y + 28,
          size.width + 24,
          y - 10,
        );
      canvas.drawPath(path, paint);
    }
  }

  @override
  bool shouldRepaint(covariant _HomeWavePainter oldDelegate) {
    return oldDelegate.color != color;
  }
}

class _WaveLegend extends StatelessWidget {
  const _WaveLegend({required this.label, required this.color});

  final String label;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 7,
          height: 7,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        const SizedBox(width: 5),
        Text(
          label,
          style: GoogleFonts.jetBrainsMono(
            fontSize: 11,
            fontWeight: FontWeight.w600,
            color: const Color(0xFFC8C6BE),
          ),
        ),
      ],
    );
  }
}

class _RecordingActivityChip extends StatelessWidget {
  const _RecordingActivityChip({required this.label, required this.active});

  final String label;
  final bool active;

  @override
  Widget build(BuildContext context) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 160),
      curve: Curves.easeOut,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: active
            ? const Color(0xFF18382D)
            : Colors.white.withValues(alpha: 0.06),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(
          color: active
              ? const Color(0xFF2C8A66).withValues(alpha: 0.42)
              : Colors.white.withValues(alpha: 0.07),
        ),
      ),
      child: Text(
        label,
        style: GoogleFonts.interTight(
          fontSize: 12,
          height: 1.2,
          fontWeight: FontWeight.w600,
          color: active ? const Color(0xFF6ED0A8) : const Color(0xFF8C8E86),
        ),
      ),
    );
  }
}

class _RecordingStopButton extends StatelessWidget {
  const _RecordingStopButton({required this.onPressed});

  final VoidCallback onPressed;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 78,
      height: 78,
      decoration: const BoxDecoration(
        shape: BoxShape.circle,
        boxShadow: [
          BoxShadow(color: Color(0x33FF5757), blurRadius: 0, spreadRadius: 10),
          BoxShadow(color: Color(0x1FFF5757), blurRadius: 0, spreadRadius: 22),
          BoxShadow(color: Color(0x66FF5757), blurRadius: 18),
        ],
      ),
      child: Material(
        color: const Color(0xFFFF5757),
        shape: const CircleBorder(),
        child: InkWell(
          onTap: onPressed,
          customBorder: const CircleBorder(),
          child: const Icon(Icons.stop_rounded, color: Colors.white, size: 34),
        ),
      ),
    );
  }
}

class _RecordingWaveformPainter extends CustomPainter {
  const _RecordingWaveformPainter({required this.samples});

  final List<SensorSample> samples;

  @override
  void paint(Canvas canvas, Size size) {
    final gridPaint = Paint()
      ..color = Colors.white.withValues(alpha: 0.04)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1;

    for (var i = 1; i < 4; i++) {
      final y = size.height * i / 4;
      canvas.drawLine(Offset(0, y), Offset(size.width, y), gridPaint);
    }

    if (samples.length < 2) {
      final idlePaint = Paint()
        ..color = const Color(0xFF6ED0A8).withValues(alpha: 0.22)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2
        ..strokeCap = StrokeCap.round;
      final y = size.height / 2;
      canvas.drawLine(Offset(0, y), Offset(size.width, y), idlePaint);
      return;
    }

    _drawAxis(
      canvas: canvas,
      size: size,
      color: const Color(0xFF6ED0A8),
      valueFor: (sample) => sample.ax,
    );
    _drawAxis(
      canvas: canvas,
      size: size,
      color: const Color(0xFFE6C968),
      valueFor: (sample) => sample.ay,
    );
    _drawAxis(
      canvas: canvas,
      size: size,
      color: const Color(0xFF8AB9E0),
      valueFor: (sample) => sample.az,
    );
  }

  void _drawAxis({
    required Canvas canvas,
    required Size size,
    required Color color,
    required double Function(SensorSample sample) valueFor,
  }) {
    var maxAbs = 1.0;
    for (final sample in samples) {
      maxAbs = math.max(maxAbs, valueFor(sample).abs());
    }

    final path = Path();
    for (var i = 0; i < samples.length; i++) {
      final x = samples.length == 1
          ? 0.0
          : size.width * i / (samples.length - 1);
      final normalized = (valueFor(samples[i]) / maxAbs).clamp(-1.0, 1.0);
      final y = size.height * 0.5 - normalized * size.height * 0.36;
      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }

    final paint = Paint()
      ..color = color.withValues(alpha: 0.78)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2
      ..strokeJoin = StrokeJoin.round
      ..strokeCap = StrokeCap.round;

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant _RecordingWaveformPainter oldDelegate) {
    return oldDelegate.samples != samples;
  }
}

class _FallCountdownRingPainter extends CustomPainter {
  const _FallCountdownRingPainter({required this.progress});

  final double progress;

  @override
  void paint(Canvas canvas, Size size) {
    final strokeWidth = 9.0;
    final rect =
        Offset(strokeWidth / 2, strokeWidth / 2) &
        Size(size.width - strokeWidth, size.height - strokeWidth);

    final trackPaint = Paint()
      ..color = Colors.white.withValues(alpha: 0.08)
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth;

    final ringPaint = Paint()
      ..color = const Color(0xFFFF8378)
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeWidth = strokeWidth;

    canvas.drawOval(rect, trackPaint);
    canvas.drawArc(
      rect,
      -math.pi / 2,
      math.pi * 2 * progress.clamp(0.0, 1.0),
      false,
      ringPaint,
    );
  }

  @override
  bool shouldRepaint(covariant _FallCountdownRingPainter oldDelegate) {
    return oldDelegate.progress != progress;
  }
}

enum _SaveOutcome { none, success, partial, skipped, failed }

class _ActivityBreakdownItem {
  const _ActivityBreakdownItem({
    required this.label,
    required this.fraction,
    required this.durationSeconds,
    required this.color,
  });

  final String label;
  final double fraction;
  final double durationSeconds;
  final Color color;
}

class _StackedActivityBar extends StatelessWidget {
  const _StackedActivityBar({required this.items});

  final List<_ActivityBreakdownItem> items;

  @override
  Widget build(BuildContext context) {
    final visibleItems = items.where((item) => item.fraction > 0).toList();
    if (visibleItems.isEmpty) {
      visibleItems.add(
        const _ActivityBreakdownItem(
          label: 'Unclassified',
          fraction: 1,
          durationSeconds: 0,
          color: Color(0xFFE5E1D4),
        ),
      );
    }

    return ClipRRect(
      borderRadius: BorderRadius.circular(6),
      child: SizedBox(
        height: 12,
        child: Row(
          children: [
            for (var i = 0; i < visibleItems.length; i++) ...[
              Expanded(
                flex: math.max(1, (visibleItems[i].fraction * 1000).round()),
                child: Container(color: visibleItems[i].color),
              ),
              if (i != visibleItems.length - 1)
                Container(width: 2, color: Colors.white),
            ],
          ],
        ),
      ),
    );
  }
}

class _ActivityLegendTile extends StatelessWidget {
  const _ActivityLegendTile({
    required this.item,
    required this.duration,
    required this.percent,
  });

  final _ActivityBreakdownItem item;
  final String duration;
  final String percent;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFFE5E1D4)),
      ),
      child: Row(
        children: [
          Container(
            width: 10,
            height: 10,
            decoration: BoxDecoration(
              color: item.color,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 9),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  item.label,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.interTight(
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                    color: const Color(0xFF141713),
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  '$duration · $percent',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.jetBrainsMono(
                    fontSize: 11,
                    fontWeight: FontWeight.w500,
                    color: const Color(0xFF8E918A),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
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
