import 'dart:convert';

import 'package:flutter/foundation.dart'
    show TargetPlatform, defaultTargetPlatform, kIsWeb;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;

import '../config/runtime_config.dart';
import '../models/sensor_sample.dart';
import '../services/runtime_api_service.dart';
import '../services/runtime_identity_service.dart';
import '../services/sensor_recorder.dart';
import '../services/session_storage_service.dart';
import 'saved_sessions_page.dart';

import '../models/api_result_summary.dart';

class RuntimeHomePage extends StatefulWidget {
  const RuntimeHomePage({super.key});

  @override
  State<RuntimeHomePage> createState() => _RuntimeHomePageState();
}

class _RuntimeHomePageState extends State<RuntimeHomePage> {
  static const Color _accent = Color(0xFFC14953); // buttons / selected tab
  static const Color _pageBackground = Color(
    0xFF848FA5,
  ); // whole app background
  static const Color _cardBackground = Color(0xFFF9FAFC);
  static const Color _softBackground = Color(0xFFF1F3F7);
  static const Color _border = Color(0xFFD8DEE8);
  static const Color _textPrimary = Color(0xFF17202D);
  static const Color _textSecondary = Color(0xFF5F6878);
  static const Color _success = Color(0xFF2FA36B);
  static const Color _danger = Color(0xFFD64545);
  static const Color _warning = Color(0xFFE79A1F);

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

  final TextEditingController _subjectController = TextEditingController();

  String _selectedPlacement = 'pocket';
  int _selectedSection = 0; // 0 monitor, 1 session, 2 checks

  String? _savedSessionsDirPath;
  String? _savedSessionsDirError;
  String? _lastSavedSessionPath;
  String _lastSaveStatus = 'No local save attempted yet.';
  _SaveOutcome _lastSaveOutcome = _SaveOutcome.none;
  DateTime? _lastSaveTimestamp;

  bool _isSending = false;
  bool _isCheckingHealth = false;
  bool _serverHealthy = false;

  String _status = 'Idle';
  String? _activeSessionId;
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

  Future<void> _bootstrapIdentityAndHealth() async {
    if (!mounted) return;

    setState(() {
      _isCheckingHealth = true;
      _status = 'Setting up your personal account...';
    });

    try {
      final identity = await RuntimeIdentityService.instance.ensureIdentity();
      _api?.dispose();
      _api = RuntimeApiService(
        baseUrl: runtimeApiBaseUrl,
        basicAuthUsername: identity.username,
        basicAuthPassword: identity.password,
      );

      if (!mounted) return;
      setState(() {
        _subjectController.text = identity.subjectId;
        _status = 'Account ready. Checking server health...';
      });

      await _runHealthCheck(updateStatus: false);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _health = null;
        _serverHealthy = false;
        _status = 'Failed to prepare your account: $e';
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

    setState(() {
      _result = null;
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
        _status = 'Recording...';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _status = 'Failed to start recording: $e';
      });
    }
  }

  Future<void> _stopRecording() async {
    try {
      await _recorder.stop();

      final path = await _recorder.saveSessionLocally(
        subjectId: _normalisedSubjectId(),
        placement: _normalisedPlacement(),
      );

      await _refreshSavedSessionsPath();
      if (!mounted) return;
      setState(() {
        if (path == null) {
          _status = 'Recording stopped. No samples saved.';
          _recordSaveOutcome(
            outcome: _SaveOutcome.skipped,
            status: 'No samples were available to save.',
          );
        } else {
          _status = 'Recording stopped. Saved locally at $path';
          _recordSaveOutcome(
            outcome: _SaveOutcome.success,
            status: 'Saved session to disk.',
            savedPath: path,
          );
        }
      });
    } catch (e) {
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

  Future<void> _sendForInference() async {
    final api = _api;
    if (api == null) {
      setState(() {
        _status =
            'Your account is still being prepared. Try again in a moment.';
      });
      return;
    }
    if (_recorder.samples.isEmpty) {
      setState(() {
        _status = 'No samples recorded yet.';
      });
      return;
    }

    final sessionId = _activeSessionId ?? _newSessionId();

    setState(() {
      _isSending = true;
      _status = 'Sending session to server...';
    });

    try {
      final summary = await api.submitSession(
        sessionId: sessionId,
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

      if (!mounted) return;
      setState(() {
        _result = summary;
        _status = 'Inference complete.';
        _selectedSection = 0;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _status = 'Inference failed: $e';
      });
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
    setState(() {
      _isSending = true;
      _status = 'Sending bundled demo session...';
      _result = null;
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
        _result = summary;
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
      return;
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

  String _healthText() {
    if (_health == null) {
      return 'Backend not checked';
    }
    return '${_health!.serviceName} · ${_health!.version}';
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

  Widget _buildSessionBanner() {
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

            final left = Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'ACTIVE SESSION',
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 0.9,
                    color: Color(0xFFF3F4F8),
                  ),
                ),
                const SizedBox(height: 12),
                Text(
                  _activeSessionId ?? 'No active session yet',
                  style: const TextStyle(
                    fontSize: 30,
                    fontWeight: FontWeight.w800,
                    letterSpacing: -1.0,
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
                const Text(
                  'Switch between monitoring, session setup, and checks using the tabs below.',
                  style: TextStyle(
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
                      label: _normalisedSubjectId(),
                      textColor: Colors.white,
                      icon: Icons.person_outline,
                      background: Colors.white.withValues(alpha: 0.16),
                    ),
                    _chip(
                      label: _normalisedPlacement(),
                      textColor: Colors.white,
                      icon: Icons.phone_android_outlined,
                      background: Colors.white.withValues(alpha: 0.16),
                    ),
                    _chip(
                      label: _serverHealthy ? 'Server Ready' : 'Server Offline',
                      textColor: Colors.white,
                      icon: _serverHealthy ? Icons.cloud_done : Icons.cloud_off,
                      background: _serverHealthy
                          ? _success.withValues(alpha: 0.32)
                          : _danger.withValues(alpha: 0.26),
                    ),
                  ],
                ),
              ],
            );

            final right = SizedBox(
              width: wide ? 220 : double.infinity,
              child: _uniformActionButton(
                label: _isCheckingHealth ? 'Checking...' : 'Check Backend',
                icon: Icons.health_and_safety_outlined,
                onPressed: _isCheckingHealth ? null : _checkHealth,
                primary: true,
                loading: _isCheckingHealth,
              ),
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

  Widget _tabButton({
    required String label,
    required bool selected,
    required VoidCallback onTap,
  }) {
    return Expanded(
      child: GestureDetector(
        onTap: onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 100),
          curve: Curves.easeOut,
          padding: const EdgeInsets.symmetric(vertical: 14),
          decoration: BoxDecoration(
            color: selected ? _accent : Colors.transparent,
            borderRadius: BorderRadius.circular(14),
            boxShadow: selected
                ? const [
                    BoxShadow(
                      color: Color(0x1AC14953),
                      blurRadius: 12,
                      offset: Offset(0, 4),
                    ),
                  ]
                : null,
          ),
          child: Center(
            child: Text(
              label,
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w800,
                color: selected ? Colors.white : _textSecondary,
                fontFamilyFallback: const [
                  'SF Pro Display',
                  'Inter',
                  'Segoe UI',
                  'Roboto',
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildTabShell() {
    return Column(
      children: [
        Container(
          decoration: BoxDecoration(
            color: Colors.white.withValues(alpha: 0.88),
            borderRadius: BorderRadius.circular(18),
            border: Border.all(color: Colors.white.withValues(alpha: 0.35)),
            boxShadow: const [
              BoxShadow(
                color: Color(0x140F172A),
                blurRadius: 18,
                offset: Offset(0, 8),
              ),
            ],
          ),
          padding: const EdgeInsets.all(6),
          child: Row(
            children: [
              _tabButton(
                label: 'Monitor',
                selected: _selectedSection == 0,
                onTap: () => setState(() => _selectedSection = 0),
              ),
              const SizedBox(width: 6),
              _tabButton(
                label: 'Session',
                selected: _selectedSection == 1,
                onTap: () => setState(() => _selectedSection = 1),
              ),
              const SizedBox(width: 6),
              _tabButton(
                label: 'Checks',
                selected: _selectedSection == 2,
                onTap: () => setState(() => _selectedSection = 2),
              ),
            ],
          ),
        ),
        const SizedBox(height: 16),
        _buildSelectedSection(),
      ],
    );
  }

  Widget _buildSelectedSection() {
    switch (_selectedSection) {
      case 1:
        return _buildSessionTab();
      case 2:
        return _buildChecksTab();
      default:
        return _buildMonitorTab();
    }
  }

  Widget _buildMonitorTab() {
    final result = _result;

    return Column(
      children: [
        _card(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _sectionTitle(
                'Quick Actions',
                'Primary controls for recording, replay, and inference.',
              ),
              LayoutBuilder(
                builder: (context, constraints) {
                  const gap = 12.0;
                  final cols = constraints.maxWidth >= 720
                      ? 4
                      : constraints.maxWidth >= 520
                      ? 2
                      : 1;
                  final width =
                      (constraints.maxWidth - gap * (cols - 1)) / cols;

                  final liveSensorsSupported = _recorder.supportsLiveSensors;
                  final isRecording = _recorder.isRecording;

                  return Wrap(
                    spacing: gap,
                    runSpacing: gap,
                    children: [
                      SizedBox(
                        width: width,
                        child: _uniformActionButton(
                          label: 'Start Recording',
                          icon: Icons.play_arrow_rounded,
                          onPressed:
                              (!liveSensorsSupported ||
                                  isRecording ||
                                  _isSending)
                              ? null
                              : _startRecording,
                          primary: liveSensorsSupported,
                        ),
                      ),
                      SizedBox(
                        width: width,
                        child: _uniformActionButton(
                          label: 'Stop Recording',
                          icon: Icons.stop_circle_outlined,
                          onPressed: (!liveSensorsSupported || !isRecording)
                              ? null
                              : _stopRecording,
                        ),
                      ),
                      SizedBox(
                        width: width,
                        child: _uniformActionButton(
                          label: 'Run Demo Session',
                          icon: Icons.bolt_outlined,
                          onPressed: (_isSending || isRecording)
                              ? null
                              : _sendBundledDemoSession,
                          primary: !liveSensorsSupported,
                        ),
                      ),
                      SizedBox(
                        width: width,
                        child: _uniformActionButton(
                          label: _isSending ? 'Sending...' : 'Send to Server',
                          icon: Icons.cloud_upload_outlined,
                          onPressed:
                              (_isSending ||
                                  isRecording ||
                                  !_serverHealthy ||
                                  _recorder.samples.isEmpty)
                              ? null
                              : _sendForInference,
                          primary: liveSensorsSupported,
                          loading: _isSending,
                        ),
                      ),
                    ],
                  );
                },
              ),
              const SizedBox(height: 16),
              Text(
                _recorder.supportsLiveSensors
                    ? 'Live recording is available on supported mobile devices.'
                    : 'This platform is using replay mode because live motion sensors are not available here.',
                style: const TextStyle(
                  fontSize: 14,
                  color: _textSecondary,
                  height: 1.4,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 16),
        if (result == null)
          _card(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _sectionTitle(
                  'Latest Result',
                  'Inference output will appear here after a successful run.',
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
                        'No inference result yet',
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
          )
        else
          _buildResultCardBody(result),
      ],
    );
  }

  Widget _buildResultCardBody(ApiResultSummary result) {
    final levelColor = _levelColor(result.warningLevel);
    final stateCounts = result.placementSummary.stateCounts;

    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionTitle(
            'Latest Result',
            'Prediction summary from the most recent session.',
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
                      label: 'Activity',
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
                      onPressed: _isSending
                          ? null
                          : () => _submitFeedback('confirmed_fall'),
                    ),
                  ),
                  SizedBox(
                    width: width,
                    child: _uniformActionButton(
                      label: 'False Alarm',
                      icon: Icons.close_rounded,
                      onPressed: _isSending
                          ? null
                          : () => _submitFeedback('false_alarm'),
                    ),
                  ),
                  SizedBox(
                    width: width,
                    child: _uniformActionButton(
                      label: 'Uncertain',
                      icon: Icons.help_outline_rounded,
                      onPressed: _isSending
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

  Widget _buildSessionTab() {
    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionTitle(
            'Session Setup',
            'Your account is assigned automatically on first launch. Only placement is editable here.',
          ),
          TextField(
            controller: _subjectController,
            readOnly: true,
            decoration: const InputDecoration(
              labelText: 'Subject ID',
              prefixIcon: Icon(Icons.person_outline),
              helperText: 'Generated for this device during first launch.',
            ),
          ),
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _pageBackground,
      appBar: AppBar(
        backgroundColor: _pageBackground,
        foregroundColor: Colors.white,
        elevation: 0,
        title: const Text(
          'Fall Monitor',
          style: TextStyle(
            fontSize: 30,
            fontWeight: FontWeight.w800,
            letterSpacing: -1.0,
            color: Colors.white,
            fontFamilyFallback: [
              'SF Pro Display',
              'Inter',
              'Segoe UI',
              'Roboto',
            ],
          ),
        ),
        actions: [
          IconButton(
            onPressed: _openSavedSessions,
            icon: const Icon(Icons.folder_open_rounded, color: Colors.white),
            tooltip: 'Saved Sessions',
          ),
          const SizedBox(width: 6),
        ],
      ),
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 1220),
            child: ListView(
              padding: const EdgeInsets.fromLTRB(16, 10, 16, 24),
              children: [
                _buildSessionBanner(),
                const SizedBox(height: 16),
                _buildTabShell(),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

enum _SaveOutcome { none, success, skipped, failed }

double? _asDouble(dynamic value) {
  if (value == null) {
    return null;
  }
  if (value is num) {
    return value.toDouble();
  }
  return double.tryParse(value.toString());
}
