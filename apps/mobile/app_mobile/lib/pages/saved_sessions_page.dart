import 'package:flutter/material.dart';

import '../config/runtime_config.dart';
import '../models/saved_session.dart';
import '../services/runtime_api_service.dart';
import '../services/runtime_identity_service.dart';
import '../services/session_storage_service.dart';
import 'session_detail_page.dart';

class SavedSessionsPage extends StatefulWidget {
  const SavedSessionsPage({super.key, this.initialSubjectId});

  final String? initialSubjectId;

  @override
  State<SavedSessionsPage> createState() => _SavedSessionsPageState();
}

class _SavedSessionsPageState extends State<SavedSessionsPage> {
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
  RuntimeApiService? _api;

  bool _loading = true;
  String? _errorMessage;
  Object? _bootstrapError;
  List<SavedSession> _sessions = const <SavedSession>[];

  @override
  void initState() {
    super.initState();
    _bootstrapAndLoadSessions();
  }

  @override
  void dispose() {
    _api?.dispose();
    super.dispose();
  }

  Future<void> _bootstrapAndLoadSessions() async {
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

    await _loadSessions();
  }

  Future<void> _loadSessions() async {
    if (!mounted) return;

    setState(() {
      _loading = true;
      _errorMessage = null;
    });

    List<SavedSession> localSessions = const <SavedSession>[];
    Object? localError;
    try {
      localSessions = await _storage.listSessions();
    } catch (e) {
      localError = e;
    }

    List<SavedSession> remoteSessions = const <SavedSession>[];
    Object? remoteError;
    try {
      final api = _api;
      if (api == null) {
        throw _bootstrapError ?? StateError('Your account is not ready yet.');
      }
      final persisted = await api.listPersistedSessions(
        subjectId: widget.initialSubjectId,
        limit: 100,
        offset: 0,
      );
      remoteSessions = persisted
          .map(SavedSession.fromPersistedSummary)
          .toList(growable: false);
    } catch (e) {
      remoteError = e;
    }

    if (!mounted) return;
    setState(() {
      _sessions = _mergeSessions(
        localSessions: localSessions,
        remoteSessions: remoteSessions,
      );
      _loading = false;
      if (localError != null && remoteError != null) {
        _errorMessage =
            'Failed to load local sessions ($localError) and remote history ($remoteError).';
      } else if (localError != null) {
        _errorMessage = 'Failed to load local sessions: $localError';
      } else if (remoteError != null) {
        _errorMessage = 'Remote history unavailable: $remoteError';
      } else {
        _errorMessage = null;
      }
    });
  }

  Future<void> _openSession(SavedSession session) async {
    await Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => SessionDetailPage(session: session)),
    );

    await _loadSessions();
  }

  Future<void> _deleteSession(SavedSession session) async {
    if (!session.hasLocalFile) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('This session exists only on the server.'),
        ),
      );
      return;
    }

    final confirmed =
        await showDialog<bool>(
          context: context,
          builder: (context) {
            return AlertDialog(
              title: const Text('Delete session'),
              content: Text(
                'Delete "${session.fileName}"?\n\nThis removes the saved local session file.',
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.of(context).pop(false),
                  child: const Text('Cancel'),
                ),
                FilledButton(
                  onPressed: () => Navigator.of(context).pop(true),
                  child: const Text('Delete'),
                ),
              ],
            );
          },
        ) ??
        false;

    if (!confirmed) {
      return;
    }

    try {
      await _storage.deleteSession(session.filePath!);
      await _loadSessions();

      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Deleted ${session.fileName}')));
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Failed to delete session: $e')));
    }
  }

  String _fmt(DateTime dt) {
    return '${dt.year}-${dt.month.toString().padLeft(2, '0')}-${dt.day.toString().padLeft(2, '0')} '
        '${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}';
  }

  String _labelText(SavedSession session) {
    final label = session.activityLabel?.trim();
    if (label == null || label.isEmpty) {
      return 'unlabelled';
    }
    return label;
  }

  List<SavedSession> _mergeSessions({
    required List<SavedSession> localSessions,
    required List<SavedSession> remoteSessions,
  }) {
    final merged = <SavedSession>[];
    final localByPersistedId = <String, SavedSession>{};
    final unmatchedLocal = <SavedSession>[];

    for (final session in localSessions) {
      final persistedId = session.persistedSessionId;
      if (persistedId != null && persistedId.trim().isNotEmpty) {
        localByPersistedId[persistedId] = session;
      } else {
        unmatchedLocal.add(session);
      }
    }

    for (final remote in remoteSessions) {
      final persistedId = remote.persistedSessionId;
      final localMatch = persistedId == null
          ? null
          : localByPersistedId.remove(persistedId);

      if (localMatch == null) {
        merged.add(remote);
        continue;
      }

      final savedAt = remote.savedAt.isAfter(localMatch.savedAt)
          ? remote.savedAt
          : localMatch.savedAt;

      merged.add(
        localMatch.copyWith(
          subjectId: remote.subjectId,
          placement: remote.placement,
          sampleCount: remote.sampleCount,
          savedAt: savedAt,
          activityLabel: localMatch.activityLabel ?? remote.activityLabel,
          notes: localMatch.notes ?? remote.notes,
          persistedUserId: remote.persistedUserId,
          persistedSessionId: remote.persistedSessionId,
          persistedInferenceId: remote.persistedInferenceId,
          isRemote: true,
        ),
      );
    }

    merged.addAll(localByPersistedId.values);
    merged.addAll(unmatchedLocal);
    merged.sort((a, b) => b.savedAt.compareTo(a.savedAt));
    return merged;
  }

  Color _labelColor(SavedSession session) {
    final label = session.activityLabel?.trim().toLowerCase();
    switch (label) {
      case 'fall':
        return _danger;
      case 'walking':
      case 'stairs':
      case 'sitting':
      case 'standing':
      case 'lying':
      case 'transition':
        return _accent;
      case 'unknown':
      case '':
      case null:
        return _textSecondary;
      default:
        return _warning;
    }
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

  Widget _card({
    required Widget child,
    EdgeInsets padding = const EdgeInsets.all(20),
  }) {
    return Container(
      decoration: BoxDecoration(
        color: _cardBackground,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: _border),
        boxShadow: const [
          BoxShadow(
            color: Color(0x0D0F172A),
            blurRadius: 20,
            offset: Offset(0, 10),
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

  Widget _buildHeaderBanner() {
    return Container(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(28),
        gradient: const LinearGradient(
          colors: [Color(0xFF9099AD), Color(0xFF7A859A)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        boxShadow: const [
          BoxShadow(
            color: Color(0x1E5F697A),
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
                  'SAVED + SYNCED',
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 0.9,
                    color: Color(0xFFF3F4F8),
                  ),
                ),
                const SizedBox(height: 12),
                Text(
                  _sessions.isEmpty
                      ? 'No saved sessions yet'
                      : '${_sessions.length} saved session${_sessions.length == 1 ? '' : 's'}',
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
                Text(
                  _errorMessage ??
                      'Browse local recordings and server-synced sessions, then open them for annotation and review.',
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
                      label: _loading ? 'Refreshing' : 'Ready',
                      textColor: Colors.white,
                      icon: _loading
                          ? Icons.sync_rounded
                          : Icons.check_circle_outline,
                      background: _loading
                          ? Colors.white.withValues(alpha: 0.16)
                          : _success.withValues(alpha: 0.28),
                    ),
                    _chip(
                      label: 'Tap a session to review',
                      textColor: Colors.white,
                      icon: Icons.touch_app_outlined,
                      background: Colors.white.withValues(alpha: 0.16),
                    ),
                  ],
                ),
              ],
            );

            final right = SizedBox(
              width: wide ? 220 : double.infinity,
              child: ElevatedButton.icon(
                onPressed: _loading ? null : _loadSessions,
                style: ElevatedButton.styleFrom(
                  backgroundColor: _accent,
                  foregroundColor: Colors.white,
                  disabledBackgroundColor: const Color(0xFFDDE2EA),
                  disabledForegroundColor: const Color(0xFF98A2B3),
                  minimumSize: const Size.fromHeight(52),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16),
                  ),
                  elevation: 0,
                  textStyle: const TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w700,
                    fontFamilyFallback: [
                      'SF Pro Display',
                      'Inter',
                      'Segoe UI',
                      'Roboto',
                    ],
                  ),
                ),
                icon: _loading
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: Colors.white,
                        ),
                      )
                    : const Icon(Icons.refresh_rounded),
                label: Text(_loading ? 'Refreshing...' : 'Refresh Sessions'),
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

  Widget _buildEmptyState() {
    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionTitle(
            'Saved Sessions',
            'Local files and synced server sessions appear here.',
          ),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(28),
            decoration: BoxDecoration(
              color: _softBackground,
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: _border),
            ),
            child: const Column(
              children: [
                Icon(
                  Icons.folder_open_rounded,
                  size: 48,
                  color: _textSecondary,
                ),
                SizedBox(height: 12),
                Text(
                  'No saved sessions yet',
                  style: TextStyle(
                    fontSize: 17,
                    fontWeight: FontWeight.w800,
                    color: _textPrimary,
                  ),
                ),
                SizedBox(height: 6),
                Text(
                  'Record on mobile or submit a session to the backend to populate this history.',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 14,
                    height: 1.45,
                    color: _textSecondary,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSessionTile(SavedSession session) {
    final activityChipColor = _labelColor(session);
    final activityLabel = _labelText(session);
    final hasPlacementLabel =
        session.placementLabel != null &&
        session.placementLabel!.trim().isNotEmpty;

    return InkWell(
      borderRadius: BorderRadius.circular(24),
      onTap: () => _openSession(session),
      child: _card(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            LayoutBuilder(
              builder: (context, constraints) {
                final wide = constraints.maxWidth >= 680;

                final info = Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      session.fileName,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        fontSize: 19,
                        fontWeight: FontWeight.w800,
                        letterSpacing: -0.5,
                        color: _textPrimary,
                        fontFamilyFallback: [
                          'SF Pro Display',
                          'Inter',
                          'Segoe UI',
                          'Roboto',
                        ],
                      ),
                    ),
                    const SizedBox(height: 10),
                    Wrap(
                      spacing: 10,
                      runSpacing: 10,
                      children: [
                        _chip(
                          label: session.subjectId,
                          textColor: _textPrimary,
                          icon: Icons.person_outline,
                          background: _softBackground,
                        ),
                        _chip(
                          label: session.placement,
                          textColor: _accent,
                          icon: Icons.phone_android_outlined,
                          background: _accent.withValues(alpha: 0.10),
                        ),
                        _chip(
                          label: '${session.sampleCount} samples',
                          textColor: _textSecondary,
                          icon: Icons.data_usage_rounded,
                          background: _softBackground,
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    Text(
                      'Saved: ${_fmt(session.savedAt)}',
                      style: const TextStyle(
                        fontSize: 14,
                        color: _textSecondary,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                );

                final trailing = Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    if (session.hasLocalFile)
                      IconButton(
                        onPressed: () => _deleteSession(session),
                        icon: const Icon(Icons.delete_outline_rounded),
                        color: _danger,
                        tooltip: 'Delete local file',
                      )
                    else
                      const Icon(Icons.cloud_done_outlined, color: _success),
                    const Icon(
                      Icons.chevron_right_rounded,
                      color: _textSecondary,
                    ),
                  ],
                );

                if (!wide) {
                  return Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      info,
                      const SizedBox(height: 8),
                      Align(alignment: Alignment.centerRight, child: trailing),
                    ],
                  );
                }

                return Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(child: info),
                    const SizedBox(width: 12),
                    trailing,
                  ],
                );
              },
            ),
            const SizedBox(height: 14),
            Wrap(
              spacing: 10,
              runSpacing: 10,
              children: [
                _chip(
                  label: activityLabel,
                  textColor: activityChipColor,
                  background: activityChipColor.withValues(alpha: 0.10),
                ),
                if (session.hasPersistedSession)
                  _chip(
                    label: 'Server synced',
                    textColor: _success,
                    icon: Icons.cloud_done_outlined,
                    background: _success.withValues(alpha: 0.10),
                  ),
                if (session.hasLocalFile)
                  _chip(
                    label: 'Local file',
                    textColor: _textSecondary,
                    icon: Icons.folder_open_rounded,
                    background: _softBackground,
                  ),
                if (hasPlacementLabel)
                  _chip(
                    label: 'Placement label: ${session.placementLabel}',
                    textColor: _textPrimary,
                    background: _softBackground,
                  ),
                if (session.notes != null && session.notes!.trim().isNotEmpty)
                  _chip(
                    label: 'Has notes',
                    textColor: _warning,
                    icon: Icons.notes_rounded,
                    background: _warning.withValues(alpha: 0.10),
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSessionsList() {
    if (_sessions.isEmpty) {
      return _buildEmptyState();
    }

    return Column(
      children: _sessions
          .map(
            (session) => Padding(
              padding: const EdgeInsets.only(bottom: 16),
              child: _buildSessionTile(session),
            ),
          )
          .toList(growable: false),
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
          'Saved Sessions',
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
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 1220),
            child: _loading && _sessions.isEmpty
                ? const Center(child: CircularProgressIndicator())
                : ListView(
                    padding: const EdgeInsets.fromLTRB(16, 10, 16, 24),
                    children: [
                      _buildHeaderBanner(),
                      const SizedBox(height: 16),
                      _buildSessionsList(),
                    ],
                  ),
          ),
        ),
      ),
    );
  }
}
