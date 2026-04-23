import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

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

  bool _loading = true;
  String? _errorMessage;
  Object? _bootstrapError;
  List<SavedSession> _sessions = const <SavedSession>[];
  _SessionFilter _filter = _SessionFilter.all;

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

  String _labelText(SavedSession session) {
    final label = session.activityLabel?.trim();
    if (label == null || label.isEmpty) {
      return 'unlabelled';
    }
    return label;
  }

  String _titleCase(String value) {
    final cleaned = value.trim().replaceAll('_', ' ');
    if (cleaned.isEmpty) {
      return 'Unlabelled';
    }
    return cleaned
        .split(RegExp(r'\s+'))
        .where((part) => part.isNotEmpty)
        .map((part) => '${part[0].toUpperCase()}${part.substring(1)}')
        .join(' ');
  }

  bool _isThisWeek(SavedSession session) {
    final now = DateTime.now();
    final startOfToday = DateTime(now.year, now.month, now.day);
    final startOfWeek = startOfToday.subtract(
      Duration(days: startOfToday.weekday - 1),
    );
    return !session.savedAt.toLocal().isBefore(startOfWeek);
  }

  bool _isFlagged(SavedSession session) {
    final label = session.activityLabel?.trim().toLowerCase();
    final notes = session.notes?.trim().toLowerCase() ?? '';
    return label == 'fall' || notes.contains('flag');
  }

  List<SavedSession> _filteredSessions() {
    switch (_filter) {
      case _SessionFilter.thisWeek:
        return _sessions.where(_isThisWeek).toList(growable: false);
      case _SessionFilter.flagged:
        return _sessions.where(_isFlagged).toList(growable: false);
      case _SessionFilter.all:
        return _sessions;
    }
  }

  int _filterCount(_SessionFilter filter) {
    switch (filter) {
      case _SessionFilter.thisWeek:
        return _sessions.where(_isThisWeek).length;
      case _SessionFilter.flagged:
        return _sessions.where(_isFlagged).length;
      case _SessionFilter.all:
        return _sessions.length;
    }
  }

  String _timeLabel(DateTime value) {
    final local = value.toLocal();
    final hour = local.hour.toString().padLeft(2, '0');
    final minute = local.minute.toString().padLeft(2, '0');
    return '$hour:$minute';
  }

  String _dayLabel(DateTime value) {
    const months = [
      'JAN',
      'FEB',
      'MAR',
      'APR',
      'MAY',
      'JUN',
      'JUL',
      'AUG',
      'SEP',
      'OCT',
      'NOV',
      'DEC',
    ];
    final local = value.toLocal();
    final today = DateTime.now();
    final todayDate = DateTime(today.year, today.month, today.day);
    final valueDate = DateTime(local.year, local.month, local.day);
    if (valueDate == todayDate) {
      return 'TODAY';
    }
    if (valueDate == todayDate.subtract(const Duration(days: 1))) {
      return 'YESTERDAY';
    }
    return '${months[local.month - 1]} ${local.day}, ${local.year}';
  }

  String _dayKey(DateTime value) {
    final local = value.toLocal();
    return '${local.year}-${local.month.toString().padLeft(2, '0')}-${local.day.toString().padLeft(2, '0')}';
  }

  String _durationLabel(SavedSession session) {
    if (session.sampleCount <= 0) {
      return 'remote';
    }
    final seconds = session.sampleCount / 50;
    if (seconds < 60) {
      return '${seconds.round()}s';
    }
    final minutes = seconds / 60;
    return minutes < 10
        ? '${minutes.toStringAsFixed(1)}m'
        : '${minutes.round()}m';
  }

  Color _toneColor(SavedSession session) {
    final label = session.activityLabel?.trim().toLowerCase();
    switch (label) {
      case 'fall':
        return _danger;
      case 'standing':
      case 'transition':
        return _warning;
      case 'sitting':
      case 'lying':
        return _sky;
      case 'walking':
      case 'stairs':
        return _accent;
      default:
        return _textTertiary;
    }
  }

  IconData _sessionIcon(SavedSession session) {
    final label = session.activityLabel?.trim().toLowerCase();
    switch (label) {
      case 'fall':
        return Icons.priority_high_rounded;
      case 'walking':
      case 'stairs':
        return Icons.directions_walk_rounded;
      case 'sitting':
      case 'lying':
        return Icons.event_seat_outlined;
      case 'standing':
        return Icons.accessibility_new_rounded;
      default:
        return session.isRemoteOnly
            ? Icons.cloud_done_outlined
            : Icons.monitor_heart_outlined;
    }
  }

  String _verdictLabel(SavedSession session) {
    if (_isFlagged(session)) {
      return 'Review';
    }
    if (session.hasPersistedSession || session.hasLocalFile) {
      return 'Safe';
    }
    return 'Saved';
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

  Widget _buildHeaderBanner() {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        Expanded(
          child: Text(
            'Sessions.',
            style: GoogleFonts.instrumentSerif(
              fontSize: 42,
              height: 1.02,
              letterSpacing: -1,
              color: _textPrimary,
            ),
          ),
        ),
        const SizedBox(width: 14),
        FloatingActionButton(
          heroTag: 'saved-sessions-add',
          onPressed: () => Navigator.of(context).maybePop(),
          elevation: 0,
          backgroundColor: _accent,
          foregroundColor: Colors.white,
          shape: const CircleBorder(),
          tooltip: 'New session',
          child: const Icon(Icons.add_rounded),
        ),
      ],
    );
  }

  Widget _buildFilterPills() {
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      child: Row(
        children: [
          _FilterPill(
            label: 'All · ${_filterCount(_SessionFilter.all)}',
            selected: _filter == _SessionFilter.all,
            onTap: () => setState(() => _filter = _SessionFilter.all),
          ),
          const SizedBox(width: 8),
          _FilterPill(
            label: 'This week',
            selected: _filter == _SessionFilter.thisWeek,
            onTap: () => setState(() => _filter = _SessionFilter.thisWeek),
          ),
          const SizedBox(width: 8),
          _FilterPill(
            label: 'Flagged · ${_filterCount(_SessionFilter.flagged)}',
            selected: _filter == _SessionFilter.flagged,
            onTap: () => setState(() => _filter = _SessionFilter.flagged),
          ),
        ],
      ),
    );
  }

  Widget _buildEmptyState() {
    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionTitle(
            _sessions.isEmpty ? 'No sessions yet' : 'No matches',
            _sessions.isEmpty
                ? 'Record on mobile or submit a session to populate this history.'
                : 'Try a different filter to see more sessions.',
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
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSessionTile(SavedSession session) {
    final tone = _toneColor(session);
    final activityLabel = _titleCase(_labelText(session));
    final verdictLabel = _verdictLabel(session);
    final verdictTone = verdictLabel == 'Review' ? _danger : _sageDeep;
    final verdictBackground = verdictLabel == 'Review'
        ? const Color(0xFFF6DDD8)
        : _sageSoft;

    return InkWell(
      borderRadius: BorderRadius.circular(18),
      onTap: () => _openSession(session),
      onLongPress: session.hasLocalFile ? () => _deleteSession(session) : null,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 0, vertical: 12),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Container(
              width: 44,
              height: 44,
              decoration: BoxDecoration(
                color: tone.withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: tone.withValues(alpha: 0.16)),
              ),
              child: Icon(_sessionIcon(session), color: tone, size: 22),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '${_timeLabel(session.savedAt)} · ${_durationLabel(session)}',
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: GoogleFonts.jetBrainsMono(
                      fontSize: 12,
                      fontWeight: FontWeight.w500,
                      color: _textSecondary,
                    ),
                  ),
                  const SizedBox(height: 5),
                  Text(
                    activityLabel,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: GoogleFonts.interTight(
                      fontSize: 14,
                      height: 1.2,
                      fontWeight: FontWeight.w600,
                      color: _textPrimary,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 10),
            SizedBox(
              width: 58,
              height: 24,
              child: CustomPaint(
                painter: _SessionSparklinePainter(
                  color: tone,
                  seed: session.fileName.hashCode ^ session.sampleCount,
                ),
              ),
            ),
            const SizedBox(width: 10),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
              decoration: BoxDecoration(
                color: verdictBackground,
                borderRadius: BorderRadius.circular(999),
              ),
              child: Text(
                verdictLabel,
                style: GoogleFonts.interTight(
                  fontSize: 11.5,
                  fontWeight: FontWeight.w700,
                  color: verdictTone,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSessionsList() {
    final sessions = _filteredSessions();
    if (sessions.isEmpty) {
      return _buildEmptyState();
    }

    final children = <Widget>[];
    String? lastDayKey;
    for (final session in sessions) {
      final dayKey = _dayKey(session.savedAt);
      if (dayKey != lastDayKey) {
        if (children.isNotEmpty) {
          children.add(const SizedBox(height: 16));
        }
        children.add(_DayLabel(label: _dayLabel(session.savedAt)));
        children.add(const SizedBox(height: 7));
        lastDayKey = dayKey;
      }
      children.add(_buildSessionTile(session));
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: children,
    );
  }

  Widget _buildFloatingTabBar() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.88),
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
          _NavItem(
            icon: Icons.monitor_heart_outlined,
            label: 'Home',
            selected: false,
            onTap: () => Navigator.of(context).maybePop(),
          ),
          _NavItem(
            icon: Icons.folder_open_rounded,
            label: 'Sessions',
            selected: true,
            onTap: () {},
          ),
          _NavItem(
            icon: Icons.refresh_rounded,
            label: 'Refresh',
            selected: false,
            onTap: _loading ? null : _loadSessions,
          ),
          _NavItem(
            icon: Icons.add_rounded,
            label: 'New',
            selected: false,
            onTap: () => Navigator.of(context).maybePop(),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _pageBackground,
      body: SafeArea(
        child: Stack(
          children: [
            Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 760),
                child: _loading && _sessions.isEmpty
                    ? const Center(
                        child: CircularProgressIndicator(color: _accent),
                      )
                    : ListView(
                        padding: const EdgeInsets.fromLTRB(20, 14, 20, 118),
                        children: [
                          _buildHeaderBanner(),
                          const SizedBox(height: 18),
                          _buildFilterPills(),
                          if (_errorMessage != null) ...[
                            const SizedBox(height: 12),
                            _ErrorBanner(message: _errorMessage!),
                          ],
                          const SizedBox(height: 22),
                          _buildSessionsList(),
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

enum _SessionFilter { all, thisWeek, flagged }

class _FilterPill extends StatelessWidget {
  const _FilterPill({
    required this.label,
    required this.selected,
    required this.onTap,
  });

  final String label;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(999),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 160),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 9),
        decoration: BoxDecoration(
          color: selected ? const Color(0xFF141713) : Colors.white,
          borderRadius: BorderRadius.circular(999),
          border: Border.all(
            color: selected ? const Color(0xFF141713) : const Color(0xFFE5E1D4),
          ),
        ),
        child: Text(
          label,
          style: GoogleFonts.interTight(
            fontSize: 12,
            fontWeight: FontWeight.w700,
            color: selected ? const Color(0xFFF5F3EE) : const Color(0xFF141713),
          ),
        ),
      ),
    );
  }
}

class _DayLabel extends StatelessWidget {
  const _DayLabel({required this.label});

  final String label;

  @override
  Widget build(BuildContext context) {
    return Text(
      label,
      style: GoogleFonts.interTight(
        fontSize: 11,
        height: 1.2,
        fontWeight: FontWeight.w700,
        letterSpacing: 1.1,
        color: const Color(0xFF8E918A),
      ),
    );
  }
}

class _ErrorBanner extends StatelessWidget {
  const _ErrorBanner({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFFF6DDD8),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFFEBC6C0)),
      ),
      child: Text(
        message,
        style: GoogleFonts.interTight(
          fontSize: 13,
          height: 1.35,
          fontWeight: FontWeight.w600,
          color: const Color(0xFFC14C41),
        ),
      ),
    );
  }
}

class _NavItem extends StatelessWidget {
  const _NavItem({
    required this.icon,
    required this.label,
    required this.selected,
    required this.onTap,
  });

  final IconData icon;
  final String label;
  final bool selected;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
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
                width: 38,
                height: 32,
                decoration: BoxDecoration(
                  color: selected
                      ? const Color(0xFFDCEBE3)
                      : Colors.transparent,
                  borderRadius: BorderRadius.circular(999),
                ),
                child: Icon(
                  icon,
                  size: 20,
                  color: selected
                      ? const Color(0xFF1A5A44)
                      : const Color(0xFF8E918A),
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
                  color: selected
                      ? const Color(0xFF1A5A44)
                      : const Color(0xFF8E918A),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _SessionSparklinePainter extends CustomPainter {
  const _SessionSparklinePainter({required this.color, required this.seed});

  final Color color;
  final int seed;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color.withValues(alpha: 0.82)
      ..strokeWidth = 1.8
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;

    final path = Path();
    final phase = (seed.abs() % 628) / 100;
    for (var i = 0; i < 18; i++) {
      final x = size.width * i / 17;
      final wave =
          math.sin(i * 0.85 + phase) * 0.35 +
          math.sin(i * 0.32 + phase * 0.7) * 0.18;
      final y = size.height * (0.52 - wave);
      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant _SessionSparklinePainter oldDelegate) {
    return oldDelegate.color != color || oldDelegate.seed != seed;
  }
}
