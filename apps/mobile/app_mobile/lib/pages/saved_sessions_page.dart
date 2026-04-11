import 'package:flutter/material.dart';

import '../models/saved_session.dart';
import '../services/session_storage_service.dart';
import 'session_detail_page.dart';

class SavedSessionsPage extends StatefulWidget {
  const SavedSessionsPage({super.key});

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

  bool _loading = true;
  String? _errorMessage;
  List<SavedSession> _sessions = const <SavedSession>[];

  @override
  void initState() {
    super.initState();
    _loadSessions();
  }

  Future<void> _loadSessions() async {
    if (!mounted) return;

    setState(() {
      _loading = true;
      _errorMessage = null;
    });

    try {
      final sessions = await _storage.listSessions();

      if (!mounted) return;
      setState(() {
        _sessions = sessions;
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _sessions = const <SavedSession>[];
        _loading = false;
        _errorMessage = 'Failed to load saved sessions: $e';
      });
    }
  }

  Future<void> _openSession(SavedSession session) async {
    await Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => SessionDetailPage(session: session)),
    );

    await _loadSessions();
  }

  Future<void> _deleteSession(SavedSession session) async {
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
      await _storage.deleteSession(session.filePath);
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

  Color _labelColor(SavedSession session) {
    final label = session.activityLabel?.trim().toLowerCase();
    switch (label) {
      case 'fall':
        return _danger;
      case 'walking':
      case 'stairs':
      case 'sitting':
        return _accent;
      case 'unknown':
      case '':
      case null:
        return _textSecondary;
      default:
        return _warning;
    }
  }

  String? _testTitleFromNotes(SavedSession session) {
    final notes = session.notes?.trim();
    if (notes == null || notes.isEmpty) {
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

  String? _preferredTestTitle(SavedSession session) {
    final savedTitle = session.testTitle?.trim();
    if (savedTitle != null && savedTitle.isNotEmpty) {
      return savedTitle;
    }

    return _testTitleFromNotes(session);
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
                  'LOCAL STORAGE',
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
                      'Browse recorded or replayed sessions, then open them for annotation and review.',
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
            'Your local session list will appear here.',
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
                  'Run a demo session or record on mobile to create your first local session file.',
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
    final testTitle = _preferredTestTitle(session);

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
                    IconButton(
                      onPressed: () => _deleteSession(session),
                      icon: const Icon(Icons.delete_outline_rounded),
                      color: _danger,
                      tooltip: 'Delete',
                    ),
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
                if (hasPlacementLabel)
                  _chip(
                    label: 'Placement label: ${session.placementLabel}',
                    textColor: _textPrimary,
                    background: _softBackground,
                  ),
                if (testTitle != null)
                  _chip(
                    label: testTitle,
                    textColor: _accent,
                    icon: Icons.assignment_outlined,
                    background: _accent.withValues(alpha: 0.10),
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
