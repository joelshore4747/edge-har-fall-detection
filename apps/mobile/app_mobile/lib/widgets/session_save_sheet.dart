import 'package:flutter/material.dart';

import '../config/runtime_labels.dart';

enum SessionSaveDestination { localOnly, localAndUpload }

class SessionSaveRequest {
  const SessionSaveRequest({
    required this.fileName,
    required this.category,
    required this.destination,
  });

  final String fileName;
  final String category;
  final SessionSaveDestination destination;
}

class SessionSaveSheet extends StatefulWidget {
  const SessionSaveSheet({
    super.key,
    required this.initialFileName,
    required this.initialCategory,
    required this.sampleCount,
    required this.allowUpload,
  });

  final String initialFileName;
  final String initialCategory;
  final int sampleCount;
  final bool allowUpload;

  @override
  State<SessionSaveSheet> createState() => _SessionSaveSheetState();
}

class _SessionSaveSheetState extends State<SessionSaveSheet> {
  static const Color _accent = Color(0xFFC14953);
  static const Color _textPrimary = Color(0xFF17202D);
  static const Color _textSecondary = Color(0xFF5F6878);
  static const Color _border = Color(0xFFD8DEE8);
  static const Color _softBackground = Color(0xFFF1F3F7);
  static const Color _danger = Color(0xFFD64545);

  late final TextEditingController _fileNameController;
  late String _selectedCategory;
  String? _nameError;

  @override
  void initState() {
    super.initState();
    _fileNameController = TextEditingController(text: widget.initialFileName);
    _selectedCategory = normaliseRuntimeActivityLabel(
      widget.initialCategory,
      fallback: 'other',
    );
  }

  @override
  void dispose() {
    _fileNameController.dispose();
    super.dispose();
  }

  void _submit(SessionSaveDestination destination) {
    final fileName = _fileNameController.text.trim();
    if (fileName.isEmpty) {
      setState(() {
        _nameError = 'Enter a session name before saving.';
      });
      return;
    }

    Navigator.of(context).pop(
      SessionSaveRequest(
        fileName: fileName,
        category: _selectedCategory,
        destination: destination,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final bottomInset = MediaQuery.of(context).viewInsets.bottom;

    return SafeArea(
      top: false,
      child: Padding(
        padding: EdgeInsets.fromLTRB(20, 20, 20, bottomInset + 20),
        child: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                width: 44,
                height: 5,
                decoration: BoxDecoration(
                  color: _border,
                  borderRadius: BorderRadius.circular(999),
                ),
              ),
              const SizedBox(height: 18),
              const Text(
                'Save Session',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.w800,
                  color: _textPrimary,
                ),
              ),
              const SizedBox(height: 6),
              Text(
                '${widget.sampleCount} samples recorded. Name this session, choose a category, then save it locally or upload it.',
                style: const TextStyle(
                  fontSize: 14,
                  height: 1.45,
                  color: _textSecondary,
                  fontWeight: FontWeight.w500,
                ),
              ),
              const SizedBox(height: 18),
              TextField(
                controller: _fileNameController,
                decoration: InputDecoration(
                  labelText: 'Session Name',
                  prefixIcon: const Icon(Icons.edit_note_rounded),
                  errorText: _nameError,
                ),
                textInputAction: TextInputAction.done,
                onChanged: (_) {
                  if (_nameError == null) {
                    return;
                  }
                  setState(() {
                    _nameError = null;
                  });
                },
              ),
              const SizedBox(height: 18),
              const Text(
                'Category',
                style: TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.w800,
                  color: _textPrimary,
                ),
              ),
              const SizedBox(height: 10),
              Wrap(
                spacing: 10,
                runSpacing: 10,
                children: runtimeSaveCategories
                    .map(
                      (category) => ChoiceChip(
                        label: Text(category),
                        selected: _selectedCategory == category,
                        onSelected: (_) {
                          setState(() {
                            _selectedCategory = category;
                          });
                        },
                        selectedColor: _accent.withValues(alpha: 0.16),
                        backgroundColor: _softBackground,
                        side: BorderSide(
                          color: _selectedCategory == category
                              ? _accent
                              : _border,
                        ),
                        labelStyle: TextStyle(
                          color: _selectedCategory == category
                              ? _accent
                              : _textPrimary,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    )
                    .toList(growable: false),
              ),
              if (!widget.allowUpload) ...[
                const SizedBox(height: 16),
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(14),
                  decoration: BoxDecoration(
                    color: _danger.withValues(alpha: 0.08),
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(color: _danger.withValues(alpha: 0.18)),
                  ),
                  child: const Text(
                    'Server upload is unavailable right now, so only local save is enabled.',
                    style: TextStyle(
                      fontSize: 13,
                      height: 1.4,
                      color: _danger,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              ],
              const SizedBox(height: 20),
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: () =>
                          _submit(SessionSaveDestination.localOnly),
                      icon: const Icon(Icons.save_alt_rounded),
                      label: const Text('Save Locally'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: FilledButton.icon(
                      onPressed: widget.allowUpload
                          ? () => _submit(SessionSaveDestination.localAndUpload)
                          : null,
                      icon: const Icon(Icons.cloud_upload_outlined),
                      label: const Text('Save + Upload'),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
