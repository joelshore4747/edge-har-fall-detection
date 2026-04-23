import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'package:path_provider/path_provider.dart';

import '../config/runtime_config.dart';
import '../models/runtime_identity.dart';
import 'runtime_api_service.dart';

class RuntimeIdentityException implements Exception {
  RuntimeIdentityException(this.message);

  final String message;

  @override
  String toString() => message;
}

class RuntimeIdentityService {
  RuntimeIdentityService._();

  static final RuntimeIdentityService instance = RuntimeIdentityService._();
  static const int minPasswordLength = 8;

  static const String _identityFileName = 'runtime_identity.json';
  static const String _identityFolderName = 'runtime_auth';
  static const String _lowerAlphaNum = 'abcdefghijklmnopqrstuvwxyz0123456789';
  static const String _passwordChars =
      'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-';

  RuntimeIdentity? _currentIdentity;

  RuntimeIdentity? get currentIdentity => _currentIdentity;

  Future<RuntimeIdentity?> loadExistingIdentity() async {
    if (_currentIdentity != null) {
      return _currentIdentity;
    }

    final configuredIdentity = _configuredIdentity();
    if (configuredIdentity != null) {
      _currentIdentity = configuredIdentity;
      return configuredIdentity;
    }

    final storedIdentity = await _loadIdentity();
    if (storedIdentity == null ||
        storedIdentity.username.isEmpty ||
        storedIdentity.password.isEmpty ||
        !storedIdentity.registered) {
      return null;
    }

    _currentIdentity = storedIdentity;
    return storedIdentity;
  }

  Future<RuntimeIdentity> ensureIdentity() async {
    if (_currentIdentity != null) {
      return _currentIdentity!;
    }

    final configuredIdentity = _configuredIdentity();
    if (configuredIdentity != null) {
      _currentIdentity = configuredIdentity;
      return configuredIdentity;
    }

    final storedIdentity = await _loadIdentity();
    if (storedIdentity != null) {
      if (storedIdentity.registered) {
        _currentIdentity = storedIdentity;
        return storedIdentity;
      }
      return _registerAndPersist(storedIdentity);
    }

    final generatedIdentity = _generateIdentity();
    await _saveIdentity(generatedIdentity);
    return _registerAndPersist(generatedIdentity);
  }

  Future<RuntimeIdentity> signIn({
    required String username,
    required String password,
  }) async {
    final normalizedUsername = username.trim();
    if (normalizedUsername.isEmpty || password.isEmpty) {
      throw RuntimeIdentityException('Username and password are required.');
    }
    if (password.length < minPasswordLength) {
      throw RuntimeIdentityException(
        'Password must be at least $minPasswordLength characters.',
      );
    }

    final identity = RuntimeIdentity(
      username: normalizedUsername,
      password: password,
      subjectId: normalizedUsername,
      displayName: normalizedUsername,
      registered: true,
      provisioningMode: 'manual_login',
    );

    return verifyIdentity(identity);
  }

  Future<RuntimeIdentity> signUp({
    required String username,
    required String password,
    String? displayName,
  }) async {
    final normalizedUsername = username.trim();
    if (normalizedUsername.isEmpty || password.isEmpty) {
      throw RuntimeIdentityException('Username and password are required.');
    }
    if (password.length < minPasswordLength) {
      throw RuntimeIdentityException(
        'Password must be at least $minPasswordLength characters.',
      );
    }

    final subjectId = _generateSubjectId(normalizedUsername);
    final identity = RuntimeIdentity(
      username: normalizedUsername,
      password: password,
      subjectId: subjectId,
      displayName: _cleanText(displayName) ?? normalizedUsername,
      registered: false,
      provisioningMode: 'manual_signup',
    );
    return _registerAndPersist(identity);
  }

  Future<RuntimeIdentity> verifyIdentity(RuntimeIdentity identity) async {
    final api = RuntimeApiService(
      baseUrl: runtimeApiBaseUrl,
      basicAuthUsername: identity.username,
      basicAuthPassword: identity.password,
    );
    try {
      final profile = await api.fetchAuthProfile();
      final subjectId =
          (profile.subjectId == null || profile.subjectId!.trim().isEmpty)
          ? identity.subjectId
          : profile.subjectId!.trim();
      final resolvedIdentity = identity.copyWith(
        username: profile.username ?? identity.username,
        subjectId: subjectId,
        displayName: identity.displayName ?? subjectId,
        registered: true,
        provisioningMode: identity.provisioningMode,
        registeredAt: identity.registeredAt ?? DateTime.now().toUtc(),
      );
      await _saveIdentity(resolvedIdentity);
      _currentIdentity = resolvedIdentity;
      return resolvedIdentity;
    } on RuntimeApiException catch (error) {
      throw RuntimeIdentityException(
        'Login was rejected by the server: ${error.message}',
      );
    } finally {
      api.dispose();
    }
  }

  Future<void> clearIdentity() async {
    _currentIdentity = null;
    final file = await _identityFile();
    if (await file.exists()) {
      await file.delete();
    }
  }

  RuntimeIdentity? _configuredIdentity() {
    final username = runtimeApiUsername.trim();
    final password = runtimeApiPassword;
    if (username.isEmpty || password.isEmpty) {
      return null;
    }

    final subjectId = runtimeApiSubjectId.trim().isEmpty
        ? username
        : runtimeApiSubjectId.trim();
    return RuntimeIdentity(
      username: username,
      password: password,
      subjectId: subjectId,
      displayName: subjectId,
      registered: true,
      provisioningMode: 'configured',
    );
  }

  Future<RuntimeIdentity> _registerAndPersist(RuntimeIdentity identity) async {
    final api = RuntimeApiService(baseUrl: runtimeApiBaseUrl);
    try {
      final registered = await api.registerSelfServiceUser(
        username: identity.username,
        password: identity.password,
        subjectId: identity.subjectId,
        displayName: identity.displayName,
      );

      final resolvedIdentity = identity.copyWith(
        subjectId: registered.subjectId,
        displayName: registered.displayName ?? identity.displayName,
        registered: true,
        provisioningMode: 'self_service',
        registeredAt: DateTime.now().toUtc(),
      );
      await _saveIdentity(resolvedIdentity);
      _currentIdentity = resolvedIdentity;
      return resolvedIdentity;
    } on RuntimeApiException catch (error) {
      throw RuntimeIdentityException(
        'Failed to provision a personal account for this device: ${error.message}',
      );
    } finally {
      api.dispose();
    }
  }

  Future<RuntimeIdentity?> _loadIdentity() async {
    final file = await _identityFile();
    if (!await file.exists()) {
      return null;
    }

    final raw = await file.readAsString();
    if (raw.trim().isEmpty) {
      return null;
    }

    final decoded = jsonDecode(raw);
    if (decoded is Map<String, dynamic>) {
      return RuntimeIdentity.fromJson(decoded);
    }
    if (decoded is Map) {
      return RuntimeIdentity.fromJson(
        decoded.map((key, value) => MapEntry(key.toString(), value)),
      );
    }
    return null;
  }

  Future<void> _saveIdentity(RuntimeIdentity identity) async {
    final file = await _identityFile();
    if (!await file.parent.exists()) {
      await file.parent.create(recursive: true);
    }
    await file.writeAsString(jsonEncode(identity.toJson()));
  }

  String? _cleanText(String? value) {
    final trimmed = value?.trim();
    if (trimmed == null || trimmed.isEmpty) {
      return null;
    }
    return trimmed;
  }

  String _generateSubjectId(String username) {
    final normalized = username
        .trim()
        .toLowerCase()
        .replaceAll(RegExp(r'[^a-z0-9]+'), '_')
        .replaceAll(RegExp(r'_+'), '_')
        .replaceAll(RegExp(r'^_|_$'), '');
    final fallback = normalized.isEmpty ? 'subject' : normalized;
    final suffix = _randomString(length: 6, alphabet: _lowerAlphaNum);
    return '${fallback}_$suffix';
  }

  Future<File> _identityFile() async {
    final docs = await getApplicationDocumentsDirectory();
    return File('${docs.path}/$_identityFolderName/$_identityFileName');
  }

  RuntimeIdentity _generateIdentity() {
    final token = _randomString(length: 12, alphabet: _lowerAlphaNum);
    return RuntimeIdentity(
      username: 'app_$token',
      password: _randomString(length: 24, alphabet: _passwordChars),
      subjectId: 'subject_$token',
      displayName: 'Participant ${token.substring(token.length - 6)}',
      registered: false,
      provisioningMode: 'self_service',
    );
  }

  String _randomString({required int length, required String alphabet}) {
    final random = Random.secure();
    final codeUnits = List<int>.generate(
      length,
      (_) => alphabet.codeUnitAt(random.nextInt(alphabet.length)),
      growable: false,
    );
    return String.fromCharCodes(codeUnits);
  }
}
