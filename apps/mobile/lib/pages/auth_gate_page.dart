import 'package:flutter/material.dart';

import '../config/runtime_config.dart';
import '../models/runtime_identity.dart';
import '../services/runtime_identity_service.dart';
import 'runtime_home_page.dart';

class AuthGatePage extends StatefulWidget {
  const AuthGatePage({super.key});

  @override
  State<AuthGatePage> createState() => _AuthGatePageState();
}

class _AuthGatePageState extends State<AuthGatePage> {
  final TextEditingController _usernameController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  final TextEditingController _subjectController = TextEditingController();
  final TextEditingController _displayNameController = TextEditingController();

  RuntimeIdentity? _identity;
  bool _checkingSavedIdentity = true;
  bool _submitting = false;
  bool _signUpMode = false;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _loadSavedIdentity();
  }

  @override
  void dispose() {
    _usernameController.dispose();
    _passwordController.dispose();
    _subjectController.dispose();
    _displayNameController.dispose();
    super.dispose();
  }

  Future<void> _loadSavedIdentity() async {
    try {
      final existing = await RuntimeIdentityService.instance
          .loadExistingIdentity();
      if (existing == null) {
        if (!mounted) return;
        setState(() {
          _checkingSavedIdentity = false;
        });
        return;
      }

      _usernameController.text = existing.username;
      _subjectController.text = existing.subjectId;
      _displayNameController.text = existing.displayName ?? '';

      final verified = await RuntimeIdentityService.instance.verifyIdentity(
        existing,
      );
      if (!mounted) return;
      setState(() {
        _identity = verified;
        _checkingSavedIdentity = false;
      });
    } catch (error) {
      await RuntimeIdentityService.instance.clearIdentity();
      if (!mounted) return;
      setState(() {
        _checkingSavedIdentity = false;
        _errorMessage = error.toString();
      });
    }
  }

  Future<void> _submit() async {
    FocusManager.instance.primaryFocus?.unfocus();

    final username = _usernameController.text.trim();
    final password = _passwordController.text;
    final subjectId = _subjectController.text.trim();

    if (username.length < 6) {
      setState(() {
        _errorMessage = 'Username must be at least 6 characters.';
      });
      return;
    }
    if (password.length < 12) {
      setState(() {
        _errorMessage = 'Password must be at least 12 characters.';
      });
      return;
    }
    if (_signUpMode && subjectId.length < 6) {
      setState(() {
        _errorMessage = 'Subject ID must be at least 6 characters.';
      });
      return;
    }

    setState(() {
      _submitting = true;
      _errorMessage = null;
    });

    try {
      final identity = _signUpMode
          ? await RuntimeIdentityService.instance.signUp(
              username: username,
              password: password,
              subjectId: subjectId,
              displayName: _displayNameController.text,
            )
          : await RuntimeIdentityService.instance.signIn(
              username: username,
              password: password,
            );

      if (!mounted) return;
      setState(() {
        _identity = identity;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _errorMessage = error.toString();
      });
    } finally {
      if (mounted) {
        setState(() {
          _submitting = false;
        });
      }
    }
  }

  Future<void> _signOut() async {
    await RuntimeIdentityService.instance.clearIdentity();
    if (!mounted) return;
    setState(() {
      _identity = null;
      _passwordController.clear();
      _errorMessage = null;
      _checkingSavedIdentity = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    final identity = _identity;
    if (identity != null) {
      return RuntimeHomePage(initialIdentity: identity, onSignOut: _signOut);
    }

    if (_checkingSavedIdentity) {
      return const Scaffold(
        body: SafeArea(child: Center(child: CircularProgressIndicator())),
      );
    }

    return Scaffold(
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(20),
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 460),
              child: Card(
                child: Padding(
                  padding: const EdgeInsets.all(22),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Text(
                        'Fall Monitor',
                        style: Theme.of(context).textTheme.headlineMedium,
                      ),
                      const SizedBox(height: 6),
                      Text(
                        runtimeApiBaseUrl,
                        style: Theme.of(context).textTheme.bodyMedium,
                      ),
                      const SizedBox(height: 20),
                      SegmentedButton<bool>(
                        segments: const [
                          ButtonSegment<bool>(
                            value: false,
                            label: Text('Sign in'),
                            icon: Icon(Icons.login_rounded),
                          ),
                          ButtonSegment<bool>(
                            value: true,
                            label: Text('Sign up'),
                            icon: Icon(Icons.person_add_alt_1_rounded),
                          ),
                        ],
                        selected: {_signUpMode},
                        onSelectionChanged: _submitting
                            ? null
                            : (selection) {
                                setState(() {
                                  _signUpMode = selection.first;
                                  _errorMessage = null;
                                });
                              },
                      ),
                      const SizedBox(height: 18),
                      TextField(
                        controller: _usernameController,
                        enabled: !_submitting,
                        textInputAction: TextInputAction.next,
                        autocorrect: false,
                        decoration: const InputDecoration(
                          labelText: 'Username',
                          prefixIcon: Icon(Icons.person_outline),
                        ),
                      ),
                      const SizedBox(height: 12),
                      TextField(
                        controller: _passwordController,
                        enabled: !_submitting,
                        obscureText: true,
                        textInputAction: _signUpMode
                            ? TextInputAction.next
                            : TextInputAction.done,
                        onSubmitted: (_) {
                          if (!_signUpMode) {
                            _submit();
                          }
                        },
                        decoration: const InputDecoration(
                          labelText: 'Password',
                          prefixIcon: Icon(Icons.lock_outline),
                        ),
                      ),
                      if (_signUpMode) ...[
                        const SizedBox(height: 12),
                        TextField(
                          controller: _subjectController,
                          enabled: !_submitting,
                          textInputAction: TextInputAction.next,
                          autocorrect: false,
                          decoration: const InputDecoration(
                            labelText: 'Subject ID',
                            prefixIcon: Icon(Icons.badge_outlined),
                          ),
                        ),
                        const SizedBox(height: 12),
                        TextField(
                          controller: _displayNameController,
                          enabled: !_submitting,
                          textInputAction: TextInputAction.done,
                          onSubmitted: (_) => _submit(),
                          decoration: const InputDecoration(
                            labelText: 'Display name',
                            prefixIcon: Icon(Icons.edit_outlined),
                          ),
                        ),
                      ],
                      if (_errorMessage != null) ...[
                        const SizedBox(height: 14),
                        Text(
                          _errorMessage!,
                          style: TextStyle(
                            color: Theme.of(context).colorScheme.error,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                      const SizedBox(height: 20),
                      FilledButton.icon(
                        onPressed: _submitting ? null : _submit,
                        icon: _submitting
                            ? const SizedBox(
                                width: 18,
                                height: 18,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                ),
                              )
                            : Icon(
                                _signUpMode
                                    ? Icons.person_add_alt_1_rounded
                                    : Icons.login_rounded,
                              ),
                        label: Text(_signUpMode ? 'Create account' : 'Sign in'),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
