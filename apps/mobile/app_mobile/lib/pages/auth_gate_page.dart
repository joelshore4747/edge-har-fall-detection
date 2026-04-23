import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../models/runtime_identity.dart';
import '../services/runtime_identity_service.dart';
import 'runtime_home_page.dart';

class _AuthColor {
  static const bg = Color(0xFFF4F1E9);
  static const surface = Color(0xFFFFFFFF);
  static const surfaceAlt = Color(0xFFEDE9DF);
  static const ink = Color(0xFF141713);
  static const ink2 = Color(0xFF5A5E58);
  static const ink3 = Color(0xFF8E918A);
  static const hair = Color(0xFFE5E1D4);
  static const sage = Color(0xFF2C8A66);
  static const sageDeep = Color(0xFF1A5A44);
  static const sageSoft = Color(0xFFDCEBE3);
  static const coral = Color(0xFFC14C41);
  static const coralSoft = Color(0xFFF6DDD8);
}

const String _appVersionLabel = 'v1.0.0+1';

class AuthGatePage extends StatefulWidget {
  const AuthGatePage({super.key});

  @override
  State<AuthGatePage> createState() => _AuthGatePageState();
}

class _AuthGatePageState extends State<AuthGatePage> {
  final TextEditingController _usernameController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
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

    if (username.length < 6) {
      setState(() {
        _errorMessage = 'Username must be at least 6 characters.';
      });
      return;
    }
    if (password.length < RuntimeIdentityService.minPasswordLength) {
      setState(() {
        _errorMessage =
            'Password must be at least ${RuntimeIdentityService.minPasswordLength} characters.';
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
        backgroundColor: _AuthColor.bg,
        body: SafeArea(
          child: Center(
            child: CircularProgressIndicator(color: _AuthColor.sage),
          ),
        ),
      );
    }

    return Scaffold(
      backgroundColor: _AuthColor.bg,
      body: SafeArea(
        child: LayoutBuilder(
          builder: (context, constraints) {
            return SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(20, 28, 20, 24),
              child: ConstrainedBox(
                constraints: BoxConstraints(minHeight: constraints.maxHeight),
                child: Center(
                  child: ConstrainedBox(
                    constraints: const BoxConstraints(maxWidth: 430),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        const _ShieldWell(),
                        const SizedBox(height: 26),
                        Text(
                          'A quiet watch\nover the people\nyou love.',
                          textAlign: TextAlign.left,
                          style: GoogleFonts.instrumentSerif(
                            fontSize: 42,
                            height: 1.02,
                            letterSpacing: 0,
                            color: _AuthColor.ink,
                          ),
                        ),
                        const SizedBox(height: 28),
                        _AuthTextFieldCard(
                          label: 'Username',
                          controller: _usernameController,
                          enabled: !_submitting,
                          textInputAction: TextInputAction.next,
                          autofillHints: _signUpMode
                              ? const [AutofillHints.newUsername]
                              : const [AutofillHints.username],
                          hintText: 'margaret.white',
                        ),
                        const SizedBox(height: 10),
                        _AuthTextFieldCard(
                          label: 'Password',
                          controller: _passwordController,
                          enabled: !_submitting,
                          obscureText: true,
                          textInputAction: _signUpMode
                              ? TextInputAction.next
                              : TextInputAction.done,
                          autofillHints: _signUpMode
                              ? const [AutofillHints.newPassword]
                              : const [AutofillHints.password],
                          hintText: 'Minimum 8 characters',
                          onSubmitted: (_) {
                            if (!_signUpMode) {
                              _submit();
                            }
                          },
                        ),
                        AnimatedSwitcher(
                          duration: const Duration(milliseconds: 180),
                          switchInCurve: Curves.easeOut,
                          switchOutCurve: Curves.easeIn,
                          child: _signUpMode
                              ? Padding(
                                  key: const ValueKey('display-name-field'),
                                  padding: const EdgeInsets.only(top: 10),
                                  child: _AuthTextFieldCard(
                                    label: 'Display name',
                                    controller: _displayNameController,
                                    enabled: !_submitting,
                                    textInputAction: TextInputAction.done,
                                    autofillHints: const [AutofillHints.name],
                                    hintText: 'Margaret',
                                    onSubmitted: (_) => _submit(),
                                  ),
                                )
                              : const SizedBox.shrink(
                                  key: ValueKey('no-display-name-field'),
                                ),
                        ),
                        if (_errorMessage != null) ...[
                          const SizedBox(height: 14),
                          _ErrorNotice(message: _errorMessage!),
                        ],
                        const SizedBox(height: 18),
                        SizedBox(
                          height: 54,
                          child: FilledButton(
                            onPressed: _submitting ? null : _submit,
                            style: FilledButton.styleFrom(
                              elevation: 0,
                              backgroundColor: _AuthColor.ink,
                              foregroundColor: const Color(0xFFF5F3EE),
                              disabledBackgroundColor: _AuthColor.surfaceAlt,
                              disabledForegroundColor: _AuthColor.ink3,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(16),
                              ),
                              textStyle: GoogleFonts.interTight(
                                fontSize: 15,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                            child: _submitting
                                ? const SizedBox(
                                    width: 18,
                                    height: 18,
                                    child: CircularProgressIndicator(
                                      strokeWidth: 2,
                                      color: Color(0xFFF5F3EE),
                                    ),
                                  )
                                : Text(
                                    _signUpMode ? 'Create account' : 'Sign in',
                                  ),
                          ),
                        ),
                        const SizedBox(height: 18),
                        Center(
                          child: _AuthModeLink(
                            enabled: !_submitting,
                            prompt: _signUpMode
                                ? 'Already have an account?'
                                : 'New here?',
                            action: _signUpMode ? 'Sign in' : 'Create account',
                            onTap: () {
                              setState(() {
                                _signUpMode = !_signUpMode;
                                _errorMessage = null;
                              });
                            },
                          ),
                        ),
                        const SizedBox(height: 34),
                        Text(
                          '$_appVersionLabel  //  end-to-end encrypted',
                          textAlign: TextAlign.center,
                          style: GoogleFonts.jetBrainsMono(
                            fontSize: 11,
                            height: 1.4,
                            letterSpacing: 0,
                            color: _AuthColor.ink3,
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
    );
  }
}

class _ShieldWell extends StatelessWidget {
  const _ShieldWell();

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        width: 52,
        height: 52,
        decoration: BoxDecoration(
          color: _AuthColor.sageSoft,
          borderRadius: BorderRadius.circular(17),
        ),
        child: const Icon(
          Icons.shield_outlined,
          color: _AuthColor.sageDeep,
          size: 26,
        ),
      ),
    );
  }
}

class _AuthTextFieldCard extends StatelessWidget {
  const _AuthTextFieldCard({
    required this.label,
    required this.controller,
    required this.enabled,
    required this.textInputAction,
    required this.hintText,
    this.obscureText = false,
    this.autofillHints,
    this.onSubmitted,
  });

  final String label;
  final TextEditingController controller;
  final bool enabled;
  final bool obscureText;
  final TextInputAction textInputAction;
  final String hintText;
  final Iterable<String>? autofillHints;
  final ValueChanged<String>? onSubmitted;

  @override
  Widget build(BuildContext context) {
    return AnimatedOpacity(
      duration: const Duration(milliseconds: 160),
      opacity: enabled ? 1 : 0.66,
      child: TextField(
        controller: controller,
        enabled: enabled,
        obscureText: obscureText,
        textInputAction: textInputAction,
        autocorrect: false,
        enableSuggestions: !obscureText,
        autofillHints: autofillHints,
        cursorColor: _AuthColor.sage,
        onSubmitted: onSubmitted,
        style: GoogleFonts.interTight(
          fontSize: 16,
          height: 1.2,
          fontWeight: FontWeight.w500,
          color: _AuthColor.ink,
        ),
        decoration: InputDecoration(
          labelText: label,
          hintText: hintText,
          filled: true,
          fillColor: _AuthColor.surface,
          contentPadding: const EdgeInsets.fromLTRB(16, 16, 16, 15),
          labelStyle: GoogleFonts.interTight(
            fontSize: 14,
            height: 1.2,
            fontWeight: FontWeight.w600,
            color: _AuthColor.ink2,
          ),
          floatingLabelStyle: GoogleFonts.interTight(
            fontSize: 13,
            height: 1.2,
            fontWeight: FontWeight.w700,
            color: _AuthColor.sageDeep,
          ),
          hintStyle: GoogleFonts.interTight(
            fontSize: 15,
            height: 1.2,
            fontWeight: FontWeight.w500,
            color: _AuthColor.ink3,
          ),
          errorBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
            borderSide: const BorderSide(color: _AuthColor.coral),
          ),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
            borderSide: const BorderSide(color: _AuthColor.hair),
          ),
          focusedErrorBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
            borderSide: const BorderSide(color: _AuthColor.coral, width: 1.4),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
            borderSide: const BorderSide(color: _AuthColor.hair),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
            borderSide: const BorderSide(color: _AuthColor.sage, width: 1.4),
          ),
          disabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
            borderSide: BorderSide(
              color: _AuthColor.hair.withValues(alpha: 0.72),
            ),
          ),
        ),
      ),
    );
  }
}

class _ErrorNotice extends StatelessWidget {
  const _ErrorNotice({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: _AuthColor.coralSoft,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFFEBC6C0)),
      ),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(
            Icons.info_outline_rounded,
            color: _AuthColor.coral,
            size: 18,
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              message,
              style: GoogleFonts.interTight(
                fontSize: 13,
                height: 1.35,
                fontWeight: FontWeight.w600,
                color: _AuthColor.coral,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _AuthModeLink extends StatelessWidget {
  const _AuthModeLink({
    required this.enabled,
    required this.prompt,
    required this.action,
    required this.onTap,
  });

  final bool enabled;
  final String prompt;
  final String action;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final promptStyle = GoogleFonts.interTight(
      fontSize: 13.5,
      height: 1.35,
      fontWeight: FontWeight.w500,
      color: _AuthColor.ink2,
    );
    final actionStyle = GoogleFonts.interTight(
      fontSize: 13.5,
      height: 1.35,
      fontWeight: FontWeight.w600,
      color: enabled ? _AuthColor.sageDeep : _AuthColor.ink3,
      decoration: TextDecoration.underline,
      decorationColor: enabled ? _AuthColor.sageDeep : _AuthColor.ink3,
      decorationThickness: 1.2,
    );

    return Wrap(
      alignment: WrapAlignment.center,
      crossAxisAlignment: WrapCrossAlignment.center,
      spacing: 5,
      children: [
        Text(prompt, style: promptStyle),
        InkWell(
          onTap: enabled ? onTap : null,
          borderRadius: BorderRadius.circular(6),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 2, vertical: 4),
            child: Text(action, style: actionStyle),
          ),
        ),
      ],
    );
  }
}
