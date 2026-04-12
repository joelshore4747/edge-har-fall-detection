import 'dart:async';
import 'dart:ui';

import 'package:flutter/material.dart';

import 'pages/runtime_home_page.dart';

void main() {
  runZonedGuarded(
    () {
      WidgetsFlutterBinding.ensureInitialized();

      FlutterError.onError = (FlutterErrorDetails details) {
        debugPrint('[AppStartup] FlutterError: ${details.exceptionAsString()}');
        debugPrintStack(
          stackTrace: details.stack,
          label: '[AppStartup] FlutterError stack',
        );
        FlutterError.presentError(details);
      };

      PlatformDispatcher.instance.onError =
          (Object error, StackTrace stackTrace) {
            debugPrint('[AppStartup] PlatformDispatcher error: $error');
            debugPrintStack(
              stackTrace: stackTrace,
              label: '[AppStartup] PlatformDispatcher stack',
            );
            return true;
          };

      debugPrint('[AppStartup] Launching FallMonitorApp');
      runApp(const FallMonitorApp());
    },
    (Object error, StackTrace stackTrace) {
      debugPrint('[AppStartup] Uncaught zone error: $error');
      debugPrintStack(
        stackTrace: stackTrace,
        label: '[AppStartup] Uncaught zone stack',
      );
    },
  );
}

class FallMonitorApp extends StatelessWidget {
  const FallMonitorApp({super.key});

  @override
  Widget build(BuildContext context) {
    const background = Color(0xFFF2F2F7);
    const surface = Color(0xFFFFFFFF);
    const primaryText = Color(0xFF111111);
    const secondaryText = Color(0xFF6E6E73);
    const border = Color(0xFFE5E5EA);
    const iosBlue = Color(0xFF007AFF);

    final colorScheme = const ColorScheme.light(
      primary: iosBlue,
      secondary: iosBlue,
      surface: surface,
      error: Color(0xFFFF3B30),
      onPrimary: Colors.white,
      onSecondary: Colors.white,
      onSurface: primaryText,
      onError: Colors.white,
    );

    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Fall Monitor',
      themeMode: ThemeMode.light,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: colorScheme,
        scaffoldBackgroundColor: background,
        canvasColor: background,
        splashFactory: InkRipple.splashFactory,
        appBarTheme: const AppBarTheme(
          backgroundColor: background,
          foregroundColor: primaryText,
          elevation: 0,
          scrolledUnderElevation: 0,
          centerTitle: false,
          titleTextStyle: TextStyle(
            fontSize: 30,
            fontWeight: FontWeight.w700,
            color: primaryText,
            letterSpacing: -0.8,
          ),
        ),
        textTheme: const TextTheme(
          headlineLarge: TextStyle(
            fontSize: 32,
            fontWeight: FontWeight.w700,
            color: primaryText,
            letterSpacing: -1.0,
          ),
          headlineMedium: TextStyle(
            fontSize: 28,
            fontWeight: FontWeight.w700,
            color: primaryText,
            letterSpacing: -0.8,
          ),
          titleLarge: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.w700,
            color: primaryText,
            letterSpacing: -0.3,
          ),
          titleMedium: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
            color: primaryText,
          ),
          bodyLarge: TextStyle(fontSize: 16, height: 1.4, color: primaryText),
          bodyMedium: TextStyle(
            fontSize: 14,
            height: 1.4,
            color: secondaryText,
          ),
          labelLarge: TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.w600,
            color: primaryText,
          ),
        ),
        cardTheme: CardThemeData(
          color: surface,
          elevation: 0,
          margin: EdgeInsets.zero,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(24),
            side: const BorderSide(color: border),
          ),
        ),
        dividerTheme: const DividerThemeData(
          color: border,
          thickness: 1,
          space: 1,
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: surface,
          contentPadding: const EdgeInsets.symmetric(
            horizontal: 18,
            vertical: 18,
          ),
          labelStyle: const TextStyle(
            color: secondaryText,
            fontSize: 15,
            fontWeight: FontWeight.w500,
          ),
          hintStyle: const TextStyle(color: secondaryText),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(18),
            borderSide: const BorderSide(color: border),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(18),
            borderSide: const BorderSide(color: border),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(18),
            borderSide: const BorderSide(color: iosBlue, width: 1.3),
          ),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            elevation: 0,
            backgroundColor: iosBlue,
            foregroundColor: Colors.white,
            disabledBackgroundColor: const Color(0xFFE5E5EA),
            disabledForegroundColor: const Color(0xFF8E8E93),
            padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 15),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
            textStyle: const TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
        outlinedButtonTheme: OutlinedButtonThemeData(
          style: OutlinedButton.styleFrom(
            elevation: 0,
            foregroundColor: primaryText,
            backgroundColor: surface,
            side: const BorderSide(color: border),
            padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 15),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
            textStyle: const TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
        filledButtonTheme: FilledButtonThemeData(
          style: FilledButton.styleFrom(
            elevation: 0,
            backgroundColor: iosBlue,
            foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 15),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
            textStyle: const TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
        chipTheme: ChipThemeData(
          backgroundColor: const Color(0xFFF9F9FB),
          selectedColor: const Color(0xFFEAF2FF),
          disabledColor: const Color(0xFFF1F1F4),
          side: const BorderSide(color: border),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(999),
          ),
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
          labelStyle: const TextStyle(
            fontSize: 13,
            fontWeight: FontWeight.w600,
            color: primaryText,
          ),
        ),
        snackBarTheme: SnackBarThemeData(
          behavior: SnackBarBehavior.floating,
          backgroundColor: const Color(0xFF1C1C1E),
          contentTextStyle: const TextStyle(
            color: Colors.white,
            fontSize: 14,
            fontWeight: FontWeight.w500,
          ),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
        ),
      ),
      home: const RuntimeHomePage(),
    );
  }
}
