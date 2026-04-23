import 'dart:async';

import 'package:flutter/foundation.dart'
    show TargetPlatform, defaultTargetPlatform, kIsWeb;
import 'package:sensors_plus/sensors_plus.dart';

import '../models/sensor_sample.dart';
import 'session_storage_service.dart';

enum SensorRecorderState { idle, starting, recording, stopping, error }

class SensorRecorderService {
  SensorRecorderService({
    SessionStorageService? storage,
    Duration samplingPeriod = const Duration(milliseconds: 20),
    Duration maxGyroAge = const Duration(milliseconds: 250),
    bool useUserAccelerometer = false,
  }) : _storage = storage ?? SessionStorageService(),
       _samplingPeriod = samplingPeriod,
       _maxGyroAge = maxGyroAge,
       _useUserAccelerometer = useUserAccelerometer;

  final SessionStorageService _storage;
  final Duration _samplingPeriod;
  final Duration _maxGyroAge;
  final bool _useUserAccelerometer;

  final List<SensorSample> _samples = [];

  StreamSubscription<_AccelReading>? _accSub;
  StreamSubscription<GyroscopeEvent>? _gyroSub;

  SensorRecorderState _state = SensorRecorderState.idle;
  Object? _lastError;
  StackTrace? _lastStackTrace;

  DateTime? _recordingStartedAtUtc;
  DateTime? _firstSensorTimestampUtc;

  double? _lastGx;
  double? _lastGy;
  double? _lastGz;
  DateTime? _lastGyroTimestampUtc;

  List<SensorSample> get samples => List.unmodifiable(_samples);
  SensorRecorderState get state => _state;
  bool get isRecording => _state == SensorRecorderState.recording;
  Object? get lastError => _lastError;
  StackTrace? get lastStackTrace => _lastStackTrace;
  int get sampleCount => _samples.length;

  DateTime? get recordingStartedAtUtc => _recordingStartedAtUtc;

  double? get durationSeconds {
    if (_samples.length < 2) {
      return null;
    }
    final duration = _samples.last.timestamp - _samples.first.timestamp;
    if (duration <= 0) {
      return null;
    }
    return duration;
  }

  double? get estimatedSamplingRateHz {
    if (_samples.length < 2) {
      return null;
    }
    final duration = _samples.last.timestamp - _samples.first.timestamp;
    if (duration <= 0) {
      return null;
    }
    return (_samples.length - 1) / duration;
  }

  bool get supportsLiveSensors {
    if (kIsWeb) return false;
    return defaultTargetPlatform == TargetPlatform.iOS ||
        defaultTargetPlatform == TargetPlatform.android;
  }

  Future<void> start({
    void Function(SensorSample sample)? onSample,
    void Function(Object error, StackTrace stackTrace)? onError,
  }) async {
    if (!supportsLiveSensors) {
      throw UnsupportedError(
        'Live motion sensors are only supported on Android and iOS. '
        'For desktop, use a saved or demo session.',
      );
    }

    if (_state == SensorRecorderState.starting ||
        _state == SensorRecorderState.recording) {
      throw StateError('Sensor recording is already active.');
    }

    _state = SensorRecorderState.starting;
    _lastError = null;
    _lastStackTrace = null;

    await _cancelSubscriptions();
    _resetSession();

    _recordingStartedAtUtc = DateTime.now().toUtc();

    try {
      _gyroSub = gyroscopeEventStream(samplingPeriod: _samplingPeriod).listen(
        (event) {
          _lastGx = _finiteOrNull(event.x);
          _lastGy = _finiteOrNull(event.y);
          _lastGz = _finiteOrNull(event.z);
          _lastGyroTimestampUtc = event.timestamp.toUtc();
        },
        onError: (Object error, StackTrace stackTrace) {
          _state = SensorRecorderState.error;
          _lastError = error;
          _lastStackTrace = stackTrace;
          onError?.call(error, stackTrace);
        },
        cancelOnError: false,
      );

      final accelStream = _buildAccelerometerStream();

      _accSub = accelStream.listen(
        (reading) {
          final eventTimestampUtc = reading.timestamp.toUtc();
          _firstSensorTimestampUtc ??= eventTimestampUtc;

          final start = _firstSensorTimestampUtc!;
          final tsSeconds =
              eventTimestampUtc.difference(start).inMicroseconds / 1000000.0;

          final gx = _gyroValueIfFresh(eventTimestampUtc, _lastGx);
          final gy = _gyroValueIfFresh(eventTimestampUtc, _lastGy);
          final gz = _gyroValueIfFresh(eventTimestampUtc, _lastGz);

          final hasCompleteGyro = gx != null && gy != null && gz != null;

          final sample = SensorSample(
            timestamp: tsSeconds,
            ax: reading.x,
            ay: reading.y,
            az: reading.z,
            gx: hasCompleteGyro ? gx : null,
            gy: hasCompleteGyro ? gy : null,
            gz: hasCompleteGyro ? gz : null,
          );

          _samples.add(sample);
          _state = SensorRecorderState.recording;
          onSample?.call(sample);
        },
        onError: (Object error, StackTrace stackTrace) {
          _state = SensorRecorderState.error;
          _lastError = error;
          _lastStackTrace = stackTrace;
          onError?.call(error, stackTrace);
        },
        cancelOnError: false,
      );

      _state = SensorRecorderState.recording;
    } catch (error, stackTrace) {
      _state = SensorRecorderState.error;
      _lastError = error;
      _lastStackTrace = stackTrace;
      await _cancelSubscriptions();
      onError?.call(error, stackTrace);
      rethrow;
    }
  }

  Future<void> stop() async {
    if (_state == SensorRecorderState.idle) {
      return;
    }

    _state = SensorRecorderState.stopping;
    await _cancelSubscriptions();

    if (_state != SensorRecorderState.error) {
      _state = SensorRecorderState.idle;
    }
  }

  void clear() {
    if (isRecording) {
      throw StateError('Cannot clear samples while recording is active.');
    }
    _resetSession();
    _state = SensorRecorderState.idle;
    _lastError = null;
    _lastStackTrace = null;
  }

  String _platformValue() {
    if (kIsWeb) {
      return 'web';
    }

    switch (defaultTargetPlatform) {
      case TargetPlatform.iOS:
        return 'ios';
      case TargetPlatform.android:
        return 'android';
      case TargetPlatform.linux:
        return 'linux';
      case TargetPlatform.macOS:
        return 'macos';
      case TargetPlatform.windows:
        return 'windows';
      case TargetPlatform.fuchsia:
        return 'fuchsia';
    }
  }

  String _deviceModelValue() {
    if (kIsWeb) {
      return 'web_browser';
    }

    switch (defaultTargetPlatform) {
      case TargetPlatform.iOS:
        return 'ios_device';
      case TargetPlatform.android:
        return 'android_device';
      case TargetPlatform.linux:
        return 'linux_desktop';
      case TargetPlatform.macOS:
        return 'macos_desktop';
      case TargetPlatform.windows:
        return 'windows_desktop';
      case TargetPlatform.fuchsia:
        return 'fuchsia_device';
    }
  }

  Future<String?> saveSessionLocally({
    required String subjectId,
    required String placement,
  }) async {
    final started = _recordingStartedAtUtc;
    final sessionId = started == null
        ? 'session_${DateTime.now().millisecondsSinceEpoch}'
        : 'session_${started.millisecondsSinceEpoch}';

    return _storage.saveSession(
      sessionId: sessionId,
      subjectId: subjectId,
      placement: placement,
      datasetName: 'APP_RUNTIME',
      sourceType: supportsLiveSensors ? 'mobile_app' : 'debug',
      devicePlatform: _platformValue(),
      deviceModel: _deviceModelValue(),
      recordingMode: supportsLiveSensors ? 'live_capture' : 'demo',
      runtimeMode: supportsLiveSensors ? 'mobile_live' : 'desktop_demo',
      samples: _samples.map((s) => s.toJson()).toList(growable: false),
    );
  }

  Future<void> dispose() async {
    await stop();
  }

  Stream<_AccelReading> _buildAccelerometerStream() {
    if (_useUserAccelerometer) {
      return userAccelerometerEventStream(samplingPeriod: _samplingPeriod).map(
        (event) => _AccelReading(
          x: _finiteOrZero(event.x),
          y: _finiteOrZero(event.y),
          z: _finiteOrZero(event.z),
          timestamp: event.timestamp,
        ),
      );
    }

    return accelerometerEventStream(samplingPeriod: _samplingPeriod).map(
      (event) => _AccelReading(
        x: _finiteOrZero(event.x),
        y: _finiteOrZero(event.y),
        z: _finiteOrZero(event.z),
        timestamp: event.timestamp,
      ),
    );
  }

  double _finiteOrZero(double value) {
    return value.isFinite ? value : 0.0;
  }

  double? _finiteOrNull(double value) {
    return value.isFinite ? value : null;
  }

  double? _gyroValueIfFresh(DateTime accelTimestampUtc, double? value) {
    if (value == null || _lastGyroTimestampUtc == null) {
      return null;
    }

    final age = accelTimestampUtc.difference(_lastGyroTimestampUtc!).abs();
    if (age > _maxGyroAge) {
      return null;
    }

    return value;
  }

  Future<void> _cancelSubscriptions() async {
    await _accSub?.cancel();
    await _gyroSub?.cancel();
    _accSub = null;
    _gyroSub = null;
  }

  void _resetSession() {
    _samples.clear();
    _firstSensorTimestampUtc = null;
    _recordingStartedAtUtc = null;

    _lastGx = null;
    _lastGy = null;
    _lastGz = null;
    _lastGyroTimestampUtc = null;
  }
}

class _AccelReading {
  const _AccelReading({
    required this.x,
    required this.y,
    required this.z,
    required this.timestamp,
  });

  final double x;
  final double y;
  final double z;
  final DateTime timestamp;
}
