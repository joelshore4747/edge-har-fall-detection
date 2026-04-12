import 'package:flutter/foundation.dart';

@immutable
class SensorSample {
  SensorSample({
    required this.timestamp,
    required this.ax,
    required this.ay,
    required this.az,
    this.gx,
    this.gy,
    this.gz,
  }) : assert(timestamp >= 0, 'timestamp must be >= 0'),
       assert(
         !timestamp.isNaN && !timestamp.isInfinite,
         'timestamp must be finite',
       ),
       assert(!ax.isNaN && !ax.isInfinite, 'ax must be finite'),
       assert(!ay.isNaN && !ay.isInfinite, 'ay must be finite'),
       assert(!az.isNaN && !az.isInfinite, 'az must be finite'),
       assert(
         gx == null || (!gx.isNaN && !gx.isInfinite),
         'gx must be finite when provided',
       ),
       assert(
         gy == null || (!gy.isNaN && !gy.isInfinite),
         'gy must be finite when provided',
       ),
       assert(
         gz == null || (!gz.isNaN && !gz.isInfinite),
         'gz must be finite when provided',
       );

  final double timestamp;
  final double ax;
  final double ay;
  final double az;
  final double? gx;
  final double? gy;
  final double? gz;

  bool get hasAnyGyro => gx != null || gy != null || gz != null;

  bool get hasCompleteGyro => gx != null && gy != null && gz != null;

  SensorSample copyWith({
    double? timestamp,
    double? ax,
    double? ay,
    double? az,
    double? gx,
    double? gy,
    double? gz,
    bool clearGx = false,
    bool clearGy = false,
    bool clearGz = false,
  }) {
    return SensorSample(
      timestamp: timestamp ?? this.timestamp,
      ax: ax ?? this.ax,
      ay: ay ?? this.ay,
      az: az ?? this.az,
      gx: clearGx ? null : (gx ?? this.gx),
      gy: clearGy ? null : (gy ?? this.gy),
      gz: clearGz ? null : (gz ?? this.gz),
    );
  }

  Map<String, dynamic> toJson() {
    return <String, dynamic>{
      'timestamp': timestamp,
      'ax': ax,
      'ay': ay,
      'az': az,
      'gx': gx,
      'gy': gy,
      'gz': gz,
    };
  }

  factory SensorSample.fromJson(Map<String, dynamic> json) {
    final timestamp = _parseRequiredDouble(json['timestamp'], 'timestamp');
    final ax = _parseRequiredDouble(json['ax'], 'ax');
    final ay = _parseRequiredDouble(json['ay'], 'ay');
    final az = _parseRequiredDouble(json['az'], 'az');

    final gx = _parseOptionalDouble(json['gx'], 'gx');
    final gy = _parseOptionalDouble(json['gy'], 'gy');
    final gz = _parseOptionalDouble(json['gz'], 'gz');

    return SensorSample(
      timestamp: timestamp,
      ax: ax,
      ay: ay,
      az: az,
      gx: gx,
      gy: gy,
      gz: gz,
    );
  }

  static SensorSample? tryFromJson(
    dynamic json, {
    void Function(String message)? onInvalid,
  }) {
    if (json is! Map) {
      onInvalid?.call('Sample is not a JSON object.');
      return null;
    }

    try {
      return SensorSample.fromJson(
        json.map((key, value) => MapEntry(key.toString(), value)),
      );
    } catch (error) {
      onInvalid?.call(error.toString());
      return null;
    }
  }

  static double _parseRequiredDouble(dynamic value, String fieldName) {
    final parsed = _parseOptionalDouble(value, fieldName);
    if (parsed == null) {
      throw FormatException(
        'Missing or invalid required numeric field: $fieldName',
      );
    }
    return parsed;
  }

  static double? _parseOptionalDouble(dynamic value, String fieldName) {
    if (value == null) {
      return null;
    }

    final parsed = switch (value) {
      num v => v.toDouble(),
      String v => double.tryParse(v.trim()),
      _ => null,
    };

    if (parsed == null) {
      throw FormatException('Invalid numeric field: $fieldName');
    }

    if (parsed.isNaN || parsed.isInfinite) {
      throw FormatException('Field must be finite: $fieldName');
    }

    return parsed;
  }

  @override
  String toString() {
    return 'SensorSample('
        'timestamp: $timestamp, '
        'ax: $ax, ay: $ay, az: $az, '
        'gx: $gx, gy: $gy, gz: $gz'
        ')';
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;

    return other is SensorSample &&
        other.timestamp == timestamp &&
        other.ax == ax &&
        other.ay == ay &&
        other.az == az &&
        other.gx == gx &&
        other.gy == gy &&
        other.gz == gz;
  }

  @override
  int get hashCode => Object.hash(timestamp, ax, ay, az, gx, gy, gz);
}
