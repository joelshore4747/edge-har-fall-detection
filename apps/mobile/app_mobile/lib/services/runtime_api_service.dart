import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

import '../models/api_result_summary.dart';
import '../models/persisted_session.dart';
import '../models/sensor_sample.dart';

class RuntimeApiException implements Exception {
  RuntimeApiException(this.message, {this.statusCode, this.details});

  final String message;
  final int? statusCode;
  final Map<String, dynamic>? details;

  @override
  String toString() {
    if (statusCode == null) {
      return message;
    }
    return '[$statusCode] $message';
  }
}

class RuntimeHealthResponse {
  RuntimeHealthResponse({
    required this.status,
    required this.serviceName,
    required this.version,
  });

  final String status;
  final String serviceName;
  final String version;

  factory RuntimeHealthResponse.fromJson(Map<String, dynamic> json) {
    return RuntimeHealthResponse(
      status: _asString(json['status']) ?? 'unknown',
      serviceName: _asString(json['service_name']) ?? 'runtime-inference-api',
      version: _asString(json['version']) ?? 'unknown',
    );
  }
}

class RuntimeRegistrationResponse {
  RuntimeRegistrationResponse({
    required this.status,
    required this.username,
    required this.subjectId,
    required this.created,
    this.displayName,
  });

  final String status;
  final String username;
  final String subjectId;
  final String? displayName;
  final bool created;

  factory RuntimeRegistrationResponse.fromJson(Map<String, dynamic> json) {
    return RuntimeRegistrationResponse(
      status: _asString(json['status']) ?? 'registered',
      username: _asString(json['username']) ?? '',
      subjectId: _asString(json['subject_id']) ?? '',
      displayName: _asString(json['display_name']),
      created: json['created'] as bool? ?? true,
    );
  }
}

class RuntimeAuthProfile {
  RuntimeAuthProfile({
    required this.status,
    required this.role,
    required this.authRequired,
    this.username,
    this.subjectId,
  });

  final String status;
  final String? username;
  final String? subjectId;
  final String role;
  final bool authRequired;

  factory RuntimeAuthProfile.fromJson(Map<String, dynamic> json) {
    return RuntimeAuthProfile(
      status: _asString(json['status']) ?? 'unknown',
      username: _asString(json['username']),
      subjectId: _asString(json['subject_id']),
      role: _asString(json['role']) ?? 'anonymous',
      authRequired: json['auth_required'] as bool? ?? true,
    );
  }
}

class RuntimeFeedbackAck {
  RuntimeFeedbackAck({
    required this.sessionId,
    required this.userFeedback,
    required this.message,
    required this.status,
    this.requestId,
    this.persistedSessionId,
    this.persistedInferenceId,
    this.persistedFeedbackId,
    this.targetType,
    this.eventId,
    this.windowId,
    this.recordedAt,
  });

  final String? requestId;
  final String sessionId;
  final String? persistedSessionId;
  final String? persistedInferenceId;
  final String? persistedFeedbackId;
  final String? targetType;
  final String? eventId;
  final String? windowId;
  final String userFeedback;
  final String message;
  final String status;
  final DateTime? recordedAt;

  factory RuntimeFeedbackAck.fromJson(Map<String, dynamic> json) {
    return RuntimeFeedbackAck(
      requestId: _asString(json['request_id']),
      sessionId: _asString(json['session_id']) ?? '',
      persistedSessionId: _asString(json['persisted_session_id']),
      persistedInferenceId: _asString(json['persisted_inference_id']),
      persistedFeedbackId: _asString(json['persisted_feedback_id']),
      targetType: _asString(json['target_type']),
      eventId: _asString(json['event_id']),
      windowId: _asString(json['window_id']),
      userFeedback: _asString(json['user_feedback']) ?? 'unknown',
      message: _asString(json['message']) ?? 'Feedback recorded.',
      status: _asString(json['status']) ?? 'accepted',
      recordedAt: _tryParseDateTime(json['recorded_at']),
    );
  }
}

class RuntimeApiService {
  RuntimeApiService({
    required String baseUrl,
    String? basicAuthUsername,
    String? basicAuthPassword,
    http.Client? client,
    Duration? timeout,
  }) : _baseUrl = _normaliseBaseUrl(baseUrl),
       _basicAuthHeader = _buildBasicAuthHeader(
         username: basicAuthUsername,
         password: basicAuthPassword,
       ),
       _client = client ?? http.Client(),
       _timeout = timeout ?? const Duration(seconds: 30);

  final String _baseUrl;
  final String? _basicAuthHeader;
  final http.Client _client;
  final Duration _timeout;

  void dispose() {
    _client.close();
  }

  Future<RuntimeHealthResponse> checkHealth() async {
    final uri = _buildUri('/health');

    http.Response response;
    try {
      response = await _client
          .get(uri, headers: _requestHeaders())
          .timeout(_timeout);
    } on TimeoutException {
      throw RuntimeApiException('Health check timed out.');
    } catch (e) {
      throw RuntimeApiException('Failed to connect to backend: $e');
    }

    final payload = _decodeJsonMap(response.body);

    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw _toApiException(
        response.statusCode,
        payload,
        fallbackMessage: 'Health check failed.',
      );
    }

    return RuntimeHealthResponse.fromJson(payload);
  }

  Future<RuntimeRegistrationResponse> registerSelfServiceUser({
    required String username,
    required String password,
    required String subjectId,
    String? displayName,
    String devicePlatform = 'mobile_app',
    String? deviceModel,
    String? appVersion,
    String? appBuild,
  }) async {
    final json = await _postJson('/v1/auth/register', <String, dynamic>{
      'username': username,
      'password': password,
      'subject_id': subjectId,
      'display_name': displayName,
      'device_platform': devicePlatform,
      'device_model': deviceModel,
      'app_version': appVersion,
      'app_build': appBuild,
    });
    return RuntimeRegistrationResponse.fromJson(json);
  }

  Future<RuntimeAuthProfile> fetchAuthProfile() async {
    final json = await _getJson('/v1/auth/me');
    return RuntimeAuthProfile.fromJson(json);
  }

  Future<ApiResultSummary> submitSession({
    required String sessionId,
    required String subjectId,
    required String placement,
    required List<SensorSample> samples,
    String taskType = 'runtime',
    String datasetName = 'APP_RUNTIME',
    String sourceType = 'mobile_app',
    String devicePlatform = 'unknown',
    String? deviceModel,
    String? appVersion,
    String? appBuild,
    String recordingMode = 'live_capture',
    String runtimeMode = 'mobile_live',
    double? samplingRateHz,
    String? sessionName,
    String? activityLabel,
    String? notes,
    bool includeHarWindows = false,
    bool includeFallWindows = false,
    bool includeCombinedTimeline = true,
    bool includeGroupedFallEvents = true,
    bool includePointTimeline = true,
    bool includeTimelineEvents = true,
    bool includeTransitionEvents = true,
    String? traceId,
    String? clientVersion,
  }) async {
    if (samples.length < 32) {
      throw RuntimeApiException(
        'At least 32 samples are required before submitting a session.',
      );
    }

    final payload = <String, dynamic>{
      'metadata': <String, dynamic>{
        'session_id': sessionId,
        'subject_id': subjectId.trim().isEmpty
            ? 'anonymous_user'
            : subjectId.trim(),
        'placement': _normalisePlacement(placement),
        'task_type': taskType,
        'dataset_name': datasetName,
        'source_type': sourceType,
        'device_platform': devicePlatform,
        'device_model': deviceModel,
        'sampling_rate_hz': samplingRateHz,
        'session_name': _cleanOptionalString(sessionName),
        'activity_label': _cleanOptionalString(activityLabel),
        'notes': notes,
        'app_version': appVersion,
        'app_build': appBuild,
        'recording_mode': recordingMode,
        'runtime_mode': runtimeMode,
      },
      'samples': samples
          .map((sample) => sample.toJson())
          .toList(growable: false),
      'include_har_windows': includeHarWindows,
      'include_fall_windows': includeFallWindows,
      'include_combined_timeline': includeCombinedTimeline,
      'include_grouped_fall_events': includeGroupedFallEvents,
      'include_point_timeline': includePointTimeline,
      'include_timeline_events': includeTimelineEvents,
      'include_transition_events': includeTransitionEvents,
      'request_context': <String, dynamic>{
        if (traceId != null && traceId.trim().isNotEmpty)
          'trace_id': traceId.trim(),
        if (clientVersion != null && clientVersion.trim().isNotEmpty)
          'client_version': clientVersion.trim(),
      },
    };

    return inferPayload(payload);
  }

  Future<ApiResultSummary> inferPayload(Map<String, dynamic> payload) async {
    final json = await _postJson('/v1/infer/session', payload);
    return ApiResultSummary.fromJson(json);
  }

  Future<RuntimeFeedbackAck> submitFeedback({
    required String sessionId,
    required String userFeedback,
    String? subjectId,
    String? persistedSessionId,
    String? persistedInferenceId,
    String? targetType,
    String? eventId,
    String? windowId,
    String? correctedLabel,
    String? reviewerId,
    String? notes,
    String? traceId,
    String? clientVersion,
  }) async {
    final payload = <String, dynamic>{
      'session_id': sessionId,
      'subject_id': subjectId,
      'persisted_session_id': persistedSessionId,
      'persisted_inference_id': persistedInferenceId,
      'target_type': targetType,
      'event_id': eventId,
      'window_id': windowId,
      'user_feedback': userFeedback,
      'corrected_label': correctedLabel,
      'reviewer_id': reviewerId,
      'notes': notes,
      'request_context': <String, dynamic>{
        if (traceId != null && traceId.trim().isNotEmpty)
          'trace_id': traceId.trim(),
        if (clientVersion != null && clientVersion.trim().isNotEmpty)
          'client_version': clientVersion.trim(),
      },
    };

    final json = await _postJson('/v1/feedback', payload);
    return RuntimeFeedbackAck.fromJson(json);
  }

  Future<List<PersistedSessionSummary>> listPersistedSessions({
    String? subjectId,
    int limit = 50,
    int offset = 0,
  }) async {
    final json = await _getJson(
      '/v1/sessions',
      queryParameters: <String, String?>{
        'subject_id': subjectId,
        'limit': limit.toString(),
        'offset': offset.toString(),
      },
    );

    final sessions = json['sessions'];
    if (sessions is! List) {
      return const <PersistedSessionSummary>[];
    }

    return sessions
        .whereType<Map>()
        .map(
          (item) => PersistedSessionSummary.fromJson(
            item.map((key, value) => MapEntry(key.toString(), value)),
          ),
        )
        .toList(growable: false);
  }

  Future<PersistedSessionDetail> fetchPersistedSessionDetail(
    String appSessionId,
  ) async {
    final json = await _getJson('/v1/sessions/$appSessionId');
    return PersistedSessionDetail.fromJson(json);
  }

  Future<Map<String, dynamic>> _postJson(
    String path,
    Map<String, dynamic> payload,
  ) async {
    final uri = _buildUri(path);

    http.Response response;
    try {
      response = await _client
          .post(
            uri,
            headers: _requestHeaders(includeContentType: true),
            body: jsonEncode(_stripNulls(payload)),
          )
          .timeout(_timeout);
    } on TimeoutException {
      throw RuntimeApiException('Request to $path timed out.');
    } catch (e) {
      throw RuntimeApiException('Failed to connect to backend: $e');
    }

    final json = _decodeJsonMap(response.body);

    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw _toApiException(
        response.statusCode,
        json,
        fallbackMessage: 'Request failed for $path.',
      );
    }

    return json;
  }

  Future<Map<String, dynamic>> _getJson(
    String path, {
    Map<String, String?>? queryParameters,
  }) async {
    final uri = _buildUri(path, queryParameters: queryParameters);

    http.Response response;
    try {
      response = await _client
          .get(uri, headers: _requestHeaders())
          .timeout(_timeout);
    } on TimeoutException {
      throw RuntimeApiException('Request to $path timed out.');
    } catch (e) {
      throw RuntimeApiException('Failed to connect to backend: $e');
    }

    final json = _decodeJsonMap(response.body);

    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw _toApiException(
        response.statusCode,
        json,
        fallbackMessage: 'Request failed for $path.',
      );
    }

    return json;
  }

  Uri _buildUri(String path, {Map<String, String?>? queryParameters}) {
    final normalisedPath = path.startsWith('/') ? path : '/$path';
    final baseUri = Uri.parse('$_baseUrl$normalisedPath');
    if (queryParameters == null || queryParameters.isEmpty) {
      return baseUri;
    }

    final cleaned = <String, String>{};
    queryParameters.forEach((key, value) {
      final trimmed = value?.trim();
      if (trimmed == null || trimmed.isEmpty) {
        return;
      }
      cleaned[key] = trimmed;
    });

    if (cleaned.isEmpty) {
      return baseUri;
    }

    return baseUri.replace(queryParameters: cleaned);
  }

  Map<String, String> _requestHeaders({bool includeContentType = false}) {
    final headers = <String, String>{'Accept': 'application/json'};
    if (includeContentType) {
      headers['Content-Type'] = 'application/json';
    }
    if (_basicAuthHeader != null) {
      headers['Authorization'] = _basicAuthHeader;
    }
    return headers;
  }

  RuntimeApiException _toApiException(
    int statusCode,
    Map<String, dynamic> payload, {
    required String fallbackMessage,
  }) {
    final message =
        _asString(payload['message']) ??
        _asString(payload['detail']) ??
        fallbackMessage;

    final details = payload['details'] is Map<String, dynamic>
        ? payload['details'] as Map<String, dynamic>
        : payload['details'] is Map
        ? (payload['details'] as Map).map(
            (key, value) => MapEntry(key.toString(), value),
          )
        : null;

    return RuntimeApiException(
      message,
      statusCode: statusCode,
      details: details,
    );
  }

  static String _normaliseBaseUrl(String value) {
    final trimmed = value.trim();
    if (trimmed.isEmpty) {
      throw ArgumentError.value(value, 'baseUrl', 'baseUrl cannot be empty');
    }
    return trimmed.endsWith('/')
        ? trimmed.substring(0, trimmed.length - 1)
        : trimmed;
  }

  static String _normalisePlacement(String value) {
    switch (value.trim().toLowerCase()) {
      case 'pocket':
        return 'pocket';
      case 'hand':
        return 'hand';
      case 'desk':
      case 'on_surface':
        return 'desk';
      case 'bag':
        return 'bag';
      case 'arm_mounted':
        return 'hand';
      case 'repositioning':
      case 'unknown':
      default:
        return 'unknown';
    }
  }

  static String? _buildBasicAuthHeader({
    required String? username,
    required String? password,
  }) {
    final normalizedUsername = username?.trim() ?? '';
    final normalizedPassword = password ?? '';
    if (normalizedUsername.isEmpty || normalizedPassword.isEmpty) {
      return null;
    }

    final token = base64Encode(
      utf8.encode('$normalizedUsername:$normalizedPassword'),
    );
    return 'Basic $token';
  }
}

Map<String, dynamic> _decodeJsonMap(String raw) {
  if (raw.trim().isEmpty) {
    return <String, dynamic>{};
  }

  final decoded = jsonDecode(raw);
  if (decoded is Map<String, dynamic>) {
    return decoded;
  }
  if (decoded is Map) {
    return decoded.map((key, value) => MapEntry(key.toString(), value));
  }

  throw const FormatException('Expected a JSON object response.');
}

dynamic _stripNulls(dynamic value) {
  if (value is Map) {
    final result = <String, dynamic>{};
    value.forEach((key, item) {
      if (item == null) {
        return;
      }
      final cleaned = _stripNulls(item);
      if (cleaned != null) {
        result[key.toString()] = cleaned;
      }
    });
    return result;
  }

  if (value is List) {
    return value
        .map(_stripNulls)
        .where((item) => item != null)
        .toList(growable: false);
  }

  return value;
}

String? _asString(dynamic value) {
  if (value == null) {
    return null;
  }
  final text = value.toString();
  return text.isEmpty ? null : text;
}

String? _cleanOptionalString(String? value) {
  if (value == null) {
    return null;
  }
  final trimmed = value.trim();
  return trimmed.isEmpty ? null : trimmed;
}

DateTime? _tryParseDateTime(dynamic value) {
  final raw = _asString(value);
  if (raw == null) {
    return null;
  }
  return DateTime.tryParse(raw);
}
