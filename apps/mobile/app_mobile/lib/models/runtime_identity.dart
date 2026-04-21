class RuntimeIdentity {
  const RuntimeIdentity({
    required this.username,
    required this.password,
    required this.subjectId,
    required this.displayName,
    required this.registered,
    required this.provisioningMode,
    this.registeredAt,
  });

  final String username;
  final String password;
  final String subjectId;
  final String? displayName;
  final bool registered;
  final String provisioningMode;
  final DateTime? registeredAt;

  factory RuntimeIdentity.fromJson(Map<String, dynamic> json) {
    return RuntimeIdentity(
      username: (json['username'] as String?)?.trim() ?? '',
      password: json['password'] as String? ?? '',
      subjectId: (json['subject_id'] as String?)?.trim() ?? '',
      displayName: (json['display_name'] as String?)?.trim(),
      registered: json['registered'] as bool? ?? false,
      provisioningMode:
          (json['provisioning_mode'] as String?)?.trim() ?? 'self_service',
      registeredAt: DateTime.tryParse(json['registered_at'] as String? ?? ''),
    );
  }

  Map<String, dynamic> toJson() {
    return <String, dynamic>{
      'username': username,
      'password': password,
      'subject_id': subjectId,
      'display_name': displayName,
      'registered': registered,
      'provisioning_mode': provisioningMode,
      'registered_at': registeredAt?.toUtc().toIso8601String(),
    };
  }

  RuntimeIdentity copyWith({
    String? username,
    String? password,
    String? subjectId,
    String? displayName,
    bool? registered,
    String? provisioningMode,
    DateTime? registeredAt,
  }) {
    return RuntimeIdentity(
      username: username ?? this.username,
      password: password ?? this.password,
      subjectId: subjectId ?? this.subjectId,
      displayName: displayName ?? this.displayName,
      registered: registered ?? this.registered,
      provisioningMode: provisioningMode ?? this.provisioningMode,
      registeredAt: registeredAt ?? this.registeredAt,
    );
  }
}
