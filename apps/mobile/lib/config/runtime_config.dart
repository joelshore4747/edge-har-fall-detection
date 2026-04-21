const String runtimeApiBaseUrl = String.fromEnvironment(
  'API_BASE_URL',
  defaultValue: 'http://127.0.0.1:8000',
);

const String runtimeApiUsername = String.fromEnvironment(
  'API_USERNAME',
  defaultValue: '',
);

const String runtimeApiSubjectId = String.fromEnvironment(
  'API_SUBJECT_ID',
  defaultValue: '',
);

const String runtimeApiPassword = String.fromEnvironment(
  'API_PASSWORD',
  defaultValue: '',
);
