import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault(
    "ADMIN_SESSION_SECRET",
    "test-only-session-secret-not-for-production-use",
)

os.environ.setdefault(
    "HAR_ARTIFACT_PATH",
    str(_PROJECT_ROOT / "artifacts" / "har" / "har_rf_ucihar.joblib"),
)
os.environ.setdefault(
    "FALL_ARTIFACT_PATH",
    str(_PROJECT_ROOT / "artifacts" / "fall" / "fall_meta_phone_negatives_v1" / "model.joblib"),
)
