import pandas as pd
import pytest

from pipeline.schema import COMMON_SCHEMA_COLUMNS, REQUIRED_CORE_COLUMNS, build_empty_common_frame, check_common_schema_columns
from pipeline.validation import validate_ingestion_dataframe


def _valid_minimal_df() -> pd.DataFrame:
    df = build_empty_common_frame()
    df = pd.DataFrame(
        [
            {
                "dataset_name": "TEST",
                "task_type": "har",
                "subject_id": "S1",
                "session_id": None,
                "timestamp": None,
                "ax": 0.1,
                "ay": 0.2,
                "az": 9.8,
                "gx": None,
                "gy": None,
                "gz": None,
                "label_raw": "walking",
                "label_mapped": "locomotion",
                "placement": None,
                "sampling_rate_hz": None,
                "source_file": "fixture.csv",
                "row_index": 0,
            }
        ],
        columns=COMMON_SCHEMA_COLUMNS,
    )
    return df


def test_common_schema_constants_include_required_core_columns():
    for col in REQUIRED_CORE_COLUMNS:
        assert col in COMMON_SCHEMA_COLUMNS


def test_build_empty_common_frame_has_expected_columns():
    df = build_empty_common_frame()
    assert list(df.columns) == COMMON_SCHEMA_COLUMNS
    assert df.empty


def test_missing_required_column_detection():
    df = _valid_minimal_df().drop(columns=["ax"])
    result = validate_ingestion_dataframe(df)
    assert not result.is_valid
    assert any("Missing common schema columns" in err for err in result.errors)


def test_nullable_gyro_columns_are_allowed():
    df = _valid_minimal_df()
    result = validate_ingestion_dataframe(df)
    assert result.is_valid, result.errors
    assert any("gx" in warning for warning in result.warnings)


def test_invalid_task_type_is_detected():
    df = _valid_minimal_df()
    df.loc[0, "task_type"] = "unsupported"
    schema_check = check_common_schema_columns(df)
    assert not schema_check.ok
    assert schema_check.invalid_task_types == ("unsupported",)
