from pathlib import Path

import pandas as pd
import pytest

from pipeline.ingest import load_mobifall, load_pamap2, load_sisfall, load_uci_har, load_wisdm
from pipeline.schema import COMMON_SCHEMA_COLUMNS
from pipeline.validation import validate_ingestion_dataframe

FIXTURES = Path("tests/fixtures")


@pytest.mark.parametrize(
    "loader,path,expected_dataset,expected_task",
    [
        (load_uci_har, FIXTURES / "uci_har_sample.csv", "UCIHAR", "har"),
        (load_pamap2, FIXTURES / "pamap2_sample.csv", "PAMAP2", "har"),
        (load_wisdm, FIXTURES / "wisdm_sample.csv", "WISDM", "har"),
        (load_mobifall, FIXTURES / "mobifall_sample.csv", "MOBIFALL", "fall"),
        (load_sisfall, FIXTURES / "sisfall_sample.csv", "SISFALL", "fall"),
    ],
)
def test_loader_returns_common_schema_dataframe(loader, path, expected_dataset, expected_task):
    df = loader(path)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == COMMON_SCHEMA_COLUMNS
    assert set(df["dataset_name"].dropna().unique().tolist()) == {expected_dataset}
    assert set(df["task_type"].dropna().unique().tolist()) == {expected_task}
    assert df["label_mapped"].notna().all()
    assert df["label_raw"].notna().all()

    result = validate_ingestion_dataframe(df)
    assert result.is_valid, result.errors


def test_mobifall_loader_adds_missing_gyro_columns_as_nulls():
    df = load_mobifall(FIXTURES / "mobifall_sample.csv")
    assert {"gx", "gy", "gz"}.issubset(df.columns)
    assert df[["gx", "gy", "gz"]].isna().all().all()


def test_uci_loader_maps_expected_categories():
    df = load_uci_har(FIXTURES / "uci_har_sample.csv")
    mapped = set(df["label_mapped"].astype(str).unique().tolist())
    assert {"static", "locomotion", "stairs"}.issubset(mapped)


def test_wisdm_loader_maps_numeric_activity_codes():
    df = load_wisdm(FIXTURES / "wisdm_sample.csv")
    mapped = set(df["label_mapped"].astype(str).unique().tolist())
    assert {"static", "locomotion", "stairs"}.issubset(mapped)
