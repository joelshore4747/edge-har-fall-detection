from pathlib import Path

import pandas as pd

from pipeline.ingest import load_pamap2
from pipeline.schema import COMMON_SCHEMA_COLUMNS
from pipeline.validation import validate_ingestion_dataframe


def _pamap2_row(timestamp: float, activity_id: int, *, ax: float, ay: float, az: float, gx: float, gy: float, gz: float) -> str:
    """Build one PAMAP2-style 54-column whitespace row.

    Column layout follows the documented PAMAP2 Protocol file structure.
    For this test we only need a valid row shape plus the selected hand IMU columns.
    """
    row = [0.0] * 54
    row[0] = timestamp
    row[1] = float(activity_id)
    row[2] = 100.0  # heart rate (unused by common schema parser)

    # hand IMU acc16 + gyro (the parser's current selected sensor source)
    row[4] = ax
    row[5] = ay
    row[6] = az
    row[10] = gx
    row[11] = gy
    row[12] = gz

    return " ".join(str(v) for v in row)


def test_load_pamap2_protocol_dat_directory_parses_common_schema(tmp_path: Path):
    root = tmp_path / "PAMAP2_Dataset"
    protocol = root / "Protocol"
    protocol.mkdir(parents=True)
    dat_file = protocol / "subject101.dat"

    lines = [
        _pamap2_row(0.00, 4, ax=1.0, ay=2.0, az=3.0, gx=0.1, gy=0.2, gz=0.3),   # walking -> locomotion
        _pamap2_row(0.01, 12, ax=1.1, ay=2.1, az=3.1, gx=0.2, gy=0.3, gz=0.4),  # ascending stairs -> stairs
        _pamap2_row(0.02, 2, ax=1.2, ay=2.2, az=3.2, gx=0.3, gy=0.4, gz=0.5),   # sitting -> static
    ]
    dat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    df = load_pamap2(root, max_files=1)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == COMMON_SCHEMA_COLUMNS
    assert set(df["dataset_name"].dropna().unique().tolist()) == {"PAMAP2"}
    assert set(df["task_type"].dropna().unique().tolist()) == {"har"}
    assert set(df["subject_id"].dropna().astype(str).unique().tolist()) == {"101"}
    assert df["session_id"].astype(str).str.contains("protocol_subject101").all()
    assert df["source_file"].astype(str).str.endswith("subject101.dat").all()
    assert df["label_raw"].astype(str).str.contains(":").all()
    assert set(df["label_mapped"].astype(str).unique().tolist()) == {"locomotion", "stairs", "static"}

    result = validate_ingestion_dataframe(df)
    assert result.is_valid, result.errors
    assert bool(df.attrs.get("pamap2_sensor_source"))
