from pathlib import Path

import pandas as pd

from pipeline.ingest import load_wisdm
from pipeline.schema import COMMON_SCHEMA_COLUMNS
from pipeline.validation import validate_ingestion_dataframe


def test_load_wisdm_split_csv_maps_to_common_schema(tmp_path: Path):
    wisdm_dir = tmp_path / "WISDM"
    wisdm_dir.mkdir(parents=True)
    train_csv = wisdm_dir / "train.csv"
    train_csv.write_text(
        "\n".join(
            [
                "acc_x,acc_y,acc_z,timestamp,Activity",
                "-1.2,9.8,0.1,0.00,5",
                "-1.1,9.7,0.2,0.05,5",
                "-1.0,9.6,0.1,0.10,5",
                "0.4,10.1,-0.2,2.50,0",
                "0.3,10.0,-0.1,2.55,0",
                "0.1,9.7,0.0,5.50,2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    df = load_wisdm(wisdm_dir, split="train")

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == COMMON_SCHEMA_COLUMNS
    assert set(df["dataset_name"].dropna().unique().tolist()) == {"WISDM"}
    assert set(df["task_type"].dropna().unique().tolist()) == {"har"}
    assert set(df["subject_id"].dropna().astype(str).unique().tolist()) == {"split_train"}
    assert df["session_id"].astype(str).str.startswith("train_session_").all()
    assert set(df["label_raw"].astype(str).unique().tolist()) == {"walking", "downstairs", "sitting"}
    assert set(df["label_mapped"].astype(str).unique().tolist()) == {"locomotion", "stairs", "static"}

    result = validate_ingestion_dataframe(df)
    assert result.is_valid, result.errors
    assert bool(df.attrs.get("wisdm_has_true_subject_ids")) is False
