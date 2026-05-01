from pathlib import Path

import pandas as pd

from pipeline.ingest.weather import load_weather_csv, load_weather_csvs


def test_load_weather_csv_parses_time_and_infers_location(tmp_path: Path):
    csv_path = tmp_path / "weather_ben_nevis.csv"
    csv_path.write_text(
        "time,pressure_msl,temperature_2m\n"
        "2023-01-01T00:00,998.4,-1.2\n"
        "2023-01-01T01:00,998.0,-1.4\n",
        encoding="utf-8",
    )

    df = load_weather_csv(csv_path)

    assert list(df.columns)[:3] == ["time", "pressure_msl", "temperature_2m"]
    assert pd.api.types.is_datetime64_any_dtype(df["time"])
    assert df["location_name"].iloc[0] == "ben_nevis"
    assert df["source_file"].iloc[0] == str(csv_path)
    assert float(df["pressure_msl"].iloc[0]) == 998.4


def test_load_weather_csvs_concatenates_multiple_files(tmp_path: Path):
    p1 = tmp_path / "weather_site_a.csv"
    p2 = tmp_path / "weather_site_b.csv"
    p1.write_text("time,pressure_msl\n2023-01-01T00:00,1000.0\n", encoding="utf-8")
    p2.write_text("time,pressure_msl\n2023-01-01T01:00,999.5\n", encoding="utf-8")

    df = load_weather_csvs([p1, p2])

    assert len(df) == 2
    assert set(df["location_name"].astype(str).tolist()) == {"site_a", "site_b"}
