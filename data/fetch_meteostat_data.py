from datetime import datetime
from pathlib import Path

import pandas as pd
from meteostat import Point, config, hourly, stations


LOCATIONS = [
    {"name": "ben_nevis", "lat": 56.7969, "lon": -5.0036},
    {"name": "fort_william", "lat": 56.8198, "lon": -5.1052},
    {"name": "aviemore_cairngorms", "lat": 57.1894, "lon": -3.8170},
    {"name": "wales_snowdonia", "lat": 53.0685, "lon": -4.0764},
]

START = datetime(2023, 1, 1)
END = datetime(2023, 12, 31, 23, 59)

OUTPUT_DIR = Path(__file__).resolve().parent / "raw" / "METEOSTAT_Dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METEOSTAT_CACHE_DIR = OUTPUT_DIR / "cache"
METEOSTAT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
config.cache_directory = str(METEOSTAT_CACHE_DIR)
config.stations_db_file = str(METEOSTAT_CACHE_DIR / "stations.db")


def fetch_nearest_station(lat: float, lon: float) -> pd.Series:
    nearby_df = stations.nearby(Point(lat, lon), limit=1)
    if nearby_df.empty:
        raise RuntimeError(f"No Meteostat station found near {lat}, {lon}")
    return nearby_df.iloc[0]


def save_location_weather(name: str, lat: float, lon: float) -> dict:
    station = fetch_nearest_station(lat, lon)
    station_id = station.name

    hourly_df = hourly(station_id, START, END).fetch()
    if hourly_df is None or hourly_df.empty:

        raise RuntimeError(f"No hourly Meteostat data returned for {name} ({station_id})")

    weather_path = OUTPUT_DIR / f"meteostat_hourly_{name}.csv"
    hourly_df.reset_index().to_csv(weather_path, index=False)

    return {
        "location": name,
        "lat": lat,
        "lon": lon,
        "station_id": station_id,
        "station_name": station.get("name"),
        "station_lat": station.get("latitude"),
        "station_lon": station.get("longitude"),
        "station_elevation_m": station.get("elevation"),
        "hourly_rows": len(hourly_df),
        "weather_file": weather_path.name,
    }


def main() -> None:
    station_rows = []
    for loc in LOCATIONS:
        print(f"Fetching Meteostat for {loc['name']}...")
        station_rows.append(save_location_weather(loc["name"], loc["lat"], loc["lon"]))

    stations_path = OUTPUT_DIR / "meteostat_stations_used.csv"
    pd.DataFrame(station_rows).to_csv(stations_path, index=False)
    print(f"Saved station metadata to: {stations_path}")
    print(f"Meteostat dataset folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
