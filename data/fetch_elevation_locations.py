import os
import requests
import pandas as pd


LOCATIONS = [
    {"label": "ben_nevis", "lat": 56.7969, "lon": -5.0036},
    {"label": "fort_william", "lat": 56.8198, "lon": -5.1052},
    {"label": "aviemore_cairngorms", "lat": 57.1894, "lon": -3.8170},
    {"label": "wales_snowdonia", "lat": 53.0685, "lon": -4.0764},
]

OUTPUT_DIR = "data/environment"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "elevation_locations.csv")

API_URL = "https://api.open-meteo.com/v1/elevation"

def fetch_elevation(lat: float, lon: float):
    params = {"latitude": lat, "longitude": lon}
    r = requests.get(API_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    if "elevation" not in data or not data["elevation"]:
        raise ValueError(f"No elevation returned for ({lat}, {lon})")

    return float(data["elevation"][0])

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []
    for loc in LOCATIONS:
        label = loc["label"]
        lat = loc["lat"]
        lon = loc["lon"]

        print(f"Fetching elevation for {label} ({lat}, {lon})...")
        elev_m = fetch_elevation(lat, lon)

        rows.append({
            "label": label,
            "latitude": lat,
            "longitude": lon,
            "elevation_m": elev_m
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nSaved:", OUTPUT_CSV)
    print("\nPreview:")
    print(df)

if __name__ == "__main__":
    main()