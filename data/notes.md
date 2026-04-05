# Datasets and References Register
## Project: Adaptive Edge Intelligence System for Mountain Environments


This file tracks the datasets, APIs, and reference sources used in the dissertation project.

---

## 1) Core Human Activity / Fall Datasets
> The core of my project is the Human Activity Recognition (HAR) dataset, 
> which provides the primary training and evaluation data for the activity classification component.
> The MobiFall dataset is used for fall detection,
> and the PAMAP2 dataset is used for cross-dataset activity generalisation.
### 1.1 UCI HAR (Kaggle mirror) ✅ Selected / Downloaded
- **Name:** UCI-HAR Dataset (Kaggle mirror)
- **Purpose:** Baseline activity classification (walking, stairs, sitting, etc.)
- **URL (Kaggle mirror used):**
  - https://www.kaggle.com/datasets/drsaeedmohsen/ucihar-dataset
- **Official reference (UCI page):**
  - https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- **Planned role in project:** Primary activity model training / baseline benchmarking

### 1.2 MobiFall Dataset v2.0 (Kaggle mirror) ✅ Selected / Downloaded
- **Name:** MobiFall_Dataset_v2.0 (Kaggle mirror)
- **Purpose:** Fall detection training and event classification
- **URL (Kaggle mirror used):**
  - https://www.kaggle.com/datasets/kmknation/mobifall-dataset-v20
- **Official family dataset page (MobiFall / MobiAct):**
  - https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/
- **Planned role in project:** Fall detection model development and threshold/ML comparison

### 1.3 SisFall Dataset ⚠️ Optional / Recommended
- **Name:** SisFall fall detection dataset
- **Purpose:** Fall robustness testing / cross-dataset validation
- **URL (Kaggle mirror):**
  - https://www.kaggle.com/datasets/kushajm/sisfall-dataset-fall-detection
- **Paper / dataset reference:**
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC5298771/
- **Planned role in project:** Generalisation and robustness benchmark for fall detection

### 1.4 PAMAP2 Dataset ⚠️ Optional / Recommended
- **Name:** PAMAP2 Physical Activity Monitoring
- **Purpose:** Cross-dataset activity generalisation testing
- **Official dataset page (UCI):**
  - https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring
- **Direct UCI repository / readme reference:**
  - https://archive.ics.uci.edu/ml/machine-learning-databases/00231/readme.pdf
- **Planned role in project:** Out-of-domain activity recognition evaluation

---

## 2) Environmental Weather Data (Main + Validation)

> I am including weather data for future reference, the crucial structure of this project 
> does not use or require these datasets.

### 2.1 Open-Meteo Historical Weather API (Main Weather Source) ✅ In Use
- **Purpose:** Historical weather variables for mountain risk context (pressure, wind, precipitation, temperature, humidity)
- **Docs URL:**
  - https://open-meteo.com/en/docs/historical-weather-api
- **Chosen model (initial):**
  - ERA5
- **Why chosen (project decision):**
  - Consistent historical reanalysis suitable for research reproducibility
- **Planned hourly variables:**
  - pressure_msl
  - surface_pressure
  - temperature_2m
  - relative_humidity_2m
  - wind_speed_10m
  - wind_gusts_10m
  - precipitation
  - weather_code

### 2.2 Meteostat (Secondary / Validation Weather Source) ✅ In Use (Optional in early phase)
- **Purpose:** Validate / cross-check station-based hourly weather trends (e.g., pressure/wind)
- **Python docs (overview):**
  - https://dev.meteostat.net/python
- **Nearby stations (Python):**
  - https://dev.meteostat.net/python/stations/nearby
- **Hourly data (Python):**
  - https://dev.meteostat.net/python/api/meteostat.hourly
- **Planned role in project:**
  - Station-based validation against Open-Meteo reanalysis trends

---

## 3) Elevation / Terrain Data
> Similar to Weather Data elevation is for future reference,
### 3.1 Open-Meteo Elevation API ✅ In Use
- **Purpose:** Fetch terrain elevation for configured mountain / base locations
- **Docs URL:**
  - https://open-meteo.com/en/docs/elevation-api
- **Planned role in project:**
  - Add elevation context to weather + risk logic
  - Label mountain/base locations
  - Support route/ascent simulation inputs

### 3.2 Do later (optional): SRTM 30m raster if you need slope maps / route elevation profiles ⚠️ Optional
- **Recommended dataset (Earth Engine catalog):**
  - NASA SRTM Digital Elevation 30m
  - https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003
- **Why later / optional:**
  - Useful for GIS-style terrain analysis (slope, route elevation profiles)
  - Not required for initial dissertation prototype if using point elevation + weather APIs

### 3.3 Alternative SRTM (90m) reference (catalog only) ℹ️ Reference
- **CGIAR SRTM90 V4 (Earth Engine catalog):**
  - https://developers.google.com/earth-engine/datasets/catalog/CGIAR_SRTM90_V4
- **Use case:**
  - Lower-resolution DEM reference if needed for broader terrain context

---

## 4) Project Locations (UK Weather/Elevation Scenarios)
> These are for future reference,
### 4.1 Ben Nevis area (Mountain Scenario)
- **Role:** Harsh mountain weather scenario (Scotland)
- **Approx coords:** 56.7969, -5.0036

### 4.2 Fort William area (Lower / Base Comparison)
- **Role:** Nearby lower-altitude comparison (Scotland)
- **Approx coords:** 56.8198, -5.1052

### 4.3 Aviemore / Cairngorms (Second Mountain Scenario)
- **Role:** Secondary mountain region for robustness (Scotland)
- **Approx coords:** 57.1894, -3.8170

### 4.4 Snowdonia (Wales) / Eryri (Mountain Scenario)
- **Role:** Additional UK mountain region (Wales)
- **Approx coords (Snowdon/Yr Wyddfa area):** 53.0685, -4.0764

## 5) Generated Files (Expected Local Outputs)

### 5.1 Weather CSVs (from Open-Meteo script)
- `data/weather/weather_ben_nevis.csv`
- `data/weather/weather_fort_william.csv`
- `data/weather/weather_aviemore_cairngorms.csv`
- `data/weather/weather_wales_snowdonia.csv`

### 5.2 Elevation CSV (from Open-Meteo elevation script)
- `data/environment/elevation_locations.csv`

### 5.3 Activity / Fall Datasets (local raw storage example)
- `data/uci_har/`
- `data/mobifall/`
- `data/sisfall/` (if added)
- `data/pamap2/` (if added)


