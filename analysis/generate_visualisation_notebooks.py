from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "analysis"


def _lines(text: str) -> list[str]:
    return dedent(text).lstrip("\n").splitlines(keepends=True)


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _lines(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def write_notebook(path: Path, cells: list[dict]) -> None:
    payload = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.13",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


BOOTSTRAP_CELL = """
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

CWD = Path.cwd().resolve()
for candidate in (CWD, *CWD.parents):
    if (candidate / "pipeline").exists() and (candidate / "analysis").exists():
        REPO_ROOT = candidate
        break
else:
    raise RuntimeError("Could not locate repo root from the current working directory.")

for extra_path in (REPO_ROOT, REPO_ROOT / "analysis"):
    extra_str = str(extra_path)
    if extra_str not in sys.path:
        sys.path.insert(0, extra_str)

import notebook_utils as nb_utils

nb_utils.configure_matplotlib()
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 160)
"""


def build_audit_overview_notebook() -> None:
    cells = [
        md_cell(
            """
            # Cross-Dataset Audit Overview

            This notebook visualises the structured audit already saved in
            `results/validation/dataset_distribution_audit.json`. It is the quickest
            way to compare the public inertial sources already flowing through the
            dissertation ingest pipeline.
            """
        ),
        code_cell(BOOTSTRAP_CELL),
        code_cell(
            """
            AUDIT_PATH = REPO_ROOT / "results" / "validation" / "dataset_distribution_audit.json"
            if not AUDIT_PATH.exists():
                raise FileNotFoundError(AUDIT_PATH)

            with AUDIT_PATH.open("r", encoding="utf-8") as fh:
                audit = json.load(fh)

            summary_df = nb_utils.audit_summary_frame(audit)
            comparison_df = pd.DataFrame([audit.get("comparison", {})])
            errors_df = pd.DataFrame(
                [
                    {"dataset_name": dataset_name, "error": error_text}
                    for dataset_name, error_text in audit.get("errors", {}).items()
                ]
            )

            display(summary_df)
            display(comparison_df)
            display(errors_df if not errors_df.empty else pd.DataFrame({"status": ["No audit errors recorded"]}))
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 3, figsize=(18, 4))
            plot_specs = [
                ("rows", "Rows per dataset", "#355070"),
                ("subjects", "Subjects per dataset", "#6d597a"),
                ("sessions", "Sessions per dataset", "#b56576"),
            ]

            for ax, (column, title, color) in zip(axes, plot_specs):
                ordered = summary_df.sort_values(column, ascending=False, kind="stable")
                ax.bar(ordered["dataset_name"], ordered[column], color=color, alpha=0.9)
                ax.set_title(title)
                ax.tick_params(axis="x", rotation=25)
                ax.set_ylabel(column)

            fig.tight_layout()
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 3, figsize=(18, 4))
            plot_specs = [
                ("sampling_rate_median_hz", "Median sampling rate (Hz)", "#588157"),
                ("acc_q50", "Acceleration magnitude q50", "#3a5a40"),
                ("acc_q99", "Acceleration magnitude q99", "#a3b18a"),
            ]

            for ax, (column, title, color) in zip(axes, plot_specs):
                ordered = summary_df.sort_values(column, ascending=False, kind="stable")
                ax.bar(ordered["dataset_name"], ordered[column], color=color, alpha=0.9)
                ax.set_title(title)
                ax.tick_params(axis="x", rotation=25)
                ax.set_ylabel(column)

            fig.tight_layout()
            """
        ),
        code_cell(
            """
            scale_hints_df = summary_df[
                ["dataset_name", "accel_scale_hint", "gyro_scale_hint", "acc_q50", "acc_q99", "gyro_q99"]
            ].copy()
            scale_hints_df
            """
        ),
    ]

    write_notebook(ANALYSIS_DIR / "00_cross_dataset_audit_overview.ipynb", cells)


def build_uci_har_notebook() -> None:
    cells = [
        md_cell(
            """
            # UCI HAR Visualisation

            This notebook loads UCI HAR through the project loader, keeping the same
            total-acceleration harmonisation choices used elsewhere in the repo.
            """
        ),
        code_cell(BOOTSTRAP_CELL),
        code_cell(
            """
            from pipeline.ingest import load_uci_har

            DATASET_PATH = REPO_ROOT / "data" / "raw" / "UCIHAR_Dataset"
            LOAD_FULL_DATASET = False
            MAX_WINDOWS_PER_SPLIT = None if LOAD_FULL_DATASET else 250

            if not DATASET_PATH.exists():
                raise FileNotFoundError(DATASET_PATH)

            df = load_uci_har(DATASET_PATH, max_windows_per_split=MAX_WINDOWS_PER_SPLIT)
            df = nb_utils.add_vector_magnitudes(df)
            df["split"] = df["session_id"].astype(str).str.extract(r"^(train|test)_", expand=False).fillna("unknown")

            print(f"Loaded {len(df):,} rows from {DATASET_PATH}")
            display(df.head())
            """
        ),
        code_cell(
            """
            profile_df = nb_utils.dataset_profile(df)
            loader_notes_df = pd.DataFrame({"loader_note": df.attrs.get("loader_notes", [])})
            label_raw_df = nb_utils.count_table(df, "label_raw", top_n=10)
            label_mapped_df = nb_utils.count_table(df, "label_mapped", top_n=10)

            display(profile_df)
            display(loader_notes_df if not loader_notes_df.empty else pd.DataFrame({"loader_note": ["No loader notes recorded"]}))
            display(label_raw_df)
            display(label_mapped_df)
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 3, figsize=(18, 4))
            nb_utils.plot_count_bars(nb_utils.count_table(df, "split", top_n=None), "split", ax=axes[0], title="Train / test window counts", color="#7f5539")
            nb_utils.plot_count_bars(label_raw_df, "label_raw", ax=axes[1], title="Raw activity labels", color="#3a86ff")
            nb_utils.plot_count_bars(label_mapped_df, "label_mapped", ax=axes[2], title="Mapped HAR labels", color="#8338ec")
            fig.tight_layout()
            """
        ),
        code_cell(
            """
            _ = nb_utils.plot_signal_histograms(df, dataset_label="UCI HAR")
            """
        ),
        code_cell(
            """
            duration_df = nb_utils.session_duration_table(df)
            display(duration_df.head())

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(duration_df["duration_s"].dropna(), bins=30, color="#ffb703", alpha=0.9)
            ax.set_title("Session duration distribution")
            ax.set_xlabel("duration_s")
            ax.set_ylabel("count")
            fig.tight_layout()
            """
        ),
        code_cell(
            """
            example_seq = nb_utils.pick_representative_sequence(df, preferred_label="locomotion", min_rows=64)
            display(example_seq[["subject_id", "session_id", "timestamp", "ax", "ay", "az", "gx", "gy", "gz", "label_mapped"]].head(12))
            _ = nb_utils.plot_sequence_axes(example_seq, title="Representative UCI HAR sequence")
            """
        ),
    ]

    write_notebook(ANALYSIS_DIR / "01_uci_har_visualisation.ipynb", cells)


def build_pamap2_notebook() -> None:
    cells = [
        md_cell(
            """
            # PAMAP2 Visualisation

            This notebook uses the same real `.dat` parser as the dissertation pipeline,
            with the hand IMU channels collapsed into the common schema.
            """
        ),
        code_cell(BOOTSTRAP_CELL),
        code_cell(
            """
            from pipeline.ingest import load_pamap2

            DATASET_PATH = REPO_ROOT / "data" / "raw" / "PAMAP2_Dataset"
            LOAD_FULL_DATASET = False
            MAX_FILES = None if LOAD_FULL_DATASET else 3
            INCLUDE_OPTIONAL = False

            if not DATASET_PATH.exists():
                raise FileNotFoundError(DATASET_PATH)

            df = load_pamap2(DATASET_PATH, max_files=MAX_FILES, include_optional=INCLUDE_OPTIONAL)
            df = nb_utils.add_vector_magnitudes(df)
            df["activity_name"] = df["label_raw"].astype(str).str.split(":", n=1).str[1].fillna(df["label_raw"].astype(str))

            print(f"Loaded {len(df):,} rows from {DATASET_PATH}")
            display(df.head())
            """
        ),
        code_cell(
            """
            profile_df = nb_utils.dataset_profile(df)
            loader_notes_df = pd.DataFrame({"loader_note": df.attrs.get("loader_notes", [])})
            activity_name_df = nb_utils.count_table(df, "activity_name", top_n=12)
            label_mapped_df = nb_utils.count_table(df, "label_mapped", top_n=10)
            subject_df = nb_utils.count_table(df, "subject_id", top_n=10)

            display(profile_df)
            display(loader_notes_df if not loader_notes_df.empty else pd.DataFrame({"loader_note": ["No loader notes recorded"]}))
            display(activity_name_df)
            display(label_mapped_df)
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 3, figsize=(18, 4))
            nb_utils.plot_count_bars(activity_name_df, "activity_name", ax=axes[0], title="Raw activity names", color="#3a86ff")
            nb_utils.plot_count_bars(label_mapped_df, "label_mapped", ax=axes[1], title="Mapped HAR labels", color="#8338ec")
            nb_utils.plot_count_bars(subject_df, "subject_id", ax=axes[2], title="Rows per subject", color="#ff006e")
            fig.tight_layout()
            """
        ),
        code_cell(
            """
            _ = nb_utils.plot_signal_histograms(df, dataset_label="PAMAP2")
            """
        ),
        code_cell(
            """
            duration_df = nb_utils.session_duration_table(df)
            display(duration_df)

            fig, ax = plt.subplots(figsize=(10, 4))
            ordered = duration_df.sort_values("duration_s", ascending=False, kind="stable")
            ax.bar(ordered["session_id"].astype(str), ordered["duration_s"], color="#2a9d8f", alpha=0.9)
            ax.set_title("Session durations by parsed file")
            ax.set_xlabel("session_id")
            ax.set_ylabel("duration_s")
            ax.tick_params(axis="x", rotation=25)
            fig.tight_layout()
            """
        ),
        code_cell(
            """
            example_seq = nb_utils.pick_representative_sequence(df, preferred_label="locomotion", min_rows=256)
            display(example_seq[["subject_id", "session_id", "timestamp", "ax", "ay", "az", "gx", "gy", "gz", "label_mapped"]].head(12))
            _ = nb_utils.plot_sequence_axes(example_seq, title="Representative PAMAP2 sequence")
            """
        ),
    ]

    write_notebook(ANALYSIS_DIR / "02_pamap2_visualisation.ipynb", cells)


def build_mobifall_notebook() -> None:
    cells = [
        md_cell(
            """
            # MobiFall Visualisation

            This notebook loads the smartphone fall dataset through the grouped-trial
            parser used by the fall pipeline.
            """
        ),
        code_cell(BOOTSTRAP_CELL),
        code_cell(
            """
            from pipeline.ingest import load_mobifall

            DATASET_PATH = REPO_ROOT / "data" / "raw" / "MOBIACT_Dataset" / "MobiFall_Dataset_v2.0"
            LOAD_FULL_DATASET = False
            MAX_TRIALS = None if LOAD_FULL_DATASET else 40

            if not DATASET_PATH.exists():
                raise FileNotFoundError(DATASET_PATH)

            df = load_mobifall(DATASET_PATH, max_files=MAX_TRIALS)
            df = nb_utils.add_vector_magnitudes(df)
            df["activity_code"] = df["session_id"].astype(str).str.split(":", n=1).str[0]

            print(f"Loaded {len(df):,} rows from {DATASET_PATH}")
            display(df.head())
            """
        ),
        code_cell(
            """
            profile_df = nb_utils.dataset_profile(df)
            loader_notes_df = pd.DataFrame({"loader_note": df.attrs.get("loader_notes", [])})
            label_raw_df = nb_utils.count_table(df, "label_raw", top_n=10)
            label_mapped_df = nb_utils.count_table(df, "label_mapped", top_n=10)
            activity_code_df = nb_utils.count_table(df, "activity_code", top_n=12)

            trial_sensor_df = (
                df.groupby("session_id", dropna=False, sort=False)[["has_gyro", "has_orientation"]]
                .max()
                .reset_index()
            )
            gyro_presence_df = (
                trial_sensor_df["has_gyro"].astype(str).value_counts().rename_axis("has_gyro").reset_index(name="count")
            )
            orientation_presence_df = (
                trial_sensor_df["has_orientation"].astype(str).value_counts().rename_axis("has_orientation").reset_index(name="count")
            )

            display(profile_df)
            display(loader_notes_df if not loader_notes_df.empty else pd.DataFrame({"loader_note": ["No loader notes recorded"]}))
            display(label_raw_df)
            display(activity_code_df)
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(2, 2, figsize=(16, 8))
            nb_utils.plot_count_bars(label_raw_df, "label_raw", ax=axes[0, 0], title="Raw labels", color="#3a86ff")
            nb_utils.plot_count_bars(label_mapped_df, "label_mapped", ax=axes[0, 1], title="Mapped fall labels", color="#ff006e")
            nb_utils.plot_count_bars(activity_code_df, "activity_code", ax=axes[1, 0], title="Trial activity codes", color="#8338ec")
            nb_utils.plot_count_bars(gyro_presence_df, "has_gyro", ax=axes[1, 1], title="Trials with gyroscope stream", color="#2a9d8f")
            fig.tight_layout()
            """
        ),
        code_cell(
            """
            _ = nb_utils.plot_signal_histograms(df, dataset_label="MobiFall")
            """
        ),
        code_cell(
            """
            duration_df = nb_utils.session_duration_table(df)
            display(duration_df.head())

            fig, ax = plt.subplots(figsize=(10, 4))
            duration_df.boxplot(column="duration_s", by="label_mapped", ax=ax, grid=False)
            ax.set_title("Trial duration by mapped label")
            ax.set_xlabel("label_mapped")
            ax.set_ylabel("duration_s")
            fig.suptitle("")
            fig.tight_layout()
            """
        ),
        code_cell(
            """
            example_seq = nb_utils.pick_representative_sequence(df, preferred_label="fall", min_rows=128)
            display(example_seq[["subject_id", "session_id", "timestamp", "ax", "ay", "az", "gx", "gy", "gz", "label_mapped"]].head(12))
            _ = nb_utils.plot_sequence_axes(example_seq, title="Representative MobiFall fall sequence")
            """
        ),
    ]

    write_notebook(ANALYSIS_DIR / "03_mobifall_visualisation.ipynb", cells)


def build_sisfall_notebook() -> None:
    cells = [
        md_cell(
            """
            # SisFall Visualisation

            This notebook inspects the SisFall files using the same raw-count conversion
            assumptions already encoded in the dissertation loader.
            """
        ),
        code_cell(BOOTSTRAP_CELL),
        code_cell(
            """
            from pipeline.ingest import load_sisfall

            DATASET_PATH = REPO_ROOT / "data" / "raw" / "SISFALL_Dataset" / "SisFall_dataset"
            LOAD_FULL_DATASET = False
            MAX_FILES = None if LOAD_FULL_DATASET else 60

            if not DATASET_PATH.exists():
                raise FileNotFoundError(DATASET_PATH)

            df = load_sisfall(DATASET_PATH, max_files=MAX_FILES)
            df = nb_utils.add_vector_magnitudes(df)
            df["event_code"] = df["session_id"].astype(str).str.split("_").str[0]
            df["subject_group"] = df["subject_id"].astype(str).str[:2]

            print(f"Loaded {len(df):,} rows from {DATASET_PATH}")
            display(df.head())
            """
        ),
        code_cell(
            """
            profile_df = nb_utils.dataset_profile(df)
            loader_notes_df = pd.DataFrame({"loader_note": df.attrs.get("loader_notes", [])})
            label_raw_df = nb_utils.count_table(df, "label_raw", top_n=10)
            label_mapped_df = nb_utils.count_table(df, "label_mapped", top_n=10)
            event_code_df = nb_utils.count_table(df, "event_code", top_n=15)
            subject_group_df = nb_utils.count_table(df, "subject_group", top_n=10)

            display(profile_df)
            display(loader_notes_df if not loader_notes_df.empty else pd.DataFrame({"loader_note": ["No loader notes recorded"]}))
            display(label_mapped_df)
            display(event_code_df)
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(2, 2, figsize=(16, 8))
            nb_utils.plot_count_bars(label_raw_df, "label_raw", ax=axes[0, 0], title="Raw labels", color="#3a86ff")
            nb_utils.plot_count_bars(label_mapped_df, "label_mapped", ax=axes[0, 1], title="Mapped fall labels", color="#ff006e")
            nb_utils.plot_count_bars(event_code_df, "event_code", ax=axes[1, 0], title="Event code frequency", color="#8338ec")
            nb_utils.plot_count_bars(subject_group_df, "subject_group", ax=axes[1, 1], title="Subject-group rows", color="#2a9d8f")
            fig.tight_layout()
            """
        ),
        code_cell(
            """
            _ = nb_utils.plot_signal_histograms(df, dataset_label="SisFall")
            """
        ),
        code_cell(
            """
            duration_df = nb_utils.session_duration_table(df)
            display(duration_df.head())

            fig, ax = plt.subplots(figsize=(10, 4))
            duration_df.boxplot(column="duration_s", by="label_mapped", ax=ax, grid=False)
            ax.set_title("Trial duration by mapped label")
            ax.set_xlabel("label_mapped")
            ax.set_ylabel("duration_s")
            fig.suptitle("")
            fig.tight_layout()
            """
        ),
        code_cell(
            """
            example_seq = nb_utils.pick_representative_sequence(df, preferred_label="fall", min_rows=256)
            display(example_seq[["subject_id", "session_id", "timestamp", "ax", "ay", "az", "gx", "gy", "gz", "label_mapped"]].head(12))
            _ = nb_utils.plot_sequence_axes(example_seq, title="Representative SisFall fall sequence")
            """
        ),
    ]

    write_notebook(ANALYSIS_DIR / "04_sisfall_visualisation.ipynb", cells)


def build_weather_notebook() -> None:
    cells = [
        md_cell(
            """
            # Weather Context Visualisation

            This notebook inspects the weather context sources already kept in `data/raw/`:
            Open-Meteo location exports, the earlier pressure-only CSV, and Meteostat
            hourly station pulls.
            """
        ),
        code_cell(BOOTSTRAP_CELL),
        code_cell(
            """
            from pipeline.ingest.weather import load_weather_csv, load_weather_csvs

            OPENMETEO_DIR = REPO_ROOT / "data" / "raw" / "weather"
            METEOSTAT_DIR = REPO_ROOT / "data" / "raw" / "METEOSTAT_Dataset"
            OPENMETEO_PRESSURE_PATH = REPO_ROOT / "data" / "raw" / "OPENMETEO_Dataset" / "open_meteo_pressure.csv"
            METEOSTAT_STATION_PATH = METEOSTAT_DIR / "meteostat_stations_used.csv"

            openmeteo_paths = sorted(OPENMETEO_DIR.glob("*.csv"))
            meteostat_paths = sorted(METEOSTAT_DIR.glob("meteostat_hourly_*.csv"))

            if not openmeteo_paths:
                raise FileNotFoundError(f"No Open-Meteo weather CSVs found in {OPENMETEO_DIR}")
            if not meteostat_paths:
                raise FileNotFoundError(f"No Meteostat weather CSVs found in {METEOSTAT_DIR}")

            openmeteo_df = load_weather_csvs(openmeteo_paths)
            meteostat_df = load_weather_csvs(meteostat_paths)
            pressure_only_df = load_weather_csv(OPENMETEO_PRESSURE_PATH) if OPENMETEO_PRESSURE_PATH.exists() else pd.DataFrame()
            station_df = pd.read_csv(METEOSTAT_STATION_PATH) if METEOSTAT_STATION_PATH.exists() else pd.DataFrame()

            display(openmeteo_df.head())
            display(meteostat_df.head())
            """
        ),
        code_cell(
            """
            openmeteo_profile_df = nb_utils.dataset_profile(openmeteo_df)
            meteostat_profile_df = nb_utils.dataset_profile(meteostat_df)
            openmeteo_missing_df = nb_utils.missing_ratio_table(openmeteo_df)
            meteostat_missing_df = nb_utils.missing_ratio_table(meteostat_df)

            print("Open-Meteo profile")
            display(openmeteo_profile_df)
            print("Meteostat profile")
            display(meteostat_profile_df)
            display(station_df if not station_df.empty else pd.DataFrame({"status": ["No Meteostat station metadata found"]}))
            """
        ),
        code_cell(
            """
            _ = nb_utils.plot_weather_lines(
                openmeteo_df,
                value_cols=["pressure_msl", "temperature_2m", "wind_speed_10m"],
                title_prefix="Open-Meteo",
            )
            """
        ),
        code_cell(
            """
            _ = nb_utils.plot_weather_lines(
                meteostat_df,
                value_cols=["pres", "temp", "wspd"],
                title_prefix="Meteostat",
            )
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 2, figsize=(16, 4))
            openmeteo_top_missing = openmeteo_missing_df.head(10)
            meteostat_top_missing = meteostat_missing_df.head(10)

            axes[0].bar(openmeteo_top_missing["column"], openmeteo_top_missing["missing_ratio"], color="#3a86ff", alpha=0.9)
            axes[0].set_title("Open-Meteo missing ratio (top 10 columns)")
            axes[0].set_ylabel("missing_ratio")
            axes[0].tick_params(axis="x", rotation=35)

            axes[1].bar(meteostat_top_missing["column"], meteostat_top_missing["missing_ratio"], color="#2a9d8f", alpha=0.9)
            axes[1].set_title("Meteostat missing ratio (top 10 columns)")
            axes[1].set_ylabel("missing_ratio")
            axes[1].tick_params(axis="x", rotation=35)

            fig.tight_layout()

            display(openmeteo_missing_df.head(10))
            display(meteostat_missing_df.head(10))
            """
        ),
        code_cell(
            """
            if pressure_only_df.empty:
                print("Pressure-only Open-Meteo CSV not found.")
            else:
                display(pressure_only_df.head())
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(pd.to_datetime(pressure_only_df["time"]), pressure_only_df["pressure_msl"], color="#ff006e", linewidth=1.2)
                ax.set_title("Open-Meteo pressure-only reference file")
                ax.set_xlabel("time")
                ax.set_ylabel("pressure_msl")
                fig.tight_layout()
            """
        ),
    ]

    write_notebook(ANALYSIS_DIR / "05_weather_context_visualisation.ipynb", cells)


def build_phone1_notebook() -> None:
    cells = [
        md_cell(
            """
            # Phone1 Runtime Visualisation

            This notebook inspects the local `phone1/` capture with the runtime folder
            adapter and, when present, overlays the saved HAR and fall outputs from the
            validation runs already stored in `results/validation/`.
            """
        ),
        code_cell(BOOTSTRAP_CELL),
        code_cell(
            """
            from pipeline.ingest.runtime_phone_folder import load_runtime_phone_folder

            PHONE_FOLDER = REPO_ROOT / "phone1"
            METADATA_PATH = PHONE_FOLDER / "Metadata.csv"
            HAR_WINDOWS_PATH = REPO_ROOT / "results" / "validation" / "phone1_har.csv"
            FALL_WINDOWS_PATH = REPO_ROOT / "results" / "validation" / "phone1_fall.csv"
            TIMELINE_PATH = REPO_ROOT / "results" / "validation" / "phone1_timeline.csv"

            if not PHONE_FOLDER.exists():
                raise FileNotFoundError(PHONE_FOLDER)

            phone_df = load_runtime_phone_folder(PHONE_FOLDER)
            phone_df = nb_utils.add_vector_magnitudes(phone_df)

            metadata_df = pd.read_csv(METADATA_PATH) if METADATA_PATH.exists() else pd.DataFrame()
            har_windows_df = pd.read_csv(HAR_WINDOWS_PATH) if HAR_WINDOWS_PATH.exists() else pd.DataFrame()
            fall_windows_df = pd.read_csv(FALL_WINDOWS_PATH) if FALL_WINDOWS_PATH.exists() else pd.DataFrame()
            timeline_df = pd.read_csv(TIMELINE_PATH) if TIMELINE_PATH.exists() else pd.DataFrame()

            print(f"Loaded {len(phone_df):,} rows from {PHONE_FOLDER}")
            display(phone_df.head())
            display(metadata_df if not metadata_df.empty else pd.DataFrame({"status": ["No Metadata.csv found"]}))
            """
        ),
        code_cell(
            """
            profile_df = nb_utils.dataset_profile(phone_df)
            missing_df = nb_utils.missing_ratio_table(phone_df, columns=["timestamp", "ax", "ay", "az", "gx", "gy", "gz", "sampling_rate_hz"])
            display(profile_df)
            display(missing_df)
            """
        ),
        code_cell(
            """
            _ = nb_utils.plot_sequence_axes(phone_df, title="Phone1 raw runtime trace", max_points=2500)
            """
        ),
        code_cell(
            """
            _ = nb_utils.plot_signal_histograms(phone_df, dataset_label="Phone1 runtime")
            """
        ),
        code_cell(
            """
            if har_windows_df.empty or fall_windows_df.empty:
                print("Saved HAR / fall window outputs were not found in results/validation.")
            else:
                har_counts_df = nb_utils.count_table(har_windows_df, "predicted_label", top_n=10)
                fall_counts_df = nb_utils.count_table(fall_windows_df, "predicted_label", top_n=10)

                fig, axes = plt.subplots(1, 3, figsize=(18, 4))
                nb_utils.plot_count_bars(har_counts_df, "predicted_label", ax=axes[0], title="HAR predicted labels", color="#3a86ff")
                nb_utils.plot_count_bars(fall_counts_df, "predicted_label", ax=axes[1], title="Fall predicted labels", color="#ff006e")

                axes[2].plot(fall_windows_df["midpoint_ts"], fall_windows_df["predicted_probability"], color="#2a9d8f", linewidth=1.2)
                axes[2].axhline(0.4, color="#6d597a", linestyle="--", linewidth=1.0, label="threshold=0.4")
                fall_hits = fall_windows_df[fall_windows_df["predicted_is_fall"].astype(bool)]
                axes[2].scatter(fall_hits["midpoint_ts"], fall_hits["predicted_probability"], color="#d00000", s=18, label="predicted fall")
                axes[2].set_title("Fall probability over runtime windows")
                axes[2].set_xlabel("midpoint_ts")
                axes[2].set_ylabel("predicted_probability")
                axes[2].legend()
                fig.tight_layout()

                display(har_windows_df.head())
                display(fall_windows_df.head())
            """
        ),
        code_cell(
            """
            if timeline_df.empty:
                print("Timeline output not found.")
            else:
                display(timeline_df.head())

                fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
                axes[0].plot(timeline_df["midpoint_ts"], timeline_df["har_predicted_confidence"], color="#3a86ff", linewidth=1.1)
                axes[0].set_title("HAR confidence over runtime timeline")
                axes[0].set_ylabel("har_predicted_confidence")

                axes[1].plot(timeline_df["midpoint_ts"], timeline_df["fall_predicted_probability"], color="#ff006e", linewidth=1.1)
                axes[1].axhline(0.4, color="#6d597a", linestyle="--", linewidth=1.0)
                fall_events = timeline_df[timeline_df["fall_predicted_is_fall"].astype(bool)]
                axes[1].scatter(fall_events["midpoint_ts"], fall_events["fall_predicted_probability"], color="#d00000", s=18)
                axes[1].set_title("Fall probability timeline")
                axes[1].set_xlabel("midpoint_ts")
                axes[1].set_ylabel("fall_predicted_probability")
                fig.tight_layout()
            """
        ),
    ]

    write_notebook(ANALYSIS_DIR / "06_phone1_runtime_visualisation.ipynb", cells)


def build_har_model_selection_notebook() -> None:
    cells = [
        md_cell(
            """
            # HAR Model Selection Evidence

            This notebook pulls together the baseline HAR result artifacts already in
            `results/runs/` and `results/validation/` so the model choice can be tied
            back to measured performance rather than a prose-only claim.
            """
        ),
        code_cell(BOOTSTRAP_CELL),
        code_cell(
            """
            from analysis.confusion_matrix_plots import confusion_matrix_to_dataframe

            har_metric_paths = nb_utils.list_matching_paths("results/runs/har_baseline_*/metrics.json", base_dir=REPO_ROOT)
            if not har_metric_paths:
                raise FileNotFoundError("No HAR baseline metrics.json files found under results/runs/")

            har_rows = []
            latest_metrics_by_dataset = {}
            for metrics_path in har_metric_paths:
                payload = nb_utils.load_json(metrics_path)
                dataset = str(payload["dataset"]).upper()
                latest_metrics_by_dataset[dataset] = payload
                split = payload.get("split", {})
                preprocessing = payload.get("preprocessing_summary", {})

                for model_name, model_key in [("Heuristic", "heuristic"), ("Random Forest", "random_forest")]:
                    model_metrics = payload.get(model_key, {}).get("metrics", {})
                    har_rows.append(
                        {
                            "run_id": metrics_path.parent.name,
                            "dataset": dataset,
                            "model": model_name,
                            "accuracy": model_metrics.get("accuracy"),
                            "macro_f1": model_metrics.get("macro_f1"),
                            "support_total": model_metrics.get("support_total"),
                            "train_rows": split.get("train_rows"),
                            "test_rows": split.get("test_rows"),
                            "windows_total": preprocessing.get("windows_total"),
                            "feature_rows": preprocessing.get("feature_rows"),
                        }
                    )

            har_baseline_df = pd.DataFrame(har_rows).sort_values(["dataset", "model", "run_id"], kind="stable").reset_index(drop=True)
            display(har_baseline_df)

            latest_importance_paths = {
                "UCI_HAR": nb_utils.latest_matching_path("results/runs/har_baseline_uci_har__*/feature_importances_random_forest.csv", base_dir=REPO_ROOT),
                "PAMAP2": nb_utils.latest_matching_path("results/runs/har_baseline_pamap2__*/feature_importances_random_forest.csv", base_dir=REPO_ROOT),
            }
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))

            accuracy_pivot = har_baseline_df.pivot_table(index="dataset", columns="model", values="accuracy", aggfunc="max")
            macro_f1_pivot = har_baseline_df.pivot_table(index="dataset", columns="model", values="macro_f1", aggfunc="max")

            accuracy_pivot.plot(kind="bar", ax=axes[0], color=["#4c78a8", "#f58518"])
            axes[0].set_title("HAR baseline accuracy")
            axes[0].set_ylabel("accuracy")
            axes[0].tick_params(axis="x", rotation=0)

            macro_f1_pivot.plot(kind="bar", ax=axes[1], color=["#4c78a8", "#f58518"])
            axes[1].set_title("HAR baseline macro F1")
            axes[1].set_ylabel("macro_f1")
            axes[1].tick_params(axis="x", rotation=0)

            fig.tight_layout()
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(2, 2, figsize=(14, 11))
            matrix_specs = [
                ("UCI_HAR", "heuristic", "UCI HAR Heuristic"),
                ("UCI_HAR", "random_forest", "UCI HAR Random Forest"),
                ("PAMAP2", "heuristic", "PAMAP2 Heuristic"),
                ("PAMAP2", "random_forest", "PAMAP2 Random Forest"),
            ]

            for ax, (dataset_key, model_key, title) in zip(axes.flatten(), matrix_specs):
                payload = latest_metrics_by_dataset[dataset_key]
                metrics = payload[model_key]["metrics"]
                matrix_df = confusion_matrix_to_dataframe(metrics["confusion_matrix"], metrics["labels"])
                nb_utils.plot_heatmap(matrix_df, title=title, ax=ax)

            fig.tight_layout()
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            for ax, (dataset_key, importance_path) in zip(axes, latest_importance_paths.items()):
                if importance_path is None or not importance_path.exists():
                    ax.text(0.5, 0.5, f"No importance file for {dataset_key}", ha="center", va="center")
                    ax.set_axis_off()
                    continue
                importance_df = pd.read_csv(importance_path).sort_values("importance", ascending=False, kind="stable").head(15)
                ax.barh(importance_df["feature"][::-1], importance_df["importance"][::-1], color="#2a9d8f", alpha=0.9)
                ax.set_title(f"{dataset_key} top random-forest features")
                ax.set_xlabel("importance")

            fig.tight_layout()
            """
        ),
        code_cell(
            """
            cross_eval_path = REPO_ROOT / "results" / "validation" / "har_cross_dataset_eval.json"
            if not cross_eval_path.exists():
                raise FileNotFoundError(cross_eval_path)

            cross_eval = nb_utils.load_json(cross_eval_path)

            cross_rows = []
            for dataset_name, block in cross_eval["within_dataset"].items():
                for model_key, label in [("heuristic", "Heuristic"), ("random_forest", "Random Forest")]:
                    metrics = block[model_key]
                    cross_rows.append(
                        {
                            "evaluation_type": "within_dataset",
                            "source_dataset": dataset_name,
                            "target_dataset": dataset_name,
                            "model": label,
                            "accuracy": metrics.get("accuracy"),
                            "macro_f1": metrics.get("macro_f1"),
                            "support_total": metrics.get("support_total"),
                        }
                    )

            for transfer_name, block in cross_eval["cross_dataset"].items():
                for model_key, label in [("heuristic", "Heuristic"), ("random_forest", "Random Forest")]:
                    metrics = block[model_key]
                    cross_rows.append(
                        {
                            "evaluation_type": "cross_dataset",
                            "source_dataset": block.get("source_dataset"),
                            "target_dataset": block.get("target_dataset"),
                            "model": label,
                            "accuracy": metrics.get("accuracy"),
                            "macro_f1": metrics.get("macro_f1"),
                            "support_total": metrics.get("support_total"),
                        }
                    )

            cross_df = pd.DataFrame(cross_rows)
            cross_df["transfer"] = cross_df["source_dataset"].astype(str) + " -> " + cross_df["target_dataset"].astype(str)
            display(cross_df)
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 2, figsize=(16, 4))

            accuracy_plot_df = cross_df.pivot_table(index="transfer", columns="model", values="accuracy", aggfunc="mean")
            macro_f1_plot_df = cross_df.pivot_table(index="transfer", columns="model", values="macro_f1", aggfunc="mean")

            accuracy_plot_df.plot(kind="bar", ax=axes[0], color=["#4c78a8", "#f58518"])
            axes[0].set_title("HAR transfer accuracy")
            axes[0].set_ylabel("accuracy")
            axes[0].tick_params(axis="x", rotation=30)

            macro_f1_plot_df.plot(kind="bar", ax=axes[1], color=["#4c78a8", "#f58518"])
            axes[1].set_title("HAR transfer macro F1")
            axes[1].set_ylabel("macro_f1")
            axes[1].tick_params(axis="x", rotation=30)

            fig.tight_layout()
            """
        ),
    ]

    write_notebook(ANALYSIS_DIR / "07_har_model_selection_evidence.ipynb", cells)


def build_har_phone_evidence_notebook() -> None:
    cells = [
        md_cell(
            """
            # HAR Phone Transfer And Adaptation Evidence

            This notebook focuses on the phone-labelled HAR evidence: which public
            training source transfers best to the phone windows, and what changes after
            adding a small amount of phone-labelled adaptation data.
            """
        ),
        code_cell(BOOTSTRAP_CELL),
        code_cell(
            """
            from analysis.confusion_matrix_plots import confusion_matrix_to_dataframe

            eval_paths = {
                "uci_har": nb_utils.latest_matching_path("results/validation/phone_har_eval_uci_har__*/phone_har_evaluation.json", base_dir=REPO_ROOT),
                "pamap2": nb_utils.latest_matching_path("results/validation/phone_har_eval_pamap2__*/phone_har_evaluation.json", base_dir=REPO_ROOT),
                "both": nb_utils.latest_matching_path("results/validation/phone_har_eval_both__*/phone_har_evaluation.json", base_dir=REPO_ROOT),
            }
            if not any(eval_paths.values()):
                raise FileNotFoundError("No phone HAR evaluation JSON files were found.")

            eval_rows = []
            eval_payloads = {}
            for train_dataset, path in eval_paths.items():
                if path is None or not path.exists():
                    continue
                payload = nb_utils.load_json(path)
                eval_payloads[train_dataset] = payload
                metrics = payload["metrics"]
                eval_rows.append(
                    {
                        "train_dataset": train_dataset,
                        "run_id": payload["run_id"],
                        "accuracy": metrics.get("accuracy"),
                        "macro_f1": metrics.get("macro_f1"),
                        "phone_labeled_rows": payload.get("phone_labeled_rows"),
                        "public_train_rows": payload.get("public_train_rows"),
                    }
                )

            phone_eval_df = pd.DataFrame(eval_rows).sort_values("train_dataset", kind="stable").reset_index(drop=True)
            display(phone_eval_df)
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            axes[0].bar(phone_eval_df["train_dataset"], phone_eval_df["accuracy"], color="#4c78a8", alpha=0.9)
            axes[0].set_title("Phone HAR accuracy by public training source")
            axes[0].set_ylabel("accuracy")

            axes[1].bar(phone_eval_df["train_dataset"], phone_eval_df["macro_f1"], color="#f58518", alpha=0.9)
            axes[1].set_title("Phone HAR macro F1 by public training source")
            axes[1].set_ylabel("macro_f1")

            fig.tight_layout()
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, len(eval_payloads), figsize=(6 * max(1, len(eval_payloads)), 5))
            axes = np.atleast_1d(axes)

            for ax, (train_dataset, payload) in zip(axes, eval_payloads.items()):
                metrics = payload["metrics"]
                matrix_df = confusion_matrix_to_dataframe(metrics["confusion_matrix"], metrics["labels"])
                nb_utils.plot_heatmap(matrix_df, title=f"Phone HAR eval: {train_dataset}", ax=ax)

            fig.tight_layout()
            """
        ),
        code_cell(
            """
            adapt_paths = {
                "uci_har": nb_utils.latest_matching_path("results/validation/phone_har_adapt_uci_har__*/phone_har_adaptation_summary.json", base_dir=REPO_ROOT),
                "pamap2": nb_utils.latest_matching_path("results/validation/phone_har_adapt_pamap2__*/phone_har_adaptation_summary.json", base_dir=REPO_ROOT),
            }

            adapt_rows = []
            adapt_payloads = {}
            for train_dataset, path in adapt_paths.items():
                if path is None or not path.exists():
                    continue
                payload = nb_utils.load_json(path)
                adapt_payloads[train_dataset] = payload
                for run_name, run_payload in payload["runs"].items():
                    metrics = run_payload["metrics"]
                    if run_name == "public_only":
                        adapt_frac = 0.0
                        phone_adapt_rows = 0
                    else:
                        phone_adapt_rows = run_payload.get("phone_adapt_rows", 0)
                        pool_rows = max(1, int(payload.get("phone_adapt_pool_rows", 1)))
                        adapt_frac = float(phone_adapt_rows / pool_rows)

                    adapt_rows.append(
                        {
                            "train_dataset": train_dataset,
                            "run_name": run_name,
                            "adapt_frac": adapt_frac,
                            "phone_adapt_rows": phone_adapt_rows,
                            "accuracy": metrics.get("accuracy"),
                            "macro_f1": metrics.get("macro_f1"),
                            "train_rows": run_payload.get("train_rows"),
                        }
                    )

            adapt_df = pd.DataFrame(adapt_rows).sort_values(["train_dataset", "adapt_frac"], kind="stable").reset_index(drop=True)
            display(adapt_df)
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))

            for train_dataset, group in adapt_df.groupby("train_dataset", sort=True):
                group = group.sort_values("adapt_frac", kind="stable")
                axes[0].plot(group["adapt_frac"], group["accuracy"], marker="o", linewidth=1.5, label=train_dataset)
                axes[1].plot(group["adapt_frac"], group["macro_f1"], marker="o", linewidth=1.5, label=train_dataset)

            axes[0].set_title("Phone adaptation accuracy")
            axes[0].set_xlabel("fraction of phone adaptation pool used")
            axes[0].set_ylabel("accuracy")
            axes[0].legend()

            axes[1].set_title("Phone adaptation macro F1")
            axes[1].set_xlabel("fraction of phone adaptation pool used")
            axes[1].set_ylabel("macro_f1")
            axes[1].legend()

            fig.tight_layout()
            """
        ),
        code_cell(
            """
            best_runs = []
            for train_dataset, group in adapt_df.groupby("train_dataset", sort=True):
                best_row = group.sort_values(["macro_f1", "accuracy"], ascending=[False, False], kind="stable").iloc[0]
                best_runs.append(best_row.to_dict())

            best_runs_df = pd.DataFrame(best_runs)
            display(best_runs_df)

            fig, axes = plt.subplots(1, len(best_runs_df), figsize=(6 * max(1, len(best_runs_df)), 5))
            axes = np.atleast_1d(axes)
            for ax, row in zip(axes, best_runs_df.to_dict(orient="records")):
                summary_path = adapt_paths[row["train_dataset"]]
                run_dir = summary_path.parent
                matrix_path = run_dir / f"{row['run_name']}_confusion_matrix.csv"
                matrix_df = pd.read_csv(matrix_path, index_col=0) if matrix_path.exists() else pd.DataFrame()
                if matrix_df.empty:
                    ax.text(0.5, 0.5, f"Missing confusion matrix\\n{matrix_path.name}", ha="center", va="center")
                    ax.set_axis_off()
                    continue
                nb_utils.plot_heatmap(matrix_df, title=f"{row['train_dataset']} best adaptation\\n{row['run_name']}", ax=ax)

            fig.tight_layout()
            """
        ),
        code_cell(
            """
            per_label_rows = []
            for row in best_runs_df.to_dict(orient="records"):
                summary_path = adapt_paths[row["train_dataset"]]
                pred_path = summary_path.parent / f"{row['run_name']}_predictions.csv"
                if not pred_path.exists():
                    continue
                pred_df = pd.read_csv(pred_path)
                label_acc = (
                    pred_df.groupby("phone_target_label", dropna=False)["is_correct"]
                    .mean()
                    .reset_index(name="accuracy")
                )
                label_acc["train_dataset"] = row["train_dataset"]
                label_acc["run_name"] = row["run_name"]
                per_label_rows.append(label_acc)

            per_label_df = pd.concat(per_label_rows, ignore_index=True) if per_label_rows else pd.DataFrame()
            display(per_label_df)
            """
        ),
    ]

    write_notebook(ANALYSIS_DIR / "08_har_phone_transfer_adaptation_evidence.ipynb", cells)


def build_fall_threshold_evidence_notebook() -> None:
    cells = [
        md_cell(
            """
            # Fall Threshold Selection Evidence

            This notebook collects the threshold sweep outputs, the selected threshold
            baseline runs, the false-alarm exports, and the logistic fall meta-model
            results so the fall-detection choice can be justified explicitly.
            """
        ),
        code_cell(BOOTSTRAP_CELL),
        code_cell(
            """
            from analysis.fall_threshold_tradeoffs import (
                best_config_by_f1,
                best_config_by_false_alarms_with_sensitivity_floor,
                best_config_by_sensitivity_with_specificity_floor,
            )

            sweep_paths = nb_utils.list_matching_paths("results/runs/fall_threshold_sweep_*/threshold_sweep_results.csv", base_dir=REPO_ROOT)
            if not sweep_paths:
                raise FileNotFoundError("No threshold sweep CSVs found under results/runs/")

            sweep_frames = []
            selection_rows = []
            for path in sweep_paths:
                df = pd.read_csv(path)
                run_id = path.parent.name
                dataset = "MOBIFALL" if "mobifall" in run_id.lower() else "SISFALL"
                sweep_variant = run_id.replace("fall_threshold_sweep_", "")
                df["dataset"] = dataset
                df["sweep_variant"] = sweep_variant
                df["run_id"] = run_id
                sweep_frames.append(df)

                best_f1 = best_config_by_f1(df)
                if best_f1:
                    selection_rows.append(
                        {
                            "dataset": dataset,
                            "sweep_variant": sweep_variant,
                            "selection_rule": "best_f1",
                            "impact_threshold": best_f1.get("impact_threshold"),
                            "confirm_post_dyn_ratio_mean_max": best_f1.get("confirm_post_dyn_ratio_mean_max"),
                            "confirm_post_var_max": best_f1.get("confirm_post_var_max"),
                            "jerk_threshold": best_f1.get("jerk_threshold"),
                            "accuracy": best_f1.get("accuracy"),
                            "sensitivity": best_f1.get("sensitivity"),
                            "specificity": best_f1.get("specificity"),
                            "precision": best_f1.get("precision"),
                            "f1": best_f1.get("f1"),
                            "false_alarms_count": best_f1.get("false_alarms_count"),
                        }
                    )

                best_sens = best_config_by_sensitivity_with_specificity_floor(df, specificity_floor=0.70)
                if best_sens:
                    selection_rows.append(
                        {
                            "dataset": dataset,
                            "sweep_variant": sweep_variant,
                            "selection_rule": "best_sensitivity_with_specificity_floor",
                            "impact_threshold": best_sens.get("impact_threshold"),
                            "confirm_post_dyn_ratio_mean_max": best_sens.get("confirm_post_dyn_ratio_mean_max"),
                            "confirm_post_var_max": best_sens.get("confirm_post_var_max"),
                            "jerk_threshold": best_sens.get("jerk_threshold"),
                            "accuracy": best_sens.get("accuracy"),
                            "sensitivity": best_sens.get("sensitivity"),
                            "specificity": best_sens.get("specificity"),
                            "precision": best_sens.get("precision"),
                            "f1": best_sens.get("f1"),
                            "false_alarms_count": best_sens.get("false_alarms_count"),
                        }
                    )

                best_low_fp = best_config_by_false_alarms_with_sensitivity_floor(df, sensitivity_floor=0.50)
                if best_low_fp:
                    selection_rows.append(
                        {
                            "dataset": dataset,
                            "sweep_variant": sweep_variant,
                            "selection_rule": "lowest_false_alarms_with_sensitivity_floor",
                            "impact_threshold": best_low_fp.get("impact_threshold"),
                            "confirm_post_dyn_ratio_mean_max": best_low_fp.get("confirm_post_dyn_ratio_mean_max"),
                            "confirm_post_var_max": best_low_fp.get("confirm_post_var_max"),
                            "jerk_threshold": best_low_fp.get("jerk_threshold"),
                            "accuracy": best_low_fp.get("accuracy"),
                            "sensitivity": best_low_fp.get("sensitivity"),
                            "specificity": best_low_fp.get("specificity"),
                            "precision": best_low_fp.get("precision"),
                            "f1": best_low_fp.get("f1"),
                            "false_alarms_count": best_low_fp.get("false_alarms_count"),
                        }
                    )

            sweeps_df = pd.concat(sweep_frames, ignore_index=True)
            selection_df = pd.DataFrame(selection_rows)
            display(selection_df)
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, sweeps_df["dataset"].nunique(), figsize=(7 * max(1, sweeps_df["dataset"].nunique()), 5))
            axes = np.atleast_1d(axes)

            for ax, (dataset_name, group) in zip(axes, sweeps_df.groupby("dataset", sort=True)):
                scatter = ax.scatter(
                    group["false_alarms_count"],
                    group["sensitivity"],
                    c=group["f1"],
                    cmap="viridis",
                    alpha=0.8,
                    edgecolors="black",
                    linewidths=0.2,
                )
                ax.set_title(f"{dataset_name} threshold sweep")
                ax.set_xlabel("false_alarms_count")
                ax.set_ylabel("sensitivity")
                plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="f1")

            fig.tight_layout()
            """
        ),
        code_cell(
            """
            top_sweep_df = (
                sweeps_df.sort_values(["dataset", "f1", "specificity", "sensitivity"], ascending=[True, False, False, False], kind="stable")
                .groupby("dataset", group_keys=False)
                .head(10)
                .reset_index(drop=True)
            )
            display(top_sweep_df[[
                "dataset",
                "sweep_variant",
                "impact_threshold",
                "confirm_post_dyn_ratio_mean_max",
                "confirm_post_var_max",
                "jerk_threshold",
                "sensitivity",
                "specificity",
                "precision",
                "f1",
                "false_alarms_count",
            ]])
            """
        ),
        code_cell(
            """
            threshold_metric_paths = {
                "MOBIFALL": nb_utils.latest_matching_path("results/runs/fall_threshold_mobifall__*/metrics.json", base_dir=REPO_ROOT),
                "SISFALL": nb_utils.latest_matching_path("results/runs/fall_threshold_sisfall__*/metrics.json", base_dir=REPO_ROOT),
            }
            threshold_rows = []
            threshold_configs = []
            false_alarm_frames = []

            for dataset_name, metrics_path in threshold_metric_paths.items():
                if metrics_path is None or not metrics_path.exists():
                    continue
                payload = nb_utils.load_json(metrics_path)
                metrics = payload["metrics"]
                config = payload["threshold_detector"]["config"]
                threshold_rows.append(
                    {
                        "dataset": dataset_name,
                        "run_id": metrics_path.parent.name,
                        "accuracy": metrics.get("accuracy"),
                        "sensitivity": metrics.get("sensitivity"),
                        "specificity": metrics.get("specificity"),
                        "precision": metrics.get("precision"),
                        "f1": metrics.get("f1"),
                        "false_alarm_count": payload.get("false_alarm_summary", {}).get("false_alarm_count"),
                    }
                )
                threshold_configs.append({"dataset": dataset_name, "run_id": metrics_path.parent.name, **config})

                false_alarm_path = metrics_path.parent / "false_alarms.csv"
                if false_alarm_path.exists():
                    false_alarm_df = pd.read_csv(false_alarm_path)
                    false_alarm_df["dataset"] = dataset_name
                    false_alarm_frames.append(false_alarm_df)

            threshold_summary_df = pd.DataFrame(threshold_rows)
            threshold_config_df = pd.DataFrame(threshold_configs)
            false_alarms_df = pd.concat(false_alarm_frames, ignore_index=True) if false_alarm_frames else pd.DataFrame()

            display(threshold_summary_df)
            display(threshold_config_df)
            """
        ),
        code_cell(
            """
            stage_comparison_path = REPO_ROOT / "results" / "reports" / "results_comparison.json"
            stage_comparison_df = pd.DataFrame(nb_utils.load_json(stage_comparison_path))
            fall_stage_df = stage_comparison_df[stage_comparison_df["stage"].isin(["threshold_baseline", "fall_meta_model"])].copy()
            display(fall_stage_df)

            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            accuracy_pivot = fall_stage_df.pivot_table(index="dataset", columns="stage", values="accuracy", aggfunc="max")
            f1_pivot = fall_stage_df.pivot_table(index="dataset", columns="stage", values="f1", aggfunc="max")

            accuracy_pivot.plot(kind="bar", ax=axes[0], color=["#577590", "#f94144"])
            axes[0].set_title("Threshold baseline vs fall meta-model accuracy")
            axes[0].set_ylabel("accuracy")
            axes[0].tick_params(axis="x", rotation=0)

            f1_pivot.plot(kind="bar", ax=axes[1], color=["#577590", "#f94144"])
            axes[1].set_title("Threshold baseline vs fall meta-model F1")
            axes[1].set_ylabel("f1")
            axes[1].tick_params(axis="x", rotation=0)

            fig.tight_layout()
            """
        ),
        code_cell(
            """
            if false_alarms_df.empty:
                print("No false_alarms.csv files found for the selected threshold runs.")
            else:
                reason_df = (
                    false_alarms_df.groupby(["dataset", "detector_reason"], dropna=False)
                    .size()
                    .reset_index(name="count")
                    .sort_values(["dataset", "count"], ascending=[True, False], kind="stable")
                )
                session_df = (
                    false_alarms_df.groupby(["dataset", "session_id"], dropna=False)
                    .size()
                    .reset_index(name="count")
                    .sort_values(["dataset", "count"], ascending=[True, False], kind="stable")
                )
                display(reason_df.head(20))
                display(session_df.head(20))

                fig, axes = plt.subplots(1, 2, figsize=(16, 4))
                for ax, (dataset_name, group) in zip(axes, reason_df.groupby("dataset", sort=True)):
                    top_group = group.head(10)
                    ax.bar(top_group["detector_reason"].astype(str), top_group["count"], color="#e76f51", alpha=0.9)
                    ax.set_title(f"{dataset_name} false-alarm reasons")
                    ax.set_ylabel("count")
                    ax.tick_params(axis="x", rotation=30)
                fig.tight_layout()
            """
        ),
    ]

    write_notebook(ANALYSIS_DIR / "09_fall_threshold_selection_evidence.ipynb", cells)


def build_fall_transfer_runtime_notebook() -> None:
    cells = [
        md_cell(
            """
            # Fall Transfer And Runtime Adaptation Evidence

            This notebook brings together the cross-dataset fall evaluations and the
            phone-hard-negative adaptation outputs so the generalisation claims can be
            tied to the validation artifacts already present in the repo.
            """
        ),
        code_cell(BOOTSTRAP_CELL),
        code_cell(
            """
            cross_paths = nb_utils.list_matching_paths("results/validation/fall_cross_dataset_eval*.json", base_dir=REPO_ROOT)
            if not cross_paths:
                raise FileNotFoundError("No fall cross-dataset evaluation JSON files found.")

            cross_rows = []
            for path in cross_paths:
                payload = nb_utils.load_json(path)
                run_name = path.stem
                train_info = payload["cross_dataset"]["train"]
                eval_info = payload["cross_dataset"]["eval"]
                threshold_metrics = payload["cross_dataset"]["threshold_target_reference"]["metrics"]
                meta_metrics = payload["cross_dataset"]["meta_model_transfer"]["metrics"]

                for method_name, metrics in [("threshold_target_reference", threshold_metrics), ("meta_model_transfer", meta_metrics)]:
                    cross_rows.append(
                        {
                            "run_name": run_name,
                            "train_source": train_info.get("train_source"),
                            "eval_source": eval_info.get("eval_source"),
                            "eval_dataset_name": eval_info.get("eval_dataset_name"),
                            "method": method_name,
                            "accuracy": metrics.get("accuracy"),
                            "sensitivity": metrics.get("sensitivity"),
                            "specificity": metrics.get("specificity"),
                            "precision": metrics.get("precision"),
                            "f1": metrics.get("f1"),
                        }
                    )

            fall_transfer_df = pd.DataFrame(cross_rows).sort_values(["train_source", "eval_source", "method"], kind="stable").reset_index(drop=True)
            fall_transfer_df["transfer"] = fall_transfer_df["train_source"].astype(str) + " -> " + fall_transfer_df["eval_source"].astype(str)
            display(fall_transfer_df)
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 2, figsize=(15, 4))
            f1_pivot = fall_transfer_df.pivot_table(index="transfer", columns="method", values="f1", aggfunc="mean")
            sens_pivot = fall_transfer_df.pivot_table(index="transfer", columns="method", values="sensitivity", aggfunc="mean")

            f1_pivot.plot(kind="bar", ax=axes[0], color=["#577590", "#f94144"])
            axes[0].set_title("Cross-dataset fall F1")
            axes[0].set_ylabel("f1")
            axes[0].tick_params(axis="x", rotation=30)

            sens_pivot.plot(kind="bar", ax=axes[1], color=["#577590", "#f94144"])
            axes[1].set_title("Cross-dataset fall sensitivity")
            axes[1].set_ylabel("sensitivity")
            axes[1].tick_params(axis="x", rotation=30)

            fig.tight_layout()
            """
        ),
        code_cell(
            """
            adaptation_paths = nb_utils.list_matching_paths("results/validation/fall_phone_adaptation_comparison*.json", base_dir=REPO_ROOT)
            if not adaptation_paths:
                raise FileNotFoundError("No fall phone adaptation comparison JSON files found.")

            adaptation_rows = []
            breakdown_rows = []
            for path in adaptation_paths:
                payload = nb_utils.load_json(path)
                variant = path.stem.replace("fall_phone_adaptation_comparison_", "")
                for eval_scope, eval_key in [("phone_eval", "phone_eval"), ("public_eval", "public_eval")]:
                    eval_block = payload[eval_key]
                    if eval_key == "phone_eval":
                        for model_variant in ["baseline", "adapted"]:
                            metrics = eval_block[model_variant]
                            adaptation_rows.append(
                                {
                                    "variant": variant,
                                    "evaluation_scope": eval_scope,
                                    "model_variant": model_variant,
                                    "accuracy": metrics.get("accuracy"),
                                    "sensitivity": metrics.get("sensitivity"),
                                    "specificity": metrics.get("specificity"),
                                    "precision": metrics.get("precision"),
                                    "f1": metrics.get("f1"),
                                    "probability_threshold": metrics.get("probability_threshold"),
                                }
                            )
                        for breakdown_name in ["baseline_false_positive_breakdown", "adapted_false_positive_breakdown"]:
                            for label_name, count in eval_block.get(breakdown_name, {}).items():
                                breakdown_rows.append(
                                    {
                                        "variant": variant,
                                        "breakdown": breakdown_name,
                                        "label": label_name,
                                        "count": count,
                                    }
                                )
                    else:
                        for model_variant in ["baseline", "adapted"]:
                            metrics = eval_block[model_variant]
                            adaptation_rows.append(
                                {
                                    "variant": variant,
                                    "evaluation_scope": eval_scope,
                                    "model_variant": model_variant,
                                    "accuracy": metrics.get("accuracy"),
                                    "sensitivity": metrics.get("sensitivity"),
                                    "specificity": metrics.get("specificity"),
                                    "precision": metrics.get("precision"),
                                    "f1": metrics.get("f1"),
                                    "probability_threshold": metrics.get("probability_threshold"),
                                }
                            )

            fall_adaptation_df = pd.DataFrame(adaptation_rows).sort_values(
                ["evaluation_scope", "variant", "model_variant"],
                kind="stable"
            ).reset_index(drop=True)
            fp_breakdown_df = pd.DataFrame(breakdown_rows)
            display(fall_adaptation_df)
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 2, figsize=(16, 4))
            phone_eval_plot = fall_adaptation_df[fall_adaptation_df["evaluation_scope"] == "phone_eval"].copy()
            public_eval_plot = fall_adaptation_df[fall_adaptation_df["evaluation_scope"] == "public_eval"].copy()

            if not phone_eval_plot.empty:
                phone_eval_pivot = phone_eval_plot.pivot_table(index="variant", columns="model_variant", values="f1", aggfunc="mean")
                phone_eval_pivot.plot(kind="bar", ax=axes[0], color=["#577590", "#f94144"])
                axes[0].set_title("Phone evaluation F1 before/after adaptation")
                axes[0].set_ylabel("f1")
                axes[0].tick_params(axis="x", rotation=20)

            if not public_eval_plot.empty:
                public_eval_pivot = public_eval_plot.pivot_table(index="variant", columns="model_variant", values="f1", aggfunc="mean")
                public_eval_pivot.plot(kind="bar", ax=axes[1], color=["#577590", "#f94144"])
                axes[1].set_title("Public evaluation F1 before/after adaptation")
                axes[1].set_ylabel("f1")
                axes[1].tick_params(axis="x", rotation=20)

            fig.tight_layout()
            """
        ),
        code_cell(
            """
            if fp_breakdown_df.empty:
                print("No phone false-positive breakdowns were recorded.")
            else:
                fp_breakdown_summary = (
                    fp_breakdown_df.groupby(["variant", "breakdown", "label"], dropna=False)["count"]
                    .sum()
                    .reset_index()
                    .sort_values(["variant", "breakdown", "count"], ascending=[True, True, False], kind="stable")
                )
                display(fp_breakdown_summary)

                fig, axes = plt.subplots(1, min(3, fp_breakdown_summary["variant"].nunique()), figsize=(6 * min(3, fp_breakdown_summary["variant"].nunique()), 4))
                axes = np.atleast_1d(axes)
                for ax, (variant, group) in zip(axes, fp_breakdown_summary.groupby("variant", sort=True)):
                    plot_df = group[group["breakdown"] == "baseline_false_positive_breakdown"].head(8)
                    ax.bar(plot_df["label"], plot_df["count"], color="#f4a261", alpha=0.9)
                    ax.set_title(f"{variant} baseline false positives")
                    ax.tick_params(axis="x", rotation=30)
                    ax.set_ylabel("count")
                fig.tight_layout()
            """
        ),
    ]

    write_notebook(ANALYSIS_DIR / "10_fall_transfer_runtime_adaptation_evidence.ipynb", cells)


def build_vulnerability_evidence_notebook() -> None:
    cells = [
        md_cell(
            """
            # Vulnerability Pipeline Evidence

            This notebook focuses on the event-state and vulnerability-state layer, using
            the aggregated comparison table plus the vulnerability evaluation summaries
            and prediction exports.
            """
        ),
        code_cell(BOOTSTRAP_CELL),
        code_cell(
            """
            comparison_path = REPO_ROOT / "results" / "reports" / "results_comparison.json"
            if not comparison_path.exists():
                raise FileNotFoundError(comparison_path)

            comparison_df = pd.DataFrame(nb_utils.load_json(comparison_path))
            display(comparison_df)
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 2, figsize=(16, 4))

            f1_pivot = comparison_df.pivot_table(index="dataset", columns="stage", values="f1", aggfunc="max")
            sens_pivot = comparison_df.pivot_table(index="dataset", columns="stage", values="sensitivity", aggfunc="max")

            f1_pivot.plot(kind="bar", ax=axes[0], colormap="tab20")
            axes[0].set_title("Stage-by-stage F1")
            axes[0].set_ylabel("f1")
            axes[0].tick_params(axis="x", rotation=0)

            sens_pivot.plot(kind="bar", ax=axes[1], colormap="tab20")
            axes[1].set_title("Stage-by-stage sensitivity")
            axes[1].set_ylabel("sensitivity")
            axes[1].tick_params(axis="x", rotation=0)

            fig.tight_layout()
            """
        ),
        code_cell(
            """
            summary_paths = nb_utils.list_matching_paths("results/runs/vulnerability_eval_*/vulnerability_eval_summary.json", base_dir=REPO_ROOT)
            if not summary_paths:
                raise FileNotFoundError("No vulnerability_eval_summary.json files found.")

            vulnerability_rows = []
            state_rows = []
            level_rows = []
            monitoring_rows = []
            latest_prediction_paths = {}

            for path in summary_paths:
                payload = nb_utils.load_json(path)
                run_id = path.parent.name
                dataset_name = ",".join(payload.get("datasets_present", [])) or run_id
                vulnerability_rows.append(
                    {
                        "run_id": run_id,
                        "dataset": dataset_name,
                        "rows_total": payload.get("rows_total"),
                        "event_profile": payload.get("event_profile"),
                        "vulnerability_profile": payload.get("vulnerability_profile"),
                        "escalations": payload.get("escalations"),
                        "deescalations": payload.get("deescalations"),
                        "fall_event_f1": payload.get("fall_event_binary_metrics", {}).get("f1"),
                        "vulnerability_f1": payload.get("vulnerability_binary_metrics", {}).get("f1"),
                    }
                )
                latest_prediction_paths[dataset_name] = path.parent / "vulnerability_eval_predictions.csv"

                for state_name, count in payload.get("event_state_counts", {}).items():
                    state_rows.append({"run_id": run_id, "dataset": dataset_name, "event_state": state_name, "count": count})
                for level_name, count in payload.get("vulnerability_level_counts", {}).items():
                    level_rows.append({"run_id": run_id, "dataset": dataset_name, "vulnerability_level": level_name, "count": count})
                for monitoring_name, count in payload.get("monitoring_state_counts", {}).items():
                    monitoring_rows.append({"run_id": run_id, "dataset": dataset_name, "monitoring_state": monitoring_name, "count": count})

            vulnerability_summary_df = pd.DataFrame(vulnerability_rows).sort_values("run_id", kind="stable").reset_index(drop=True)
            event_state_df = pd.DataFrame(state_rows)
            vulnerability_level_df = pd.DataFrame(level_rows)
            monitoring_state_df = pd.DataFrame(monitoring_rows)

            display(vulnerability_summary_df)
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 3, figsize=(18, 4))

            for ax, (title, frame, label_col, color) in zip(
                axes,
                [
                    ("Event states", event_state_df, "event_state", "#577590"),
                    ("Vulnerability levels", vulnerability_level_df, "vulnerability_level", "#f94144"),
                    ("Monitoring states", monitoring_state_df, "monitoring_state", "#2a9d8f"),
                ],
            ):
                pivot_df = frame.pivot_table(index="dataset", columns=label_col, values="count", aggfunc="sum").fillna(0.0)
                pivot_df.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
                ax.set_title(title)
                ax.set_ylabel("count")
                ax.tick_params(axis="x", rotation=0)

            fig.tight_layout()
            """
        ),
        code_cell(
            """
            prediction_rows = []
            for dataset_name, pred_path in latest_prediction_paths.items():
                if pred_path.exists():
                    pred_df = pd.read_csv(pred_path)
                    pred_df["dataset"] = dataset_name
                    prediction_rows.append(pred_df)

            predictions_df = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
            if predictions_df.empty:
                print("No vulnerability prediction CSVs found.")
            else:
                event_counts = (
                    predictions_df.groupby(["dataset", "fall_event_state"], dropna=False)
                    .size()
                    .reset_index(name="count")
                )
                monitor_counts = (
                    predictions_df.groupby(["dataset", "monitoring_state"], dropna=False)
                    .size()
                    .reset_index(name="count")
                )
                escalation_df = (
                    predictions_df.groupby("dataset", dropna=False)[["escalated", "deescalated"]]
                    .sum(numeric_only=True)
                    .reset_index()
                )
                display(event_counts)
                display(monitor_counts)
                display(escalation_df)
            """
        ),
        code_cell(
            """
            if predictions_df.empty:
                print("No prediction-level state distributions available.")
            else:
                fig, axes = plt.subplots(1, 2, figsize=(16, 4))

                event_pivot = predictions_df.pivot_table(index="dataset", columns="fall_event_state", values="window_id", aggfunc="count").fillna(0.0)
                monitor_pivot = predictions_df.pivot_table(index="dataset", columns="monitoring_state", values="window_id", aggfunc="count").fillna(0.0)

                event_pivot.plot(kind="bar", stacked=True, ax=axes[0], colormap="tab20")
                axes[0].set_title("Prediction-level fall event states")
                axes[0].set_ylabel("window count")
                axes[0].tick_params(axis="x", rotation=0)

                monitor_pivot.plot(kind="bar", stacked=True, ax=axes[1], colormap="tab20")
                axes[1].set_title("Prediction-level monitoring states")
                axes[1].set_ylabel("window count")
                axes[1].tick_params(axis="x", rotation=0)

                fig.tight_layout()
            """
        ),
    ]

    write_notebook(ANALYSIS_DIR / "11_vulnerability_pipeline_evidence.ipynb", cells)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    build_audit_overview_notebook()
    build_uci_har_notebook()
    build_pamap2_notebook()
    build_mobifall_notebook()
    build_sisfall_notebook()
    build_weather_notebook()
    build_phone1_notebook()
    build_har_model_selection_notebook()
    build_har_phone_evidence_notebook()
    build_fall_threshold_evidence_notebook()
    build_fall_transfer_runtime_notebook()
    build_vulnerability_evidence_notebook()
    print("Generated visualisation notebooks in analysis/")


if __name__ == "__main__":
    main()
