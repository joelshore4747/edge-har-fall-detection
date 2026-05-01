"""Shared helpers for dataset visualisation notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def configure_matplotlib() -> None:
    for style_name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(style_name)
            break
        except OSError:
            continue

    plt.rcParams.update(
        {
            "figure.figsize": (12, 4.5),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, out_path: str | Path, *, dpi: int = 160) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    return out


def resolve_repo_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pipeline").exists() and (candidate / "analysis").exists():
            return candidate
    raise RuntimeError("Could not locate repo root from the current working directory.")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_matching_paths(pattern: str, *, base_dir: str | Path | None = None) -> list[Path]:
    root = Path(base_dir) if base_dir is not None else resolve_repo_root()
    return sorted(root.glob(pattern))


def latest_matching_path(pattern: str, *, base_dir: str | Path | None = None) -> Path | None:
    matches = list_matching_paths(pattern, base_dir=base_dir)
    return matches[-1] if matches else None


def add_vector_magnitudes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for out_col, axes in (
        ("acc_magnitude", ("ax", "ay", "az")),
        ("gyro_magnitude", ("gx", "gy", "gz")),
    ):
        cols = [col for col in axes if col in out.columns]
        if not cols:
            continue
        numeric = out[cols].apply(pd.to_numeric, errors="coerce")
        out[out_col] = np.sqrt(numeric.pow(2).sum(axis=1))
    return out


def dataset_profile(df: pd.DataFrame) -> pd.DataFrame:
    metrics: list[dict[str, Any]] = [
        {"metric": "rows", "value": int(len(df))},
        {"metric": "columns", "value": int(len(df.columns))},
    ]

    for column in ("dataset_name", "task_type", "subject_id", "session_id", "source_file", "label_raw", "label_mapped"):
        if column in df.columns:
            metrics.append({"metric": f"{column}_unique", "value": int(df[column].nunique(dropna=True))})

    if "sampling_rate_hz" in df.columns:
        sampling = pd.to_numeric(df["sampling_rate_hz"], errors="coerce").dropna()
        metrics.extend(
            [
                {"metric": "sampling_rate_median_hz", "value": float(sampling.median()) if not sampling.empty else None},
                {"metric": "sampling_rate_min_hz", "value": float(sampling.min()) if not sampling.empty else None},
                {"metric": "sampling_rate_max_hz", "value": float(sampling.max()) if not sampling.empty else None},
            ]
        )

    if "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce").dropna()
        metrics.extend(
            [
                {"metric": "timestamp_min", "value": float(ts.min()) if not ts.empty else None},
                {"metric": "timestamp_max", "value": float(ts.max()) if not ts.empty else None},
                {"metric": "timestamp_span", "value": float(ts.max() - ts.min()) if len(ts) > 1 else None},
            ]
        )

    return pd.DataFrame(metrics)


def count_table(
    df: pd.DataFrame,
    column: str,
    *,
    top_n: int | None = 20,
    dropna: bool = False,
) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame(columns=[column, "count", "ratio"])

    series = df[column].astype("string")
    if not dropna:
        series = series.fillna("<NA>")

    total = int(series.notna().sum()) if dropna else int(len(series))
    counts = series.value_counts(dropna=dropna)
    if top_n is not None:
        counts = counts.head(int(top_n))

    out = counts.rename_axis(column).reset_index(name="count")
    out["ratio"] = out["count"] / total if total else 0.0
    return out


def missing_ratio_table(df: pd.DataFrame, *, columns: Iterable[str] | None = None) -> pd.DataFrame:
    cols = [col for col in (columns or df.columns) if col in df.columns]
    out = pd.DataFrame(
        {
            "column": cols,
            "missing_ratio": [float(df[col].isna().mean()) for col in cols],
            "missing_count": [int(df[col].isna().sum()) for col in cols],
        }
    )
    return out.sort_values(["missing_ratio", "missing_count"], ascending=[False, False], kind="stable").reset_index(drop=True)


def session_duration_table(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [col for col in ("dataset_name", "subject_id", "session_id", "source_file") if col in df.columns]
    if "timestamp" not in df.columns or not group_cols:
        return pd.DataFrame(
            columns=group_cols + ["n_rows", "start_ts", "end_ts", "duration_s", "label_mapped"]
        )

    timestamp = pd.to_numeric(df["timestamp"], errors="coerce")
    working = df.copy()
    working["timestamp"] = timestamp

    aggregations: dict[str, Any] = {
        "timestamp": ["min", "max", "size"],
    }
    if "label_mapped" in working.columns:
        aggregations["label_mapped"] = "first"
    if "label_raw" in working.columns:
        aggregations["label_raw"] = "first"

    grouped = working.groupby(group_cols, dropna=False, sort=False).agg(aggregations)
    grouped.columns = [
        "start_ts" if col == ("timestamp", "min") else
        "end_ts" if col == ("timestamp", "max") else
        "n_rows" if col == ("timestamp", "size") else
        col[0]
        for col in grouped.columns.to_flat_index()
    ]
    grouped = grouped.reset_index()
    grouped["duration_s"] = grouped["end_ts"] - grouped["start_ts"]
    return grouped.sort_values("duration_s", ascending=False, kind="stable").reset_index(drop=True)


def plot_count_bars(
    table: pd.DataFrame,
    label_col: str,
    *,
    ax: plt.Axes | None = None,
    title: str = "",
    color: str = "#1f77b4",
    rotate_xticks: int = 35,
    horizontal: bool = False,
) -> plt.Axes:
    axes = ax or plt.subplots(figsize=(10, 4))[1]
    if table.empty:
        axes.text(0.5, 0.5, "No data", ha="center", va="center")
        axes.set_axis_off()
        return axes

    labels = table[label_col].astype(str).tolist()
    values = table["count"].astype(float).tolist()
    if horizontal:
        axes.barh(labels, values, color=color, alpha=0.9)
        axes.invert_yaxis()
        axes.set_xlabel("count")
    else:
        axes.bar(labels, values, color=color, alpha=0.9)
        axes.set_ylabel("count")
        axes.tick_params(axis="x", rotation=rotate_xticks)

    axes.set_title(title)
    return axes


def metric_row(
    name: str,
    metrics: dict[str, Any],
    *,
    fields: Iterable[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {"name": name}
    for field in fields:
        row[field] = metrics.get(field)
    if extra:
        row.update(extra)
    return row


def plot_heatmap(
    matrix_df: pd.DataFrame,
    *,
    title: str,
    ax: plt.Axes | None = None,
    cmap: str = "Blues",
    value_fmt: str = "auto",
) -> plt.Axes:
    axes = ax or plt.subplots(figsize=(6, 5))[1]
    values = matrix_df.to_numpy(dtype=float)
    im = axes.imshow(values, cmap=cmap)
    axes.set_title(title)
    axes.set_xticks(range(len(matrix_df.columns)))
    axes.set_yticks(range(len(matrix_df.index)))
    axes.set_xticklabels([str(v) for v in matrix_df.columns], rotation=35, ha="right")
    axes.set_yticklabels([str(v) for v in matrix_df.index])
    axes.set_xlabel(matrix_df.columns.name or "Predicted")
    axes.set_ylabel(matrix_df.index.name or "True")

    vmax = float(np.nanmax(values)) if values.size else 0.0
    threshold = vmax * 0.5 if vmax > 0 else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if value_fmt == "auto":
                text = f"{int(val)}" if float(val).is_integer() else f"{val:.2f}"
            else:
                text = format(val, value_fmt)
            color = "white" if val > threshold else "black"
            axes.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    plt.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    return axes


def _sample_series(series: pd.Series, *, max_points: int) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) <= max_points:
        return clean
    return clean.sample(n=max_points, random_state=42)


def plot_signal_histograms(
    df: pd.DataFrame,
    *,
    dataset_label: str = "",
    max_points: int = 50000,
) -> plt.Figure:
    working = add_vector_magnitudes(df)
    plot_specs = [("acc_magnitude", "Acceleration magnitude", "#2a6f97")]
    if "gyro_magnitude" in working.columns and working["gyro_magnitude"].notna().any():
        plot_specs.append(("gyro_magnitude", "Gyroscope magnitude", "#c1121f"))

    fig, axes = plt.subplots(1, len(plot_specs), figsize=(6.2 * len(plot_specs), 4.2))
    axes = np.atleast_1d(axes)

    for ax, (column, title, color) in zip(axes, plot_specs, strict=False):
        sampled = _sample_series(working[column], max_points=max_points)
        if sampled.empty:
            ax.text(0.5, 0.5, "No numeric samples", ha="center", va="center")
            ax.set_axis_off()
            continue
        ax.hist(sampled, bins=50, color=color, alpha=0.85)
        ax.set_title(f"{dataset_label} {title}".strip())
        ax.set_xlabel(column)
        ax.set_ylabel("count")

    fig.tight_layout()
    return fig


def pick_representative_sequence(
    df: pd.DataFrame,
    *,
    preferred_label: str | None = None,
    label_col: str = "label_mapped",
    min_rows: int = 32,
) -> pd.DataFrame:
    working = df.copy()
    if preferred_label and label_col in working.columns:
        filtered = working[working[label_col].astype(str) == preferred_label].copy()
        if not filtered.empty:
            working = filtered

    group_cols = [col for col in ("subject_id", "session_id", "source_file") if col in working.columns]
    if not group_cols:
        return _sort_sequence(working)

    sizes = working.groupby(group_cols, dropna=False, sort=False).size().sort_values(ascending=False, kind="stable")
    for key in sizes.index:
        key_tuple = key if isinstance(key, tuple) else (key,)
        mask = pd.Series(True, index=working.index)
        for col, value in zip(group_cols, key_tuple, strict=False):
            if pd.isna(value):
                mask &= working[col].isna()
            else:
                mask &= working[col].eq(value)
        candidate = _sort_sequence(working.loc[mask].copy())
        if len(candidate) >= min_rows:
            return candidate

    first_key = sizes.index[0]
    key_tuple = first_key if isinstance(first_key, tuple) else (first_key,)
    mask = pd.Series(True, index=working.index)
    for col, value in zip(group_cols, key_tuple, strict=False):
        if pd.isna(value):
            mask &= working[col].isna()
        else:
            mask &= working[col].eq(value)
    return _sort_sequence(working.loc[mask].copy())


def _sort_sequence(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [col for col in ("timestamp", "row_index") if col in df.columns]
    if not sort_cols:
        return df.reset_index(drop=True)
    return df.sort_values(sort_cols, kind="stable").reset_index(drop=True)


def plot_sequence_axes(
    sequence_df: pd.DataFrame,
    *,
    title: str,
    max_points: int = 1200,
) -> plt.Figure:
    working = add_vector_magnitudes(_sort_sequence(sequence_df))
    if len(working) > max_points:
        step = max(1, int(np.ceil(len(working) / max_points)))
        working = working.iloc[::step].reset_index(drop=True)

    time_axis = (
        pd.to_numeric(working["timestamp"], errors="coerce").to_numpy(dtype=float)
        if "timestamp" in working.columns
        else np.arange(len(working), dtype=float)
    )

    has_gyro = any(col in working.columns and working[col].notna().any() for col in ("gx", "gy", "gz"))
    nrows = 2 if has_gyro else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 3.8 * nrows), sharex=True)
    axes = np.atleast_1d(axes)

    for col, color in (("ax", "#1d3557"), ("ay", "#457b9d"), ("az", "#2a9d8f")):
        if col in working.columns:
            axes[0].plot(time_axis, pd.to_numeric(working[col], errors="coerce"), label=col, linewidth=1.1, alpha=0.9, color=color)
    if "acc_magnitude" in working.columns:
        axes[0].plot(time_axis, working["acc_magnitude"], label="acc_magnitude", linewidth=1.6, color="#e76f51")
    axes[0].set_title(title)
    axes[0].set_ylabel("accel")
    axes[0].legend(ncol=4, loc="upper right")

    if has_gyro:
        for col, color in (("gx", "#6a040f"), ("gy", "#9d0208"), ("gz", "#dc2f02")):
            if col in working.columns:
                axes[1].plot(time_axis, pd.to_numeric(working[col], errors="coerce"), label=col, linewidth=1.1, alpha=0.9, color=color)
        if "gyro_magnitude" in working.columns:
            axes[1].plot(time_axis, working["gyro_magnitude"], label="gyro_magnitude", linewidth=1.6, color="#f48c06")
        axes[1].set_ylabel("gyro")
        axes[1].legend(ncol=4, loc="upper right")

    axes[-1].set_xlabel("timestamp (s)")
    fig.tight_layout()
    return fig


def plot_weather_lines(
    df: pd.DataFrame,
    *,
    value_cols: Iterable[str],
    time_col: str = "time",
    title_prefix: str = "",
) -> plt.Figure:
    cols = [col for col in value_cols if col in df.columns and df[col].notna().any()]
    if not cols:
        raise ValueError("No requested weather columns are available for plotting.")
    if time_col not in df.columns:
        raise ValueError(f"Missing required time column: {time_col}")

    working = df.copy()
    working[time_col] = pd.to_datetime(working[time_col], errors="coerce")
    working = working.dropna(subset=[time_col]).sort_values([time_col], kind="stable")

    fig, axes = plt.subplots(len(cols), 1, figsize=(12, 3.5 * len(cols)), sharex=True)
    axes = np.atleast_1d(axes)

    location_groups = (
        working.groupby("location_name", dropna=False, sort=True)
        if "location_name" in working.columns
        else [("all", working)]
    )

    for ax, column in zip(axes, cols, strict=False):
        for location_name, group in location_groups:
            label = str(location_name)
            ax.plot(group[time_col], pd.to_numeric(group[column], errors="coerce"), label=label, linewidth=1.1, alpha=0.9)
        ax.set_title(f"{title_prefix} {column}".strip())
        ax.set_ylabel(column)
        ax.legend(ncol=3, loc="upper right")

    axes[-1].set_xlabel(time_col)
    fig.tight_layout()
    return fig


def audit_summary_frame(audit_payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_name, summary in audit_payload.get("datasets", {}).items():
        rows.append(
            {
                "dataset_name": dataset_name,
                "rows": summary.get("rows"),
                "subjects": summary.get("subjects"),
                "sessions": summary.get("sessions"),
                "sampling_rate_median_hz": summary.get("sampling_rate_hz", {}).get("median"),
                "acc_q50": summary.get("acc_magnitude_quantiles", {}).get("q50"),
                "acc_q99": summary.get("acc_magnitude_quantiles", {}).get("q99"),
                "gyro_q99": summary.get("gyro_magnitude_quantiles", {}).get("q99"),
                "accel_scale_hint": summary.get("accel_scale_hint"),
                "gyro_scale_hint": summary.get("gyro_scale_hint"),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_name", kind="stable").reset_index(drop=True)
