"""Export concise, browser-ready figures for the admin dashboard."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import notebook_utils as nb_utils


REPO_ROOT = nb_utils.resolve_repo_root(Path(__file__).resolve())
DEFAULT_OUTPUT_DIR = REPO_ROOT / "apps" / "admin" / "public" / "figures"


def _manifest_row(
    *,
    slug: str,
    title: str,
    purpose: str,
    recommended_page: str,
    source_files: list[str],
    output_path: Path,
) -> dict[str, Any]:
    return {
        "slug": slug,
        "title": title,
        "purpose": purpose,
        "recommended_page": recommended_page,
        "source_files": source_files,
        "output_path": str(output_path.relative_to(REPO_ROOT)),
    }


def _save_manifest(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return manifest_path


def export_dataset_inventory(output_dir: Path) -> dict[str, Any]:
    audit_path = REPO_ROOT / "results" / "validation" / "dataset_distribution_audit.json"
    audit_payload = nb_utils.load_json(audit_path)
    summary_df = nb_utils.audit_summary_frame(audit_payload).sort_values("rows", ascending=False, kind="stable")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    specs = [
        ("rows", "Rows per dataset (log scale)", "#355070"),
        ("subjects", "Subjects per dataset", "#6d597a"),
        ("sessions", "Sessions per dataset", "#b56576"),
    ]
    for ax, (column, title, color) in zip(axes, specs, strict=False):
        ax.bar(summary_df["dataset_name"], summary_df[column], color=color, alpha=0.9)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylabel(column)
    axes[0].set_yscale("log")
    fig.suptitle("UniFallMonitor dataset inventory", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_path = nb_utils.save_figure(fig, output_dir / "dataset_inventory.svg")
    plt.close(fig)
    return _manifest_row(
        slug="dataset_inventory",
        title="Dataset Inventory",
        purpose="Compact cross-dataset inventory for admin docs and research-library views.",
        recommended_page="docs or library",
        source_files=[str(audit_path.relative_to(REPO_ROOT))],
        output_path=out_path,
    )


def export_primary_model_comparison(output_dir: Path) -> dict[str, Any]:
    comparison_path = REPO_ROOT / "results" / "reports" / "primary_comparison.csv"
    df = pd.read_csv(comparison_path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    specs = [
        ("f1", "F1", "#1d4ed8"),
        ("sensitivity", "Sensitivity", "#dc2626"),
        ("specificity", "Specificity", "#059669"),
    ]
    x_positions = range(len(df))
    width = 0.35

    for ax, (metric, title, color) in zip(axes, specs, strict=False):
        threshold_col = f"threshold_{metric}"
        vulnerability_col = f"vulnerability_{metric}"
        ax.bar(
            [x - width / 2 for x in x_positions],
            df[threshold_col],
            width=width,
            color="#94a3b8",
            label="threshold",
        )
        ax.bar(
            [x + width / 2 for x in x_positions],
            df[vulnerability_col],
            width=width,
            color=color,
            label="vulnerability",
        )
        ax.set_title(title)
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(df["dataset"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("score")
        if ax is axes[0]:
            ax.legend()

    fig.suptitle("Threshold vs vulnerability pipeline comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_path = nb_utils.save_figure(fig, output_dir / "primary_model_comparison.svg")
    plt.close(fig)
    return _manifest_row(
        slug="primary_model_comparison",
        title="Primary Model Comparison",
        purpose="Results-page comparison between the threshold baseline and the vulnerability pipeline.",
        recommended_page="results",
        source_files=[str(comparison_path.relative_to(REPO_ROOT))],
        output_path=out_path,
    )


def export_har_domain_adaptation_delta(output_dir: Path) -> dict[str, Any]:
    source_data_path = REPO_ROOT / "results" / "validation" / "har_domain_adaptation_eval.csv"
    source_svg_path = REPO_ROOT / "results" / "validation" / "har_domain_adaptation_delta.svg"
    out_path = output_dir / "har_domain_adaptation_delta.svg"
    if source_svg_path.exists() and source_svg_path.resolve() != out_path.resolve():
        shutil.copyfile(source_svg_path, out_path)

    return _manifest_row(
        slug="har_domain_adaptation_delta",
        title="HAR Domain Adaptation Delta",
        purpose="Dissertation figure showing offline HAR adaptation macro-F1 deltas and paired bootstrap confidence intervals.",
        recommended_page="docs or results",
        source_files=[str(source_data_path.relative_to(REPO_ROOT))],
        output_path=out_path,
    )


def export_phone1_runtime_timeline(output_dir: Path) -> dict[str, Any]:
    timeline_path = REPO_ROOT / "results" / "validation" / "phone1_timeline.csv"
    df = pd.read_csv(timeline_path)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(df["midpoint_ts"], df["har_predicted_confidence"], color="#2563eb", linewidth=1.4)
    axes[0].set_title("Self-collected session HAR confidence")
    axes[0].set_ylabel("confidence")
    axes[0].set_ylim(0, 1.05)

    axes[1].plot(df["midpoint_ts"], df["fall_predicted_probability"], color="#dc2626", linewidth=1.4)
    axes[1].axhline(0.4, color="#7c3aed", linestyle="--", linewidth=1.0, label="fall threshold")
    fall_hits = df[df["fall_predicted_is_fall"].astype(str).str.lower() == "true"]
    axes[1].scatter(
        fall_hits["midpoint_ts"],
        fall_hits["fall_predicted_probability"],
        color="#111827",
        s=18,
        label="predicted fall",
        zorder=3,
    )
    axes[1].set_title("Self-collected session fall probability")
    axes[1].set_xlabel("midpoint_ts")
    axes[1].set_ylabel("probability")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    fig.suptitle("Self-collected phone-session evidence timeline", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_path = nb_utils.save_figure(fig, output_dir / "phone1_runtime_timeline.svg")
    plt.close(fig)
    return _manifest_row(
        slug="phone1_runtime_timeline",
        title="Self-Collected Phone-Session Runtime Timeline",
        purpose="Session-detail visual showing HAR confidence and fall probability over time.",
        recommended_page="session detail",
        source_files=[str(timeline_path.relative_to(REPO_ROOT))],
        output_path=out_path,
    )


def export_phone1_annotation_alignment(output_dir: Path) -> dict[str, Any]:
    intervals_path = REPO_ROOT / "results" / "validation" / "phone1_har_interval_comparison.csv"
    df = pd.read_csv(intervals_path)
    alignment_df = pd.crosstab(df["annotation_label"], df["dominant_har_label"])

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    nb_utils.plot_heatmap(alignment_df, title="Self-collected session: annotation vs HAR label", ax=ax)
    fig.tight_layout()

    out_path = nb_utils.save_figure(fig, output_dir / "phone1_annotation_alignment.svg")
    plt.close(fig)
    return _manifest_row(
        slug="phone1_annotation_alignment",
        title="Self-Collected Phone-Session Annotation Alignment",
        purpose="Evaluation visual for checking how dominant HAR labels align with manual runtime interval annotations.",
        recommended_page="docs or results",
        source_files=[str(intervals_path.relative_to(REPO_ROOT))],
        output_path=out_path,
    )


def export_phone1_grouped_fall_events(output_dir: Path) -> dict[str, Any]:
    events_path = REPO_ROOT / "results" / "validation" / "phone1_fall_grouped_events.csv"
    df = pd.read_csv(events_path).sort_values("peak_probability", ascending=False, kind="stable").head(12)
    ordered = df.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.barh(ordered["event_id"], ordered["peak_probability"], color="#be123c", alpha=0.9)
    for idx, row in ordered.iterrows():
        ax.text(
            min(1.02, float(row["peak_probability"]) + 0.015),
            idx,
            f"{int(row['n_positive_windows'])} windows | {row['event_duration_seconds']:.2f}s",
            va="center",
            fontsize=8.5,
            color="#334155",
        )
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("peak fall probability")
    ax.set_ylabel("event_id")
    ax.set_title("Grouped fall events from self-collected replay")
    fig.tight_layout()

    out_path = nb_utils.save_figure(fig, output_dir / "phone1_grouped_fall_events.svg")
    plt.close(fig)
    return _manifest_row(
        slug="phone1_grouped_fall_events",
        title="Self-Collected Phone-Session Grouped Fall Events",
        purpose="Evidence summary showing the highest-probability grouped fall events for admin review.",
        recommended_page="session detail or evidence",
        source_files=[str(events_path.relative_to(REPO_ROOT))],
        output_path=out_path,
    )


def export_admin_visuals(output_dir: Path | None = None) -> list[dict[str, Any]]:
    nb_utils.configure_matplotlib()
    resolved_output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        export_dataset_inventory(resolved_output_dir),
        export_primary_model_comparison(resolved_output_dir),
        export_har_domain_adaptation_delta(resolved_output_dir),
        export_phone1_runtime_timeline(resolved_output_dir),
        export_phone1_annotation_alignment(resolved_output_dir),
        export_phone1_grouped_fall_events(resolved_output_dir),
    ]
    _save_manifest(rows, resolved_output_dir)
    return rows


def main() -> None:
    rows = export_admin_visuals()
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
