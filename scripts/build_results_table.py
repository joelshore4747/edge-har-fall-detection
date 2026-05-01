from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DATASET_NAMES = ("mobifall", "sisfall")

RESEARCH_QUESTION = (
    "Does the fused vulnerability pipeline improve alert quality relative to "
    "threshold-only fall detection while preserving enough specificity to remain credible?"
)

POSITIVE_DEFINITION = {
    "threshold_baseline": "predicted fall window",
    "vulnerability_pipeline": "vulnerability_level in {medium, high}",
    "ground_truth": "true_label == fall",
}

EVALUATION_GUARDRAILS = {
    "min_absolute_f1_gain": 0.20,
    "min_vulnerability_specificity": 0.50,
    "max_f1_drop_vs_meta_model": 0.05,
}


@dataclass(slots=True)
class ResultRow:
    dataset: str
    stage: str
    profile: str
    run_name: str
    source_json: str
    accuracy: float | None
    sensitivity: float | None
    specificity: float | None
    precision: float | None
    f1: float | None
    roc_auc: float | None = None
    average_precision: float | None = None
    brier_score: float | None = None
    support_total: int | None = None
    support_positive: int | None = None
    support_negative: int | None = None


PRIMARY_STAGE_ORDER = {
    "threshold_baseline": 0,
    "vulnerability_pipeline": 1,
    "fall_meta_model": 2,
    "fall_event_pipeline": 3,
}


def _safe_float(value: Any) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_matching_files(root: Path, pattern: str) -> list[Path]:
    return sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)


def _latest_matching_file(root: Path, pattern: str) -> Path | None:
    matches = _latest_matching_files(root, pattern)
    return matches[0] if matches else None


def _infer_dataset_from_path(path: Path) -> str:
    text = str(path).lower()
    for dataset in DATASET_NAMES:
        if dataset in text:
            return dataset
    return "unknown"


def _metrics_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "metrics" in payload and isinstance(payload["metrics"], dict):
        return payload["metrics"]

    if {"accuracy", "sensitivity", "specificity", "precision", "f1"}.issubset(payload.keys()):
        return payload

    if "test_metrics" in payload and isinstance(payload["test_metrics"], dict):
        return payload["test_metrics"]

    raise ValueError("Could not locate metric payload in JSON")


def _row_from_metrics(
    *,
    dataset: str,
    stage: str,
    profile: str,
    run_name: str,
    source_json: Path,
    metrics: dict[str, Any],
) -> ResultRow:
    return ResultRow(
        dataset=dataset.upper(),
        stage=stage,
        profile=profile,
        run_name=run_name,
        source_json=str(source_json),
        accuracy=_safe_float(metrics.get("accuracy")),
        sensitivity=_safe_float(metrics.get("sensitivity")),
        specificity=_safe_float(metrics.get("specificity")),
        precision=_safe_float(metrics.get("precision")),
        f1=_safe_float(metrics.get("f1")),
        roc_auc=_safe_float(metrics.get("roc_auc")),
        average_precision=_safe_float(metrics.get("average_precision")),
        brier_score=_safe_float(metrics.get("brier_score")),
        support_total=_safe_int(metrics.get("support_total")),
        support_positive=_safe_int(metrics.get("support_positive") or metrics.get("support_fall")),
        support_negative=_safe_int(metrics.get("support_negative") or metrics.get("support_non_fall")),
    )


def parse_threshold_metrics(path: Path) -> ResultRow:
    payload = _load_json(path)
    metrics = _metrics_from_payload(payload)
    dataset = _infer_dataset_from_path(path)
    return _row_from_metrics(
        dataset=dataset,
        stage="threshold_baseline",
        profile="default",
        run_name=path.parent.name,
        source_json=path,
        metrics=metrics,
    )


def parse_meta_run_summary(path: Path, *, profile: str = "logistic_regression") -> ResultRow:
    payload = _load_json(path)
    metrics = _metrics_from_payload(payload)
    dataset = _infer_dataset_from_path(path)
    return _row_from_metrics(
        dataset=dataset,
        stage="fall_meta_model",
        profile=profile,
        run_name=path.parent.name,
        source_json=path,
        metrics=metrics,
    )


def parse_vulnerability_eval_summary(path: Path) -> list[ResultRow]:
    payload = _load_json(path)
    dataset = _infer_dataset_from_path(path)
    run_name = path.parent.name

    rows: list[ResultRow] = []

    if "fall_event_binary_metrics" in payload:
        rows.append(
            _row_from_metrics(
                dataset=dataset,
                stage="fall_event_pipeline",
                profile=str(payload.get("event_profile", "unknown")),
                run_name=run_name,
                source_json=path,
                metrics=payload["fall_event_binary_metrics"],
            )
        )

    if "vulnerability_binary_metrics" in payload:
        rows.append(
            _row_from_metrics(
                dataset=dataset,
                stage="vulnerability_pipeline",
                profile=str(payload.get("vulnerability_profile", "unknown")),
                run_name=run_name,
                source_json=path,
                metrics=payload["vulnerability_binary_metrics"],
            )
        )

    return rows


def _best_vulnerability_eval_summary_for_dataset(results_root: Path, dataset: str) -> Path | None:
    """
    Choose the best vulnerability evaluation summary for a dataset.

    Preference order:
    1. highest vulnerability F1
    2. if tied, highest fall-event F1
    3. newest file
    """
    candidates = _latest_matching_files(
        results_root,
        f"vulnerability_eval_{dataset}__*/vulnerability_eval_summary.json",
    )
    if not candidates:
        return None

    scored: list[tuple[float, float, float, Path]] = []
    for path in candidates:
        try:
            payload = _load_json(path)
            vuln_metrics = payload.get("vulnerability_binary_metrics", {}) or {}
            event_metrics = payload.get("fall_event_binary_metrics", {}) or {}
            vuln_f1 = _safe_float(vuln_metrics.get("f1")) or float("-inf")
            event_f1 = _safe_float(event_metrics.get("f1")) or float("-inf")
            mtime = path.stat().st_mtime
            scored.append((vuln_f1, event_f1, mtime, path))
        except Exception:
            continue

    if not scored:
        return candidates[0]

    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return scored[0][3]


def _best_meta_summary_for_dataset(results_root: Path, dataset: str) -> tuple[Path | None, str]:
    """
    Choose the best meta-model summary for a dataset.

    Preference order:
    1. tuned run if it has higher F1
    2. otherwise untuned logistic regression
    """
    candidates: list[tuple[Path, str]] = []

    latest_regular = _latest_matching_file(
        results_root,
        f"fall_meta_{dataset}_lr__*/run_summary.json",
    )
    if latest_regular is not None:
        candidates.append((latest_regular, "logistic_regression"))

    latest_tuned = _latest_matching_file(
        results_root,
        f"fall_meta_{dataset}_tuned_lr__*/run_summary.json",
    )
    if latest_tuned is not None:
        candidates.append((latest_tuned, "logistic_regression_tuned"))

    if not candidates:
        return None, "unknown"

    best_path: Path | None = None
    best_profile = "unknown"
    best_f1 = float("-inf")
    best_mtime = float("-inf")

    for path, profile in candidates:
        try:
            payload = _load_json(path)
            metrics = _metrics_from_payload(payload)
            f1 = _safe_float(metrics.get("f1")) or float("-inf")
            mtime = path.stat().st_mtime
            if (f1 > best_f1) or (f1 == best_f1 and mtime > best_mtime):
                best_path = path
                best_profile = profile
                best_f1 = f1
                best_mtime = mtime
        except Exception:
            continue

    if best_path is None:
        return candidates[0]

    return best_path, best_profile


def discover_latest_rows(results_root: Path) -> list[ResultRow]:
    rows: list[ResultRow] = []

    for dataset in DATASET_NAMES:
        threshold_json = _latest_matching_file(
            results_root,
            f"fall_threshold_{dataset}__*/metrics.json",
        )
        if threshold_json is not None:
            rows.append(parse_threshold_metrics(threshold_json))

        meta_summary, meta_profile = _best_meta_summary_for_dataset(results_root, dataset)
        if meta_summary is not None:
            rows.append(parse_meta_run_summary(meta_summary, profile=meta_profile))

        vuln_summary = _best_vulnerability_eval_summary_for_dataset(results_root, dataset)
        if vuln_summary is not None:
            rows.extend(parse_vulnerability_eval_summary(vuln_summary))

    return rows


def build_results_dataframe(rows: list[ResultRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset",
                "stage",
                "profile",
                "run_name",
                "accuracy",
                "sensitivity",
                "specificity",
                "precision",
                "f1",
                "roc_auc",
                "average_precision",
                "brier_score",
                "support_total",
                "support_positive",
                "support_negative",
                "source_json",
            ]
        )

    df = pd.DataFrame([asdict(row) for row in rows])

    df["_stage_order"] = df["stage"].map(PRIMARY_STAGE_ORDER).fillna(999)
    df = df.sort_values(["dataset", "_stage_order", "profile", "run_name"]).drop(columns=["_stage_order"])
    return df.reset_index(drop=True)


def build_primary_comparison_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset",
        "threshold_f1",
        "vulnerability_f1",
        "absolute_f1_gain",
        "relative_f1_gain_pct",
        "threshold_sensitivity",
        "vulnerability_sensitivity",
        "threshold_specificity",
        "vulnerability_specificity",
        "threshold_run_name",
        "vulnerability_run_name",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for dataset, dataset_df in df.groupby("dataset", sort=True):
        threshold_rows = dataset_df[dataset_df["stage"] == "threshold_baseline"]
        vulnerability_rows = dataset_df[dataset_df["stage"] == "vulnerability_pipeline"]
        if threshold_rows.empty or vulnerability_rows.empty:
            continue

        threshold_row = threshold_rows.iloc[0]
        vulnerability_row = vulnerability_rows.iloc[0]

        threshold_f1 = _safe_float(threshold_row.get("f1"))
        vulnerability_f1 = _safe_float(vulnerability_row.get("f1"))
        absolute_f1_gain = None
        relative_f1_gain_pct = None
        if threshold_f1 is not None and vulnerability_f1 is not None:
            absolute_f1_gain = vulnerability_f1 - threshold_f1
            if threshold_f1 > 0:
                relative_f1_gain_pct = (absolute_f1_gain / threshold_f1) * 100.0

        rows.append(
            {
                "dataset": dataset,
                "threshold_f1": threshold_f1,
                "vulnerability_f1": vulnerability_f1,
                "absolute_f1_gain": absolute_f1_gain,
                "relative_f1_gain_pct": relative_f1_gain_pct,
                "threshold_sensitivity": _safe_float(threshold_row.get("sensitivity")),
                "vulnerability_sensitivity": _safe_float(vulnerability_row.get("sensitivity")),
                "threshold_specificity": _safe_float(threshold_row.get("specificity")),
                "vulnerability_specificity": _safe_float(vulnerability_row.get("specificity")),
                "threshold_run_name": str(threshold_row.get("run_name", "")),
                "vulnerability_run_name": str(vulnerability_row.get("run_name", "")),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def build_evaluation_contract(df: pd.DataFrame) -> dict[str, Any]:
    primary_df = build_primary_comparison_dataframe(df)
    meta_rows = {
        str(row["dataset"]): row
        for _, row in df[df["stage"] == "fall_meta_model"].iterrows()
    } if not df.empty else {}

    dataset_verdicts: list[dict[str, Any]] = []
    for _, row in primary_df.iterrows():
        dataset = str(row["dataset"])
        threshold_f1 = _safe_float(row.get("threshold_f1"))
        vulnerability_f1 = _safe_float(row.get("vulnerability_f1"))
        threshold_sensitivity = _safe_float(row.get("threshold_sensitivity"))
        vulnerability_sensitivity = _safe_float(row.get("vulnerability_sensitivity"))
        threshold_specificity = _safe_float(row.get("threshold_specificity"))
        vulnerability_specificity = _safe_float(row.get("vulnerability_specificity"))
        absolute_f1_gain = _safe_float(row.get("absolute_f1_gain"))

        meta_row = meta_rows.get(dataset)
        meta_f1 = _safe_float(meta_row.get("f1")) if meta_row is not None else None
        f1_drop_vs_meta = None
        if meta_f1 is not None and vulnerability_f1 is not None:
            f1_drop_vs_meta = meta_f1 - vulnerability_f1

        sensitivity_uplift = None
        if threshold_sensitivity is not None and vulnerability_sensitivity is not None:
            sensitivity_uplift = vulnerability_sensitivity - threshold_sensitivity

        threshold_false_positive_rate = None
        if threshold_specificity is not None:
            threshold_false_positive_rate = 1.0 - threshold_specificity

        vulnerability_false_positive_rate = None
        if vulnerability_specificity is not None:
            vulnerability_false_positive_rate = 1.0 - vulnerability_specificity

        checks = {
            "absolute_f1_gain": (
                absolute_f1_gain is not None
                and absolute_f1_gain >= float(EVALUATION_GUARDRAILS["min_absolute_f1_gain"])
            ),
            "sensitivity_not_worse_than_threshold": (
                threshold_sensitivity is not None
                and vulnerability_sensitivity is not None
                and vulnerability_sensitivity >= threshold_sensitivity
            ),
            "specificity_floor": (
                vulnerability_specificity is not None
                and vulnerability_specificity >= float(EVALUATION_GUARDRAILS["min_vulnerability_specificity"])
            ),
            "retains_meta_model_signal": (
                f1_drop_vs_meta is not None
                and f1_drop_vs_meta <= float(EVALUATION_GUARDRAILS["max_f1_drop_vs_meta_model"])
            ),
        }
        passes_contract = bool(checks) and all(checks.values())

        dataset_verdicts.append(
            {
                "dataset": dataset,
                "metrics": {
                    "threshold_f1": threshold_f1,
                    "vulnerability_f1": vulnerability_f1,
                    "absolute_f1_gain": absolute_f1_gain,
                    "relative_f1_gain_pct": _safe_float(row.get("relative_f1_gain_pct")),
                    "threshold_sensitivity": threshold_sensitivity,
                    "vulnerability_sensitivity": vulnerability_sensitivity,
                    "sensitivity_uplift": sensitivity_uplift,
                    "threshold_specificity": threshold_specificity,
                    "vulnerability_specificity": vulnerability_specificity,
                    "threshold_false_positive_rate": threshold_false_positive_rate,
                    "vulnerability_false_positive_rate": vulnerability_false_positive_rate,
                    "meta_model_f1": meta_f1,
                    "f1_drop_vs_meta_model": f1_drop_vs_meta,
                },
                "source_runs": {
                    "threshold_run_name": str(row.get("threshold_run_name", "")),
                    "vulnerability_run_name": str(row.get("vulnerability_run_name", "")),
                    "meta_model_run_name": str(meta_row.get("run_name", "")) if meta_row is not None else None,
                },
                "checks": checks,
                "passes_contract": passes_contract,
            }
        )

    overall = {
        "datasets_evaluated": int(len(dataset_verdicts)),
        "datasets_passing": int(sum(1 for item in dataset_verdicts if item["passes_contract"])),
        "passes_contract": bool(dataset_verdicts) and all(item["passes_contract"] for item in dataset_verdicts),
    }

    return {
        "research_question": RESEARCH_QUESTION,
        "primary_comparison": "threshold_baseline vs vulnerability_pipeline",
        "positive_definition": POSITIVE_DEFINITION,
        "guardrails": EVALUATION_GUARDRAILS,
        "dataset_verdicts": dataset_verdicts,
        "overall": overall,
    }


def _format_for_markdown(df: pd.DataFrame, float_columns: list[str]) -> pd.DataFrame:
    printable = df.copy()
    for col in float_columns:
        if col in printable.columns:
            printable[col] = printable[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    return printable


def _to_markdown_text(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def build_evaluation_contract_markdown(contract: dict[str, Any]) -> str:
    lines = [
        "# Evaluation Contract",
        "",
        f"Research question: {contract['research_question']}",
        "",
        "## Primary Comparison",
        "",
        f"- `{contract['primary_comparison']}`",
        "- Positive alert for the vulnerability pipeline: `vulnerability_level in {medium, high}`.",
        "- Positive ground truth: `true_label == fall`.",
        "",
        "## Quantitative Guardrails",
        "",
        f"- Absolute F1 gain over threshold baseline must be at least `{contract['guardrails']['min_absolute_f1_gain']:.2f}` on each dataset.",
        "- Vulnerability sensitivity must not be worse than threshold sensitivity.",
        f"- Vulnerability specificity must stay at or above `{contract['guardrails']['min_vulnerability_specificity']:.2f}`.",
        f"- Vulnerability F1 must remain within `{contract['guardrails']['max_f1_drop_vs_meta_model']:.2f}` of the fall meta-model F1.",
        "",
        "## Dataset Verdicts",
        "",
    ]

    verdict_rows: list[dict[str, Any]] = []
    for item in contract["dataset_verdicts"]:
        metrics = item["metrics"]
        verdict_rows.append(
            {
                "dataset": item["dataset"],
                "pass": item["passes_contract"],
                "threshold_f1": metrics.get("threshold_f1"),
                "vulnerability_f1": metrics.get("vulnerability_f1"),
                "absolute_f1_gain": metrics.get("absolute_f1_gain"),
                "sensitivity_uplift": metrics.get("sensitivity_uplift"),
                "vulnerability_specificity": metrics.get("vulnerability_specificity"),
                "f1_drop_vs_meta_model": metrics.get("f1_drop_vs_meta_model"),
            }
        )

    verdict_df = pd.DataFrame(verdict_rows)
    if verdict_df.empty:
        lines.append("No dataset verdicts were generated.")
    else:
        verdict_df = _format_for_markdown(
            verdict_df,
            [
                "threshold_f1",
                "vulnerability_f1",
                "absolute_f1_gain",
                "sensitivity_uplift",
                "vulnerability_specificity",
                "f1_drop_vs_meta_model",
            ],
        )
        lines.append(_to_markdown_text(verdict_df))

    lines.extend(
        [
            "",
            "## Overall Verdict",
            "",
            f"- Datasets evaluated: `{contract['overall']['datasets_evaluated']}`",
            f"- Datasets passing: `{contract['overall']['datasets_passing']}`",
            f"- Contract pass: `{contract['overall']['passes_contract']}`",
            "",
        ]
    )
    return "\n".join(lines)


def save_results_artifacts(df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "results_comparison.csv"
    json_path = output_dir / "results_comparison.json"
    md_path = output_dir / "results_comparison.md"
    primary_csv_path = output_dir / "primary_comparison.csv"
    primary_json_path = output_dir / "primary_comparison.json"
    primary_md_path = output_dir / "primary_comparison.md"
    contract_json_path = output_dir / "evaluation_contract.json"
    contract_md_path = output_dir / "evaluation_contract.md"

    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    primary_df = build_primary_comparison_dataframe(df)
    primary_df.to_csv(primary_csv_path, index=False)
    primary_json_path.write_text(primary_df.to_json(orient="records", indent=2), encoding="utf-8")
    contract = build_evaluation_contract(df)
    contract_json_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    display_columns = [
        "dataset",
        "stage",
        "profile",
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "roc_auc",
        "average_precision",
        "brier_score",
        "run_name",
    ]
    printable = df[display_columns].copy() if not df.empty else df.copy()
    printable = _format_for_markdown(
        printable,
        ["accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc", "average_precision", "brier_score"],
    )

    md_sections: list[str] = [
        "# Primary Comparison",
        "",
        "This report treats threshold-only fall detection versus fused vulnerability assessment as the main dissertation comparison.",
        "",
        _to_markdown_text(
            _format_for_markdown(
                primary_df,
                [
                    "threshold_f1",
                    "vulnerability_f1",
                    "absolute_f1_gain",
                    "relative_f1_gain_pct",
                    "threshold_sensitivity",
                    "vulnerability_sensitivity",
                    "threshold_specificity",
                    "vulnerability_specificity",
                ],
            )
        ) if not primary_df.empty else "No threshold/vulnerability pairs were found.",
        "",
        "# Supporting Stages",
        "",
        "Intermediate fall meta-model and fall-event rows are included below as supporting evidence for the primary comparison.",
        "",
        _to_markdown_text(printable),
        "",
    ]

    md_path.write_text("\n".join(md_sections), encoding="utf-8")
    primary_md_path.write_text(_to_markdown_text(_format_for_markdown(
        primary_df,
        [
            "threshold_f1",
            "vulnerability_f1",
            "absolute_f1_gain",
            "relative_f1_gain_pct",
            "threshold_sensitivity",
            "vulnerability_sensitivity",
            "threshold_specificity",
            "vulnerability_specificity",
        ],
    )) + "\n", encoding="utf-8")
    contract_md_path.write_text(build_evaluation_contract_markdown(contract) + "\n", encoding="utf-8")

    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "markdown": str(md_path),
        "primary_csv": str(primary_csv_path),
        "primary_json": str(primary_json_path),
        "primary_markdown": str(primary_md_path),
        "contract_json": str(contract_json_path),
        "contract_markdown": str(contract_md_path),
    }


def _print_compact_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No result artifacts were found.")
        return

    display_columns = [
        "dataset",
        "stage",
        "profile",
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
    ]
    printable = df[display_columns].copy()
    for col in ["accuracy", "sensitivity", "specificity", "precision", "f1"]:
        printable[col] = printable[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")

    try:
        print(printable.to_markdown(index=False))
    except Exception:
        print(printable.to_string(index=False))


def _print_primary_summary(df: pd.DataFrame) -> None:
    primary_df = build_primary_comparison_dataframe(df)
    if primary_df.empty:
        print("No primary threshold-vs-vulnerability comparison rows were found.")
        return

    printable = _format_for_markdown(
        primary_df,
        [
            "threshold_f1",
            "vulnerability_f1",
            "absolute_f1_gain",
            "relative_f1_gain_pct",
            "threshold_sensitivity",
            "vulnerability_sensitivity",
            "threshold_specificity",
            "vulnerability_specificity",
        ],
    )
    print(_to_markdown_text(printable))


def _print_contract_summary(df: pd.DataFrame) -> None:
    contract = build_evaluation_contract(df)
    print(
        "Contract verdict: "
        f"datasets_passing={contract['overall']['datasets_passing']}/"
        f"{contract['overall']['datasets_evaluated']} "
        f"pass={contract['overall']['passes_contract']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the dissertation comparison tables. "
            "The primary artifact is threshold-only fall detection versus fused vulnerability assessment."
        )
    )
    parser.add_argument(
        "--results-root",
        default="results/runs",
        help="Root directory containing run artifacts",
    )
    parser.add_argument(
        "--output-dir",
        default="results/reports",
        help="Directory where the comparison table artifacts will be written",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    output_dir = Path(args.output_dir)

    rows = discover_latest_rows(results_root)
    df = build_results_dataframe(rows)
    artifacts = save_results_artifacts(df, output_dir=output_dir)

    print("Built results comparison table")
    _print_contract_summary(df)
    print("Primary threshold-vs-vulnerability comparison")
    _print_primary_summary(df)
    print("Supporting stage summary")
    _print_compact_summary(df)
    print(f"Saved CSV to: {artifacts['csv']}")
    print(f"Saved JSON to: {artifacts['json']}")
    print(f"Saved Markdown to: {artifacts['markdown']}")
    print(f"Saved primary CSV to: {artifacts['primary_csv']}")
    print(f"Saved primary JSON to: {artifacts['primary_json']}")
    print(f"Saved primary Markdown to: {artifacts['primary_markdown']}")
    print(f"Saved contract JSON to: {artifacts['contract_json']}")
    print(f"Saved contract Markdown to: {artifacts['contract_markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
