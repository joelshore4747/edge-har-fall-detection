"""Run paired significance tests for reported threshold-vs-vulnerability claims."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lib.paired_tests import binary_from_labels, mcnemar_chi_square_p_value, mcnemar_test


DEFAULT_RUNS = {
    "MOBIFALL": REPO_ROOT / "results" / "runs" / "vulnerability_eval_mobifall__v2" / "vulnerability_eval_predictions.csv",
    "SISFALL": REPO_ROOT / "results" / "runs" / "vulnerability_eval_sisfall__v3" / "vulnerability_eval_predictions.csv",
}
VULNERABILITY_POSITIVE_LEVELS = {"medium", "high"}


def _prediction_from_vulnerability_level(values: pd.Series) -> list[bool]:
    return [str(value).strip().lower() in VULNERABILITY_POSITIVE_LEVELS for value in values]


def _f1(y_true: list[bool], y_pred: list[bool]) -> float:
    tp = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth and pred)
    fp = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if not truth and pred)
    fn = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth and not pred)
    denom = (2 * tp) + fp + fn
    return float((2 * tp) / denom) if denom else 0.0


def _accuracy(y_true: list[bool], y_pred: list[bool]) -> float:
    if not y_true:
        return float("nan")
    return sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth == pred) / len(y_true)


def run_for_file(dataset: str, path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    required = {"true_label", "threshold_predicted_label", "vulnerability_level"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    y_true = binary_from_labels(frame["true_label"]).tolist()
    threshold_pred = binary_from_labels(frame["threshold_predicted_label"]).tolist()
    vulnerability_pred = _prediction_from_vulnerability_level(frame["vulnerability_level"])
    result = mcnemar_test(y_true, threshold_pred, vulnerability_pred)

    return {
        "dataset": dataset,
        "source": str(path.relative_to(REPO_ROOT)),
        "n": result.n,
        "positive_rule": "vulnerability_level in {'medium', 'high'}",
        "threshold_f1": _f1(y_true, threshold_pred),
        "vulnerability_f1": _f1(y_true, vulnerability_pred),
        "threshold_accuracy": _accuracy(y_true, threshold_pred),
        "vulnerability_accuracy": _accuracy(y_true, vulnerability_pred),
        "mcnemar": {
            "statistic_continuity_corrected": result.statistic,
            "exact_p_value": result.p_value,
            "chi_square_p_value": mcnemar_chi_square_p_value(result),
            "both_correct": result.both_correct,
            "threshold_correct_vulnerability_wrong": result.a_correct_b_wrong,
            "threshold_wrong_vulnerability_correct": result.a_wrong_b_correct,
            "both_wrong": result.both_wrong,
            "discordant_pairs": result.discordant,
        },
    }


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Paired Significance Tests",
        "",
        "This report applies an exact McNemar test to paired window-level predictions.",
        "The threshold baseline and the vulnerability pipeline are compared on the same rows;",
        "the test asks whether their correctness differs symmetrically across discordant pairs.",
        "",
        "| Dataset | n | Threshold F1 | Vulnerability F1 | Threshold correct / vulnerability wrong | Threshold wrong / vulnerability correct | Exact p-value | Source |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        test = row["mcnemar"]
        lines.append(
            "| {dataset} | {n} | {threshold_f1:.3f} | {vulnerability_f1:.3f} | {a} | {b} | {p:.3g} | `{source}` |".format(
                dataset=row["dataset"],
                n=row["n"],
                threshold_f1=row["threshold_f1"],
                vulnerability_f1=row["vulnerability_f1"],
                a=test["threshold_correct_vulnerability_wrong"],
                b=test["threshold_wrong_vulnerability_correct"],
                p=test["exact_p_value"],
                source=row["source"],
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: a small p-value supports the claim that the vulnerability pipeline changes the paired error pattern rather than merely moving an unpaired aggregate metric.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "reports")
    args = parser.parse_args()

    rows = [run_for_file(dataset, path) for dataset, path in DEFAULT_RUNS.items()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "paired_significance.json"
    md_path = args.out_dir / "paired_significance.md"
    json_path.write_text(json.dumps({"comparisons": rows}, indent=2), encoding="utf-8")
    _write_markdown(rows, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
