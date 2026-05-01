from __future__ import annotations

from scripts.build_results_table import (
    ResultRow,
    build_evaluation_contract,
    build_primary_comparison_dataframe,
    build_results_dataframe,
)


def test_build_results_dataframe_prioritizes_primary_comparison_order() -> None:
    rows = [
        ResultRow(
            dataset="MOBIFALL",
            stage="fall_meta_model",
            profile="balanced",
            run_name="meta",
            source_json="meta.json",
            accuracy=0.93,
            sensitivity=0.88,
            specificity=0.95,
            precision=0.83,
            f1=0.86,
        ),
        ResultRow(
            dataset="MOBIFALL",
            stage="vulnerability_pipeline",
            profile="balanced",
            run_name="vuln",
            source_json="vuln.json",
            accuracy=0.94,
            sensitivity=0.90,
            specificity=0.95,
            precision=0.82,
            f1=0.87,
        ),
        ResultRow(
            dataset="MOBIFALL",
            stage="threshold_baseline",
            profile="default",
            run_name="threshold",
            source_json="threshold.json",
            accuracy=0.63,
            sensitivity=0.34,
            specificity=0.71,
            precision=0.23,
            f1=0.28,
        ),
        ResultRow(
            dataset="MOBIFALL",
            stage="fall_event_pipeline",
            profile="balanced",
            run_name="event",
            source_json="event.json",
            accuracy=0.94,
            sensitivity=0.89,
            specificity=0.95,
            precision=0.83,
            f1=0.86,
        ),
    ]

    df = build_results_dataframe(rows)

    assert df["stage"].tolist() == [
        "threshold_baseline",
        "vulnerability_pipeline",
        "fall_meta_model",
        "fall_event_pipeline",
    ]


def test_build_primary_comparison_dataframe_computes_threshold_vs_vulnerability_gain() -> None:
    rows = [
        ResultRow(
            dataset="MOBIFALL",
            stage="threshold_baseline",
            profile="default",
            run_name="threshold_run",
            source_json="threshold.json",
            accuracy=0.63,
            sensitivity=0.3351,
            specificity=0.7094,
            precision=0.2345,
            f1=0.2759,
        ),
        ResultRow(
            dataset="MOBIFALL",
            stage="vulnerability_pipeline",
            profile="balanced",
            run_name="vulnerability_run",
            source_json="vuln.json",
            accuracy=0.9370,
            sensitivity=0.8976,
            specificity=0.9474,
            precision=0.8193,
            f1=0.8567,
        ),
    ]

    df = build_results_dataframe(rows)
    primary_df = build_primary_comparison_dataframe(df)

    assert primary_df["dataset"].tolist() == ["MOBIFALL"]
    row = primary_df.iloc[0]
    assert row["threshold_run_name"] == "threshold_run"
    assert row["vulnerability_run_name"] == "vulnerability_run"
    assert row["absolute_f1_gain"] == row["vulnerability_f1"] - row["threshold_f1"]
    assert row["relative_f1_gain_pct"] > 0.0


def test_build_evaluation_contract_marks_dataset_as_passing_when_guardrails_hold() -> None:
    rows = [
        ResultRow(
            dataset="MOBIFALL",
            stage="threshold_baseline",
            profile="default",
            run_name="threshold_run",
            source_json="threshold.json",
            accuracy=0.63,
            sensitivity=0.3351,
            specificity=0.7094,
            precision=0.2345,
            f1=0.2759,
        ),
        ResultRow(
            dataset="MOBIFALL",
            stage="fall_meta_model",
            profile="logistic_regression",
            run_name="meta_run",
            source_json="meta.json",
            accuracy=0.9384,
            sensitivity=0.8854,
            specificity=0.9525,
            precision=0.8320,
            f1=0.8579,
        ),
        ResultRow(
            dataset="MOBIFALL",
            stage="vulnerability_pipeline",
            profile="balanced",
            run_name="vulnerability_run",
            source_json="vuln.json",
            accuracy=0.9370,
            sensitivity=0.8976,
            specificity=0.9474,
            precision=0.8193,
            f1=0.8567,
        ),
    ]

    df = build_results_dataframe(rows)
    contract = build_evaluation_contract(df)

    assert contract["overall"]["passes_contract"] is True
    assert contract["overall"]["datasets_evaluated"] == 1
    assert contract["overall"]["datasets_passing"] == 1

    verdict = contract["dataset_verdicts"][0]
    assert verdict["dataset"] == "MOBIFALL"
    assert verdict["passes_contract"] is True
    assert verdict["checks"]["absolute_f1_gain"] is True
    assert verdict["checks"]["sensitivity_not_worse_than_threshold"] is True
    assert verdict["checks"]["specificity_floor"] is True
    assert verdict["checks"]["retains_meta_model_signal"] is True
