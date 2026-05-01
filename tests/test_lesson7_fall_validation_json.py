from pathlib import Path

from scripts.validate_lesson7_fall import (
    assemble_lesson7_validation_report,
    build_lesson7_methodology_checks,
)


def test_lesson7_validation_report_contains_required_fields(tmp_path: Path):
    dataset_path = tmp_path / "mobifall_root"
    dataset_path.mkdir(parents=True)

    grouped_rate = {
        "groups_checked": 3,
        "median_hz": 91.7,
        "min_hz": 81.9,
        "max_hz": 95.7,
        "estimated_rates_hz_preview": [95.7, 91.7, 81.9],
    }
    split = {
        "strategy": "group_shuffle_split_by_subject",
        "train_subject_groups": ["MOBIFALL::sub1", "MOBIFALL::sub2"],
        "test_subject_groups": ["MOBIFALL::sub3"],
        "train_rows": 100,
        "test_rows": 40,
    }
    evaluation = {
        "accuracy": 0.8,
        "sensitivity": 0.75,
        "specificity": 0.85,
        "precision": 0.78,
        "f1": 0.76,
        "support_total": 40,
        "per_class_support": {"fall": 20, "non_fall": 20},
        "per_class_precision": {"fall": 0.78, "non_fall": 0.82},
        "per_class_recall": {"fall": 0.75, "non_fall": 0.85},
        "confusion_matrix_available": True,
    }

    methodology = build_lesson7_methodology_checks(
        dataset_path=dataset_path,
        grouped_sampling_rate_summary=grouped_rate,
        split_summary=split,
    )

    report = assemble_lesson7_validation_report(
        repo_root=Path("/tmp/repo"),
        dataset="mobifall",
        dataset_path=dataset_path,
        pipeline_run={
            "executed": True,
            "passed": True,
            "return_code": 0,
            "stdout_preview": "ok",
            "stderr_preview": "",
        },
        grouped_sampling_rate_summary=grouped_rate,
        split_summary=split,
        evaluation_summary=evaluation,
        false_alarm_summary={
            "count": 5,
            "artifact_path": "/tmp/repo/results/runs/x/false_alarms.csv",
            "artifact_exists": True,
        },
        methodology_checks=methodology,
        artifacts={
            "run_dir": "/tmp/repo/results/runs/x",
            "metrics_json": "/tmp/repo/results/runs/x/metrics.json",
        },
        notes=[],
    )

    expected_top_keys = {
        "validation_name",
        "dataset",
        "status",
        "repo_root",
        "data_source",
        "pipeline_run",
        "grouped_sampling_rate_summary",
        "evaluation",
        "split",
        "false_alarms",
        "methodology_checks",
        "artifacts",
        "notes",
    }
    assert expected_top_keys.issubset(report.keys())
    assert report["validation_name"] == "lesson7_fall_validation"
    assert report["status"] == "ok"

    grouped = report["grouped_sampling_rate_summary"]
    assert grouped["groups_checked"] == 3
    assert grouped["median_hz"] == 91.7
    assert grouped["min_hz"] == 81.9
    assert grouped["max_hz"] == 95.7
    assert isinstance(grouped["estimated_rates_hz_preview"], list)

    checks = report["methodology_checks"]
    assert checks["grouped_rate_estimation_used"] is True
    assert checks["subject_aware_split_detected"] is True
    assert checks["real_data_used"] is True
