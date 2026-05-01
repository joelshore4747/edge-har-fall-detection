from pathlib import Path

from scripts.validate_lesson5_har import (
    assemble_validation_report,
    build_methodology_checks,
    summarize_evaluation_from_metrics,
)


def test_lesson5_validation_json_helpers_produce_expected_structure():
    metrics_payload = {
        "config": {"keep_unacceptable": False},
        "preprocessing_summary": {"windows_total": 12, "feature_rows": 10},
        "split": {
            "train_subject_groups": ["UCIHAR::1", "UCIHAR::2"],
            "test_subject_groups": ["UCIHAR::3"],
        },
        "heuristic": {
            "metrics": {
                "accuracy": 0.5,
                "macro_f1": 0.4,
                "per_class": {
                    "static": {"precision": 1.0, "recall": 0.5},
                    "locomotion": {"precision": 0.0, "recall": 0.0},
                },
                "confusion_matrix": [[1, 1], [0, 0]],
            }
        },
        "random_forest": {
            "metrics": {
                "accuracy": 0.75,
                "macro_f1": 0.7,
                "per_class": {
                    "static": {"precision": 1.0, "recall": 1.0},
                    "locomotion": {"precision": 0.5, "recall": 0.5},
                },
                "confusion_matrix": [[1, 0], [1, 1]],
            }
        },
    }
    artifacts_found = {
        "confusion_matrix_heuristic_csv": True,
        "confusion_matrix_random_forest_csv": True,
    }
    feature_table_summary = {
        "available": True,
        "rows": 10,
        "columns": 50,
        "feature_column_count": 36,
        "metadata_columns": ["window_id", "subject_id", "label_mapped_majority"],
        "label_column": "label_mapped_majority",
        "label_counts": {"static": 5, "locomotion": 5},
        "subjects_count": 3,
        "sessions_count": 3,
        "datasets_present": ["UCIHAR"],
        "notes": [],
    }

    evaluation_summary = summarize_evaluation_from_metrics(metrics_payload, artifacts_found=artifacts_found)
    methodology_checks = build_methodology_checks(
        metrics_payload=metrics_payload,
        evaluation_summary=evaluation_summary,
        feature_table_summary=feature_table_summary,
    )
    report = assemble_validation_report(
        repo_root=Path("/tmp/repo"),
        tests_summary={
            "executed": True,
            "passed": True,
            "pytest_targets": [],
            "return_code": 0,
            "stdout_preview": "....",
            "stderr_preview": "",
        },
        feature_table_summary=feature_table_summary,
        pipeline_run_summary={
            "executed": True,
            "passed": True,
            "return_code": 0,
            "stdout_preview": "...",
            "stderr_preview": "",
            "artifacts_found": {"run_dir": "/tmp/repo/results/runs/x", "files": artifacts_found},
        },
        evaluation_summary=evaluation_summary,
        methodology_checks=methodology_checks,
        notes=[],
    )

    expected_top_keys = {
        "validation_name",
        "status",
        "repo_root",
        "tests",
        "feature_table",
        "pipeline_run",
        "evaluation",
        "methodology_checks",
        "notes",
    }
    assert expected_top_keys.issubset(report.keys())
    assert report["validation_name"] == "lesson5_har_validation"
    assert report["status"] == "ok"
    assert report["evaluation"]["heuristic_baseline"]["available"] is True
    assert report["evaluation"]["rf_baseline"]["available"] is True
    assert report["evaluation"]["confusion_matrix_available"] is True
    assert report["methodology_checks"]["subject_aware_split_detected"] is True
    assert report["methodology_checks"]["macro_f1_reported"] is True
