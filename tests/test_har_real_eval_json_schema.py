from scripts.run_har_real_evaluation import build_comparison_summary


def test_har_real_comparison_json_schema_has_expected_keys():
    uci_summary = {
        "dataset": "UCIHAR",
        "status": "ok",
        "evaluation": {
            "heuristic_baseline": {"accuracy": 0.60, "macro_f1": 0.55},
            "rf_baseline": {"accuracy": 0.85, "macro_f1": 0.82},
        },
    }
    pamap2_summary = {
        "dataset": "PAMAP2",
        "status": "ok",
        "evaluation": {
            "heuristic_baseline": {"accuracy": 0.35, "macro_f1": 0.28},
            "rf_baseline": {"accuracy": 0.62, "macro_f1": 0.54},
        },
    }

    comparison = build_comparison_summary([uci_summary, pamap2_summary])

    expected_keys = {
        "comparison_name",
        "status",
        "datasets",
        "metrics",
        "methodology_notes",
        "interpretation_notes",
    }
    assert expected_keys.issubset(comparison.keys())
    assert comparison["comparison_name"] == "lesson5_har_real_comparison"
    assert comparison["status"] == "ok"
    assert comparison["datasets"] == ["UCIHAR", "PAMAP2"]
    assert set(comparison["metrics"].keys()) == {"UCIHAR", "PAMAP2"}
    assert comparison["metrics"]["UCIHAR"]["rf_macro_f1"] == 0.82
    assert comparison["metrics"]["PAMAP2"]["heuristic_accuracy"] == 0.35
    assert any("pre-windowed" in note.lower() or "prewindowed" in note.lower() for note in comparison["methodology_notes"])
