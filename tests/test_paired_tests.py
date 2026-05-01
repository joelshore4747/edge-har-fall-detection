from scripts.lib.paired_tests import mcnemar_test


def test_mcnemar_counts_paired_correctness():
    y_true = ["fall", "fall", "non_fall", "non_fall"]
    pred_a = ["fall", "non_fall", "non_fall", "fall"]
    pred_b = ["fall", "fall", "non_fall", "non_fall"]

    result = mcnemar_test(y_true, pred_a, pred_b)

    assert result.n == 4
    assert result.both_correct == 2
    assert result.a_correct_b_wrong == 0
    assert result.a_wrong_b_correct == 2
    assert result.both_wrong == 0
    assert result.discordant == 2
    assert result.p_value == 0.5


def test_mcnemar_handles_no_discordant_pairs():
    result = mcnemar_test([True, False], [True, True], [True, True])

    assert result.discordant == 0
    assert result.statistic == 0.0
    assert result.p_value == 1.0


def test_mcnemar_rejects_unaligned_inputs():
    try:
        mcnemar_test([True], [True, False], [True])
    except ValueError as exc:
        assert "must align" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
