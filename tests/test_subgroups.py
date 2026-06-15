"""Tests for T3.3 subgroup evaluation utilities."""

import pandas as pd
import pytest

from binary_classifier.evaluation.subgroups import subgroup_report


def test_subgroup_report_metrics_and_suppression() -> None:
    df = pd.DataFrame(
        {
            "EIN2": ["00-1", "00-2", "00-3", "00-4", "00-5"],
            "ntee_major": ["X", "X", "Y", "Y", "Z"],
            "text": [
                "one",
                "one two",
                "one two three",
                "one two three four five",
                "one two three four five six",
            ],
        }
    )

    rows = subgroup_report(
        df,
        y_true=[1, 0, 1, 0, 1],
        y_pred=[1, 1, 0, 0, 1],
        y_prob=[0.9, 0.8, 0.4, 0.2, 0.7],
        by="ntee_major",
        length_bins=[2, 4],
        min_n=2,
    )
    by_key = {(row["grouping"], row["value"]): row for row in rows}

    assert by_key[("ntee_major", "X")] == {
        "grouping": "ntee_major",
        "value": "X",
        "n": 2,
        "suppressed": False,
        "minority_f1": pytest.approx(2 / 3),
        "fpr": pytest.approx(1.0),
        "fnr": pytest.approx(0.0),
    }
    assert by_key[("ntee_major", "Y")]["minority_f1"] == pytest.approx(0.0)
    assert by_key[("ntee_major", "Y")]["fpr"] == pytest.approx(0.0)
    assert by_key[("ntee_major", "Y")]["fnr"] == pytest.approx(1.0)
    assert by_key[("ntee_major", "Z")] == {
        "grouping": "ntee_major",
        "value": "Z",
        "n": 1,
        "suppressed": True,
        "minority_f1": None,
        "fpr": None,
        "fnr": None,
    }

    assert by_key[("word_count_bin", "0-2")]["n"] == 2
    assert by_key[("word_count_bin", "0-2")]["minority_f1"] == pytest.approx(
        2 / 3
    )
    assert by_key[("word_count_bin", "3-4")]["suppressed"] is True
    assert by_key[("word_count_bin", "5+")]["minority_f1"] == pytest.approx(1.0)


def test_subgroup_report_validates_grouping_column() -> None:
    df = pd.DataFrame({"EIN2": ["00-1"], "text": ["one"]})

    with pytest.raises(KeyError, match="Grouping column not found"):
        subgroup_report(
            df,
            y_true=[1],
            y_pred=[1],
            y_prob=[0.9],
            by="ntee_major",
            length_bins=[2],
            min_n=1,
        )
