"""Tests for gated training arms."""

import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues

from binary_classifier.train import arms


def test_pruned_arm_drops_cleanlab_issues_only_in_disagreement_band() -> None:
    """Pruning intersects cleanlab flags with the low-vote-share band."""
    frame = _training_frame(100)
    noisy_indices = [7, 18, 41, 72]
    band_indices = [7, 18, 10, 11]
    frame.loc[band_indices, "p_pos"] = 0.50
    oof = _oof_probs(frame, noisy_indices=noisy_indices)

    cleanlab_mask = find_label_issues(
        labels=frame["hard_label"].to_numpy(dtype=int),
        pred_probs=oof[["p0", "p1"]].to_numpy(dtype=float),
    )
    expected_drop = set(
        frame.loc[
            np.asarray(cleanlab_mask, dtype=bool)
            & frame["p_pos"].between(0.34, 0.66, inclusive="neither"),
            "EIN2",
        ],
    )
    assert {"E007", "E018"}.issubset(expected_drop)

    pruned, spec = arms.pruned_arm(frame, oof.sample(frac=1.0, random_state=4))

    assert spec.targets == "soft"
    assert spec.arm == "pruned"
    assert spec.class_weights is None
    assert set(frame["EIN2"]) - set(pruned["EIN2"]) == expected_drop
    assert {"E041", "E072"}.issubset(set(pruned["EIN2"]))
    assert {"E010", "E011"}.issubset(set(pruned["EIN2"]))


def test_class_weighted_arm_computes_inverse_frequency_weights() -> None:
    """Weights are n / (2 * class_count), ordered as (w0, w1)."""
    frame = _training_frame(100)
    frame["hard_label"] = [0] * 80 + [1] * 20
    frame["p_pos"] = frame["hard_label"].astype(float)

    weighted, spec = arms.class_weighted_arm(frame)

    assert len(weighted) == len(frame)
    assert spec.targets == "soft"
    assert spec.arm == "class_weighted"
    assert spec.class_weights == (100 / (2 * 80), 100 / (2 * 20))
    assert arms.class_weights(frame) == spec.class_weights


def test_each_arm_returns_training_schema_and_loss_spec() -> None:
    """Arm runners preserve EIN2-bearing training rows and expose loss specs."""
    frame = _training_frame(12)
    oof = _oof_probs(frame, noisy_indices=[])

    hard_df, hard_spec = arms.run_arm("hard", frame)
    pruned_df, pruned_spec = arms.run_arm("pruned", frame, oof_probs=oof)
    weighted_df, weighted_spec = arms.run_arm("class_weighted", frame)

    for out in [hard_df, pruned_df, weighted_df]:
        assert {"EIN2", "text", "ntee_major_group", "p_pos", "hard_label"}.issubset(
            out.columns,
        )
        assert out["EIN2"].is_unique
        assert out["p_pos"].between(0, 1).all()
        assert set(out["hard_label"]).issubset({0, 1})

    assert set(hard_df["p_pos"]) == {0.0, 1.0}
    assert hard_spec.finetune_kwargs() == {"targets": "hard", "arm": "hard"}
    assert pruned_spec.finetune_kwargs() == {"targets": "soft", "arm": "pruned"}
    assert weighted_spec.finetune_kwargs() == {
        "targets": "soft",
        "arm": "class_weighted",
    }
    assert weighted_spec.class_weights == (1.0, 1.0)


def _training_frame(n_rows: int) -> pd.DataFrame:
    """Build a balanced synthetic training frame."""
    rows = []
    for i in range(n_rows):
        label = i % 2
        rows.append(
            {
                "EIN2": f"E{i:03d}",
                "text": f"mission text {i}",
                "ntee_major_group": "X",
                "p_pos": 0.9 if label else 0.1,
                "hard_label": label,
            },
        )
    return pd.DataFrame(rows)


def _oof_probs(frame: pd.DataFrame, *, noisy_indices: list[int]) -> pd.DataFrame:
    """Build OOF probabilities with selected rows confidently wrong."""
    noisy = set(noisy_indices)
    rows = []
    for i, row in frame.reset_index(drop=True).iterrows():
        label = int(row["hard_label"])
        if i in noisy:
            p1 = 0.01 if label else 0.99
        else:
            p1 = 0.99 if label else 0.01
        rows.append(
            {
                "EIN2": row["EIN2"],
                "fold": i % 5,
                "p0": 1.0 - p1,
                "p1": p1,
            },
        )
    return pd.DataFrame(rows)
