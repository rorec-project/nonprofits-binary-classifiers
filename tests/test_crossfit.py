"""Tests for out-of-fold cross-fit prediction probabilities."""

import numpy as np
import pandas as pd

from binary_classifier.paths import PathRegistry
from binary_classifier.train.crossfit import compute_oof_pred_probs


class _DeterministicPredictor:
    """Predictor stub returning deterministic probabilities by EIN2."""

    def predict_proba(self, eval_df: pd.DataFrame) -> np.ndarray:
        scores = np.array(
            [0.1 + (int(str(ein2)[-2:]) % 8) / 10 for ein2 in eval_df["EIN2"]],
            dtype=float,
        )
        return np.column_stack([1.0 - scores, scores])


class _RecordingFinetune:
    """Fine-tune stub recording fold partitions and returning a predictor."""

    def __init__(self) -> None:
        self.calls = []

    def __call__(
        self,
        cfg,
        registry,
        encoder,
        train_df: pd.DataFrame,
        dev_df: pd.DataFrame,
        **kwargs,
    ):
        del cfg, registry, encoder
        self.calls.append(
            {
                "train_ein2": train_df["EIN2"].tolist(),
                "heldout_ein2": dev_df["EIN2"].tolist(),
                "seed": kwargs["seed"],
                "targets": kwargs["targets"],
                "arm": kwargs["arm"],
            },
        )
        return {"predictor": _DeterministicPredictor()}


def test_oof_partition_property_fold_ids_and_schema(tiny_config, tiny_registry) -> None:
    """Every EIN2 appears once, with valid fold ids and artifact schema."""
    tiny_config.SEED = 11
    tiny_config.training.crossfit_folds = 3
    frame = _training_frame(12)
    finetune = _RecordingFinetune()

    path = compute_oof_pred_probs(
        tiny_config,
        tiny_registry,
        frame,
        finetune_fn=finetune,
    )
    out = pd.read_parquet(path)

    assert list(out.columns) == ["EIN2", "fold", "p0", "p1"]
    assert len(out) == len(frame)
    assert out["EIN2"].nunique() == len(frame)
    assert set(out["EIN2"]) == set(frame["EIN2"])
    assert set(out["fold"]) == {0, 1, 2}
    assert out.groupby("fold").size().to_dict() == {0: 4, 1: 4, 2: 4}
    assert np.allclose(out["p0"] + out["p1"], 1.0)
    assert out["p0"].between(0, 1).all()
    assert out["p1"].between(0, 1).all()

    assert len(finetune.calls) == 3
    for call in finetune.calls:
        assert len(call["train_ein2"]) == 8
        assert len(call["heldout_ein2"]) == 4
        assert set(call["train_ein2"]).isdisjoint(call["heldout_ein2"])
        assert call["targets"] == tiny_config.training.targets
        assert call["arm"] == "default"


def test_oof_crossfit_is_deterministic(tiny_config, tmp_path) -> None:
    """Same config and frame produce identical partitions and probabilities."""
    tiny_config.SEED = 23
    tiny_config.training.crossfit_folds = 4
    frame = _training_frame(16)
    registry_a = _registry(tiny_config, tmp_path / "a")
    registry_b = _registry(tiny_config, tmp_path / "b")
    finetune_a = _RecordingFinetune()
    finetune_b = _RecordingFinetune()

    out_a = pd.read_parquet(
        compute_oof_pred_probs(
            tiny_config,
            registry_a,
            frame,
            finetune_fn=finetune_a,
        ),
    )
    out_b = pd.read_parquet(
        compute_oof_pred_probs(
            tiny_config,
            registry_b,
            frame,
            finetune_fn=finetune_b,
        ),
    )

    pd.testing.assert_frame_equal(out_a, out_b)
    assert finetune_a.calls == finetune_b.calls


def test_oof_crossfit_resumes_from_existing_fold_shards(
    tiny_config,
    tiny_registry,
) -> None:
    """Existing valid fold shards are reused without calling fine-tune again."""
    tiny_config.SEED = 31
    tiny_config.training.crossfit_folds = 3
    frame = _training_frame(12)
    finetune = _RecordingFinetune()
    first_path = compute_oof_pred_probs(
        tiny_config,
        tiny_registry,
        frame,
        finetune_fn=finetune,
    )
    first = pd.read_parquet(first_path)
    first_path.unlink()

    def fail_finetune(*args, **kwargs):
        del args, kwargs
        raise AssertionError("resume should skip completed fold shards")

    resumed_path = compute_oof_pred_probs(
        tiny_config,
        tiny_registry,
        frame,
        finetune_fn=fail_finetune,
    )
    resumed = pd.read_parquet(resumed_path)

    assert len(finetune.calls) == 3
    pd.testing.assert_frame_equal(first, resumed)


def test_oof_schema_accepts_callable_p1_predictor(tiny_config, tiny_registry) -> None:
    """Callable predictors may return one-dimensional positive probabilities."""
    tiny_config.SEED = 41
    tiny_config.training.crossfit_folds = 2
    frame = _training_frame(8)

    def finetune_stub(*args, **kwargs):
        del args, kwargs

        def predictor(eval_df: pd.DataFrame) -> np.ndarray:
            return np.full(len(eval_df), 0.75, dtype=float)

        return predictor

    out = pd.read_parquet(
        compute_oof_pred_probs(
            tiny_config,
            tiny_registry,
            frame,
            finetune_fn=finetune_stub,
        ),
    )

    assert list(out.columns) == ["EIN2", "fold", "p0", "p1"]
    assert out["p0"].tolist() == [0.25] * len(frame)
    assert out["p1"].tolist() == [0.75] * len(frame)


def _training_frame(n_rows: int) -> pd.DataFrame:
    """Build a balanced training frame for cross-fit tests."""
    rows = []
    for i in range(n_rows):
        label = i % 2
        rows.append(
            {
                "EIN2": f"E{i:03d}",
                "text": f"mission text {i}",
                "ntee_major_group": "X",
                "p_pos": float(label),
                "hard_label": label,
            },
        )
    return pd.DataFrame(rows)


def _registry(tiny_config, root) -> PathRegistry:
    """Build an isolated registry for determinism checks."""
    registry = PathRegistry.from_config(tiny_config, root=root)
    registry.ensure_dirs()
    return registry
