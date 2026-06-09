"""Typed configuration loader for the binary classifier package.

Reads YAML config files into a validated Pydantic model. All downstream
stages (data loading, annotation, training, evaluation) consume the same
config object so that paths, seeds, thresholds, and model slates are defined
in a single source of truth.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

# ── Sub-configs ──────────────────────────────────────────────────────────────


class PathsConfig(BaseModel):
    """Input/output directory paths.

    Attributes:
        upstream_repo: Path to the upstream NonProfitData repository.
        missions_parquet: Name of the missions cross-section parquet file.
        bmf_parquet: Name of the BMF unified processed parquet file.
        output_dir: Directory for intermediate and final artifacts.
        data_dir: Sub-directory for labelled CSVs and raw outputs.
        train_test_dir: Sub-directory for train/test splits.
        results_dir: Sub-directory for evaluation metrics and plots.
        models_dir: Sub-directory for persisted fine-tuned models.

    """

    upstream_repo: Path = Field(default=Path("../NonProfitData"))
    missions_parquet: str = (
        "data/processed/corpus/missions/missions_cross_section.parquet"
    )
    bmf_parquet: str = "data/processed/panel/core/bmf_unified_processed.parquet"
    output_dir: Path = Field(default=Path("."))
    data_dir: Path = Field(default=Path("data"))
    train_test_dir: Path = Field(default=Path("train_test_datasets"))
    results_dir: Path = Field(default=Path("results"))
    models_dir: Path = Field(default=Path("models"))


class ModelSlateConfig(BaseModel):
    """Models available for the annotation bake-off and production runs.

    Attributes:
        closed_reference: A snapshot model from a closed API (e.g. OpenAI).
        open_production: The primary open-weight model served via vLLM.
        open_lightweight: A smaller open-weight control model.

    """

    closed_reference: str = "gpt-4o-mini"
    open_production: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    open_lightweight: str = "google/gemma-4-31B-it"


class QThresholdsConfig(BaseModel):
    """Quality-score thresholds for the computable rubric Q.

    Tiers are defined in the annex
    (see ``.agents/plans/we-work-on-the-floofy-wreath-annex.md``):
    HIGH missions are concrete (purpose + beneficiary + activity), MEDIUM are
    decent but thinner on one dimension, and LOW are fragments or bare labels
    that are handled by the rule layer at inference.
    """

    HIGH: float = 5.0
    MEDIUM: float = 3.0
    LOW: float = 0.0


class SampleSizesConfig(BaseModel):
    """Target sizes for the silver pool, gold set, and prompt-dev split.

    The silver pool is over-provisioned so that the learning-curve sweep in
    point 3 can empirically find the optimal training N. The gold set is kept
    small (~400) because it is hand-coded by a human engineer; the frozen test
    is reported with bootstrap confidence intervals.
    """

    silver: int = 20_000
    gold: int = 400
    prompt_dev: int = 50


class AnnotationConfig(BaseModel):
    """Hyperparameters for the LLM-as-primary labeler.

    All temperature and seed values are fixed to ensure reproducible
    annotations. Resume is keyed by (EIN2, source_id) rather than row count
    to avoid the audit R-08 idempotency gap.
    """

    temperature: float = 0.0
    max_retries: int = 5
    checkpoint_every: int = 100
    guided_json: bool = True
    seed: int = 42


class TrainingConfig(BaseModel):
    """Stubs for the fine-tuning stage (point 3 in the roadmap).

    These values are placeholders; the actual hyperparameters will be chosen
    empirically via the learning-curve sweep and best-model selection on the
    human validation set.
    """

    learning_rate: float = 5e-5
    batch_size: int = 16
    epochs: int = 10
    weight_decay: float = 0.01
    metric_for_best_model: str = "pr_auc"
    greater_is_better: bool = True
    early_stopping_patience: int = 4
    save_total_limit: int = 2
    fp16: bool = True


class BinaryClassifierConfig(BaseModel):
    """Root configuration object for a binary classification task.

    Attributes:
        SEED: Global random seed for all stochastic steps (sampling, splitting,
            training, annotation temperature).
        entity: Name of the data entity being classified (e.g. "missions",
            "activities").
        field: Column in the upstream parquet that contains the text to
            classify (e.g. "LONGEST_MISSION").
        label_name: Semantic label for the positive class (e.g. "religious").
        paths: Input/output directory paths.
        model_slate: Models for the annotation bake-off.
        q_thresholds: Quality-score tiers for the Q rubric.
        sample_sizes: Target sizes for silver/gold/prompt-dev.
        annotation: LLM annotation hyperparameters.
        training: Fine-tuning hyperparameter stubs.

    """

    SEED: int = 42
    entity: str = "missions"
    field: str = "LONGEST_MISSION"
    label_name: str = "religious"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    model_slate: ModelSlateConfig = Field(default_factory=ModelSlateConfig)
    q_thresholds: QThresholdsConfig = Field(default_factory=QThresholdsConfig)
    sample_sizes: SampleSizesConfig = Field(default_factory=SampleSizesConfig)
    annotation: AnnotationConfig = Field(default_factory=AnnotationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


# ── Public API ───────────────────────────────────────────────────────────────


def load_config(path: Path | str) -> BinaryClassifierConfig:
    """Load and validate a YAML configuration file.

    Args:
        path: Path to the YAML config file (e.g.
            ``config/religious_missions.yaml``).

    Returns:
        A validated ``BinaryClassifierConfig`` instance.

    """
    path = Path(path)
    raw_yaml = yaml.safe_load(path.read_text())
    return BinaryClassifierConfig.model_validate(raw_yaml)
