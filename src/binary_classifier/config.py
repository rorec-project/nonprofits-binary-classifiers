"""Typed configuration loader for the binary classifier package.

Reads YAML config files into a validated Pydantic model. All downstream
stages (data loading, annotation, training, evaluation) consume the same
config object so that paths, seeds, thresholds, and model slates are defined
in a single source of truth.
"""

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

# ── Sub-configs ──────────────────────────────────────────────────────────────


class PathsConfig(BaseModel):
    """Input/output directory paths.

    Follows the cookiecutter-data-science convention: ``raw/`` for
    immutable upstream inputs, ``interim/`` for intermediate pipeline
    artifacts (cloud-symlinked), ``processed/`` for final ready-to-train
    datasets, and ``models/`` for persisted trained models.

    Attributes:
        raw_dir: Directory for upstream parquet files (e.g.
            ``missions_cross_section.parquet``, ``bmf_unified_processed.parquet``).
        interim_dir: Cloud-symlinked directory for intermediate pipeline
            outputs (manifests, bake-off scores, annotation stores).
        processed_dir: Directory for final datasets. The ``gold/``
            sub-directory is git-committed (human-coded labels and the
            production slate); ``silver_labels.csv`` is gitignored.
        models_dir: Directory for persisted fine-tuned model artifacts
            (future roadmap).

    """

    raw_dir: Path = Field(default=Path("data/raw"))
    interim_dir: Path = Field(default=Path("data/interim"))
    processed_dir: Path = Field(default=Path("data/processed"))
    models_dir: Path = Field(default=Path("data/models"))


class BakeoffCandidate(BaseModel):
    """A single model in the bake-off slate, tagged by serving provider.

    Attributes:
        id: Model identifier. For ``openai`` this is the API model/snapshot
            (e.g. ``gpt-5-mini``); for ``vllm`` the HuggingFace id
            (e.g. ``google/gemma-3-27b-it``).
        provider: Serving backend — ``openai`` (closed API) or ``vllm``
            (local open-weight endpoint). Routes annotator construction.
        reasoning_effort: Optional reasoning-effort knob for GPT-5-class
            models (e.g. ``minimal``). Forwarded to the annotator only when
            set; ignored by providers that do not support it.

    """

    id: str
    provider: Literal["openai", "vllm"]
    reasoning_effort: str | None = None


def _default_bakeoff_candidates() -> list[BakeoffCandidate]:
    """Seed slate: three closed OpenAI tiers + one open-weight Gemma arm.

    OpenAI ids are floating aliases (placeholders); pin them to dated
    snapshots before a production run. The Gemma arm requires the vLLM
    server to be up during the bake-off; comment it out (in the YAML) for a
    pure-OpenAI bake-off.
    """
    return [
        BakeoffCandidate(id="gpt-4o-mini", provider="openai"),
        BakeoffCandidate(
            id="gpt-5-mini",
            provider="openai",
            reasoning_effort="minimal",
        ),
        BakeoffCandidate(
            id="gpt-5-nano",
            provider="openai",
            reasoning_effort="minimal",
        ),
        BakeoffCandidate(id="google/gemma-3-27b-it", provider="vllm"),
    ]


class ModelSlateConfig(BaseModel):
    """Config-driven model slate for the bake-off and production annotation.

    Attributes:
        bakeoff_candidates: The list of ``(id, provider, reasoning_effort?)``
            candidates scored in stage 02. Each is tagged with its serving
            provider so the annotator factory can route construction.
        production: The default production model id. Stage 03 only runs what
            the human confirms in ``production_slate.json`` (gate G2), so this
            is a default/seed rather than a binding choice.

    """

    bakeoff_candidates: list[BakeoffCandidate] = Field(
        default_factory=_default_bakeoff_candidates,
    )
    production: str = "gpt-5-mini"


class Slate(BaseModel):
    """A bake-off slate file (proposed or human-confirmed).

    The same shape is written by stage 02 as the *proposed* slate
    (``confirmed=False``) and copied/edited by the human into the
    *production* slate (``confirmed=True``) that gate G2 requires.

    Attributes:
        confirmed: ``True`` only after a human has reviewed the scores and
            promoted the slate. Stage 03 refuses to run an unconfirmed slate.
        models: The authoritative model set stage 03 runs (each × the prompt
            set). The human edits this list to control production.
        selected: Per-(model, prompt) detail retained for review — which combos
            cleared the agreement threshold and their scores. Not consumed by
            stage 03; informational.

    """

    confirmed: bool = False
    models: list[BakeoffCandidate] = Field(default_factory=list)
    selected: list[dict[str, Any]] = Field(default_factory=list)


def load_slate(path: Path | str) -> Slate:
    """Load and validate a slate JSON file.

    Args:
        path: Path to ``proposed_slate.json`` or ``production_slate.json``.

    Returns:
        A validated ``Slate`` instance.

    """
    raw = json.loads(Path(path).read_text())
    return Slate.model_validate(raw)


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
    """Target sizes for the silver pool, gold set, and human splits.

    The silver pool is over-provisioned so that the learning-curve sweep in
    point 3 can empirically find the optimal training N. The gold set is kept
    small because it is hand-coded by a human engineer; the monitor slice is an
    incremental holdout, so ``gold`` includes it instead of shrinking
    validation/test.
    """

    silver: int = 20_000
    gold: int = 450
    prompt_dev: int = 50
    monitor: int = Field(
        default=50,
        description=(
            "Held-out drift-monitor slice drawn from gold in addition to "
            "prompt_dev/validation/test, so gold must be bumped by this size."
        ),
    )


class AnnotationConfig(BaseModel):
    """Hyperparameters for the LLM-as-primary labeler.

    Temperature is fixed to keep annotations low-variance / best-effort
    reproducible (closed APIs do not guarantee determinism across backend
    changes). The global ``SEED`` from :class:`BinaryClassifierConfig` is
    used for seeding. Resume is keyed by (EIN2, source_id) rather than row
    count to avoid the audit R-08 idempotency gap.
    """

    temperature: float = 0.0
    max_retries: int = 5
    checkpoint_every: int = 100
    guided_json: bool = True


class DataConfig(BaseModel):
    """Data-loading behaviour.

    Attributes:
        allow_synthetic: When ``False`` (default), a missing upstream parquet
            is a hard error. When ``True``, a synthetic dataset is generated
            (with a loud warning and a ``data_source="synthetic"`` stamp) for
            local smoke-testing only.

    """

    allow_synthetic: bool = False


class QCConfig(BaseModel):
    """Quality-control gate thresholds.

    Attributes:
        agreement_threshold: Legacy raw LLM-vs-human agreement benchmark on the
            validation overlap. This is logged for continuity, but the blocking
            freeze gate now uses chance-corrected agreement and minority-F1 CI.
        kappa_threshold: Minimum Cohen's κ required to freeze silver labels.
            The default of 0.70 matches the old ≈0.85 raw-agreement operating
            point on the roughly balanced validation gate set.
        f1_ci_floor: Minimum lower bound of the bootstrap 95% confidence
            interval for minority-class F1 required to freeze silver labels.
        abstain_on_fabricated_positive: When ``True``, any positive label
            (``binary_label == "religious"``) that carries a fabricated
            evidence span is treated as an abstain (``binary_label = None``)
            before aggregation.

    """

    agreement_threshold: float = 0.85
    kappa_threshold: float = 0.70
    f1_ci_floor: float = 0.70
    abstain_on_fabricated_positive: bool = False


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
        data: Data-loading behaviour (synthetic opt-in).
        qc: Quality-control gate thresholds.
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
    data: DataConfig = Field(default_factory=DataConfig)
    qc: QCConfig = Field(default_factory=QCConfig)
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
