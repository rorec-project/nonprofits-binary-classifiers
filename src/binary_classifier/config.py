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
from pydantic import BaseModel, ConfigDict, Field

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
        models: The authoritative model set stage 03 runs. The human edits this
            list to control production.
        selected: Per-(model, prompt) detail retained for review — which combos
            cleared the agreement threshold and their scores. When present,
            stage 03 consumes these selected ``(model_id, prompt_id)`` pairs.

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


class AcceptanceCriteria(BaseModel):
    """Frozen-test and calibration acceptance thresholds.

    Attributes:
        min_pr_auc: Minimum precision-recall AUC on the frozen test set.
        min_minority_f1_ci_lower: Minimum lower confidence bound for minority
            F1 on the frozen test set.
        max_ece: Maximum expected calibration error on anchor OOF scores.
            Kept as the sole acceptance gate for this pass (Brier/log-loss
            are future work, out of scope). Retained per confirmed design
            decision (DO NOT change).

    """

    min_pr_auc: float = 0.90
    min_minority_f1_ci_lower: float = 0.70
    max_ece: float = 0.05


def _default_calibration_methods() -> list[Literal["platt", "temperature"]]:
    """Return the default calibration method slate.

    Platt scaling + temperature scaling are compared, then the best is
    deployed. Kept per confirmed design decision: DO NOT reduce to
    temperature-only; DO NOT add isotonic (overfits at anchor n=500).
    """
    return ["platt", "temperature"]


def _default_length_bins() -> list[int]:
    """Return default word-count bin edges for subgroup evaluation."""
    return [10, 25, 50]


class EvaluationConfig(BaseModel):
    """Evaluation, calibration, and test-unlock configuration.

    Attributes:
        acceptance: Threshold snapshot required for test unlock approval.
        calibration_methods: Calibration methods considered for score scaling.
        crossfit_folds: Number of folds for anchor out-of-fold scoring.
        threshold_policy: Rule for selecting the operating threshold.
        precision_floor: Minimum precision under the precision-floor policy.
        ece_bins: Number of bins for expected calibration error.
        bootstrap_resamples: Bootstrap draws for confidence intervals.
        length_bins: Word-count bin edges for length subgroup reporting.
        base_rate_precision_target: Target deployment precision after adjusting
            anchor operating characteristics to the population base rate.
        population_base_rate: Optional externally supplied population base rate;
            when omitted, stage 07 derives a design-weighted anchor fallback.

    """

    acceptance: AcceptanceCriteria = Field(default_factory=AcceptanceCriteria)
    calibration_methods: list[Literal["platt", "temperature"]] = Field(
        default_factory=_default_calibration_methods,
    )
    crossfit_folds: int = 5
    threshold_policy: Literal["precision_floor", "max_f1"] = "precision_floor"
    precision_floor: float = 0.80
    ece_bins: int = 10
    bootstrap_resamples: int = 2000
    length_bins: list[int] = Field(default_factory=_default_length_bins)
    base_rate_precision_target: float = 0.90
    population_base_rate: float | None = None


class InferenceConfig(BaseModel):
    """Batch inference configuration.

    Attributes:
        batch_size: Number of rows scored per classifier batch.
        shard_size: Number of input rows processed per inference shard.
        device: Compute device selection policy for classifier inference.
        route_low_to_rules: Whether LOW-quality rows use the rule layer before
            classifier scoring.
        rule_ambiguous_to_classifier: Whether rule-layer abstentions on
            LOW-quality rows fall through to the classifier.

    """

    batch_size: int = 512
    shard_size: int = 50_000
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    route_low_to_rules: bool = True
    rule_ambiguous_to_classifier: bool = True


def _default_prevalence_cross_checks() -> list[Literal["emq", "kdey"]]:
    """Return the default prevalence cross-check estimators.

    PPI++ is the primary prevalence estimator (Angelopoulos 2023). EMQ
    (vendored SLD, Saerens et al. 2002) is the single quantification
    sensitivity check retained in the default path. KDEy (Moreo et al.
    2021) is available as an opt-in extra under the ``quant`` dependency
    group.
    """
    return ["emq"]


class PrevalenceConfig(BaseModel):
    """Population-prevalence estimation configuration.

    Attributes:
        alpha: Significance level used for prevalence confidence intervals.
        cross_checks: Secondary quantification estimators reported alongside the
            primary prevalence estimate.
        use_design_weights: Whether anchor-sample design weights are used when
            estimating prevalence.
        per_ntee: Whether prevalence is also reported by NTEE major group.
        ntee_min_n: Minimum labeled/inferred rows required for an NTEE subgroup
            estimate.
        low_tier_sensitivity: Whether to report sensitivity bounds for
            LOW-quality missions routed through the rule layer.

    """

    alpha: float = 0.05
    cross_checks: list[Literal["emq", "kdey"]] = Field(
        default_factory=_default_prevalence_cross_checks,
    )
    use_design_weights: bool = True
    per_ntee: bool = True
    ntee_min_n: int = 10
    low_tier_sensitivity: bool = True


class AggregationConfig(BaseModel):
    """Production aggregation plus stage-11 diagnostic comparison arms."""

    method: Literal["majority"] = "majority"
    comparison_arms: list[Literal["dawid_skene", "crowdlab"]] = Field(
        default_factory=list,
    )


class TestUnlock(BaseModel):
    """A human-confirmed frozen-test unlock artifact.

    Attributes:
        confirmed: ``True`` only after human review authorizes test evaluation.
        checkpoint: Human-readable checkpoint identifier or relative path.
        checkpoint_sha256: SHA-256 digest for the selected model checkpoint.
        acceptance: Snapshot of acceptance criteria at unlock time.
        rationale: Human rationale for unlocking the frozen test set.

    """

    confirmed: bool = False
    checkpoint: str = ""
    checkpoint_sha256: str = ""
    acceptance: AcceptanceCriteria = Field(default_factory=AcceptanceCriteria)
    rationale: str = ""


def load_test_unlock(path: Path | str) -> TestUnlock:
    """Load and validate a test-unlock JSON file.

    Args:
        path: Path to ``test_unlock.json``.

    Returns:
        A validated ``TestUnlock`` instance.

    """
    raw = json.loads(Path(path).read_text())
    return TestUnlock.model_validate(raw)


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


class AnchorConfig(BaseModel):
    """Stage 05 anchor sample over the FULL frame (incl. LOW)."""

    n: int = 500
    oversample_low_factor: float = 1.5
    min_stratum_frame: int = 200


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
    openai_max_concurrency: int = Field(default=2, gt=0)
    vllm_max_concurrency: int = Field(default=8, gt=0)
    openai_batch: bool = False
    openai_batch_poll_seconds: int = Field(default=30, gt=0)
    openai_batch_completion_window: Literal["24h"] = "24h"


class DataConfig(BaseModel):
    """Data-loading behaviour.

    Attributes:
        allow_synthetic: When ``False`` (default), a missing upstream parquet
            is a hard error. When ``True``, a synthetic dataset is generated
            (with a loud warning and a ``data_source="synthetic"`` stamp) for
            local smoke-testing only.

    """

    allow_synthetic: bool = False


class NamesExpectedCounts(BaseModel):
    """Snapshot counts that reconcile the names-arm input populations."""

    panel_has_mission: int
    panel_name_only: int
    panel_no_name_no_mission: int
    panel_name_only_flagged: int
    bmf_only: int
    bmf_only_flagged: int


class NamesConfig(BaseModel):
    """Names-arm filenames, diagnostics, and optional snapshot reconciliation.

    ``diagnostic_threshold`` is deliberately not a transferable operating point;
    it only turns raw name scores into labels for the external-flag diagnostic.
    """

    panel_final_filename: str = "panel_final.parquet"
    panel_filled_gaps_filename: str = "panel_filled_gaps.parquet"
    panel_tax_year_column: str = "TAX_YEAR"
    panel_scope_column: str = "COMMON_LEVEL1"
    panel_raw_name_columns: list[str] = Field(
        default_factory=lambda: ["F9_00_ORG_NAME_L1", "NAME_CASED"],
        min_length=1,
    )
    panel_best_name_cased_column: str = "BEST_NAME_CASED"
    panel_best_name_bare_column: str = "BEST_NAME_BARE_CASED"
    panel_dba_cased_column: str = "BEST_DBA_CASED"
    panel_has_dba_column: str = "HAS_DBA"
    panel_cleaned_filename: str = "panel_cleaned_names.parquet"
    bmf_only_cleaned_filename: str = "bmf_only_cleaned_names.parquet"
    divergence_audit_filename: str = "name_divergence_audit.json"
    scores_filename: str = "name_scores.parquet"
    validation_filename: str = "name_validation.json"
    probe_diagnostics_filename: str = "name_probe_diagnostics.json"
    dba_case_study_filename: str = "name_dba_case_study.parquet"
    dba_case_study_report_filename: str = "name_dba_case_study_report.json"
    gold_sample_size: int = Field(default=400, gt=0)
    gold_seed: int | None = None
    gold_stratum_quotas: dict[str, int] = Field(
        default_factory=lambda: {
            "ntee_x_only": 100,
            "church_foundation_only": 100,
            "both_external_flags": 100,
            "neither_external_flag": 100,
        }
    )
    gold_conflict_quotas: dict[str, int] = Field(
        default_factory=lambda: {
            "saint_name": 20,
            "faith_heritage": 20,
            "non_christian_tradition": 20,
            "non_english_name": 20,
        }
    )
    diagnostic_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    base_rate_shift_ratio_tolerance: float = Field(default=0.25, ge=0.0)
    expected_counts: NamesExpectedCounts | None = None


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


TrainingArm = Literal["hard", "pruned", "class_weighted"]
BaselineName = Literal["tfidf_logreg", "minilm_logreg"]


def _default_training_arms() -> list[TrainingArm]:
    """Return the default training-data variants for the sweep.

    Soft vote-share targets are the default (see ``training.targets``).
    The ``pruned`` arm is dropped as redundant: soft labels already
    down-weight the disagreement band that pruning removes
    (arXiv:2605.20642). The ``pruned`` arm remains available as an
    opt-in by overriding this default.
    """
    return ["hard", "class_weighted"]


def _default_curve_fractions() -> list[float]:
    """Return the default learning-curve fractions.

    The multi-fraction learning curve (0.25, 0.5, 1.0) is dropped in
    favour of a single full-data run. This simplifies the sweep stage
    while retaining the documented variance estimate.
    """
    return [1.0]


def _default_sweep_seeds() -> list[int]:
    """Return the default model-selection seeds."""
    return [42, 43, 44]


def _default_final_seeds() -> list[int]:
    """Return the default final-refit seeds.

    Five seeds are required for the reported variance estimate.
    Retained per confirmed design decision (DO NOT reduce below 5).
    """
    return [42, 43, 44, 45, 46]


def _default_baselines() -> list[BaselineName]:
    """Return the default baseline model families."""
    return ["tfidf_logreg", "minilm_logreg"]


class EncoderArm(BaseModel):
    """Encoder model participating in the training comparison slate.

    Attributes:
        id: HuggingFace model identifier for the encoder.
        arm: Whether this encoder is the primary production candidate or a
            comparison arm.
        max_length: Maximum tokenized sequence length for training/evaluation.

    """

    id: str
    arm: Literal["primary", "comparison"] = "comparison"
    max_length: int = 256
    precision: str | None = None


class TrainingConfig(BaseModel):
    """Fine-tuning, baseline, and learning-curve configuration.

    Attributes:
        dev_fraction: Fraction of silver labels reserved for development during
            training-data construction.
        targets: Whether encoder training consumes soft aggregate scores or hard
            labels.
        arms: Training-data variants included in the learning-curve sweep.
        curve_fractions: Fractions of available training data used for learning
            curve comparisons.
        sweep_seeds: Random seeds used during the model-selection sweep.
        final_seeds: Random seeds used for final model refits.
        crossfit_folds: Number of folds for cross-fit silver-label prediction.
        encoders: Encoder slate evaluated by the training stage.
        baselines: Baseline model families evaluated alongside encoders.
        minilm_id: Sentence-transformer id for the MiniLM logistic baseline.
        learning_rate: Optimizer learning rate for encoder fine-tuning.
        batch_size: Per-device training batch size.
        epochs: Maximum number of training epochs.
        weight_decay: Optimizer weight decay.
        warmup_fraction: Fraction of training steps used for learning-rate
            warmup.
        early_stopping_patience: Validation checks without improvement before
            stopping.
        metric_for_best_model: Validation metric used for checkpoint selection.
        greater_is_better: Whether larger values of the selection metric are
            preferred.
        save_total_limit: Maximum checkpoints retained per training run.
        precision: Numeric precision policy for encoder fine-tuning.
        device: Compute device selection policy.
        report_to: External training loggers enabled for trainer integrations.

    """

    dev_fraction: float = 0.1
    targets: Literal["soft", "hard"] = "soft"
    arms: list[TrainingArm] = Field(default_factory=_default_training_arms)
    curve_fractions: list[float] = Field(default_factory=_default_curve_fractions)
    sweep_seeds: list[int] = Field(default_factory=_default_sweep_seeds)
    final_seeds: list[int] = Field(default_factory=_default_final_seeds)
    crossfit_folds: int = 5
    encoders: list[EncoderArm] = Field(
        default_factory=lambda: [
            EncoderArm(id="microsoft/deberta-v3-base", arm="primary"),
            EncoderArm(id="answerdotai/ModernBERT-base", arm="comparison"),
        ],
    )
    baselines: list[BaselineName] = Field(default_factory=_default_baselines)
    minilm_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    learning_rate: float = 2e-5
    batch_size: int = 32
    epochs: int = 10
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_fraction: float = 0.06
    early_stopping_patience: int = 4
    metric_for_best_model: str = "pr_auc"
    greater_is_better: bool = True
    save_total_limit: int = 2
    precision: Literal["auto", "bf16", "fp32"] = "auto"
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    report_to: list[str] = Field(default_factory=list)


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
        anchor: Anchor-sample sizing and LOW-quality oversampling controls.
        annotation: LLM annotation hyperparameters.
        data: Data-loading behaviour (synthetic opt-in).
        qc: Quality-control gate thresholds.
        aggregation: Majority-only production aggregation and stage-11
            diagnostic comparison arms.
        training: Fine-tuning, baseline, and learning-curve settings.
        evaluation: Evaluation, calibration, and test-unlock settings.
        inference: Batch inference and LOW-tier routing settings.
        prevalence: Population-prevalence estimation settings.
        names: Names-arm upstream filename settings.

    """

    model_config = ConfigDict(extra="forbid")

    SEED: int = 42
    entity: str = "missions"
    field: str = "LONGEST_MISSION"
    label_name: str = "religious"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    model_slate: ModelSlateConfig = Field(default_factory=ModelSlateConfig)
    q_thresholds: QThresholdsConfig = Field(default_factory=QThresholdsConfig)
    sample_sizes: SampleSizesConfig = Field(default_factory=SampleSizesConfig)
    anchor: AnchorConfig = Field(default_factory=AnchorConfig)
    annotation: AnnotationConfig = Field(default_factory=AnnotationConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    names: NamesConfig = Field(default_factory=NamesConfig)
    qc: QCConfig = Field(default_factory=QCConfig)
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    prevalence: PrevalenceConfig = Field(default_factory=PrevalenceConfig)


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
