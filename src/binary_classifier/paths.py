"""Path registry for the binary classifier package.

All paths are resolved as ``pathlib.Path`` objects. No string concatenation is
used anywhere. The registry reads the config object and exposes both upstream
input paths and downstream output paths.
"""

from pathlib import Path

from binary_classifier.config import BinaryClassifierConfig, load_config


class PathRegistry:
    """Resolved path registry for a given config.

    Attributes are exposed as ``pathlib.Path`` objects so that downstream
    modules can use them directly in ``pd.read_parquet`` or
    ``Path.mkdir(parents=True, exist_ok=True)``.

    Args:
        config_path: Path to the YAML config file (e.g.
            ``config/religious_missions.yaml``).

    """

    def __init__(self, config_path: Path | str) -> None:
        self.cfg: BinaryClassifierConfig = load_config(config_path)
        self._root: Path = Path(".").resolve()

    @classmethod
    def from_config(
        cls,
        cfg: BinaryClassifierConfig,
        root: Path | str | None = None,
    ) -> "PathRegistry":
        """Build a registry from an in-memory config (no YAML on disk).

        Useful for tests and programmatic use. ``root`` overrides the path
        anchor (defaults to the current working directory).

        Args:
            cfg: A validated ``BinaryClassifierConfig`` instance.
            root: Directory all relative paths are resolved against.

        Returns:
            A ``PathRegistry`` bound to ``cfg``.

        """
        registry = cls.__new__(cls)
        registry.cfg = cfg
        registry._root = Path(root if root is not None else ".").resolve()
        return registry

    # ── Upstream inputs ──────────────────────────────────────────────────────

    @property
    def missions_parquet(self) -> Path:
        """Cross-section missions parquet (one row per EIN2)."""
        return self._root / self.cfg.paths.raw_dir / "missions_cross_section.parquet"

    @property
    def bmf_parquet(self) -> Path:
        """BMF unified processed parquet (for NTEE major-group join)."""
        return self._root / self.cfg.paths.raw_dir / "bmf_unified_processed.parquet"

    @property
    def panel_final_parquet(self) -> Path:
        """Upstream longitudinal panel artifact used by the names arm."""
        return self._root / self.cfg.paths.raw_dir / self.cfg.names.panel_final_filename

    @property
    def panel_filled_gaps_parquet(self) -> Path:
        """Upstream panel artifact retaining suffix-stripped organization names."""
        return (
            self._root
            / self.cfg.paths.raw_dir
            / self.cfg.names.panel_filled_gaps_filename
        )

    # ── Downstream directories ───────────────────────────────────────────────

    @property
    def interim_dir(self) -> Path:
        """Cloud-symlinked directory for intermediate pipeline artifacts."""
        return self._root / self.cfg.paths.interim_dir

    @property
    def processed_dir(self) -> Path:
        """Directory for final, ready-to-train datasets."""
        return self._root / self.cfg.paths.processed_dir

    @property
    def names_interim_dir(self) -> Path:
        """Isolated interim directory for names-arm artifacts."""
        return self.interim_dir / "names"

    @property
    def names_processed_dir(self) -> Path:
        """Isolated processed directory for names-arm artifacts."""
        return self.processed_dir / "names"

    @property
    def names_panel_frame(self) -> Path:
        """501(c)(3) panel name frame."""
        return self.names_interim_dir / "panel_name_frame.parquet"

    @property
    def names_bmf_only_frame(self) -> Path:
        """BMF-only name frame."""
        return self.names_interim_dir / "bmf_only_name_frame.parquet"

    @property
    def names_panel_cleaned(self) -> Path:
        """Panel frame with the shared name cleaner applied."""
        return self.names_interim_dir / self.cfg.names.panel_cleaned_filename

    @property
    def names_bmf_only_cleaned(self) -> Path:
        """BMF-only frame with the shared name cleaner applied."""
        return self.names_interim_dir / self.cfg.names.bmf_only_cleaned_filename

    @property
    def names_divergence_audit(self) -> Path:
        """Divergence audit produced by the names cleaner stage."""
        return self.names_interim_dir / self.cfg.names.divergence_audit_filename

    @property
    def names_scores(self) -> Path:
        """Cross-field transfer scores for both name-input variants."""
        return self.names_processed_dir / self.cfg.names.scores_filename

    @property
    def names_validation(self) -> Path:
        """Paired transfer-validation report for the names arm."""
        return self.names_processed_dir / self.cfg.names.validation_filename

    @property
    def names_probe_diagnostics(self) -> Path:
        """Synthetic-probe diagnostic report for cross-field transfer."""
        return self.names_processed_dir / self.cfg.names.probe_diagnostics_filename

    @property
    def names_dba_case_study(self) -> Path:
        """Reviewable EIN2-level legal-name and DBA comparison cases."""
        return self.names_processed_dir / self.cfg.names.dba_case_study_filename

    @property
    def names_dba_case_study_report(self) -> Path:
        """Summary report for the DBA case study."""
        return self.names_processed_dir / self.cfg.names.dba_case_study_report_filename

    @property
    def names_gold_manifest(self) -> Path:
        """Seeded BMF-only name-gold draw with stratum provenance."""
        return self.names_interim_dir / "names_gold_manifest.csv"

    @property
    def names_gold_coding_template(self) -> Path:
        """Human-coding template for the BMF-only names gold sample."""
        return self.names_processed_dir / "gold" / "names_gold_to_code.csv"

    @property
    def names_gold_coding_instructions(self) -> Path:
        """Unchanged mission-construct rubric accompanying the names template."""
        return self.names_processed_dir / "gold" / "names_gold_coding_instructions.md"

    @property
    def gold_dir(self) -> Path:
        """Git-committed directory for human-coded gold artifacts."""
        return self.processed_dir / "gold"

    @property
    def bakeoff_dir(self) -> Path:
        """Sub-directory under ``interim_dir`` for bake-off scores and
        the proposed (unconfirmed) slate."""
        return self.interim_dir / "bakeoff"

    @property
    def models_dir(self) -> Path:
        """Directory for persisted fine-tuned models."""
        return self._root / self.cfg.paths.models_dir

    # ── Manifests ────────────────────────────────────────────────────────────

    @property
    def silver_manifest(self) -> Path:
        """EIN2 manifest for the silver pool (~20k)."""
        return self.interim_dir / "manifests" / "silver_manifest.csv"

    @property
    def gold_manifest(self) -> Path:
        """EIN2 manifest for gold rows, including the monitor split."""
        return self.interim_dir / "manifests" / "gold_manifest.csv"

    @property
    def prompt_dev_manifest(self) -> Path:
        """EIN2 manifest for the prompt-dev set (~50)."""
        return self.interim_dir / "manifests" / "prompt_dev_manifest.csv"

    @property
    def validation_manifest(self) -> Path:
        """EIN2 manifest for the validation set."""
        return self.interim_dir / "manifests" / "validation_manifest.csv"

    @property
    def test_manifest(self) -> Path:
        """EIN2 manifest for the frozen test set."""
        return self.interim_dir / "manifests" / "test_manifest.csv"

    @property
    def monitor_manifest(self) -> Path:
        """EIN2 manifest for the held-out drift-monitor slice.

        The monitor split is written beside the other manifests so stage 03 can
        later run canary/drift checks without mixing those rows into the
        validation freeze gate.
        """
        return self.interim_dir / "manifests" / "monitor_manifest.csv"

    @property
    def anchor_manifest(self) -> Path:
        """EIN2 manifest for the full-frame anchor sample."""
        return self.interim_dir / "manifests" / "anchor_manifest.csv"

    # ── Human-coding & slate artifacts ───────────────────────────────────────

    @property
    def gold_coding_template(self) -> Path:
        """In-place human-coding template (EIN2, split, text, human_label)."""
        return self.gold_dir / "gold_to_code.csv"

    @property
    def anchor_coding_template(self) -> Path:
        """In-place human-coding template for the anchor sample."""
        return self.gold_dir / "anchor_to_code.csv"

    @property
    def proposed_slate(self) -> Path:
        """Machine-proposed (unconfirmed) slate from the bake-off (stage 02)."""
        return self.bakeoff_dir / "proposed_slate.json"

    @property
    def production_slate(self) -> Path:
        """Human-confirmed production slate (committed); gate G2 requires it."""
        return self.gold_dir / "production_slate.json"

    @property
    def selected_model(self) -> Path:
        """Human-confirmed selected model artifact."""
        return self.gold_dir / "selected_model.json"

    @property
    def test_unlock(self) -> Path:
        """Human-confirmed test-unlock artifact."""
        return self.gold_dir / "test_unlock.json"

    @property
    def bakeoff_results(self) -> Path:
        """Full per-candidate bake-off score bundle (stage 02)."""
        return self.bakeoff_dir / "bakeoff_results.json"

    # ── Label stores ─────────────────────────────────────────────────────────

    @property
    def annotation_store(self) -> Path:
        """Long/tidy annotation store for the full silver run (stage 03)."""
        return self.interim_dir / "annotation_store.csv"

    @property
    def bakeoff_store(self) -> Path:
        """Long/tidy bake-off label store (stage 02)."""
        return self.bakeoff_dir / "bakeoff_labels.csv"

    @property
    def prompts_dir(self) -> Path:
        """Directory containing built-in prompt text files."""
        return self._root / "src" / "binary_classifier" / "annotate" / "prompts"

    @property
    def silver_labels(self) -> Path:
        """Aggregated silver labels for downstream training."""
        return self.processed_dir / "silver_labels.csv"

    @property
    def runs_dir(self) -> Path:
        """Directory for training sweep run artifacts."""
        return self.models_dir / "runs"

    @property
    def checkpoints_dir(self) -> Path:
        """Directory for model checkpoints."""
        return self.models_dir / "checkpoints"

    @property
    def learning_curve_results(self) -> Path:
        """JSONL learning-curve results artifact."""
        return self.runs_dir / "results.jsonl"

    @property
    def selection_report(self) -> Path:
        """Model-selection report artifact."""
        return self.models_dir / "selection_report.json"

    @property
    def oof_pred_probs(self) -> Path:
        """Out-of-fold prediction probabilities for training rows."""
        return self.interim_dir / "oof_pred_probs.parquet"

    @property
    def embeddings_dir(self) -> Path:
        """Directory for cached text embeddings."""
        return self.interim_dir / "embeddings"

    @property
    def evaluation_dir(self) -> Path:
        """Directory for evaluation artifacts."""
        return self.processed_dir / "evaluation"

    @property
    def test_evaluation(self) -> Path:
        """Frozen test-set evaluation report."""
        return self.evaluation_dir / "test_evaluation.json"

    @property
    def calibrator_path(self) -> Path:
        """Persisted calibration model artifact."""
        return self.evaluation_dir / "calibrator.json"

    @property
    def anchor_oof_scores(self) -> Path:
        """Out-of-fold scores for anchor rows."""
        return self.evaluation_dir / "anchor_oof_scores.parquet"

    @property
    def rule_validation(self) -> Path:
        """Rule-layer validation report."""
        return self.evaluation_dir / "rule_validation.json"

    @property
    def base_rate_precision(self) -> Path:
        """Base-rate-adjusted precision report."""
        return self.evaluation_dir / "base_rate_precision.json"

    @property
    def predictions_dir(self) -> Path:
        """Directory for inference-at-scale predictions."""
        return self.processed_dir / "predictions"

    @property
    def predictions_parquet(self) -> Path:
        """Full predictions parquet artifact."""
        return self.predictions_dir / "predictions.parquet"

    @property
    def predictions_full_parquet(self) -> Path:
        """Per-organization predictions expanded back to raw EIN2 rows."""
        return self.predictions_dir / "predictions_full.parquet"

    @property
    def monitor_scores(self) -> Path:
        """Monitor-slice prediction scores."""
        return self.predictions_dir / "monitor_scores.json"

    @property
    def prevalence_dir(self) -> Path:
        """Directory for prevalence-estimation artifacts."""
        return self.processed_dir / "prevalence"

    @property
    def prevalence_report(self) -> Path:
        """Prevalence-estimation report."""
        return self.prevalence_dir / "prevalence_report.json"

    @property
    def prevalence_by_ntee(self) -> Path:
        """Prevalence estimates grouped by NTEE major group."""
        return self.prevalence_dir / "prevalence_by_ntee.csv"

    @property
    def figures_dir(self) -> Path:
        """Directory for generated figures."""
        return self.processed_dir / "figures"

    @property
    def run_manifest(self) -> Path:
        """Reproducibility manifest for the current local run."""
        return self.processed_dir / "run_manifest.json"

    @property
    def aggregation_compare(self) -> Path:
        """Aggregation-comparison diagnostics artifact."""
        return self.interim_dir / "aggregation_compare.json"

    # ── Convenience helpers ──────────────────────────────────────────────────

    def ensure_dirs(self) -> None:
        """Create all output directories if they do not exist."""
        for d in (
            self.interim_dir / "manifests",
            self.bakeoff_dir,
            self.models_dir,
            self.runs_dir,
            self.checkpoints_dir,
            self.gold_dir,
            self.interim_dir,
            self.embeddings_dir,
            self.processed_dir,
            self.evaluation_dir,
            self.predictions_dir / "shards",
            self.prevalence_dir,
            self.figures_dir,
            self.names_interim_dir,
            self.names_processed_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
