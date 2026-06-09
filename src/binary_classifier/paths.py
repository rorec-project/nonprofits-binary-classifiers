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
        return (
            self._root / self.cfg.paths.upstream_repo / self.cfg.paths.missions_parquet
        )

    @property
    def bmf_parquet(self) -> Path:
        """BMF unified processed parquet (for NTEE major-group join)."""
        return self._root / self.cfg.paths.upstream_repo / self.cfg.paths.bmf_parquet

    # ── Downstream directories ───────────────────────────────────────────────

    @property
    def data_dir(self) -> Path:
        """Directory for labelled CSVs and intermediate outputs."""
        return self._root / self.cfg.paths.data_dir

    @property
    def train_test_dir(self) -> Path:
        """Directory for train/test split manifests."""
        return self._root / self.cfg.paths.train_test_dir

    @property
    def results_dir(self) -> Path:
        """Directory for evaluation metrics and plots."""
        return self._root / self.cfg.paths.results_dir

    @property
    def models_dir(self) -> Path:
        """Directory for persisted fine-tuned models."""
        return self._root / self.cfg.paths.models_dir

    @property
    def gold_dir(self) -> Path:
        """Directory for committed, human-coded gold artifacts."""
        return self._root / self.cfg.paths.gold_dir

    @property
    def silver_dir(self) -> Path:
        """Directory for the cloud-symlinked machine-labelled silver pool."""
        return self._root / self.cfg.paths.silver_dir

    # ── Manifests ────────────────────────────────────────────────────────────

    @property
    def silver_manifest(self) -> Path:
        """EIN2 manifest for the silver pool (~20k)."""
        return self.train_test_dir / "silver_manifest.csv"

    @property
    def gold_manifest(self) -> Path:
        """EIN2 manifest for the gold set (~400)."""
        return self.train_test_dir / "gold_manifest.csv"

    @property
    def prompt_dev_manifest(self) -> Path:
        """EIN2 manifest for the prompt-dev set (~50)."""
        return self.train_test_dir / "prompt_dev_manifest.csv"

    @property
    def validation_manifest(self) -> Path:
        """EIN2 manifest for the validation set."""
        return self.train_test_dir / "validation_manifest.csv"

    @property
    def test_manifest(self) -> Path:
        """EIN2 manifest for the frozen test set."""
        return self.train_test_dir / "test_manifest.csv"

    # ── Human-coding & slate artifacts ───────────────────────────────────────

    @property
    def gold_coding_template(self) -> Path:
        """In-place human-coding template (EIN2, split, text, human_label)."""
        return self.gold_dir / "gold_to_code.csv"

    @property
    def proposed_slate(self) -> Path:
        """Machine-proposed (unconfirmed) slate from the bake-off (stage 02)."""
        return self.results_dir / "proposed_slate.json"

    @property
    def production_slate(self) -> Path:
        """Human-confirmed production slate (committed); gate G2 requires it."""
        return self.gold_dir / "production_slate.json"

    @property
    def bakeoff_results(self) -> Path:
        """Full per-candidate bake-off score bundle (stage 02)."""
        return self.results_dir / "bakeoff_results.json"

    # ── Label stores ─────────────────────────────────────────────────────────

    @property
    def annotation_store(self) -> Path:
        """Long/tidy annotation store for the full silver run (stage 03)."""
        return self.silver_dir / "annotation_store.csv"

    @property
    def bakeoff_store(self) -> Path:
        """Long/tidy bake-off label store (stage 02)."""
        return self.silver_dir / "bakeoff_labels.csv"

    @property
    def prompts_dir(self) -> Path:
        """Directory containing built-in prompt text files."""
        return self._root / "src" / "binary_classifier" / "annotate" / "prompts"

    # ── Convenience helpers ──────────────────────────────────────────────────

    def ensure_dirs(self) -> None:
        """Create all output directories if they do not exist."""
        for d in (
            self.data_dir,
            self.train_test_dir,
            self.results_dir,
            self.models_dir,
            self.gold_dir,
            self.silver_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
