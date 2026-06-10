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

    # ── Human-coding & slate artifacts ───────────────────────────────────────

    @property
    def gold_coding_template(self) -> Path:
        """In-place human-coding template (EIN2, split, text, human_label)."""
        return self.gold_dir / "gold_to_code.csv"

    @property
    def proposed_slate(self) -> Path:
        """Machine-proposed (unconfirmed) slate from the bake-off (stage 02)."""
        return self.bakeoff_dir / "proposed_slate.json"

    @property
    def production_slate(self) -> Path:
        """Human-confirmed production slate (committed); gate G2 requires it."""
        return self.gold_dir / "production_slate.json"

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
        return self.interim_dir / "bakeoff_labels.csv"

    @property
    def prompts_dir(self) -> Path:
        """Directory containing built-in prompt text files."""
        return self._root / "src" / "binary_classifier" / "annotate" / "prompts"

    # ── Convenience helpers ──────────────────────────────────────────────────

    def ensure_dirs(self) -> None:
        """Create all output directories if they do not exist."""
        for d in (
            self.interim_dir / "manifests",
            self.bakeoff_dir,
            self.models_dir,
            self.gold_dir,
            self.interim_dir,
            self.processed_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
