"""Shared pytest fixtures for the binary-classifier test suite.

The fixtures here build a config in memory and anchor a ``PathRegistry`` in an
isolated ``tmp_path`` so tests never touch the real (cloud-symlinked) data
directories and never require GPU/API access.
"""

import pytest

from binary_classifier.config import BinaryClassifierConfig
from binary_classifier.paths import PathRegistry


@pytest.fixture
def tiny_config() -> BinaryClassifierConfig:
    """A minimal config for tiny synthetic fixtures.

    The production default enforces a minority-F1 CI floor, but many legacy unit
    fixtures use two validation rows where the nonparametric bootstrap lower
    bound is intentionally zero. T9-specific tests opt back into the floor with
    larger fixtures so routine plumbing tests can stay small.
    """
    cfg = BinaryClassifierConfig()
    cfg.qc.f1_ci_floor = 0.0
    return cfg


@pytest.fixture
def tiny_registry(tmp_path, tiny_config: BinaryClassifierConfig) -> PathRegistry:
    """A ``PathRegistry`` anchored in an isolated tmp dir, dirs pre-created."""
    registry = PathRegistry.from_config(tiny_config, root=tmp_path)
    registry.ensure_dirs()
    return registry
