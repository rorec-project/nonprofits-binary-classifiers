"""Tests for T1.2: synthetic data is opt-in, loud, stamped, and cleaned up."""

import pytest

from binary_classifier.data import load as load_module
from binary_classifier.data.load import load_missions
from binary_classifier.config import BinaryClassifierConfig, DataConfig, PathsConfig


def _cfg(tmp_path, allow_synthetic: bool) -> BinaryClassifierConfig:
    # raw_dir points at an empty dir → the parquet files do not exist.
    return BinaryClassifierConfig(
        paths=PathsConfig(raw_dir=tmp_path),
        data=DataConfig(allow_synthetic=allow_synthetic),
    )


def test_missing_parquet_disallowed_raises(tmp_path) -> None:
    """Missing parquet + allow_synthetic=False → hard error with the path."""
    cfg = _cfg(tmp_path, allow_synthetic=False)
    with pytest.raises(FileNotFoundError, match="Upstream parquet not found"):
        load_missions(cfg)


def test_missing_parquet_allowed_stamps_synthetic(tmp_path, caplog) -> None:
    """allow_synthetic=True warns, returns synthetic-stamped data."""
    cfg = _cfg(tmp_path, allow_synthetic=True)
    with caplog.at_level("WARNING"):
        df = load_missions(cfg)
    assert "data_source" in df.columns
    assert (df["data_source"] == "synthetic").all()
    assert any("SYNTHETIC" in r.message for r in caplog.records)


def test_synthetic_temp_dir_cleanup(tmp_path) -> None:
    """The synthetic temp dir is recorded and removable at exit."""
    cfg = _cfg(tmp_path, allow_synthetic=True)
    load_module._SYNTHETIC_DIRS.clear()
    load_missions(cfg)
    assert load_module._SYNTHETIC_DIRS, "temp dir was not recorded"
    created = load_module._SYNTHETIC_DIRS[0]
    assert created.exists()
    load_module._cleanup_synthetic_dirs()
    assert not created.exists()
    assert load_module._SYNTHETIC_DIRS == []
