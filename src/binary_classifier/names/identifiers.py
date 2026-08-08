"""Identifier normalization shared by names-arm stage boundaries."""

import pandas as pd


def normalize_ein2(values: pd.Series) -> pd.Series:
    """Return stripped nullable string ``EIN2`` values without inventing IDs.

    Args:
        values: Source identifiers, potentially with surrounding whitespace or nulls.

    Returns:
        Identifiers represented with pandas' nullable string dtype. Null handling is
        deliberately left to the consuming stage because frame construction and
        validation have different contracts for incomplete source data.
    """
    return values.astype("string").str.strip()
