"""NTEE major-group letter to human-readable label lookup.

The table ships inside the package (``data/resources/ntee_major_group_labels.csv``)
rather than under ``data/``, since it is a fixed code-adjacent reference table, not
generated pipeline output. See the CSV's header comment for provenance.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

import pandas as pd

_RESOURCE_PACKAGE = "binary_classifier.data.resources"
_RESOURCE_NAME = "ntee_major_group_labels.csv"


@lru_cache(maxsize=1)
def load_ntee_labels() -> pd.DataFrame:
    """Load the NTEE major-group label table.

    Returns:
        A frame with columns ``ntee_major_group`` and ``label``, one row per
        letter ``A`` through ``Z``.

    """
    resource = resources.files(_RESOURCE_PACKAGE) / _RESOURCE_NAME
    with resources.as_file(resource) as path:
        return pd.read_csv(path, comment="#")


def ntee_label_map() -> dict[str, str]:
    """Return the NTEE major-group label table as a ``{letter: label}`` dict."""
    labels = load_ntee_labels()
    return dict(zip(labels["ntee_major_group"], labels["label"], strict=True))
