"""Tools for working with pandas indexes."""

import pandas as pd


def is_unnamed_range_index(index: pd.Index) -> bool:
    """Assert that the index is trivial, i.e. equal to the default RangeIndex."""
    if index.name is not None:
        return False
    unnamed_range_index = pd.RangeIndex(len(index))
    return index.equals(unnamed_range_index)
