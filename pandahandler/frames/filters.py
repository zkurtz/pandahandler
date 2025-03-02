"""Data frame row-filtering functions."""

import logging

import pandas as pd

from pandahandler.frames.decorators.framesize import log_rowcount_change

logger = logging.getLogger(__name__)


@log_rowcount_change(logger=logger)
def drop_if_any_null(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with any null values."""
    return df.dropna()


@log_rowcount_change(logger=logger)
def apply_mask(df: pd.DataFrame, *, mask: pd.Series, name: str = "unnamed mask") -> pd.DataFrame:
    """Apply a mask to filter a data frame.

    Args:
        df: The data frame to filter.
        mask: A boolean series with the same index as df, where True values indicate rows to keep.
        name: The name of the mask, for logging purposes only

    Raises:
        TypeError: If the mask is off boolean type.
        ValueError: If the mask index is not identical to the data frame index.
    """
    del name  # used only by the log_rowcount_change decorator
    if not mask.index.equals(df.index):
        raise ValueError("The mask index must be identical to the data frame index.")
    if mask.dtype != bool:
        raise TypeError("The mask must be of boolean type.")
    return df.loc[mask]
