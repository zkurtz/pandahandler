"""Data frame row-filtering functions."""

import logging

import pandas as pd

from pandahandler.frames.decorators.framesize import log_rowcount_change
from pandahandler.frames.filtering import masks

logger = logging.getLogger(__name__)


@log_rowcount_change
def drop_if_any_null(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with any null values."""
    return df.dropna()


# Additional filters defined from masks; here is a toy example:
@log_rowcount_change
def my_mask_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Toy example of a filter that uses a mask."""
    mask = masks.my_mask(df)
    return df.loc[mask]
