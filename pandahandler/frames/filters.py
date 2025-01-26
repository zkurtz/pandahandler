"""Data frame filtering functions."""

import logging

import pandas as pd

from pandahandler.frames.decorators.framesize import log_rowcount_change

logger = logging.getLogger(__name__)


@log_rowcount_change(logger=logger)
def drop_if_any_null(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with any null values."""
    return df.dropna()
