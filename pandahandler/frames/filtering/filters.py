"""Data frame row-filtering functions."""

import logging

import pandas as pd

from pandahandler.frames.decorators.framesize import log_rowcount_change
from pandahandler.frames.filtering import masks
from pandahandler.frames.filtering.masktools import as_filter

logger = logging.getLogger(__name__)


@log_rowcount_change(logger=logger)
def drop_if_any_null(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with any null values."""
    return df.dropna()


# Additional filters defined from masks; here is a toy example:
my_mask_filter = as_filter(masks.my_mask, logger=logger)
