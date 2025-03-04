"""Defining data frame filters in terms of masks."""

import functools
import logging

import pandas as pd

from pandahandler.frames.constants import DataframeToDataframe, DataframeToSeries
from pandahandler.frames.decorators.framesize import log_rowcount_change

logger = logging.getLogger(__name__)


def _describe_mask_filter(func: DataframeToDataframe, *args, **kwargs) -> str:
    """Describe a filter function that uses a named mask."""
    del args
    assert "name" in kwargs, "apply_mask must be called with a 'name' keyword argument."
    mask_name = kwargs["name"]
    func_name = func.__name__
    return f"{func_name}:{mask_name}"


def _apply_mask(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Safely apply a mask to filter a data frame.

    Raises:
        TypeError: If the mask is off boolean type.
        ValueError: If the mask index is not identical to the data frame index.
    """
    if not mask.index.equals(df.index):
        raise ValueError("The mask index must be identical to the data frame index.")
    if mask.dtype != bool:
        raise TypeError("The mask must be of boolean type.")
    return df.loc[mask]


@log_rowcount_change(
    describe_func=_describe_mask_filter,
)
def apply_mask(df: pd.DataFrame, *, mask: pd.Series | DataframeToSeries, name: str = "unnamed_mask") -> pd.DataFrame:
    """Apply a mask to filter a data frame.

    Args:
        df: The data frame to filter.
        mask: A boolean series with the same index as df, where True values indicate rows to keep.
        name: The name of the mask, for logging purposes only
    """
    del name  # used only by the log_rowcount_change decorator
    if not isinstance(mask, pd.Series):
        mask = mask(df)
    return _apply_mask(df, mask)


def as_filter(mask_func: DataframeToSeries, logger: logging.Logger = logger, **kwargs) -> DataframeToDataframe:
    """Convert a mask function to a filter function.

    Typically users should define masks as standalone functions and then use this decorator to convert them to filters.
    This allows the mask to be used or tested independently of the filter:
    ```
    def my_mask(df: pd.DataFrame) -> pd.Series:
        return df["a"] > 1

    my_filter = as_filter(my_mask)
    ```

    Args:
        mask_func: A function that accepts a data frame and returns a boolean series with the same index.
        logger: The logger to use for logging row count changes.
        **kwargs: Additional keyword arguments to pass to log_rowcount_change.

    Returns:
        A function that accepts a data frame and returns a data frame, where the rows are filtered by the mask.
    """

    @log_rowcount_change(**kwargs)
    @functools.wraps(mask_func)
    def filter_func(df: pd.DataFrame) -> pd.DataFrame:
        mask = mask_func(df)
        return _apply_mask(df, mask=mask)

    return filter_func
