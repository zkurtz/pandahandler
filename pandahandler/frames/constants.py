"""Constants for working with data frames."""

from typing import Any, Protocol

import pandas as pd


class DataframeToDataframe(Protocol):
    """Protocol for functions that take a data frame and return a data frame."""

    __name__: str

    def __call__(self, df: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Specify the function signature."""
        ...


class DataframeToSeries(Protocol):
    """Protocol for functions that take a data frame and return a series."""

    __name__: str

    def __call__(self, df: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.Series:
        """Specify the function signature."""
        ...
