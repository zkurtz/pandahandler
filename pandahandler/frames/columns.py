"""Tools for working with groups of columns in data frames."""

from typing import Hashable

import pandas as pd


def list_categoricals(types_: pd.Series) -> list[Hashable]:
    """Return the names of categorical columns. Call as `get_categoricals(df.dtypes)`."""
    return [col for col, dtype in types_.items() if isinstance(dtype, pd.CategoricalDtype)]


def list_numerics(types_: pd.Series) -> list[Hashable]:
    """Return the names of numeric columns. Call as `list_numerics(df.dtypes)`."""
    return [col for col, dtype in types_.items() if pd.api.types.is_numeric_dtype(dtype)]
