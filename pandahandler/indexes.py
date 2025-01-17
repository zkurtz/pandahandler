"""Tools for working with pandas indexes."""

from typing import Any

import pandas as pd
from attr.validators import min_len
from attrs import field, frozen
from pandas.core.indexes.frozen import FrozenList


def is_unnamed_range_index(index: pd.Index) -> bool:
    """Assert that the index is trivial, i.e. equal to the default RangeIndex."""
    if index.name is not None:
        return False
    unnamed_range_index = pd.RangeIndex(len(index))
    return index.equals(unnamed_range_index)


def assert_no_nulls(index: pd.Index) -> None:
    """Assert that the index has no null values."""
    is_any_null = index.to_frame().isnull().any(axis=None)
    # A numpy bool type gets returned, confusing pyright, so we can simplify that:
    is_any_null = bool(is_any_null)
    if is_any_null:
        raise ValueError("Null values are not allowed in the index.")


def _validate_sort_vs_null(instance: "Index", _: Any, __: Any) -> None:
    if instance.allow_null and instance.sort:
        raise ValueError("`sorted=True` is not allowed with and `allow_null=True`.")


@frozen
class Index:
    """A functional wrapper around pandas indexes.

    An instance of this class can be used to simplify the following types of operations:
    - Coercing an input data frame to have a particular index.
    - Enforcing or ensuring various index properties such as monotonicity or uniqueness.
    - Unset and reset indexes cleanly before and after operations that require columnar access to index columns.

    This class applies both for pandas.Index and MultiIndex objects to reduce the amount of special-casing needed based
    on the number of columns in the index.

    Attributes:
        names: The names of the columns of the index.
        allow_null: Whether to allow null values in the index. Applicable only for single-column indexes, since pandas
            does not support null values in a MultiIndex.
        sort: Whether to sort the index.
        require_unique: Whether the index should be unique
    """

    names: FrozenList = field(converter=FrozenList, validator=min_len(1))
    allow_null: bool = field(default=False, kw_only=True)
    sort: bool = field(default=False, kw_only=True, validator=_validate_sort_vs_null)
    require_unique: bool = field(default=True, kw_only=True)

    def __call__(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Set the index on the data frame.

        Any named columns of the current df index that are not part of the new index will be converted to new
        data frame columns.

        Args:
            df: The data frame to set the index on.
            **kwargs: Additional arguments to pass to the set_index method.
        """
        verify_integrity = kwargs.get("verify_integrity", self.require_unique)
        if verify_integrity is not self.require_unique:
            raise ValueError("The verify_integrity argument must be consistent with the require_unique attribute.")
        kwargs["verify_integrity"] = self.require_unique

        if not is_unnamed_range_index(df.index):
            df = df.reset_index(drop=False)
        df = df.set_index(self.names, **kwargs)

        if not self.allow_null:
            assert_no_nulls(index=df.index)
        if self.sort:
            df = df.sort_index()
        return df
