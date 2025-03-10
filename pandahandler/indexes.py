"""Tools for working with pandas indexes."""

from typing import Any

import pandas as pd
from attr.validators import min_len
from attrs import field, frozen
from pandas.core.indexes.frozen import FrozenList

__all__ = [
    "Index",
    "assert_no_nulls",
    "is_unnamed_range_index",
    "unset",
]


def index_has_any_unnamed_col(index: pd.Index) -> bool:
    """Check if the index has any unnamed columns."""
    return any(name is None for name in index.names)


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


def _assert_index_vs_cols_disjoint(df: pd.DataFrame) -> None:
    """Assert that the index names are disjoint from the column names of the data frame."""
    index = df.index
    column_name_overlap = set(df.columns).intersection(index.names)
    if column_name_overlap:
        msg = "Data frame index column names match the names of non-index columns"
        raise ValueError(f"{msg}: {column_name_overlap}.")


def unset(df: pd.DataFrame, require_names: bool = True) -> pd.DataFrame:
    """Safely unset the index.

    Details:

    * This is more an "unset" than a "reset" in the sense that it makes the index as trivial as possible. If we
      could totally remove the index of the data frame, that's what this function would do, but the unnamed
      range index is the next closest thing.
    * This is "safe" in the sense that it will not drop any existing data encoded in the index. (We assume that an
      unnamed range index does not count as "data" in this context.) Existing index columns get converted to
      regular columns in the data frame.

    Args:
        df: The data frame to unset the index on.
        require_names: Whether to raise an error if the existing index is unnamed:

          * This setting is ignored whenever the index is an unnamed RangeIndex.
          * With require_names=False, a unnamed index typically is converted to a new column called "index".
          * Using require_names=True (default) forces users to declare how to handle an unnamed index. Either:

            * Call reset_index(drop=True) directly to drop the index instead of calling this function.
            * Set the name(s) of the index prior to calling this function.

    Raises:
        ValueError: If the data frame column names overlap with the index names.
        ValueError: If all of the following apply:
          - require_names is True (the default)
          - the index is unnamed
          - the index is not a trivial RangeIndex

    Returns:
        A copy of the input data frame with the index columns reset as regular columns. The new index is
        a simple RangeIndex.
    """
    if is_unnamed_range_index(df.index):
        return df
    if require_names and index_has_any_unnamed_col(df.index):
        raise ValueError(
            "At least one column of the index is unnamed while the index itself is not a RangeIndex. "
            "Please set the names of the index columns before calling unset, or just call reset_index(drop=True) "
            "directly."
        )
    return df.reset_index(drop=False, allow_duplicates=False)


@frozen
class Index:
    """A functional wrapper around pandas indexes.

    An instance of this class can be used to simplify the following types of operations:

    * Coercing an input data frame to have a particular index.
    * Enforcing or ensuring various index properties such as monotonicity or uniqueness.
    * Unset and reset indexes cleanly before and after operations that require columnar access to index columns.

    This class applies both for pandas.Index and MultiIndex objects to reduce the amount of special-casing needed based
    on the number of columns in the index.

    Example:
        .. code-block:: python

            # Set an index with sorting
            df = pd.DataFrame({"a": [1, 3, 2], "b": [4, 5, 6]})
            index = Index(names=["a"], sort=True)
            df = index(df)
            assert df.index.tolist() == [1, 2, 3]

            # Switch over to another column as the index
    """

    names: FrozenList = field(converter=FrozenList, validator=min_len(1))
    """The names of the columns of the index."""

    allow_null: bool = field(default=False, kw_only=True)
    """Whether to allow null values in the index. Applicable only for single-column indexes, since pandas does not
    support null values in a MultiIndex."""

    sort: bool = field(default=False, kw_only=True, validator=_validate_sort_vs_null)
    """Whether to sort the index."""

    require_unique: bool = field(default=True, kw_only=True)
    """Whether to require the index to be unique. E.g. if True, raise an error if the index is not unique."""

    def __call__(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Set the index on the data frame.

        Any named columns of the current df index that are not part of the new index will be converted to new
        data frame columns.

        Args:
            df: The data frame to set the index on.
            **kwargs: Additional arguments to pass to the set_index method.

        Raises:
            ValueError: If the verify_integrity kwarg is inconsistent with the require_unique attribute.
            ValueError: If there is overlap between the existing index column names and the column names.
        """
        # resolve the verify_integrity argument in a way that's not inconsistent with the require_unique attribute:
        verify_integrity = kwargs.get("verify_integrity", self.require_unique)
        if verify_integrity is not self.require_unique:
            raise ValueError("The verify_integrity argument must be consistent with the require_unique attribute.")
        kwargs["verify_integrity"] = self.require_unique

        # raise value error if there is overlap between the existing index column names and the column names:
        _assert_index_vs_cols_disjoint(df)

        # set the index, retaining the columns of any existing index as regular columns:
        if not is_unnamed_range_index(df.index):
            df = df.reset_index(drop=False)
        df = df.set_index(self.names, **kwargs)

        # check nulls if applicable
        if not self.allow_null:
            assert_no_nulls(index=df.index)

        # sort if required
        if self.sort:
            df = df.sort_index()
        return df

    def assert_equal_names(self, index: pd.Index) -> None:
        """Assert that names of the provided index match the names of this index."""
        if not index.names == self.names:
            expected = f"Expected: {self.names}"
            provided = f"Provided: {index.names}"
            raise ValueError(f"Index names mismatch:\n   {expected}\n   {provided}.")
