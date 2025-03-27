"""Tools for working with pandas indexes."""

from typing import Any, Mapping

import pandas as pd
from attr.validators import min_len
from attrs import field, frozen
from pandas.core.indexes.frozen import FrozenList

__all__ = [
    "Index",
    "is_unnamed_range_index",
    "unset",
]


def _get_dtypes(index: pd.Index) -> pd.Series:
    """Get the dtypes of the index column(s).

    The purpose of this function is to handle edge cases such as non-MultiIndex indices, where the dtypes attribute
    is not available.
    """
    if isinstance(index, pd.MultiIndex):
        return pd.Series(index.dtypes, index=index.names)
    return pd.Series(index.dtype, index=index.names)


def index_has_any_unnamed_col(index: pd.Index) -> bool:
    """Check if the index has any unnamed columns."""
    return any(name is None for name in index.names)


def is_unnamed_range_index(index: pd.Index) -> bool:
    """Assert that the index is trivial, i.e. equal to the default RangeIndex."""
    if index.name is not None:
        return False
    unnamed_range_index = pd.RangeIndex(len(index))
    return index.equals(unnamed_range_index)


def _validate_sort_vs_null(instance: "Index", _: Any, __: Any) -> None:
    if instance.allow_null and instance.sort:
        raise ValueError("`sorted=True` is not allowed with and `allow_null=True`.")


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
    try:
        return df.reset_index(drop=False, allow_duplicates=False)
    except ValueError as err:
        error = str(err)
        if "already exists" in error and "cannot insert" in error:
            raise ValueError(
                "A name of a column of the existing index matches a column that already exists in the data frame. "
                "Deduplicate your columns (including index columns) before attempting to unset or reset the index."
            ) from err
        raise


@frozen(kw_only=True)
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

    allow_null: bool = field(default=False)
    """Whether to allow null values in the index. Applicable only for single-column indexes, since pandas does not
    support null values in a MultiIndex."""

    sort: bool = field(default=False, validator=_validate_sort_vs_null)
    """Whether to sort the index."""

    require_unique: bool = field(default=True)
    """Whether to require the index to be unique. E.g. if True, raise an error if the index is not unique."""

    dtypes: Mapping[str, Any] | None = None
    """The data types of the index columns."""

    def _validate_dtypes(self, index: pd.Index, coerce_dtypes: bool = False) -> None:
        """Assert that the provided index complies with this index specification."""
        if not self.dtypes:
            if coerce_dtypes:
                raise ValueError("coerce_dtypes is True but dtypes is not specified.")
            return
        specified_dtypes = pd.Series(self.dtypes)
        existing_dtypes = _get_dtypes(index)
        if specified_dtypes.rename("").equals(existing_dtypes.rename("")):
            return
        # Since the types are not equal, let's raise a type error with details about which types are not matching:
        types_df = pd.DataFrame({"specified": specified_dtypes, "existing": existing_dtypes})
        mismatch_mask = types_df["specified"] != types_df["existing"]
        mismatch_df = types_df.loc[mismatch_mask]
        raise TypeError(f"Index dtypes mismatch:\n{mismatch_df}")

    def validate(self, index: pd.Index, coerce_dtypes: bool = False) -> None:
        """Assert that the provided index complies with this index specification."""
        if not index.names == self.names:
            raise ValueError(f"Index names mismatch: {index.names} != {self.names}.")
        # Validate dtypes
        self._validate_dtypes(index, coerce_dtypes)

        # Validate uniqueness
        if self.require_unique and index.has_duplicates:
            raise ValueError("The index has duplicate values.")

        # Validate no nulls
        if not self.allow_null and index.hasnans:
            raise ValueError("The index has null values.")

        # Validate sorting
        if self.sort and not index.is_monotonic_increasing:
            raise ValueError("The index is not sorted.")

    def __call__(self, df: pd.DataFrame, coerce_dtypes: bool = False, **kwargs: Any) -> pd.DataFrame:
        """Set the index on the data frame.

        Any named columns of the current df index that are not part of the new index will be converted to new
        data frame columns.

        Args:
            df: The data frame to set the index on.
            coerce_dtypes: Whether to coerce the types of the index columns to the specified dtypes.
            **kwargs: Additional arguments to pass to the set_index method.

        Raises:
            ValueError: If the verify_integrity kwarg is inconsistent with the require_unique attribute.
            ValueError: If there is overlap between the existing index column names and the column names.
            ValueError: If coerce_dtypes is True and self.dtypes is None.
            ValueError: If coerce_dtypes is False and the input df.types do not match self.dtypes.
        """
        # This is a no-op if the existing index already matches the index specification:
        try:
            self.validate(df.index, coerce_dtypes=coerce_dtypes)
            return df
        except ValueError:
            pass

        # resolve the verify_integrity argument in a way that's not inconsistent with the require_unique attribute:
        verify_integrity = kwargs.get("verify_integrity", self.require_unique)
        if verify_integrity is not self.require_unique:
            raise ValueError("The verify_integrity argument must be consistent with the require_unique attribute.")
        kwargs["verify_integrity"] = self.require_unique

        # unset any existing nontrivial index to retain the columns of any existing index as regular columns:
        df = unset(df)  # A no-op if the existing index is an unnamed RangeIndex

        # if specified, coerce the types of the index columns to the specified dtypes:
        if coerce_dtypes:
            if not self.dtypes:
                raise ValueError("coerce_dtypes is True but dtypes is not specified.")
            df = df.astype(self.dtypes)

        # set the index:
        df = df.set_index(self.names, **kwargs)

        # check nulls if applicable. Do this early, before sorting, to fail fast
        if not self.allow_null and df.index.hasnans:
            raise ValueError("The index has null values.")

        # sort if required
        if self.sort:
            df = df.sort_index()

        self.validate(df.index, coerce_dtypes=coerce_dtypes)
        return df

    def assert_equal_names(self, index: pd.Index) -> None:
        """Assert that names of the provided index match the names of this index."""
        if not index.names == self.names:
            expected = f"Expected: {self.names}"
            provided = f"Provided: {index.names}"
            raise ValueError(f"Index names mismatch:\n   {expected}\n   {provided}.")
