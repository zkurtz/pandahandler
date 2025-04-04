"""Univariate data tabulation."""

from functools import cached_property
from typing import Hashable, Iterable

import pandas as pd
from attrs import define
from typing_extensions import Self

__all__ = [
    "Tabulation",
    "tabulate",
]

NAME = "name"
RATE = "rate"


@define(kw_only=True)
class Tabulation:
    """Counts and associated metadata for a univariate data set."""

    counts: pd.Series
    """The table of counts."""

    name: str | None = None
    """A name for the data set being tabulated."""

    n_values: int
    """The number of values in the input series."""

    n_distinct: int
    """The number of distinct values in the input series (i.e. the number of rows in `df`)."""

    def __attrs_post_init__(self) -> None:
        """Data validation.

        Raises:
            ValueError: If counts index is not monotonic increasing, after removing nulls.
        """
        isnull_mask = pd.isnull(self.counts.index)
        nonnull_counts = self.counts[~isnull_mask]
        assert isinstance(nonnull_counts, pd.Series), "Expected counts to be a Series."
        if not nonnull_counts.index.is_monotonic_increasing:
            raise ValueError("Expected counts.index to be monotonic increasing (ignoring nulls).")

    def select(self, keep: Iterable[Hashable]) -> Self:
        """Derive a new tabulation that includes only a subset of the distinct values.

        Args:
            keep: The distinct values to include.

        Raises:
            KeyError: If any of the named index values are not present in index of self.counts.
        """
        keep_idxs = pd.Index(keep)
        missing = set(keep_idxs.difference(self.counts.index))
        if missing:
            raise KeyError(f"Named index values not found in index: {missing}")
        cls = type(self)
        counts = self.counts.loc[keep_idxs].copy()
        if isinstance(counts.index.dtype, pd.CategoricalDtype):
            idx_name = self.counts.index.name
            _cat_idx = pd.CategoricalIndex(counts.index)
            _setter = _cat_idx.set_categories  # pyright: ignore[reportAttributeAccessIssue]
            counts.index = _setter(keep_idxs.dropna())
            counts.index.name = idx_name
        return cls(
            counts=counts,
            name=self.name,
            n_values=counts.sum(),
            n_distinct=len(keep_idxs),
        )

    @cached_property
    def rates(self) -> pd.Series:
        """Generate the empirical multinomial probabilities."""
        return pd.Series(self.counts / self.n_values, name=RATE)


def tabulate(data: Iterable[Hashable], name: str | None = None, dropna: bool = False) -> Tabulation:
    """Create a tabulation of data.

    Args:
        data: The data to tabulate.
        name: A name for the data set being tabulated. Defaults to None, but inherits the name of the input data if
            it has a `name` attribute.
        dropna: Whether to drop NA values before tabulating. Defaults to False.
    """
    series = pd.Series(data)  # pyright: ignore
    name = getattr(data, NAME, None)
    counts = series.value_counts(sort=False, dropna=dropna)
    counts = counts.sort_index()
    return Tabulation(
        counts=counts,
        name=name,
        n_values=series.size,
        n_distinct=counts.size,
    )
