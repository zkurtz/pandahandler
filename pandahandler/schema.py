"""Tools to learn or coerce the schema of an open-ended input data frame."""

from dataclasses import dataclass
from functools import cached_property
from typing import Hashable

import pandas as pd
from typing_extensions import Self

from pandahandler.frames import columns


def categorize_non_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Categorize columns that are neither categorical nor numeric."""
    categoricals = columns.list_categoricals(df.dtypes)
    numerics = columns.list_numerics(df.dtypes)
    num_or_cat = set(categoricals + numerics)
    others = [col for col in df if col not in num_or_cat]
    df[others] = df[others].astype("category")
    return df


@dataclass(frozen=True)
class Schema:
    """Using and applying a data frame's schema information.

    The primary intended use case is in open-world data exploration, where the schema of the input data is not known in
    advance. If you know the schema in advance, consider using a more declarative approach such as pandera.

    Note that the categorical encodings attribute is important information that's not traditionally captured in
    "schema" information, although it is important for encoding any new data in a way that's consistent with training
    data for scoring in machine learning applications.

    Attributes:
        types_: The data types of the columns.
        categorical_encodings: The categories of the categorical columns. The keys are the column names (for columns
            of categorical type) and the values are index objects expressing the integeger-category mappings defining
            that column's categorical encoding.
    """

    types_: pd.Series
    categorical_encodings: dict[Hashable, pd.Index]

    def __post_init__(self):
        """Run consistency checks."""
        categoricals = set(self.categoricals)
        numerics = set(self.numerics)
        others = set(self.others)
        assert not categoricals & numerics, "Categorical and numeric columns overlap."
        assert not categoricals & others, "Categorical and other columns overlap."
        assert not numerics & others, "Numeric and other columns overlap."
        assert set(self.types_.index) == categoricals | numerics | others, "Column names are inconsistent."

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Self:
        """Create a ColumnTypes object from a data frame."""
        categoricals = columns.list_categoricals(df.dtypes)
        return cls(
            types_=df.dtypes,
            categorical_encodings={col: df[col].cat.categories for col in categoricals},
        )

    @cached_property
    def categoricals(self) -> list[Hashable]:
        """Return the names of categorical columns."""
        return list(self.categorical_encodings) if self.categorical_encodings else []

    @cached_property
    def numerics(self) -> list[Hashable]:
        """Return the names of numeric columns."""
        return columns.list_numerics(self.types_)

    @cached_property
    def others(self) -> list[Hashable]:
        """Return the names of columns that are neither categorical nor numeric."""
        categorical_or_numeric = set(self.categoricals + self.numerics)
        return [col for col in self.types_.index if col not in categorical_or_numeric]

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce the data frame to the schema."""
        df = df.copy()
        other_types = self.types_.loc[pd.Index(self.others)]
        df[self.others] = df[self.others].astype(other_types)

        numeric_types = self.types_.loc[pd.Index(self.numerics)]
        df[self.numerics] = df[self.numerics].astype(numeric_types)

        for col in self.categoricals:
            df[col] = df[col].astype("category")
            df[col] = df[col].cat.set_categories(self.categorical_encodings[col])

        if not df.dtypes.equals(self.types_):
            raise ValueError("The schema coercion failed.")

        return df
