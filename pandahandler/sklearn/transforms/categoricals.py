"""Sklearn-compatible transformer to encode non-numeric columns as pandas categorical features."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass(frozen=True)
class SeriesEncoder:
    """Categorical encoding of the values for a pandas Series.

    Attributes:
        categories: A list of non-null categories.
        has_seen_null: Whether null values were seen in the training data.
    """

    categories: list[Any]
    has_seen_null: bool

    @classmethod
    def fit(cls, series: pd.Series) -> "SeriesEncoder":
        """Learn categorical codes of a data series.

        Args:
            series: The pandas Series to fit.

        Returns:
            A SeriesEncoder object with the unique categories and null presence.
        """
        series = series.astype("category")
        values = series.cat.categories.to_list()
        return cls(
            categories=values,
            has_seen_null=bool(series.isnull().any()),
        )

    def __call__(self, series: pd.Series) -> pd.Series:
        """Encode a series as categorical.

        This encoder maintains a distinction between null values and "never-before-seen" values:
        - Pandas maps null values to -1.
        - Pandas also maps any never-before-seen values to -1, which is loses information. Instead, we identify
          such values and postpend them to the list of categories, so that they are encoded as new categories.

        This distinction can matter for certain ML algorithms, such as XGBoost. See also
        https://github.com/microsoft/LightGBM/issues/6908.
        """
        new_values = set(series.dropna()) - set(self.categories)
        categories = self.categories + sorted(list(new_values))
        categorical = pd.Categorical(
            series,
            categories=categories,
            ordered=False,
        )
        return pd.Series(categorical, index=series.index, name=series.name)


def infer_categoricals(df: pd.DataFrame) -> list[str]:
    """Identify columns that should be coded as categorical."""
    ret = df.select_dtypes(exclude=[np.number]).columns.to_list()
    assert isinstance(ret, list), "Expected a list of column names."
    return ret


class Encoder(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer to encode non-numeric columns as pandas categorical features.

    This stores category mappings during fit and applies them during transform, ensuring consistency in the
    value-to-code mapping between training and scoring.
    """

    def __init__(self, specified_columns: list[str] | None = None) -> None:
        """Initialize the Encoder.

        Args:
            specified_columns: Optional list of column names to be encoded as pandas categorical. If not specified,
                the fit method will automatically detect non-numeric columns and treat them all as categorical.
        """
        self.specified_columns = specified_columns
        self.encoders: dict[str, SeriesEncoder] = {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> "Encoder":
        """Fit the encoder to the data.

        Learns the unique categories for each specified categorical column and saves the category codes.

        Args:
            X: The data to fit the encoder on.
            y: Ignored, present for API consistency.

        Raises:
            ValueError: If the encoder has already been fitted.
        """
        del y
        if self.encoders:
            raise ValueError("Encoder has already been fitted. Please create a new instance.")
        categorical_columns: list[str] = self.specified_columns or infer_categoricals(X)
        for col in categorical_columns:
            self.encoders[col] = SeriesEncoder.fit(X[col])  # pyright: ignore[reportArgumentType]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the category encodings to new data.

        Args:
            X: The data to transform

        Returns: The transformed data with consistent categorical encodings.
        """
        X = X.copy()
        columns = list(self.encoders)
        encoded_cols: list[pd.Series] = [
            encoder(X[col])  # pyright: ignore[reportArgumentType]
            for col, encoder in self.encoders.items()
        ]
        X[columns] = pd.concat(encoded_cols, axis=1)
        return X
