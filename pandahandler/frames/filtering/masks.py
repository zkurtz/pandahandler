"""Mask functions."""

import pandas as pd


def my_mask(df: pd.DataFrame) -> pd.Series:
    """Define a toy mask function."""
    return df["a"] > 1
