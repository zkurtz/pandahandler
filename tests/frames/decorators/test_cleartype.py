import pandas as pd
import pytest

from pandahandler.frames.decorators.cleartype import assert_returns_dataframe


def test_assert_returns_dataframe():
    @assert_returns_dataframe
    def return_df(df: pd.DataFrame) -> pd.DataFrame:
        return df

    df = pd.DataFrame({"a": [1, 2, 3]})
    return_df(df)

    @assert_returns_dataframe  # pyright: ignore
    def return_zero(df: pd.DataFrame):  # pyright: ignore
        del df
        return 0

    with pytest.raises(AssertionError, match="did not return a DataFrame"):
        return_zero(df)
