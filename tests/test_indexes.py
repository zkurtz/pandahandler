"""Test index utilities."""

import pandas as pd
import pytest

from pandahandler.indexes import Index, is_unnamed_range_index, unset


def test_is_unnamed_range_index():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert is_unnamed_range_index(df.index)

    df = pd.DataFrame({"a": [1, 2, 3]}, index=pd.RangeIndex(3))
    assert is_unnamed_range_index(df.index)

    df = pd.DataFrame({"a": [1, 2, 3]}, index=pd.RangeIndex(3, name=None))
    assert is_unnamed_range_index(df.index)

    # name index to get False:
    df = pd.DataFrame({"a": [1, 2, 3]}, index=pd.RangeIndex(3, name="foo"))
    assert not is_unnamed_range_index(df.index)

    # obvious not-range index
    df = pd.DataFrame({"a": [1, 2, 3]}, index=pd.Index(["x", "y", "z"]))
    assert not is_unnamed_range_index(df.index)

    # multiindex should fail:
    df = pd.DataFrame({"a": [1, 2]}, index=pd.MultiIndex.from_tuples([("a", "b"), ("c", "d")]))
    assert not is_unnamed_range_index(df.index)


def test_index():
    df = pd.DataFrame(
        {
            "cats": ["siamese", "little", "persian"],
            "timestamp": ["2021-01-01", "2021-01-02", None],
        },
        index=pd.RangeIndex(3, name="idcol"),
    )
    df["cats"] = df["cats"].astype("category")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    original_df = df.copy()

    CatIndex = Index(["cats"], sort=True)
    TimestampIndex = Index(["timestamp"])
    NullTimestampIndex = Index(["timestamp"], allow_null=True)
    IdIndex = Index(["idcol"], sort=True)
    CatTimeIndex = Index(["cats", "timestamp"], allow_null=True)

    df = CatIndex(df)
    assert df.index.names == ["cats"]
    assert df.index.is_monotonic_increasing
    assert df.index.is_unique

    # Any index operation on a df where a column name of the existing index matches the name of any non-index column
    # should raise an error:
    with pytest.raises(ValueError, match="index column names match the names of non-index columns: {'cats'}"):
        dfx = df.copy()
        dfx["cats"] = dfx.index.to_numpy()
        TimestampIndex(dfx)

    df = NullTimestampIndex(df)
    assert df.index.names == ["timestamp"]

    with pytest.raises(ValueError, match="Null values are not allowed in the index."):
        df = TimestampIndex(df)

    df = CatTimeIndex(df)
    assert df.index.names == ["cats", "timestamp"]

    pd.testing.assert_frame_equal(IdIndex(df), original_df)


def test_unset() -> None:
    df = pd.DataFrame(
        {
            "cats": ["siamese", "little", "persian"],
            "timestamp": ["2021-01-01", "2021-01-02", None],
        }
    )
    df.index = df["cats"]  # pyright: ignore

    # Any index operation on a df where a column name of the existing index matches the name of any non-index column
    # should raise an error:
    with pytest.raises(ValueError, match="index column names match the names of non-index columns: {'cats'}"):
        unset(df)

    # But we can rename the index and then unset it
    df.index.name = "cats_copy"
    df = unset(df)

    # Running the operation again should not change anything since df already has a range index:
    dfx = unset(df)
    pd.testing.assert_frame_equal(dfx, df)
