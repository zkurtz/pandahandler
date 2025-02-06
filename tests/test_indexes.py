"""Test index utilities."""

import pandas as pd
import pytest

from pandahandler.indexes import Index, index_has_any_unnamed_col, is_unnamed_range_index, unset


def test_index_has_any_unnamed_col():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert index_has_any_unnamed_col(df.index)

    df.index.name = "blah"
    assert not index_has_any_unnamed_col(df.index)

    # multi-index case:
    df = pd.DataFrame({"a": [1, 2]}, index=pd.MultiIndex.from_tuples([("a", "b"), ("c", "d")]))
    assert index_has_any_unnamed_col(df.index)
    df.index.names = ["blah", None]
    assert index_has_any_unnamed_col(df.index)
    df.index.names = ["blah", "blah"]
    assert not index_has_any_unnamed_col(df.index)


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
    with pytest.raises(ValueError, match="cannot insert cats, already exists"):
        unset(df)

    # But we can rename the index and then unset it
    df.index.name = "cats_copy"
    df = unset(df)

    # Running the operation again should not change anything since df already has a range index:
    dfx = unset(df)
    pd.testing.assert_frame_equal(dfx, df)

    # Now let's test the require_names parameter: expecting a value error if the index is unnamed but not a RangeIndex:
    df = pd.DataFrame(
        {
            "cats": ["siamese", "little", "persian"],
            "timestamp": ["2021-01-01", "2021-01-02", None],
        },
        index=pd.Index(["a", "b", "c"]),
    )
    msg = "At least one column of the index is unnamed while the index itself is not a RangeIndex"
    with pytest.raises(ValueError, match=msg):
        unset(df)
    # but we can override that:
    dfx = unset(df, require_names=False)
    assert "index" in dfx
