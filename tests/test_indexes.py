"""Test index utilities."""

import numpy as np
import pandas as pd
import pytest

from pandahandler.indexes import (
    DTypesError,
    DuplicateValuesError,
    Index,
    NullValuesError,
    index_has_any_unnamed_col,
    is_unnamed_range_index,
    unset,
)


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
    # test coerce_dtypes:
    df = pd.DataFrame({"a": [1, 2, 3]}, index=pd.Index(["x", "y", "z"], name="letters"))
    # expect an error if coerce_dtypes is False while the dtypes are not conforming:
    with pytest.raises(DTypesError, match="Index dtypes mismatch:\n"):
        index = Index(names=["letters"], dtypes={"letters": float})
        _ = index(df)
    # Create index with category dtype
    index = Index(names=["a"], dtypes={"a": "category"})
    df = index(df, coerce_dtypes=True)
    assert df.index.dtype == "category"
    # expect an error if coerce_dtypes is True but dtypes is not specified:
    index = Index(names=["a"])
    with pytest.raises(ValueError, match="coerce_dtypes is True but dtypes is not specified."):
        df = index(df, coerce_dtypes=True)

    # test coercion involving datetimes that are formatted as strings:
    df = pd.DataFrame(
        {"dates": ["2021-01-01", "2021-01-02", "2021-01-03"]},
        index=pd.Index([1, 2, 3.0], name="numbers"),
    )
    index = Index(
        names=["dates", "numbers"],
        dtypes={"numbers": np.int64, "dates": np.dtype("datetime64[ns]")},
    )
    dfc = index(df, coerce_dtypes=True)
    index._validate_dtypes(index=dfc.index)

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

    CatIndex = Index(names=["cats"], sort=True)
    TimestampIndex = Index(names=["timestamp"])
    NullTimestampIndex = Index(names=["timestamp"], allow_null=True)
    IdIndex = Index(names=["idcol"], sort=True)
    CatTimeIndex = Index(names=["cats", "timestamp"], allow_null=True)
    CatTimeStrictIndex = Index(names=["cats", "timestamp"], allow_null=False, require_unique=True)

    df = CatIndex(df)
    assert df.index.names == ["cats"]
    assert df.index.is_monotonic_increasing
    assert df.index.is_unique

    # Any index operation on a df where a column name of the existing index matches the name of any non-index column
    # should raise an error:
    with pytest.raises(ValueError, match="a column of the existing index matches a column that already exists"):
        dfx = df.copy()
        dfx["cats"] = dfx.index.to_numpy()
        TimestampIndex(dfx)

    df = NullTimestampIndex(df)
    assert df.index.names == ["timestamp"]
    with pytest.raises(NullValuesError, match="The index has null values."):
        df = TimestampIndex(df)

    df = CatTimeIndex(df)
    assert df.index.names == ["cats", "timestamp"]
    with pytest.raises(NullValuesError, match="The index has null values."):
        CatTimeStrictIndex(df)

    pd.testing.assert_frame_equal(IdIndex(df), original_df)

    # Test Index.validate():
    index = Index(names=["a"], sort=True)
    pd_index = pd.Index([1, 3, 2], name="a")
    with pytest.raises(ValueError, match="The index is not sorted."):
        index.validate(pd_index)

    # Test duplicate values error:
    pd_index = pd.Index([1, 3, 2, 3], name="a")
    with pytest.raises(DuplicateValuesError, match="The index has duplicate values"):
        index.validate(pd_index)


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
    with pytest.raises(ValueError, match="a column of the existing index matches a column that already exists"):
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
