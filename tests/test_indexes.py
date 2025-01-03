"""Test index utilities."""

import pandas as pd

from pandahandler.indexes import is_unnamed_range_index


def test_is_unnamed_range_index():
    """Test that is_unnamed_range_index works as expected."""

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
