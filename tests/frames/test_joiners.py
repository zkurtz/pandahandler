import pandas as pd
import pytest

from pandahandler.frames import joiners


def test_safe_hstack():
    # empty input should raise an error:
    with pytest.raises(ValueError, match="At least one data frame must be provided."):
        joiners.safe_hstack([])

    # frames with different indexes should raise an error:
    df1 = pd.DataFrame({"a": [1, 2, 3]}, index=pd.Index(["x", "y", "z"]))
    df2 = pd.DataFrame({"a": [4, 5, 6]}, index=pd.Index(["x", "y", "w"]))
    with pytest.raises(ValueError, match="All data frames must share the same index"):
        joiners.safe_hstack([df1, df2])

    # frames with overlapping columns should raise an error:
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = pd.DataFrame({"b": [7, 8, 9], "c": [10, 11, 12]})
    with pytest.raises(ValueError, match="Column names must be unique across data frames"):
        joiners.safe_hstack([df1, df2])

    # happy path:
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = pd.DataFrame({"c": [7, 8, 9], "d": [10, 11, 12]})
    df = joiners.safe_hstack([df1, df2])
    expected_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [10, 11, 12]})
    pd.testing.assert_frame_equal(df, expected_df)

    # should also be fine with a single frame:
    df = joiners.safe_hstack([df1])
    pd.testing.assert_frame_equal(df, df1)
