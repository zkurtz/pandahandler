import pandas as pd

from pandahandler.frames import columns


def test_list_categoricals():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", None],
            "c": [1.0, 2.0, 3.0],
        }
    )
    assert columns.list_categoricals(df.dtypes) == []
    assert columns.list_numerics(df.dtypes) == ["a", "c"]
    df = df.astype("category")
    assert columns.list_categoricals(df.dtypes) == list(df.columns)
