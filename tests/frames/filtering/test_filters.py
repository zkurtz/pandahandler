import logging

import pandas as pd
import pytest

from pandahandler.frames.filtering.filters import drop_if_any_null


def test_drop_if_any_null(caplog: pytest.LogCaptureFixture) -> None:
    df = pd.DataFrame({"a": [1, 2, None], "b": [4, 5, 6]})
    with caplog.at_level(logging.INFO):
        result = drop_if_any_null(df)
        assert "drop_if_any_null returned 2 rows, down 1 rows (-33.3%)" in caplog.text
        pd.testing.assert_frame_equal(result, df.dropna())
