import logging

import pandas as pd
import pytest

from pandahandler.frames.decorators.framesize import log_rowcount_change


def test_log_rowcount_change(caplog: pytest.LogCaptureFixture):
    @log_rowcount_change
    def add_row(df: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([df, df], axis=0)

    @log_rowcount_change
    def no_change(df: pd.DataFrame) -> pd.DataFrame:
        return df

    @log_rowcount_change(level=logging.WARN)
    def filter_smalls(df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[df["a"] > 1]

    df = pd.DataFrame({"a": [1, 2, 3]})
    with caplog.at_level(logging.INFO):
        caplog.clear()
        add_row(df)
        assert caplog.text.startswith("INFO     test_framesize:framesize.py")
        assert caplog.text.endswith("add_row returned 6 rows, up 3 rows (100.0%).\n")

        caplog.clear()
        no_change(df)
        assert caplog.text.endswith("no_change did not affect the row count.\n")

    with caplog.at_level(logging.CRITICAL):
        # Since the decorator was defined only for level WARN, nothing should be logged at CRITICAL level:
        caplog.clear()
        filter_smalls(df)
        assert caplog.text == ""

    with caplog.at_level(logging.INFO):
        # But at INFO level, the decorator should of course log:
        caplog.clear()
        filter_smalls(df)
        assert caplog.text.startswith("WARNING  test_framesize:framesize.py")
        assert caplog.text.endswith("filter_smalls returned 2 rows, down 1 rows (-33.3%).\n")
