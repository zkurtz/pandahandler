import logging

import pandas as pd
import pytest

from pandahandler.frames.decorators.framesize import log_rowcount_change

logger = logging.getLogger(__name__)


def test_log_rowcount_change(caplog: pytest.LogCaptureFixture):
    @log_rowcount_change
    def double(df: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([df, df], axis=0)

    @log_rowcount_change(logger=logging.getLogger("my_test"))
    def add_row(df: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([df, df.iloc[[0]]], axis=0)

    @log_rowcount_change(logger=logger)
    def no_change(df: pd.DataFrame) -> pd.DataFrame:
        return df

    @log_rowcount_change(level=logging.WARN)
    def filter_smalls(df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[df["a"] > 1]

    @log_rowcount_change()
    def drop_all_rows(df: pd.DataFrame) -> pd.DataFrame:
        return df.iloc[0:0]

    @log_rowcount_change(allow_empty_input=False, allow_empty_output=False)
    def drop_all_rows_strict(df: pd.DataFrame) -> pd.DataFrame:
        ret = df.iloc[0:0]
        return ret

    df = pd.DataFrame({"a": [1, 2, 3]})

    with caplog.at_level(logging.INFO):
        caplog.clear()
        double(df)
        assert caplog.text.startswith(f"INFO     {__name__}:test_framesize.py")
        assert caplog.text.endswith("double returned 6 rows, up 3 rows (100.0%).\n")

        caplog.clear()
        no_change(df)
        assert caplog.text.endswith("no_change did not affect the row count.\n")

        caplog.clear()
        df2 = pd.concat([df, df.iloc[[0]]], axis=0)
        add_row(df2)
        assert caplog.text.startswith("INFO     my_test:test_framesize.py")
        assert caplog.text.endswith("add_row returned 5 rows, up 1 rows (25.0%).\n")

    with caplog.at_level(logging.CRITICAL):
        # Since the decorator was defined only for level WARN, nothing should be logged at CRITICAL level:
        caplog.clear()
        filter_smalls(df)
        assert caplog.text == ""

    with caplog.at_level(logging.INFO):
        # But at INFO level, the decorator should of course log:
        caplog.clear()
        filter_smalls(df)
        assert caplog.text.startswith(f"WARNING  {__name__}")
        assert caplog.text.endswith("filter_smalls returned 2 rows, down 1 rows (-33.3%).\n")

    with caplog.at_level(logging.INFO):
        msg = "drop_all_rows_strict received an empty data frame but allow_empty_input is False"
        with pytest.raises(ValueError, match=msg):
            drop_all_rows_strict(df.iloc[0:0])

        msg = "drop_all_rows_strict produced an empty data frame but allow_empty_output is False"
        with pytest.raises(RuntimeError, match=msg):
            drop_all_rows_strict(df)

    with caplog.at_level(logging.INFO):
        caplog.clear()
        drop_all_rows(df)
        assert caplog.text.endswith("drop_all_rows returned an empty data frame.\n")
