"""Illustrate the use of decorators.

Run demo as:
    python -m pandahandler.frames.decorators.demo

Expected log:
    INFO:__main__:drop_if_any_null returned 2 rows, down 1 rows (-33.3%).
    WARNING:__main__:local_filter returned 1 rows, down 1 rows (-50.0%).
"""

import logging

import pandas as pd

from pandahandler.frames.decorators.framesize import log_rowcount_change
from pandahandler.frames.filtering.filters import drop_if_any_null


@log_rowcount_change(level=logging.WARNING)
def local_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Local filter that drops rows with any null values."""
    return df.loc[df["a"] < 2]


def _filtering() -> None:
    df = pd.DataFrame({"a": [1, 2, None], "b": [1, 4, 5]})
    df = drop_if_any_null(df)
    _ = local_filter(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    print("Running filtering decorators demo:")
    _filtering()
