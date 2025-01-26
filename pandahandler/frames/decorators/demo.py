"""Illustrate the use of decorators.

Run demo as:
    python -m pandahandler.frames.decorators.demo
"""

import logging

import pandas as pd

from pandahandler.frames.filters import drop_if_any_null


def _filtering() -> None:
    df = pd.DataFrame({"a": [1, 2, None], "b": [None, 4, 5]})
    print(f"Original data frame:\n{df}")
    df_filtered = drop_if_any_null(df)
    print(f"Filtered data frame:\n{df_filtered}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Running filtering decorators demo:")
    _filtering()
