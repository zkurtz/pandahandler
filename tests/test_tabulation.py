import numpy as np
import pandas as pd

from pandahandler.tabulation import tabulate


def test_tabulate():
    # Define a series of categorical data
    data = ["a", "a", "b", "c", "c", "c", None, None, None]
    series = pd.Series(data, name="test_name").astype("category")
    tb = tabulate(series)

    # Validate counts
    expected_counts = pd.Series({"a": 2, "b": 1, "c": 3, np.nan: 3}, name="count")
    expected_counts.index = expected_counts.index.astype("category")
    expected_counts.index.name = "test_name"
    pd.testing.assert_series_equal(expected_counts, tb.counts)

    # Validate metadata
    assert tb.n_values == len(data)
    assert tb.n_distinct == len(set(data))
    assert tb.name == "test_name"

    # Validate rates
    expected_rates = pd.Series(expected_counts / len(data), name="rate")
    expected_rates.index = expected_rates.index.astype("category")
    pd.testing.assert_series_equal(expected_rates, tb.rates)

    # Validate `select` method
    sub_tb = tb.select(keep=["a", np.nan])
    expected_counts = pd.Series({"a": 2, np.nan: 3}, name="count")
    expected_counts.index = expected_counts.index.astype("category")
    expected_counts.index.name = "test_name"
    pd.testing.assert_series_equal(expected_counts, sub_tb.counts)
    assert sub_tb.n_values == 5
    assert sub_tb.n_distinct == 2
    assert sub_tb.name == "test_name"
    expected_rates = pd.Series(expected_counts / sub_tb.n_values, name="rate")
    expected_rates.index = expected_rates.index.astype("category")
    expected_rates.index.name = "test_name"
    pd.testing.assert_series_equal(expected_rates, sub_tb.rates)
