"""Tools to clarify the type of data-frame-valued variables."""

from typing import Callable

import pandas as pd

from pandahandler.frames.types import DfTransform


def assert_returns_dataframe(func: DfTransform) -> Callable:
    """Decorator to assert that a function returns a DataFrame.

    This uses an assertion rather than an exception because the primary intent is for the programmer to use this as
    clarification for a static analysis type checker, not validation of user input or other run time behavior.

    Args:
        func: The function to decorate.
    """

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        assert isinstance(result, pd.DataFrame), f"{func.__name__} did not return a DataFrame"
        return result

    return wrapper
