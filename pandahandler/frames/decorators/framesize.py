"""Function decorators for functions that accept a data frame and return a data frame."""

import functools
import logging
from typing import Callable

import pandas as pd
from sigfig import round as sround

from pandahandler.frames.constants import DataframeToDataframe


def _get_function_name(func: DataframeToDataframe, *args, **kwargs) -> str:
    """Default method to describe the wrapped function."""
    del args
    del kwargs
    return func.__name__


def log_rowcount_change(
    logger: logging.Logger,
    level: int = logging.INFO,
    stacklevel: int = 2,
    allow_empty_input: bool = True,
    allow_empty_output: bool = True,
    describe_func: Callable[..., str] = _get_function_name,
) -> Callable[[DataframeToDataframe], DataframeToDataframe]:
    """Log the change in the number of rows of a data frame processed by func.

    Args:
        args: If provided, the function to decorate. If not provided, the decorator is being called with arguments,
            as a decorator factory, so it returns the actual decorator.
        logger: The logger to use. If None, the logger for the calling module is used.
        level: The logging level to use, defaulting to INFO.
        stacklevel: Passed into the logger.log call:
            1: The log message shows the line number of the logging call inside the decorator itself.
            2 (default): the log message shows the line number of the call to the decorated function.
        allow_empty_input: If False, raise an exception if the input data frame is empty.
        allow_empty_output: If False, raise an exception if the output data frame is empty.
        describe_func: A function that takes the decorated function and its arguments and returns a string description
            of the function. The default implementation returns the function's name.

    Raises:
        ValueError: If the input data frame is empty and allow_empty_input is False.
        RuntimeError: If the output data frame is empty and allow_empty_output is False.
    """

    def decorator(func: DataframeToDataframe) -> DataframeToDataframe:
        @functools.wraps(func)
        def wrapper(df: pd.DataFrame, *args, **kwargs):
            # Don't bother to compute logging inputs if the logging level so high that nothing would get logged:
            if not logger.isEnabledFor(level):
                return func(df, *args, **kwargs)

            logfunc = functools.partial(logger.log, level=level, stacklevel=stacklevel)
            n_input = len(df)

            func_name = describe_func(func, *args, **kwargs)
            if not allow_empty_input and df.empty:
                raise ValueError(f"{func_name} received an empty data frame but allow_empty_input is False.")

            df = func(df, *args, **kwargs)
            n_output = len(df)

            if df.empty:
                if not allow_empty_output:
                    raise RuntimeError(f"{func_name} produced an empty data frame but allow_empty_output is False.")
                logfunc(msg=f"{func_name} returned an empty data frame.")
                return df

            n_delta = n_output - n_input
            if n_delta == 0:
                logfunc(msg=f"{func_name} did not affect the row count.")
                return df
            pct_delta = sround((n_delta / n_input) * 100, sigfigs=3, warning=False)
            delta_str = "down" if n_delta <= 0 else "up"
            logfunc(msg=f"{func_name} returned {n_output} rows, {delta_str} {abs(n_delta)} rows ({pct_delta}%).")
            return df

        return wrapper

    return decorator
