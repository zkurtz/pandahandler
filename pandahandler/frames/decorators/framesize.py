"""Function decorators for functions that accept a data frame and return a data frame."""

import functools
import logging
from typing import Callable, TypeAlias

import pandas as pd
from sigfig import round as sround

FunType: TypeAlias = Callable[[pd.DataFrame], pd.DataFrame]


def log_rowcount_change(
    logger: logging.Logger,
    level: int = logging.INFO,
    stacklevel: int = 2,
    allow_empty_input: bool = True,
    allow_empty_output: bool = True,
) -> Callable:
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

    Raises:
        ValueError: If the input data frame is empty and allow_empty_input is False.
        RuntimeError: If the output data frame is empty and allow_empty_output is False.
    """

    def decorator(func: FunType) -> FunType:
        def wrapper(df: pd.DataFrame, *args, **kwargs):
            # Don't bother to compute logging inputs if the logging level so high that nothing would get logged:
            if not logger.isEnabledFor(level):
                return func(df, *args, **kwargs)

            logfunc = functools.partial(logger.log, level=level, stacklevel=stacklevel)
            n_input = len(df)

            if not allow_empty_input and df.empty:
                raise ValueError(f"{func.__name__} received an empty data frame but allow_empty_input is False.")

            df = func(df, *args, **kwargs)
            n_output = len(df)

            if df.empty:
                if not allow_empty_output:
                    raise RuntimeError(f"{func.__name__} produced an empty data frame but allow_empty_output is False.")
                logfunc(msg=f"{func.__name__} returned an empty data frame.")
                return df

            n_delta = n_output - n_input
            if n_delta == 0:
                logfunc(msg=f"{func.__name__} did not affect the row count.")
                return df
            pct_delta = sround((n_delta / n_input) * 100, sigfigs=3, warning=False)
            delta_str = "down" if n_delta <= 0 else "up"
            logfunc(msg=f"{func.__name__} returned {n_output} rows, {delta_str} {abs(n_delta)} rows ({pct_delta}%).")
            return df

        return wrapper

    return decorator
