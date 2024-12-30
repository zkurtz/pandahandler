"""Function decorators for functions that accept a data frame and return a data frame."""

import inspect
import logging
from typing import Callable, TypeAlias

import pandas as pd
from sigfig import round as sround

FunType: TypeAlias = Callable[[pd.DataFrame], pd.DataFrame]


def _get_default_logger() -> logging.Logger:
    """Get the logger for the caller's calling module."""
    stack = inspect.stack()
    # here = stack[0]
    # caller = stack[1]
    # but we want the caller of the caller:
    context_frame = stack[2]
    module = inspect.getmodule(context_frame[0])
    assert module is not None, "Could not determine the calling module"
    return logging.getLogger(module.__name__)


def log_rowcount_change(*args: FunType, logger: logging.Logger | None = None, level: int = logging.INFO) -> Callable:
    """Log the change in the number of rows of a data frame processed by func.

    Args:
        args: If provided, the function to decorate. If not provided, the decorator is being called with arguments,
            as a decorator factory, so it returns the actual decorator.
        logger: The logger to use. If None, the logger for the calling module is used.
        level: The logging level to use, defaulting to INFO.
    """
    logger = logger or _get_default_logger()

    def decorator(func: FunType) -> FunType:
        def wrapper(df: pd.DataFrame, *args, **kwargs):
            # Don't bother to compute logging inputs if the logging level so high that nothing would get logged:
            if not logger.isEnabledFor(level):
                return func(df, *args, **kwargs)

            n_input = len(df)
            df = func(df, *args, **kwargs)
            n_output = len(df)
            n_delta = n_output - n_input
            if n_delta == 0:
                logger.log(level=level, msg=f"{func.__name__} did not affect the row count.")
                return df
            pct_delta = sround((n_delta / n_input) * 100, sigfigs=3)
            delta_str = "down" if n_delta <= 0 else "up"
            msg = f"{func.__name__} returned {n_output} rows, {delta_str} {abs(n_delta)} rows ({pct_delta}%)."
            logger.log(level=level, msg=msg)
            return df

        return wrapper

    if args:
        if len(args) > 1:
            raise ValueError("There should be at most one unnamed argument")
        assert len(args) == 1
        func = args[0]
        if not callable(func):
            raise ValueError("When called without named args, the sole argument should be a callable")
        return decorator(func=func)
    return decorator
