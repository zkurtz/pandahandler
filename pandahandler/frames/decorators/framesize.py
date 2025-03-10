"""Function decorators for functions that accept a data frame and return a data frame."""

import functools
import inspect
import logging
from typing import Any, Callable, cast

import pandas as pd
from sigfig import round as sround

from pandahandler.frames.constants import DataframeToDataframe

__all__ = ["log_rowcount_change"]


def _get_function_name(func: DataframeToDataframe, *args, **kwargs) -> str:
    """Default method to describe the wrapped function."""
    del args
    del kwargs
    return func.__name__


def _get_stack_modules(max_num_levels: int) -> list[str]:
    """Get the list of module names from the call stack.

    Args:
        max_num_levels: Number of levels to traverse up the call stack.

    Returns:
        list: Module names in order from current frame (index 0) up to max_num_levels.
    """
    stack = inspect.stack()
    num_levels = min(max_num_levels, len(stack))
    stack = stack[:num_levels]
    modules = []
    try:
        for item in stack:
            module = inspect.getmodule(item.frame)
            module_name = module.__name__ if module else "<unknown>"
            modules.append(module_name)
    finally:
        # Explicitly delete references to frame objects to avoid reference cycles
        del stack

    return modules


def log_rowcount_change(
    func: DataframeToDataframe | None = None,
    *,
    logger: logging.Logger | None = None,
    level: int = logging.INFO,
    stacklevel: int = 2,
    allow_empty_input: bool = True,
    allow_empty_output: bool = True,
    describe_func: Callable[..., str] = _get_function_name,
    # Return type should be Callable[[DataframeToDataframe], DataframeToDataframe] | DataframeToDataframe, but
    # pyright is not buying it.
) -> Any:
    """Log the change in the number of rows of a data frame processed by func.

    This decorator can be used with or without arguments:

        .. code-block:: python

            @log_rowcount_change
            def my_func(df): ...

            @log_rowcount_change(level=logging.DEBUG)
            def my_func(df): ...

    Args:
        func: The function to decorate (when used without parentheses).
        logger: The logger to use. If None, the logger for the module calling the decorated function is used.
        level: The logging level to use, defaulting to INFO. Note that this sets the logging level for your decorator,
            NOT the logger. For example, if you set level=logging.DEBUG, the decorator will log only when the logger is
            set to DEBUG or lower.
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

    def create_decorator(func: DataframeToDataframe) -> DataframeToDataframe:
        @functools.wraps(func)
        def wrapper(df: pd.DataFrame, *args, **kwargs):
            # Resolve what logger to use
            if logger:
                _logger = logger
            else:
                modules_stack = _get_stack_modules(stacklevel + 1)
                _logger = logging.getLogger(modules_stack[-1])

            # Don't bother to compute logging inputs if the logging level so high that nothing would get logged:
            if not _logger.isEnabledFor(level):
                return func(df, *args, **kwargs)

            logfunc = functools.partial(_logger.log, level=level, stacklevel=stacklevel)
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

        return cast(DataframeToDataframe, wrapper)

    # Check if called directly with a function
    if func is not None:
        return create_decorator(func)

    # Otherwise, return a decorator
    return create_decorator
