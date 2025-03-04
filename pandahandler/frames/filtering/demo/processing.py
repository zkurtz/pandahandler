"""Illustrate usage of filtering tools.

Shows three ways to apply the same mask as a data frame filter:
  1. Execute the mask function to get a boolean series; then apply the mask using masktools.apply_mask.
  2. Apply masktools.apply_mask directly using the mask function.
  3. Transform the mask function into a filter using masktools.as_filter, and apply the filter to the data frame.

Run as:
    python -m pandahandler.frames.filtering.demo
"""

import logging

import pandas as pd

from pandahandler.frames.filtering import filters, masks, masktools

logger = logging.getLogger(__name__)


def process() -> None:
    """Demonstrate usage of filtering tools."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    my_mask_series = masks.my_mask(df)

    result1 = masktools.apply_mask(df, mask=my_mask_series, name="precomputed_mask")
    result2 = masktools.apply_mask(df, mask=masks.my_mask, name="mask_func")
    result3 = filters.my_mask_filter(df)
    assert result1.equals(result2), "apply_mask should work with both precomputed masks and mask functions."
    assert result1.equals(result3), "The filter should produce the same result as apply_mask"
