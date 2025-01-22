"""Methods to join/merge/stack multiple data frames."""

import pandas as pd


def safe_hstack(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Horizontally concatenate (i.e. axis=1) data frames after asserting strong assumptions.

    The naming of this method suggests an apt analogy with numpy's `hstack` function, which horizontally stacks
    arrays. This method is a convenience wrapper around `pd.concat` that includes the following assertions about the
    inputs frames:
      - all data frames must share the same index
      - there is no overlap among the columns of the data frames (ignoring the indexes)

    Args:
        frames: The data frames to concatenate.

    Raises:
        ValueError: If no data frames are provided in `frames`.
        ValueError: If the data frames do not all share the same index.
        ValueError: If any column name appears in more than one of the input data frames.
    """
    # We need at least one frame to be able to return a frame:
    if len(frames) == 0:
        raise ValueError("At least one data frame must be provided.")

    # Check that all data frames share the same index:
    index = frames[0].index.copy()
    for num, frame in enumerate(frames[1:]):
        which_idx = num + 1
        if not index.equals(frame.index):
            req = "All data frames must share the same index"
            raise ValueError(f"{req}, but frames[{which_idx}].index is not equal to frames[0].index.")

    # Check that there is no overlap among the columns of the data frames:
    columns = pd.concat([pd.Series(df.columns) for df in frames], axis=0)
    counts = columns.value_counts(dropna=False)
    overlap = counts[counts > 1]
    assert isinstance(overlap, pd.Series), "Expected overlap to be a Series."
    if not overlap.empty:
        raise ValueError(f"Column names must be unique across data frames, but these repeat: {overlap.index}.")
    df = pd.concat(frames, axis=1)
    assert df.index.equals(index), "Expected the concatenated data frame to have the same index as the input frames."
    return df
