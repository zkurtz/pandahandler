import logging

import pandas as pd
import pytest

from pandahandler.frames.filtering.masktools import apply_mask


def test_apply_mask(caplog: pytest.LogCaptureFixture) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    mask = pd.Series([True, False, True], index=df.index)

    with caplog.at_level(logging.INFO):
        result = apply_mask(df, mask=mask, name="test_mask")
        assert "apply_mask:test_mask returned 2 rows, down 1 rows (-33.3%)" in caplog.text
        pd.testing.assert_frame_equal(result, df.loc[mask])

    with pytest.raises(ValueError, match="The mask index must be identical to the data frame index."):
        apply_mask(df, mask=pd.Series([True, False, True], index=["a", "b", "c"]))

    with pytest.raises(TypeError, match="The mask must be of boolean type."):
        apply_mask(df, mask=pd.Series([1, 0, 1], index=df.index))
