import logging
import textwrap

import pytest

from pandahandler.frames.filtering.demo import processing

INFO_PROCESSING = "INFO     pandahandler.frames.filtering.demo.processing"
EXPECTED_LOG = f"""
    {INFO_PROCESSING}:processing.py:26 apply_mask:precomputed_mask returned 2 rows, down 1 rows (-33.3%).
    {INFO_PROCESSING}:processing.py:27 apply_mask:mask_func returned 2 rows, down 1 rows (-33.3%).
    {INFO_PROCESSING}:processing.py:28 my_mask_filter returned 2 rows, down 1 rows (-33.3%).
    """


def test_filtering_demo_process(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        processing.process()
        assert caplog.text.strip() == textwrap.dedent(EXPECTED_LOG).strip()
        caplog.clear()
