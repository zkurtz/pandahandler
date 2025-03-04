import logging
import textwrap

import pytest

from pandahandler.frames.decorators.demo import _filtering

EXPECTED_LOG = """
    INFO     pandahandler.frames.decorators.demo:demo.py:27 drop_if_any_null returned 2 rows, down 1 rows (-33.3%).
    WARNING  pandahandler.frames.decorators.demo:demo.py:28 local_filter returned 1 rows, down 1 rows (-50.0%).
    """


def test__filtering(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        _filtering()
        assert caplog.text.strip() == textwrap.dedent(EXPECTED_LOG).strip()
