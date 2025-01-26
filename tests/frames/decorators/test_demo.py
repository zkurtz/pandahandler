import logging

import pytest

from pandahandler.frames.decorators.demo import _filtering


def test__filtering(caplog: pytest.LogCaptureFixture):
    msg = "INFO     pandahandler.frames.filters:demo.py:17 drop_if_any_null returned 1 rows, down 2 rows (-66.7%).\n"
    with caplog.at_level(logging.INFO):
        _filtering()
        assert caplog.text == msg
