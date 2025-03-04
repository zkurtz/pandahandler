"""Run the filtering processes demo as `python -m pandahandler.frames.filtering.demo.run`."""

import logging

from pandahandler.frames.filtering.demo import processing

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    print("\nbasic logging:")
    logging.basicConfig(level=logging.INFO)
    processing.process()

    print("\ndetailed logging including line number:")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(lineno)d:%(message)s", force=True)
    processing.process()
