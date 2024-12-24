from importlib.metadata import version

__version__ = version("pandahandler")
__all__ = ["Schema", "categorize_non_numerics"]

from pandahandler.schema import Schema as Schema
from pandahandler.schema import categorize_non_numerics as categorize_non_numerics
