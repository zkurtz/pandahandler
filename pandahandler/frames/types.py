"""Constants for the frames module."""

from typing import Callable, Concatenate, ParamSpec, TypeAlias

import pandas as pd

P = ParamSpec("P")


DfTransform: TypeAlias = Callable[Concatenate[pd.DataFrame, P], pd.DataFrame]
