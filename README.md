# pandahandler

Tools to simplify working with pandas objects DataFrame and Series.

pandahandler is a staging area for experimental tooling that could eventually be added to pandas itself. If that were to happen with any of these tools, we would remove them from this package after adding a deprecation notice and migration notes etc.

## Documentation

Users currently must rely on source code docstrings for most documentation. We'll briefly describe a few of the main methods below.

- [safe_hstack](https://github.com/zkurtz/pandahandler/blob/7ac841181595cd538f6f3f42a0f73be9e156e206/pandahandler/frames/joiners.py#L6-L22) offers safe/strict horizontal concatenation of data frames. Calling `safe_hstack([df1, df2, df3])` is much like `pd.concat([df1, df2, df3])` but first guarantees that (a) the columns of the input data frames are disjoint, and (b) the row-indexes of the input data frames are identical, raising helpful error messages if these conditions are not met. [On stackoverlow](https://stackoverflow.com/a/79405137/2232265).
- [log_rowcount_change](https://github.com/zkurtz/pandahandler/blob/7ac841181595cd538f6f3f42a0f73be9e156e206/pandahandler/frames/decorators/framesize.py#L13-L28) is a decorator ([example application](https://github.com/zkurtz/pandahandler/blob/7ac841181595cd538f6f3f42a0f73be9e156e206/pandahandler/frames/filters.py#L12-L15)) for data frame filtering functions that logs the change in row count both in raw count and percentage terms. This is useful for debugging or for monitoring the impact of a filters in data pipelines. [On stackoverflow](https://stackoverflow.com/a/79405155/2232265).
- [indexes.Index](https://github.com/zkurtz/pandahandler/blob/7ac841181595cd538f6f3f42a0f73be9e156e206/pandahandler/indexes.py#L62-L79) simplifies the process of declaring data frame indexes. A user can define an index as a constant to replace index setting/checking boilerplate in across any product-specific ecosystem.[On stackoverflow](https://stackoverflow.com/a/79405178/2232265).
- [indexes.unset](https://github.com/zkurtz/pandahandler/blob/7ac841181595cd538f6f3f42a0f73be9e156e206/pandahandler/indexes.py#L42-L54) is a utility for unsetting the index of a data frame, converting any existing index columns while (a) asserting that no columns of the same name already exist on the frame and (b) reverting to a no-op in case the index is already an unnamed range index. [On stackoverflow](https://stackoverflow.com/a/79405083/2232265).


## Installation

We're [on pypi](https://pypi.org/project/pandahandler/), so `pip install pandahandler`.

Consider using the [simplest-possible virtual environment](https://gist.github.com/zkurtz/4c61572b03e667a7596a607706463543) if working directly on this repo.
