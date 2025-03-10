pandahandler reference guide
============================

- pandahandler is collection of utilities for working with pandas objects.
- We're `on pypi <https://pypi.org/project/pandahandler/>`_, so ``pip install pandahandler``
- `Github repo <https://github.com/zkurtz/pandahandler>`_

.. rst-class:: quick-links

Quick Links
-----------

Here's an overview of the main features. Click on the links for detailed API documentation:

* :func:`pandahandler.frames.decorators.framesize.log_rowcount_change` - a decorator for data frame filtering functions that logs the change in row count both in raw count and percentage terms. This is useful for debugging or for monitoring the impact of a filters in data pipelines.
* :func:`pandahandler.frames.filtering.masktools.apply_mask`: applies a mask (either as a boolean series or a function that generates one) to a data frame, using ``log_rowcount_change`` under the hood to log the effects of the filter.
* :func:`pandahandler.frames.filtering.masktools.as_filter`: converts any mask-generating function into a filter function, again invoking ``log_rowcount_change`` under the hood to log the effects of the filter.
* :func:`pandahandler.frames.joiners.safe_hstack` - safe/strict horizontal concatenation of data frames. Calling ``safe_hstack([df1, df2, df3])`` is much like ``pd.concat([df1, df2, df3])`` but first guarantees that (a) the columns of the input data frames are disjoint, and (b) the row-indexes of the input data frames are identical, raising helpful error messages if these conditions are not met.
* :func:`pandahandler.indexes.unset` - unsets the index of a data frame as safely as possible, converting any existing index columns to regular columns while (a) asserting that no columns of the same name already exist on the frame and (b) reverting to a no-op in case the index is already an unnamed range index.
* :class:`pandahandler.indexes.Index` - simplifies the process of declaring data frame indexes. A user can define an index as a constant to replace index setting/checking boilerplate in across any product-specific ecosystem.


.. toctree::
   :maxdepth: 3
   :caption: Standard docs tree

   autoapi/pandahandler/index
