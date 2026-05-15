=========
Smoothers
=========

The ``greybox.smoothers`` module provides two non-parametric smoothers
that reproduce R's :func:`stats::lowess` and :func:`stats::supsmu` to
machine precision. Each wraps a native pybind11 extension and returns
a dictionary with sorted abscissa and smoothed ordinate.

LOWESS
------

.. autofunction:: greybox.smoothers.lowess

**Example**::

   import numpy as np
   from greybox import lowess

   rng = np.random.default_rng(0)
   x = np.linspace(0, 6, 80)
   y = np.sin(x) + rng.normal(0, 0.2, 80)

   smoothed = lowess(x, y, f=0.4)
   # smoothed["x"] are the sorted x values; smoothed["y"] is the smoothed curve.

**Reference**

Cleveland, W. S. (1979). "Robust Locally Weighted Regression and Smoothing
Scatterplots". *Journal of the American Statistical Association*, 74(368),
829-836. DOI: `10.1080/01621459.1979.10481038
<https://doi.org/10.1080/01621459.1979.10481038>`_.

SuperSmoother (supsmu)
----------------------

.. autofunction:: greybox.smoothers.supsmu

**Example**::

   import numpy as np
   from greybox import supsmu

   rng = np.random.default_rng(1)
   x = np.arange(100, dtype=float)
   y = 0.05 * x + rng.normal(0, 1.0, 100)

   # Default: cross-validated span selection
   cv = supsmu(x, y)

   # Fixed span (0 < span <= 1)
   fixed = supsmu(x, y, span=0.3)

   # Bass-tone control increases smoothness in noisy regions
   smoother = supsmu(x, y, bass=5.0)

**Reference**

Friedman, J. H. (1984). "A Variable Span Smoother". Technical Report 5
(SLAC-PUB-3477; STAN-LCS-005), Laboratory for Computational Statistics,
Department of Statistics, Stanford University.
`OSTI 1447470 <https://www.osti.gov/biblio/1447470>`_.
