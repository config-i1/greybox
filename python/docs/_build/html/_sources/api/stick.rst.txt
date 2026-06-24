===================================================
Seasonality, Trend, and Irregular Contribution Kit
===================================================

The ``greybox.stick`` module decomposes the variance of a time series
into seasonal, trend and irregular parts based on an Analysis of
Variance (ANOVA) of the series on the seasonal and trend factors, and
measures the contribution of each component. It is a Python port of R's
:func:`greybox::stick` and implements the Seasonality, Trend, and
Irregular (STI) classification of Hans Levenbach.

stick()
-------

.. autofunction:: greybox.stick.stick

**Example**::

   import numpy as np
   from greybox import stick

   # Monthly series with a trend and a seasonal pattern
   t = np.arange(1, 121)
   y = 100 + 0.3 * t + 20 * np.sin(2 * np.pi * t / 12)

   result = stick(y, lags=12)
   print(result)
   # Seasonality, Trend, and Irregular Contribution Kit
   # Seasonal lags: 12
   #
   # Strength of the components:
   # seasonal12    ...
   # trend         ...
   # irregular     ...

   # Multiple seasonal lags (e.g. hourly data)
   result2 = stick(y_hourly, lags=[24, 168])

**Accessing the result**

The :class:`~greybox.stick.StickResult` exposes the strength of each
component as a :class:`pandas.Series` and the full ANOVA table as a
:class:`pandas.DataFrame`::

   result.strength            # one entry per lag, plus trend & irregular
   result.strength["trend"]   # strength of the trend
   result.anova               # Df / Sum Sq / Mean Sq / F value / Pr(>F)

The strengths are the shares of the respective Sum of Squares in the
total Sum of Squares and sum up to one.

**Plotting**

``StickResult.plot()`` mirrors R's ``plot.stick()``. For every seasonal
lag it draws a *seasonal plot* (the series reshaped into one grey line
per cycle, with the average seasonal profile overlaid as a bold dashed
black line); the final plot is the *trend plot* (one grey line per
seasonal position drawn across the cycles, with the average level per
cycle -- the trend -- overlaid as a bold dashed black line). The
``which`` argument selects which panes to draw: ``1, ..., k-1`` are the
seasonal plots, ``k`` is the trend plot; ``None`` (the default) draws
all of them::

   import matplotlib.pyplot as plt

   axes = result.plot()          # all panes
   ax = result.plot(which=2)     # only the trend plot
   plt.show()

Result class
------------

.. autoclass:: greybox.stick.StickResult
   :members: plot
   :no-undoc-members:
