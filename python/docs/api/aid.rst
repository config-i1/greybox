==================================
Automatic Identification of Demand
==================================

The ``greybox.aid`` module classifies a time series into one of six
demand types and flags stockouts, new products, and obsolete products.
It is a one-to-one Python port of R's :func:`greybox::aid` and
:func:`greybox::aidCat`.

aid()
-----

.. autofunction:: greybox.aid.aid

**Example**::

   import numpy as np
   from greybox import aid

   rng = np.random.default_rng(42)

   # Intermittent count demand: Poisson(0.7) gives many zeros
   y = rng.poisson(0.7, 120).astype(float)
   result = aid(y)
   print(result)
   # The provided time series is smooth intermittent count

   # Detect an injected stockout
   y2 = rng.poisson(3, 100).astype(float)
   y2[40:50] = 0
   result2 = aid(y2)
   print(result2.stockouts.start, result2.stockouts.end)
   # [41] [50]  (1-based positions, matches R)

**Accessing the result**

The ``type`` and ``stockouts`` fields are typed dataclasses
(:class:`AidType` / :class:`Stockouts`) that support both R-style
attribute access and dict-style subscript access::

   result2.stockouts.start    # numpy.ndarray of 1-based positions
   result2.stockouts["start"] # same data, dict-style fallback
   result2.type.type1         # "count" or "fractional"

**Plotting**

``AidResult.plot()`` mirrors R's ``plot.aid()`` — line plot of the
series with each stockout shaded in light grey and bracketed by a
solid red (start) / dashed green (end) vertical line. Leading-zero
runs flagged as a new product get a light-blue shading, trailing-zero
runs flagged as obsolete get a light-orange shading::

   import matplotlib.pyplot as plt

   ax = result2.plot()
   plt.show()

aid_cat()
---------

.. autofunction:: greybox.aid.aid_cat

**Example**::

   import numpy as np
   from greybox import aid_cat

   rng = np.random.default_rng(1)
   series = {
       "a": rng.poisson(1, 80).astype(float),
       "b": rng.poisson(5, 80).astype(float),
       "c": rng.normal(10, 2, 80),
   }
   result = aid_cat(series)

   print(result.types)        # 2x3 demand-type counts
   print(result.anomalies)    # New / Stockouts / Old counts
   print(result.categories)   # categorical vector per series

   ax = result.plot()         # 2x3 demand-category panel

Result classes
--------------

.. autoclass:: greybox.aid.AidResult
   :members: plot
   :no-undoc-members:

.. autoclass:: greybox.aid.AidCatResult
   :members: plot
   :no-undoc-members:

Nested dataclasses
------------------

.. autoclass:: greybox.aid.AidType
   :no-members:

.. autoclass:: greybox.aid.Stockouts
   :no-members:
