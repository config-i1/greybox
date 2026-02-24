===
ALM
===

.. autoclass:: greybox.alm.ALM
   :members:
   :undoc-members:
   :show-inheritance:

PredictionResult
----------------

.. autoclass:: greybox.alm.PredictionResult
   :members:
   :undoc-members:
   :no-index:

Dynamic Multipliers
-------------------

.. autofunction:: greybox.xreg.multipliers

**Example**::

   from greybox import ALM, formula, multipliers

   y, X = formula("y ~ x + B(x, 1) + B(x, 2)", data)
   model = ALM().fit(y, X)
   m = multipliers(model, "x", h=10)
   # m == {"h1": ..., "h2": ..., ..., "h10": ...}
