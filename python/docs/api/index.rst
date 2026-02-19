===
API
===

.. toctree::
   :maxdepth: 2

   alm
   formula
   selection
   measures
   xreg
   transforms
   distributions

Core Classes
------------

* :class:`greybox.ALM` - Augmented Linear Model estimator
* :class:`greybox.formula` - Formula parsing function
* :class:`greybox.expand_formula` - Formula expansion function
* :class:`greybox.bc_transform` - Box-Cox transformation
* :class:`greybox.bc_transform_inv` - Inverse Box-Cox transformation
* :class:`greybox.mean_fast` - Trimmed/hubered mean

Model Selection
---------------

* :func:`greybox.stepwise` - Stepwise IC-based variable selection
* :func:`greybox.CALM` - Combine ALM models based on IC weights

Accuracy Measures
-----------------

* :func:`greybox.measures.MAE` - Mean Absolute Error
* :func:`greybox.measures.MSE` - Mean Squared Error
* :func:`greybox.measures.RMSE` - Root Mean Squared Error
* :func:`greybox.measures.MAPE` - Mean Absolute Percentage Error
* :func:`greybox.measures.MASE` - Mean Absolute Scaled Error

Variable Processing
-------------------

* :func:`greybox.xreg.xreg_expander` - Lag/lead expansion
* :func:`greybox.xreg.xreg_multiplier` - Cross-product generation
* :func:`greybox.xreg.temporal_dummy` - Temporal dummy variables
