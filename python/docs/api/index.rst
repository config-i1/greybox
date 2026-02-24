===
API
===

.. toctree::
   :maxdepth: 2

   alm
   formula
   selection
   error_measures
   association
   hm
   quantile_measures
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

* :func:`greybox.point_measures.mae` - Mean Absolute Error
* :func:`greybox.point_measures.mse` - Mean Squared Error
* :func:`greybox.point_measures.rmse` - Root Mean Squared Error
* :func:`greybox.point_measures.mape` - Mean Absolute Percentage Error
* :func:`greybox.point_measures.mase` - Mean Absolute Scaled Error
* :func:`greybox.measures` - Comprehensive error measures

Association Measures
--------------------

* :mod:`greybox.association` - Association and correlation measures

Half-Moments
------------

* :mod:`greybox.hm` - Half-moment functions

Quantile Measures
-----------------

* :mod:`greybox.quantile_measures` - Pinball loss and interval scores

Variable Processing
-------------------

* :func:`greybox.xreg.xreg_expander` - Lag/lead expansion
* :func:`greybox.xreg.xreg_multiplier` - Cross-product generation
* :func:`greybox.xreg.temporal_dummy` - Temporal dummy variables
