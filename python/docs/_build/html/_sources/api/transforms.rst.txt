===========
Transforms
===========

Box-Cox Transformations
-----------------------

.. autofunction:: greybox.transforms.bc_transform

.. autofunction:: greybox.transforms.bc_transform_inv

Box-Cox Transformation
~~~~~~~~~~~~~~~~~~~~~~~

The Box-Cox transformation is defined as:

.. math::

    y^{(\lambda)} =
    \begin{cases}
    \frac{y^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\
    \ln(y) & \text{if } \lambda = 0
    \end{cases}

Common Values
~~~~~~~~~~~~~~

* ``lambda=0`` : Log transformation
* ``lambda=0.5`` : Square root transformation
* ``lambda=1`` : No transformation (linear)
* ``lambda=-1`` : Inverse transformation

Example
~~~~~~~

::

    from greybox import bc_transform, bc_transform_inv

    # Apply Box-Cox transformation with lambda=0.5
    y_transformed = bc_transform([1, 4, 9, 16], lambda_bc=0.5)
    # Result: [0, 2, 4, 6]

    # Inverse transformation
    y_original = bc_transform_inv([0, 2, 4, 6], lambda_bc=0.5)
    # Result: [1, 4, 9, 16]

mean_fast
---------

.. autofunction:: greybox.transforms.mean_fast

The mean_fast function computes a trimmed/hubered mean that is robust to outliers.

Parameters
~~~~~~~~~~

* ``x`` : array-like - Input data
* ``df`` : int - Degrees of freedom (default: 1)
* ``trim`` : float - Trim proportion (0 to 0.5, default: 0.0)
* ``side`` : str - Trim side: "both" (default), "lower", or "upper"

Example
~~~~~~~

::

    from greybox import mean_fast
    import numpy as np

    data = np.array([1, 2, 3, 4, 5, 100])

    # Standard mean (affected by outlier)
    np.mean(data)  # ~19.17

    # Trimmed mean (remove 10% from each tail)
    mean_fast(data, trim=0.1)  # ~3.0
