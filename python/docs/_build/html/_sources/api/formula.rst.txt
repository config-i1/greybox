=======
Formula
=======

.. autofunction:: greybox.formula.formula

.. autofunction:: greybox.formula.expand_formula

Formula Syntax Reference
------------------------

Basic Operators
~~~~~~~~~~~~~~~

* ``~`` : Separates response from predictors
* ``+`` : Adds a term (include variable)
* ``-`` : Removes a term
* ``0`` or ``-1`` : Removes intercept
* ``1`` : Adds intercept (default)
* ``*`` : Main effects and interactions (a*b = a + b + a:b)
* ``:`` : Interaction only
* ``I()`` : Protect expression from interpretation

Transformations
~~~~~~~~~~~~~~~

Supported transformations in formula terms:

* ``log(x)`` - Natural logarithm
* ``log10(x)`` - Base 10 logarithm
* ``log2(x)`` - Base 2 logarithm
* ``sqrt(x)`` - Square root
* ``exp(x)`` - Exponential
* ``abs(x)`` - Absolute value
* ``sin(x)``, ``cos(x)``, ``tan(x)`` - Trigonometric

Polynomial Terms
~~~~~~~~~~~~~~~~

* ``I(x^2)`` - Squared term (protected)
* ``I(x^3)`` - Cubed term
* ``poly(x, 2)`` - Polynomial (if supported)

Special Variables
~~~~~~~~~~~~~~~~~

* ``trend`` - Linear time trend (1, 2, 3, ...)

Examples
~~~~~~~~

Basic linear regression::

    y, X = formula("y ~ x1 + x2", data)

Without intercept::

    y, X = formula("y ~ 0 + x1 + x2", data)

With log transformation::

    y, X = formula("log(y) ~ log(x1) + sqrt(x2)", data)

Polynomial regression::

    y, X = formula("y ~ x + I(x^2) + I(x^3)", data)

Interactions::

    y, X = formula("y ~ x1 * x2", data)  # equivalent to x1 + x2 + x1:x2
