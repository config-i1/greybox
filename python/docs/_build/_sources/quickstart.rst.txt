==========
Quickstart
==========

This guide will help you get started with greybox.

Basic Workflow
--------------

The typical workflow with greybox involves:

1. Prepare your data in a dictionary or pandas DataFrame
2. Use the ``formula`` function to create design matrix and response
3. Create and fit an ALM model
4. Inspect the model and make predictions

Basic Example
-------------

Here's a simple example using the mtcars dataset::

    from greybox import formula, ALM

    # Prepare data as a dictionary
    data = {
        'mpg': [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2],
        'wt': [2.620, 2.875, 2.320, 3.215, 3.440, 3.460, 3.570, 3.190, 3.150, 3.440],
        'am': [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    }

    # Parse the formula and get X (design matrix) and y (response)
    y, X = formula("mpg ~ wt + am", data)

    # Create and fit the model
    model = ALM(distribution="dnorm", loss="likelihood")
    model.fit(X, y)

    # View coefficients
    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)

    # Make predictions
    predictions = model.predict(X)
    print("Predicted values:", predictions.mean)

Using Different Distributions
-----------------------------

greybox supports many distributions for different types of data::

    # Normal distribution (default)
    model = ALM(distribution="dnorm", loss="likelihood")

    # Laplace distribution (heavier tails)
    model = ALM(distribution="dlaplace", loss="likelihood")

    # Log-normal distribution (for positive data)
    model = ALM(distribution="dlnorm", loss="likelihood")

    # Poisson distribution (for count data)
    model = ALM(distribution="dpois", loss="likelihood")

Prediction Intervals
--------------------

You can obtain prediction intervals for forecasts::

    result = model.predict(X, interval="prediction", level=0.95)

    print("Mean:", result.mean)
    print("Lower bound:", result.lower)
    print("Upper bound:", result.upper)

Formula Syntax
--------------

The formula parser supports various transformations::

    # Basic formula
    y, X = formula("y ~ x1 + x2", data)

    # With intercept (default)
    y, X = formula("y ~ 1 + x1 + x2", data)

    # Without intercept
    y, X = formula("y ~ 0 + x1 + x2", data)

    # Log transformation
    y, X = formula("log(y) ~ log(x1) + x2", data)

    # Polynomial terms
    y, X = formula("y ~ x1 + I(x1^2)", data)

    # Interactions
    y, X = formula("y ~ x1 * x2", data)

Model Summary
-------------

Get detailed model information::

    summary = model.summary()
    print(summary)

ARIMA Orders
------------

The ``orders`` parameter allows you to include AR (autoregressive) terms and
differencing in your model, similar to ARIMA::

    from greybox import formula, ALM
    import pandas as pd

    # Load mtcars dataset (included with greybox)
    from greybox import mtcars
    data = mtcars.to_dict(orient="list")

    # Parse formula
    y, X = formula("mpg ~ wt", data)

    # Fit model with AR(1) term
    model = ALM(distribution="dnorm", orders=(1, 0, 0))
    model.fit(X, y)

    print("AR(1) model coefficients:")
    print("Intercept:", model.intercept_)
    print("wt:", model.coef[0])
    print("mpgLag1:", model.coef[1])

    # Fit model with AR(2) terms
    model2 = ALM(distribution="dnorm", orders=(2, 0, 0))
    model2.fit(X, y)

    print("\nAR(2) model coefficients:")
    print("Intercept:", model2.intercept_)
    print("wt:", model2.coef[0])
    print("mpgLag1:", model2.coef[1])
    print("mpgLag2:", model2.coef[2])

The ``orders`` parameter is a tuple ``(p, d, q)``:

- ``p``: AR (autoregressive) order - number of lagged response variables
- ``d``: Differencing order - order of differencing for non-stationary data
- ``q``: MA (moving average) order - not yet implemented

Note: MA(q) is not supported and will raise an error if used.
