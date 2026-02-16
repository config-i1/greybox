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
