# greybox
[![CRAN_Status_Badge](http://www.r-pkg.org/badges/version/greybox)](https://cran.r-project.org/package=greybox)
[![Downloads](http://cranlogs.r-pkg.org/badges/greybox)](https://cran.r-project.org/package=greybox)

The package _greybox_ contains functions for model building, which is currently done via the model selection and combinations based on information criteria. The resulting model can then be used in analysis and forecasting.

The list of included functions:
1. xregExpander - Function produces lags and leads of the provided data.
2. stepwise - Function implements stepwise AIC based on partial correlations.
3. combiner - Function combines the regression models from the provided data based on IC weigths and returns the combined lm object.
4. ro - rolling origin evaluation (see the vignette).

Future functions:
1. nonlinearExpander - Function produces non-linear transformations of the provided data.
2. lmAdvanced - linear regression with MAE / HAM / MAPE / etc.
3. vcov, confint, forecast - methods for stepwise and combined regressions.

Methods already implemented:
1. summary - returns summary of the regression (either selected or combined).
2. nParam - returns number of parameters of a model.
3. getResponse - returns the response variable from the model.
4. plot - plots the basic linear graph of actuals and fitted.
5. AICc - AICc for regression with normally distributed residuals.

## Installation

The stable version of the package is available on CRAN, so you can install it by running:
> install.packages("greybox")

A recent, development version, is available via github and can be installed using "devtools" in R. First make sure that you have devtools:
> if (!require("devtools")){install.packages("devtools")}

and after that run:
> devtools::install_github("config-i1/greybox")
