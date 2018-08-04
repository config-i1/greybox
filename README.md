# greybox
[![CRAN_Status_Badge](http://www.r-pkg.org/badges/version/greybox)](https://cran.r-project.org/package=greybox)
[![Downloads](http://cranlogs.r-pkg.org/badges/greybox)](https://cran.r-project.org/package=greybox)

The package _greybox_ contains functions for model building, which is currently done via the model selection and combinations based on information criteria. The resulting model can then be used in analysis and forecasting.

The list of included functions:
1. alm - advanced linear regression model that implements likelihood estimation of parameters for Laplace, S, Folded Normaland Chi-Squared distributions. In a sense this is similar to glm, but with a different set of distributions.
2. xregExpander - Function produces lags and leads of the provided data.
3. stepwise - Function implements stepwise AIC based on partial correlations.
4. combine - Function combines the regression models from the provided data based on IC weigths and returns the combined lm object.
5. ro - rolling origin evaluation (see the vignette).
<!--5. nemenyi - non-parametric test for comparison of multiple classifiers / methods. This function not only conducts the test, but also provide the plots, showing the ranks of the different methods together with their confidence intervals.-->
6. rmc - Regression for Multiple Comparison of forecasting methods. Can be used, for example, when RelMAE is calculated for several forecasting methods and an analysis of statistical significance in accuracy of methods needs to be carried out. This can be especially useful when you have a lot of methods to compare. The test is faster than Nemenyi in this case and becomes more powerful and accurate.
7. lmDynamic - linear regression with time varying parameters based on pAIC.
8. determination - the function returns the vector of coefficients of determination (R^2) for the provided data. This is useful for the diagnostics of multicollinearity.
9. qlaplace, dlaplace, rlaplace, plaplace - functions for Laplace distribution.
10. qs, ds, rs, ps - functions for S distribution.

Future functions:
1. nonlinearExpander - Function produces non-linear transformations of the provided data.

Methods already implemented:
1. pointLik - point likelihood method for the time series models.
2. pAIC - point AIC based on pointLik.
3. summary - returns summary of the regression (either selected or combined).
4. vcov - covariance matrix for combined models. This is an approximate thing. The real one is quite messy and not yet available.
5. confint - confidence intervals for combined models.
6. forecast - point and interval forecasts for the response variable.
7. nParam - returns number of parameters of a model.
8. getResponse - returns the response variable from the model.
9. plot - plots the basic linear graph of actuals and fitted. Similar thing plots graphs for forecasts of greybox functions.
10. AICc - AICc for regression with normally distributed residuals.
11. BICc - BICc for regression with normally distributed residuals.

## Installation

The stable version of the package is available on CRAN, so you can install it by running:
> install.packages("greybox")

A recent, development version, is available via github and can be installed using "devtools" in R. First make sure that you have devtools:
> if (!require("devtools")){install.packages("devtools")}

and after that run:
> devtools::install_github("config-i1/greybox")
