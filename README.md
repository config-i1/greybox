# greybox
[![CRAN_Status_Badge](http://www.r-pkg.org/badges/version/greybox)](https://cran.r-project.org/package=greybox)
[![Downloads](http://cranlogs.r-pkg.org/badges/greybox)](https://cran.r-project.org/package=greybox)
[![ko-fi](https://ivan.svetunkov.ru/ko-fi.png)](https://ko-fi.com/G2G51C4C4)

The package _greybox_ contains functions for model building, which is currently done via the model selection and combinations based on information criteria. The resulting model can then be used in analysis and forecasting.

There are several groups of functions in the package.

### Regression model functions
1. alm - augmented linear regression model that implements likelihood estimation of parameters for Normal, Laplace, Asymmetric Laplace, Logistic, Student's t, S, Folded Normal, Log Normal, Box-Cox Normal, Inverse Gaussian, Chi-Squared, Beta, Poisson, Negative Binomial, Cumulative Logistic and Cumulative Normal distributions. In a sense this is similar to `glm()` function, but with a different set of distributions and with a focus on forecasting.
2. stepwise - function implements stepwise IC based on partial correlations.
3. lmCombine - function combines the regression models from the provided data, based on IC weights and returns the combined alm object.

### Exogenous variables transformation functions
1. xregExpander - function produces lags and leads of the provided data.
2. xregTransformer - function produces non-linear transformations of the provided data (logs, inverse etc).
3. xregMultiplier - function produces cross-products of the variables in the matrix. Could be useful when exploring interaction effects of dummy variables.

### The data analysis functions
1. cramer - calculates Cramer's V for two categorical variables. Plus tests the significance of such association.
2. mcor - function returns the coefficients of multiple correlation between the variables. This is useful when measuring association between categorical and numerical variables.
3. association (aka 'assoc()') - function returns matrix of measures of association, choosing between cramer(), mcor() and cor() depending on the types of variables.
4. determination (and the method 'determ()') - function returns the vector of coefficients of determination (R^2) for the provided data. This is useful for the diagnostics of multicollinearity.
5. tableplot - plots the graph for two categorical variables.
6. spread - plots the matrix of scatter / boxplot / tableplot diagrams - depending on the type of the provided variables.
7. graphmaker - plots the original series, the fitted values and the forecasts.

### Models evaluation functions
1. ro - rolling origin evaluation (see the vignette).
2. rmcb - Regression for Multiple Comparison with the Best. This is a simplified version of the nemenyi / MCB test, relying on regression on ranks of methods.
3. measures - the error measures for the provided forecasts. Includes MPE, MAPE, MASE, sMAE, sMSE, RelMAE, RelRMSE, MIS, sMIS, RelMIS, pinball and others.
<!--5. nemenyi - non-parametric test for comparison of multiple classifiers / methods. This function not only conducts the test, but also provide the plots, showing the ranks of the different methods together with their confidence intervals.-->

### Distribution functions:
1. qlaplace, dlaplace, rlaplace, plaplace - functions for Laplace distribution.
2. qalaplace, dalaplace, ralaplace, palaplace - functions for Asymmetric Laplace distribution.
3. qs, ds, rs, ps - functions for S distribution.
4. qfnorm, dfnorm, rfnorm, pfnorm - functions for folded normal distribution.
5. qtplnorm, dtplnorm, rtplnorm, ptplnorm - functions for three parameter log normal distribution.
6. qbcnorm, dbcnorm, rbcnorm, pbcnorm - functions for Box-Cox normal distribution (discussed in Box & Cox, 1964).
7. qlogitnorm, dlogitnorm, rlogitnorm, plogitnorm - functions for Logit-normal distribution.

### Methods for the introduced and some existing classes:
1. temporaldummy - the method that creates a matrix of dummy variables for an object based on the selected frequency. e.g. this can create week of year based on the provided zoo object.
2. outlierdummy - the method that creates a matrix of dummy variables based on the residuals of an object, selected confidence level and type of residuals.
3. pointLik - point likelihood method for the time series models.
4. pAIC, pAICc, pBIC, pBICc - respective point values for the information criteria, based on pointLik.
5. coefbootstrap - the method that returns bootstrapped coefficients of the model. Useful for the calculation of covariance matrix and confidence intervals for parameters.
6. summary - returns summary of the regression (either selected or combined).
7. vcov - covariance matrix for combined models. This is an approximate thing. The real one is quite messy and not yet available.
8. confint - confidence intervals for combined models.
9. predict, forecast - point and interval forecasts for the response variable. forecast method relies on the parameter h (the forecast horizon), while predict is focused on the newdata. See vignettes for the details.
10. nparam - returns the number of estimated parameters in the model (including location, scale, shift).
11. getResponse - returns the response variable from the model.
12. plot - plots several graphs for the analysis of the residuals (see documentation for more details).
13. AICc - AICc for regression with normally distributed residuals.
14. BICc - BICc for regression with normally distributed residuals.
15. is.greybox, is.alm etc. - functions to check if the object was generated by respective functions.

### Experimental functions:
1. lmDynamic - linear regression with time varying parameters based on pAIC.

## Installation

The stable version of the package is available on CRAN, so you can install it by running:
> install.packages("greybox")

A recent, development version, is available via github and can be installed using "devtools" in R. First make sure that you have devtools:
> if (!require("devtools")){install.packages("devtools")}

and after that run:
> devtools::install_github("config-i1/greybox")
