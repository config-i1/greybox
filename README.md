# greybox

The package _greybox_ contains functions allowing selecting and combining regression models based on AIC and other information criteria. The resulting models can be then used in analysis and forecasting.

The list of included functions:
1. xregExpander - Function produces lags and leads of the provided data.
2. stepwise - Function implements stepwise AIC based on partial correlations.
3. combiner - Function combines the regression models from the provided data based on IC weigths and returns the combined lm object.

Future functions:
1. nonlinearExpander - Function produces non-linear transformations of the provided data.
2. multicor - multiple correlations coefficient.

Methods implemented:
1. summary.lm.combined - returns summary of the combined regression.
2. nobs.lm.combined - returns number of observations in the data of the model.
3. getResponse - returns the response variable from the combined model.
4. plot - plots the basic linear graph of actuals and fitted.

## Installation

A recent, development version, is available via github and can be installed using "devtools" in R. First make sure that you have devtools:
> if (!require("devtools")){install.packages("devtools")}

and after that run:
> devtools::install_github("config-i1/greybox")
