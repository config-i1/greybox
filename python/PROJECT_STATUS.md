# Greybox Python Port - Project Status

## Overview

This is a Python port of the R package `greybox` - a toolbox for model building and forecasting. The Python implementation provides feature parity with the R package for regression model building and forecasting tasks.

## Project Summary

### Main Features Implemented

1. **Regression Models**
   - ALM (Augmented Linear Model) with various distributions
   - Support for multiple loss functions (likelihood, MSE, MAE, HAM, LASSO, RIDGE, ROLE)

2. **Distributions Supported (27)**
   - Normal: `dnorm`, `plogis`, `pnorm`
   - Laplace: `dlaplace`, `dalaplace`, `dllaplace`
   - Logistic: `dlogis`
   - Student-t: `dt`
   - Normal variants: `ds`, `dgnorm`, `dlnorm`, `dlgnorm`, `dfnorm`, `drectnorm`, `dbcnorm`
   - Gamma family: `dgamma`, `dexp`, `dchisq`, `dinvgauss`
   - Count distributions: `dpois`, `dnbinom`, `dbinom`, `dgeom`
   - Other: `dbeta`, `dlogitnorm`, `dls`

3. **Model Selection**
   - Stepwise regression based on information criteria (AIC, AICc, BIC, BICc)
   - Model combination (`lm_combine`) using information criteria weights

4. **Variable Processing**
   - `xreg_transformer` - Mathematical transformations (log, exp, inv, sqrt, square)
   - `xreg_multiplier` - Cross-product generation
   - `xreg_expander` - Lag/lead expansion
   - `temporal_dummy` - Temporal dummies

5. **Forecasting**
   - Point forecasts with confidence/prediction intervals
   - Rolling origin evaluation for time series cross-validation

6. **Statistical Measures**
   - Accuracy metrics: MAE, MSE, RMSE, MPE, MAPE, MASE
   - Correlation: `pcor` (partial correlations), `mcor` (multiple correlation)
   - `determination` - R-squared and adjusted R-squared
   - `association` - Measures of association for different variable types

7. **Diagnostics**
   - `outlier_dummy` - Outlier detection and dummy variable creation

8. **Model Methods (S3-style)**
   - `predict()` - Point forecasts with intervals
   - `vcov()` - Variance-covariance matrix
   - `summary()` - Model summary with coefficients, std errors, t-stats, p-values
   - `confint()` - Confidence intervals for parameters
   - `forecast()` - Forecasting
   - Properties: `nobs`, `nparam`, `residuals`, `fitted`, `actuals`, `sigma`, `log_lik`, `formula`

## Project Structure

```
python/src/greybox/
├── __init__.py              # Package initialization
├── alm.py                   # Main ALM model class
├── formula.py               # Formula parser
├── fitters.py               # Internal fitters
├── cost_function.py         # Loss functions
├── transforms.py            # Data transformations
├── predict.py               # Prediction utilities
│
├── distributions/           # Distribution functions
│   ├── __init__.py
│   ├── alaplace.py
│   ├── beta.py
│   ├── bcnorm.py
│   ├── binom.py
│   ├── chi2.py
│   ├── exp.py
│   ├── fnorm.py
│   ├── geom.py
│   ├── gnorm.py
│   ├── helper.py
│   ├── gamma.py
│   ├── lapace.py
│   ├── llaplace.py
│   ├── lgnorm.py
│   ├── logis.py
│   ├── logitnorm.py
│   ├── lnorm.py
│   ├── nbinoom.py
│   ├── pois.py
│   ├── rectnorm.py
│   ├── s.py
│   ├── t.py
│   └── (and more)
│
├── methods/               # Model methods (Python best practice)
│   ├── __init__.py
│   └── summary.py        # SummaryResult class
│
├── diagnostics.py         # outlier_dummy function
├── measures.py           # Statistical measures (accuracy, pcor, mcor, etc.)
├── selection.py          # stepwise, lm_combine
├── xreg.py              # Variable manipulation functions
├── pointlik.py          # Point likelihood functions
└── rolling.py           # Rolling origin evaluation
```

## Test Status

- **Total Tests**: 55
- **Status**: All passing
- **Coverage**: Formula parsing, ALM fitting, prediction intervals, R comparison tests

## Key Implementation Details

### Prediction Intervals
- Uses t-distribution for ALL distributions (matches R)
- Variance calculation: `X @ vcov @ X'` directly
- Scale recalculated using `df_residual = n - k`

### ALM Properties
- `nobs` - Number of observations
- `nparam` - Number of parameters (including scale when applicable)
- `residuals` - Model residuals
- `fitted` - Fitted values
- `actuals` - Actual response values
- `sigma` - Scale parameter
- `log_lik` - Log-likelihood
- `formula` - Formula string

## Usage Examples

### Basic ALM Model
```python
from greybox.formula import formula
from greybox.alm import ALM

data = {'y': [1, 2, 3, 4, 5], 'x1': [1, 2, 3, 4, 5], 'x2': [2, 4, 6, 8, 10]}
y, X = formula("y ~ x1 + x2", data)

model = ALM(distribution="dnorm", loss="likelihood")
model.fit(X, y)

# Predictions with confidence intervals
result = model.predict(X, interval="confidence", level=0.95)
```

### Stepwise Regression
```python
from greybox.selection import stepwise

data = {'y': [1, 2, 3, 4, 5], 'x1': [1, 2, 3, 4, 5], 
        'x2': [2, 4, 6, 8, 10], 'x3': [3, 6, 9, 12, 15]}
model = stepwise(data, ic="AICc")
```

### Accuracy Measures
```python
from greybox.measures import accuracy, mae, mape

actual = [1, 2, 3, 4, 5]
forecast = [1.1, 2.0, 3.2, 3.9, 5.1]

measures = accuracy(actual, forecast)
print(measures)  # {'ME': ..., 'MAE': ..., 'MSE': ..., ...}
```

## Remaining Items

1. **lm_dynamic** - Dynamic linear regression with AR components (skipped)
2. **Additional R compatibility** - Some edge cases may differ

## Testing Commands

```bash
# Run all tests
pytest

# Run specific test file
pytest python/tests/test_alm.py

# Run linting
flake8 python/src/greybox/
```

## References

- Original R package: https://github.com/config/Python/Libraries/greybox
- R greybox documentation: https://cran.r-project.org/web/packages/greybox/

---

*Last Updated: February 2026*
