# greybox

[![PyPI version](https://img.shields.io/pypi/v/greybox.svg)](https://pypi.org/project/greybox/)
[![Python CI](https://github.com/config-i1/greybox/actions/workflows/python-test.yml/badge.svg)](https://github.com/config-i1/greybox/actions/workflows/python-test.yml)
[![Python versions](https://img.shields.io/pypi/pyversions/greybox.svg)](https://pypi.org/project/greybox/)
[![License: LGPL-2.1](https://img.shields.io/badge/License-LGPL--2.1-blue.svg)](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html)

Python port of the [R greybox package](https://cran.r-project.org/package=greybox) — a toolbox for regression model building and forecasting.

![hex-sticker of the greybox package for Python](https://github.com/config-i1/greybox/blob/master/python/img/greybox-python-web.png?raw=true)

## Installation

```bash
pip install greybox
```

For more installation options, see the [Installation](https://github.com/config-i1/greybox/wiki/Installation) wiki page.

## Quick Example

```python
import numpy as np
import pandas as pd
from greybox import ALM, formula

# Generate sample data
np.random.seed(42)
n = 200
data = pd.DataFrame({
    "y": np.random.normal(10, 2, n),
    "x1": np.random.normal(5, 1, n),
    "x2": np.random.normal(3, 1, n),
})
data["y"] = 2 + 0.5 * data["x1"] - 0.3 * data["x2"] + np.random.normal(0, 1, n)

# Parse formula and fit model
result = formula("y ~ x1 + x2", data=data)
model = ALM(distribution="dnorm")
model.fit(result.data, result.formula)

# Summary
print(model.summary())

# Predict with intervals
pred = model.predict(result.data, interval="prediction", level=0.95)
print(pred.mean[:5])

# Include AR terms (ARIMA-like models)
# For example, ARIMA(1,1,0) model with Log-Normal distribution:
model = ALM(distribution="dlnorm", orders=(1, 1, 0))
model.fit(X, y)
```

## Supported Distributions

| Category | Distributions |
|----------|--------------|
| **Continuous** | `dnorm`, `dlaplace`, `ds`, `dgnorm`, `dlgnorm`, `dfnorm`, `drectnorm`, `dt` |
| **Positive** | `dlnorm`, `dinvgauss`, `dgamma`, `dexp`, `dchisq` |
| **Count** | `dpois`, `dnbinom`, `dgeom` |
| **Bounded** | `dbeta`, `dlogitnorm`, `dbcnorm` |
| **CDF-based** | `pnorm`, `plogis` |
| **Other** | `dalaplace`, `dbinom` |

## Features

- **[ALM (Augmented Linear Model)](https://github.com/config-i1/greybox/wiki/ALM)**: Likelihood-based regression with 26 distributions
- **Formula parser**: R-style formulas (`y ~ x1 + x2`, `log(y) ~ .`, `y ~ 0 + x1`) with support for backshift operator
- **[stepwise()](https://github.com/config-i1/greybox/wiki/stepwise)**: IC-based variable selection with partial correlations
- **[CALM()](https://github.com/config-i1/greybox/wiki/CALM)**: Combine ALM models based on IC weights
- **[Forecast error measures](https://github.com/config-i1/greybox/wiki/measures)**: MAE, MSE, RMSE, MAPE, MASE, MPE, sMAPE, and more
- **[Variable processing](https://github.com/config-i1/greybox/wiki/manipulations)**: `xreg_expander` (lags/leads), `xreg_multiplier` (interactions), `temporal_dummy`
- **[Distributions](https://github.com/config-i1/greybox/wiki/distributions)**: 27 distribution families with density, CDF, quantile, and random generation
- **[Association](https://github.com/config-i1/greybox/wiki/association)**: Partial correlations and measures of association
- **[Diagnostics](https://github.com/config-i1/greybox/wiki/diagnostics)**: Model diagnostics and validation

## Links

- [Wiki documentation](https://github.com/config-i1/greybox/wiki)
- [GitHub repository](https://github.com/config-i1/greybox)
- [R package on CRAN](https://cran.r-project.org/package=greybox)

## License

[LGPL-2.1](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html)
