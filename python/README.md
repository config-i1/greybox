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

- **ALM (Augmented Linear Model)**: Likelihood-based regression with 26 distributions
- **Formula parser**: R-style formulas (`y ~ x1 + x2`, `log(y) ~ .`, `y ~ 0 + x1`)
- **Model selection**: `stepwise()` IC-based variable selection with partial correlations
- **Model combination**: `lm_combine()` weighted model averaging based on IC
- **Accuracy measures**: MAE, MSE, RMSE, MAPE, MASE, MPE, sMAPE, and more
- **Variable processing**: `xreg_expander` (lags/leads), `xreg_multiplier` (interactions), `temporal_dummy`
- **27 distribution families**: Each with density, CDF, quantile, and random generation functions

## Links

- [GitHub repository](https://github.com/config-i1/greybox)
- [R package on CRAN](https://cran.r-project.org/package=greybox)

## License

[LGPL-2.1](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html)
