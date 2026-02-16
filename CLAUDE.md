# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python port of the R `greybox` package — a toolbox for regression model building and forecasting. The R package (root directory) is the mature original; the Python port (`python/`) is the active development target on the `Python` branch.

## Build, Lint, and Test Commands

All commands run from `python/`:

```bash
make build          # pip install -e src/
make lint           # flake8 .
make test           # pytest
pytest tests/test_alm.py                              # single file
pytest tests/test_alm.py::TestFormula::test_formula_with_response  # single test
ptw                 # watch mode
```

R-vs-Python comparison tests require `rpy2` and R `greybox` installed:
```bash
pytest tests/test_r_python_compare.py
pytest tests/test_alm_distributions_compare.py
```

## Architecture

### Python package (`python/src/greybox/`)

- **`alm.py`** — Core `ALM` class (scikit-learn-style estimator). Supports 26 distributions and 7 loss functions. Uses `nlopt` for optimization (default: Nelder-Mead).
- **`formula.py`** — R-style formula parser (`formula()`, `expand_formula()`). Supports `y ~ x1 + x2`, `y ~ 0 + x1` (no intercept), `log(y) ~ x`, `I(x+z)` (protected expressions), `trend` auto-generation.
- **`fitters.py`** — Internal fitting machinery: `scaler_internal`, `fitter`, `fitter_recursive`. Wraps `cost_function.py`.
- **`cost_function.py`** — Loss functions: `likelihood`, `MSE`, `MAE`, `HAM`, `LASSO`, `RIDGE`, `ROLE`.
- **`predict.py`** — `predict_basic()` returning `PredictResult` dataclass with `.mean`, `.lower`, `.upper`.
- **`distributions/`** — 27 distributions with d/p/q/r functions (density, CDF, quantile, random).
- **`methods/summary.py`** — `SummaryResult` dataclass with formatted `__str__`.
- **`selection.py`** — `stepwise()`, `lm_combine()`.
- **`measures.py`** — Accuracy metrics (MAE, MSE, RMSE, MAPE, MASE, etc.).
- **`xreg.py`** — Variable processing: lag/lead expansion, transformations, temporal dummies.

### Key design decisions

- Prediction intervals use t-distribution for all distributions (matching R behavior). Variance = `X @ vcov @ X'`, scale uses `df_residual = n - k`.
- `nlopt` is a runtime dependency (not listed in pyproject.toml, must be installed separately).

## Code Style

- 4-space indentation, 80-char line limit
- `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- Type hints on all function signatures
- Import order: stdlib, third-party, local (blank line between groups)
- Linter: `flake8`; formatter: `black`; type checker: `mypy`
