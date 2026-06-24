# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IMPORTANT: Always use the python/.venv environment when running Python commands!
Example: cd python && .venv/bin/python -m pytest tests/

IMPORTANT: Never create cache files, virtual environments, or other generated
artifacts in the project's root folder. All Python work — including the
virtual environment (`python/.venv`), caches, and build outputs — belongs in
the `python/` subfolder.

Python port of the R `greybox` package — a toolbox for regression model building and forecasting. The R package (root directory) is the mature original; the Python port (`python/`) is the active development target on the `Python` branch.

## Build, Lint, and Test Commands

All commands run from `python/`:

```bash
make build          # pip install -e src/
make lint           # ruff check + format
make typecheck      # mypy src/greybox/ --ignore-missing-imports
make test           # pytest
pytest tests/test_alm.py                              # single file
pytest tests/test_alm.py::TestFormula::test_formula_with_response  # single test
ptw                 # watch mode
```

IMPORTANT: Always run `make typecheck` (mypy) together with the linting
checks — treat type checking as part of "linting", never skip it. New or
changed code must pass `make lint` AND `make typecheck` before it is
considered done.

R-vs-Python comparison tests require `rpy2` and R `greybox` installed:
```bash
pytest tests/test_r_python_compare.py
pytest tests/test_alm_distributions_compare.py
```

## Architecture

### Python package (`python/src/greybox/`)

- **`alm.py`** — Core `ALM` class (scikit-learn-style estimator). Supports 27 distributions and 7 loss functions. Uses `nlopt` for optimization (default: Nelder-Mead).
- **`formula.py`** — R-style formula parser (`formula()`, `expand_formula()`). Supports `y ~ x1 + x2`, `y ~ 0 + x1` (no intercept), `log(y) ~ x`, `I(x+z)` (protected expressions), `trend` auto-generation.
- **`fitters.py`** — Internal fitting machinery: `scaler_internal`, `fitter`, `fitter_recursive`. Wraps `cost_function.py`.
- **`cost_function.py`** — Loss functions: `likelihood`, `MSE`, `MAE`, `HAM`, `LASSO`, `RIDGE`, `ROLE`.
- **`predict.py`** — `predict_basic()` returning `PredictionResult` dataclass with `.mean`, `.lower`, `.upper`.
- **`distributions/`** — 27 distributions with d/p/q/r functions (density, CDF, quantile, random).
- **`methods/summary.py`** — `SummaryResult` dataclass with formatted `__str__`.
- **`selection.py`** — `stepwise()`, `CALM()`.
- **`point_measures.py`** — 20+ accuracy metrics (MAE, MSE, RMSE, MAPE, MASE, etc.).
- **`xreg.py`** — Variable processing: lag/lead expansion, transformations, temporal dummies.
- **`rolling.py`** — `rolling_origin()` cross-validation with `RollingOriginResult`.
- **`association.py`** — `pcor()`, `mcor()`, `association()`, `determination()` (partial/multiple correlations).
- **`hm.py`** — Half-moment measures: `hm()`, `ham()`, `asymmetry()`, `extremity()`, `mre()`.
- **`quantile_measures.py`** — `pinball()` quantile/expectile scoring, `MIS`.
- **`diagnostics.py`** — `outlier_dummy()` with `OutlierResult`.
- **`stick.py`** — `stick()` (Seasonality, Trend, Irregular contribution via ANOVA) with `StickResult` (`.strength`, `.anova`, `.plot()`).
- **`transforms.py`** — `bc_transform()`, `bc_transform_inv()`, `mean_fast()`.
- **`pointlik.py`** — `point_lik()` point-wise likelihoods per observation.
- **`data.py`** — `mtcars` dataset.

### Key design decisions

- Variance = `X @ vcov @ X'`, scale uses `df_residual = n - k`.
- `nlopt>=2.7.0` is a required runtime dependency (declared in pyproject.toml).

## Code Style

- 4-space indentation, 80-char line limit
- `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- Type hints on all function signatures
- Import order: stdlib, third-party, local (blank line between groups)
- Linter/formatter: `ruff`; type checker: `mypy`
