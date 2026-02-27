# Greybox

## Project Summary

This is a Python port of the R package `greybox` - a toolbox for model building and forecasting. The original R package implements functions and instruments for regression model building and its application to forecasting.

### Main Features
- **Regression Models**: Advanced Linear Model (ALM), Scale Model (SM), Dynamic Linear Regression
- **Model Selection**: Stepwise regression based on information criteria, partial correlations. Combined ALM (CALM)
- **Variable Processing**: 
  - Lag/lead expansion (`xregExpander`)
  - Mathematical transformations (log, sqrt, etc.)
  - Cross-product generation
  - Temporal dummies, outlier dummies
- **Forecasting**: Rolling origin evaluation, error measures
- **Statistical Measures**: AIC, BIC, determination coefficients, association measures
- **Bootstrap Methods**: Parameter bootstrap, bootstrap for time series
- **Distributions**: Normal, Laplace, LogitNormal, Beta-Normal, etc.

### Target Users
- Data scientists and forecasters
- Marketing analysts
- Econometric modelers
- Anyone building predictive models with time series data

---

## Build, Lint, and Test Commands
These commands ensure your code adheres to standards and functions as expected.

IMPORTANT: Always use the `python/.venv` environment when running Python commands!
Example:
```bash
cd python && .venv/bin/python -m pytest tests/
```

### Build Command:
```bash
make build
```

This command compiles the project. Ensure you're in the root directory of the project before running it.
Lint Command:
```bash
flake8 .
```

The flake8 tool checks your Python code for issues, including errors and deviations from PEP 8 style guidelines. It's a combination of PyFlakes, pycodestyle (formerly pep8), and McCabe script.
Test Commands:
- Run all tests:
    `pytest`
  - Run tests in watch mode:
    `ptw`
    This command runs pytest in a loop, re-running the tests whenever one of your files changes.
- Run a specific test file:
    `pytest [path_to_test_file]`
  - Run a specific test within a test file:
    `pytest [path_to_test_file]::[TestClassOrFunctionName]`
  
## Code Style Guidelines
### Imports
- Place standard library imports first, then third-party libraries, and lastly the local application/library-specific imports. Each section should be separated by an empty line.

Example:
```python
import os
import sys
import requests
from flask import Flask
from myapp.forms import MyForm
```

### Formatting
- Use 4 spaces per indentation level.
- Ensure lines do not exceed 80 characters to prevent horizontal scrolling.
- Separate top-level function and class definitions with two blank lines, and method definitions inside a class body with one blank line.
- Surround binary operators (+, -, *, / etc) with a single space on either side for readability.
Types
Enforce type hints in your codebase to improve readability and maintainability. This helps tools like mypy (used for static type checking).
Example:
```python
def multiply(a: int, b: int) -> int:
    return a * b
```

##Naming Conventions
- Variables: Use lowercase with words separated by underscores.
    `max_retries = 3`
  - Constants: Capitalize each word and separate using underscores.
    `MAX_CONNECTIONS = 100`
  - Functions & Methods: Lowercase with words separated by underscores.
```python
def send_email(to_address: str, body: str) -> None:
  pass
```
  - Classes: Capitalize the first letter of each word (CamelCase).
```pyhon
class UserAuthenticator:
  pass
```
  
## Error Handling
Handle exceptions gracefully. Use specific exceptions rather than catching a broad Exception which can hide many errors.

Example:
```python
try:
    response = requests.get('https://api.example.com/data')
    response.raise_for_status()
except requests.exceptions.HTTPError as e:
    print(f"HTTP error occurred: {e}")
```
