"""Pytest configuration and fixtures."""

import os
import shutil
from pathlib import Path

import pandas
import pytest


def has_rpy2():
    """Check if rpy2 and R are available."""
    if shutil.which("R") is None:
        return False
    try:
        import rpy2.robjects as ro

        ro.r["library"]("greybox")
        return True
    except Exception:
        return False


def is_ci_environment() -> bool:
    """Detect common CI environments (GitHub Actions, generic CI)."""
    return bool(
        os.environ.get("GITHUB_ACTIONS")
        or os.environ.get("CI")
        or os.environ.get("CONTINUOUS_INTEGRATION")
    )


def pytest_collection_modifyitems(config, items):
    """Skip R/Python comparison test files on CI environments.

    These tests require a full R installation and the greybox R package;
    they're intended to run locally during development. Detect CI via
    standard environment variables and skip any test in a file whose
    basename ends with ``_compare.py``.
    """
    if not is_ci_environment():
        return
    skip_marker = pytest.mark.skip(
        reason="R/Python comparison tests run only locally, not on CI"
    )
    for item in items:
        if str(item.fspath).endswith("_compare.py"):
            item.add_marker(skip_marker)


@pytest.fixture(scope="session")
def mtcars():
    """Load mtcars dataset from CSV."""
    return pandas.read_csv(Path(__file__).parent / "mtcars.csv")


pytest.mark.skipif_rpy2 = pytest.mark.skipif(
    not has_rpy2(), reason="Requires rpy2 and R"
)
