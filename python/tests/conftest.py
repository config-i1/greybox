"""Pytest configuration and fixtures."""
from pathlib import Path

import pandas
import pytest


def has_rpy2():
    """Check if rpy2 and R are available."""
    try:
        import rpy2.robjects as ro
        ro.r["library"]("greybox")
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def mtcars():
    """Load mtcars dataset from CSV."""
    return pandas.read_csv(Path(__file__).parent / "mtcars.csv")


pytest.mark.skipif_rpy2 = pytest.mark.skipif(
    not has_rpy2(), reason="Requires rpy2 and R"
)
