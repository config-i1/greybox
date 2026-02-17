"""Pytest configuration and fixtures."""
import pytest


def has_rpy2():
    """Check if rpy2 and R are available."""
    try:
        import rpy2.robjects as ro
        ro.r["library"]("greybox")
        return True
    except Exception:
        return False


pytest.mark.skipif_rpy2 = pytest.mark.skipif(
    not has_rpy2(), reason="Requires rpy2 and R"
)
