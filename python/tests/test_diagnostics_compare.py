"""Tests for diagnostic functions.

These tests check the outlier_dummy function works correctly
and produces results consistent with R's outlierdummy.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from greybox.alm import ALM
from greybox.formula import formula
from greybox.diagnostics import outlier_dummy


class TestOutlierDummy:
    """Test outlier_dummy function."""

    def test_smoke_dnorm(self):
        """Smoke test: outlier_dummy runs without error."""
        np.random.seed(42)
        n = 50
        x = np.random.randn(n)
        y = 2 + 3 * x + np.random.randn(n) * 0.5
        # Add an outlier
        y[0] = 100.0
        data = {"y": y, "x": x}
        y_vec, X = formula("y ~ x", data)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y_vec)
        result = outlier_dummy(model)

        assert hasattr(result, "outliers")
        assert hasattr(result, "statistic")
        assert hasattr(result, "id")
        assert hasattr(result, "level")
        assert hasattr(result, "type")
        assert hasattr(result, "errors")
        assert result.level == 0.999

    def test_detects_outlier(self):
        """Test that a clear outlier is detected."""
        np.random.seed(42)
        n = 50
        x = np.random.randn(n)
        y = 2 + 3 * x + np.random.randn(n) * 0.5
        # Add a very extreme outlier
        y[0] = 200.0
        data = {"y": y, "x": x}
        y_vec, X = formula("y ~ x", data)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y_vec)
        result = outlier_dummy(model)

        # The outlier at index 0 should be detected
        assert 0 in result.id

    def test_no_outliers(self):
        """Test with data that has no outliers."""
        np.random.seed(42)
        n = 30
        x = np.linspace(0, 1, n)
        y = 2 + 3 * x  # Perfect linear relationship, no noise
        data = {"y": y, "x": x}
        y_vec, X = formula("y ~ x", data)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y_vec)
        result = outlier_dummy(model)

        assert result.outliers is None
        assert len(result.id) == 0

    def test_rstudent_type(self):
        """Test with rstudent residual type."""
        np.random.seed(42)
        n = 50
        x = np.random.randn(n)
        y = 2 + 3 * x + np.random.randn(n) * 0.5
        y[0] = 200.0
        data = {"y": y, "x": x}
        y_vec, X = formula("y ~ x", data)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y_vec)
        result = outlier_dummy(model, type="rstudent")

        assert result.type == "rstudent"
        assert 0 in result.id

    def test_custom_level(self):
        """Test with a lower confidence level (more outliers)."""
        np.random.seed(42)
        n = 50
        x = np.random.randn(n)
        y = 2 + 3 * x + np.random.randn(n) * 2.0
        # Add moderate outliers
        y[0] = 30.0
        y[1] = -20.0
        data = {"y": y, "x": x}
        y_vec, X = formula("y ~ x", data)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y_vec)

        result_strict = outlier_dummy(model, level=0.999)
        result_loose = outlier_dummy(model, level=0.95)

        # Lower confidence level should detect at least as many outliers
        assert len(result_loose.id) >= len(result_strict.id)

    def test_not_fitted_raises(self):
        """Test that unfitted model raises error."""
        model = ALM(distribution="dnorm", loss="likelihood")
        with pytest.raises((ValueError, AttributeError)):
            outlier_dummy(model)

    def test_outlier_matrix_shape(self):
        """Test that outlier matrix has correct shape."""
        np.random.seed(42)
        n = 50
        x = np.random.randn(n)
        y = 2 + 3 * x + np.random.randn(n) * 0.5
        y[0] = 200.0
        y[1] = -100.0
        data = {"y": y, "x": x}
        y_vec, X = formula("y ~ x", data)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y_vec)
        result = outlier_dummy(model)

        if result.outliers is not None:
            assert result.outliers.shape[0] == n
            assert result.outliers.shape[1] == len(result.id)
