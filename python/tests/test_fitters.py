"""Tests for fitter functions."""

import numpy as np
from greybox.fitters import (
    scaler_internal,
    extractor_fitted,
    extractor_residuals,
    fitter,
    fitter_recursive,
    cf,
)


class TestScalerInternal:
    """Tests for scaler_internal function."""

    def test_scaler_internal_dnorm(self):
        """Test scale calculation for normal distribution."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mu = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        matrix_xreg = np.ones((5, 1))
        B = np.array([2.0])
        otU = np.array([True, True, True, True, True])

        result = scaler_internal(B, "dnorm", y, matrix_xreg, mu, None, otU, 5)
        expected = np.sqrt(np.mean((y - mu) ** 2))
        assert np.isclose(result, expected)

    def test_scaler_internal_dlaplace(self):
        """Test scale calculation for Laplace distribution."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mu = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        matrix_xreg = np.ones((5, 1))
        B = np.array([2.0])
        otU = np.array([True, True, True, True, True])

        result = scaler_internal(B, "dlaplace", y, matrix_xreg, mu, None, otU, 5)
        expected = np.mean(np.abs(y - mu))
        assert np.isclose(result, expected)

    def test_scaler_internal_dlnorm(self):
        """Test scale calculation for log-normal distribution."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mu = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        matrix_xreg = np.ones((5, 1))
        B = np.array([0.0])
        otU = np.array([True, True, True, True, True])

        result = scaler_internal(B, "dlnorm", y, matrix_xreg, mu, None, otU, 5)
        expected = np.sqrt(np.mean((np.log(y) - mu) ** 2))
        assert np.isclose(result, expected)

    def test_scaler_internal_default(self):
        """Test default scale for unknown distributions."""
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.0, 2.0, 3.0])
        matrix_xreg = np.ones((3, 1))
        B = np.array([1.0])
        otU = np.array([True, True, True])

        result = scaler_internal(B, "unknown", y, matrix_xreg, mu, None, otU, 3)
        assert result == 1.0


class TestExtractorFitted:
    """Tests for extractor_fitted function."""

    def test_extractor_fitted_dnorm(self):
        """Test fitted values for normal distribution."""
        mu = np.array([1.0, 2.0, 3.0])
        scale = 1.0
        result = extractor_fitted("dnorm", mu, scale)
        np.testing.assert_array_almost_equal(result, mu)

    def test_extractor_fitted_dlnorm(self):
        """Test fitted values for log-normal distribution."""
        mu = np.array([0.0, 1.0, 2.0])
        scale = 1.0
        result = extractor_fitted("dlnorm", mu, scale)
        expected = np.exp(mu)
        np.testing.assert_array_almost_equal(result, expected)

    def test_extractor_fitted_dbcnorm(self):
        """Test fitted values for Box-Cox normal distribution."""
        mu = np.array([0.0, 1.0, 2.0])
        scale = 1.0
        lambda_bc = 0.5
        from greybox.transforms import bc_transform_inv

        expected = bc_transform_inv(mu, lambda_bc)
        result = extractor_fitted("dbcnorm", mu, scale, lambda_bc)
        np.testing.assert_array_almost_equal(result, expected)

    def test_extractor_fitted_pnorm(self):
        """Test fitted values for probit (pnorm) distribution."""
        mu = np.array([0.0])
        scale = 1.0
        result = extractor_fitted("pnorm", mu, scale)
        assert result[0] == 0.5


class TestExtractorResiduals:
    """Tests for extractor_residuals function."""

    def test_extractor_residuals_dnorm(self):
        """Test residuals for normal distribution."""
        mu = np.array([2.0, 2.0, 2.0])
        y = np.array([1.0, 2.0, 3.0])
        result = extractor_residuals("dnorm", mu, y)
        expected = y - mu
        np.testing.assert_array_almost_equal(result, expected)

    def test_extractor_residuals_dlnorm(self):
        """Test residuals for log-normal distribution."""
        mu = np.array([0.0, 0.0, 0.0])
        y = np.array([1.0, 2.0, 3.0])
        result = extractor_residuals("dlnorm", mu, y)
        expected = np.log(y) - mu
        np.testing.assert_array_almost_equal(result, expected)

    def test_extractor_residuals_dbcnorm(self):
        """Test residuals for Box-Cox normal distribution."""
        mu = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 4.0, 9.0])
        lambda_bc = 0.5
        from greybox.transforms import bc_transform

        expected = bc_transform(y, lambda_bc) - mu
        result = extractor_residuals("dbcnorm", mu, y, lambda_bc)
        np.testing.assert_array_almost_equal(result, expected)


class TestFitter:
    """Tests for fitter function."""

    def test_fitter_basic(self):
        """Test basic fitting with normal distribution."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        X = np.column_stack([np.ones(5), y])
        B = np.array([0.0, 1.0])

        result = fitter(B, "dnorm", y, X)

        assert "mu" in result
        assert "scale" in result
        assert "other" in result
        assert "poly1" in result
        assert result["mu"].shape == y.shape

    def test_fitter_with_dlaplace(self):
        """Test fitting with Laplace distribution."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        X = np.column_stack([np.ones(5), y])
        B = np.array([0.0, 1.0])

        result = fitter(B, "dlaplace", y, X)

        assert "mu" in result
        assert "scale" in result


class TestCF:
    """Tests for cost function."""

    def test_cf_mse(self):
        """Test MSE loss function."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        X = np.column_stack([np.ones(5), y])
        B = np.array([0.0, 1.0])

        result = cf(B, "dnorm", "MSE", y, X)

        assert isinstance(result, (float, np.floating))
        assert result >= 0

    def test_cf_mae(self):
        """Test MAE loss function."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        X = np.column_stack([np.ones(5), y])
        B = np.array([0.0, 1.0])

        result = cf(B, "dnorm", "MAE", y, X)

        assert isinstance(result, (float, np.floating))
        assert result >= 0

    def test_cf_ham(self):
        """Test HAM loss function."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        X = np.column_stack([np.ones(5), y])
        B = np.array([0.0, 1.0])

        result = cf(B, "dnorm", "HAM", y, X)

        assert isinstance(result, (float, np.floating))
        assert result >= 0

    def test_cf_returns_scalar(self):
        """Test that cf returns a scalar."""
        y = np.array([1.0, 2.0, 3.0])
        X = np.column_stack([np.ones(3), y])
        B = np.array([0.0, 1.0])

        result = cf(B, "dnorm", "MSE", y, X)

        assert np.isscalar(result) or result.shape == ()
