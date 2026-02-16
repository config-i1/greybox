"""Tests for custom distribution functions."""

import numpy as np
from greybox.distributions import (
    ds,
    ps,
    qs,
    rs,
    dalaplace,
    palaplace,
    qalaplace,
    ralaplace,
    dbcnorm,
    pbcnorm,
    qbcnorm,
    rbcnorm,
    dfnorm,
    pfnorm,
    qfnorm,
    rfnorm,
    dlogitnorm,
    plogitnorm,
    qlogitnorm,
    rlogitnorm,
    drectnorm,
    prectnorm,
    qrectnorm,
    rrectnorm,
    dgnorm,
    pgnorm,
    qgnorm,
    rgnorm,
    plogis,
    pnorm,
    qnorm,
)


class TestS:
    """Tests for S-distribution."""

    def test_ds_basic(self):
        """Test basic density calculation."""
        y = np.array([0.0, 1.0, 2.0])
        result = ds(y, mu=1.0, scale=1.0)
        assert result.shape == y.shape
        assert np.all(result >= 0)

    def test_ds_log_false(self):
        """Test log=False returns density."""
        y = np.array([1.0])
        result = ds(y, mu=1.0, scale=1.0, log=False)
        assert result[0] >= 0

    def test_ds_log_true(self):
        """Test log=True returns log-density."""
        y = np.array([1.0])
        result = ds(y, mu=1.0, scale=1.0, log=True)
        assert np.isfinite(result[0])

    def test_ps_basic(self):
        """Test CDF."""
        y = np.array([0.0])
        result = ps(y, mu=0.0, scale=1.0)
        assert 0 <= result[0] <= 1

    def test_qs_basic(self):
        """Test quantile function."""
        p = np.array([0.5])
        result = qs(p, mu=0.0, scale=1.0)
        assert np.isfinite(result[0])

    def test_rs_basic(self):
        """Test random generation."""
        result = rs(5, mu=0.0, scale=1.0)
        assert len(result) == 5


class TestGnorm:
    """Tests for Generalized Normal distribution."""

    def test_dgnorm_basic(self):
        """Test basic density calculation."""
        y = np.array([0.0, 1.0, 2.0])
        result = dgnorm(y, mu=1.0, scale=1.0, shape=2.0)
        assert result.shape == y.shape
        assert np.all(result >= 0)

    def test_dgnorm_shape_one(self):
        """Test shape=1 (Laplace)."""
        y = np.array([0.0])
        result = dgnorm(y, mu=0.0, scale=1.0, shape=1.0)
        assert result[0] > 0

    def test_pgnorm_basic(self):
        """Test CDF."""
        y = np.array([0.0])
        result = pgnorm(y, mu=0.0, scale=1.0, shape=2.0)
        assert 0 <= result[0] <= 1

    def test_qgnorm_basic(self):
        """Test quantile function."""
        p = np.array([0.5])
        result = qgnorm(p, mu=0.0, scale=1.0, shape=2.0)
        assert np.isfinite(result[0])

    def test_rgnorm_basic(self):
        """Test random generation."""
        result = rgnorm(5, mu=0.0, scale=1.0, shape=2.0)
        assert len(result) == 5


class TestALaplace:
    """Tests for Asymmetric Laplace distribution."""

    def test_dalaplace_basic(self):
        """Test basic density calculation."""
        y = np.array([0.0, 1.0, 2.0])
        result = dalaplace(y, mu=1.0, scale=1.0)
        assert result.shape == y.shape
        assert np.all(result >= 0)

    def test_dalaplace_default_alpha(self):
        """Test default alpha=0.5."""
        y = np.array([1.0])
        result = dalaplace(y, mu=1.0, scale=1.0, alpha=0.5)
        assert result[0] > 0

    def test_dalaplace_asymmetric(self):
        """Test asymmetric alpha values."""
        y = np.array([1.0, 2.0])
        result_low = dalaplace(y, mu=1.0, scale=1.0, alpha=0.25)
        result_high = dalaplace(y, mu=1.0, scale=1.0, alpha=0.75)
        assert not np.array_equal(result_low, result_high)

    def test_palaplace_basic(self):
        """Test CDF."""
        y = np.array([0.0, 1.0, 2.0])
        result = palaplace(y, mu=1.0, scale=1.0)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_palaplace_at_mu(self):
        """Test CDF at mu equals alpha."""
        y = np.array([1.0])
        result = palaplace(y, mu=1.0, scale=1.0, alpha=0.5)
        assert result[0] == 0.5

    def test_qalaplace_basic(self):
        """Test quantile function."""
        p = np.array([0.25, 0.5, 0.75])
        result = qalaplace(p, mu=0.0, scale=1.0, alpha=0.5)
        assert result.shape == p.shape

    def test_qalaplace_bounds(self):
        """Test quantile at bounds."""
        result_0 = qalaplace(0.0, mu=0.0, scale=1.0, alpha=0.5)
        result_1 = qalaplace(1.0, mu=0.0, scale=1.0, alpha=0.5)
        assert np.isinf(result_1)
        assert result_0 == -np.inf

    def test_ralaplace_basic(self):
        """Test random generation."""
        result = ralaplace(5, mu=0.0, scale=1.0, alpha=0.5)
        assert len(result) == 5


class TestBcnorm:
    """Tests for Box-Cox Normal distribution."""

    def test_dbcnorm_basic(self):
        """Test basic density calculation."""
        y = np.array([1.0, 2.0, 3.0])
        result = dbcnorm(y, mu=0.0, sigma=1.0, lambda_bc=0.5)
        assert result.shape == y.shape
        assert np.all(result >= 0)

    def test_dbcnorm_lambda_zero(self):
        """Test lambda=0 (log-normal case)."""
        y = np.array([1.0, 2.0, 3.0])
        result = dbcnorm(y, mu=0.0, sigma=1.0, lambda_bc=0.0)
        assert result.shape == y.shape
        assert np.all(result >= 0)

    def test_dbcnorm_lambda_one(self):
        """Test lambda=1 (normal case)."""
        y = np.array([1.0, 2.0, 3.0])
        result = dbcnorm(y, mu=0.0, sigma=1.0, lambda_bc=1.0)
        assert result.shape == y.shape

    def test_dbcnorm_log(self):
        """Test log=True returns log-density."""
        y = np.array([1.0, 2.0])
        result = dbcnorm(y, mu=0.0, sigma=1.0, lambda_bc=0.5, log=True)
        assert np.all(np.isfinite(result))

    def test_pbcnorm_basic(self):
        """Test CDF."""
        y = np.array([1.0, 2.0, 3.0])
        result = pbcnorm(y, mu=0.0, sigma=1.0, lambda_bc=0.5)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_pbcnorm_zero(self):
        """Test CDF at zero."""
        y = np.array([0.0])
        result = pbcnorm(y, mu=0.0, sigma=1.0, lambda_bc=0.5)
        assert result[0] == 0

    def test_qbcnorm_basic(self):
        """Test quantile function."""
        p = np.array([0.25, 0.5, 0.75])
        result = qbcnorm(p, mu=0.0, sigma=1.0, lambda_bc=0.5)
        assert result.shape == p.shape
        assert np.all(result >= 0)

    def test_rbcnorm_basic(self):
        """Test random generation."""
        result = rbcnorm(5, mu=0.0, sigma=1.0, lambda_bc=0.5)
        assert len(result) == 5
        assert np.all(result >= 0)


class TestFnorm:
    """Tests for Folded Normal distribution."""

    def test_dfnorm_basic(self):
        """Test basic density calculation."""
        y = np.array([0.0, 1.0, 2.0])
        result = dfnorm(y, mu=1.0, sigma=1.0)
        assert result.shape == y.shape
        assert np.all(result >= 0)

    def test_dfnorm_negative_input(self):
        """Test negative input returns 0."""
        y = np.array([-1.0])
        result = dfnorm(y, mu=1.0, sigma=1.0)
        assert result[0] == 0

    def test_dfnorm_at_zero(self):
        """Test at y=0."""
        y = np.array([0.0])
        result = dfnorm(y, mu=1.0, sigma=1.0)
        assert result[0] >= 0

    def test_pfnorm_basic(self):
        """Test CDF."""
        y = np.array([0.0, 1.0, 2.0])
        result = pfnorm(y, mu=1.0, sigma=1.0)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_pfnorm_negative(self):
        """Test CDF for negative values."""
        y = np.array([-1.0])
        result = pfnorm(y, mu=1.0, sigma=1.0)
        assert result[0] == 0

    def test_qfnorm_basic(self):
        """Test quantile function."""
        p = np.array([0.25, 0.5, 0.75])
        result = qfnorm(p, mu=1.0, sigma=1.0)
        assert result.shape == p.shape
        assert np.all(result >= 0)

    def test_qfnorm_at_bounds(self):
        """Test quantile at bounds."""
        result_0 = qfnorm(0.0, mu=1.0, sigma=1.0)
        result_1 = qfnorm(1.0, mu=1.0, sigma=1.0)
        assert result_0 == 0
        assert np.isinf(result_1)

    def test_rfnorm_basic(self):
        """Test random generation."""
        result = rfnorm(5, mu=1.0, sigma=1.0)
        assert len(result) == 5
        assert np.all(result >= 0)


class TestLogitnorm:
    """Tests for Logit-Normal distribution."""

    def test_dlogitnorm_basic(self):
        """Test basic density calculation."""
        y = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        result = dlogitnorm(y, mu=0.0, sigma=1.0)
        assert result.shape == y.shape
        assert np.all(result >= 0)

    def test_dlogitnorm_log(self):
        """Test log=True returns log-density."""
        y = np.array([0.3, 0.5, 0.7])
        result = dlogitnorm(y, mu=0.0, sigma=1.0, log=True)
        assert np.all(np.isfinite(result))

    def test_dlogitnorm_at_extremes(self):
        """Test at extreme values."""
        y = np.array([0.001, 0.999])
        result = dlogitnorm(y, mu=0.0, sigma=1.0)
        assert np.all(result >= 0)

    def test_plogitnorm_basic(self):
        """Test CDF."""
        y = np.array([0.1, 0.5, 0.9])
        result = plogitnorm(y, mu=0.0, sigma=1.0)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_plogitnorm_bounds(self):
        """Test CDF at bounds."""
        y_low = np.array([-0.1])
        y_high = np.array([1.1])
        result_low = plogitnorm(y_low, mu=0.0, sigma=1.0)
        result_high = plogitnorm(y_high, mu=0.0, sigma=1.0)
        assert result_low[0] == 0
        assert result_high[0] == 1

    def test_qlogitnorm_basic(self):
        """Test quantile function."""
        p = np.array([0.25, 0.5, 0.75])
        result = qlogitnorm(p, mu=0.0, sigma=1.0)
        assert result.shape == p.shape
        assert np.all(result > 0)
        assert np.all(result < 1)

    def test_rlogitnorm_basic(self):
        """Test random generation."""
        result = rlogitnorm(5, mu=0.0, sigma=1.0)
        assert len(result) == 5
        assert np.all(result > 0)
        assert np.all(result < 1)


class TestRectnorm:
    """Tests for Rectified Normal distribution."""

    def test_drectnorm_basic(self):
        """Test basic density calculation."""
        y = np.array([-1.0, 0.0, 1.0])
        result = drectnorm(y, mu=1.0, sigma=1.0)
        assert result.shape == y.shape
        assert np.all(result >= 0)

    def test_drectnorm_positive_only(self):
        """Test that output is non-negative."""
        y = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        result = drectnorm(y, mu=1.0, sigma=1.0)
        assert np.all(result >= 0)

    def test_drectnorm_at_zero(self):
        """Test at y=0."""
        y = np.array([0.0])
        result = drectnorm(y, mu=1.0, sigma=1.0)
        assert result[0] >= 0

    def test_prectnorm_basic(self):
        """Test CDF."""
        y = np.array([-1.0, 0.0, 1.0])
        result = prectnorm(y, mu=1.0, sigma=1.0)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_prectnorm_at_zero(self):
        """Test CDF at zero."""
        y = np.array([0.0])
        result = prectnorm(y, mu=1.0, sigma=1.0)
        assert result[0] > 0

    def test_qrectnorm_basic(self):
        """Test quantile function."""
        p = np.array([0.25, 0.5, 0.75])
        result = qrectnorm(p, mu=1.0, sigma=1.0)
        assert result.shape == p.shape
        assert np.all(result >= 0)

    def test_rrectnorm_basic(self):
        """Test random generation."""
        result = rrectnorm(5, mu=1.0, sigma=1.0)
        assert len(result) == 5
        assert np.all(result >= 0)


class TestHelperFunctions:
    """Tests for helper distribution functions."""

    def test_plogis(self):
        """Test logistic CDF."""
        y = np.array([0.0])
        result = plogis(y, location=0.0, scale=1.0)
        assert result[0] == 0.5

    def test_pnorm(self):
        """Test normal CDF."""
        y = np.array([0.0])
        result = pnorm(y, mean=0.0, sd=1.0)
        assert result[0] == 0.5

    def test_pnorm_lower_tail_false(self):
        """Test upper tail."""
        y = np.array([0.0])
        result = pnorm(y, mean=0.0, sd=1.0, lower_tail=False)
        assert result[0] == 0.5

    def test_qnorm(self):
        """Test normal quantile function."""
        p = np.array([0.5])
        result = qnorm(p, mean=0.0, sd=1.0)
        np.testing.assert_array_almost_equal(result, np.array([0.0]))
