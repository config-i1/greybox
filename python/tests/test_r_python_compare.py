"""Tests comparing R and Python distribution functions.

These tests ensure that the Python implementations produce identical results
to the R greybox package functions.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

# Import Python distribution functions
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
    dllaplace,
    qllaplace,
    dls,
    qls,
    dlgnorm,
    qlgnorm,
)


# Set up R environment
import rpy2.robjects as ro

ro.r["library"]("greybox")


def to_r_array(arr):
    """Convert numpy array to R vector."""
    return ro.FloatVector(arr.tolist())


def call_r_func(func_name, *args, **kwargs):
    """Call an R function from greybox package."""
    r_args = [to_r_array(a) if isinstance(a, np.ndarray) else a for a in args]
    r_kwargs = {
        k: (to_r_array(v) if isinstance(v, np.ndarray) else v)
        for k, v in kwargs.items()
    }
    return np.array(ro.r[func_name](*r_args, **r_kwargs))


class TestSvsR:
    """Compare S-distribution between R and Python."""

    def test_ds(self):
        """Test density function."""
        q = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        mu, scale = 1.0, 1.0

        py_result = ds(q, mu=mu, scale=scale)
        r_result = call_r_func("ds", q, mu, scale)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_ds_log(self):
        """Test density function with log=True."""
        q = np.array([0.5, 1.0, 1.5])
        mu, scale = 1.0, 1.0

        py_result = ds(q, mu=mu, scale=scale, log=True)
        r_result = call_r_func("ds", q, mu, scale, log=True)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_ps(self):
        """Test CDF function."""
        q = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        mu, scale = 1.0, 1.0

        py_result = ps(q, mu=mu, scale=scale)
        r_result = call_r_func("ps", q, mu, scale)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_qs(self):
        """Test quantile function."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        mu, scale = 1.0, 1.0

        py_result = qs(p, mu=mu, scale=scale)
        r_result = call_r_func("qs", p, mu, scale)

        assert_allclose(py_result, r_result, rtol=1e-10)


class TestGnormvsR:
    """Compare Generalized Normal distribution between R and Python."""

    def test_dgnorm(self):
        """Test density function."""
        q = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        mu, scale, shape = 0.0, 1.0, 2.0

        py_result = dgnorm(q, mu=mu, scale=scale, shape=shape)
        r_result = call_r_func("dgnorm", q, mu, scale, shape)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_dgnorm_log(self):
        """Test density function with log=True."""
        q = np.array([0.5, 1.0, 1.5])
        mu, scale, shape = 0.0, 1.0, 2.0

        py_result = dgnorm(q, mu=mu, scale=scale, shape=shape, log=True)
        r_result = call_r_func("dgnorm", q, mu, scale, shape, log=True)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_pgnorm(self):
        """Test CDF function."""
        q = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        mu, scale, shape = 0.0, 1.0, 2.0

        py_result = pgnorm(q, mu=mu, scale=scale, shape=shape)
        r_result = call_r_func("pgnorm", q, mu, scale, shape)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_qgnorm(self):
        """Test quantile function."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        mu, scale, shape = 0.0, 1.0, 2.0

        py_result = qgnorm(p, mu=mu, scale=scale, shape=shape)
        r_result = call_r_func("qgnorm", p, mu, scale, shape)

        assert_allclose(py_result, r_result, rtol=1e-10)


class TestALapacevsR:
    """Compare Asymmetric Laplace distribution between R and Python."""

    def test_dalaplace(self):
        """Test density function."""
        q = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        mu, scale, alpha = 0.0, 1.0, 0.5

        py_result = dalaplace(q, mu=mu, scale=scale, alpha=alpha)
        r_result = call_r_func("dalaplace", q, mu, scale, alpha)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_dalaplace_log(self):
        """Test density function with log=True."""
        q = np.array([0.5, 1.0, 1.5])
        mu, scale, alpha = 0.0, 1.0, 0.5

        py_result = dalaplace(q, mu=mu, scale=scale, alpha=alpha, log=True)
        r_result = call_r_func("dalaplace", q, mu, scale, alpha, log=True)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_palaplace(self):
        """Test CDF function."""
        q = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        mu, scale, alpha = 0.0, 1.0, 0.5

        py_result = palaplace(q, mu=mu, scale=scale, alpha=alpha)
        r_result = call_r_func("palaplace", q, mu, scale, alpha)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_qalaplace(self):
        """Test quantile function."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        mu, scale, alpha = 0.0, 1.0, 0.5

        py_result = qalaplace(p, mu=mu, scale=scale, alpha=alpha)
        r_result = call_r_func("qalaplace", p, mu, scale, alpha)

        assert_allclose(py_result, r_result, rtol=1e-10)


class TestBcnormvsR:
    """Compare Box-Cox Normal distribution between R and Python."""

    def test_dbcnorm(self):
        """Test density function."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        mu, sigma, lambda_bc = 1.0, 1.0, 0.5

        py_result = dbcnorm(q, mu=mu, sigma=sigma, lambda_bc=lambda_bc)
        r_result = call_r_func("dbcnorm", q, mu, sigma, lambda_bc)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_dbcnorm_log(self):
        """Test density function with log=True."""
        q = np.array([0.5, 1.0, 1.5, 2.0])
        mu, sigma, lambda_bc = 1.0, 1.0, 0.5

        py_result = dbcnorm(q, mu=mu, sigma=sigma, lambda_bc=lambda_bc, log=True)
        r_result = call_r_func("dbcnorm", q, mu, sigma, lambda_bc, log=True)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_pbcnorm(self):
        """Test CDF function."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        mu, sigma, lambda_bc = 1.0, 1.0, 0.5

        py_result = pbcnorm(q, mu=mu, sigma=sigma, lambda_bc=lambda_bc)
        r_result = call_r_func("pbcnorm", q, mu, sigma, lambda_bc)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_qbcnorm(self):
        """Test quantile function."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        mu, sigma, lambda_bc = 1.0, 1.0, 0.5

        py_result = qbcnorm(p, mu=mu, sigma=sigma, lambda_bc=lambda_bc)
        r_result = call_r_func("qbcnorm", p, mu, sigma, lambda_bc)

        assert_allclose(py_result, r_result, rtol=1e-10)


class TestFnormvsR:
    """Compare Folded Normal distribution between R and Python."""

    def test_dfnorm(self):
        """Test density function."""
        q = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        mu, sigma = 1.0, 1.0

        py_result = dfnorm(q, mu=mu, sigma=sigma)
        r_result = call_r_func("dfnorm", q, mu, sigma)

        assert_allclose(py_result, r_result, rtol=1e-9)

    def test_dfnorm_log(self):
        """Test density function with log=True."""
        q = np.array([0.5, 1.0, 1.5])
        mu, sigma = 1.0, 1.0

        py_result = dfnorm(q, mu=mu, sigma=sigma, log=True)
        r_result = call_r_func("dfnorm", q, mu, sigma, log=True)

        assert_allclose(py_result, r_result, rtol=1e-9)

    def test_pfnorm(self):
        """Test CDF function."""
        q = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        mu, sigma = 1.0, 1.0

        py_result = pfnorm(q, mu=mu, sigma=sigma)
        r_result = call_r_func("pfnorm", q, mu, sigma)

        assert_allclose(py_result, r_result, rtol=1e-9)

    def test_qfnorm(self):
        """Test quantile function."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        mu, sigma = 1.0, 1.0

        py_result = qfnorm(p, mu=mu, sigma=sigma)
        r_result = call_r_func("qfnorm", p, mu, sigma)

        assert_allclose(py_result, r_result, rtol=1e-6)


class TestLogitnormvsR:
    """Compare Logit-Normal distribution between R and Python."""

    def test_dlogitnorm(self):
        """Test density function."""
        q = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        mu, sigma = 0.0, 1.0

        py_result = dlogitnorm(q, mu=mu, sigma=sigma)
        r_result = call_r_func("dlogitnorm", q, mu, sigma)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_dlogitnorm_log(self):
        """Test density function with log=True."""
        q = np.array([0.1, 0.25, 0.5, 0.75])
        mu, sigma = 0.0, 1.0

        py_result = dlogitnorm(q, mu=mu, sigma=sigma, log=True)
        r_result = call_r_func("dlogitnorm", q, mu, sigma, log=True)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_plogitnorm(self):
        """Test CDF function."""
        q = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        mu, sigma = 0.0, 1.0

        py_result = plogitnorm(q, mu=mu, sigma=sigma)
        r_result = call_r_func("plogitnorm", q, mu, sigma)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_qlogitnorm(self):
        """Test quantile function."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        mu, sigma = 0.0, 1.0

        py_result = qlogitnorm(p, mu=mu, sigma=sigma)
        r_result = call_r_func("qlogitnorm", p, mu, sigma)

        assert_allclose(py_result, r_result, rtol=1e-10)


class TestRectnormvsR:
    """Compare Rectified Normal distribution between R and Python."""

    def test_drectnorm(self):
        """Test density function."""
        q = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        mu, sigma = 1.0, 1.0

        py_result = drectnorm(q, mu=mu, sigma=sigma)
        r_result = call_r_func("drectnorm", q, mu, sigma)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_drectnorm_log(self):
        """Test density function with log=True."""
        q = np.array([0.5, 1.0, 1.5])
        mu, sigma = 1.0, 1.0

        py_result = drectnorm(q, mu=mu, sigma=sigma, log=True)
        r_result = call_r_func("drectnorm", q, mu, sigma, log=True)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_prectnorm(self):
        """Test CDF function."""
        q = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        mu, sigma = 1.0, 1.0

        py_result = prectnorm(q, mu=mu, sigma=sigma)
        r_result = call_r_func("prectnorm", q, mu, sigma)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_qrectnorm(self):
        """Test quantile function."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        mu, sigma = 1.0, 1.0

        py_result = qrectnorm(p, mu=mu, sigma=sigma)
        r_result = call_r_func("qrectnorm", p, mu, sigma)

        assert_allclose(py_result, r_result, rtol=1e-10)


class TestLLapacevsR:
    """Compare Log-Laplace distribution between R and Python.

    Note: R greybox does not have dllaplace, qllaplace functions.
    These tests are for Python implementation verification only.
    """

    def test_dllaplace_python_only(self):
        """Test density function (R does not have this)."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        mu, scale = 1.0, 1.0

        py_result = dllaplace(q, loc=mu, scale=scale)
        assert py_result.shape == q.shape
        assert np.all(np.isfinite(py_result))

    def test_dllaplace_log_python_only(self):
        """Test density function with log=True (R does not have this)."""
        q = np.array([0.5, 1.0, 1.5, 2.0])
        mu, scale = 1.0, 1.0

        py_result = dllaplace(q, loc=mu, scale=scale, log=True)
        assert py_result.shape == q.shape
        assert np.all(np.isfinite(py_result))

    def test_qllaplace_python_only(self):
        """Test quantile function (R does not have this)."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        mu, scale = 1.0, 1.0

        py_result = qllaplace(p, loc=mu, scale=scale)
        assert py_result.shape == p.shape
        assert np.all(np.isfinite(py_result))
        assert np.all(py_result > 0)


class TestLSvsR:
    """Compare Log-S distribution between R and Python.

    Note: R greybox does not have dls, qls functions.
    These tests are for Python implementation verification only.
    """

    def test_dls_python_only(self):
        """Test density function (R does not have this)."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        mu, scale = 1.0, 1.0

        py_result = dls(q, loc=mu, scale=scale)
        assert py_result.shape == q.shape
        assert np.all(np.isfinite(py_result))

    def test_dls_log_python_only(self):
        """Test density function with log=True (R does not have this)."""
        q = np.array([0.5, 1.0, 1.5, 2.0])
        mu, scale = 1.0, 1.0

        py_result = dls(q, loc=mu, scale=scale, log=True)
        assert py_result.shape == q.shape
        assert np.all(np.isfinite(py_result))

    def test_qls_python_only(self):
        """Test quantile function (R does not have this)."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        mu, scale = 1.0, 1.0

        py_result = qls(p, loc=mu, scale=scale)
        assert py_result.shape == p.shape
        assert np.all(np.isfinite(py_result))
        assert np.all(py_result > 0)


class TestLGnormvsR:
    """Compare Log-Generalized Normal distribution between R and Python.

    Note: R greybox does not have dlgnorm, qlgnorm functions.
    These tests are for Python implementation verification only.
    """

    def test_dlgnorm_python_only(self):
        """Test density function (R does not have this)."""
        q = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        mu, scale, shape = 1.0, 1.0, 2.0

        py_result = dlgnorm(q, mu=mu, scale=scale, shape=shape)
        assert py_result.shape == q.shape
        assert np.all(np.isfinite(py_result))

    def test_dlgnorm_log_python_only(self):
        """Test density function with log=True (R does not have this)."""
        q = np.array([0.5, 1.0, 1.5, 2.0])
        mu, scale, shape = 1.0, 1.0, 2.0

        py_result = dlgnorm(q, mu=mu, scale=scale, shape=shape, log=True)
        assert py_result.shape == q.shape
        assert np.all(np.isfinite(py_result))

    def test_qlgnorm_python_only(self):
        """Test quantile function (R does not have this)."""
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        mu, scale, shape = 1.0, 1.0, 2.0

        py_result = qlgnorm(p, mu=mu, scale=scale, shape=shape)
        assert py_result.shape == p.shape
        assert np.all(np.isfinite(py_result))
        assert np.all(py_result > 0)


class TestEdgeCases:
    """Test edge cases and special parameter values."""

    def test_gnorm_shape_2(self):
        """Test gnorm with shape=2 (similar to normal)."""
        q = np.array([-1, 0, 1])
        mu, scale, shape = 0, 1, 2

        py_result = dgnorm(q, mu=mu, scale=scale, shape=shape)
        r_result = call_r_func("dgnorm", q, mu, scale, shape)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_bcnorm_lambda_zero(self):
        """Test bcnorm with lambda=0 (log transform)."""
        q = np.array([0.5, 1.0, 1.5, 2.0])
        mu, sigma, lambda_bc = 1.0, 1.0, 0.0

        py_result = dbcnorm(q, mu=mu, sigma=sigma, lambda_bc=lambda_bc)
        r_result = call_r_func("dbcnorm", q, mu, sigma, lambda_bc)

        assert_allclose(py_result, r_result, rtol=1e-10)

    def test_bcnorm_lambda_one(self):
        """Test bcnorm with lambda=1 (identity transform)."""
        q = np.array([0.5, 1.0, 1.5, 2.0])
        mu, sigma, lambda_bc = 1.0, 1.0, 1.0

        py_result = dbcnorm(q, mu=mu, sigma=sigma, lambda_bc=lambda_bc)
        r_result = call_r_func("dbcnorm", q, mu, sigma, lambda_bc)

        assert_allclose(py_result, r_result, rtol=1e-10)
