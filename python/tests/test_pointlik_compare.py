"""Tests comparing pointLik between R and Python.

These tests ensure that the Python point_lik() function produces identical
results to R's pointLik() for various distributions.
"""
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("rpy2.robjects", reason="Requires rpy2 and R")

import rpy2.robjects as ro

from greybox.alm import ALM
from greybox.formula import formula
from greybox.pointlik import point_lik

ro.r["library"]("greybox")


def _get_mtcars():
    """Load mtcars dataset from R."""
    mtcars = ro.r["mtcars"]
    col_names = list(mtcars.colnames)
    data = {}
    for i, name in enumerate(col_names):
        data[name] = np.array(mtcars.rx2(name))
    return data


class TestPointLikVsR:
    """Compare point_lik against R's pointLik."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = _get_mtcars()

    def test_dnorm(self):
        """Test point likelihood for normal distribution."""
        y, X = formula("mpg ~ wt + am", self.data)

        # Python
        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y)
        py_result = point_lik(model, log=True)

        # R
        r_code = """
        data(mtcars)
        r_model <- alm(mpg ~ wt + am, data=mtcars, distribution="dnorm")
        pointLik(r_model)
        """
        r_result = np.array(ro.r(r_code))

        assert_allclose(py_result, r_result, rtol=1e-3)

    def test_dlaplace(self):
        """Test point likelihood for Laplace distribution."""
        y, X = formula("mpg ~ wt + am", self.data)

        model = ALM(distribution="dlaplace", loss="likelihood")
        model.fit(X, y)
        py_result = point_lik(model, log=True)

        r_code = """
        data(mtcars)
        r_model <- alm(mpg ~ wt + am, data=mtcars, distribution="dlaplace")
        pointLik(r_model)
        """
        r_result = np.array(ro.r(r_code))

        assert_allclose(py_result, r_result, rtol=1e-3)

    def test_dlogis(self):
        """Test point likelihood for logistic distribution."""
        y, X = formula("mpg ~ wt + am", self.data)

        model = ALM(distribution="dlogis", loss="likelihood")
        model.fit(X, y)
        py_result = point_lik(model, log=True)

        r_code = """
        data(mtcars)
        r_model <- alm(mpg ~ wt + am, data=mtcars, distribution="dlogis")
        pointLik(r_model)
        """
        r_result = np.array(ro.r(r_code))

        assert_allclose(py_result, r_result, rtol=1e-3)

    def test_dlnorm(self):
        """Test point likelihood for log-normal distribution."""
        y, X = formula("mpg ~ wt + am", self.data)

        model = ALM(distribution="dlnorm", loss="likelihood")
        model.fit(X, y)
        py_result = point_lik(model, log=True)

        r_code = """
        data(mtcars)
        r_model <- alm(mpg ~ wt + am, data=mtcars, distribution="dlnorm")
        pointLik(r_model)
        """
        r_result = np.array(ro.r(r_code))

        assert_allclose(py_result, r_result, rtol=1e-3)

    def test_log_false(self):
        """Test point likelihood with log=False."""
        y, X = formula("mpg ~ wt + am", self.data)

        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y)
        py_log = point_lik(model, log=True)
        py_no_log = point_lik(model, log=False)

        # exp(log_lik) should equal lik
        assert_allclose(np.exp(py_log), py_no_log, rtol=1e-10)
