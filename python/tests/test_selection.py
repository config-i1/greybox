"""Smoke tests for greybox.selection module."""

import numpy as np
import pytest
import pandas as pd

import rpy2.robjects as ro

from greybox.selection import stepwise, lm_combine


def _load_mtcars():
    """Load mtcars dataset from R."""
    ro.r('data(mtcars, package="datasets")')
    mtcars_r = ro.r("mtcars")
    return pd.DataFrame({col: list(mtcars_r.rx2(col)) for col in mtcars_r.names})


@pytest.fixture(scope="module")
def mtcars():
    return _load_mtcars()


class TestStepwise:
    def test_smoke_mtcars(self, mtcars):
        data = {
            "mpg": mtcars["mpg"].tolist(),
            "cyl": mtcars["cyl"].tolist(),
            "disp": mtcars["disp"].tolist(),
            "hp": mtcars["hp"].tolist(),
            "wt": mtcars["wt"].tolist(),
        }
        model = stepwise(data, ic="AICc", distribution="dnorm")
        assert model is not None
        assert model.coef is not None
        assert model.nobs == 32

    def test_smoke_simple(self):
        np.random.seed(42)
        n = 50
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = 2 * x1 + 0.5 * x2 + np.random.randn(n) * 0.1
        data = {"y": y.tolist(), "x1": x1.tolist(), "x2": x2.tolist()}
        model = stepwise(data)
        assert model is not None
        assert model.coef is not None

    def test_not_dict_raises(self):
        with pytest.raises(ValueError):
            stepwise("not a dict")

    def test_no_predictors_raises(self):
        with pytest.raises(ValueError):
            stepwise({"y": [1, 2, 3]})


class TestLmCombine:
    def test_smoke_mtcars(self, mtcars):
        data = {
            "mpg": mtcars["mpg"].tolist(),
            "cyl": mtcars["cyl"].tolist(),
            "hp": mtcars["hp"].tolist(),
            "wt": mtcars["wt"].tolist(),
        }
        result = lm_combine(data, ic="AICc", distribution="dnorm")
        assert "coefficients" in result
        assert "fitted" in result
        assert "residuals" in result
        assert "R2" in result
        assert len(result["fitted"]) == 32

    def test_smoke_simple(self):
        np.random.seed(42)
        n = 50
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = 2 * x1 + 0.5 * x2 + np.random.randn(n) * 0.1
        data = {"y": y.tolist(), "x1": x1.tolist(), "x2": x2.tolist()}
        result = lm_combine(data)
        assert "coefficients" in result
        assert len(result["coefficients"]) == 3  # intercept + 2 vars

    def test_not_dict_raises(self):
        with pytest.raises(ValueError):
            lm_combine("not a dict")
