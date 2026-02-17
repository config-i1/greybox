"""Smoke tests for greybox.selection module."""

import numpy as np
import pytest
import pandas as pd

from greybox.selection import stepwise, lm_combine


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
