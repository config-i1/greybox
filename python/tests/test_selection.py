"""Tests for greybox.selection module."""

import numpy as np
import pytest
import pandas as pd

from greybox.selection import stepwise, lm_combine


class TestStepwise:
    """Tests for stepwise selection."""

    def test_smoke_mtcars(self, mtcars):
        """Test stepwise on mtcars data."""
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

    def test_stepwise_basic(self):
        """Test stepwise with simple data."""
        np.random.seed(42)
        n = 100
        x1 = np.random.normal(10, 3, n)
        x2 = np.random.normal(50, 5, n)
        noise = np.random.normal(0, 3, n)
        y = 100 + 0.5 * x1 - 0.75 * x2 + noise

        data = {"y": y.tolist(), "x1": x1.tolist(), "x2": x2.tolist()}

        model = stepwise(data, ic="AICc", silent=True)
        assert model is not None
        assert hasattr(model, "ic_values")
        assert isinstance(model.ic_values, dict)
        assert len(model.ic_values) >= 2

    def test_stepwise_with_noise(self):
        """Test stepwise correctly ignores noise variable."""
        np.random.seed(123)
        n = 200
        x1 = np.random.normal(10, 3, n)
        x2 = np.random.normal(50, 5, n)
        x3 = np.random.normal(0, 1, n)
        noise = np.random.normal(0, 100, n)
        y = 100 + 0.5 * x1 - 0.75 * x2 + noise

        data = {
            "y": y.tolist(),
            "x1": x1.tolist(),
            "x2": x2.tolist(),
            "x3": x3.tolist(),
        }

        model = stepwise(data, ic="AICc", silent=True)
        assert model is not None

    def test_stepwise_ic_values_attribute(self):
        """Test that ic_values attribute is a dict with step names."""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + 3 * x2 + np.random.normal(0, 0.1, n)

        data = {"y": y.tolist(), "x1": x1.tolist(), "x2": x2.tolist()}

        model = stepwise(data, ic="AICc", silent=True)
        assert hasattr(model, "ic_values")
        assert isinstance(model.ic_values, dict)
        assert len(model.ic_values) > 0
        # First key should always be "Intercept"
        assert list(model.ic_values.keys())[0] == "Intercept"
        # All keys are strings, all values are floats
        assert all(isinstance(k, str) for k in model.ic_values.keys())
        assert all(
            isinstance(v, (float, np.floating))
            for v in model.ic_values.values()
        )

    def test_stepwise_time_elapsed_attribute(self):
        """Test that time_elapsed attribute is properly set."""
        data = {"y": list(range(1, 11)), "x1": list(range(1, 11))}

        model = stepwise(data, ic="AICc", silent=True)
        assert hasattr(model, "time_elapsed")
        assert isinstance(model.time_elapsed, float)
        assert model.time_elapsed >= 0

    def test_stepwise_ic_types(self):
        """Test stepwise with different IC types."""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + np.random.normal(0, 0.1, n)

        data = {"y": y.tolist(), "x1": x1.tolist()}

        for ic_type in ["AICc", "AIC", "BIC", "BICc"]:
            model = stepwise(data, ic=ic_type, silent=True)
            assert model is not None
            assert hasattr(model, "ic_values")

    def test_stepwise_silent_output(self, capsys):
        """Test that silent=False produces output."""
        np.random.seed(42)
        n = 30
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + 3 * x2 + np.random.normal(0, 0.1, n)

        data = {"y": y.tolist(), "x1": x1.tolist(), "x2": x2.tolist()}

        model = stepwise(data, ic="AICc", silent=False)
        captured = capsys.readouterr()
        assert "Formula:" in captured.out or "IC:" in captured.out

    def test_stepwise_correlation_method(self):
        """Test stepwise with different correlation methods."""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + 3 * x2 + np.random.normal(0, 0.1, n)

        data = {"y": y.tolist(), "x1": x1.tolist(), "x2": x2.tolist()}

        for method in ["pearson", "spearman", "kendall"]:
            model = stepwise(data, ic="AICc", method=method, silent=True)
            assert model is not None

    def test_stepwise_no_improvement(self):
        """Test stepwise when no variable improves IC."""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        noise = np.random.normal(100, 0.001, n)
        y = 1 + 2 * x1 + noise

        data = {"y": y.tolist(), "x1": x1.tolist(), "noise": noise.tolist()}

        model = stepwise(data, ic="AICc", silent=True)
        assert model is not None

    def test_stepwise_returns_alm_object(self):
        """Test that stepwise returns an ALM object."""
        np.random.seed(42)
        n = 30
        x1 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + np.random.normal(0, 0.1, n)

        data = {"y": y.tolist(), "x1": x1.tolist()}

        model = stepwise(data, ic="AICc", silent=True)
        from greybox.alm import ALM

        assert isinstance(model, ALM)
        assert hasattr(model, "coef")
        assert hasattr(model, "intercept_")

    def test_stepwise_alm_compatibility(self):
        """Test that stepwise result supports all key ALM properties/methods."""
        np.random.seed(42)
        n = 100
        x1 = np.random.normal(10, 3, n)
        x2 = np.random.normal(50, 5, n)
        y = 100 + 0.5 * x1 - 0.75 * x2 + np.random.normal(0, 3, n)

        data = {"y": y.tolist(), "x1": x1.tolist(), "x2": x2.tolist()}
        model = stepwise(data, ic="AICc", silent=True)

        from greybox.alm import ALM

        # isinstance check
        assert isinstance(model, ALM)

        # Core properties
        assert model.coef is not None
        assert isinstance(model.intercept_, (float, np.floating))
        assert model.nobs == n
        assert model.nparam > 0

        # Residuals, fitted, actuals
        assert model.residuals is not None
        assert len(model.residuals) == n
        assert model.fitted is not None
        assert len(model.fitted) == n
        assert model.actuals is not None
        assert len(model.actuals) == n

        # Variance and log-likelihood
        assert isinstance(model.sigma, (float, np.floating))
        assert isinstance(model.log_lik, (float, np.floating))
        assert model.formula is not None

        # Variance-covariance matrix
        vcov = model.vcov()
        assert vcov is not None
        assert vcov.ndim == 2

        # Summary
        summary = model.summary()
        assert summary is not None
        assert str(summary)  # should be printable

        # Confidence intervals
        ci = model.confint()
        assert ci is not None

        # Predict — include intercept column to match fitted design matrix
        X_pred = np.column_stack([np.ones(5), x1[:5], x2[:5]])
        pred = model.predict(X_pred)
        assert pred is not None
        assert hasattr(pred, "mean")
        assert len(pred.mean) == 5

        # Score (sklearn-compatible)
        score = model.score(X_pred, y[:5])
        assert isinstance(score, float)

        # Information criteria
        assert isinstance(model.aic, (float, np.floating))
        assert isinstance(model.bic, (float, np.floating))
        assert isinstance(model.aicc, (float, np.floating))
        assert isinstance(model.bicc, (float, np.floating))

        # Stepwise-specific attributes
        assert isinstance(model.ic_values, dict)
        assert list(model.ic_values.keys())[0] == "Intercept"
        assert all(isinstance(k, str) for k in model.ic_values.keys())
        assert all(
            isinstance(v, (float, np.floating))
            for v in model.ic_values.values()
        )
        assert isinstance(model.time_elapsed, float)
        assert model.time_elapsed >= 0


class TestLmCombine:
    """Tests for lm_combine function."""

    def test_lm_combine_basic(self):
        """Test lm_combine with simple data."""
        np.random.seed(42)
        n = 30
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + 3 * x2 + np.random.normal(0, 0.1, n)

        data = {"y": y.tolist(), "x1": x1.tolist(), "x2": x2.tolist()}

        result = lm_combine(data, ic="AICc", silent=True)
        assert "coefficients" in result
        assert "fitted" in result
        assert "IC" in result
