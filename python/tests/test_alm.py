"""Tests for formula parser and ALM model."""

import numpy as np
from greybox.formula import formula, expand_formula
from greybox.alm import ALM


class TestFormula:
    """Tests for formula parser."""

    def test_formula_with_response(self):
        """Test formula with response variable."""
        data = {'y': [1, 2, 3], 'x1': [4, 5, 6], 'x2': [7, 8, 9]}
        y, X = formula("y ~ x1 + x2", data)
        
        assert np.array_equal(y, np.array([1, 2, 3]))
        assert X.shape == (3, 3)
        assert np.all(X[:, 0] == 1)
        assert np.array_equal(X[:, 1], np.array([4, 5, 6]))
        assert np.array_equal(X[:, 2], np.array([7, 8, 9]))

    def test_formula_without_response(self):
        """Test formula without response variable."""
        data = {'y': [1, 2, 3], 'x1': [4, 5, 6], 'x2': [7, 8, 9]}
        X = formula("~ x1 + x2", data, return_type="X")
        
        assert X.shape == (3, 3)
        assert np.all(X[:, 0] == 1)
        assert np.array_equal(X[:, 1], np.array([4, 5, 6]))
        assert np.array_equal(X[:, 2], np.array([7, 8, 9]))

    def test_formula_intercept_only(self):
        """Test formula with intercept only."""
        data = {'y': [1, 2, 3], 'x1': [4, 5, 6]}
        y, X = formula("y ~ 1", data)
        
        assert np.array_equal(y, np.array([1, 2, 3]))
        assert X.shape == (3, 1)
        assert np.all(X == 1)

    def test_formula_terms(self):
        """Test formula terms extraction."""
        data = {'y': [1, 2, 3], 'x1': [4, 5, 6]}
        terms = formula("y ~ x1", data, return_type="terms")
        
        assert terms == ['x1']

    def test_formula_trend_not_in_data(self):
        """Test formula with trend not in data - should auto-generate."""
        data = {'y': [1, 2, 3, 4, 5], 'x': [1, 2, 3, 4, 5]}
        y, X = formula("y ~ x + trend", data)
        
        assert y is not None
        assert X.shape == (5, 3)
        np.testing.assert_array_equal(X[:, 2], np.array([1, 2, 3, 4, 5]))

    def test_formula_trend_in_data(self):
        """Test formula with trend in data - should use provided values."""
        data = {'y': [1, 2, 3, 4, 5], 'x': [1, 2, 3, 4, 5], 'trend': [10, 20, 30, 40, 50]}
        y, X = formula("y ~ x + trend", data)
        
        assert X.shape == (5, 3)
        np.testing.assert_array_equal(X[:, 2], np.array([10, 20, 30, 40, 50]))

    def test_formula_log_y(self):
        """Test formula with log(y) on LHS."""
        data = {'y': [1, 2, 3, 4, 5], 'x': [1, 2, 3, 4, 5]}
        y, X = formula("log(y) ~ x", data)
        
        expected_y = np.log([1, 2, 3, 4, 5])
        np.testing.assert_array_almost_equal(y, expected_y)

    def test_formula_polynomial(self):
        """Test formula with polynomial terms."""
        data = {'y': [1, 2, 3, 4, 5], 'x': [1, 2, 3, 4, 5]}
        y, X = formula("y ~ x + x^2 + x^3", data)
        
        assert X.shape == (5, 4)
        np.testing.assert_array_equal(X[:, 2], np.array([1, 4, 9, 16, 25]))  # x^2
        np.testing.assert_array_equal(X[:, 3], np.array([1, 8, 27, 64, 125]))  # x^3

    def test_formula_log_x(self):
        """Test formula with log(x) transformation."""
        data = {'y': [1, 2, 3, 4, 5], 'x': [1, 2, 3, 4, 5]}
        y, X = formula("y ~ log(x)", data)
        
        expected_log_x = np.log([1, 2, 3, 4, 5])
        np.testing.assert_array_almost_equal(X[:, 1], expected_log_x)

    def test_formula_sqrt(self):
        """Test formula with sqrt transformation."""
        data = {'y': [1, 2, 3, 4, 5], 'x': [1, 2, 3, 4, 5]}
        y, X = formula("y ~ sqrt(x)", data)
        
        expected_sqrt_x = np.sqrt([1, 2, 3, 4, 5])
        np.testing.assert_array_almost_equal(X[:, 1], expected_sqrt_x)

    def test_formula_I_protected(self):
        """Test formula with I() protected expression."""
        data = {'y': [1, 2, 3, 4, 5], 'x': [1, 2, 3, 4, 5]}
        y, X = formula("y ~ I(x^2)", data)
        
        assert X.shape == (5, 2)
        np.testing.assert_array_equal(X[:, 1], np.array([1, 4, 9, 16, 25]))

    def test_expand_formula(self):
        """Test formula expansion."""
        expanded = expand_formula("y ~ x1 * x2")
        assert "x1:x2" in expanded


class TestALM:
    """Tests for ALM model."""

    def test_alm_fit_predict(self):
        """Test ALM model fit and predict."""
        data = {'y': [1.0, 2.0, 3.0, 4.0, 5.0], 
                'x1': [1.0, 2.0, 3.0, 4.0, 5.0], 
                'x2': [2.0, 3.0, 4.0, 5.0, 6.0]}
        y, X = formula("y ~ x1 + x2", data)
        
        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y)
        
        assert model.intercept_ is not None
        assert model.coef_ is not None
        assert model.scale_ is not None
        assert model.log_lik_ is not None
        assert model.aic_ is not None
        
        y_pred = model.predict(X)
        assert y_pred.shape == y.shape

    def test_alm_get_params(self):
        """Test ALM get_params."""
        model = ALM(distribution="dgamma", loss="MSE")
        params = model.get_params()
        
        assert params["distribution"] == "dgamma"
        assert params["loss"] == "MSE"

    def test_alm_set_params(self):
        """Test ALM set_params."""
        model = ALM()
        model.set_params(distribution="dlogis", loss="MAE")
        
        assert model.distribution == "dlogis"
        assert model.loss == "MAE"

    def test_alm_score(self):
        """Test ALM score method."""
        data = {'y': [1.0, 2.0, 3.0, 4.0, 5.0], 
                'x1': [1.0, 2.0, 3.0, 4.0, 5.0], 
                'x2': [2.0, 3.0, 4.0, 5.0, 6.0]}
        y, X = formula("y ~ x1 + x2", data)
        
        model = ALM(distribution="dnorm", loss="likelihood")
        model.fit(X, y)
        
        score_mse = model.score(X, y, metric="MSE")
        score_r2 = model.score(X, y, metric="R2")
        
        assert score_mse >= 0
        assert score_r2 <= 1.1

    def test_alm_repr(self):
        """Test ALM repr."""
        model = ALM(distribution="dnorm", loss="likelihood")
        repr_str = repr(model)
        
        assert "ALM" in repr_str
        assert "dnorm" in repr_str

    def test_alm_nlopt_kargs(self):
        """Test ALM with custom nlopt_kargs."""
        import csv
        data = {}
        with open('/tmp/mtcars.csv', 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        for key in rows[0].keys():
            data[key] = [float(row[key]) for row in rows]
        
        y, X = formula("mpg ~ wt", data)

        model = ALM(
            distribution="dnorm",
            loss="likelihood",
            nlopt_kargs={
                "algorithm": "NLOPT_LN_SBPLX",
                "maxeval": 100,
                "xtol_rel": 1e-4
            }
        )
        model.fit(X, y)

        assert model.intercept_ is not None
        assert model.coef_ is not None
        assert model.log_lik_ is not None
        assert model.nlopt_result_ is not None


class TestALMMtcars:
    """Tests comparing Python ALM with R alm() using mtcars dataset.

    Note: For normal distribution (dnorm), likelihood and MSE give the same
    results because they are equivalent for Gaussian errors.
    """

    @staticmethod
    def load_mtcars():
        """Load mtcars data from CSV."""
        import csv
        data = {}
        with open('/tmp/mtcars.csv', 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        for key in rows[0].keys():
            data[key] = [float(row[key]) for row in rows]
        return data

    def test_alm_mtcars_dnorm(self):
        """Test ALM with mtcars: mpg ~ wt, dnorm, likelihood.

        R results:
        - Intercept: 37.29
        - wt: -5.34
        - scale: 2.95
        - log-likelihood: -80.01
        """
        data = self.load_mtcars()
        y, X = formula("mpg ~ wt", data)

        model = ALM(distribution="dnorm", loss="likelihood",
                   nlopt_kargs={"maxeval": 1000})
        model.fit(X, y)

        np.testing.assert_allclose(
            model.intercept_, 37.29, rtol=1e-1,
            err_msg="Intercept doesn't match R"
        )
        np.testing.assert_allclose(
            model.coef_[0], -5.34, rtol=1e-1,
            err_msg="wt coefficient doesn't match R"
        )
        np.testing.assert_allclose(
            model.scale_, 2.95, rtol=2e-1,
            err_msg="Scale doesn't match R"
        )
        np.testing.assert_allclose(
            model.log_lik_, -80.01, rtol=1e-1,
            err_msg="Log-likelihood doesn't match R"
        )

    def test_alm_mtcars_dnorm_two_predictors(self):
        """Test ALM with mtcars: mpg ~ wt + hp, dnorm, likelihood.

        R results:
        - Intercept: 37.23
        - wt: -3.88
        - hp: -0.032
        - scale: 2.47
        - log-likelihood: -74.33
        """
        data = self.load_mtcars()
        y, X = formula("mpg ~ wt + hp", data)

        model = ALM(distribution="dnorm", loss="likelihood", nlopt_kargs={"maxeval": 1000})
        model.fit(X, y)

        np.testing.assert_allclose(
            model.intercept_, 37.23, rtol=1e-1,
            err_msg="Intercept doesn't match R"
        )
        np.testing.assert_allclose(
            model.coef_[0], -3.88, rtol=1e-1,
            err_msg="wt coefficient doesn't match R"
        )
        np.testing.assert_allclose(
            model.coef_[1], -0.032, rtol=2e-1,
            err_msg="hp coefficient doesn't match R"
        )
        np.testing.assert_allclose(
            model.scale_, 2.47, rtol=2e-1,
            err_msg="Scale doesn't match R"
        )

    def test_alm_mtcars_dlaplace(self):
        """Test ALM with mtcars: mpg ~ wt, dlaplace, likelihood.

        R results:
        - Intercept: 34.33
        - wt: -4.56
        - scale: 2.32
        """
        data = self.load_mtcars()
        y, X = formula("mpg ~ wt", data)

        model = ALM(distribution="dlaplace", loss="likelihood", nlopt_kargs={"maxeval": 1000})
        model.fit(X, y)

        np.testing.assert_allclose(
            model.intercept_, 34.33, rtol=1e-1,
            err_msg="Intercept doesn't match R"
        )
        np.testing.assert_allclose(
            model.coef_[0], -4.56, rtol=1e-1,
            err_msg="wt coefficient doesn't match R"
        )
        np.testing.assert_allclose(
            model.scale_, 2.32, rtol=2e-1,
            err_msg="Scale doesn't match R"
        )

    def test_alm_mtcars_dnorm_mse(self):
        """Test ALM with mtcars: mpg ~ wt, dnorm, MSE loss.

        For dnorm, likelihood and MSE are equivalent (Gaussian).
        R results:
        - Intercept: 37.29
        - wt: -5.34
        - scale: 2.95
        """
        data = self.load_mtcars()
        y, X = formula("mpg ~ wt", data)

        model = ALM(distribution="dnorm", loss="MSE", nlopt_kargs={"maxeval": 1000})
        model.fit(X, y)

        np.testing.assert_allclose(
            model.intercept_, 37.29, rtol=1e-1,
            err_msg="Intercept doesn't match R"
        )
        np.testing.assert_allclose(
            model.coef_[0], -5.34, rtol=1e-1,
            err_msg="wt coefficient doesn't match R"
        )

    def test_alm_mtcars_dnorm_mae(self):
        """Test ALM with mtcars: mpg ~ wt, dnorm, MAE loss.

        R results (approximate):
        - Intercept: ~35-36
        - wt: ~-4.8 to -5.0
        """
        data = self.load_mtcars()
        y, X = formula("mpg ~ wt", data)

        model = ALM(distribution="dnorm", loss="MAE", nlopt_kargs={"maxeval": 1000})
        model.fit(X, y)

        np.testing.assert_allclose(
            model.intercept_, 35.6, rtol=2e-1,
            err_msg="Intercept doesn't match R"
        )
        np.testing.assert_allclose(
            model.coef_[0], -4.9, rtol=2e-1,
            err_msg="wt coefficient doesn't match R"
        )

    def test_alm_mtcars_dlogis(self):
        """Test ALM with mtcars: mpg ~ wt, dlogis, likelihood.

        R results:
        - Intercept: 36.73
        - wt: -5.26
        - scale: 1.63
        """
        data = self.load_mtcars()
        y, X = formula("mpg ~ wt", data)

        model = ALM(distribution="dlogis", loss="likelihood", nlopt_kargs={"maxeval": 1000})
        model.fit(X, y)

        np.testing.assert_allclose(
            model.intercept_, 36.73, rtol=1e-1,
            err_msg="Intercept doesn't match R"
        )
        np.testing.assert_allclose(
            model.coef_[0], -5.26, rtol=1e-1,
            err_msg="wt coefficient doesn't match R"
        )
        np.testing.assert_allclose(
            model.scale_, 1.63, rtol=2e-1,
            err_msg="Scale doesn't match R"
        )

    def test_alm_mtcars_dt(self):
        """Test ALM with mtcars: mpg ~ wt, dt, likelihood.

        R results:
        - Intercept: ~37
        - wt: ~-5.3
        - scale: ~2.9
        """
        data = self.load_mtcars()
        y, X = formula("mpg ~ wt", data)

        model = ALM(distribution="dt", loss="likelihood", nu=4, nlopt_kargs={"maxeval": 1000})
        model.fit(X, y)

        np.testing.assert_allclose(
            model.intercept_, 37.0, rtol=1e-1,
            err_msg="Intercept doesn't match R"
        )
        np.testing.assert_allclose(
            model.coef_[0], -5.3, rtol=1e-1,
            err_msg="wt coefficient doesn't match R"
        )

    def test_alm_mtcars_dgnorm(self):
        """Test ALM with mtcars: mpg ~ wt, dgnorm, likelihood.

        R results (shape=2):
        - Intercept: ~37
        - wt: ~-5.3
        - scale: ~2.9
        """
        data = self.load_mtcars()
        y, X = formula("mpg ~ wt", data)

        model = ALM(distribution="dgnorm", loss="likelihood", shape=2.0, nlopt_kargs={"maxeval": 1000})
        model.fit(X, y)

        np.testing.assert_allclose(
            model.intercept_, 37.0, rtol=1e-1,
            err_msg="Intercept doesn't match R"
        )
        np.testing.assert_allclose(
            model.coef_[0], -5.3, rtol=1e-1,
            err_msg="wt coefficient doesn't match R"
        )
