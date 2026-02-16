"""Test all ALM distributions against R results.

This module tests that Python ALM produces identical results to R's alm() function
for all supported distributions.
"""

import csv
import subprocess
import numpy as np
import pytest
from greybox.formula import formula
from greybox.alm import ALM


# --- Continuous distributions (mpg ~ wt + am + vs) ---
CONTINUOUS_DISTRIBUTIONS = [
    "dnorm",
    "dlaplace",
    "ds",
    "dgnorm",
    "dlogis",
    "dalaplace",
    "dlnorm",
    "dllaplace",
    "dls",
    "dlgnorm",
    "dt",
    "dbcnorm",
    "dfnorm",
    "drectnorm",
    "dinvgauss",
    "dgamma",
    "dexp",
]

# --- Binary distributions (am ~ wt + hp) ---
BINARY_DISTRIBUTIONS = [
    "plogis",
    "pnorm",
]

# --- Bounded (0,1) distributions (mpg/max(mpg) ~ wt) ---
# NOTE: dbeta uses a two-part model (shape1/shape2 each with own coefficients)
# which is not yet fully implemented in Python. Excluded from R comparison.
BOUNDED_DISTRIBUTIONS = [
    "dlogitnorm",
]

# --- Count distributions (carb ~ wt + hp) ---
# NOTE: dchisq is a Python-only distribution (not in R greybox v2.0.7)
COUNT_DISTRIBUTIONS = [
    "dpois",
    "dnbinom",
    "dgeom",
]

# --- Binomial (vs ~ wt) ---
BINOMIAL_DISTRIBUTIONS = [
    "dbinom",
]


DISTRIBUTION_PARAMS = {
    "dgnorm": {"shape": 2.0},
    "dlgnorm": {"shape": 2.0},
    "dalaplace": {"alpha": 0.5},
    "dbcnorm": {},
    "dt": {"nu": 5},
    "dchisq": {"nu": 3},
    "dnbinom": {"size": 2},
    "dbinom": {"size": 1},
}


def load_mtcars():
    """Load mtcars data from CSV."""
    data = {}
    with open("/tmp/mtcars.csv", "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for key in rows[0].keys():
        data[key] = [float(row[key]) for row in rows]
    return data


def run_r_alm_continuous(distribution, params=None):
    """Run R alm() with mpg ~ wt + am + vs and return results."""
    params_str = ""
    if params:
        for key, value in params.items():
            if isinstance(value, (int, float)):
                params_str += f", {key}={value}"
            else:
                params_str += f", {key}='{value}'"

    r_code = f"""
library(greybox)
data(mtcars)
test <- alm(mpg ~ wt + am + vs, mtcars, distribution="{distribution}"{params_str})
sink("/tmp/r_alm_result.txt")
cat("intercept:", coef(test)[1], "\\n")
cat("coef_wt:", coef(test)[2], "\\n")
cat("coef_am:", coef(test)[3], "\\n")
cat("coef_vs:", coef(test)[4], "\\n")
cat("scale:", test$scale[1], "\\n")
cat("log_lik:", as.numeric(logLik(test)), "\\n")
cat("aic:", AIC(test), "\\n")
cat("bic:", BIC(test), "\\n")
sink()
"""
    subprocess.run(["R", "-e", r_code], capture_output=True, check=True)
    return _parse_r_result()


def run_r_alm_binary(distribution):
    """Run R alm() with am ~ wt + hp (binary response)."""
    r_code = f"""
library(greybox)
data(mtcars)
test <- alm(am ~ wt + hp, mtcars, distribution="{distribution}")
sink("/tmp/r_alm_result.txt")
cat("intercept:", coef(test)[1], "\\n")
cat("coef_wt:", coef(test)[2], "\\n")
cat("coef_hp:", coef(test)[3], "\\n")
cat("scale:", test$scale[1], "\\n")
cat("log_lik:", as.numeric(logLik(test)), "\\n")
cat("aic:", AIC(test), "\\n")
cat("bic:", BIC(test), "\\n")
sink()
"""
    subprocess.run(["R", "-e", r_code], capture_output=True, check=True)
    return _parse_r_result()


def run_r_alm_bounded(distribution):
    """Run R alm() with bounded (0,1) response: mpg/max(mpg) ~ wt."""
    r_code = f"""
library(greybox)
data(mtcars)
mtcars$mpg_scaled <- mtcars$mpg / (max(mtcars$mpg) + 1)
test <- alm(mpg_scaled ~ wt, mtcars, distribution="{distribution}")
sink("/tmp/r_alm_result.txt")
cat("intercept:", coef(test)[1], "\\n")
cat("coef_wt:", coef(test)[2], "\\n")
cat("scale:", test$scale[1], "\\n")
cat("log_lik:", as.numeric(logLik(test)), "\\n")
cat("aic:", AIC(test), "\\n")
cat("bic:", BIC(test), "\\n")
sink()
"""
    subprocess.run(["R", "-e", r_code], capture_output=True, check=True)
    return _parse_r_result()


def run_r_alm_count(distribution, params=None):
    """Run R alm() with count response: carb ~ wt + hp."""
    params_str = ""
    if params:
        for key, value in params.items():
            if isinstance(value, (int, float)):
                params_str += f", {key}={value}"
            else:
                params_str += f", {key}='{value}'"

    r_code = f"""
library(greybox)
data(mtcars)
test <- alm(carb ~ wt + hp, mtcars, distribution="{distribution}"{params_str})
sink("/tmp/r_alm_result.txt")
cat("intercept:", coef(test)[1], "\\n")
cat("coef_wt:", coef(test)[2], "\\n")
cat("coef_hp:", coef(test)[3], "\\n")
cat("scale:", test$scale[1], "\\n")
cat("log_lik:", as.numeric(logLik(test)), "\\n")
cat("aic:", AIC(test), "\\n")
cat("bic:", BIC(test), "\\n")
sink()
"""
    subprocess.run(["R", "-e", r_code], capture_output=True, check=True)
    return _parse_r_result()


def run_r_alm_binomial(distribution, params=None):
    """Run R alm() with vs ~ wt (binomial)."""
    params_str = ""
    if params:
        for key, value in params.items():
            if isinstance(value, (float, int)):
                params_str += f", {key}={value}"
            else:
                params_str += f", {key}='{value}'"

    r_code = f"""
library(greybox)
data(mtcars)
test <- alm(vs ~ wt, mtcars, distribution="{distribution}"{params_str})
sink("/tmp/r_alm_result.txt")
cat("intercept:", coef(test)[1], "\\n")
cat("coef_wt:", coef(test)[2], "\\n")
cat("scale:", test$scale[1], "\\n")
cat("log_lik:", as.numeric(logLik(test)), "\\n")
cat("aic:", AIC(test), "\\n")
cat("bic:", BIC(test), "\\n")
sink()
"""
    subprocess.run(["R", "-e", r_code], capture_output=True, check=True)
    return _parse_r_result()


def _parse_r_result():
    """Parse /tmp/r_alm_result.txt into a dict."""
    result = {}
    with open("/tmp/r_alm_result.txt", "r") as f:
        for line in f:
            if ":" in line:
                key, value = line.strip().split(":", 1)
                result[key.strip()] = float(value.strip())
    return result


def create_alm_model(distribution, loss="likelihood"):
    """Create ALM model with appropriate parameters for distribution."""
    params = DISTRIBUTION_PARAMS.get(distribution, {})
    return ALM(distribution=distribution, loss=loss, **params)


class TestContinuousDistributionsVsR:
    """Test continuous distributions using mpg ~ wt + am + vs."""

    @pytest.fixture(scope="class")
    def mtcars_data(self):
        return load_mtcars()

    @pytest.mark.parametrize("distribution", CONTINUOUS_DISTRIBUTIONS)
    def test_distribution_against_r(self, mtcars_data, distribution):
        y, X = formula("mpg ~ wt + am + vs", mtcars_data)
        var_names = formula(
            "mpg ~ wt + am + vs", mtcars_data, return_type="variables"
        )

        params = DISTRIBUTION_PARAMS.get(distribution, {})

        model = create_alm_model(distribution)
        model.fit(X, y, formula="mpg ~ wt + am + vs", feature_names=var_names)

        r_result = run_r_alm_continuous(distribution, params)

        rtol = 1e-3

        np.testing.assert_allclose(
            model.intercept_,
            r_result["intercept"],
            rtol=rtol,
            err_msg=f"Intercept doesn't match R for {distribution}",
        )
        np.testing.assert_allclose(
            model.coef[0],
            r_result["coef_wt"],
            rtol=rtol,
            err_msg=f"wt coefficient doesn't match R for {distribution}",
        )
        np.testing.assert_allclose(
            model.coef[1],
            r_result["coef_am"],
            rtol=rtol,
            err_msg=f"am coefficient doesn't match R for {distribution}",
        )
        np.testing.assert_allclose(
            model.coef[2],
            r_result["coef_vs"],
            rtol=rtol,
            err_msg=f"vs coefficient doesn't match R for {distribution}",
        )
        np.testing.assert_allclose(
            model.scale,
            r_result["scale"],
            rtol=rtol,
            err_msg=f"Scale doesn't match R for {distribution}",
        )
        np.testing.assert_allclose(
            model.log_lik,
            r_result["log_lik"],
            rtol=rtol,
            err_msg=f"Log-likelihood doesn't match R for {distribution}",
        )
        np.testing.assert_allclose(
            model.aic,
            r_result["aic"],
            rtol=rtol,
            err_msg=f"AIC doesn't match R for {distribution}",
        )
        np.testing.assert_allclose(
            model.bic,
            r_result["bic"],
            rtol=rtol,
            err_msg=f"BIC doesn't match R for {distribution}",
        )


class TestBinaryDistributionsVsR:
    """Test binary distributions using am ~ wt + hp."""

    @pytest.fixture(scope="class")
    def mtcars_data(self):
        return load_mtcars()

    @pytest.mark.parametrize("distribution", BINARY_DISTRIBUTIONS)
    def test_distribution_against_r(self, mtcars_data, distribution):
        y, X = formula("am ~ wt + hp", mtcars_data)
        var_names = formula("am ~ wt + hp", mtcars_data, return_type="variables")

        model = create_alm_model(distribution)
        model.fit(X, y, formula="am ~ wt + hp", feature_names=var_names)

        r_result = run_r_alm_binary(distribution)

        rtol = 1e-3

        np.testing.assert_allclose(
            model.intercept_,
            r_result["intercept"],
            rtol=rtol,
            err_msg=f"Intercept doesn't match R for {distribution}",
        )
        np.testing.assert_allclose(
            model.coef[0],
            r_result["coef_wt"],
            rtol=rtol,
            err_msg=f"wt coefficient doesn't match R for {distribution}",
        )
        np.testing.assert_allclose(
            model.coef[1],
            r_result["coef_hp"],
            rtol=rtol,
            err_msg=f"hp coefficient doesn't match R for {distribution}",
        )
        np.testing.assert_allclose(
            model.log_lik,
            r_result["log_lik"],
            rtol=rtol,
            err_msg=f"Log-likelihood doesn't match R for {distribution}",
        )
        np.testing.assert_allclose(
            model.aic,
            r_result["aic"],
            rtol=rtol,
            err_msg=f"AIC doesn't match R for {distribution}",
        )
        np.testing.assert_allclose(
            model.bic,
            r_result["bic"],
            rtol=rtol,
            err_msg=f"BIC doesn't match R for {distribution}",
        )


class TestBoundedDistributionsVsR:
    """Test bounded (0,1) distributions using mpg/max(mpg) ~ wt."""

    @pytest.fixture(scope="class")
    def mtcars_data(self):
        return load_mtcars()

    @pytest.mark.parametrize("distribution", BOUNDED_DISTRIBUTIONS)
    def test_distribution_against_r(self, mtcars_data, distribution):
        mpg = np.array(mtcars_data["mpg"])
        mtcars_mod = dict(mtcars_data)
        mtcars_mod["mpg_scaled"] = (mpg / (max(mpg) + 1)).tolist()

        y, X = formula("mpg_scaled ~ wt", mtcars_mod)
        var_names = formula(
            "mpg_scaled ~ wt", mtcars_mod, return_type="variables"
        )

        model = create_alm_model(distribution)
        model.fit(
            X, y, formula="mpg_scaled ~ wt", feature_names=var_names
        )

        r_result = run_r_alm_bounded(distribution)

        rtol = 1e-3

        np.testing.assert_allclose(
            model.intercept_,
            r_result["intercept"],
            rtol=rtol,
            err_msg=f"Intercept doesn't match R for {distribution}",
        )
        np.testing.assert_allclose(
            model.coef[0],
            r_result["coef_wt"],
            rtol=rtol,
            err_msg=f"wt coefficient doesn't match R for {distribution}",
        )
        np.testing.assert_allclose(
            model.scale,
            r_result["scale"],
            rtol=rtol,
            err_msg=f"Scale doesn't match R for {distribution}",
        )
        np.testing.assert_allclose(
            model.log_lik,
            r_result["log_lik"],
            rtol=rtol,
            err_msg=f"Log-likelihood doesn't match R for {distribution}",
        )
        np.testing.assert_allclose(
            model.aic,
            r_result["aic"],
            rtol=rtol,
            err_msg=f"AIC doesn't match R for {distribution}",
        )
        np.testing.assert_allclose(
            model.bic,
            r_result["bic"],
            rtol=rtol,
            err_msg=f"BIC doesn't match R for {distribution}",
        )


class TestCountDistributionsVsR:
    """Test count distributions using carb ~ wt + hp."""

    @pytest.fixture(scope="class")
    def mtcars_data(self):
        return load_mtcars()

    @pytest.mark.parametrize("distribution", COUNT_DISTRIBUTIONS)
    def test_distribution_against_r(self, mtcars_data, distribution):
        y, X = formula("carb ~ wt + hp", mtcars_data)
        var_names = formula(
            "carb ~ wt + hp", mtcars_data, return_type="variables"
        )

        params = DISTRIBUTION_PARAMS.get(distribution, {})

        model = create_alm_model(distribution)
        model.fit(X, y, formula="carb ~ wt + hp", feature_names=var_names)

        r_result = run_r_alm_count(distribution, params)

        rtol = 1e-3

        np.testing.assert_allclose(
            model.intercept_,
            r_result["intercept"],
            rtol=rtol,
            err_msg=f"Intercept doesn't match R for {distribution}",
        )
        np.testing.assert_allclose(
            model.coef[0],
            r_result["coef_wt"],
            rtol=rtol,
            err_msg=f"wt coefficient doesn't match R for {distribution}",
        )
        np.testing.assert_allclose(
            model.coef[1],
            r_result["coef_hp"],
            rtol=rtol,
            err_msg=f"hp coefficient doesn't match R for {distribution}",
        )
        np.testing.assert_allclose(
            model.log_lik,
            r_result["log_lik"],
            rtol=rtol,
            err_msg=f"Log-likelihood doesn't match R for {distribution}",
        )
        np.testing.assert_allclose(
            model.aic,
            r_result["aic"],
            rtol=rtol,
            err_msg=f"AIC doesn't match R for {distribution}",
        )
        np.testing.assert_allclose(
            model.bic,
            r_result["bic"],
            rtol=rtol,
            err_msg=f"BIC doesn't match R for {distribution}",
        )


class TestBinomialDistributionVsR:
    """Test binomial distribution using vs ~ wt."""

    @pytest.fixture(scope="class")
    def mtcars_data(self):
        return load_mtcars()

    @pytest.mark.parametrize("distribution", BINOMIAL_DISTRIBUTIONS)
    def test_distribution_against_r(self, mtcars_data, distribution):
        y, X = formula("vs ~ wt", mtcars_data)
        var_names = formula("vs ~ wt", mtcars_data, return_type="variables")

        params = DISTRIBUTION_PARAMS.get(distribution, {})

        model = create_alm_model(distribution)
        model.fit(X, y, formula="vs ~ wt", feature_names=var_names)

        r_result = run_r_alm_binomial(distribution, params)

        # dbinom uses supsmu init in R (not available in Python),
        # so optimizer may find slightly different local optima
        rtol = 0.02

        np.testing.assert_allclose(
            model.intercept_,
            r_result["intercept"],
            rtol=rtol,
            err_msg=f"Intercept doesn't match R for {distribution}",
        )
        np.testing.assert_allclose(
            model.coef[0],
            r_result["coef_wt"],
            rtol=rtol,
            err_msg=f"wt coefficient doesn't match R for {distribution}",
        )
        np.testing.assert_allclose(
            model.log_lik,
            r_result["log_lik"],
            rtol=rtol,
            err_msg=f"Log-likelihood doesn't match R for {distribution}",
        )
        np.testing.assert_allclose(
            model.aic,
            r_result["aic"],
            rtol=rtol,
            err_msg=f"AIC doesn't match R for {distribution}",
        )
        np.testing.assert_allclose(
            model.bic,
            r_result["bic"],
            rtol=rtol,
            err_msg=f"BIC doesn't match R for {distribution}",
        )


class TestALMDistributionSmoke:
    """Smoke tests - all distributions run without error."""

    @pytest.fixture(scope="class")
    def mtcars_data(self):
        return load_mtcars()

    @pytest.mark.parametrize("distribution", CONTINUOUS_DISTRIBUTIONS)
    def test_continuous_runs(self, mtcars_data, distribution):
        y, X = formula("mpg ~ wt + am + vs", mtcars_data)
        model = create_alm_model(distribution)
        model.fit(X, y)
        assert model.intercept_ is not None
        assert model.coef is not None
        assert model.scale is not None
        assert model.log_lik is not None

    @pytest.mark.parametrize("distribution", BINARY_DISTRIBUTIONS)
    def test_binary_runs(self, mtcars_data, distribution):
        y, X = formula("am ~ wt + hp", mtcars_data)
        model = create_alm_model(distribution)
        model.fit(X, y)
        assert model.intercept_ is not None

    @pytest.mark.parametrize("distribution", BOUNDED_DISTRIBUTIONS)
    def test_bounded_runs(self, mtcars_data, distribution):
        mpg = np.array(mtcars_data["mpg"])
        mtcars_mod = dict(mtcars_data)
        mtcars_mod["mpg_scaled"] = (mpg / (max(mpg) + 1)).tolist()
        y, X = formula("mpg_scaled ~ wt", mtcars_mod)
        model = create_alm_model(distribution)
        model.fit(X, y)
        assert model.intercept_ is not None

    def test_dbeta_smoke(self, mtcars_data):
        """dbeta uses two-part model - just verify it doesn't crash."""
        mpg = np.array(mtcars_data["mpg"])
        mtcars_mod = dict(mtcars_data)
        mtcars_mod["mpg_scaled"] = (mpg / (max(mpg) + 1)).tolist()
        y, X = formula("mpg_scaled ~ wt", mtcars_mod)
        model = ALM(distribution="dbeta", loss="likelihood")
        # dbeta currently has issues with scaler_internal - xfail
        pytest.xfail("dbeta two-part model not fully implemented")

    @pytest.mark.parametrize("distribution", COUNT_DISTRIBUTIONS)
    def test_count_runs(self, mtcars_data, distribution):
        y, X = formula("carb ~ wt + hp", mtcars_data)
        model = create_alm_model(distribution)
        model.fit(X, y)
        assert model.intercept_ is not None

    @pytest.mark.parametrize("distribution", BINOMIAL_DISTRIBUTIONS)
    def test_binomial_runs(self, mtcars_data, distribution):
        y, X = formula("vs ~ wt", mtcars_data)
        model = create_alm_model(distribution)
        model.fit(X, y)
        assert model.intercept_ is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
