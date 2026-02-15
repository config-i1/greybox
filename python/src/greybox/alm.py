"""Augmented Linear Model (ALM) for greybox.

This module provides the ALM estimator following scikit-learn principles.
Users should first use the formula module to get X and y, then pass them
to the fit() method.
"""

from typing import Literal

import time as time_module

import numpy as np
import nlopt
from scipy import stats

from .fitters import scaler_internal, extractor_fitted, extractor_residuals
from .cost_function import cf
from . import distributions as dist
from .methods.summary import SummaryResult


class PredictionResult:
    """Prediction result object with mean and interval bounds.

    Attributes
    ----------
    mean : np.ndarray
        Predicted values (point forecasts).
    lower : np.ndarray or None
        Lower prediction bounds.
    upper : np.ndarray or None
        Upper prediction bounds.
    """

    def __init__(self):
        self.mean = None
        self.lower = None
        self.upper = None


NLOPT_ALGORITHMS = {
    "NLOPT_LN_NELDERMEAD": nlopt.LN_NELDERMEAD,
    "NLOPT_LN_SBPLX": nlopt.LN_SBPLX,
    "NLOPT_LN_COBYLA": nlopt.LN_COBYLA,
    "NLOPT_LN_BOBYQA": nlopt.LN_BOBYQA,
    "NLOPT_LN_NEWUOA": nlopt.LN_NEWUOA,
    "NLOPT_LN_PRAXIS": nlopt.LN_PRAXIS,
    "NLOPT_LD_LBFGS": nlopt.LD_LBFGS,
    "NLOPT_LD_SLSQP": nlopt.LD_SLSQP,
    "NLOPT_LD_MMA": nlopt.LD_MMA,
}


class ALM:
    """Augmented Linear Model estimator.

    This estimator fits a linear model with various distributions and loss
    functions, following scikit-learn principles.

    Parameters
    ----------
    distribution : str, default="dnorm"
        Distribution name. Options: "dnorm", "dlaplace", "ds", "dgnorm",
        "dlogis", "dt", "dalaplace", "dlnorm", "dllaplace", "dls",
        "dlgnorm", "dbcnorm", "dfnorm", "drectnorm", "dinvgauss",
        "dgamma", "dexp", "dchisq", "dgeom", "dpois", "dnbinom",
        "dbinom", "dlogitnorm", "dbeta", "plogis", "pnorm".
    loss : str, default="likelihood"
        Loss function. Options: "likelihood", "MSE", "MAE", "HAM",
        "LASSO", "RIDGE", "ROLE".
    occurrence : str, default="none"
        Occurrence model for zero-inflated data. Options: "none",
        "plogis", "pnorm".
    scale_formula : array-like or None, default=None
        Formula for scale parameter. If None, scale is constant.
    orders : tuple, default=(0, 0, 0)
        ARIMA orders (p, d, q).
    alpha : float, optional
        Additional parameter for Asymmetric Laplace distribution.
    shape : float, optional
        Shape parameter for Generalized Normal distribution.
    lambda_bc : float, optional
        Box-Cox lambda parameter for Box-Cox Normal distribution.
    size : float, optional
        Size parameter for Negative Binomial/Binomial distributions.
    nu : float, optional
        Degrees of freedom for Student's t or Chi-squared distributions.
    trim : float, default=0.0
        Trim proportion for ROLE loss.
    lambda_l1 : float, optional
        L1 regularization parameter for LASSO.
    lambda_l2 : float, optional
        L2 regularization parameter for RIDGE.
    nlopt_kargs : dict, optional
        Dictionary of nlopt parameters. Options:
        - "algorithm": str, default="NLOPT_LN_NELDERMEAD"
        - "maxeval": int, default=40 per parameter
        - "maxtime": float, default=600 seconds
        - "xtol_rel": float, default=1e-6
        - "xtol_abs": float, default=1e-8
        - "ftol_rel": float, default=1e-4
        - "ftol_abs": float, default=0
        - "print_level": int, default=0 (0=none, 3=full)
    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients (excluding intercept).
    intercept_ : float
        Estimated intercept.
    scale_ : float
        Estimated scale parameter.
    other_ : dict
        Other estimated parameters (alpha, shape, etc.).
    fitted_values_ : ndarray of shape (n_samples,)
        Fitted values.
    residuals_ : ndarray of shape (n_samples,)
        Model residuals.
    loss_value_ : float
        Final value of the loss function.
    log_lik_ : float or None
        Log-likelihood (only for likelihood-based losses).
    aic_ : float or None
        Akaike Information Criterion.
    bic_ : float or None
        Bayesian Information Criterion.
    n_iter_ : int
        Number of optimization iterations.

    Examples
    --------
    >>> from greybox.formula import formula
    >>> from greybox.alm import ALM
    >>> data = {'y': [1, 2, 3, 4, 5], 'x1': [1, 2, 3, 4, 5], 'x2': [2, 3, 4, 5, 6]}
    >>> y, X = formula("y ~ x1 + x2", data)
    >>> model = ALM(distribution="dnorm", loss="likelihood")
    >>> model.fit(X, y)
    >>> print(model.coef_)

    >>> # Using nlopt with custom parameters (like R)
    >>> model = ALM(
    ...     distribution="dnorm",
    ...     loss="likelihood",
    ...     nlopt_kargs={
    ...         "algorithm": "NLOPT_LN_SBPLX",
    ...         "maxeval": 1000,
    ...         "maxtime": 600,
    ...         "xtol_rel": 1e-8,
    ...         "print_level": 1
    ...     }
    ... )
    >>> model.fit(X, y)
    """

    DISTRIBUTIONS = [
        "dnorm",
        "dlaplace",
        "ds",
        "dgnorm",
        "dlogis",
        "dt",
        "dalaplace",
        "dlnorm",
        "dllaplace",
        "dls",
        "dlgnorm",
        "dbcnorm",
        "dinvgauss",
        "dgamma",
        "dexp",
        "dchisq",
        "dfnorm",
        "drectnorm",
        "dpois",
        "dnbinom",
        "dbinom",
        "dgeom",
        "dbeta",
        "dlogitnorm",
        "plogis",
        "pnorm",
    ]

    LOSS_FUNCTIONS = ["likelihood", "MSE", "MAE", "HAM", "LASSO", "RIDGE", "ROLE"]

    def __init__(
        self,
        distribution="dnorm",
        loss="likelihood",
        occurrence="none",
        scale_formula=None,
        orders=(0, 0, 0),
        alpha=None,
        shape=None,
        lambda_bc=None,
        size=None,
        nu=None,
        trim=0.0,
        lambda_l1=None,
        lambda_l2=None,
        nlopt_kargs=None,
        verbose=0,
    ):
        self.distribution = distribution
        self.loss = loss
        self.occurrence = occurrence
        self.scale_formula = scale_formula
        self.orders = orders
        self.alpha = alpha
        self.shape = shape
        self.lambda_bc = lambda_bc
        self.size = size
        self.nu = nu
        self.trim = trim
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.nlopt_kargs = nlopt_kargs if nlopt_kargs is not None else {}
        self.verbose = verbose

        self.coef_ = None
        self.intercept_ = None
        self.scale_ = None
        self.other_ = None
        self.fitted_values_ = None
        self.residuals_ = None
        self.loss_value_ = None
        self.log_lik_ = None
        self.aic_ = None
        self.bic_ = None
        self.n_iter_ = None
        self.time_elapsed_ = None
        self._result = None
        self._n_features = None
        self._X_train_ = None
        self._y_train_ = None
        self._formula_ = None
        self._feature_names = None
        self.df_residual_ = None

    def _validate_params(self):
        """Validate input parameters."""
        if self.distribution not in self.DISTRIBUTIONS:
            raise ValueError(
                f"Invalid distribution: {self.distribution}. "
                f"Choose from: {self.DISTRIBUTIONS}"
            )

        if self.loss not in self.LOSS_FUNCTIONS:
            raise ValueError(
                f"Invalid loss: {self.loss}. Choose from: {self.LOSS_FUNCTIONS}"
            )

        if not isinstance(self.orders, (tuple, list)) or len(self.orders) != 3:
            raise ValueError("orders must be a tuple of 3 integers (p, d, q)")

    def fit(self, X, y, formula=None, feature_names=None):
        """Fit the ALM model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Design matrix. Should include intercept column (column of ones)
            as first column if you want an intercept.
        y : array-like of shape (n_samples,)
            Target values.
        formula : str, optional
            Formula string used to generate X and y. Stored for reference.
        feature_names : list of str, optional
            Names for feature columns. If provided, used in print output.

        Returns
        -------
        self : ALM
            Fitted estimator.
        """
        start_time = time_module.time()

        self._validate_params()
        self._formula_ = formula
        self._feature_names = feature_names

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        self._n_features = n_features

        other_val = self._get_other_parameter()
        a_parameter_provided = other_val is not None

        if not a_parameter_provided:
            other_val = 1.0

        B_init = np.zeros(n_features)

        if self.distribution in ("dlnorm", "dllaplace", "dls", "dlgnorm"):
            y_pos = y[y > 0]
            X_pos = X[y > 0]
            if len(y_pos) > 0:
                try:
                    B_init = np.linalg.lstsq(X_pos, np.log(y_pos), rcond=None)[0]
                except Exception:
                    B_init = np.zeros(n_features)
        elif self.distribution in ("dlogitnorm",):
            y_clip = np.clip(y, 1e-10, 1 - 1e-10)
            y_transformed = np.log(y_clip / (1 - y_clip))
            try:
                B_init = np.linalg.lstsq(X, y_transformed, rcond=None)[0]
            except Exception:
                B_init = np.zeros(n_features)
        else:
            try:
                B_init = np.linalg.lstsq(X, y, rcond=None)[0]
            except Exception:
                B_init = np.zeros(n_features)

        n_params = n_features

        def objective_func(B, grad):
            if grad.size > 0:
                grad[:] = 0
            return cf(
                B,
                self.distribution,
                self.loss,
                y,
                X,
                ar_order=0,
                i_order=0,
                lambda_val=self.lambda_l1
                if self.loss == "LASSO"
                else (self.lambda_l2 if self.loss == "RIDGE" else 0.0),
                other=other_val,
                a_parameter_provided=a_parameter_provided,
                trim=self.trim,
                lambda_bc=self.lambda_bc if self.distribution == "dbcnorm" else 0.0,
                size=self.size if self.distribution == "dbinom" else 1.0,
            )

        algorithm_name = self.nlopt_kargs.get("algorithm", "NLOPT_LN_NELDERMEAD")
        if algorithm_name not in NLOPT_ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm: {algorithm_name}. Choose from: {list(NLOPT_ALGORITHMS.keys())}"
            )

        algorithm = NLOPT_ALGORITHMS[algorithm_name]

        opt = nlopt.opt(algorithm, n_params)

        opt.set_lower_bounds([-1e10] * n_params)
        opt.set_upper_bounds([1e10] * n_params)
        opt.set_min_objective(objective_func)

        opt.set_xtol_rel(self.nlopt_kargs.get("xtol_rel", 1e-6))
        opt.set_xtol_abs(self.nlopt_kargs.get("xtol_abs", 1e-8))
        opt.set_ftol_rel(self.nlopt_kargs.get("ftol_rel", 1e-4))
        opt.set_ftol_abs(self.nlopt_kargs.get("ftol_abs", 0))
        opt.set_maxeval(self.nlopt_kargs.get("maxeval", 40 * n_params))
        opt.set_maxtime(self.nlopt_kargs.get("maxtime", 600))

        print_level = self.nlopt_kargs.get("print_level", 0)
        if self.verbose > 0 or print_level > 0:
            opt.set_verbose(True)

        try:
            B_opt = opt.optimize(B_init)
            self.n_iter_ = opt.get_numevals()
            self._result = opt.last_optimum_value()
            self.nlopt_result_ = opt.last_optimize_result()
        except Exception as e:
            if self.verbose > 0:
                print(f"Optimization failed: {e}")
            B_opt = B_init
            self.n_iter_ = 0
            self._result = None
            self.nlopt_result_ = None

        if n_features > 0:
            self.intercept_ = B_opt[0]
            self.coef_ = B_opt[1:] if len(B_opt) > 1 else np.array([])

        fitter_return = {
            "mu": X @ B_opt,
            "scale": 1.0,
            "other": other_val,
        }

        scale = scaler_internal(
            B_opt,
            self.distribution,
            y,
            X,
            fitter_return["mu"],
            other_val,
            np.ones(len(y), dtype=bool),
            np.sum(np.ones(len(y), dtype=bool)),
        )
        fitter_return["scale"] = scale
        self.scale_ = scale
        self.other_ = other_val

        self.fitted_values_ = extractor_fitted(
            self.distribution,
            fitter_return["mu"],
            scale,
            self.lambda_bc if self.distribution == "dbcnorm" else 0.0,
        )

        self.residuals_ = extractor_residuals(
            self.distribution,
            fitter_return["mu"],
            y,
            self.lambda_bc if self.distribution == "dbcnorm" else 0.0,
        )

        self.loss_value_ = objective_func(B_opt, np.zeros(n_params))

        if self.loss == "likelihood":
            self.log_lik_ = -self.loss_value_
            n_params = n_features
            if self.distribution not in (
                "dexp",
                "dpois",
                "dgeom",
                "dbinom",
                "plogis",
                "pnorm",
            ):
                n_params += 1
            self.aic_ = 2 * n_params - 2 * self.log_lik_
            self.bic_ = n_params * np.log(n_samples) - 2 * self.log_lik_

        self.df_residual_ = n_samples - n_params
        self._X_train_ = X.copy()
        self._y_train_ = y.copy()

        scale = scaler_internal(
            B_opt,
            self.distribution,
            y,
            X,
            fitter_return["mu"],
            other_val,
            np.ones(len(y), dtype=bool),
            self.df_residual_,
        )
        self.scale_ = scale

        XtX = X.T @ X
        XtX += np.eye(XtX.shape[0]) * 1e-10
        self._XtX_inv_ = np.linalg.inv(XtX)

        self.time_elapsed_ = time_module.time() - start_time

        return self

    def vcov(self) -> np.ndarray:
        """Calculate variance-covariance matrix of parameter estimates.

        Returns the variance-covariance matrix of the parameter estimates,
        computed as: scale² × (X'X)⁻¹

        Returns
        -------
        vcov_matrix : np.ndarray
            Covariance matrix of shape (n_params, n_params)
        """
        if self._X_train_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._X_train_
        XtX = X.T @ X
        XtX_reg = XtX + np.eye(XtX.shape[0]) * 1e-10

        try:
            XtX_inv = np.linalg.inv(XtX_reg)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Covariance matrix is singular. Results may be inaccurate. "
                "Using pseudoinverse instead."
            )
            XtX_inv = np.linalg.pinv(XtX_reg)

        return (self.scale_**2) * XtX_inv

    def _calculate_variance(self, X, interval="confidence"):
        """Calculate conditional variance for prediction intervals.

        Parameters
        ----------
        X : np.ndarray
            Design matrix for predictions.
        interval : str
            Type of interval: "confidence" or "prediction".

        Returns
        -------
        variances : np.ndarray
            Variance values for each observation.
        """
        vcov_matrix = self.vcov()
        variances = np.diag(X @ vcov_matrix @ X.T)

        if interval == "prediction":
            return variances + (self.scale_**2)
        return variances

    def _calculate_quantiles(self, mean, variances, interval, level, side):
        """Calculate lower and upper quantiles for prediction intervals.

        Parameters
        ----------
        mean : np.ndarray
            Predicted mean values.
        variances : np.ndarray
            Variance values.
        interval : str
            Type of interval: "confidence" or "prediction".
        level : float or list
            Confidence level(s).
        side : str
            Side of interval: "both", "upper", or "lower".

        Returns
        -------
        lower : np.ndarray or None
            Lower bounds.
        upper : np.ndarray or None
            Upper bounds.
        """
        if interval == "none":
            return None, None

        if not isinstance(level, list):
            level = [level]

        n_levels = len(level)
        n_obs = len(mean)
        se = np.sqrt(variances)

        if side == "upper":
            level_low = [0.0] * n_levels
            level_up = level
        elif side == "lower":
            level_low = [1 - lev for lev in level]
            level_up = [1.0] * n_levels
        else:
            level_low = [(1 - lev) / 2 for lev in level]
            level_up = [(1 + lev) / 2 for lev in level]

        quantiles_low = stats.t.ppf(level_low, df=self.df_residual_)
        quantiles_up = stats.t.ppf(level_up, df=self.df_residual_)

        lower = np.zeros((n_obs, n_levels))
        upper = np.zeros((n_obs, n_levels))

        for i in range(n_levels):
            lower[:, i] = mean + quantiles_low[i] * se
            upper[:, i] = mean + quantiles_up[i] * se

        if n_levels == 1:
            lower = lower.ravel()
            upper = upper.ravel()

        if side == "upper":
            lower = None
        elif side == "lower":
            upper = None

        return lower, upper

    def predict(
        self,
        X: np.ndarray,
        interval: Literal["none", "confidence", "prediction"] = "none",
        level: float | list[float] = 0.95,
        side: Literal["both", "upper", "lower"] = "both",
    ) -> PredictionResult:
        """Predict using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Design matrix. Should have same number of features as training data.
        interval : {"none", "confidence", "prediction"}, default="none"
            Type of interval to calculate:
            - "none": No intervals, return only point forecasts
            - "confidence": Confidence interval for the mean
            - "prediction": Prediction interval for new observations
        level : float or list of float, default=0.95
            Confidence level(s) for intervals. Can be a single float (e.g., 0.95)
            or a list of floats (e.g., [0.8, 0.9, 0.95]). Default is 0.95 (95%).
        side : {"both", "upper", "lower"}, default="both"
            Side of interval:
            - "both": Return both lower and upper bounds
            - "upper": Return only upper bounds
            - "lower": Return only lower bounds

        Returns
        -------
        PredictionResult
            Object with the following attributes:
            - mean : np.ndarray - Predicted values (point forecasts)
            - lower : np.ndarray or None - Lower prediction bounds
            - upper : np.ndarray or None - Upper prediction bounds
        """
        if self.fitted_values_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if interval not in ("none", "confidence", "prediction"):
            raise ValueError(
                f"interval must be 'none', 'confidence', or 'prediction', got {interval!r}"
            )

        if side not in ("both", "upper", "lower"):
            raise ValueError(f"side must be 'both', 'upper', or 'lower', got {side!r}")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != self._n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was fitted with {self._n_features}"
            )

        if self.coef_ is not None and len(self.coef_) > 0:
            B = np.concatenate([[self.intercept_], self.coef_])
        else:
            B = np.array([self.intercept_])

        mu = X @ B

        mean = extractor_fitted(
            self.distribution,
            mu,
            self.scale_,
            self.lambda_bc if self.distribution == "dbcnorm" else 0.0,
        )

        if interval == "none":
            lower = None
            upper = None
        else:
            variances = self._calculate_variance(X, interval)
            lower, upper = self._calculate_quantiles(
                mean, variances, interval, level, side
            )

        result = PredictionResult()
        result.mean = mean
        result.lower = lower
        result.upper = upper

        return result

    def score(self, X, y, metric="likelihood"):
        """Calculate model score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Design matrix.
        y : array-like of shape (n_samples,)
            True values.
        metric : str, optional
            Metric to use: "likelihood", "MSE", "MAE", or "R2".

        Returns
        -------
        score : float
            Score value.
        """
        y_pred_result = self.predict(X)
        y_pred = y_pred_result.mean

        if metric == "likelihood":
            if self.log_lik_ is not None:
                return self.log_lik_
            y = np.asarray(y)
            log_lik = 0.0
            y_otU = (
                y[y != 0]
                if self.distribution not in ("dlnorm", "dllaplace", "dls", "dlgnorm")
                else y
            )
            mu_otU = (
                y_pred[y != 0]
                if self.distribution not in ("dlnorm", "dllaplace", "dls", "dlgnorm")
                else y_pred
            )
            if self.distribution == "dnorm":
                log_lik = np.sum(
                    dist.dnorm(y_otU, mean=mu_otU, sd=self.scale_, log=True)
                )
            return log_lik
        elif metric == "MSE":
            return np.mean((y - y_pred) ** 2)
        elif metric == "MAE":
            return np.mean(np.abs(y - y_pred))
        elif metric == "R2":
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - ss_res / ss_tot
        else:
            raise ValueError(f"Unknown metric: {metric}")

    @property
    def nobs(self) -> int:
        """Number of observations.

        Returns
        -------
        nobs : int
            Number of observations used in the model.
        """
        if self._X_train_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._X_train_.shape[0]

    @property
    def nparam(self) -> int:
        """Number of parameters.

        Returns
        -------
        nparam : int
            Number of parameters in the model (including intercept and scale).
        """
        if self._X_train_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        n_params = self._X_train_.shape[1]
        if self.distribution not in (
            "dexp",
            "dpois",
            "dgeom",
            "dbinom",
            "plogis",
            "pnorm",
        ):
            n_params += 1
        return n_params

    @property
    def residuals(self) -> np.ndarray:
        """Model residuals.

        Returns
        -------
        residuals : np.ndarray
            Residuals (y - fitted values).
        """
        if self.residuals_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.residuals_

    @property
    def fitted(self) -> np.ndarray:
        """Fitted values.

        Returns
        -------
        fitted : np.ndarray
            Fitted values (predictions on training data).
        """
        if self.fitted_values_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.fitted_values_

    @property
    def actuals(self) -> np.ndarray:
        """Actual values (response variable).

        Returns
        -------
        actuals : np.ndarray
            Actual response values from training data.
        """
        if self._y_train_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._y_train_

    @property
    def formula(self) -> str | None:
        """Formula string used to fit the model.

        Returns
        -------
        formula : str or None
            Formula string if provided during fit.
        """
        return self._formula_

    @property
    def sigma(self) -> float:
        """Scale parameter (sigma).

        Returns
        -------
        sigma : float
            Scale parameter of the model.
        """
        if self.scale_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.scale_

    @property
    def log_lik(self) -> float:
        """Log-likelihood of the model.

        Returns
        -------
        log_lik : float
            Log-likelihood value.
        """
        if self.log_lik_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.log_lik_

    def summary(self, level: float = 0.95) -> "SummaryResult":
        """Model summary.

        Parameters
        ----------
        level : float, optional
            Confidence level for parameter intervals. Default is 0.95.

        Returns
        -------
        SummaryResult
            Summary of the model with coefficient estimates, standard errors,
            t-statistics, p-values, and confidence intervals.
        """
        from .methods.summary import SummaryResult

        if self.coef_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        coefficients = np.concatenate([[self.intercept_], self.coef_])
        vcov_matrix = self.vcov()
        se = np.sqrt(np.diag(vcov_matrix))
        t_stat = coefficients / se
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=self.df_residual_))

        t_crit = stats.t.ppf((1 + level) / 2, df=self.df_residual_)
        lower_ci = coefficients - t_crit * se
        upper_ci = coefficients + t_crit * se

        return SummaryResult(
            coefficients=coefficients,
            se=se,
            t_stat=t_stat,
            p_value=p_value,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            scale=self.scale_,
            df_residual=self.df_residual_,
            distribution=self.distribution,
            log_lik=self.log_lik_,
            aic=self.aic_,
            bic=self.bic_,
            nobs=self.nobs,
            nparam=self.nparam,
        )

    def confint(
        self,
        parm: int | list[int] | None = None,
        level: float = 0.95,
    ) -> np.ndarray:
        """Confidence intervals for parameters.

        Parameters
        ----------
        parm : int or list of int, optional
            Which parameters to include. If None, all parameters are included.
            0 = intercept, 1, 2, ... = coefficients.
        level : float, optional
            Confidence level. Default is 0.95.

        Returns
        -------
        confint : np.ndarray
            Array with shape (n_params, 2) containing lower and upper bounds.
        """
        if self.coef_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        coefficients = np.concatenate([[self.intercept_], self.coef_])
        vcov_matrix = self.vcov()
        se = np.sqrt(np.diag(vcov_matrix))

        t_crit = stats.t.ppf((1 + level) / 2, df=self.df_residual_)
        lower_ci = coefficients - t_crit * se
        upper_ci = coefficients + t_crit * se

        if parm is not None:
            if isinstance(parm, int):
                parm = [parm]
            lower_ci = lower_ci[parm]
            upper_ci = upper_ci[parm]

        return np.column_stack([lower_ci, upper_ci])

    def forecast(
        self,
        h: int = 1,
        newdata: np.ndarray | None = None,
        interval: Literal["none", "confidence", "prediction"] = "none",
        level: float = 0.95,
        side: Literal["both", "upper", "lower"] = "both",
    ) -> PredictionResult:
        """Forecast future values.

        Parameters
        ----------
        h : int
            Forecast horizon. Not used if newdata is provided.
        newdata : array-like, optional
            Future values of exogenous variables.
        interval : {"none", "confidence", "prediction"}, optional
            Type of interval to calculate. Default is "none".
        level : float, optional
            Confidence level. Default is 0.95.
        side : {"both", "upper", "lower"}, optional
            Side of interval. Default is "both".

        Returns
        -------
        PredictionResult
            Forecast results with mean and interval bounds.
        """
        if newdata is not None:
            return self.predict(newdata, interval=interval, level=level, side=side)

        raise NotImplementedError(
            "Forecasting without newdata requires time series functionality. "
            "Please provide newdata for exogenous variables."
        )

    def _get_other_parameter(self):
        """Get the additional parameter for distributions that require it."""
        if self.distribution == "dalaplace":
            return self.alpha
        elif self.distribution in ("dgnorm", "dlgnorm"):
            return self.shape
        elif self.distribution == "dbcnorm":
            return self.lambda_bc
        elif self.distribution in ("dt", "dchisq"):
            return self.nu
        elif self.distribution in ("dnbinom",):
            return self.size
        else:
            return None

    def get_params(self):
        """Get model parameters.

        Returns
        -------
        params : dict
            Dictionary of model parameters.
        """
        return {
            "distribution": self.distribution,
            "loss": self.loss,
            "occurrence": self.occurrence,
            "orders": self.orders,
            "alpha": self.alpha,
            "shape": self.shape,
            "lambda_bc": self.lambda_bc,
            "size": self.size,
            "nu": self.nu,
            "trim": self.trim,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "nlopt_kargs": self.nlopt_kargs,
            "verbose": self.verbose,
        }

    def set_params(self, **params):
        """Set model parameters.

        Parameters
        ----------
        **params : dict
            Parameters to set.

        Returns
        -------
        self : ALM
            Model with updated parameters.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def __str__(self):
        if self.coef_ is None:
            return f"ALM(distribution={self.distribution!r}, loss={self.loss!r})"

        time_str = f"Time elapsed: {self.time_elapsed_:.1f} seconds"

        class_str = f'ALM(distribution="{self.distribution}", loss="{self.loss}")'

        coef_names = ["(Intercept)"]
        coef_values = [self.intercept_]

        if self.coef_ is not None and len(self.coef_) > 0:
            n_coefs = len(self.coef_)
            if self._feature_names is not None and len(self._feature_names) == n_coefs:
                coef_names.extend(self._feature_names)
            else:
                for i in range(n_coefs):
                    coef_names.append(f"x{i + 1}")
            coef_values.extend(self.coef_)

        coef_lines = []
        for name, value in zip(coef_names, coef_values):
            coef_lines.append(f"{name:>15} {value:>15.8f}")

        return f"{time_str}\n{class_str}\n\nCoefficients:\n" + "\n".join(coef_lines)

    def __repr__(self):
        return f"ALM(distribution={self.distribution!r}, loss={self.loss!r})"
