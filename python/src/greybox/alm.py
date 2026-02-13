"""Augmented Linear Model (ALM) for greybox.

This module provides the ALM estimator following scikit-learn principles.
Users should first use the formula module to get X and y, then pass them
to the fit() method.
"""

import numpy as np
from scipy import optimize

from .fitters import scaler_internal, extractor_fitted, extractor_residuals
from .cost_function import cf
from . import distributions as dist


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
    maxiter : int, default=500
        Maximum number of optimization iterations.
    tol : float, default=1e-6
        Tolerance for convergence.
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
    """

    DISTRIBUTIONS = [
        "dnorm", "dlaplace", "ds", "dgnorm", "dlogis", "dt", "dalaplace",
        "dlnorm", "dllaplace", "dls", "dlgnorm", "dbcnorm",
        "dinvgauss", "dgamma", "dexp", "dchisq",
        "dfnorm", "drectnorm",
        "dpois", "dnbinom", "dbinom", "dgeom",
        "dbeta", "dlogitnorm",
        "plogis", "pnorm"
    ]

    LOSS_FUNCTIONS = [
        "likelihood", "MSE", "MAE", "HAM", "LASSO", "RIDGE", "ROLE"
    ]

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
        maxiter=500,
        tol=1e-6,
        verbose=0
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
        self.maxiter = maxiter
        self.tol = tol
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
        self._result = None
        self._n_features = None

    def _validate_params(self):
        """Validate input parameters."""
        if self.distribution not in self.DISTRIBUTIONS:
            raise ValueError(
                f"Invalid distribution: {self.distribution}. "
                f"Choose from: {self.DISTRIBUTIONS}"
            )

        if self.loss not in self.LOSS_FUNCTIONS:
            raise ValueError(
                f"Invalid loss: {self.loss}. "
                f"Choose from: {self.LOSS_FUNCTIONS}"
            )

        if not isinstance(self.orders, (tuple, list)) or len(self.orders) != 3:
            raise ValueError("orders must be a tuple of 3 integers (p, d, q)")

    def fit(self, X, y):
        """Fit the ALM model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Design matrix. Should include intercept column (column of ones)
            as first column if you want an intercept.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : ALM
            Fitted estimator.
        """
        self._validate_params()

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
                except:
                    B_init = np.zeros(n_features)
        elif self.distribution in ("dlogitnorm",):
            y_clip = np.clip(y, 1e-10, 1 - 1e-10)
            y_transformed = np.log(y_clip / (1 - y_clip))
            try:
                B_init = np.linalg.lstsq(X, y_transformed, rcond=None)[0]
            except:
                B_init = np.zeros(n_features)
        else:
            try:
                B_init = np.linalg.lstsq(X, y, rcond=None)[0]
            except:
                B_init = np.zeros(n_features)

        ar_order = 0
        i_order = 0

        def objective(B):
            return cf(
                B,
                self.distribution,
                self.loss,
                y,
                X,
                ar_order=0,
                i_order=0,
                lambda_val=self.lambda_l1 if self.loss == "LASSO" else (self.lambda_l2 if self.loss == "RIDGE" else 0.0),
                other=other_val,
                a_parameter_provided=a_parameter_provided,
                trim=self.trim,
                lambda_bc=self.lambda_bc if self.distribution == "dbcnorm" else 0.0,
                size=self.size if self.distribution == "dbinom" else 1.0
            )

        try:
            result = optimize.minimize(
                objective,
                B_init,
                method="Nelder-Mead",
                options={"maxiter": self.maxiter, "xatol": self.tol, "fatol": self.tol}
            )
            B_opt = result.x
            self.n_iter_ = result.nit
            self._result = result
        except Exception as e:
            if self.verbose > 0:
                print(f"Optimization failed: {e}")
            B_opt = B_init
            self.n_iter_ = 0

        if n_features > 0:
            self.intercept_ = B_opt[0]
            self.coef_ = B_opt[1:] if len(B_opt) > 1 else np.array([])

        fitter_return = {
            "mu": X @ B_opt,
            "scale": 1.0,
            "other": other_val,
        }

        scale = scaler_internal(
            B_opt, self.distribution, y, X,
            fitter_return["mu"], other_val,
            np.ones(len(y), dtype=bool),
            np.sum(np.ones(len(y), dtype=bool))
        )
        fitter_return["scale"] = scale
        self.scale_ = scale
        self.other_ = other_val

        self.fitted_values_ = extractor_fitted(
            self.distribution,
            fitter_return["mu"],
            scale,
            self.lambda_bc if self.distribution == "dbcnorm" else 0.0
        )

        self.residuals_ = extractor_residuals(
            self.distribution,
            fitter_return["mu"],
            y,
            self.lambda_bc if self.distribution == "dbcnorm" else 0.0
        )

        self.loss_value_ = objective(B_opt)

        if self.loss == "likelihood":
            self.log_lik_ = -self.loss_value_
            n_params = n_features
            if self.distribution not in ("dexp", "dpois", "dgeom", "dbinom", "plogis", "pnorm"):
                n_params += 1
            self.aic_ = 2 * n_params - 2 * self.log_lik_
            self.bic_ = n_params * np.log(n_samples) - 2 * self.log_lik_

        return self

    def predict(self, X):
        """Predict using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Design matrix. Should have same number of features as training data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        if self.fitted_values_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

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

        return extractor_fitted(
            self.distribution,
            mu,
            self.scale_,
            self.lambda_bc if self.distribution == "dbcnorm" else 0.0
        )

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
        y_pred = self.predict(X)

        if metric == "likelihood":
            if self.log_lik_ is not None:
                return self.log_lik_
            y = np.asarray(y)
            log_lik = 0.0
            y_otU = y[y != 0] if self.distribution not in ("dlnorm", "dllaplace", "dls", "dlgnorm") else y
            mu_otU = y_pred[y != 0] if self.distribution not in ("dlnorm", "dllaplace", "dls", "dlgnorm") else y_pred
            if self.distribution == "dnorm":
                log_lik = np.sum(dist.dnorm(y_otU, mean=mu_otU, sd=self.scale_, log=True))
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
            "maxiter": self.maxiter,
            "tol": self.tol,
            "verbose": self.verbose
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

    def __repr__(self):
        return (
            f"ALM(distribution={self.distribution!r}, "
            f"loss={self.loss!r})"
        )
