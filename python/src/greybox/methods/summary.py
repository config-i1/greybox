"""Summary results for ALM model."""

import numpy as np
from dataclasses import dataclass


@dataclass
class SummaryResult:
    """Summary results for ALM model.

    Attributes
    ----------
    coefficients : np.ndarray
        Parameter estimates (including intercept).
    se : np.ndarray
        Standard errors of coefficients.
    t_stat : np.ndarray
        T-statistics for each coefficient.
    p_value : np.ndarray
        P-values for each coefficient.
    lower_ci : np.ndarray
        Lower confidence interval bounds.
    upper_ci : np.ndarray
        Upper confidence interval bounds.
    scale : float
        Scale parameter (sigma).
    df_residual : int
        Degrees of freedom for residuals.
    distribution : str
        Distribution used in the model.
    log_lik : float
        Log-likelihood of the model.
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    nobs : int
        Number of observations.
    nparam : int
        Number of parameters.
    """

    coefficients: np.ndarray
    se: np.ndarray
    t_stat: np.ndarray
    p_value: np.ndarray
    lower_ci: np.ndarray
    upper_ci: np.ndarray
    scale: float
    df_residual: int
    distribution: str
    log_lik: float
    aic: float
    bic: float
    nobs: int
    nparam: int

    def __str__(self) -> str:
        """String representation of summary."""
        names = ["(Intercept)", "coef"]
        if len(self.coefficients) > 2:
            names = ["(Intercept)"] + [
                f"x{i}" for i in range(1, len(self.coefficients))
            ]

        lines = [
            "=" * 60,
            "ALM Model Summary",
            "=" * 60,
            "",
            f"Distribution: {self.distribution}",
            f"Number of observations: {self.nobs}",
            f"Number of parameters: {self.nparam}",
            f"Degrees of freedom (residual): {self.df_residual}",
            "",
            f"Scale (sigma): {self.scale:.6f}",
            f"Log-likelihood: {self.log_lik:.4f}",
            f"AIC: {self.aic:.4f}",
            f"BIC: {self.bic:.4f}",
            "",
            "=" * 60,
            "Coefficients:",
            "-" * 60,
            f"{'Name':<15} {'Estimate':>12} {'Std.Err':>12} {'t-value':>12} {'Pr(>|t|)':>12}",
            "-" * 60,
        ]

        for i, name in enumerate(names[: len(self.coefficients)]):
            lines.append(
                f"{name:<15} {self.coefficients[i]:>12.4f} "
                f"{self.se[i]:>12.4f} {self.t_stat[i]:>12.4f} "
                f"{self.p_value[i]:>12.4f}"
            )

        lines.extend(["-" * 60, ""])

        return "\n".join(lines)
