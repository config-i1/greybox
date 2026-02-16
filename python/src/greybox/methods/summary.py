"""Summary results for ALM model."""

import numpy as np
from dataclasses import dataclass


DISTRIBUTION_NAMES = {
    "dnorm": "Normal",
    "dlaplace": "Laplace",
    "ds": "S",
    "dgnorm": "Generalized Normal",
    "dlogis": "Logistic",
    "dt": "Student-t",
    "dalaplace": "Asymmetric Laplace",
    "dlnorm": "Log-Normal",
    "dllaplace": "Log-Laplace",
    "dls": "Log-S",
    "dlgnorm": "Log-Generalized Normal",
    "dbcnorm": "Box-Cox Normal",
    "dinvgauss": "Inverse Gaussian",
    "dgamma": "Gamma",
    "dexp": "Exponential",
    "dchisq": "Chi-Squared",
    "dfnorm": "Folded Normal",
    "drectnorm": "Rectified Normal",
    "dpois": "Poisson",
    "dnbinom": "Negative Binomial",
    "dbinom": "Binomial",
    "dgeom": "Geometric",
    "dbeta": "Beta",
    "dlogitnorm": "Logit-Normal",
    "plogis": "Logit",
    "pnorm": "Probit",
}


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
    response_variable : str
        Name of the response variable.
    loss : str
        Loss function used in estimation.
    aicc : float
        Corrected AIC.
    bicc : float
        Corrected BIC.
    feature_names : list
        Names of the predictor variables.
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
    response_variable: str = "y"
    loss: str = "likelihood"
    aicc: float | None = None
    bicc: float | None = None
    feature_names: list | None = None

    def __str__(self) -> str:
        """String representation of summary."""
        dist_name = DISTRIBUTION_NAMES.get(self.distribution, self.distribution)

        names = ["(Intercept)"]
        if self.feature_names is not None:
            names.extend(self.feature_names)
        else:
            for i in range(len(self.coefficients) - 1):
                names.append(f"x{i + 1}")

        level_low = (1 - 0.95) / 2 * 100
        level_high = (1 + 0.95) / 2 * 100
        col1 = f"Lower {level_low:.1f}%"
        col2 = f"Upper {level_high:.1f}%"

        lines = []
        lines.append(f"Response variable: {self.response_variable}")
        lines.append(f"Distribution used in the estimation: {dist_name}")
        lines.append(f"Loss function used in estimation: {self.loss}")
        lines.append("")
        lines.append("Coefficients:")
        header = f"{'':>12} {'Estimate':>12} {'Std. Error':>12} {col1:>12} {col2:>12}"
        lines.append(header)
        lines.append("-" * 60)

        for i, name in enumerate(names):
            sig = " *" if self.p_value[i] < 0.05 else ""
            lines.append(
                f"{name:>12} {self.coefficients[i]:>12.4f} {self.se[i]:>12.4f} "
                f"{self.lower_ci[i]:>12.4f} {self.upper_ci[i]:>12.4f}{sig}"
            )

        lines.append("")
        lines.append(f"Error standard deviation: {self.scale:.4f}")
        lines.append(f"Sample size: {self.nobs}")
        lines.append(f"Number of estimated parameters: {self.nparam}")
        lines.append(f"Number of degrees of freedom: {self.df_residual}")
        lines.append("")
        lines.append("Information criteria:")

        ic_names = ["AIC", "AICc", "BIC", "BICc"]
        ic_values = [self.aic]
        ic_values.append(self.aicc if self.aicc is not None else np.nan)
        ic_values.append(self.bic)
        ic_values.append(self.bicc if self.bicc is not None else np.nan)

        ic_line = f"{'':>5}"
        for name, val in zip(ic_names, ic_values):
            ic_line += f"{val:>12.4f}"
        lines.append(ic_line)

        return "\n".join(lines)
