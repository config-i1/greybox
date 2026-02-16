"""Log-Laplace distribution functions.

Density and quantile functions for the Log-Laplace distribution.
Log-Laplace is the distribution of exp(X) where X ~ Laplace.
Note: Random generation and CDF are not implemented as per requirements.
"""

import numpy as np
from scipy import stats


def dllaplace(q, loc=0, scale=1, log=False):
    """Log-Laplace distribution density.

    The density is obtained by transforming a Laplace distribution
    through the exponential function with Jacobian adjustment.

    f(y) = (1/scale) * exp(-(abs(log(y) - loc) / scale)) / y

    Parameters
    ----------
    q : array_like
        Quantiles (must be positive).
    loc : float
        Location parameter (of underlying Laplace).
    scale : float
        Scale parameter.
    log : bool
        If True, return log-density.

    Returns
    -------
    array
        Density values.
    """
    q = np.asarray(q)
    log_q = np.log(q)
    density = stats.laplace.pdf(log_q, loc=loc, scale=scale) / q
    density = np.maximum(density, 1e-300)
    if log:
        return np.log(density)
    return density


def qllaplace(p, loc=0, scale=1):
    """Log-Laplace distribution quantile function.

    Quantiles are obtained by exponentiating Laplace quantiles.

    Parameters
    ----------
    p : array_like
        Probabilities.
    loc : float
        Location parameter.
    scale : float
        Scale parameter.

    Returns
    -------
    array
        Quantile values.
    """
    return np.exp(stats.laplace.ppf(p, loc=loc, scale=scale))
