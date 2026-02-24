"""Log-Normal distribution functions.

Density, cumulative distribution and quantile functions for the Log-Normal distribution.
Note: Random generation is not implemented as per requirements.
"""

import numpy as np
from scipy import stats


def dlnorm(q, loc=0, scale=1, log=False):
    """Log-Normal distribution density.

    ``loc`` is the mean of the underlying normal on the log scale (meanlog),
    ``scale`` is the corresponding standard deviation (sdlog).

    Parameters
    ----------
    q : array_like
        Quantiles (must be positive).
    loc : float
        Mean of the underlying normal distribution (on log scale).
    scale : float
        Standard deviation of the underlying normal distribution.
    log : bool
        If True, return log-density.

    Returns
    -------
    array
        Density values.
    """
    if log:
        return stats.lognorm.logpdf(q, s=scale, scale=np.exp(loc))
    return stats.lognorm.pdf(q, s=scale, scale=np.exp(loc))


def plnorm(q, loc=0, scale=1):
    """Log-Normal distribution CDF.

    ``loc`` is the mean of the underlying normal on the log scale (meanlog),
    ``scale`` is the corresponding standard deviation (sdlog).

    Parameters
    ----------
    q : array_like
        Quantiles.
    loc : float
        Mean of the underlying normal distribution.
    scale : float
        Standard deviation of the underlying normal distribution.

    Returns
    -------
    array
        CDF values.
    """
    return stats.lognorm.cdf(q, s=scale, scale=np.exp(loc))


def qlnorm(p, loc=0, scale=1):
    """Log-Normal distribution quantile function.

    ``loc`` is the mean of the underlying normal on the log scale (meanlog),
    ``scale`` is the corresponding standard deviation (sdlog).

    Parameters
    ----------
    p : array_like
        Probabilities.
    loc : float
        Mean of the underlying normal distribution.
    scale : float
        Standard deviation of the underlying normal distribution.

    Returns
    -------
    array
        Quantile values.
    """
    return stats.lognorm.ppf(p, s=scale, scale=np.exp(loc))
