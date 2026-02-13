"""Log-Normal distribution functions.

Density, cumulative distribution and quantile functions for the Log-Normal distribution.
Note: Random generation is not implemented as per requirements.
"""

import numpy as np
from scipy import stats


def dlnorm(q, meanlog=0, sdlog=1, log=False):
    """Log-Normal distribution density.

    Parameters
    ----------
    q : array_like
        Quantiles (must be positive).
    meanlog : float
        Mean of the underlying normal distribution (on log scale).
    sdlog : float
        Standard deviation of the underlying normal distribution.
    log : bool
        If True, return log-density.

    Returns
    -------
    array
        Density values.
    """
    result = stats.lognorm.pdf(q, s=sdlog, scale=np.exp(meanlog))
    if log:
        return np.log(result + 1e-300)
    return result


def plnorm(q, meanlog=0, sdlog=1):
    """Log-Normal distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    meanlog : float
        Mean of the underlying normal distribution.
    sdlog : float
        Standard deviation of the underlying normal distribution.

    Returns
    -------
    array
        CDF values.
    """
    return stats.lognorm.cdf(q, s=sdlog, scale=np.exp(meanlog))


def qlnorm(p, meanlog=0, sdlog=1):
    """Log-Normal distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    meanlog : float
        Mean of the underlying normal distribution.
    sdlog : float
        Standard deviation of the underlying normal distribution.

    Returns
    -------
    array
        Quantile values.
    """
    return stats.lognorm.ppf(p, s=sdlog, scale=np.exp(meanlog))
