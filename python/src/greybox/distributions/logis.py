"""Logistic distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Logistic distribution.
"""

import numpy as np
from scipy import stats


def dlogis(q, loc=0, scale=1, log=False):
    """Logistic distribution density.

    Parameters
    ----------
    q : array_like
        Quantiles.
    loc : float
        Location parameter.
    scale : float
        Scale parameter.
    log : bool
        If True, return log-density.

    Returns
    -------
    array
        Density values.
    """
    result = stats.logistic.pdf(q, loc=loc, scale=scale)
    if log:
        return np.log(result + 1e-300)
    return result


def plogis(q, loc=0, scale=1):
    """Logistic distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    loc : float
        Location parameter.
    scale : float
        Scale parameter.

    Returns
    -------
    array
        CDF values.
    """
    return stats.logistic.cdf(q, loc=loc, scale=scale)


def qlogis(p, loc=0, scale=1):
    """Logistic distribution quantile function.

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
    return stats.logistic.ppf(p, loc=loc, scale=scale)


def rlogis(n, loc=0, scale=1):
    """Logistic distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    loc : float
        Location parameter.
    scale : float
        Scale parameter.

    Returns
    -------
    array
        Random values.
    """
    return stats.logistic.rvs(loc=loc, scale=scale, size=n)
