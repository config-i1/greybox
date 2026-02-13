"""Inverse Gaussian distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Inverse Gaussian distribution.
"""

import numpy as np
from scipy import stats


def dinvgauss(q, mu=1, scale=1, log=False):
    """Inverse Gaussian distribution density.

    Parameters
    ----------
    q : array_like
        Quantiles (must be positive).
    mu : float
        Mean parameter.
    scale : float
        Scale parameter.
    log : bool
        If True, return log-density.

    Returns
    -------
    array
        Density values.
    """
    result = stats.invgauss.pdf(q, mu=mu, scale=scale)
    if log:
        return np.log(result + 1e-300)
    return result


def pinvgauss(q, mu=1, scale=1):
    """Inverse Gaussian distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    mu : float
        Mean parameter.
    scale : float
        Scale parameter.

    Returns
    -------
    array
        CDF values.
    """
    return stats.invgauss.cdf(q, mu=mu, scale=scale)


def qinvgauss(p, mu=1, scale=1):
    """Inverse Gaussian distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    mu : float
        Mean parameter.
    scale : float
        Scale parameter.

    Returns
    -------
    array
        Quantile values.
    """
    return stats.invgauss.ppf(p, mu=mu, scale=scale)


def rinvgauss(n, mu=1, scale=1):
    """Inverse Gaussian distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    mu : float
        Mean parameter.
    scale : float
        Scale parameter.

    Returns
    -------
    array
        Random values.
    """
    return stats.invgauss.rvs(mu=mu, scale=scale, size=n)
