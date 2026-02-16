"""Gamma distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Gamma distribution.
"""

from scipy import stats


def dgamma(q, shape=1, scale=1, log=False):
    """Gamma distribution density.

    Parameters
    ----------
    q : array_like
        Quantiles (must be positive).
    shape : float
        Shape parameter (alpha).
    scale : float
        Scale parameter (theta).
    log : bool
        If True, return log-density.

    Returns
    -------
    array
        Density values.
    """
    if log:
        return stats.gamma.logpdf(q, a=shape, scale=scale)
    return stats.gamma.pdf(q, a=shape, scale=scale)


def pgamma(q, shape=1, scale=1):
    """Gamma distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    shape : float
        Shape parameter.
    scale : float
        Scale parameter.

    Returns
    -------
    array
        CDF values.
    """
    return stats.gamma.cdf(q, a=shape, scale=scale)


def qgamma(p, shape=1, scale=1):
    """Gamma distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    shape : float
        Shape parameter.
    scale : float
        Scale parameter.

    Returns
    -------
    array
        Quantile values.
    """
    return stats.gamma.ppf(p, a=shape, scale=scale)


def rgamma(n, shape=1, scale=1):
    """Gamma distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    shape : float
        Shape parameter.
    scale : float
        Scale parameter.

    Returns
    -------
    array
        Random values.
    """
    return stats.gamma.rvs(a=shape, scale=scale, size=n)
