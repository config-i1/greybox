"""Exponential distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Exponential distribution.
"""

from scipy import stats


def dexp(q, loc=0, scale=1, log=False):
    """Exponential distribution density.

    Parameters
    ----------
    q : array_like
        Quantiles (must be >= loc).
    loc : float
        Location parameter.
    scale : float
        Scale parameter (1/lambda).
    log : bool
        If True, return log-density.

    Returns
    -------
    array
        Density values.
    """
    if log:
        return stats.expon.logpdf(q, loc=loc, scale=scale)
    return stats.expon.pdf(q, loc=loc, scale=scale)


def pexp(q, loc=0, scale=1):
    """Exponential distribution CDF.

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
    return stats.expon.cdf(q, loc=loc, scale=scale)


def qexp(p, loc=0, scale=1):
    """Exponential distribution quantile function.

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
    return stats.expon.ppf(p, loc=loc, scale=scale)


def rexp(n, loc=0, scale=1):
    """Exponential distribution random number generation.

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
    return stats.expon.rvs(loc=loc, scale=scale, size=n)
