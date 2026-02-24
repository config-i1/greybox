"""Poisson distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Poisson distribution.
"""

from scipy import stats


def dpois(q, loc, log=False):
    """Poisson distribution probability mass function.

    Parameters
    ----------
    q : array_like
        Quantiles (non-negative integers).
    loc : float
        Mean parameter (lambda).
    log : bool
        If True, return log-probability.

    Returns
    -------
    array
        Probability mass values.
    """
    if log:
        return stats.poisson.logpmf(q, mu=loc)
    return stats.poisson.pmf(q, mu=loc)


def ppois(q, loc):
    """Poisson distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    loc : float
        Mean parameter.

    Returns
    -------
    array
        CDF values.
    """
    return stats.poisson.cdf(q, mu=loc)


def qpois(p, loc):
    """Poisson distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    loc : float
        Mean parameter.

    Returns
    -------
    array
        Quantile values.
    """
    return stats.poisson.ppf(p, mu=loc)


def rpois(n, loc):
    """Poisson distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    loc : float
        Mean parameter.

    Returns
    -------
    array
        Random values.
    """
    return stats.poisson.rvs(mu=loc, size=n)
