"""Geometric distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Geometric distribution.
"""

from scipy import stats


def dgeom(q, prob=0.5, log=False):
    """Geometric distribution probability mass function.

    Parameters
    ----------
    q : array_like
        Quantiles (non-negative integers, number of failures before first success).
    prob : float
        Probability of success.
    log : bool
        If True, return log-probability.

    Returns
    -------
    array
        Probability mass values.
    """
    if log:
        return stats.geom.logpmf(q + 1, p=prob)
    return stats.geom.pmf(q + 1, p=prob)


def pgeom(q, prob=0.5):
    """Geometric distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    prob : float
        Probability of success.

    Returns
    -------
    array
        CDF values.
    """
    return stats.geom.cdf(q + 1, p=prob)


def qgeom(p, prob=0.5):
    """Geometric distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    prob : float
        Probability of success.

    Returns
    -------
    array
        Quantile values.
    """
    return stats.geom.ppf(p, p=prob) - 1


def rgeom(n, prob=0.5):
    """Geometric distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    prob : float
        Probability of success.

    Returns
    -------
    array
        Random values.
    """
    return stats.geom.rvs(p=prob, size=n) - 1
