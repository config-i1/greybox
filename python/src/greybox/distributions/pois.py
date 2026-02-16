"""Poisson distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Poisson distribution.
"""

from scipy import stats


def dpois(q, mu, log=False):
    """Poisson distribution probability mass function.

    Parameters
    ----------
    q : array_like
        Quantiles (non-negative integers).
    mu : float
        Mean parameter (lambda).
    log : bool
        If True, return log-probability.

    Returns
    -------
    array
        Probability mass values.
    """
    if log:
        return stats.poisson.logpmf(q, mu=mu)
    return stats.poisson.pmf(q, mu=mu)


def ppois(q, mu):
    """Poisson distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    mu : float
        Mean parameter.

    Returns
    -------
    array
        CDF values.
    """
    return stats.poisson.cdf(q, mu=mu)


def qpois(p, mu):
    """Poisson distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    mu : float
        Mean parameter.

    Returns
    -------
    array
        Quantile values.
    """
    return stats.poisson.ppf(p, mu=mu)


def rpois(n, mu):
    """Poisson distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    mu : float
        Mean parameter.

    Returns
    -------
    array
        Random values.
    """
    return stats.poisson.rvs(mu=mu, size=n)
