"""Binomial distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Binomial distribution.
"""

from scipy import stats


def dbinom(q, size=1, prob=0.5, log=False):
    """Binomial distribution probability mass function.

    Parameters
    ----------
    q : array_like
        Quantiles (non-negative integers, <= size).
    size : int
        Number of trials.
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
        return stats.binom.logpmf(q, n=size, p=prob)
    return stats.binom.pmf(q, n=size, p=prob)


def pbinom(q, size=1, prob=0.5):
    """Binomial distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    size : int
        Number of trials.
    prob : float
        Probability of success.

    Returns
    -------
    array
        CDF values.
    """
    return stats.binom.cdf(q, n=size, p=prob)


def qbinom(p, size=1, prob=0.5):
    """Binomial distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    size : int
        Number of trials.
    prob : float
        Probability of success.

    Returns
    -------
    array
        Quantile values.
    """
    return stats.binom.ppf(p, n=size, p=prob)


def rbinom(n, size=1, prob=0.5):
    """Binomial distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    size : int
        Number of trials.
    prob : float
        Probability of success.

    Returns
    -------
    array
        Random values.
    """
    return stats.binom.rvs(n=size, p=prob, size=n)
