"""Negative Binomial distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Negative Binomial distribution.
"""

from scipy import stats


def dnbinom(q, loc=1, size=1, log=False):
    """Negative Binomial distribution probability mass function.

    Parameters
    ----------
    q : array_like
        Quantiles (non-negative integers).
    loc : float
        Mean parameter.
    size : float
        Dispersion parameter (number of successes).
    log : bool
        If True, return log-probability.

    Returns
    -------
    array
        Probability mass values.
    """
    if log:
        return stats.nbinom.logpmf(q, n=size, p=size / (size + loc))
    return stats.nbinom.pmf(q, n=size, p=size / (size + loc))


def pnbinom(q, loc=1, size=1):
    """Negative Binomial distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    loc : float
        Mean parameter.
    size : float
        Dispersion parameter.

    Returns
    -------
    array
        CDF values.
    """
    return stats.nbinom.cdf(q, n=size, p=size / (size + loc))


def qnbinom(p, loc=1, size=1):
    """Negative Binomial distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    loc : float
        Mean parameter.
    size : float
        Dispersion parameter.

    Returns
    -------
    array
        Quantile values.
    """
    return stats.nbinom.ppf(p, n=size, p=size / (size + loc))


def rnbinom(n, loc=1, size=1):
    """Negative Binomial distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    loc : float
        Mean parameter.
    size : float
        Dispersion parameter.

    Returns
    -------
    array
        Random values.
    """
    return stats.nbinom.rvs(n=size, p=size / (size + loc), size=n)
