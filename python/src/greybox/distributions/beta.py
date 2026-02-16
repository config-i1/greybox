"""Beta distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Beta distribution.
"""

from scipy import stats


def dbeta(q, a=1, b=1, log=False):
    """Beta distribution density.

    Parameters
    ----------
    q : array_like
        Quantiles (must be in [0, 1]).
    a : float
        First shape parameter (alpha).
    b : float
        Second shape parameter (beta).
    log : bool
        If True, return log-density.

    Returns
    -------
    array
        Density values.
    """
    if log:
        return stats.beta.logpdf(q, a=a, b=b)
    return stats.beta.pdf(q, a=a, b=b)


def pbeta(q, a=1, b=1):
    """Beta distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    a : float
        First shape parameter.
    b : float
        Second shape parameter.

    Returns
    -------
    array
        CDF values.
    """
    return stats.beta.cdf(q, a=a, b=b)


def qbeta(p, a=1, b=1):
    """Beta distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    a : float
        First shape parameter.
    b : float
        Second shape parameter.

    Returns
    -------
    array
        Quantile values.
    """
    return stats.beta.ppf(p, a=a, b=b)


def rbeta(n, a=1, b=1):
    """Beta distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    a : float
        First shape parameter.
    b : float
        Second shape parameter.

    Returns
    -------
    array
        Random values.
    """
    return stats.beta.rvs(a=a, b=b, size=n)
