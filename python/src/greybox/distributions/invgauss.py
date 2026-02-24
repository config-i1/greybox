"""Inverse Gaussian distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Inverse Gaussian distribution.
"""

from scipy import stats


def dinvgauss(q, loc=1, scale=1, log=False):
    """Inverse Gaussian distribution density.

    Parameters
    ----------
    q : array_like
        Quantiles (must be positive).
    loc : float
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
    if log:
        return stats.invgauss.logpdf(q, mu=loc, scale=scale)
    return stats.invgauss.pdf(q, mu=loc, scale=scale)


def pinvgauss(q, loc=1, scale=1):
    """Inverse Gaussian distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    loc : float
        Mean parameter.
    scale : float
        Scale parameter.

    Returns
    -------
    array
        CDF values.
    """
    return stats.invgauss.cdf(q, mu=loc, scale=scale)


def qinvgauss(p, loc=1, scale=1):
    """Inverse Gaussian distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    loc : float
        Mean parameter.
    scale : float
        Scale parameter.

    Returns
    -------
    array
        Quantile values.
    """
    return stats.invgauss.ppf(p, mu=loc, scale=scale)


def rinvgauss(n, loc=1, scale=1):
    """Inverse Gaussian distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    loc : float
        Mean parameter.
    scale : float
        Scale parameter.

    Returns
    -------
    array
        Random values.
    """
    return stats.invgauss.rvs(mu=loc, scale=scale, size=n)
