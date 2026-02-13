"""Chi-squared distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Chi-squared distribution.
"""

import numpy as np
from scipy import stats


def dchi2(q, df, log=False):
    """Chi-squared distribution density.

    Parameters
    ----------
    q : array_like
        Quantiles (must be non-negative).
    df : float
        Degrees of freedom.
    log : bool
        If True, return log-density.

    Returns
    -------
    array
        Density values.
    """
    result = stats.chi2.pdf(q, df=df)
    if log:
        return np.log(result + 1e-300)
    return result


def pchi2(q, df):
    """Chi-squared distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    df : float
        Degrees of freedom.

    Returns
    -------
    array
        CDF values.
    """
    return stats.chi2.cdf(q, df=df)


def qchi2(p, df):
    """Chi-squared distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    df : float
        Degrees of freedom.

    Returns
    -------
    array
        Quantile values.
    """
    return stats.chi2.ppf(p, df=df)


def rchi2(n, df):
    """Chi-squared distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    df : float
        Degrees of freedom.

    Returns
    -------
    array
        Random values.
    """
    return stats.chi2.rvs(df=df, size=n)
