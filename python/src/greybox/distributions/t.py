"""T-distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Student's t-distribution.
"""

import numpy as np
from scipy import stats


def dt(q, df, loc=0, scale=1, log=False):
    """T-distribution density.

    Parameters
    ----------
    q : array_like
        Quantiles.
    df : float
        Degrees of freedom.
    loc : float
        Location parameter.
    scale : float
        Scale parameter.
    log : bool
        If True, return log-density.

    Returns
    -------
    array
        Density values.
    """
    result = stats.t.pdf(q, df=df, loc=loc, scale=scale)
    if log:
        return np.log(result + 1e-300)
    return result


def pt(q, df, loc=0, scale=1):
    """T-distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    df : float
        Degrees of freedom.
    loc : float
        Location parameter.
    scale : float
        Scale parameter.

    Returns
    -------
    array
        CDF values.
    """
    return stats.t.cdf(q, df=df, loc=loc, scale=scale)


def qt(p, df, loc=0, scale=1):
    """T-distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    df : float
        Degrees of freedom.
    loc : float
        Location parameter.
    scale : float
        Scale parameter.

    Returns
    -------
    array
        Quantile values.
    """
    return stats.t.ppf(p, df=df, loc=loc, scale=scale)


def rt(n, df, loc=0, scale=1):
    """T-distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    df : float
        Degrees of freedom.
    loc : float
        Location parameter.
    scale : float
        Scale parameter.

    Returns
    -------
    array
        Random values.
    """
    return stats.t.rvs(df=df, loc=loc, scale=scale, size=n)
