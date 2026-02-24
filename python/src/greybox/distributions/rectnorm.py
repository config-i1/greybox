"""Rectified Normal distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Rectified Normal distribution.
If x ~ N(loc, scale^2) then y = max(0, x) follows Rectified Normal distribution.
"""

import numpy as np
from scipy import stats


def drectnorm(q, loc=0, scale=1, log=False):
    """Rectified Normal distribution density.

    f_y = I(x<=0) * F_x(loc, scale) + I(x>0) * f_x(x, loc, scale)

    Parameters
    ----------
    q : array_like
        Quantiles.
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
    q = np.asarray(q)
    is_positive = q > 0
    if log:
        log_density = np.where(
            is_positive,
            stats.norm.logpdf(q, loc=loc, scale=scale),
            stats.norm.logcdf(0, loc=loc, scale=scale),
        )
        return log_density
    indicator = is_positive.astype(float)
    density = indicator * stats.norm.pdf(q, loc=loc, scale=scale) + (
        1 - indicator
    ) * stats.norm.cdf(0, loc=loc, scale=scale)
    return density


def prectnorm(q, loc=0, scale=1):
    """Rectified Normal distribution CDF.

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
    q = np.asarray(q)
    indicator = (q > 0).astype(float)
    return indicator * stats.norm.cdf(q, loc=loc, scale=scale) + (
        1 - indicator
    ) * stats.norm.cdf(0, loc=loc, scale=scale)


def qrectnorm(p, loc=0, scale=1):
    """Rectified Normal distribution quantile function.

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
    p = np.asarray(p)
    return np.maximum(stats.norm.ppf(p, loc=loc, scale=scale), 0)


def rrectnorm(n, loc=0, scale=1):
    """Rectified Normal distribution random number generation.

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
    return np.maximum(stats.norm.rvs(loc=loc, scale=scale, size=n), 0)
