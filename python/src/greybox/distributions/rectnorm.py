"""Rectified Normal distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Rectified Normal distribution.
If x ~ N(mu, sigma^2) then y = max(0, x) follows Rectified Normal distribution.
"""

import numpy as np
from scipy import stats


def drectnorm(q, mu=0, sigma=1, log=False):
    """Rectified Normal distribution density.

    f_y = I(x<=0) * F_x(mu, sigma) + I(x>0) * f_x(x, mu, sigma)

    Parameters
    ----------
    q : array_like
        Quantiles.
    mu : float
        Location parameter.
    sigma : float
        Scale parameter.
    log : bool
        If True, return log-density.

    Returns
    -------
    array
        Density values.
    """
    q = np.asarray(q)
    indicator = (q > 0).astype(float)
    density = indicator * stats.norm.pdf(q, loc=mu, scale=sigma) + (
        1 - indicator
    ) * stats.norm.cdf(0, loc=mu, scale=sigma)
    density = np.maximum(density, 1e-300)
    if log:
        return np.log(density)
    return density


def prectnorm(q, mu=0, sigma=1):
    """Rectified Normal distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    mu : float
        Location parameter.
    sigma : float
        Scale parameter.

    Returns
    -------
    array
        CDF values.
    """
    q = np.asarray(q)
    indicator = (q > 0).astype(float)
    return indicator * stats.norm.cdf(q, loc=mu, scale=sigma) + (
        1 - indicator
    ) * stats.norm.cdf(0, loc=mu, scale=sigma)


def qrectnorm(p, mu=0, sigma=1):
    """Rectified Normal distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    mu : float
        Location parameter.
    sigma : float
        Scale parameter.

    Returns
    -------
    array
        Quantile values.
    """
    p = np.asarray(p)
    return np.maximum(stats.norm.ppf(p, loc=mu, scale=sigma), 0)


def rrectnorm(n, mu=0, sigma=1):
    """Rectified Normal distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    mu : float
        Location parameter.
    sigma : float
        Scale parameter.

    Returns
    -------
    array
        Random values.
    """
    return np.maximum(stats.norm.rvs(loc=mu, scale=sigma, size=n), 0)
