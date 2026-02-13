"""Folded Normal distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the folded normal distribution.
"""

import numpy as np
from scipy import stats
from scipy import optimize


def dfnorm(q, mu=0, sigma=1, log=False):
    """Folded Normal distribution density.

    f(x) = 1/sqrt(2*pi*sigma^2) * (exp(-(x-mu)^2 / (2*sigma^2)) + exp(-(x+mu)^2 / (2*sigma^2)))

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
    density = (
        1
        / (np.sqrt(2 * np.pi) * sigma)
        * (
            np.exp(-((np.abs(q) - mu) ** 2) / (2 * sigma**2))
            + np.exp(-((np.abs(q) + mu) ** 2) / (2 * sigma**2))
        )
    )
    is_zero = q < 0
    result = np.where(is_zero, 0.0, np.maximum(density, 1e-300))
    if log:
        return np.log(result)
    return result


def pfnorm(q, mu=0, sigma=1):
    """Folded Normal distribution CDF.

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
    result = (
        stats.norm.cdf(q, loc=mu, scale=sigma)
        + stats.norm.cdf(q, loc=-mu, scale=sigma)
        - 1
    )
    result = np.where(q < 0, 0, result)
    return result


def qfnorm(p, mu=0, sigma=1):
    """Folded Normal distribution quantile function.

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
    p = np.atleast_1d(p)
    result = np.zeros_like(p, dtype=float)

    for i, pi in enumerate(p):
        if pi == 0:
            result[i] = 0
        elif pi == 1:
            result[i] = np.inf
        else:

            def objective(x):
                return pfnorm(x, mu, sigma) - pi

            try:
                result[i] = optimize.brentq(objective, 0, mu + 10 * sigma)
            except ValueError:
                result[i] = abs(stats.norm.ppf(pi, loc=mu, scale=sigma))

    return result.squeeze()


def rfnorm(n, mu=0, sigma=1):
    """Folded Normal distribution random number generation.

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
    return np.abs(stats.norm.rvs(loc=mu, scale=sigma, size=n))
