"""Logit-Normal distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the distribution that becomes normal after the Logit
transformation.
"""

import numpy as np
from scipy import stats


def dlogitnorm(q, mu=0, sigma=1, log=False):
    """Logit-Normal distribution density.

    f(y) = 1/(sqrt(2*pi)*sigma*y*(1-y)) * exp(-(logit(y) - mu)^2 / (2*sigma^2))

    Parameters
    ----------
    q : array_like
        Quantiles (must be in (0, 1)).
    mu : float
        Location parameter (on logit scale).
    sigma : float
        Scale parameter.
    log : bool
        If True, return log-density.

    Returns
    -------
    array
        Density values.
    """
    q = np.clip(q, 1e-10, 1 - 1e-10)
    logit_q = np.log(q / (1 - q))
    density = 1 / (sigma * np.sqrt(2 * np.pi) * q * (1 - q)) * np.exp(-(logit_q - mu)**2 / (2 * sigma**2))
    density = np.maximum(density, 1e-300)
    if log:
        return np.log(density)
    return density


def plogitnorm(q, mu=0, sigma=1):
    """Logit-Normal distribution CDF.

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
    result = np.zeros_like(q, dtype=float)
    result[q >= 0] = stats.norm.cdf(np.log(q[q >= 0] / (1 - q[q >= 0])), loc=mu, scale=sigma)
    result[q < 0] = 0
    result[q >= 1] = 1
    return result


def qlogitnorm(p, mu=0, sigma=1):
    """Logit-Normal distribution quantile function.

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
    result = np.exp(stats.norm.ppf(p, loc=mu, scale=sigma)) / (1 + np.exp(stats.norm.ppf(p, loc=mu, scale=sigma)))
    result = np.where(np.isnan(result), 0, result)
    return result


def rlogitnorm(n, mu=0, sigma=1):
    """Logit-Normal distribution random number generation.

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
    return qlogitnorm(np.random.uniform(0, 1, n), mu, sigma)
