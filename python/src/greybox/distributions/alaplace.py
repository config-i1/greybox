"""Asymmetric Laplace distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Asymmetric Laplace distribution.
"""

import numpy as np


def dalaplace(q, mu=0, scale=1, alpha=0.5, log=False):
    """Asymmetric Laplace distribution density.

    f(x) = alpha * (1-alpha) / scale * exp(-(x-mu)/scale * (alpha - I(x<=mu)))

    Parameters
    ----------
    q : array_like
        Quantiles.
    mu : float
        Location parameter.
    scale : float
        Scale parameter.
    alpha : float
        Asymmetry parameter (0 < alpha < 1).
    log : bool
        If True, return log-density.

    Returns
    -------
    array
        Density values.
    """
    q = np.asarray(q)
    indicator = (q <= mu).astype(float)
    density = alpha * (1 - alpha) / scale * np.exp(-(q - mu) / scale * (alpha - indicator))
    if log:
        return np.log(density + 1e-300)
    return density


def palaplace(q, mu=0, scale=1, alpha=0.5):
    """Asymmetric Laplace distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    mu : float
        Location parameter.
    scale : float
        Scale parameter.
    alpha : float
        Asymmetry parameter (0 < alpha < 1).

    Returns
    -------
    array
        CDF values.
    """
    q = np.asarray(q)
    indicator = (q <= mu).astype(float)
    return 1 - indicator - (1 - indicator - alpha) * np.exp((indicator - alpha) / scale * (q - mu))


def qalaplace(p, mu=0, scale=1, alpha=0.5):
    """Asymmetric Laplace distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    mu : float
        Location parameter.
    scale : float
        Scale parameter.
    alpha : float
        Asymmetry parameter (0 < alpha < 1).

    Returns
    -------
    array
        Quantile values.
    """
    p = np.asarray(p)
    indicator = (p <= alpha).astype(float)
    result = mu + scale / (indicator - alpha) * np.log((1 - indicator - p) / (1 - indicator - alpha))
    result = np.where(p == 0, -np.inf, result)
    result = np.where(p == 1, np.inf, result)
    return result


def ralaplace(n, mu=0, scale=1, alpha=0.5):
    """Asymmetric Laplace distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    mu : float
        Location parameter.
    scale : float
        Scale parameter.
    alpha : float
        Asymmetry parameter (0 < alpha < 1).

    Returns
    -------
    array
        Random values.
    """
    return qalaplace(np.random.uniform(0, 1, n), mu, scale, alpha)
