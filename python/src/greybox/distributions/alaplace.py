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
    if log:
        return (
            np.log(alpha)
            + np.log(1 - alpha)
            - np.log(scale)
            - (q - mu) / scale * (alpha - indicator)
        )
    density = (
        alpha * (1 - alpha) / scale * np.exp(-(q - mu) / scale * (alpha - indicator))
    )
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
    return (
        1
        - indicator
        - (1 - indicator - alpha) * np.exp((indicator - alpha) / scale * (q - mu))
    )


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
    result = np.empty_like(p, dtype=float)

    mask_0 = p == 0
    mask_1 = p == 1
    mask_mid = ~(mask_0 | mask_1)

    result[mask_0] = -np.inf
    result[mask_1] = np.inf

    if np.any(mask_mid):
        p_mid = p[mask_mid]
        indicator = (p_mid <= alpha).astype(float)
        result[mask_mid] = mu + scale / (indicator - alpha) * np.log(
            (1 - indicator - p_mid) / (1 - indicator - alpha)
        )

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
