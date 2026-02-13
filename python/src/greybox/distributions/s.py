"""S-Distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the S distribution.
"""

import numpy as np

from .gnorm import qgnorm


def ds(q, mu=0, scale=1, log=False):
    """S-distribution density.

    Density function: f(x) = 1/(4*scale^2) * exp(-sqrt(|mu - x|) / scale)

    Parameters
    ----------
    q : array_like
        Quantiles.
    mu : float
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
    density = 1 / (4 * scale**2) * np.exp(-np.sqrt(np.abs(mu - q)) / scale)
    if log:
        return np.log(density + 1e-300)
    return density


def ps(q, mu=0, scale=1):
    """S-distribution CDF."""
    sign_val = np.sign(q - mu)
    sqrt_term = np.sqrt(np.abs(mu - q))
    return 0.5 + 0.5 * sign_val * (
        1 - 1 / scale * (sqrt_term + scale) * np.exp(-sqrt_term / scale)
    )


def qs(p, mu=0, scale=1):
    """S-distribution quantile function."""
    p = np.asarray(p)
    return qgnorm(p, mu=mu, scale=scale**2, shape=0.5)


def rs(n, mu=0, scale=1):
    """S-distribution random number generation."""
    return qs(np.random.uniform(0, 1, n), mu, scale)
