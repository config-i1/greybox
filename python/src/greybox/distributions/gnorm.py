"""Generalized Normal distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the Generalized Normal distribution.
"""

import numpy as np
from scipy import stats
from scipy.special import gamma


def dgnorm(q, mu=0, scale=1, shape=1, log=False):
    """Generalized Normal distribution density."""
    scale = np.atleast_1d(scale)
    shape = np.atleast_1d(shape)
    q = np.atleast_1d(q)

    scale = np.where(np.isnan(scale), 0, scale)
    scale = np.where(scale < 0, 0, scale)
    shape = np.where(np.isnan(shape), 0, shape)
    shape = np.where(shape == 0, 1e-10, shape)

    result = (
        np.exp(-((np.abs(q - mu) / scale) ** shape))
        * shape
        / (2 * scale * gamma(1 / shape))
    )

    if log:
        return np.log(result + 1e-300)
    return result


def pgnorm(q, mu=0, scale=1, shape=1, lower_tail=True, log_p=False):
    """Generalized Normal distribution CDF."""
    scale = np.atleast_1d(scale)
    shape = np.atleast_1d(shape)

    scale = np.where(np.isnan(scale), 0, scale)
    scale = np.where(scale < 0, 0, scale)
    shape = np.where(np.isnan(shape), 0, shape)
    shape = np.where(shape == 0, 1e-10, shape)

    if np.any(shape > 100):
        p = stats.uniform.cdf(q, loc=mu - scale, scale=2 * scale)
    else:
        p = (
            1 / 2
            + np.sign(q - mu)
            * stats.gamma.cdf(
                np.abs(q - mu) ** shape, a=1 / shape, scale=(1 / scale) ** shape
            )
            / 2
        )

    if not lower_tail:
        p = 1 - p

    if log_p:
        p = np.log(p + 1e-300)

    return p


def qgnorm(p, mu=0, scale=1, shape=1, lower_tail=True, log_p=False):
    """Generalized Normal distribution quantile function."""
    p = np.asarray(p)
    scale = np.atleast_1d(scale)
    shape = np.atleast_1d(shape)

    scale = np.where(np.isnan(scale), 0, scale)
    scale = np.where(scale < 0, 0, scale)
    shape = np.where(np.isnan(shape), 0, shape)
    shape = np.where(shape == 0, 1e-10, shape)

    if log_p:
        p = np.exp(p)
    if not lower_tail:
        p = 1 - p

    if np.all(shape > 100):
        result = stats.uniform.ppf(p, loc=mu - scale, scale=2 * scale)
    elif np.any((1 / scale) ** shape < 1e-300):
        lambdaScale = np.ceil(scale) / 10
        lambda_val = (scale / lambdaScale) ** shape
        result = (
            np.sign(p - 0.5)
            * (
                stats.gamma.ppf(np.abs(p - 0.5) * 2, a=1 / shape, scale=lambda_val)
                ** (1 / shape)
            )
            * lambdaScale
            + mu
        )
    else:
        lambda_val = scale**shape
        result = (
            np.sign(p - 0.5)
            * stats.gamma.ppf(np.abs(p - 0.5) * 2, a=1 / shape, scale=lambda_val)
            ** (1 / shape)
            + mu
        )

    return result


def rgnorm(n, mu=0, scale=1, shape=1):
    """Generalized Normal distribution random number generation."""
    return qgnorm(np.random.uniform(0, 1, n), mu=mu, scale=scale, shape=shape)
