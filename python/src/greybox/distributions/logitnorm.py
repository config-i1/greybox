"""Logit-Normal distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the distribution that becomes normal after the Logit
transformation.
"""

import numpy as np
from scipy import stats


def dlogitnorm(q, loc=0, scale=1, log=False):
    """Logit-Normal distribution density.

    f(y) = 1/(sqrt(2*pi)*scale*y*(1-y)) * exp(-(logit(y) - loc)^2 / (2*scale^2))

    Parameters
    ----------
    q : array_like
        Quantiles (must be in (0, 1)).
    loc : float
        Location parameter (on logit scale).
    scale : float
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
    density = (
        1
        / (scale * np.sqrt(2 * np.pi) * q * (1 - q))
        * np.exp(-((logit_q - loc) ** 2) / (2 * scale**2))
    )
    density = np.maximum(density, 1e-300)
    if log:
        return np.log(density)
    return density


def plogitnorm(q, loc=0, scale=1):
    """Logit-Normal distribution CDF.

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
    result = np.zeros_like(q, dtype=float)

    mask_negative = q < 0
    mask_between = (q >= 0) & (q <= 1)
    mask_above = q > 1

    result[mask_negative] = 0
    result[mask_above] = 1

    if np.any(mask_between):
        q_between = q[mask_between]
        result[mask_between] = stats.norm.cdf(
            np.log(q_between / (1 - q_between)), loc=loc, scale=scale
        )

    return result


def qlogitnorm(p, loc=0, scale=1):
    """Logit-Normal distribution quantile function.

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
    result = np.exp(stats.norm.ppf(p, loc=loc, scale=scale)) / (
        1 + np.exp(stats.norm.ppf(p, loc=loc, scale=scale))
    )
    result = np.where(np.isnan(result), 0, result)
    return result


def rlogitnorm(n, loc=0, scale=1):
    """Logit-Normal distribution random number generation.

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
    return qlogitnorm(np.random.uniform(0, 1, n), loc, scale)
