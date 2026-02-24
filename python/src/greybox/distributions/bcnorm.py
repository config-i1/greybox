"""Box-Cox Normal distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the distribution that becomes normal after the Box-Cox
transformation.
"""

import numpy as np
from scipy import stats


def dbcnorm(q, loc=0, scale=1, lambda_bc=0, log=False):
    """Box-Cox Normal distribution density.

    f(y) = y^(lambda-1) * 1/sqrt(2*pi) * exp(
        -((y^lambda-1)/lambda - loc)^2 / (2*scale^2))

    Parameters
    ----------
    q : array_like
        Quantiles (must be non-negative).
    loc : float
        Location parameter (on transformed scale).
    scale : float
        Scale parameter.
    lambda_bc : float
        Box-Cox transformation parameter.
    log : bool
        If True, return log-density.

    Returns
    -------
    array
        Density values.
    """
    q = np.asarray(q)
    if lambda_bc == 0:
        density = stats.lognorm.pdf(q, s=scale, scale=np.exp(loc))
    elif lambda_bc == 1:
        density = stats.norm.pdf(q, loc=loc + 1, scale=scale)
    else:
        jacobian = np.power(q, lambda_bc - 1)
        y_transformed = (np.power(q, lambda_bc) - 1) / lambda_bc
        density = jacobian * stats.norm.pdf(y_transformed, loc=loc, scale=scale)
        density = np.where(q <= 0, 0, density)
    density = np.maximum(density, 1e-300)
    if log:
        return np.log(density)
    return density


def pbcnorm(q, loc=0, scale=1, lambda_bc=0):
    """Box-Cox Normal distribution CDF.

    Parameters
    ----------
    q : array_like
        Quantiles.
    loc : float
        Location parameter.
    scale : float
        Scale parameter.
    lambda_bc : float
        Box-Cox transformation parameter.

    Returns
    -------
    array
        CDF values.
    """
    q = np.asarray(q)
    if lambda_bc == 0:
        return stats.lognorm.cdf(q, s=scale, scale=np.exp(loc))
    else:
        result = np.zeros_like(q, dtype=float)
        mask = q > 0
        result[mask] = stats.norm.cdf(
            (np.power(q[mask], lambda_bc) - 1) / lambda_bc, loc=loc, scale=scale
        )
        result[q <= 0] = 0
        return result


def qbcnorm(p, loc=0, scale=1, lambda_bc=0):
    """Box-Cox Normal distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    loc : float
        Location parameter.
    scale : float
        Scale parameter.
    lambda_bc : float
        Box-Cox transformation parameter.

    Returns
    -------
    array
        Quantile values.
    """
    p = np.asarray(p)
    if lambda_bc == 0:
        return stats.lognorm.ppf(p, s=scale, scale=np.exp(loc))
    else:
        result = np.power(
            stats.norm.ppf(p, loc=loc, scale=scale) * lambda_bc + 1, 1 / lambda_bc
        )
        result = np.where(np.isnan(result), 0, result)
        return result


def rbcnorm(n, loc=0, scale=1, lambda_bc=0):
    """Box-Cox Normal distribution random number generation.

    Parameters
    ----------
    n : int
        Number of observations.
    loc : float
        Location parameter.
    scale : float
        Scale parameter.
    lambda_bc : float
        Box-Cox transformation parameter.

    Returns
    -------
    array
        Random values.
    """
    return qbcnorm(np.random.uniform(0, 1, n), loc, scale, lambda_bc)
