"""Box-Cox Normal distribution functions.

Density, cumulative distribution, quantile functions and random number
generation for the distribution that becomes normal after the Box-Cox
transformation.

References
----------
.. [1] Box, G. E. P., & Cox, D. R. (1964).  An analysis of transformations.
       *Journal of the Royal Statistical Society. Series B (Methodological)*,
       26(2), 211-252.  https://www.jstor.org/stable/2984418
.. [2] Granger, C. W. J., & Newbold, P. (1976).  Forecasting transformed
       series.  *Journal of the Royal Statistical Society. Series B
       (Methodological)*, 38(2), 189-203.
       doi:10.1111/j.2517-6161.1976.tb01585.x
.. [3] Pankratz, A., & Dudley, U. (1987).  Forecasts of power-transformed
       series.  *Journal of Forecasting*, 6(4), 239-248.
       doi:10.1002/for.3980060403
.. [4] Guerrero, V. M. (1993).  Time-series analysis supported by power
       transformations.  *Journal of Forecasting*, 12(1), 37-48.
       doi:10.1002/for.3980120104
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
    elif lambda_bc > 0:
        # BC inverse support is x > -1/lambda (lower-bounded); the
        # underlying Normal can place mass below that boundary which
        # corresponds to (formally) negative y.  The naive
        # (qnorm(p)*lambda+1)**(1/lambda) becomes NaN for such p when
        # 1/lambda is non-integer; the standard convention here is to
        # map that mass to y=0 (the natural lower BC support edge).
        result = np.power(
            stats.norm.ppf(p, loc=loc, scale=scale) * lambda_bc + 1, 1 / lambda_bc
        )
        result = np.where(np.isnan(result), 0, result)
        return result
    else:
        # lambda_bc < 0: BC inverse support is x < -1/lambda
        # (upper-bounded).  The underlying Normal puts non-zero mass
        # *above* -1/lambda which would correspond to y = +infinity
        # (the mass at the upper support boundary).  The unrenormalized
        # BC CDF therefore saturates at
        #   P_valid = norm.cdf(-1/lambda, loc, scale) < 1
        # and the naive quantile is NaN for any p > P_valid (the
        # previous behaviour silently coerced those NaNs to 0, which
        # is dimensionally wrong: the model implies a *large* y, not
        # zero).
        #
        # Renormalize to the truncated distribution on (0, +infinity)
        # by conditioning on the underlying Normal falling within the
        # valid BC inverse support, i.e. invert
        #   F_trunc(y) = pbcnorm(y, ...) / P_valid
        # which IS a proper CDF.  The renormalized quantile is finite
        # for every p in (0, 1) and approaches +infinity only as
        # p -> 1.  See Granger & Newbold (1976), Pankratz & Dudley
        # (1987), Guerrero (1993) for the standard truncated-
        # distribution treatment of BC at lambda < 0.
        p_valid = stats.norm.cdf(-1 / lambda_bc, loc=loc, scale=scale)
        return np.power(
            stats.norm.ppf(p * p_valid, loc=loc, scale=scale) * lambda_bc + 1,
            1 / lambda_bc,
        )


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
