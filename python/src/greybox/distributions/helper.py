"""Helper distribution functions."""

from scipy import stats


def dnorm(q, loc=0.0, scale=1.0, log=False):
    """Normal distribution density.

    For the normal distribution, ``loc`` is the mean (μ) and ``scale`` is
    the standard deviation (σ).

    Parameters
    ----------
    q : array_like
        Quantiles.
    loc : float
        Location parameter (mean, μ).
    scale : float
        Scale parameter (standard deviation, σ).
    log : bool
        If True, return log-density.

    Returns
    -------
    array
        Density values.
    """
    if log:
        return stats.norm.logpdf(q, loc=loc, scale=scale)
    return stats.norm.pdf(q, loc=loc, scale=scale)


def plogis(y, loc=0.0, scale=1.0, log=False, lower_tail=True):
    """Logistic distribution CDF.

    Parameters
    ----------
    y : array_like
        Quantiles.
    loc : float
        Location parameter.
    scale : float
        Scale parameter.
    log : bool
        If True, return log-CDF.
    lower_tail : bool
        If True, return lower tail probability.

    Returns
    -------
    array
        CDF values.
    """
    if log:
        if lower_tail:
            return stats.logistic.logcdf(y, loc=loc, scale=scale)
        else:
            return stats.logistic.logsf(y, loc=loc, scale=scale)
    result = stats.logistic.cdf(y, loc=loc, scale=scale)
    if not lower_tail:
        result = 1 - result
    return result


def pnorm(y, loc=0.0, scale=1.0, log=False, lower_tail=True):
    """Normal distribution CDF.

    Parameters
    ----------
    y : array_like
        Quantiles.
    loc : float
        Location parameter (mean).
    scale : float
        Scale parameter (standard deviation).
    log : bool
        If True, return log-CDF.
    lower_tail : bool
        If True, return lower tail probability.

    Returns
    -------
    array
        CDF values.
    """
    if log:
        if lower_tail:
            return stats.norm.logcdf(y, loc=loc, scale=scale)
        else:
            return stats.norm.logsf(y, loc=loc, scale=scale)
    result = stats.norm.cdf(y, loc=loc, scale=scale)
    if not lower_tail:
        result = 1 - result
    return result


def qnorm(p, loc=0.0, scale=1.0):
    """Normal distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    loc : float
        Location parameter (mean).
    scale : float
        Scale parameter (standard deviation).

    Returns
    -------
    array
        Quantile values.
    """
    return stats.norm.ppf(p, loc=loc, scale=scale)
