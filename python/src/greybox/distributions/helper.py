"""Helper distribution functions."""

import numpy as np
from scipy import stats


def dnorm(q, mean=0.0, sd=1.0, log=False):
    """Normal distribution density.

    Parameters
    ----------
    q : array_like
        Quantiles.
    mean : float
        Mean.
    sd : float
        Standard deviation.
    log : bool
        If True, return log-density.

    Returns
    -------
    array
        Density values.
    """
    result = stats.norm.pdf(q, loc=mean, scale=sd)
    if log:
        return np.log(result + 1e-300)
    return result


def plogis(y, location=0.0, scale=1.0, log_p=False):
    """Logistic distribution CDF.

    Parameters
    ----------
    y : array_like
        Quantiles.
    location : float
        Location parameter.
    scale : float
        Scale parameter.
    log_p : bool
        If True, return log-CDF.

    Returns
    -------
    array
        CDF values.
    """
    return stats.logistic.cdf(y, loc=location, scale=scale)


def pnorm(y, mean=0.0, sd=1.0, log_p=False, lower_tail=True):
    """Normal distribution CDF.

    Parameters
    ----------
    y : array_like
        Quantiles.
    mean : float
        Mean.
    sd : float
        Standard deviation.
    log_p : bool
        If True, return log-CDF.
    lower_tail : bool
        If True, return lower tail probability.

    Returns
    -------
    array
        CDF values.
    """
    result = stats.norm.cdf(y, loc=mean, scale=sd)
    if not lower_tail:
        result = 1 - result
    if log_p:
        result = np.log(result + 1e-300)
    return result


def qnorm(p, mean=0.0, sd=1.0):
    """Normal distribution quantile function.

    Parameters
    ----------
    p : array_like
        Probabilities.
    mean : float
        Mean.
    sd : float
        Standard deviation.

    Returns
    -------
    array
        Quantile values.
    """
    return stats.norm.ppf(p, loc=mean, scale=sd)
