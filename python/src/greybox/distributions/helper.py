"""Helper distribution functions."""

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
    if log:
        return stats.norm.logpdf(q, loc=mean, scale=sd)
    return stats.norm.pdf(q, loc=mean, scale=sd)


def plogis(y, location=0.0, scale=1.0, log_p=False, lower_tail=True):
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
    lower_tail : bool
        If True, return lower tail probability.

    Returns
    -------
    array
        CDF values.
    """
    if log_p:
        if lower_tail:
            return stats.logistic.logcdf(y, loc=location, scale=scale)
        else:
            return stats.logistic.logsf(y, loc=location, scale=scale)
    result = stats.logistic.cdf(y, loc=location, scale=scale)
    if not lower_tail:
        result = 1 - result
    return result


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
    if log_p:
        if lower_tail:
            return stats.norm.logcdf(y, loc=mean, scale=sd)
        else:
            return stats.norm.logsf(y, loc=mean, scale=sd)
    result = stats.norm.cdf(y, loc=mean, scale=sd)
    if not lower_tail:
        result = 1 - result
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
