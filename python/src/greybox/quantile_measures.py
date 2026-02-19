"""Quantile and interval scoring measures.

This module provides functions for:
1. Pinball cost function for quantile and expectile forecasts
2. Mean Interval Score (MIS) and related measures

References
----------
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
  prediction, and estimation. Journal of the American Statistical Association,
  102(477), 359-378.
"""

import numpy as np


def pinball(
    holdout: np.ndarray,
    forecast: np.ndarray,
    level: float,
    loss: int = 1,
    na_rm: bool = True,
) -> float:
    """Pinball cost function.

    The pinball function measures the quality of quantile or expectile forecasts.
    It is used in quantile regression and forecast accuracy evaluation.

    For quantiles (loss=1):
        pinball = (1-level) * sum(|e| * I(e <= 0)) + level * sum(|e| * I(e > 0))

    where e = holdout - forecast.

    For expectiles (loss=2):
        Uses squared errors instead of absolute errors.

    Parameters
    ----------
    holdout : np.ndarray
        Actual values.
    forecast : np.ndarray
        Forecasted values (quantile or expectile).
    level : float
        The level associated with the forecast (e.g., 0.5 for median,
        0.95 for 95th percentile).
    loss : int, default=1
        Loss function type:
        - 1: L1 loss (for quantiles)
        - 2: L2 loss (for expectiles)
    na_rm : bool, default=True
        Remove NA values.

    Returns
    -------
    float
        Pinball cost value.

    Examples
    --------
    >>> holdout = np.array([1, 2, 3, 4, 5])
    >>> forecast = np.array([1.1, 2.0, 3.2, 3.9, 5.1])
    >>> pinball(holdout, forecast, level=0.5)  # Median pinball
    >>> pinball(holdout, forecast, level=0.975)  # Upper quantile
    >>> pinball(holdout, forecast, level=0.025)  # Lower quantile

    References
    ----------
    Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
    prediction, and estimation. Journal of the American Statistical Association,
    102(477), 359-378.
    """
    holdout = np.asarray(holdout, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    if len(holdout) != len(forecast):
        raise ValueError("Lengths of holdout and forecast must match")

    if na_rm:
        mask = ~(np.isnan(holdout) | np.isnan(forecast))
        holdout = holdout[mask]
        forecast = forecast[mask]

    errors = holdout - forecast

    if loss == 1:
        below = errors <= 0
        above = errors > 0
        result = (1 - level) * np.sum(np.abs(errors) * below) + level * np.sum(
            np.abs(errors) * above
        )
    elif loss == 2:
        below = errors <= 0
        above = errors > 0
        result = (1 - level) * np.sum((errors**2) * below) + level * np.sum(
            (errors**2) * above
        )
    else:
        raise ValueError("loss must be 1 (L1/quantile) or 2 (L2/expectile)")

    return result


def mis(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    level: float = 0.95,
    na_rm: bool = True,
) -> float:
    """Mean Interval Score (MIS).

    The MIS evaluates the quality of prediction intervals. It rewards
    narrow intervals and penalizes misses (when actual is outside the interval).

    Formula:
        MIS = (upper - lower) + (2/alpha) * |lower - actual| for actual < lower
                                 + (2/alpha) * |actual - upper| for actual > upper
        where alpha = 1 - level

    Parameters
    ----------
    actual : np.ndarray
        Actual (observed) values.
    lower : np.ndarray
        Lower bound of prediction interval.
    upper : np.ndarray
        Upper bound of prediction interval.
    level : float, default=0.95
        Confidence level of the interval (e.g., 0.95 for 95% interval).
    na_rm : bool, default=True
        Remove NA values.

    Returns
    -------
    float
        Mean Interval Score.

    Examples
    --------
    >>> actual = np.array([1, 2, 3, 4, 5])
    >>> lower = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    >>> upper = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    >>> mis(actual, lower, upper, level=0.95)

    References
    ----------
    Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
    prediction, and estimation. Journal of the American Statistical Association,
    102(477), 359-378.
    """
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    if level > 1:
        level = level / 100

    alpha = 1 - level
    lengths = [len(actual), len(lower), len(upper)]

    if len(set(lengths)) > 1:
        raise ValueError("Lengths of actual, lower, and upper must match")

    if na_rm:
        mask = ~(np.isnan(actual) | np.isnan(lower) | np.isnan(upper))
        actual = actual[mask]
        lower = lower[mask]
        upper = upper[mask]

    h = len(actual)
    mis_value = np.sum(upper - lower)

    below_lower = actual < lower
    above_upper = actual > upper

    mis_value += (2 / alpha) * np.sum(np.where(below_lower, lower - actual, 0))
    mis_value += (2 / alpha) * np.sum(np.where(above_upper, actual - upper, 0))

    return mis_value / h


def smis(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    scale: float | np.floating,
    level: float = 0.95,
    na_rm: bool = True,
) -> float:
    """Scaled Mean Interval Score (sMIS).

    The sMIS scales the MIS by a scale parameter.

    Formula:
        sMIS = MIS / scale

    Parameters
    ----------
    actual : np.ndarray
        Actual (observed) values.
    lower : np.ndarray
        Lower bound of prediction interval.
    upper : np.ndarray
        Upper bound of prediction interval.
    scale : float
        Scale parameter.
    level : float, default=0.95
        Confidence level of the interval.
    na_rm : bool, default=True
        Remove NA values.

    Returns
    -------
    float
        Scaled Mean Interval Score.

    Examples
    --------
    >>> actual = np.array([1, 2, 3, 4, 5])
    >>> lower = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    >>> upper = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    >>> smis(actual, lower, upper, scale=3.0)

    References
    ----------
    See Gneiting & Raftery (2007).
    """
    if scale == 0:
        return np.nan

    return mis(actual, lower, upper, level, na_rm) / scale


def rmis(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    benchmark_lower: np.ndarray,
    benchmark_upper: np.ndarray,
    level: float = 0.95,
    na_rm: bool = True,
) -> float:
    """Relative Mean Interval Score (rMIS).

    The rMIS compares the MIS of a forecast to a benchmark forecast.

    Formula:
        rMIS = MIS(forecast) / MIS(benchmark)

    Parameters
    ----------
    actual : np.ndarray
        Actual (observed) values.
    lower : np.ndarray
        Lower bound of prediction interval.
    upper : np.ndarray
        Upper bound of prediction interval.
    benchmark_lower : np.ndarray
        Lower bound of benchmark prediction interval.
    benchmark_upper : np.ndarray
        Upper bound of benchmark prediction interval.
    level : float, default=0.95
        Confidence level of the interval.
    na_rm : bool, default=True
        Remove NA values.

    Returns
    -------
    float
        Relative Mean Interval Score.

    Examples
    --------
    >>> actual = np.array([1, 2, 3, 4, 5])
    >>> lower = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    >>> upper = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    >>> benchmark_lower = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> benchmark_upper = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    >>> rmis(actual, lower, upper, benchmark_lower, benchmark_upper)

    References
    ----------
    See Gneiting & Raftery (2007).
    """
    return mis(actual, lower, upper, level, na_rm) / mis(
        actual, benchmark_lower, benchmark_upper, level, na_rm
    )
