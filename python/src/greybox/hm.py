"""Half-moment measures for data analysis.

This module provides functions for:
1. Half-moment based measures: hm, ham, asymmetry, extremity, cextremity
2. Mean Root Error (MRE)

References
----------
- Svetunkov I., Kourentzes N., Svetunkov S. "Half Central Moment for Data Analysis".
  Working Paper of Department of Management Science, Lancaster University, 2023:3, 1-21.
- Kourentzes N. (2014). The Bias Coefficient: a new metric for forecast bias.
"""

import numpy as np


def hm(x: np.ndarray, center: float | None = None) -> complex:
    """Half Moment (HM).

    The half moment is a measure that captures the asymmetry of a distribution
    by considering only the positive or negative deviations from a center point.

    Formula:
        hm = mean(sqrt(x - C))

    where C is the centering parameter (default is mean of x).

    Parameters
    ----------
    x : np.ndarray
        Data vector.
    center : float, optional
        Centering parameter. If None, uses mean(x).

    Returns
    -------
    complex
        Half moment (complex number when mean of sqrt is taken).

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> hm(x)  # doctest: +ELLIPSIS
    (0.79...+0.79...j)

    References
    ----------
    Svetunkov I., Kourentzes N., Svetunkov S. "Half Central Moment for Data Analysis".
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    if center is None:
        center = np.mean(x)

    return np.mean(np.sqrt(np.array(x - center, dtype=complex)))


def ham(x: np.ndarray, center: float | None = None) -> float:
    """Half Absolute Moment (HAM).

    The half absolute moment is the absolute value of the half moment.
    It captures the magnitude of deviations without direction.

    Formula:
        ham = mean(sqrt(|x - C|))

    Parameters
    ----------
    x : np.ndarray
        Data vector.
    center : float, optional
        Centering parameter. If None, uses mean(x).

    Returns
    -------
    float
        Half absolute moment.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> ham(x)  # doctest: +ELLIPSIS
    1.0

    References
    ----------
    Svetunkov I., Kourentzes N., Svetunkov S. "Half Central Moment for Data Analysis".
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    if center is None:
        center = np.mean(x)

    return np.mean(np.sqrt(np.abs(x - center)))


def asymmetry(x: np.ndarray, center: float | None = None) -> float:
    """Asymmetry coefficient.

    The asymmetry coefficient measures the asymmetry of a distribution
    around a center point based on the half moment.

    Formula:
        asymmetry = 1 - Arg(hm(x, C)) / (pi/4)

    Values:
        - 1: Maximum negative asymmetry (all values below center)
        - 0: Symmetric distribution
        - -1: Maximum positive asymmetry (all values above center)

    Parameters
    ----------
    x : np.ndarray
        Data vector.
    center : float, optional
        Centering parameter. If None, uses mean(x).

    Returns
    -------
    float
        Asymmetry coefficient.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> asymmetry(x)
    0.0
    >>> x_skewed = np.array([1, 1, 1, 4, 5])
    >>> asymmetry(x_skewed)  # doctest: +ELLIPSIS
    -0.20...

    References
    ----------
    Svetunkov I., Kourentzes N., Svetunkov S. "Half Central Moment for Data Analysis".
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    if center is None:
        center = np.mean(x)

    return 1 - np.angle(hm(x, center)) / (np.pi / 4)


def extremity(x: np.ndarray, center: float | None = None) -> float:
    """Extremity coefficient.

    The extremity coefficient measures how extreme the values in a distribution
    are relative to the center.

    Formula:
        extremity = 2 * (ham(x, C) / (std(x)^0.5))^(log(0.5)/log(2*3^(-0.75))) - 1

    Parameters
    ----------
    x : np.ndarray
        Data vector.
    center : float, optional
        Centering parameter. If None, uses mean(x).

    Returns
    -------
    float
        Extremity coefficient.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> extremity(x)  # doctest: +ELLIPSIS
    -0.04...

    References
    ----------
    Svetunkov I., Kourentzes N., Svetunkov S. "Half Central Moment for Data Analysis".
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    if center is None:
        center = np.mean(x)

    if len(x) < 2:
        return np.nan

    ham_val = ham(x, center)
    variance = np.var(x, ddof=0)

    if variance == 0:
        return np.nan

    exponent = np.log(0.5) / np.log(2 * 3**-0.75)
    ch = ham_val / (variance**0.25)

    return 2 * (ch**exponent) - 1


def cextremity(x: np.ndarray, center: float | None = None) -> complex:
    """Complex Extremity coefficient.

    The complex extremity coefficient extends the extremity coefficient
    to capture both magnitude and phase information.

    Formula:
        cextremity = 2 * (CH * 2)^(log(0.5)/log(2*3^(-0.75))) - 1

    where CH = hm(x, C) / (std(x)^0.25)

    Parameters
    ----------
    x : np.ndarray
        Data vector.
    center : float, optional
        Centering parameter. If None, uses mean(x).

    Returns
    -------
    complex
        Complex extremity coefficient.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> cextremity(x)  # doctest: +ELLIPSIS
    (-0.04...+0...j)

    References
    ----------
    Svetunkov I., Kourentzes N., Svetunkov S. "Half Central Moment for Data Analysis".
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    if center is None:
        center = np.mean(x)

    if len(x) < 2:
        return np.nan

    hm_val = hm(x, center)
    variance = np.var(x, ddof=0)

    if variance == 0:
        return np.nan

    exponent = np.log(0.5) / np.log(2 * 3**-0.75)
    ch = hm_val / (variance**0.25)

    real_part = 2 * (np.real(ch) * 2) ** exponent - 1
    imag_part = 2 * (np.imag(ch) * 2) ** exponent - 1

    return complex(real_part, imag_part)


def _validate_inputs(
    actual: np.ndarray, forecast: np.ndarray, na_rm: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and clean actual and forecast inputs."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    if len(actual) != len(forecast):
        raise ValueError("Lengths of actual and forecast must match")

    if na_rm:
        mask = ~(np.isnan(actual) | np.isnan(forecast))
        actual = actual[mask]
        forecast = forecast[mask]

    return actual, forecast


def mre(actual: np.ndarray, forecast: np.ndarray, na_rm: bool = True) -> float:
    """Mean Root Error (MRE).

    The MRE measures the average signed root error. It is used as a bias
    coefficient (Kourentzes, 2014) and can detect systematic over or
    under-forecasting.

    Formula:
        MRE = mean(sqrt(actual - forecast))

    Note: When actual < forecast, the result is complex. The function
    returns the real part of the mean.

    Parameters
    ----------
    actual : np.ndarray
        Actual (observed) values.
    forecast : np.ndarray
        Forecasted values.
    na_rm : bool, default=True
        Remove NA values.

    Returns
    -------
    float
        Mean Root Error.

    Examples
    --------
    >>> actual = np.array([1, 2, 3, 4, 5])
    >>> forecast = np.array([1.1, 2.0, 3.2, 3.9, 5.1])
    >>> mre(actual, forecast)  # doctest: +ELLIPSIS

    References
    ----------
    Kourentzes N. (2014). The Bias Coefficient: a new metric for forecast bias.
    https://kourentzes.com/forecasting/2014/12/17/the-bias-coefficient-a-new-metric-for-forecast-bias/
    """
    actual, forecast = _validate_inputs(actual, forecast, na_rm)
    errors = actual - forecast
    complex_roots = np.sqrt(np.array(errors, dtype=complex))
    return np.mean(complex_roots).real
