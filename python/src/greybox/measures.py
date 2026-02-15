"""Statistical measures for model evaluation.

This module provides functions for calculating various statistical measures
including correlation, association, and accuracy metrics.
"""

import numpy as np
from scipy import stats
from typing import Literal


def mae(actual: np.ndarray, forecast: np.ndarray, na_rm: bool = True) -> float:
    """Mean Absolute Error.

    Parameters
    ----------
    actual : np.ndarray
        Actual values.
    forecast : np.ndarray
        Forecasted values.
    na_rm : bool, default=True
        Remove NA values.

    Returns
    -------
    float
        Mean Absolute Error.

    Examples
    --------
    >>> actual = np.array([1, 2, 3, 4, 5])
    >>> forecast = np.array([1.1, 2.0, 3.2, 3.9, 5.1])
    >>> mae(actual, forecast)  # doctest: +ELLIPSIS
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    if len(actual) != len(forecast):
        raise ValueError("Lengths of actual and forecast must match")

    if na_rm:
        mask = ~(np.isnan(actual) | np.isnan(forecast))
        actual = actual[mask]
        forecast = forecast[mask]

    return np.mean(np.abs(actual - forecast))


def mse(actual: np.ndarray, forecast: np.ndarray, na_rm: bool = True) -> float:
    """Mean Squared Error.

    Parameters
    ----------
    actual : np.ndarray
        Actual values.
    forecast : np.ndarray
        Forecasted values.
    na_rm : bool, default=True
        Remove NA values.

    Returns
    -------
    float
        Mean Squared Error.

    Examples
    --------
    >>> actual = np.array([1, 2, 3, 4, 5])
    >>> forecast = np.array([1.1, 2.0, 3.2, 3.9, 5.1])
    >>> mse(actual, forecast)
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    if len(actual) != len(forecast):
        raise ValueError("Lengths of actual and forecast must match")

    if na_rm:
        mask = ~(np.isnan(actual) | np.isnan(forecast))
        actual = actual[mask]
        forecast = forecast[mask]

    return np.mean((actual - forecast) ** 2)


def rmse(actual, forecast, na_rm=True):
    """Root Mean Squared Error.

    Parameters
    ----------
    actual : np.ndarray
        Actual values.
    forecast : np.ndarray
        Forecasted values.
    na_rm : bool, default=True
        Remove NA values.

    Returns
    -------
    float
        Root Mean Squared Error.
    """
    return np.sqrt(mse(actual, forecast, na_rm))


def mpe(actual: np.ndarray, forecast: np.ndarray, na_rm: bool = True) -> float:
    """Mean Percentage Error.

    Parameters
    ----------
    actual : np.ndarray
        Actual values.
    forecast : np.ndarray
        Forecasted values.
    na_rm : bool, default=True
        Remove NA values.

    Returns
    -------
    float
        Mean Percentage Error.

    Examples
    --------
    >>> actual = np.array([1, 2, 3, 4, 5])
    >>> forecast = np.array([1.1, 2.0, 3.2, 3.9, 5.1])
    >>> mpe(actual, forecast)
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    if len(actual) != len(forecast):
        raise ValueError("Lengths of actual and forecast must match")

    if na_rm:
        mask = ~(np.isnan(actual) | np.isnan(forecast) | (actual == 0))
        actual = actual[mask]
        forecast = forecast[mask]

    return np.mean((actual - forecast) / actual * 100)


def mape(actual: np.ndarray, forecast: np.ndarray, na_rm: bool = True) -> float:
    """Mean Absolute Percentage Error.

    Parameters
    ----------
    actual : np.ndarray
        Actual values.
    forecast : np.ndarray
        Forecasted values.
    na_rm : bool, default=True
        Remove NA values.

    Returns
    -------
    float
        Mean Absolute Percentage Error.

    Examples
    --------
    >>> actual = np.array([1, 2, 3, 4, 5])
    >>> forecast = np.array([1.1, 2.0, 3.2, 3.9, 5.1])
    >>> mape(actual, forecast)
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    if len(actual) != len(forecast):
        raise ValueError("Lengths of actual and forecast must match")

    if na_rm:
        mask = ~(np.isnan(actual) | np.isnan(forecast) | (actual == 0))
        actual = actual[mask]
        forecast = forecast[mask]

    return np.mean(np.abs((actual - forecast) / actual) * 100)


def mase(
    actual: np.ndarray,
    forecast: np.ndarray,
    scale: float | None = None,
    na_rm: bool = True,
) -> float:
    """Mean Absolute Scaled Error.

    Parameters
    ----------
    actual : np.ndarray
        Actual values.
    forecast : np.ndarray
        Forecasted values.
    scale : float, optional
        Scale parameter. If None, uses mean absolute deviation.
    na_rm : bool, default=True
        Remove NA values.

    Returns
    -------
    float
        Mean Absolute Scaled Error.

    Examples
    --------
    >>> actual = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> forecast = np.array([1.1, 2.0, 3.2, 3.9, 5.1,
    ...                      6.0, 7.1, 8.0, 9.2, 10.1])
    >>> mase(actual, forecast)
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    if len(actual) != len(forecast):
        raise ValueError("Lengths of actual and forecast must match")

    if na_rm:
        mask = ~(np.isnan(actual) | np.isnan(forecast))
        actual = actual[mask]
        forecast = forecast[mask]

    if scale is None:
        scale = np.mean(np.abs(np.diff(actual)))

    if scale == 0:
        return np.nan

    return np.mean(np.abs(actual - forecast)) / scale


def accuracy(
    actual: np.ndarray,
    forecast: np.ndarray,
    na_rm: bool = True,
) -> dict:
    """Calculate multiple accuracy measures at once.

    Parameters
    ----------
    actual : np.ndarray
        Actual values.
    forecast : np.ndarray
        Forecasted values.
    na_rm : bool, default=True
        Remove NA values.

    Returns
    -------
    dict
        Dictionary containing: ME, MAE, MSE, RMSE, MPE, MAPE, MASE.

    Examples
    --------
    >>> actual = np.array([1, 2, 3, 4, 5])
    >>> forecast = np.array([1.1, 2.0, 3.2, 3.9, 5.1])
    >>> accuracy(actual, forecast)
    """
    return {
        "ME": np.mean(actual - forecast),
        "MAE": mae(actual, forecast, na_rm),
        "MSE": mse(actual, forecast, na_rm),
        "RMSE": rmse(actual, forecast, na_rm),
        "MPE": mpe(actual, forecast, na_rm),
        "MAPE": mape(actual, forecast, na_rm),
        "MASE": mase(actual, forecast, na_rm=na_rm),
    }


def pcor(
    x: np.ndarray,
    y: np.ndarray | None = None,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
) -> dict:
    """Calculate partial correlations.

    Function calculates partial correlations between the provided variables.
    The calculation is done based on multiple linear regressions.

    Parameters
    ----------
    x : np.ndarray
        DataFrame or matrix with numeric values.
    y : np.ndarray, optional
        The numerical variable. If provided, calculates partial correlation
        between each column of x and y.
    method : {"pearson", "spearman", "kendall"}, default="pearson"
        Which method to use for calculation.

    Returns
    -------
    dict
        Dictionary containing:
        - value: matrix of partial correlation coefficients
        - p.value: p-values for the coefficients
        - method: method used

    Examples
    --------
    >>> x = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12], [5, 10, 15]]
    ... )
    >>> result = pcor(x)
    """
    x = np.asarray(x, dtype=float)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if y is not None:
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        x = np.column_stack([x, y])

    n_obs, n_vars = x.shape

    if n_vars < 2:
        raise ValueError("Need at least 2 variables")

    mask = ~np.isnan(np.all(x, axis=1))
    x_clean = x[mask]

    if len(x_clean) < 3:
        raise ValueError("Not enough observations")

    cor_matrix = np.corrcoef(x_clean, rowvar=False)

    try:
        cor_inv = np.linalg.inv(cor_matrix)
    except np.linalg.LinAlgError:
        cor_inv = np.linalg.pinv(cor_matrix)

    pcor_matrix = np.zeros_like(cor_matrix)
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                pcor_matrix[i, j] = 1.0
            else:
                pcor_matrix[i, j] = -cor_inv[i, j] / np.sqrt(
                    cor_inv[i, i] * cor_inv[j, j]
                )

    df = len(x_clean) - n_vars
    t_stat = pcor_matrix * np.sqrt(df / (1 - pcor_matrix**2))
    t_stat = np.where(np.eye(n_vars, dtype=bool), 0, t_stat)
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df))
    p_value = np.where(np.eye(n_vars, dtype=bool), np.nan, p_value)

    return {"value": pcor_matrix, "p.value": p_value, "method": method}


def mcor(x: np.ndarray, y: np.ndarray | None = None) -> float:
    """Calculate multiple correlation coefficient.

    Function returns the multiple correlation coefficient between a set
    of variables and a dependent variable.

    Parameters
    ----------
    x : np.ndarray
        Matrix of independent variables.
    y : np.ndarray
        The dependent variable.

    Returns
    -------
    float
        Multiple correlation coefficient.

    Examples
    --------
    >>> x = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
    >>> y = np.array([3, 6, 9, 12, 15])
    >>> mcor(x, y)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    mask = ~(np.isnan(np.any(x, axis=1)) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(y_clean) < 3:
        raise ValueError("Not enough observations")

    x_with_intercept = np.column_stack([np.ones(len(x_clean)), x_clean])

    try:
        coeffs = np.linalg.lstsq(x_with_intercept, y_clean, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.nan

    y_pred = x_with_intercept @ coeffs

    ss_res = np.sum((y_clean - y_pred) ** 2)
    ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)

    if ss_tot == 0:
        return np.nan

    r_squared = 1 - ss_res / ss_tot
    return np.sqrt(r_squared)


def determination(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate coefficient of determination (R-squared).

    Parameters
    ----------
    actual : np.ndarray
        Actual values.
    predicted : np.ndarray
        Predicted values.

    Returns
    -------
    dict
        Dictionary containing R-squared and adjusted R-squared.

    Examples
    --------
    >>> actual = np.array([1, 2, 3, 4, 5])
    >>> predicted = np.array([1.1, 2.0, 3.2, 3.9, 5.1])
    >>> determination(actual, predicted)
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) != len(predicted):
        raise ValueError("Lengths of actual and predicted must match")

    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)

    if ss_tot == 0:
        r_squared = np.nan
    else:
        r_squared = 1 - ss_res / ss_tot

    n = len(actual)
    k = 1
    if n - k - 1 > 0:
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
    else:
        adj_r_squared = np.nan

    return {"r2": r_squared, "adjR2": adj_r_squared}


def association(
    x: np.ndarray,
    y: np.ndarray | None = None,
    method: str = "auto",
) -> dict:
    """Calculate measures of association.

    Function returns the matrix of measures of association for different types
    of variables.

    Parameters
    ----------
    x : np.ndarray
        DataFrame or matrix.
    y : np.ndarray, optional
        The numerical variable.
    method : str, default="auto"
        Method to use: "auto", "pearson", "spearman", "kendall", "cramer".
        "auto" selects based on variable types.

    Returns
    -------
    dict
        Dictionary containing:
        - value: matrix of association coefficients
        - p.value: p-values
        - type: matrix of types of measures used

    Examples
    --------
    >>> x = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    >>> result = association(x)
    """
    x = np.asarray(x, dtype=float)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if y is not None:
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        x = np.column_stack([x, y])

    n_obs, n_vars = x.shape

    if n_vars < 2:
        raise ValueError("Need at least 2 variables")

    mask = ~np.isnan(np.all(x, axis=1))
    x_clean = x[mask]

    if method == "auto":
        cor_matrix = np.corrcoef(x_clean, rowvar=False)
    else:
        if method == "pearson":
            cor_matrix = np.corrcoef(x_clean, rowvar=False)
        elif method == "spearman":
            cor_matrix = stats.spearmanr(x_clean).correlation
        elif method == "kendall":
            cor_matrix = stats.kendalltau(x_clean).correlation
        else:
            raise ValueError(f"Unknown method: {method}")

    df = len(x_clean) - 2
    t_stat = cor_matrix * np.sqrt(df / (1 - cor_matrix**2 + 1e-10))
    t_stat = np.where(np.eye(n_vars, dtype=bool), 0, t_stat)
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df))
    p_value = np.where(np.eye(n_vars, dtype=bool), np.nan, p_value)

    type_matrix = np.full((n_vars, n_vars), "pearson")

    return {"value": cor_matrix, "p.value": p_value, "type": type_matrix}
