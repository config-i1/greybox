"""Exogenous variables manipulation functions.

This module provides functions for transforming and expanding exogenous
variables for use in regression models.
"""

import numpy as np
from typing import Literal


def xreg_transformer(
    xreg: np.ndarray,
    functions: list[str] | None = None,
    silent: bool = True,
) -> np.ndarray:
    """Transform exogenous variables.

    Function transforms each variable in the provided matrix or vector,
    producing non-linear values, depending on the selected pool of functions.

    Parameters
    ----------
    xreg : np.ndarray
        Vector / matrix containing variables that need to be transformed.
    functions : list of str, optional
        Vector of names for functions used. Options: "log", "exp", "inv",
        "sqrt", "square". If None, uses all functions.
    silent : bool, default=True
        If False, then progress is printed. Otherwise the function
        won't print anything.

    Returns
    -------
    np.ndarray
        Matrix with the transformed and the original variables.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> xreg_transformer(x, functions=["log", "sqrt"])
    """
    if functions is None:
        functions = ["log", "exp", "inv", "sqrt", "square"]

    valid_funcs = ["log", "exp", "inv", "sqrt", "square"]
    functions = [f for f in functions if f in valid_funcs]

    if len(functions) == 0:
        raise ValueError(
            "functions parameter does not contain any valid function name. "
            "Please provide something from: log, exp, inv, sqrt, square"
        )

    functions = list(set(functions))

    if not silent:
        print("Preparing matrices...    ")

    xreg = np.asarray(xreg, dtype=float)

    if xreg.ndim == 1:
        xreg = xreg.reshape(-1, 1)

    obs, n_vars = xreg.shape
    n_functions = len(functions)

    xreg_new = np.zeros((obs, (n_functions + 1) * n_vars))

    for j in range(n_vars):
        xreg_new[:, j] = xreg[:, j]
        col_start = n_vars + j * n_functions

        for func in functions:
            func_idx = functions.index(func)
            target_col = col_start + func_idx

            if func == "log":
                xreg_new[:, target_col] = np.log(np.maximum(xreg[:, j], 1e-10))
            elif func == "exp":
                xreg_new[:, target_col] = np.exp(np.minimum(xreg[:, j], 700))
            elif func == "inv":
                xreg_new[:, target_col] = 1.0 / (xreg[:, j] + 1e-10)
            elif func == "sqrt":
                xreg_new[:, target_col] = np.sqrt(np.maximum(xreg[:, j], 0))
            elif func == "square":
                xreg_new[:, target_col] = xreg[:, j] ** 2

    return xreg_new


def xreg_multiplier(xreg: np.ndarray, silent: bool = True) -> np.ndarray:
    """Generate cross-products of exogenous variables.

    Function generates the cross-products of the provided exogenous variables.
    This might be useful when introducing interactions between dummy and
    continuous variables.

    Parameters
    ----------
    xreg : np.ndarray
        Matrix containing variables that need to be expanded. Must have
        at least two columns.
    silent : bool, default=True
        If False, then progress is printed. Otherwise the function
        won't print anything.

    Returns
    -------
    np.ndarray
        Matrix with the cross-products and the original variables.

    Examples
    --------
    >>> x = np.array([[1, 2], [3, 4], [5, 6]])
    >>> xreg_multiplier(x)
    """
    xreg = np.asarray(xreg, dtype=float)

    if xreg.ndim == 1:
        raise ValueError("xreg must be a matrix with at least 2 columns")

    n_obs, n_vars = xreg.shape

    if n_vars < 2:
        raise ValueError("xreg must have at least 2 columns")

    if not silent:
        print("Preparing matrices...    ")

    n_combinations = 0
    for i in range(n_vars):
        for j in range(i, n_vars):
            if i != j:
                n_combinations += 1

    xreg_new = np.zeros((n_obs, n_combinations + n_vars))

    xreg_new[:, :n_vars] = xreg

    k = 0
    for i in range(n_vars):
        for j in range(i, n_vars):
            if i == j:
                continue
            xreg_new[:, n_vars + k] = xreg[:, i] * xreg[:, j]
            k += 1

    return xreg_new


def xreg_expander(
    xreg: np.ndarray,
    lags: list[int] | None = None,
    silent: bool = True,
    gaps: Literal["auto", "NAs", "zero", "naive", "extrapolate"] = "auto",
) -> np.ndarray:
    """Expand exogenous variables with lags and leads.

    Function expands the provided matrix or vector of variables, producing
    values with lags and leads specified by lags parameter.

    Parameters
    ----------
    xreg : np.ndarray
        Vector / matrix containing variables that need to be expanded.
    lags : list of int, optional
        Vector of lags / leads. Negative values mean lags, positive ones
        mean leads. If None, uses -1, 0, 1 (seasonal lags if ts object).
    silent : bool, default=True
        If False, then progress is printed. Otherwise the function
        won't print anything.
    gaps : str, default="auto"
        Defines how to fill in the gaps in the data:
        - "NAs": leave missing values
        - "zero": substitute them by zeroes
        - "naive": use the last / first actual value
        - "extrapolate": use linear extrapolation
        - "auto": let the function select between "extrapolate" and "naive"

    Returns
    -------
    np.ndarray
        Matrix with the expanded variables.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> xreg_expander(x, lags=[-1, 0, 1])
    """
    xreg = np.asarray(xreg, dtype=float)

    if xreg.ndim == 1:
        xreg = xreg.reshape(-1, 1)

    n_obs, n_vars = xreg.shape

    if lags is None:
        lags = [-1, 0, 1]

    lags = [lag for lag in lags if lag != 0]

    if len(lags) == 0:
        return xreg

    lags = list(set(lags))

    if not silent:
        print("Preparing matrices...    ")

    total_cols = n_vars * len(lags)
    xreg_new = np.full((n_obs, total_cols), np.nan)

    n_lags = len(lags)
    for c_idx in range(n_vars):
        for lag_idx, lag in enumerate(lags):
            t_col = c_idx * n_lags + lag_idx

            if lag < 0:
                shift = abs(lag)
                if shift < n_obs:
                    xreg_new[shift:, t_col] = xreg[: n_obs - shift, c_idx]
            else:
                if lag < n_obs:
                    xreg_new[: n_obs - lag, t_col] = xreg[lag:n_obs, c_idx]

    if gaps == "zero":
        xreg_new = np.nan_to_num(xreg_new, nan=0.0)
    elif gaps == "naive":
        for col_idx in range(xreg_new.shape[1]):
            col = xreg_new[:, col_idx]
            first_valid = np.argmax(~np.isnan(col))
            if first_valid > 0:
                col[:first_valid] = col[first_valid]
            last_valid = n_obs - 1 - np.argmax(~np.isnan(col[::-1]))
            if last_valid < n_obs - 1:
                col[last_valid + 1 :] = col[last_valid]
            xreg_new[:, col_idx] = col
    elif gaps in ("auto", "extrapolate"):
        for col_idx in range(xreg_new.shape[1]):
            col = xreg_new[:, col_idx]
            valid_mask = ~np.isnan(col)
            if np.any(valid_mask):
                valid_indices = np.where(valid_mask)[0]
                if valid_indices[0] > 0:
                    first_val = col[valid_indices[0]]
                    col[: valid_indices[0]] = first_val
                if valid_indices[-1] < n_obs - 1:
                    last_val = col[valid_indices[-1]]
                    col[valid_indices[-1] + 1 :] = last_val

    return xreg_new


def temporal_dummy(
    x,
    freq: int | None = None,
    h: int = 0,
) -> np.ndarray:
    """Generate dummy variables for temporal data.

    Function generates dummy variables for months, days, hours, etc.
    based on the provided time index.

    Parameters
    ----------
    x : array-like
        Vector of dates or time indices.
    freq : int, optional
        Frequency of the data. If not provided, attempts to infer.
    h : int, default=0
        Number of future observations to generate dummies for.

    Returns
    -------
    np.ndarray
        Matrix with dummy variables.

    Examples
    --------
    >>> import pandas as pd
    >>> dates = pd.date_range('2020-01-01', periods=12, freq='M')
    >>> temporal_dummy(dates, freq=12)
    """
    x = np.asarray(x)
    n_obs = len(x)

    if freq is None:
        freq = 12

    total_obs = n_obs + h

    if freq == 12:
        dummies = np.zeros((total_obs, 12))
        for i in range(total_obs):
            month = i % 12
            dummies[i, month] = 1
        return dummies
    elif freq == 4:
        dummies = np.zeros((total_obs, 4))
        for i in range(total_obs):
            quarter = i % 4
            dummies[i, quarter] = 1
        return dummies
    elif freq == 7:
        dummies = np.zeros((total_obs, 7))
        for i in range(total_obs):
            day = i % 7
            dummies[i, day] = 1
        return dummies
    elif freq == 24:
        dummies = np.zeros((total_obs, 24))
        for i in range(total_obs):
            hour = i % 24
            dummies[i, hour] = 1
        return dummies
    else:
        dummies = np.zeros((total_obs, freq))
        for i in range(total_obs):
            idx = i % freq
            dummies[i, idx] = 1
        return dummies
