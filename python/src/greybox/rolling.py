"""Rolling origin evaluation.

This module provides functions for rolling origin cross-validation
for time series models.
"""

import numpy as np
from typing import Callable, Any


class RollingOriginResult:
    """Result of rolling origin evaluation.

    Attributes
    ----------
    actuals : np.ndarray
        The original data.
    holdout : np.ndarray
        Matrix of actual values corresponding to forecasts from each origin.
    forecasts : dict
        Dictionary of forecast matrices/arrays from each origin.
    origins : int
        Number of rolling origins.
    h : int
        Forecasting horizon.
    """

    def __init__(
        self,
        actuals: np.ndarray,
        holdout: np.ndarray,
        forecasts: dict,
        origins: int,
        h: int,
    ):
        self.actuals = actuals
        self.holdout = holdout
        self.forecasts = forecasts
        self.origins = origins
        self.h = h


def rolling_origin(
    data: np.ndarray,
    h: int = 1,
    origins: int = 5,
    step: int = 1,
    ci: bool = False,
    co: bool = False,
    model_fn: Callable[[np.ndarray], Any] = None,
    predict_fn: Callable[[Any, int], np.ndarray] = None,
    silent: bool = True,
) -> RollingOriginResult:
    """Rolling Origin Evaluation.

    The function produces rolling origin forecasts using the data and a
    model fitting function. This is useful for cross-validation with
    time series data.

    Parameters
    ----------
    data : np.ndarray
        Time series data.
    h : int, default=1
        Forecasting horizon (number of steps ahead to forecast).
    origins : int, default=5
        Number of rolling origins.
    step : int, default=1
        Number of observations to skip between origins.
    ci : bool, default=False
        If True, use constant in-sample window (sliding window).
        If False, use expanding window.
    co : bool, default=False
        If True, use constant holdout window.
        If False, rolling origin stops when less than h observations are left.
    model_fn : callable
        Function that takes training data and returns a fitted model.
        Example: lambda x: ALM().fit(x, y)
    predict_fn : callable
        Function that takes fitted model and horizon h and returns forecasts.
        Example: lambda model, h: model.predict(h)
    silent : bool, default=True
        If False, print progress.

    Returns
    -------
    RollingOriginResult
        Object containing:
        - actuals: original data
        - holdout: matrix of actual holdout values
        - forecasts: dictionary of forecast matrices

    Examples
    --------
    >>> import numpy as np
    >>> from greybox.alm import ALM
    >>> from greybox.formula import formula
    >>> np.random.seed(42)
    >>> y = np.cumsum(np.random.randn(50))
    >>> data = {'y': y, 'trend': range(1, 51)}
    >>> def model_fn(train_data):
    ...     y_train, X_train = formula("y ~ trend", train_data)
    ...     model = ALM(distribution="dnorm")
    ...     return model.fit(X_train, y_train)
    >>> def predict_fn(model, h):
    ...     return model.predict(model._X_train_,
    ...                          interval="none").mean
    >>> result = rolling_origin(data, h=5, origins=3,
    ...                        model_fn=model_fn, predict_fn=predict_fn)
    """
    if model_fn is None or predict_fn is None:
        raise ValueError("model_fn and predict_fn must be provided")

    data = np.asarray(data, dtype=float)
    n = len(data)

    if co:
        max_origins = (n - h) // step
        origins = min(origins, max_origins)

    forecasts_dict = {}
    holdout_matrix = np.zeros((origins, h))

    window_size = n - h - (origins - 1) * step
    origin_idx = 0
    for i in range(0, origins * step, step):
        if ci:
            train_end = window_size + i
        else:
            train_end = n - h - (origins - 1 - origin_idx) * step

        if train_end < 2:
            break

        train_data = data[:train_end]
        actual_holdout = data[train_end : train_end + h]

        try:
            model = model_fn(train_data)

            forecasts = predict_fn(model, h)

            if len(forecasts) < h:
                pad_len = h - len(forecasts)
                forecasts = np.pad(forecasts, (0, pad_len), mode="constant")

            forecasts_dict[f"origin_{origin_idx}"] = forecasts
            holdout_matrix[origin_idx, : len(actual_holdout)] = actual_holdout

            if not silent:
                print(
                    f"Origin {origin_idx + 1}/{origins}: train={train_end}, "
                    f"holdout={len(actual_holdout)}"
                )

        except Exception as e:
            if not silent:
                print(f"Origin {origin_idx + 1} failed: {e}")
            break

        origin_idx += 1

    actuals = data

    return RollingOriginResult(
        actuals=actuals,
        holdout=holdout_matrix[:origin_idx],
        forecasts=forecasts_dict,
        origins=origin_idx,
        h=h,
    )
