"""Smoothers matching R's stats::lowess and stats::supsmu.

Both functions wrap native (pybind11) implementations that reproduce R's
algorithms to machine precision.
"""

import numpy as np

from greybox import _native_lowess as _lowess_native
from greybox import _native_supsmu as _supsmu_native


def lowess(x, y=None, f=2 / 3, iter=3, delta=None):
    """LOWESS smoother. Mirrors R's stats::lowess().

    Parameters
    ----------
    x : array-like
        The x values. May also be a 2D array with two columns (x, y).
    y : array-like, optional
        The y values. Required unless x has two columns.
    f : float, default=2/3
        Smoother span (fraction of points).
    iter : int, default=3
        Number of robustifying iterations.
    delta : float, optional
        Distance threshold for interpolation. If None, uses
        ``0.01 * (max(x) - min(x))``.

    Returns
    -------
    dict
        ``{"x": sorted_x, "y": smoothed_y}``.
    """
    x = np.asarray(x, dtype=np.float64)
    if y is None:
        if x.ndim == 2 and x.shape[1] >= 2:
            y = x[:, 1].copy()
            x = x[:, 0].copy()
        else:
            raise ValueError(
                "If y is not provided, x must be a 2D array with at least 2 columns"
            )
    else:
        x = x.ravel()
        y = np.asarray(y, dtype=np.float64).ravel()

    if len(x) != len(y):
        raise ValueError(
            f"x and y must have the same length, got {len(x)} and {len(y)}"
        )

    n = len(x)
    if n < 2:
        return {"x": x.copy(), "y": y.copy()}

    if delta is None:
        delta = -1.0

    order = np.argsort(x, kind="stable")
    x_sorted = np.ascontiguousarray(x[order])
    y_sorted = np.ascontiguousarray(y[order])

    smoothed = _lowess_native.lowess(x_sorted, y_sorted, f=f, nsteps=iter, delta=delta)
    return {"x": x_sorted, "y": np.asarray(smoothed)}


def supsmu(x, y, wt=None, span="cv", periodic=False, bass=0.0):
    """Friedman's super-smoother. Mirrors R's stats::supsmu().

    Parameters
    ----------
    x : array-like
        Abscissa values; must be unique-sorted or will be sorted internally.
    y : array-like
        Ordinate values.
    wt : array-like, optional
        Per-observation weights. Default is uniform weights.
    span : float or "cv", default="cv"
        Smoother span in (0, 1]; "cv" or 0 selects automatic span selection.
    periodic : bool, default=False
        If True, x is treated as periodic in [0, 1].
    bass : float, default=0.0
        Bass tone control (0 to 10).

    Returns
    -------
    dict
        ``{"x": sorted_x, "y": smoothed_y}``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) != len(y):
        raise ValueError(
            f"x and y must have the same length, got {len(x)} and {len(y)}"
        )

    n = len(x)
    if n < 2:
        return {"x": x.copy(), "y": y.copy()}

    if isinstance(span, str):
        if span.lower() == "cv":
            span_val = 0.0
        else:
            raise ValueError(f"span must be a float or 'cv', got {span!r}")
    else:
        span_val = float(span)

    # Sort by x (R's supsmu requires sorted input).
    order = np.argsort(x, kind="stable")
    x_sorted = np.ascontiguousarray(x[order])
    y_sorted = np.ascontiguousarray(y[order])
    if wt is None:
        wt_sorted = None
    else:
        wt = np.asarray(wt, dtype=np.float64).ravel()
        if len(wt) != n:
            raise ValueError(f"wt must have length {n}, got {len(wt)}")
        wt_sorted = np.ascontiguousarray(wt[order])

    smoothed = _supsmu_native.supsmu(
        x_sorted,
        y_sorted,
        wt=wt_sorted,
        span=span_val,
        bass=bass,
        periodic=bool(periodic),
    )
    return {"x": x_sorted, "y": np.asarray(smoothed)}
