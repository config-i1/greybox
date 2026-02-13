"""Transform functions for greybox."""

import numpy as np


def bc_transform(y: np.ndarray, lambda_bc: float) -> np.ndarray:
    """Box-Cox transformation."""
    if lambda_bc == 0:
        return np.log(y)
    else:
        return (np.power(y, lambda_bc) - 1) / lambda_bc


def bc_transform_inv(y: np.ndarray, lambda_bc: float) -> np.ndarray:
    """Inverse Box-Cox transformation."""
    if lambda_bc == 0:
        return np.exp(y)
    else:
        return np.power(y * lambda_bc + 1, 1 / lambda_bc)


def mean_fast(
    x: np.ndarray,
    df: int | None = None,
    trim: float = 0.0,
    side: str = "lower"
) -> float:
    """Fast mean calculation with optional trimming."""
    if df is None:
        df = len(x)

    if trim != 0:
        if side == "both":
            n = len(x)
            n_trim = int(n * trim)
            if n_trim > 0:
                x_sorted = np.sort(x)
                return np.mean(x_sorted[n_trim:-n_trim])
            return np.mean(x)
        else:
            x_sorted = np.sort(x)
            n_trimmed = int(np.floor(len(x) * trim))
            if side == "lower":
                return np.mean(x_sorted[n_trimmed:])
            else:
                return np.mean(x_sorted[:-n_trimmed])
    else:
        return np.sum(x) / df
