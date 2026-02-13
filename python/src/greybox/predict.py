"""Prediction functions for greybox models.

This module provides prediction functionality similar to R's predict.greybox
and predict.alm methods.
"""

import numpy as np
from dataclasses import dataclass
from scipy import stats


@dataclass
class PredictResult:
    """Prediction result object.

    Attributes
    ----------
    mean : np.ndarray
        Predicted values (point forecasts).
    lower : np.ndarray or None
        Lower prediction bounds.
    upper : np.ndarray or None
        Upper prediction bounds.
    level : list
        Confidence levels used.
    variances : np.ndarray or None
        Variance values.
    newdata_provided : bool
        Whether new data was provided.
    distribution : str
        Distribution used in the model.
    scale : np.ndarray or None
        Scale values (for occurrence models).
    occurrence : PredictResult or None
        Occurrence model predictions (if applicable).
    """

    mean: np.ndarray
    lower: np.ndarray | None
    upper: np.ndarray | None
    level: list
    variances: np.ndarray | None
    newdata_provided: bool
    distribution: str
    scale: np.ndarray | None = None
    occurrence: "PredictResult | None" = None


def predict_basic(model, X, interval="none", level=0.95, side="both"):
    """Basic prediction function (similar to R's predict.greybox).

    Parameters
    ----------
    model : ALM
        Fitted ALM model.
    X : np.ndarray
        Design matrix for predictions.
    interval : str, optional
        Type of interval: "none", "confidence", or "prediction".
    level : float, optional
        Confidence level (default 0.95 for 95%).
    side : str, optional
        Side of interval: "both", "upper", or "lower".

    Returns
    -------
    PredictResult
        Prediction results with mean, lower, upper bounds, etc.
    """
    if not hasattr(model, "coef_") or model.coef_ is None:
        raise ValueError("Model not fitted. Call fit() first.")

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    if X.shape[1] != len(model.coef_) + 1:
        raise ValueError(
            f"X has {X.shape[1]} columns but model expects {len(model.coef_) + 1}"
        )

    if interval not in ("none", "confidence", "prediction"):
        raise ValueError("interval must be 'none', 'confidence', or 'prediction'")

    if side not in ("both", "upper", "lower"):
        raise ValueError("side must be 'both', 'upper', or 'lower'")

    if isinstance(level, (int, float)):
        level = [level]

    n_levels = len(level)
    n_obs = X.shape[0]

    if side == "upper":
        level_low = [0.0] * n_levels
        level_up = level
    elif side == "lower":
        level_low = [1 - lev for lev in level]
        level_up = [1.0] * n_levels
    else:
        level_low = [(1 - lev) / 2 for lev in level]
        level_up = [(1 + lev) / 2 for lev in level]

    param_quantiles = stats.t.ppf(level_low + level_up, df=model.df_residual_)

    coef_array = np.concatenate([[model.intercept_], model.coef_])
    mean = X @ coef_array

    variances = None
    lower = None
    upper = None

    if interval != "none":
        variances = np.abs(np.diag(X @ X.T)) * (model.scale_**2)

        if interval == "confidence":
            se = np.sqrt(variances)
            for i in range(n_levels):
                lower_i = mean + param_quantiles[i] * se
                upper_i = mean + param_quantiles[i + n_levels] * se
                if lower is None:
                    lower = np.zeros((n_obs, n_levels))
                    upper = np.zeros((n_obs, n_levels))
                lower[:, i] = lower_i
                upper[:, i] = upper_i

        elif interval == "prediction":
            sigma_sq = model.scale_**2
            total_var = variances + sigma_sq
            se = np.sqrt(total_var)
            for i in range(n_levels):
                lower_i = mean + param_quantiles[i] * se
                upper_i = mean + param_quantiles[i + n_levels] * se
                if lower is None:
                    lower = np.zeros((n_obs, n_levels))
                    upper = np.zeros((n_obs, n_levels))
                lower[:, i] = lower_i
                upper[:, i] = upper_i

    return PredictResult(
        mean=mean,
        lower=lower,
        upper=upper,
        level=level_low + level_up,
        variances=variances,
        newdata_provided=False,
        distribution=model.distribution,
    )


def predict(
    model,
    newdata=None,
    interval="none",
    level=0.95,
    side="both",
    occurrence=None,
):
    """Prediction function for ALM models (similar to R's predict.alm).

    Parameters
    ----------
    model : ALM
        Fitted ALM model.
    newdata : dict, DataFrame, or np.ndarray, optional
        New data for predictions. If None, uses training data.
    interval : str, optional
        Type of interval: "none", "confidence", or "prediction".
    level : float or list, optional
        Confidence level(s) (default 0.95 for 95%).
    side : str, optional
        Side of interval: "both", "upper", or "lower".
    occurrence : np.ndarray, optional
        Occurrence values for zero-inflated models.

    Returns
    -------
    PredictResult
        Prediction results with mean, lower, upper bounds, etc.
    """
    if not hasattr(model, "coef_") or model.coef_ is None:
        raise ValueError("Model not fitted. Call fit() first.")

    if interval not in ("none", "confidence", "prediction"):
        raise ValueError("interval must be 'none', 'confidence', or 'prediction'")

    if side not in ("both", "upper", "lower"):
        raise ValueError("side must be 'both', 'upper', or 'lower'")

    if isinstance(level, (int, float)):
        level = [level]

    n_levels = len(level)

    if newdata is None:
        X = model._X_train_
        newdata_provided = False
    else:
        newdata_provided = True
        from .formula import formula

        if isinstance(newdata, dict):
            if "y" in newdata:
                _, X = formula("y ~ .", newdata)
            else:
                X = formula("~ .", newdata, return_type="X")
        elif hasattr(newdata, "to_dict"):
            data_dict = newdata.to_dict(orient="list")
            if "y" in data_dict:
                _, X = formula("y ~ .", data_dict)
            else:
                X = formula("~ .", data_dict, return_type="X")
        elif isinstance(newdata, np.ndarray):
            X = newdata
        else:
            raise ValueError("newdata must be dict, DataFrame, or numpy array")

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    result = predict_basic(model, X, interval, level, side)
    result.newdata_provided = newdata_provided

    h = X.shape[0]

    if occurrence is not None:
        occurrence = np.asarray(occurrence, dtype=float)
        if len(occurrence) != h:
            raise ValueError(
                f"occurrence has {len(occurrence)} elements but newdata has {h} rows"
            )

    if interval != "none":
        level_matrix = np.tile(level, (h, 1))

        if side == "upper":
            level_low = np.zeros((h, n_levels))
            level_up = level_matrix
        elif side == "lower":
            level_low = 1 - level_matrix
            level_up = np.ones((h, n_levels))
        else:
            level_low = (1 - level_matrix) / 2
            level_up = (1 + level_matrix) / 2

        level_low = np.clip(level_low, 0, 1)
        level_up = np.clip(level_up, 0, 1)

        if model.distribution == "dnorm":
            if result.lower is not None:
                result.lower = stats.norm.ppf(
                    level_low, loc=result.mean, scale=model.scale_
                )
                result.upper = stats.norm.ppf(
                    level_up, loc=result.mean, scale=model.scale_
                )

    return result
