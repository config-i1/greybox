"""Model selection and combination functions.

This module provides functions for stepwise regression, model combination,
and forecasting accuracy measures.
"""

import itertools

import numpy as np
from typing import Literal, TypedDict, Union

from pandas import DataFrame

from .alm import ALM
from .formula import formula as formula_func


class ModelInfo(TypedDict):
    vars: list[str]
    model: ALM
    ic: float
    coef: np.ndarray


def _convert_to_dict(data: Union[dict, DataFrame]) -> dict:
    """Convert DataFrame to dict if needed."""
    if isinstance(data, DataFrame):
        return data.to_dict(orient="list")
    return data


def stepwise(
    data: Union[dict, DataFrame],
    ic: Literal["AICc", "AIC", "BIC", "BICc"] = "AICc",
    silent: bool = True,
    distribution: str = "dnorm",
) -> ALM:
    """Stepwise selection of regressors.

    Function selects variables that give linear regression with the lowest
    information criterion. The selection is done stepwise (forward) based on
    partial correlations.

    Parameters
    ----------
    data : dict or DataFrame
        Data frame containing dependent variable in the first column and
        the others in the rest.
    ic : {"AICc", "AIC", "BIC", "BICc"}, default="AICc"
        Information criterion to use.
    silent : bool, default=True
        If False, then progress is printed.
    distribution : str, default="dnorm"
        Distribution to use for the ALM model.

    Returns
    -------
    ALM
        The final fitted model.

    Examples
    --------
    >>> data = {'y': [1, 2, 3, 4, 5], 'x1': [1, 2, 3, 4, 5],
    ...         'x2': [2, 4, 6, 8, 10]}
    >>> model = stepwise(data)
    """
    data_dict = _convert_to_dict(data)

    y_var = list(data_dict.keys())[0]
    x_vars = list(data_dict.keys())[1:]

    if len(x_vars) == 0:
        raise ValueError("Need at least one predictor variable")

    best_ic = np.inf
    selected_vars: list[str] = []
    all_vars = x_vars.copy()
    feature_names: list[str] | None = None

    while all_vars:
        best_var_to_add = None
        best_var_ic = np.inf

        for var in all_vars:
            current_vars = selected_vars + [var]
            formula_str = f"{y_var} ~ " + " + ".join(current_vars)

            try:
                y_fit, X_fit = formula_func(formula_str, data_dict, as_dataframe=True)
                feature_names_loop = list(X_fit.columns)
                feature_names_loop.remove("(Intercept)")
                model = ALM(distribution=distribution, loss="likelihood")
                model.fit(
                    X_fit, y_fit, formula=formula_str, feature_names=feature_names_loop
                )

                ic_value = _get_ic_value(model, ic)

                if ic_value < best_var_ic:
                    best_var_ic = ic_value
                    best_var_to_add = var
            except Exception:
                continue

        if best_var_to_add is None or best_var_ic >= best_ic:
            break

        selected_vars.append(best_var_to_add)
        all_vars.remove(best_var_to_add)
        best_ic = best_var_ic

        if not silent:
            print(f"Added {best_var_to_add}, IC={best_var_ic:.4f}")

    if not selected_vars:
        formula_str = f"{y_var} ~ 1"
        feature_names = None
    else:
        formula_str = f"{y_var} ~ " + " + ".join(selected_vars)
        _, X_final_temp = formula_func(formula_str, data_dict, as_dataframe=True)
        feature_names = list(X_final_temp.columns)
        feature_names.remove("(Intercept)")

    y_final, X_final = formula_func(formula_str, data_dict, as_dataframe=True)
    final_model = ALM(distribution=distribution, loss="likelihood")
    final_model.fit(X_final, y_final, formula=formula_str, feature_names=feature_names)

    return final_model


def _get_ic_value(model: ALM, ic_type: str) -> float:
    """Get the IC value from a model."""
    if ic_type == "AIC":
        return model.aic if model.aic is not None else np.inf
    elif ic_type == "BIC":
        return model.bic if model.bic is not None else np.inf
    elif ic_type == "AICc":
        n = model.nobs
        k = model.nparam
        aic = model.aic if model.aic is not None else np.inf
        if n - k - 1 > 0:
            return aic + (2 * k**2 + 2 * k) / (n - k - 1)
        return np.inf
    elif ic_type == "BICc":
        n = model.nobs
        k = model.nparam
        bic = model.bic if model.bic is not None else np.inf
        return bic + k * np.log(n) ** 2 / n
    return np.inf


def lm_combine(
    data: Union[dict, DataFrame],
    ic: Literal["AICc", "AIC", "BIC", "BICc"] = "AICc",
    bruteforce: bool = True,
    silent: bool = True,
    distribution: str = "dnorm",
) -> dict:
    """Combine regressions based on information criteria.

    Function combines parameters of linear regressions of the first variable
    on all the other provided data.

    Parameters
    ----------
    data : dict or DataFrame
        Data frame containing dependent variable in the first column and
        the others in the rest.
    ic : {"AICc", "AIC", "BIC", "BICc"}, default="AICc"
        Information criterion to use.
    bruteforce : bool, default=True
        If True, all possible models are generated and combined.
        Otherwise the best model is found and then models around that
        one are produced and combined.
    silent : bool, default=True
        If False, then progress is printed.
    distribution : str, default="dnorm"
        Distribution to use for the ALM model.

    Returns
    -------
    dict
        Dictionary containing:
        - coefficients: combined parameters
        - fitted: fitted values
        - residuals: residuals
        - log_lik: combined log-likelihood
        - IC: combined information criterion
        - importance: importance of parameters

    Examples
    --------
    >>> data = {'y': [1, 2, 3, 4, 5], 'x1': [1, 2, 3, 4, 5],
    ...         'x2': [2, 4, 6, 8, 10]}
    >>> result = lm_combine(data)
    """
    data_dict = _convert_to_dict(data)

    y_var = list(data_dict.keys())[0]
    x_vars = list(data_dict.keys())[1:]

    if len(x_vars) == 0:
        raise ValueError("Need at least one predictor variable")

    n_vars = len(x_vars)

    if bruteforce:
        all_combinations: list[list[str]] = []
        for r in range(1, n_vars + 1):
            for combo in itertools.combinations(x_vars, r):
                all_combinations.append(list(combo))
    else:
        stepwise(data_dict, ic=ic, silent=True, distribution=distribution)
        included = [x_vars[i] for i in range(len(x_vars))]
        all_combinations = [included]

        for i in range(len(included)):
            for combo in itertools.combinations(included, len(included) - 1):
                if list(combo) not in all_combinations:
                    all_combinations.append(list(combo))

    models: list[ModelInfo] = []
    for combo in all_combinations:  # type: ignore[assignment]
        combo_list: list[str] = list(combo)
        formula_str = f"{y_var} ~ " + " + ".join(combo_list)
        try:
            y_fit, X_fit = formula_func(formula_str, data_dict, as_dataframe=True)
            feature_names = list(X_fit.columns)
            feature_names.remove("(Intercept)")
            model = ALM(distribution=distribution, loss="likelihood")
            model.fit(X_fit, y_fit, formula=formula_str, feature_names=feature_names)

            ic_value = _get_ic_value(model, ic)
            models.append(
                {
                    "vars": combo_list,
                    "model": model,
                    "ic": ic_value,
                    "coef": np.concatenate([[model.intercept_], model.coef]),
                }
            )
        except Exception:
            continue

    if not models:
        raise ValueError("No models could be fitted")

    models.sort(key=lambda x: x["ic"])
    best_ic = models[0]["ic"]

    weights = np.array([np.exp(-0.5 * (m["ic"] - best_ic)) for m in models])
    weights = weights / np.sum(weights)

    n_params = len(x_vars) + 1
    combined_coef = np.zeros(n_params)
    for i, model_info in enumerate(models):
        for j, var in enumerate(model_info["vars"]):
            idx = x_vars.index(var) + 1
            combined_coef[idx] += weights[i] * model_info["coef"][j]
        combined_coef[0] += weights[i] * model_info["coef"][0]

    formula_str_final = f"{y_var} ~ " + " + ".join(x_vars)
    y_final, X_final = formula_func(formula_str_final, data_dict, as_dataframe=True)
    feature_names_final = list(X_final.columns)
    feature_names_final.remove("(Intercept)")

    fitted = X_final @ combined_coef
    residuals = y_final - fitted

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_final - np.mean(y_final)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    combined_log_lik = sum(w * m["model"].log_lik for w, m in zip(weights, models))

    importance = np.zeros(n_params)
    for i, model_info in enumerate(models):
        for j, var in enumerate(model_info["vars"]):
            idx = x_vars.index(var) + 1
            importance[idx] += weights[i] * np.abs(model_info["coef"][j])

    coefficient_names = ["(Intercept)"]
    if feature_names_final is not None:
        coefficient_names.extend(feature_names_final)

    return {
        "coefficients": combined_coef,
        "coefficient_names": coefficient_names,
        "y_variable": y_var,
        "x_variables": x_vars,
        "fitted": fitted,
        "residuals": residuals,
        "log_lik": combined_log_lik,
        "IC": best_ic,
        "IC_type": ic,
        "R2": r_squared,
        "importance": importance,
        "weights": weights,
        "models": models,
    }
