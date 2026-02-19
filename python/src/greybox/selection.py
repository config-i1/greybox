"""Model selection and combination functions.

This module provides functions for stepwise regression, model combination,
and forecasting accuracy measures.
"""

import itertools
import time
from typing import Any, Literal, Optional, TypedDict, Union

import numpy as np
from pandas import DataFrame
from scipy import stats

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


def _prepare_data(
    data: Union[dict, DataFrame],
    formula_str: Optional[str] = None,
    subset: Optional[Any] = None,
) -> tuple[dict, str, list[bool]]:
    """Prepare data for stepwise selection.

    Parameters
    ----------
    data : dict or DataFrame
        Data containing variables.
    formula_str : str, optional
        Formula for variable selection after transformations.
    subset : array-like, optional
        Subset of observations to use.

    Returns
    -------
    tuple
        (data_dict, response_name, rows_selected)
    """
    data_dict = _convert_to_dict(data)

    if formula_str is not None or subset is not None:
        raise NotImplementedError(
            "formula and subset parameters are not yet implemented"
        )

    all_vars = list(data_dict.keys())
    response_name = all_vars[0]

    n_obs = len(data_dict[response_name])
    if subset is not None:
        rows_selected = np.array(subset, dtype=bool)
        if len(rows_selected) != n_obs:
            raise ValueError("subset length does not match data")
    else:
        rows_selected = np.ones(n_obs, dtype=bool)

    if any(isinstance(v, list) for v in data_dict.values()):
        for key in data_dict:
            data_dict[key] = np.array(data_dict[key], dtype=float)

    return data_dict, response_name, rows_selected


def _check_variability(data_dict: dict, var_names: list[str]) -> list[str]:
    """Check variability in variables and remove those without any.

    Parameters
    ----------
    data_dict : dict
        Data dictionary.
    var_names : list of str
        Variable names to check.

    Returns
    -------
    list of str
        Variable names that have variability.
    """
    vars_with_variability = []
    for var in var_names:
        values = np.array(data_dict[var])
        if len(np.unique(values)) > 1:
            vars_with_variability.append(var)
    return vars_with_variability


def _calculate_associations(
    residuals: np.ndarray,
    data_dict: dict,
    var_names: list[str],
    rows_selected: np.ndarray,
    method: str = "pearson",
) -> dict[str, float]:
    """Calculate association measures between residuals and variables.

    This is similar to R's assocFast() function.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals.
    data_dict : dict
        Data dictionary.
    var_names : list of str
        Variable names to calculate associations for.
    rows_selected : np.ndarray
        Boolean array indicating which rows to use.
    method : str
        Correlation method: "pearson", "kendall", or "spearman".

    Returns
    -------
    dict
        Dictionary mapping variable names to association values.
    """
    associations = {}

    for var in var_names:
        values = np.array(data_dict[var])[rows_selected]
        resid = residuals[rows_selected]

        try:
            if method == "pearson":
                corr, _ = stats.pearsonr(resid, values)
            elif method == "spearman":
                corr, _ = stats.spearmanr(resid, values)
            elif method == "kendall":
                corr, _ = stats.kendalltau(resid, values)
            else:
                corr = 0.0

            associations[var] = abs(corr) if not np.isnan(corr) else 0.0
        except Exception:
            associations[var] = 0.0

    return associations


def _get_ic_function(ic: str):
    """Get the IC function for calculation."""
    ic = ic.upper()
    if ic == "AIC":
        return lambda log_lik, df: -2 * log_lik + 2 * df
    elif ic == "BIC":
        return lambda log_lik, df: (
            -2 * log_lik
            + df * np.log(len(log_lik) if hasattr(log_lik, "__len__") else 1)
        )
    elif ic == "AICCC":
        n = 1
        return lambda log_lik, df: (
            -2 * log_lik + 2 * df + (2 * df * (df + 1)) / (n - df - 1)
            if n - df - 1 > 0
            else np.inf
        )
    else:
        return lambda log_lik, df: -2 * log_lik + 2 * df


def _calculate_ic(
    model: ALM,
    ic_type: str,
    df_add: int = 0,
) -> float:
    """Calculate information criterion from a model.

    Parameters
    ----------
    model : ALM
        Fitted model.
    ic_type : str
        Type of IC: "AIC", "BIC", "AICc", "BICc".
    df_add : int
        Additional degrees of freedom to add.

    Returns
    -------
    float
        The IC value.
    """
    n = model.nobs
    k = model.nparam + df_add
    log_lik = model.log_lik

    ic_type = ic_type.upper()
    if ic_type.endswith("CC"):
        ic_type = ic_type[:-1]  # Handle AICc -> AICC -> AICc

    n = model.nobs
    k = model.nparam + df_add
    log_lik = model.log_lik

    if ic_type == "AIC":
        return -2 * log_lik + 2 * k
    elif ic_type == "BIC":
        return -2 * log_lik + k * np.log(n)
    elif ic_type == "AICC":
        if n - k - 1 > 0:
            return -2 * log_lik + 2 * k + (2 * k**2 + 2 * k) / (n - k - 1)
        return np.inf
    elif ic_type == "BICC":
        return -2 * log_lik + k * np.log(n) + k * np.log(n) ** 2 / n
    return np.inf


def stepwise(
    data: Union[dict, DataFrame],
    ic: Literal["AICc", "AIC", "BIC", "BICc"] = "AICc",
    silent: bool = True,
    df: Optional[int] = None,
    formula: Optional[str] = None,
    subset: Optional[Any] = None,
    method: Literal["pearson", "kendall", "spearman"] = "pearson",
    distribution: str = "dnorm",
    occurrence: Literal["none", "plogis", "pnorm"] = "none",
    **kwargs,
) -> ALM:
    """Stepwise selection of regressors.

    Function selects variables that give linear regression with the lowest
    information criterion. The selection is done stepwise (forward) based on
    partial correlations. This should be a simpler and faster implementation
    than step() function from stats package.

    The algorithm uses ALM to fit different models and correlation to select
    the next regressor in the sequence.

    Parameters
    ----------
    data : dict or DataFrame
        Data frame containing dependent variable in the first column and
        the others in the rest.
    ic : {"AICc", "AIC", "BIC", "BICc"}, default="AICc"
        Information criterion to use.
    silent : bool, default=True
        If False, then progress is printed.
    df : int, optional
        Number of degrees of freedom to add (should be used if stepwise is
        used on residuals).
    formula : str, optional
        If provided, then the selection will be done from the listed
        variables in the formula after all the necessary transformations.
    subset : array-like, optional
        An optional vector specifying a subset of observations to be used
        in the fitting process.
    method : {"pearson", "kendall", "spearman"}, default="pearson"
        Method of correlations calculation. The default is Pearson's
        correlation, which should be applicable to a wide range of data
        in different scales.
    distribution : str, default="dnorm"
        Distribution to use for the ALM model. See ALM for details.
    occurrence : {"none", "plogis", "pnorm"}, default="none"
        What distribution to use for occurrence part. See ALM for details.

    Returns
    -------
    ALM
        The final fitted model with additional attributes:
        - ICs: list of IC values at each step
        - timeElapsed: time taken for calculation

    Examples
    --------
    >>> data = {'y': [1, 2, 3, 4, 5], 'x1': [1, 2, 3, 4, 5],
    ...         'x2': [2, 4, 6, 8, 10]}
    >>> model = stepwise(data)
    """
    start_time = time.time()

    if df is None:
        df = 0

    data_dict, response_name, rows_selected = _prepare_data(data, formula, subset)

    data_dict = {k: np.array(v) for k, v in data_dict.items()}

    x_vars = [k for k in data_dict.keys() if k != response_name]

    if len(x_vars) == 0:
        raise ValueError("Need at least one predictor variable")

    x_vars = _check_variability(data_dict, x_vars)

    if len(x_vars) == 0:
        raise ValueError(
            "None of exogenous variables has variability. There's nothing to select!"
        )

    selected_vars: list[str] = []
    all_ics: list[float] = []

    formula_str = f"{response_name} ~ 1"
    y_fit, X_fit = formula_func(formula_str, data_dict, as_dataframe=True)
    feature_names = [c for c in X_fit.columns if c != "(Intercept)"]

    model = ALM(distribution=distribution, loss="likelihood", **kwargs)
    model.fit(X_fit, y_fit, formula=formula_str, feature_names=feature_names)

    current_ic = _calculate_ic(model, ic, df)
    best_ic = current_ic
    all_ics.append(current_ic)

    residuals = np.array(y_fit) - np.array(model.fitted)
    if len(residuals.shape) > 1:
        residuals = residuals.flatten()

    best_formula = formula_str

    if not silent:
        print(f"Formula: {formula_str}, IC: {current_ic:.4f}\n")

    m = 2
    best_ic_not_found = True

    while best_ic_not_found:
        associations = _calculate_associations(
            residuals, data_dict, x_vars, rows_selected, method
        )

        available_vars = [v for v in x_vars if v not in selected_vars]

        if not available_vars:
            best_ic_not_found = False
            break

        max_assoc = -1
        new_element = None
        for var in available_vars:
            if var in associations and associations[var] > max_assoc:
                max_assoc = associations[var]
                new_element = var

        if new_element is None:
            best_ic_not_found = False
            break

        test_formula = best_formula + f" + {new_element}"

        try:
            y_fit, X_fit = formula_func(test_formula, data_dict, as_dataframe=True)
            feature_names = [c for c in X_fit.columns if c != "(Intercept)"]

            model = ALM(distribution=distribution, loss="likelihood", **kwargs)
            model.fit(X_fit, y_fit, formula=test_formula, feature_names=feature_names)

            current_ic = _calculate_ic(model, ic, df)

            if not silent:
                print(f"Step {m - 1}. Formula: {test_formula}, IC: {current_ic:.4f}")
                print("Correlations: ")
                for var in available_vars:
                    print(f"  {var}: {associations.get(var, 0):.3f}")
                print()

            if current_ic >= best_ic:
                best_ic_not_found = False
            else:
                best_ic = current_ic
                best_formula = test_formula
                selected_vars.append(new_element)
                residuals = np.array(y_fit) - np.array(model.fitted)
                if len(residuals.shape) > 1:
                    residuals = residuals.flatten()

            all_ics.append(current_ic)
            m += 1

        except Exception as e:
            if not silent:
                print(f"Error fitting model with {new_element}: {e}")
            best_ic_not_found = False

    y_final, X_final = formula_func(best_formula, data_dict, as_dataframe=True)
    feature_names_final = [c for c in X_final.columns if c != "(Intercept)"]

    final_model = ALM(distribution=distribution, loss="likelihood", **kwargs)
    final_model.fit(
        X_final, y_final, formula=best_formula, feature_names=feature_names_final
    )

    elapsed_time = time.time() - start_time

    final_model.ICs = all_ics
    final_model.timeElapsed = elapsed_time

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
