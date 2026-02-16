"""Formula parser for greybox models.

This module provides formula parsing functionality similar to R's formula system.
Users can call the formula function to get X (design matrix) and y, then pass
them to the model fit method.
"""

import re
import numpy as np

TRANSFORMATIONS = {
    "log": np.log,
    "log10": np.log10,
    "log2": np.log2,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "abs": np.abs,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
}


def _parse_transformation(term):
    """Parse a term to check if it contains a transformation.

    Parameters
    ----------
    term : str
        Term to parse, e.g., "log(x)", "x^2", "I(x+y)"

    Returns
    -------
    tuple : (transformation, variable, is_protected)
        - transformation: str or None (e.g., "log", "sqrt", "^2")
        - variable: str (the base variable name)
        - is_protected: bool (True if wrapped in I())
    """
    term = term.strip()

    if term.startswith("I(") and term.endswith(")"):
        inner = term[2:-1]
        return ("I", inner, True)

    for func_name in TRANSFORMATIONS.keys():
        pattern = f"{func_name}\\(([^)]+)\\)"
        match = re.match(pattern, term)
        if match:
            return (func_name, match.group(1), False)

    if "^" in term:
        parts = term.split("^")
        if len(parts) == 2:
            try:
                power = int(parts[1].strip())
                return (f"^ {power}", parts[0].strip(), False)
            except ValueError:
                pass

    return (None, term, False)


def _apply_transformation(var_name, data_dict, n_obs):
    """Apply transformation to a variable.

    Parameters
    ----------
    var_name : str
        Variable name, possibly with transformation (e.g., "log(x)", "x^2")
    data_dict : dict
        Data dictionary.
    n_obs : int
        Number of observations.

    Returns
    -------
    np.ndarray
        Transformed variable values.
    """
    transform, base_var, is_protected = _parse_transformation(var_name)

    if transform == "I":
        inner_transform, inner_base, _ = _parse_transformation(base_var)
        if inner_base in data_dict:
            base_data = np.array(data_dict[inner_base], dtype=float)
        else:
            raise ValueError(f"Variable '{inner_base}' not found in data")

        if inner_transform is None:
            return base_data
        if inner_transform.startswith("^"):
            power = int(inner_transform.split()[1])
            return base_data**power
        if inner_transform in TRANSFORMATIONS:
            return TRANSFORMATIONS[inner_transform](base_data)
        return base_data

    if base_var == "trend":
        if "trend" in data_dict:
            base_data = np.array(data_dict["trend"], dtype=float)
        else:
            base_data = np.arange(1, n_obs + 1, dtype=float)
    elif base_var in data_dict:
        base_data = np.array(data_dict[base_var], dtype=float)
    else:
        raise ValueError(f"Variable '{base_var}' not found in data")

    if transform is None:
        return base_data

    if transform.startswith("^"):
        power = int(transform.split()[1])
        return base_data**power

    if transform in TRANSFORMATIONS:
        return TRANSFORMATIONS[transform](base_data)

    raise ValueError(f"Unknown transformation: {transform}")


def formula(formula_str, data, return_type="both"):
    """Parse formula string and return design matrix and/or response.

    Parameters
    ----------
    formula_str : str
        Formula string in R-style, e.g., "y ~ x1 + x2" or "~ x1 + x2" (no y).
        Supports transformations: log(x), sqrt(x), x^2, etc.
        Use I() to protect expressions: I(x^2)
    data : dict or DataFrame
        Data containing variables. Can be dict, DataFrame, or dict of arrays.
    return_type : str, optional
        What to return: "both" (default), "X", "y", or "terms".

    Returns
    -------
    Depends on return_type:
        - "both": tuple (y, X) where y is 1D array and X is 2D array with intercept
        - "X": design matrix only
        - "y": response only
        - "terms": list of term names

    Examples
    --------
    >>> data = {'y': [1, 2, 3], 'x1': [4, 5, 6], 'x2': [7, 8, 9]}
    >>> y, X = formula("y ~ x1 + x2", data)
    >>> X_no_intercept = formula("~ x1 + x2", data, return_type="X")

    >>> # With transformations
    >>> data = {'y': [1, 2, 3], 'x': [1, 2, 3]}
    >>> # y is log-transformed, X has x and x^2
    >>> y, X = formula("log(y) ~ x + x^2", data)
    """
    formula_str = formula_str.strip()

    has_response = True
    lhs = None
    if formula_str.startswith("~"):
        has_response = False
        formula_str = formula_str[1:].strip()
    elif "~" in formula_str:
        parts = formula_str.split("~", 1)
        lhs = parts[0].strip()
        formula_str = parts[1].strip() if len(parts) > 1 else ""
        if lhs:
            has_response = True
        else:
            has_response = False

    if not formula_str:
        formula_str = "1"

    terms = _parse_formula_terms(formula_str)
    rhs_terms = terms.copy()

    if isinstance(data, dict):
        data_dict = data
    else:
        try:
            data_dict = data.to_dict(orient="list")
        except AttributeError:
            raise ValueError("data must be dict or DataFrame")

    n_obs = len(next(iter(data_dict.values())))

    if return_type == "terms":
        return terms

    y = None
    if has_response and lhs:
        transform, base_var, _ = _parse_transformation(lhs)
        if base_var not in data_dict:
            raise ValueError(f"Response variable '{base_var}' not found in data")
        y_data = np.array(data_dict[base_var], dtype=float)

        if transform is None:
            y = y_data
        elif transform == "I":
            y = y_data
        elif transform.startswith("^"):
            power = int(transform.split()[1])
            y = y_data**power
        elif transform in TRANSFORMATIONS:
            y = TRANSFORMATIONS[transform](y_data)

        terms = rhs_terms

    if return_type == "y":
        return y

    X_columns = []
    var_names = []
    intercept_added = False

    for term in terms:
        term = term.strip()
        if not term:
            continue

        if term == "1" or term == "intercept":
            if not intercept_added:
                X_columns.append(np.ones(n_obs))
                var_names.append("(Intercept)")
                intercept_added = True
            continue

        if term == "trend":
            if "trend" not in data_dict:
                X_columns.append(np.arange(1, n_obs + 1, dtype=float))
            else:
                X_columns.append(np.array(data_dict["trend"], dtype=float))
            var_names.append("trend")
            continue

        var_name = term.split("(")[0].split("^")[0].strip()
        if var_name.startswith("-"):
            var_name = var_name[1:]

        try:
            col = _apply_transformation(term, data_dict, n_obs)
            X_columns.append(col)
            var_names.append(var_name)
        except ValueError:
            raise

    if not intercept_added:
        X_columns.insert(0, np.ones(n_obs))
        var_names.insert(0, "(Intercept)")

    if not X_columns:
        X = np.ones((n_obs, 1))
    else:
        X = np.column_stack(X_columns)

    coef_names = [n for n in var_names if n != "(Intercept)"]

    if return_type == "X":
        return X
    elif return_type == "variables":
        return coef_names

    return y, X


def _parse_formula_terms(formula_str):
    """Parse formula string into list of terms."""
    terms = []

    formula_str = formula_str.replace("-", " - ")
    formula_str = formula_str.replace("+", " + ")
    formula_str = formula_str.replace("*", " * ")
    formula_str = formula_str.replace(":", " : ")

    tokens = formula_str.split()

    i = 0
    while i < len(tokens):
        token = tokens[i].strip()
        if token == "":
            i += 1
            continue
        if token == "+":
            i += 1
            continue
        if token == "-" and i + 1 < len(tokens):
            next_token = tokens[i + 1].strip()
            if next_token:
                terms.append("-" + next_token)
            i += 2
            continue
        if token == "*":
            if len(terms) > 0 and i + 1 < len(tokens):
                prev_term = terms[-1]
                next_token = tokens[i + 1].strip()
                if next_token and next_token not in ("+", "-", "*", ":", ""):
                    terms[-1] = prev_term + " * " + next_token
                    i += 2
                    continue
        if token == ":":
            if len(terms) > 0 and i + 1 < len(tokens):
                prev_term = terms[-1]
                next_token = tokens[i + 1].strip()
                if next_token and next_token not in ("+", "-", "*", ":", ""):
                    terms[-1] = prev_term + " : " + next_token
                    i += 2
                    continue
        if token:
            terms.append(token)
        i += 1

    return terms


def expand_formula(formula_str):
    """Expand formula with interaction terms.

    Parameters
    ----------
    formula_str : str
        Formula string, e.g., "y ~ x1 * x2"

    Returns
    -------
    str
        Expanded formula with explicit interaction terms.
    """
    if "~" not in formula_str:
        return formula_str

    lhs, rhs = formula_str.split("~", 1)

    rhs = rhs.strip()

    rhs = re.sub(r"(\w+)\s*\*\s*(\w+)", r"\1 + \2 + \1:\2", rhs)

    return f"{lhs.strip()} ~ {rhs}"
