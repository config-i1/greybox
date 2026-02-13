"""Formula parser for greybox models.

This module provides formula parsing functionality similar to R's formula system.
Users can call the formula function to get X (design matrix) and y, then pass
them to the model fit method.
"""

import re
import numpy as np


def formula(formula_str, data, return_type="both"):
    """Parse formula string and return design matrix and/or response.

    Parameters
    ----------
    formula_str : str
        Formula string in R-style, e.g., "y ~ x1 + x2" or "~ x1 + x2" (no y).
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
    """
    formula_str = formula_str.strip()

    has_response = True
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
            data_dict = data.to_dict(orient='list')
        except AttributeError:
            raise ValueError("data must be dict or DataFrame")

    if return_type == "terms":
        return terms

    y = None
    if has_response:
        if lhs not in data_dict:
            raise ValueError(f"Response variable '{lhs}' not found in data")
        y = np.array(data_dict[lhs], dtype=float)
        terms = rhs_terms

    if return_type == "y":
        return y

    X_columns = []
    intercept_added = False

    for term in terms:
        term = term.strip()
        if not term:
            continue

        if term == "1" or term == "intercept":
            if not intercept_added:
                X_columns.append(np.ones(len(y) if y is not None else len(next(iter(data_dict.values())))))
                intercept_added = True
            continue

        if "*" in term:
            parts = term.split("*")
            if len(parts) == 2:
                var1, var2 = parts[0].strip(), parts[1].strip()
                if var1 not in data_dict:
                    raise ValueError(f"Variable '{var1}' not found in data")
                if var2 not in data_dict:
                    raise ValueError(f"Variable '{var2}' not found in data")
                col1 = np.array(data_dict[var1], dtype=float)
                col2 = np.array(data_dict[var2], dtype=float)
                X_columns.append(col1 * col2)
            continue

        if ":" in term:
            parts = term.split(":")
            if len(parts) == 2:
                var1, var2 = parts[0].strip(), parts[1].strip()
                if var1 not in data_dict:
                    raise ValueError(f"Variable '{var1}' not found in data")
                if var2 not in data_dict:
                    raise ValueError(f"Variable '{var2}' not found in data")
                col1 = np.array(data_dict[var1], dtype=float)
                col2 = np.array(data_dict[var2], dtype=float)
                X_columns.append(col1 * col2)
            continue

        if term in data_dict:
            X_columns.append(np.array(data_dict[term], dtype=float))
        else:
            raise ValueError(f"Variable '{term}' not found in data")

    if not intercept_added:
        X_columns.insert(0, np.ones(len(y) if y is not None else len(X_columns[0])))

    if not X_columns:
        n_obs = len(y) if y is not None else len(next(iter(data_dict.values())))
        X = np.ones((n_obs, 1))
    else:
        X = np.column_stack(X_columns)

    if return_type == "X":
        return X

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

    rhs = re.sub(r'(\w+)\s*\*\s*(\w+)', r'\1 + \2 + \1:\2', rhs)

    return f"{lhs.strip()} ~ {rhs}"
