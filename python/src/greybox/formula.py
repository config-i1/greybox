"""Formula parser for greybox models.

This module provides formula parsing functionality similar to R's formula system.
Users can call the formula function to get X (design matrix) and y, then pass
them to the model fit method.
"""

import inspect
import re

import numpy as np
import pandas as pd

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


def _get_caller_globals():
    """Get globals from the caller's caller (the user's code).

    Returns
    -------
    dict
        Global namespace from the user's code.
    """
    frame = inspect.currentframe()
    if frame is None:
        return {}
    try:
        caller_frame = frame.f_back.f_back
        if caller_frame is not None:
            return caller_frame.f_globals
        return {}
    finally:
        del frame


def _parse_transformation(term, caller_globals=None):
    """Parse a term to check if it contains a transformation.

    Parameters
    ----------
    term : str
        Term to parse, e.g., "log(x)", "x^2", "I(x+y)"
    caller_globals : dict, optional
        Global namespace from user's code to resolve custom functions.

    Returns
    -------
    tuple : (transformation, variable, is_protected)
        - transformation: str or None (e.g., "log", "sqrt", "^2")
        - variable: str (the base variable name)
        - is_protected: bool (True if wrapped in I())

    Raises
    ------
    ValueError
        If the term looks like a function call but the function is not defined.
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

    func_pattern = r"^([a-zA-Z_][a-zA-Z0-9_]*)\(([^)]+)\)$"
    func_match = re.match(func_pattern, term)
    if func_match:
        func_name = func_match.group(1)
        if func_name in TRANSFORMATIONS:
            return (func_name, func_match.group(2), False)
        if caller_globals and func_name in caller_globals:
            if callable(caller_globals[func_name]):
                return (func_name, func_match.group(2), False)
            else:
                raise ValueError(f"'{func_name}' is not a callable function")
        raise ValueError(
            f"Unknown function '{func_name}' in transformation '{term}'. "
            f"Either use a built-in transformation (log, sqrt, exp, etc.) "
            f"or make sure '{func_name}' is defined or imported in your global scope."
        )

    if caller_globals:
        pattern = r"([a-zA-Z_][a-zA-Z0-9_]*)\(([^)]+)\)"
        match = re.match(pattern, term)
        if match:
            func_name = match.group(1)
            if func_name in caller_globals and callable(caller_globals[func_name]):
                return (func_name, match.group(2), False)

    if "^" in term:
        parts = term.split("^")
        if len(parts) == 2:
            try:
                power = int(parts[1].strip())
                return (f"^ {power}", parts[0].strip(), False)
            except ValueError:
                pass

    return (None, term, False)


def _apply_transformation(var_name, data_dict, n_obs, caller_globals=None):
    """Apply transformation to a variable.

    Parameters
    ----------
    var_name : str
        Variable name, possibly with transformation (e.g., "log(x)", "x^2")
    data_dict : dict
        Data dictionary.
    n_obs : int
        Number of observations.
    caller_globals : dict, optional
        Global namespace from user's code to resolve custom functions.

    Returns
    -------
    np.ndarray
        Transformed variable values.
    """
    if caller_globals is None:
        caller_globals = {}

    transform, base_var, is_protected = _parse_transformation(var_name, caller_globals)

    if transform == "I":
        inner_transform, inner_base, _ = _parse_transformation(base_var, caller_globals)
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
        if inner_transform in caller_globals:
            return caller_globals[inner_transform](base_data)
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

    if transform in caller_globals:
        return caller_globals[transform](base_data)

    raise ValueError(f"Unknown transformation: '{transform}'. "
                    f"If '{transform}' is a function you defined or imported, "
                    f"make sure it is available in the global scope where formula() is called.")


def formula(formula_str, data, return_type="both", as_dataframe=True):
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
    as_dataframe : bool, optional
        If True, return pandas DataFrames instead of numpy arrays.
        This preserves variable names in column names. Default is True.

    Returns
    -------
    Depends on return_type and as_dataframe:
        - "both": tuple (y, X) where y is 1D array (or DataFrame)
          and X is 2D array (or DataFrame) with intercept
        - "X": design matrix only (or DataFrame if as_dataframe=True)
        - "y": response only (or DataFrame if as_dataframe=True)
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

    >>> # Return as DataFrames with column names
    >>> y, X = formula("y ~ x1 + x2", data, as_dataframe=True)
    >>> print(X.columns)  # ['(Intercept)', 'x1', 'x2']

    >>> # With custom functions (defined or imported in your global scope)
    >>> def my_transform(x):
    ...     return x * 2
    >>> data = {'y': [1, 2, 3], 'x': [1, 2, 3]}
    >>> y, X = formula("y ~ my_transform(x)", data)

    >>> # With imported functions (e.g., from scipy)
    >>> from scipy.special import erfc
    >>> data = {'y': [1, 2, 3], 'x': [0.5, 1.0, 1.5]}
    >>> y, X = formula("y ~ erfc(x)", data)

    >>> # Custom function on LHS (response variable)
    >>> y, X = formula("my_transform(y) ~ x", data)
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

    if isinstance(data, dict):
        data_dict = data
    else:
        try:
            data_dict = data.to_dict(orient="list")
        except AttributeError:
            raise ValueError("data must be dict or DataFrame")

    caller_globals = _get_caller_globals()

    # Handle "." as "use all variables"
    if "." in formula_str:
        # Get all variables from data
        all_vars = set(data_dict.keys())

        # Get response variable from LHS if present
        response_var = None
        if lhs:
            transform, base_var, _ = _parse_transformation(lhs, caller_globals)
            response_var = base_var
            if response_var and response_var in all_vars:
                all_vars.remove(response_var)

        # Parse the formula string to find explicit inclusions and exclusions
        # Handle transformations: log(wt), sqrt(x), etc.
        excluded_vars = set()
        mentioned_vars = set()

        # Normalize the formula string: replace "-" with " - " and "+" with " + "
        formula_for_parsing = formula_str.replace("-", " - ").replace("+", " + ")
        tokens = formula_for_parsing.split()

        # First pass: identify excluded and mentioned variables
        i = 0
        while i < len(tokens):
            token = tokens[i].strip()
            if not token or token in (".", "+"):
                i += 1
                continue
            if token == "-":
                # Next token is excluded
                i += 1
                if i < len(tokens):
                    excluded_token = tokens[i].strip()
                    if excluded_token:
                        # Extract base variable name (handle transformations)
                        transform, base_var, _ = _parse_transformation(excluded_token, caller_globals)
                        excluded_vars.add(base_var)
                i += 1
                continue
            # This is a term - extract base variable
            transform, base_var, _ = _parse_transformation(token, caller_globals)
            mentioned_vars.add(base_var)
            i += 1

        # Remove excluded vars from all_vars
        all_vars = all_vars - excluded_vars

        # Remove mentioned vars from all_vars (they're already in the formula)
        for var in mentioned_vars:
            if var in all_vars:
                all_vars.remove(var)

        # Build the new formula string
        # Collect original terms (without ".") from tokens
        original_terms = []
        i = 0
        while i < len(tokens):
            token = tokens[i].strip()
            # Skip empty tokens and "."
            if not token or token == ".":
                i += 1
                continue
            # Handle "-" by skipping it and the next variable
            if token == "-":
                i += 2  # Skip "-" and the following variable
                continue
            # Skip standalone "+"
            if token == "+":
                i += 1
                continue
            # This is a term to include
            original_terms.append(token)
            i += 1

        # Add remaining variables from data
        remaining_vars = sorted(list(all_vars))
        original_terms.extend(remaining_vars)

        formula_str = " + ".join(original_terms)

    terms = _parse_formula_terms(formula_str)
    rhs_terms = terms.copy()

    n_obs = len(next(iter(data_dict.values())))

    if return_type == "terms":
        return terms

    y = None
    if has_response and lhs:
        transform, base_var, is_protected = _parse_transformation(lhs, caller_globals)

        # Handle I() wrapper specially - extract base variable from inside
        if is_protected:
            # For I(var), extract var and check if it exists
            inner_transform, inner_base, _ = _parse_transformation(base_var, caller_globals)
            if inner_transform is not None:
                # I(sqrt(y)) or I(y^2) - use the inner base variable
                if inner_base not in data_dict:
                    raise ValueError(
                        f"Response variable '{inner_base}' not found in data"
                    )
                y_data = np.array(data_dict[inner_base], dtype=float)
                if inner_transform.startswith("^"):
                    power = int(inner_transform.split()[1])
                    y = y_data**power
                elif inner_transform in TRANSFORMATIONS:
                    y = TRANSFORMATIONS[inner_transform](y_data)
                elif inner_transform in caller_globals:
                    y = caller_globals[inner_transform](y_data)
            else:
                # Just I(y) - use y directly
                if base_var not in data_dict:
                    raise ValueError(
                        f"Response variable '{base_var}' not found in data"
                    )
                y_data = np.array(data_dict[base_var], dtype=float)
                y = y_data
        elif transform is not None:
            if base_var not in data_dict:
                raise ValueError(f"Response variable '{base_var}' not found in data")
            y_data = np.array(data_dict[base_var], dtype=float)
            if transform is None:
                y = y_data
            elif transform.startswith("^"):
                power = int(transform.split()[1])
                y = y_data**power
            elif transform in TRANSFORMATIONS:
                y = TRANSFORMATIONS[transform](y_data)
            elif transform in caller_globals:
                y = caller_globals[transform](y_data)
        else:
            if lhs not in data_dict:
                raise ValueError(f"Response variable '{lhs}' not found in data")
            y_data = np.array(data_dict[lhs], dtype=float)
            y = y_data

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

        # Use the full term as the variable name (preserves transformations)
        var_name = term.strip()
        if var_name.startswith("-"):
            var_name = var_name[1:]

        try:
            col = _apply_transformation(term, data_dict, n_obs, caller_globals)
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

    if as_dataframe:
        X_df = pd.DataFrame(X, columns=var_names)

        if return_type == "X":
            return X_df
        elif return_type == "variables":
            return coef_names

        y_name = None
        if has_response and lhs:
            y_name = lhs.strip() if lhs else None

        if y_name is not None:
            y_df = pd.Series(y, name=y_name)
        elif y is not None:
            y_df = pd.Series(y, name="y")
        else:
            y_df = None

        return y_df, X_df

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
