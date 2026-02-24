"""Greybox - Toolbox for model building and forecasting."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("greybox")
except PackageNotFoundError:
    __version__ = "unknown"

from . import association
from . import data
from . import distributions
from . import error_measures
from . import hm
from . import quantile_measures
from .error_measures import measures
from .alm import ALM, PredictionResult
from .formula import formula, expand_formula
from .selection import stepwise, CALM, LmCombineResult
from .transforms import bc_transform, bc_transform_inv, mean_fast
from .xreg import B, multipliers
from .data import mtcars

__all__ = [
    "__version__",
    "ALM",
    "PredictionResult",
    "formula",
    "expand_formula",
    "stepwise",
    "CALM",
    "LmCombineResult",
    "bc_transform",
    "bc_transform_inv",
    "mean_fast",
    "B",
    "multipliers",
    "distributions",
    "error_measures",
    "measures",
    "association",
    "hm",
    "quantile_measures",
    "data",
    "mtcars",
]
