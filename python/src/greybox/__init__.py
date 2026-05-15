"""Greybox - Toolbox for model building and forecasting."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("greybox")
except PackageNotFoundError:
    __version__ = "unknown"

from . import association
from . import data
from . import distributions
from . import point_measures
from . import hm
from . import quantile_measures
from . import rolling
from . import smoothers
from .point_measures import measures
from .alm import ALM, PredictionResult
from .formula import formula, expand_formula
from .selection import stepwise, CALM, CALMResult
from .transforms import bc_transform, bc_transform_inv, mean_fast
from .xreg import B, multipliers
from .rolling import rolling_origin, RollingOriginResult
from .data import mtcars
from .pointlik import point_lik, point_lik_cumulative
from .aid import aid, aid_cat, AidResult, AidCatResult

__all__ = [
    "__version__",
    "ALM",
    "PredictionResult",
    "formula",
    "expand_formula",
    "stepwise",
    "CALM",
    "CALMResult",
    "bc_transform",
    "bc_transform_inv",
    "mean_fast",
    "B",
    "multipliers",
    "rolling",
    "rolling_origin",
    "RollingOriginResult",
    "distributions",
    "point_measures",
    "measures",
    "association",
    "hm",
    "quantile_measures",
    "data",
    "mtcars",
    "smoothers",
    "point_lik",
    "point_lik_cumulative",
    "aid",
    "aid_cat",
    "AidResult",
    "AidCatResult",
]
