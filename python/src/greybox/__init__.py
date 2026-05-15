"""Greybox - Toolbox for model building and forecasting.

All public functions and classes are re-exported at the top level so
users can write::

    from greybox import ALM, aid, lowess, formula, ham, association, ...

without having to remember which submodule each name lives in.

Submodules remain importable in the usual way (``import greybox.distributions``,
``from greybox.smoothers import lowess``), but the top-level package always
prefers the *function* with a given name over a submodule that shares it.
For example, ``from greybox import association`` returns the function
:func:`greybox.association.association`, not the submodule.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("greybox")
except PackageNotFoundError:
    __version__ = "unknown"

# Submodules that do not collide with a public function name and are kept
# importable as ``greybox.<submodule>``.
from . import data as data
from . import distributions as distributions
from . import point_measures as point_measures
from . import quantile_measures as quantile_measures
from . import rolling as rolling
from . import smoothers as smoothers

# Re-export every distribution function (dnorm, pnorm, qnorm, rnorm, ds, dt,
# dgeom, dpois, ...) at the top level so users can do ``from greybox import
# dnorm`` directly.
from .distributions import *  # noqa: F401,F403
from .distributions import __all__ as _distribution_names

# Core fitting and prediction
from .alm import ALM, PredictionResult

# Formula interface
from .formula import formula, expand_formula

# Model selection
from .selection import stepwise, CALM, CALMResult

# Variable processing
from .xreg import (
    B,
    multipliers,
    xreg_expander,
    xreg_multiplier,
    xreg_transformer,
    temporal_dummy,
)

# Transforms
from .transforms import bc_transform, bc_transform_inv, mean_fast

# Rolling-origin cross-validation
from .rolling import rolling_origin, RollingOriginResult

# Datasets
from .data import mtcars

# Point likelihoods
from .pointlik import point_lik, point_lik_cumulative

# Automatic Identification of Demand
from .aid import aid, aid_cat, AidResult, AidCatResult

# Smoothers
from .smoothers import lowess, supsmu

# Association / correlation
from .association import pcor, mcor, association, determination

# Half-moment measures
from .hm import hm, ham, asymmetry, extremity, cextremity, mre

# Point accuracy measures
from .point_measures import (
    measures,
    me,
    mae,
    mse,
    rmse,
    mpe,
    mape,
    mase,
    rmsse,
    same,
    rmae,
    rrmse,
    rame,
    smse,
    spis,
    sce,
    gmrae,
)

# Quantile / interval scoring measures
from .quantile_measures import pinball, mis, smis, rmis

# Diagnostics
from .diagnostics import outlier_dummy, OutlierResult

__all__ = [
    "__version__",
    # Submodules
    "data",
    "distributions",
    "point_measures",
    "quantile_measures",
    "rolling",
    "smoothers",
    # Core fitting
    "ALM",
    "PredictionResult",
    # Formula
    "formula",
    "expand_formula",
    # Selection
    "stepwise",
    "CALM",
    "CALMResult",
    # Variable processing
    "B",
    "multipliers",
    "xreg_expander",
    "xreg_multiplier",
    "xreg_transformer",
    "temporal_dummy",
    # Transforms
    "bc_transform",
    "bc_transform_inv",
    "mean_fast",
    # Rolling
    "rolling_origin",
    "RollingOriginResult",
    # Datasets
    "mtcars",
    # Point likelihoods
    "point_lik",
    "point_lik_cumulative",
    # AID
    "aid",
    "aid_cat",
    "AidResult",
    "AidCatResult",
    # Smoothers
    "lowess",
    "supsmu",
    # Association
    "pcor",
    "mcor",
    "association",
    "determination",
    # Half-moments
    "hm",
    "ham",
    "asymmetry",
    "extremity",
    "cextremity",
    "mre",
    # Accuracy measures
    "measures",
    "me",
    "mae",
    "mse",
    "rmse",
    "mpe",
    "mape",
    "mase",
    "rmsse",
    "same",
    "rmae",
    "rrmse",
    "rame",
    "smse",
    "spis",
    "sce",
    "gmrae",
    # Quantile measures
    "pinball",
    "mis",
    "smis",
    "rmis",
    # Diagnostics
    "outlier_dummy",
    "OutlierResult",
] + list(_distribution_names)
