"""Distribution functions for greybox.

This module contains implementations of various distributions
not available in scipy.stats, plus wrappers around scipy.stats.
"""

from .gnorm import dgnorm, pgnorm, qgnorm, rgnorm
from .s import ds, ps, qs, rs
from .alaplace import dalaplace, palaplace, qalaplace, ralaplace
from .bcnorm import dbcnorm, pbcnorm, qbcnorm, rbcnorm
from .fnorm import dfnorm, pfnorm, qfnorm, rfnorm
from .rectnorm import drectnorm, prectnorm, qrectnorm, rrectnorm
from .logitnorm import dlogitnorm, plogitnorm, qlogitnorm, rlogitnorm
from .laplace import dlaplace, plaplace, qlaplace, rlaplace
from .logis import dlogis, qlogis, rlogis
from .t import dt, pt, qt, rt
from .lnorm import dlnorm, plnorm, qlnorm
from .llaplace import dllaplace, qllaplace
from .ls import dls, qls
from .lgnorm import dlgnorm, qlgnorm
from .invgauss import dinvgauss, pinvgauss, qinvgauss, rinvgauss
from .gamma import dgamma, pgamma, qgamma, rgamma
from .exp import dexp, pexp, qexp, rexp
from .beta import dbeta, pbeta, qbeta, rbeta
from .pois import dpois, ppois, qpois, rpois
from .nbinom import dnbinom, pnbinom, qnbinom, rnbinom
from .binom import dbinom, pbinom, qbinom, rbinom
from .geom import dgeom, pgeom, qgeom, rgeom
from .chi2 import dchi2, pchi2, qchi2, rchi2

from .helper import plogis as plogis_helper
from .helper import pnorm as pnorm_helper
from .helper import qnorm as qnorm_helper
from .helper import dnorm as dnorm_helper

plogis = plogis_helper
pnorm = pnorm_helper
qnorm = qnorm_helper
dnorm = dnorm_helper

__all__ = [
    "dgnorm",
    "pgnorm",
    "qgnorm",
    "rgnorm",
    "ds",
    "ps",
    "qs",
    "rs",
    "dalaplace",
    "palaplace",
    "qalaplace",
    "ralaplace",
    "dbcnorm",
    "pbcnorm",
    "qbcnorm",
    "rbcnorm",
    "dfnorm",
    "pfnorm",
    "qfnorm",
    "rfnorm",
    "drectnorm",
    "prectnorm",
    "qrectnorm",
    "rrectnorm",
    "dlogitnorm",
    "plogitnorm",
    "qlogitnorm",
    "rlogitnorm",
    "dlaplace",
    "plaplace",
    "qlaplace",
    "rlaplace",
    "dlogis",
    "plogis",
    "qlogis",
    "rlogis",
    "dt",
    "pt",
    "qt",
    "rt",
    "dlnorm",
    "plnorm",
    "qlnorm",
    "dllaplace",
    "qllaplace",
    "dls",
    "qls",
    "dlgnorm",
    "qlgnorm",
    "dinvgauss",
    "pinvgauss",
    "qinvgauss",
    "rinvgauss",
    "dgamma",
    "pgamma",
    "qgamma",
    "rgamma",
    "dexp",
    "pexp",
    "qexp",
    "rexp",
    "dbeta",
    "pbeta",
    "qbeta",
    "rbeta",
    "dpois",
    "ppois",
    "qpois",
    "rpois",
    "dnbinom",
    "pnbinom",
    "qnbinom",
    "rnbinom",
    "dbinom",
    "pbinom",
    "qbinom",
    "rbinom",
    "dgeom",
    "pgeom",
    "qgeom",
    "rgeom",
    "dchi2",
    "pchi2",
    "qchi2",
    "rchi2",
    "plogis",
    "pnorm",
    "qnorm",
    "dnorm",
]
