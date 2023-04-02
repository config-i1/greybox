#' Distribution functions of the greybox package
#'
#' The greybox package implements several distribution functions. In this document,
#' I list the main functions and provide links to related resources.
#'
#' \itemize{
#' \item Generalised normal distribution (with a kurtosis parameter by Nadarajah, 2005):
#' \link[greybox]{qgnorm}, \link[greybox]{dgnorm},
#' \link[greybox]{pgnorm}, \link[greybox]{rgnorm}.
#' \item S distribution (a special case of Generalised Normal with shape=0.5):
#' \link[greybox]{qs}, \link[greybox]{ds},
#' \link[greybox]{ps}, \link[greybox]{rs}.
#' \item Laplace distribution (special case of Generalised Normal with shape=1):
#' \link[greybox]{qlaplace}, \link[greybox]{dlaplace},
#' \link[greybox]{plaplace}, \link[greybox]{rlaplace}.
#' \item Asymmetric Laplace distribution (Yu & Zhang, 2005):
#' \link[greybox]{qalaplace}, \link[greybox]{dalaplace},
#' \link[greybox]{palaplace}, \link[greybox]{ralaplace}.
#' \item Logit Normal distribution (Mead, 1965):
#' \link[greybox]{qlogitnorm}, \link[greybox]{dlogitnorm},
#' \link[greybox]{plogitnorm}, \link[greybox]{rlogitnorm}.
#' \item Box-Cox Normal distribution (Box & Cox, 1964):
#' \link[greybox]{qbcnorm}, \link[greybox]{dbcnorm},
#' \link[greybox]{pbcnorm}, \link[greybox]{rbcnorm}.
#' \item Folded Normal distribution:
#' \link[greybox]{qfnorm}, \link[greybox]{dfnorm},
#' \link[greybox]{pfnorm}, \link[greybox]{rfnorm}.
#' \item Rectified Normal distribution:
#' \link[greybox]{qrectnorm}, \link[greybox]{drectnorm},
#' \link[greybox]{prectnorm}, \link[greybox]{rrectnorm}.
#' \item Three parameter Log Normal distribution (Sangal & Biswas, 1970):
#' \link[greybox]{qtplnorm}, \link[greybox]{dtplnorm},
#' \link[greybox]{ptplnorm}, \link[greybox]{rtplnorm}.
#' }
#'
#' @name Distributions
#' @docType package
#' @author Ivan Svetunkov, \email{ivan@svetunkov.ru}
#'
#' @references
#' \itemize{
#' \item Nadarajah, Saralees. "A generalized normal distribution." Journal of
#' Applied Statistics 32.7 (2005): 685-694.
#' \item Wikipedia page on Laplace distribution:
#' \url{https://en.wikipedia.org/wiki/Laplace_distribution}.
#' \item Yu, K., & Zhang, J. (2005). A three-parameter asymmetric
#' laplace distribution and its extension. Communications in Statistics
#' - Theory and Methods, 34, 1867-1879.
#' \doi{10.1080/03610920500199018}
#' \item Mead, R. (1965). A Generalised Logit-Normal Distribution.
#' Biometrics, 21 (3), 721–732. doi: 10.2307/2528553
#' \item Box, G. E., & Cox, D. R. (1964). An Analysis of Transformations.
#' Journal of the Royal Statistical Society. Series B (Methodological),
#' 26(2), 211–252. Retrieved from https://www.jstor.org/stable/2984418
#' \item Sangal, B. P., & Biswas, A. K. (1970). The 3-Parameter
#' Distribution Applications in Hydrology. Water Resources Research,
#' 6(2), 505–515. \doi{10.1029/WR006i002p00505}
#' }
#'
#' @template author
#' @keywords distribution
#'
#' @seealso \code{\link[stats]{Distributions}} from the stats package.
#'
NULL



