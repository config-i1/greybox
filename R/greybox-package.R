#' Grey box
#'
#' Toolbox for working with multivariate models for purposes of analysis and forecasting
#'
#' \tabular{ll}{ Package: \tab greybox\cr Type: \tab Package\cr Date: \tab
#' 2018-02-13 - Inf\cr License: \tab GPL-2 \cr } The following functions are
#' included in the package:
#' \itemize{
#' \item \link[greybox]{AICc} and \link[greybox]{BICc} - AIC / BIC corrected for the
#' sample size.
#' \item \link[greybox]{pointLik} - point likelihood of the function.
#' \item \link[greybox]{pAIC}, \link[greybox]{pAICc}, \link[greybox]{pBIC},
#' \link[greybox]{pBICc} - point versions of respective information criteria.
#' \item \link[greybox]{determination} - Coefficients of determination between different
#' exogenous variables.
#' \item \link[greybox]{temporaldummy} - Matrix with seasonal dummy variables.
#' \item \link[greybox]{outlierdummy} - Matrix with dummies for outliers.
#' \item \link[greybox]{alm} - Advanced Linear Model - regression, estimated using
#' likelihood with specified distribution (e.g. Laplace or Chi-Squared).
#' \item \link[greybox]{stepwise} - Stepwise based on information criteria and partial
#' correlations. Efficient and fast.
#' \item \link[greybox]{xregExpander} - Function that expands the provided data into
#' the data with lags and leads.
#' \item \link[greybox]{xregTransformer} - Function produces mathematical transformations
#' of the variables, such as taking logarithms, square roots etc.
#' \item \link[greybox]{xregMultiplier} - Function produces cross-products of the
#' matrix of the provided variables.
#' \item \link[greybox]{lmCombine} - Function combines lm models from the estimated
#' based on information criteria weights.
#' \item \link[greybox]{lmDynamic} - Dynamic regression based on point AIC.
#' \item \link[greybox]{ro} - Rolling origin evaluation.
#' \item \link[greybox]{qlaplace}, \link[greybox]{dlaplace},
#' \link[greybox]{plaplace}, \link[greybox]{rlaplace} - Laplace distribution and the
#' respective functions.
#' \item \link[greybox]{qalaplace}, \link[greybox]{dalaplace},
#' \link[greybox]{palaplace}, \link[greybox]{ralaplace} - Asymmetric Laplace distribution and the
#' respective functions.
#' \item \link[greybox]{qfnorm}, \link[greybox]{dfnorm},
#' \link[greybox]{pfnorm}, \link[greybox]{rfnorm} - Folded normal distribution and the
#' respective functions.
#' \item \link[greybox]{qs}, \link[greybox]{ds}, \link[greybox]{ps},
#' \link[greybox]{rs} - S distribution and the respective functions.
#' \item \link[greybox]{qtplnorm}, \link[greybox]{dtplnorm},
#' \link[greybox]{ptplnorm}, \link[greybox]{rtplnorm} - Three parameter log normal
#' distribution and the respective functions.
#' \item \link[greybox]{qbcnorm}, \link[greybox]{dbcnorm},
#' \link[greybox]{pbcnorm}, \link[greybox]{rbcnorm} - Box-Cox normal distribution and
#' the respective functions.
#' \item \link[greybox]{qtplnorm}, \link[greybox]{dtplnorm},
#' \link[greybox]{ptplnorm}, \link[greybox]{rtplnorm} - Three parameter log normal distribution and
#' the respective functions.
#' \item \link[greybox]{spread} - function that produces scatterplots / boxplots / tableplots,
#' depending on the types of variables.
#' \item \link[greybox]{assoc} - function that calculates measures of association,
#' depending on the types of variables.
#' }
#'
#' @name greybox
#' @docType package
#' @author Ivan Svetunkov, \email{ivan@svetunkov.ru}
#'
#' Maintainer: Ivan Svetunkov
#' @seealso \code{\link[greybox]{stepwise}, \link[greybox]{lmCombine}}
#'
#' @template keywords
#'
#' @examples
#'
#' \dontrun{
#' xreg <- cbind(rnorm(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rnorm(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#'
#' stepwise(xreg)
#'}
#'
#' @importFrom graphics abline layout legend lines par points polygon plot
#' @importFrom stats AIC BIC logLik cov deltat end frequency is.ts cor start time ts var lm as.formula residuals
#' @importFrom utils packageVersion
NULL



