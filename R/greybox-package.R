#' Grey box
#'
#' Toolbox for working with multivariate models for purposes of analysis and forecasting
#'
#' \tabular{ll}{ Package: \tab greybox\cr Type: \tab Package\cr Date: \tab
#' 2018-02-13 - Inf\cr License: \tab GPL-2 \cr } The following functions are
#' included in the package:
#' \itemize{
#' \item \link[greybox]{stepwise} - Stepwise based on information criteria and partial
#' correlations. Efficient and fast.
#' \item \link[greybox]{xregExpander} - Function that expands the provided data into
#' the data with lags and leads.
#' }
#'
#' @name greybox
#' @docType package
#' @author Ivan Svetunkov
#'
#' Maintainer: Ivan Svetunkov <ivan@svetunkov.ru>
#' @seealso \code{\link[smooth]{es}, \link[smooth]{ssarima}}
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
#' @importFrom graphics abline layout legend lines par points polygon
#' @importFrom stats AIC BIC logLik cov deltat end frequency is.ts coef cor start time ts var lm as.formula residuals qt vcov
#' @importFrom utils packageVersion
#' @importFrom smooth es AICc iss modelType graphmaker
NULL



