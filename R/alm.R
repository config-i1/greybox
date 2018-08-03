#' Advanced Linear Model
#'
#' Function estimates model based on the selected distribution
#'
#' This is a function, similar to \link[stats]{lm}, but for the cases of several
#' non-normal distributions. These include:
#'
#' \enumerate{
#' \item Laplace distribution,
#' \item S-distribution,
#' \item Folded-normal distribution.
#' }
#'
#' Some other distributions can be used as well, given that they are formulated
#' in terms of mean and some other parameters (NOT YET IMPLEMENTED).
#'
#' The estimation is done using likelihood of respective distributions.
#'
#' @template author
#' @template keywords
#'
#' @param formula an object of class "formula" (or one that can be coerced to
#' that class): a symbolic description of the model to be fitted.
#' @param data a data frame or a matrix, containing the variables in the model.
#' @param subset an optional vector specifying a subset of observations to be
#' used in the fitting process.
#' @param na.rm	if \code{TRUE}, then observations with missing values are
#' removed. Otherwise they are interpolated.
#'
#' @return Function returns \code{model} - the final model of the class
#' "alm".
#'
#' @seealso \code{\link[greybox]{stepwise}, \link[greybox]{lmCombine}}
#'
#' @examples
#'
#'
#' @export alm
alm <- function(formula, data, subset=NULL, na.rm=TRUE,
                distribution=c("dlaplace","ds","dfnorm"), ...){
    print("This function is under development right now...");
}
