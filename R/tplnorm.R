#' Three Parameter Log Normal Distribution
#'
#' Density, cumulative distribution, quantile functions and random number
#' generation for the 3 parameter log normal distribution with the location
#' parameter mu, scale sigma (which corresponds to standard deviation in normal
#' distribution) and shifting parameter shift.
#'
#' The distribution has the following density function:
#'
#' f(x) = 1/(x-a) 1/sqrt(2 pi) exp(-(log(x-a)-mu)^2 / (2 sigma^2))
#'
#' Both \code{ptplnorm} and \code{qtplnorm} are returned for the lower
#' tail of the distribution.
#'
#' The function is based on the lnorm functions from stats package, introducing
#' the shift parameter.
#'
#' @template author
#' @keywords distribution
#'
#' @param q vector of quantiles.
#' @param p vector of probabilities.
#' @param n number of observations. Should be a single number.
#' @param mu vector of location parameters (means).
#' @param sigma vector of scale parameters.
#' @param shift vector of shift parameters.
#' @param log if \code{TRUE}, then probabilities are returned in
#' logarithms.
#'
#' @return Depending on the function, various things are returned
#' (usually either vector or scalar):
#' \itemize{
#' \item \code{dtplnorm} returns the density function value for the
#' provided parameters.
#' \item \code{ptplnorm} returns the value of the cumulative function
#' for the provided parameters.
#' \item \code{qtplnorm} returns quantiles of the distribution. Depending
#' on what was provided in \code{p}, \code{mu} and \code{sigma}, this
#' can be either a vector or a matrix, or an array.
#' \item \code{rtplnorm} returns a vector of random variables
#' generated from the tplnorm distribution. Depending on what was
#' provided in \code{mu} and \code{sigma}, this can be either a vector
#' or a matrix or an array.
#' }
#'
#' @examples
#' x <- dtplnorm(c(-1000:1000)/200, 0, 1, 1)
#' plot(c(-1000:1000)/200, x, type="l")
#'
#' x <- ptplnorm(c(-1000:1000)/200, 0, 1, 1)
#' plot(c(-1000:1000)/200, x, type="l")
#'
#' qtplnorm(c(0.025,0.975), 0, c(1,2), 1)
#'
#' x <- rtplnorm(1000, 0, 1, 1)
#' hist(x)
#'
#' @references \itemize{
#' \item Sangal, B. P., & Biswas, A. K. (1970). The 3-Parameter
#' Distribution Applications in Hydrology. Water Resources Research,
#' 6(2), 505–515. \url{https://doi.org/10.1029/WR006i002p00505}
#' }
#'
#' @rdname tplnorm-distribution
#' @importFrom stats plnorm qlnorm rlnorm
#' @export dtplnorm
dtplnorm <- function(q, mu=0, sigma=1, shift=0, log=FALSE){
    tplnormReturn <- dlnorm(x=q-shift, meanlog=mu, sdlog=sigma, log=log);
    tplnormReturn[q<shift] <- 0;

    return(tplnormReturn);
}

#' @rdname tplnorm-distribution
#' @export ptplnorm
#' @aliases ptplnorm
ptplnorm <- function(q, mu=0, sigma=1, shift=0){
    tplnormReturn <- plnorm(q=q-shift, meanlog=mu, sdlog=sigma);
    tplnormReturn[q<shift] <- 0;
    return(tplnormReturn);
}

#' @rdname tplnorm-distribution
#' @export qtplnorm
#' @aliases qtplnorm
qtplnorm <- function(p, mu=0, sigma=1, shift=0){
    tplnormReturn <- qlnorm(p=p, meanlog=mu, sdlog=sigma) + shift;
    return(tplnormReturn);
}

#' @rdname tplnorm-distribution
#' @export rtplnorm
#' @aliases rtplnorm
rtplnorm <- function(n=1, mu=0, sigma=1, shift=0){
    tplnormReturn <- rlnorm(n=n, meanlog=mu, sdlog=sigma) + shift;
    return(tplnormReturn);
}
