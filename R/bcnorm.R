#' Box-Cox Normal Distribution
#'
#' Density, cumulative distribution, quantile functions and random number
#' generation for the distribution that becomes normal after the Box-Cox
#' transformation. Note that this is based on the original Box-Cox paper.
#'
#' The distribution has the following density function:
#'
#' f(y) = y^{lambda-1} 1/sqrt(2 pi) exp(-((y^lambda-1)/lambda -mu)^2 / (2 sigma^2))
#'
#' Both \code{pbcnorm} and \code{qbcnorm} are returned for the lower
#' tail of the distribution.
#'
#' In case of lambda=0, the values of the log normal distribution are returned.
#' In case of lambda=1, the values of the normal distribution are returned with mu=mu+1.
#'
#' All the functions are defined for non-negative values only.
#'
#' @template author
#' @keywords distribution
#'
#' @param q vector of quantiles.
#' @param p vector of probabilities.
#' @param n number of observations. Should be a single number.
#' @param mu vector of location parameters (means).
#' @param sigma vector of scale parameters.
#' @param lambda the value of the Box-Cox transform parameter.
#' @param log if \code{TRUE}, then probabilities are returned in
#' logarithms.
#'
#' @return Depending on the function, various things are returned
#' (usually either vector or scalar):
#' \itemize{
#' \item \code{dbcnorm} returns the density function value for the
#' provided parameters.
#' \item \code{pbcnorm} returns the value of the cumulative function
#' for the provided parameters.
#' \item \code{qbcnorm} returns quantiles of the distribution. Depending
#' on what was provided in \code{p}, \code{mu} and \code{sigma}, this
#' can be either a vector or a matrix, or an array.
#' \item \code{rbcnorm} returns a vector of random variables
#' generated from the bcnorm distribution. Depending on what was
#' provided in \code{mu} and \code{sigma}, this can be either a vector
#' or a matrix or an array.
#' }
#'
#' @examples
#' x <- dbcnorm(c(-1000:1000)/200, 0, 1, 1)
#' plot(c(-1000:1000)/200, x, type="l")
#'
#' x <- pbcnorm(c(-1000:1000)/200, 0, 1, 1)
#' plot(c(-1000:1000)/200, x, type="l")
#'
#' qbcnorm(c(0.025,0.975), 0, c(1,2), 1)
#'
#' x <- rbcnorm(1000, 0, 1, 1)
#' hist(x)
#'
#' @references \itemize{
#' \item Box, G. E., & Cox, D. R. (1964). An Analysis of Transformations.
#' Journal of the Royal Statistical Society. Series B (Methodological),
#' 26(2), 211–252. Retrieved from https://www.jstor.org/stable/2984418
#' }
#'
#' @rdname bcnorm-distribution
#' @importFrom stats dnorm pnorm qnorm rnorm
#' @export dbcnorm
dbcnorm <- function(q, mu=0, sigma=1, lambda=0, log=FALSE){
    if(lambda==0){
        bcnormReturn <- dlnorm(x=q, meanlog=mu, sdlog=sigma, log=FALSE);
    }
    else if(lambda==1){
        bcnormReturn <- dnorm(x=q, mean=mu+1, sd=sigma, log=FALSE);
    }
    else{
        bcnormReturn <- suppressWarnings(q^(lambda-1) * 1/(sqrt(2*pi)*sigma) * exp(-((q^lambda-1)/lambda - mu)^2/ (2*sigma^2)));
        bcnormReturn[q<=0] <- 0;
    }

    # Return logs if needed
    if(log){
        bcnormReturn[] <- log(bcnormReturn);
    }

    return(bcnormReturn);
}

#' @rdname bcnorm-distribution
#' @export pbcnorm
#' @aliases pbcnorm
pbcnorm <- function(q, mu=0, sigma=1, lambda=0){
    if(lambda==0){
        bcnormReturn <- plnorm(q=q, meanlog=mu, sdlog=sigma);
    }
    else{
        bcnormReturn <- vector("numeric", length(q));
        bcnormReturn[q>=0] <- pnorm(q=(q[q>=0]^lambda-1)/lambda, mean=mu, sd=sigma);
        bcnormReturn[q<0] <- 0;
    }
    return(bcnormReturn);
}

#' @rdname bcnorm-distribution
#' @export qbcnorm
#' @aliases qbcnorm
qbcnorm <- function(p, mu=0, sigma=1, lambda=0){
    if(lambda==0){
        bcnormReturn <- qlnorm(p=p, meanlog=mu, sdlog=sigma);
    }
    else{
        bcnormReturn <- (qnorm(p=p, mean=mu, sd=sigma)*lambda+1)^(1/lambda);
        bcnormReturn[is.nan(bcnormReturn)] <- 0;
    }
    return(bcnormReturn);
}

#' @rdname bcnorm-distribution
#' @export rbcnorm
#' @aliases rbcnorm
rbcnorm <- function(n=1, mu=0, sigma=1, lambda=0){
    if(lambda==0){
        bcnormReturn <- rlnorm(n=n, meanlog=mu, sdlog=sigma);
    }
    else{
        bcnormReturn <- qbcnorm(runif(n=n, 0, 1), mu=mu, sigma=sigma, lambda=lambda);
    }
    return(bcnormReturn);
}
