#' Laplace Distribution
#'
#' Density, cumulative distribution, quantile functions and random
#' generation for the Laplace distribution with the location parameter mu
#' and Mean Absolute Error (or Mean Aboslute Deviation) equal to b.
#'
#' When mu=0 and b=1, the Laplace distribution becomes standardized with.
#' The distribution has the following density function:
#'
#' f(x) = 1/(2b) exp(-abs(x-mu) / b)
#'
#' @template author
#' @keywords distribution
#'
#' @param q vector of quantiles.
#' @param p vector of probabilities.
#' @param n number of observations. Should be a single number.
#' @param mu vector of location parameters (means).
#' @param b vector of mean absolute errors.
#' @param log if \code{TRUE}, then probabilities are returned in
#' logarithms.
#'
#' @return Depending on the function, various things are returned
#' (usually either vector or scalar):
#' \itemize{
#' \item \code{ds} returns the density function value for the
#' provided parameters.
#' \item \code{ps} returns the value of the cumulative function
#' for the provided parameters.
#' \item \code{qs} returns quantiles of the distribution.
#' \item \code{rs} returns a vector of random variables
#' generated from the Laplace distribution.
#' }
#'
#' @examples
#' x <- dlaplace(c(-1000:1000)/10, 0, 1)
#' plot(x, type="l")
#'
#' x <- plaplace(c(-1000:1000)/10, 0, 1)
#' plot(x, type="l")
#'
#' qlaplace(c(0.025,0.975), 0, 1)
#'
#' x <- rlaplace(1000, 0, 1)
#' hist(x)
#'
#' @references \itemize{
#' \item Wikipedia page on Laplace distribution:
#' \url{https://en.wikipedia.org/wiki/Laplace_distribution}.
#' }
#'
#' @rdname laplace-distribution

#' @rdname laplace-distribution
#' @export dlaplace
#' @aliases dlaplace
dlaplace <- function(q, mu=0, b=1, log=FALSE){
    laplaceReturn <- 1/(2*b)*exp(-abs(mu-q)/b);
    if(log){
        laplaceReturn <- log(laplaceReturn);
    }
    return(laplaceReturn);
}

#' @rdname laplace-distribution
#' @export plaplace
#' @aliases plaplace
plaplace <- function(q, mu=0, b=1){
    laplaceReturn <- 0.5 + 0.5*sign(q-mu)*(1-exp(-abs(q-mu)/b));
    return(laplaceReturn);
}

#' @rdname laplace-distribution
#' @export qlaplace
#' @aliases qlaplace
qlaplace <- function(p, mu=0, b=1){
    laplaceReturn <- (p==0.5)*1;
    laplaceReturn[p==0] <- rep(-Inf,sum(p==0));
    laplaceReturn[p==1] <- rep(Inf,sum(p==1));
    probsToEstimate <- which(laplaceReturn==0);
    laplaceReturn[laplaceReturn==1] <- 0;
    laplaceReturn[probsToEstimate] <- (mu - b * sign(p[probsToEstimate]-0.5) *
                                           log(1-2*abs(p[probsToEstimate]-0.5)));
    return(laplaceReturn);
}

#' @rdname laplace-distribution
#' @export rlaplace
#' @aliases rlaplace
rlaplace <- function(n=1, mu=0, b=1){
    laplaceReturn <- qlaplace(runif(n,0,1),mu,b);
    return(laplaceReturn);
}
