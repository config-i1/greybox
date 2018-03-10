#' S Distribution
#'
#' Density, cumulative distribution, quantile functions and random
#' generation for the S distribution with the location parameter mu
#' and half absolute moment equal to ham.
#'
#' When mu=0 and ham=2, the S distribution becomes standardized with
#' b=1 (this is because b=ham/2). The distribution has the following
#' density function:
#'
#' f(x) = 1/(4b^2) exp(-sqrt(abs(x-mu)) / b)
#'
#' The S distribution has fat tails and large excess.
#'
#' @param q vector of quantiles.
#' @param p vector of probabilities.
#' @param n number of observations. Should be a single number.
#' @param mu vector of location parameters (means).
#' @param ham vector of half absolute moments.
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
#' generated from the S distribution.
#' }
#'
#' @examples
#' x <- ds(c(-1000:1000)/10, 0, 2)
#' plot(x, type="l")
#'
#' x <- ps(c(-1000:1000)/10, 0, 2)
#' plot(x, type="l")
#'
#' qs(c(0.025,0.975), 0, 2)
#'
#' x <- rs(1000, 0, 2)
#' hist(x)
#'
#' @rdname s-distribution
#' @importFrom stats optimize runif

#' @rdname s-distribution
#' @export ds
#' @aliases ds
ds <- function(q, mu=0, ham=2, log=FALSE){
    b <- ham / 2;
    svetReturn <- 1/(4*b^2)*exp(-sqrt(abs(mu-q))/b);
    if(log){
        svetReturn <- log(svetReturn);
    }
    return(svetReturn);
}

#' @rdname s-distribution
#' @export ps
#' @aliases ps
ps <- function(q, mu=0, ham=2){
    b <- ham / 2;
    svetReturn <- 1/(2*b)*(sqrt(abs(mu-q))+b)*exp(-sqrt(abs(mu-q))/b)*((mu>=q)*1 - (mu<q)*1);
    svetReturn <- svetReturn + (mu<q)*1;
    return(svetReturn);
}

#' @rdname s-distribution
#' @export qs
#' @aliases qs
qs <- function(p, mu=0, ham=2){
    b <- ham / 2;
    cfFunction <- function(q,p,...){
        return(abs(ps(q,...)-p));
    }
    svetReturn <- (p==0.5)*0;
    svetReturn[p==0] <- rep(-Inf,sum(p==0));
    svetReturn[p==1] <- rep(Inf,sum(p==0));
    probsToEstimate <- which(svetReturn==0);
    for(i in 1:length(probsToEstimate)){
        j <- probsToEstimate[i];
        solution <- optimize(cfFunction,c(-100:100),mu=0,ham=2,p=p[j]);
        svetReturn[j] <- solution$minimum;
    }
    svetReturn <- svetReturn * b^2;
    svetReturn <- svetReturn + mu;
    return(svetReturn)
}

#' @rdname s-distribution
#' @export rs
#' @aliases rs
rs <- function(n=1, mu=0, ham=2){
    svetReturn <- rep(NA,n);
    for(i in 1:n){
        svetReturn[i] <- qs(runif(1,0,1),mu,ham);
    }
    return(svetReturn);
}
