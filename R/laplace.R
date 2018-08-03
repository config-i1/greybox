#' Laplace Distribution
#'
#' Density, cumulative distribution, quantile functions and random
#' generation for the Laplace distribution with the location parameter mu
#' and Mean Absolute Error (or Mean Absolute Deviation) equal to b.
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
#' \item \code{dlaplace} returns the density function value for the
#' provided parameters.
#' \item \code{plaplace} returns the value of the cumulative function
#' for the provided parameters.
#' \item \code{qlaplace} returns quantiles of the distribution. Depending
#' on what was provided in \code{p}, \code{mu} and \code{b}, this
#' can be either a vector or a matrix, or an array.
#' \item \code{rlaplace} returns a vector of random variables
#' generated from the Laplace distribution. Depending on what was
#' provided in \code{mu} and \code{b}, this can be either a vector
#' or a matrix or an array.
#' }
#'
#' @examples
#' x <- dlaplace(c(-1000:1000)/10, 0, 1)
#' plot(x, type="l")
#'
#' x <- plaplace(c(-1000:1000)/10, 0, 1)
#' plot(x, type="l")
#'
#' qlaplace(c(0.025,0.975), 0, c(1,2))
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
    laplaceReturn <- array(0,c(length(p),length(mu),length(b)),
                        dimnames=list(paste0("p=",p),paste0("mu=",mu),paste0("b=",b)));
    laplaceReturn[p==0.5,,] <- 1;
    laplaceReturn[p==0,,] <- -Inf;
    laplaceReturn[p==1,,] <- Inf;
    probsToEstimate <- which(laplaceReturn[,1,1]==0);
    laplaceReturn[laplaceReturn==1] <- 0;
    for(k in 1:length(b)){
        laplaceReturn[probsToEstimate,,k] <- (-b[k] * sign(p[probsToEstimate]-0.5) *
                                                  log(1-2*abs(p[probsToEstimate]-0.5)));
    }
    laplaceReturn <- laplaceReturn + rep(mu,each=length(p));
    if(any(dim(laplaceReturn)==1)){
        if(dim(laplaceReturn)[1]==1){
            laplaceReturn <- laplaceReturn[1,,];
        }
        else if(dim(laplaceReturn)[2]==1){
            laplaceReturn <- laplaceReturn[,1,];
        }
        else if(dim(laplaceReturn)[3]==1){
            laplaceReturn <- laplaceReturn[,,1];
        }

        if(any(dim(laplaceReturn)==1)){
            laplaceReturn <- c(laplaceReturn);
        }
    }
    return(laplaceReturn);
}

#' @rdname laplace-distribution
#' @export rlaplace
#' @aliases rlaplace
rlaplace <- function(n=1, mu=0, b=1){
    laplaceReturn <- qlaplace(runif(n,0,1),mu,b);
    return(laplaceReturn);
}
