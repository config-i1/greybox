#' S Distribution
#'
#' Density, cumulative distribution, quantile functions and random
#' generation for the S distribution with the location parameter mu
#' and a scaling parameter b.
#'
#' When mu=0 and ham=2, the S distribution becomes standardized with
#' b=1 (this is because b=ham/2). The distribution has the following
#' density function:
#'
#' f(x) = 1/(4b^2) exp(-sqrt(abs(x-mu)) / b)
#'
#' The S distribution has fat tails and large excess.
#'
#' @template author
#' @keywords distribution
#'
#' @param q vector of quantiles.
#' @param p vector of probabilities.
#' @param n number of observations. Should be a single number.
#' @param mu vector of location parameters (means).
#' @param b vector of scaling parameter (which equals to ham/2).
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
#' \item \code{qs} returns quantiles of the distribution. Depending
#' on what was provided in \code{p}, \code{mu} and \code{b}, this
#' can be either a vector or a matrix, or an array.
#' \item \code{rs} returns a vector of random variables
#' generated from the S distribution. Depending on what was provided
#' in \code{mu} and \code{b}, this can be either a vector or a matrix
#' or an array.
#' }
#'
#' @examples
#' x <- ds(c(-1000:1000)/10, 0, 1)
#' plot(x, type="l")
#'
#' x <- ps(c(-1000:1000)/10, 0, 1)
#' plot(x, type="l")
#'
#' qs(c(0.025,0.975), 0, 1)
#'
#' x <- rs(1000, 0, 1)
#' hist(x)
#'
#' @rdname s-distribution
#' @importFrom stats optimize runif
#'
#' @rdname s-distribution
#' @export ds
#' @aliases ds
ds <- function(q, mu=0, b=1, log=FALSE){
    svetReturn <- 1/(4*b^2)*exp(-sqrt(abs(mu-q))/b);
    if(log){
        svetReturn <- log(svetReturn);
    }
    return(svetReturn);
}

#' @rdname s-distribution
#' @export ps
#' @aliases ps
ps <- function(q, mu=0, b=1){
    svetReturn <- 0.5+0.5*sign(q-mu)*(1-1/b*(sqrt(abs(mu-q))+b)*exp(-sqrt(abs(mu-q))/b))
    return(svetReturn);
}

#' @rdname s-distribution
#' @export qs
#' @importFrom lamW lambertWm1
#' @aliases qs
qs <- function(p, mu=0, b=1){
    svetReturn <- array(0,c(length(p),length(mu),length(b)),
                        dimnames=list(paste0("p=",p),paste0("mu=",mu),paste0("b=",b)));
    svetReturn[p==0.5,,] <- 1;
    svetReturn[p==0,,] <- -Inf;
    svetReturn[p==1,,] <- Inf;
    probsToEstimate <- which(svetReturn[,1,1]==0);
    svetReturn[svetReturn==1] <- 0;
    if(length(probsToEstimate)!=0){
        for(i in 1:length(probsToEstimate)){
            j <- probsToEstimate[i];
            for(k in 1:length(b)){
                if(p[j]<0.5){
                    svetReturn[j,,k] <- (-b[k]^2*lambertWm1(-2*p[j]/exp(1))^2 -
                                             2*b[k]^2*lambertWm1(-2*p[j]/exp(1))-b[k]^2);
                }
                else{
                    svetReturn[j,,k] <- (b[k]^2*lambertWm1(2*(p[j]-1)/exp(1))^2 +
                                             2*b[k]^2*lambertWm1(2*(p[j]-1)/exp(1))+b[k]^2);
                }
            }
        }
        svetReturn <- svetReturn + rep(mu,each=length(p));
    }
    if(any(dim(svetReturn)==1)){
        if(dim(svetReturn)[1]==1){
            svetReturn <- svetReturn[1,,];
        }
        else if(dim(svetReturn)[2]==1){
            svetReturn <- svetReturn[,1,];
        }
        else if(dim(svetReturn)[3]==1){
            svetReturn <- svetReturn[,,1];
        }

        if(any(dim(svetReturn)==1)){
            svetReturn <- c(svetReturn);
        }
    }
    return(svetReturn);
}

#' @rdname s-distribution
#' @export rs
#' @aliases rs
rs <- function(n=1, mu=0, b=1){
    svetReturn <- qs(runif(n,0,1),mu,b);
    return(svetReturn);
}
