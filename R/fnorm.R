#' Folded Normal Distribution
#'
#' Density, cumulative distribution, quantile functions and random number
#' generation for the folded normal distribution with the location
#' parameter mu and the scale sigma (which corresponds to standard
#' deviation in normal distribution).
#'
#' The distribution has the following density function:
#'
#' f(x) = 1/sqrt(2 pi) (exp(-(x-mu)^2 / (2 sigma^2)) + exp(-(x+mu)^2 / (2 sigma^2)))
#'
#' Both \code{pfnorm} and \code{qfnorm} are returned for the lower
#' tail of the distribution.
#'
#' @template author
#' @keywords distribution
#'
#' @param q vector of quantiles.
#' @param p vector of probabilities.
#' @param n number of observations. Should be a single number.
#' @param mu vector of location parameters (means).
#' @param sigma vector of scale parameters.
#' @param log if \code{TRUE}, then probabilities are returned in
#' logarithms.
#'
#' @return Depending on the function, various things are returned
#' (usually either vector or scalar):
#' \itemize{
#' \item \code{dfnorm} returns the density function value for the
#' provided parameters.
#' \item \code{pfnorm} returns the value of the cumulative function
#' for the provided parameters.
#' \item \code{qfnorm} returns quantiles of the distribution. Depending
#' on what was provided in \code{p}, \code{mu} and \code{sigma}, this
#' can be either a vector or a matrix, or an array.
#' \item \code{rfnorm} returns a vector of random variables
#' generated from the fnorm distribution. Depending on what was
#' provided in \code{mu} and \code{sigma}, this can be either a vector
#' or a matrix or an array.
#' }
#'
#' @examples
#' x <- dfnorm(c(-1000:1000)/200, 0, 1)
#' plot(x, type="l")
#'
#' x <- pfnorm(c(-1000:1000)/200, 0, 1)
#' plot(x, type="l")
#'
#' qfnorm(c(0.025,0.975), 0, c(1,2))
#'
#' x <- rfnorm(1000, 0, 1)
#' hist(x)
#'
#' @references \itemize{
#' \item Wikipedia page on folded normal distribution:
#' \url{https://en.wikipedia.org/wiki/Folded_normal_distribution}.
#' }
#'
#' @rdname fnorm-distribution
#' @importFrom stats optim pnorm qnorm rnorm
#' @export dfnorm
dfnorm <- function(q, mu=0, sigma=1, log=FALSE){
    fnormReturn <- 1/(sqrt(2 * pi)*sigma) * (exp(-(q-mu)^2 / (2 * sigma^2)) +
                                                 exp(-(q+mu)^2 / (2 * sigma^2)));
    if(log){
        fnormReturn[] <- log(fnormReturn);
    }
    fnormReturn[q<0] <- 0;

    return(fnormReturn);
}

#' @rdname fnorm-distribution
#' @export pfnorm
#' @aliases pfnorm
pfnorm <- function(q, mu=0, sigma=1){
    fnormReturn <- pnorm(q, mu, sigma) + pnorm(q, -mu, sigma) - 1;
    fnormReturn[q<0] <- 0;
    return(fnormReturn);
}

#' @rdname fnorm-distribution
#' @export qfnorm
#' @aliases qfnorm
qfnorm <- function(p, mu=0, sigma=1){
    # p <- unique(p);
    # mu <- unique(mu);
    # sigma <- unique(sigma);
    CF <- function(A, P, mu, sigma){
        probability <- pfnorm(A, mu, sigma);
        return((P-probability)^2);
    }

    lengthMax <- max(length(p),length(mu),length(sigma));
    # If length of p, mu and sigma differs, then go difficult. Otherwise do simple stuff
    if(any(!c(length(p),length(mu),length(sigma)) %in% c(lengthMax, 1))){
        fnormReturn <- array(0,c(length(p),length(mu),length(sigma)),
                             dimnames=list(paste0("p=",p),paste0("mu=",mu),paste0("sigma=",sigma)));
        fnormReturn[p==0,,] <- 1;
        fnormReturn[p==1,,] <- Inf;
        probsToEstimate <- which(fnormReturn[,1,1]==0);
        fnormReturn[fnormReturn==1] <- 0;

        for(j in probsToEstimate){
            for(k in 1:length(sigma)){
                for(i in 1:length(mu)){
                    fnormReturn[j,i,k] <- optim(abs(qnorm(p[j],mu[i],sigma[k])), CF, method="L-BFGS-B", lower=0,
                                                P=p[j], mu=mu[i], sigma=sigma[k])$par;
                }
            }
        }

        # Drop the redundant dimensions
        fnormReturn <- fnormReturn[,,];
    }
    else{
        fnormReturn <- rep(0,lengthMax);
        p <- rep(p,lengthMax)[1:lengthMax];
        mu <- rep(mu,lengthMax)[1:lengthMax];
        sigma <- rep(sigma,lengthMax)[1:lengthMax];

        for(i in 1:lengthMax){
            if(p[i]==0){
                fnormReturn[i] <- 0;
            }
            else if(p[i]==1){
                fnormReturn[i] <- Inf;
            }
            else{
                fnormReturn[i] <- optim(abs(qnorm(p[i],mu[i],sigma[i])), CF, method="L-BFGS-B", lower=0,
                                        P=p[i], mu=mu[i], sigma=sigma[i])$par;
            }
        }
    }
    return(fnormReturn);
}

#' @rdname fnorm-distribution
#' @export rfnorm
#' @aliases rfnorm
rfnorm <- function(n=1, mu=0, sigma=1){
    fnormReturn <- abs(rnorm(n,mu,sigma));
    return(fnormReturn);
}
