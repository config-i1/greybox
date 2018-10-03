#' Asymmetric Laplace Distribution
#'
#' Density, cumulative distribution, quantile functions and random
#' generation for the Asymmetric Laplace distribution with the
#' location parameter mu, Mean Absolute Error (or Mean Absolute
#' Deviation) equal to 2 b and asymmetry parameter alpha.
#'
#' When mu=0 and b=1, the Laplace distribution becomes standardized.
#' The distribution has the following density function:
#'
#' f(x) = alpha (1-alpha) / b exp(-(x-mu)/b (alpha - I(x<=mu))),
#'
#' where I(.) is the indicator function (equal to 1 if the condition is
#' satisfied and zero otherwise).
#'
#' When alpha=0.5, then the distribution becomes Symmetric Laplace, where
#' b = 1/2 MAE.
#'
#' This distribution function aligns with the quantile estimates of
#' parameters (Geraci & Bottai, 2007).
#'
#' Finally, both \code{palaplace} and \code{qalaplace} are returned for
#' the lower tail of the distribution.
#'
#' @template author
#' @keywords distribution
#'
#' @param q vector of quantiles.
#' @param p vector of probabilities.
#' @param n number of observations. Should be a single number.
#' @param mu vector of location parameters (means).
#' @param b vector of scale parameters.
#' @param alpha value of asymmetry parameter. Varies from 0 to 1.
#' @param log if \code{TRUE}, then probabilities are returned in
#' logarithms.
#'
#' @return Depending on the function, various things are returned
#' (usually either vector or scalar):
#' \itemize{
#' \item \code{dalaplace} returns the density function value for the
#' provided parameters.
#' \item \code{palaplace} returns the value of the cumulative function
#' for the provided parameters.
#' \item \code{qalaplace} returns quantiles of the distribution. Depending
#' on what was provided in \code{p}, \code{mu} and \code{b}, this
#' can be either a vector or a matrix, or an array.
#' \item \code{ralaplace} returns a vector of random variables
#' generated from the Laplace distribution. Depending on what was
#' provided in \code{mu} and \code{b}, this can be either a vector
#' or a matrix or an array.
#' }
#'
#' @examples
#' x <- dalaplace(c(-100:100)/10, 0, 1, 0.2)
#' plot(x, type="l")
#'
#' x <- palaplace(c(-100:100)/10, 0, 1, 0.2)
#' plot(x, type="l")
#'
#' qalaplace(c(0.025,0.975), 0, c(1,2), c(0.2,0.3))
#'
#' x <- ralaplace(1000, 0, 1, 0.2)
#' hist(x)
#'
#' @references \itemize{
#' \item Geraci Marco, Bottai Matteo (2007). Quantile regression for
#' longitudinal data using the asymmetric Laplace distribution.
#' Biostatistics (2007), 8, 1, pp. 140-154
#' \url{https://doi.org/10.1093/biostatistics/kxj039}
#' \item Yu, K., & Zhang, J. (2005). A three-parameter asymmetric
#' laplace distribution and its extension. Communications in Statistics
#' - Theory and Methods, 34, 1867-1879.
#' \url{https://doi.org/10.1080/03610920500199018}
#' }
#'
#' @rdname alaplace-distribution

#' @rdname alaplace-distribution
#' @export dalaplace
#' @aliases dalaplace
dalaplace <- function(q, mu=0, b=1, alpha=0.5, log=FALSE){
    alaplaceReturn <- alpha * (1-alpha) / b * exp(-(q-mu)/b * (alpha - (q<=mu)*1));
    if(log){
        alaplaceReturn <- log(alaplaceReturn);
    }
    return(alaplaceReturn);
}

#' @rdname alaplace-distribution
#' @export palaplace
#' @aliases palaplace
palaplace <- function(q, mu=0, b=1, alpha=0.5){
    Ie <- (q<=mu)*1;
    alaplaceReturn <- 1 - Ie - (1 - Ie - alpha) * exp((Ie - alpha) / b * (q-mu));

    return(alaplaceReturn);
}

#' @rdname alaplace-distribution
#' @export qalaplace
#' @aliases qalaplace
qalaplace <- function(p, mu=0, b=1, alpha=0.5){
    p <- unique(p);
    mu <- unique(mu);
    b <- unique(b);
    alpha <- unique(alpha);
    lengthMax <- max(length(p),length(mu),length(b),length(alpha));
    # If length of p, mu, b and alpha differs, then go difficult. Otherwise do simple stuff
    if(any(!c(length(p),length(mu),length(b),length(alpha)) %in% c(lengthMax, 1))){
        alaplaceReturn <- array(0,c(length(p),length(mu),length(b),length(alpha)),
                               dimnames=list(paste0("p=",p),paste0("mu=",mu),paste0("b=",b),paste0("alpha=",alpha)));
        alaplaceReturn[p==0.5,,,] <- 1;
        alaplaceReturn[p==0,,,] <- -Inf;
        alaplaceReturn[p==1,,,] <- Inf;
        probsToEstimate <- which(alaplaceReturn[,1,1,1]==0);
        alaplaceReturn[alaplaceReturn==1] <- 0;
        for(k in 1:length(b)){
            for(i in 1:length(alpha)){
                Ie <- (p[probsToEstimate]<=alpha[i])*1;
                alaplaceReturn[probsToEstimate,,k,i] <- b[k] / (Ie - alpha[i]) * log((1-Ie-p[probsToEstimate])/(1-Ie-alpha[i]));
            }
        }
        alaplaceReturn <- alaplaceReturn + rep(mu,each=length(p));
        # Drop the redundant dimensions
        alaplaceReturn <- alaplaceReturn[,,,];
    }
    else{
        Ie <- (p<=alpha)*1;
        alaplaceReturn <- mu + b / (Ie - alpha) * log((1-Ie-p)/(1-Ie-alpha));
        if(any(p==0)){
            alaplaceReturn[p==0] <- -Inf;
        }
        if(any(p==1)){
            alaplaceReturn[p==1] <- Inf;
        }
    }

    return(alaplaceReturn);
}

#' @rdname alaplace-distribution
#' @export ralaplace
#' @aliases ralaplace
ralaplace <- function(n=1, mu=0, b=1, alpha=0.5){
    alaplaceReturn <- qalaplace(runif(n,0,1),mu,b,alpha);
    return(alaplaceReturn);
}