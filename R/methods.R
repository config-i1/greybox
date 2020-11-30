##### IC functions #####

#' Corrected Akaike's Information Criterion and Bayesian Information Criterion
#'
#' This function extracts AICc / BICc from models. It can be applied to wide
#' variety of models that use logLik() and nobs() methods (including the
#' popular lm, forecast, smooth classes).
#'
#' AICc was proposed by Nariaki Sugiura in 1978 and is used on small samples
#' for the models with normally distributed residuals. BICc was derived in
#' McQuarrie (1999) and is used in similar circumstances.
#'
#' IMPORTANT NOTE: both of the criteria can only be used for univariate models
#' (regression models, ARIMA, ETS etc) with normally distributed residuals!
#'
#' @aliases AICc
#' @template author
#' @template AICRef
#'
#' @param object Time series model.
#' @param ...  Some stuff.
#' @return This function returns numeric value.
#' @seealso \link[stats]{AIC}, \link[stats]{BIC}
#' @references \itemize{
#' \item McQuarrie A.D., A small-sample correction for the Schwarz SIC
#' model selection criterion, Statistics & Probability Letters 44 (1999)
#' pp.79-86. \doi{10.1016/S0167-7152(98)00294-6}
#' \item Sugiura Nariaki (1978) Further analysts of the data by Akaike's
#' information criterion and the finite corrections, Communications in
#' Statistics - Theory and Methods, 7:1, 13-26,
#' \doi{10.1080/03610927808827599}
#' }
#' @keywords htest
#' @examples
#'
#' xreg <- cbind(rnorm(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rnorm(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#'
#' ourModel <- stepwise(xreg)
#'
#' AICc(ourModel)
#' BICc(ourModel)
#'
#' @rdname InformationCriteria
#' @export AICc
AICc <- function(object, ...) UseMethod("AICc")

#' @rdname InformationCriteria
#' @aliases BICc
#' @export BICc
BICc <- function(object, ...) UseMethod("BICc")


#' @export
AICc.default <- function(object, ...){
    llikelihood <- logLik(object);
    nparamAll <- nparam(object);
    llikelihood <- llikelihood[1:length(llikelihood)];
    obs <- nobs(object);

    IC <- 2*nparamAll - 2*llikelihood + 2 * nparamAll * (nparamAll + 1) / (obs - nparamAll - 1);

    return(IC);
}

#' @export
BICc.default <- function(object, ...){
    llikelihood <- logLik(object);
    nparamAll <- nparam(object);
    llikelihood <- llikelihood[1:length(llikelihood)];
    obs <- nobs(object);

    IC <- - 2*llikelihood + (nparamAll * log(obs) * obs) / (obs - nparamAll - 1);

    return(IC);
}

#' @export
AICc.varest <- function(object, ...){
    llikelihood <- logLik(object);
    llikelihood <- llikelihood[1:length(llikelihood)];
    nSeries <- object$K;
    nparamAll <- nrow(coef(object)[[1]]);

    obs <- nobs(object);
    IC <- -2*llikelihood + ((2*obs*(nparamAll*nSeries + nSeries*(nSeries+1)/2)) /
                                (obs - (nparamAll + nSeries + 1)));

    return(IC);
}

#' @export
BICc.varest <- function(object, ...){
    llikelihood <- logLik(object);
    llikelihood <- llikelihood[1:length(llikelihood)];
    nSeries <- object$K;
    nparamAll <- nrow(coef(object)[[1]]) + object$K;

    obs <- nobs(object);
    IC <- -2*llikelihood + (((nparamAll + nSeries*(nSeries+1)/2) *
                                 log(obs * nSeries) * obs * nSeries) /
                                (obs * nSeries - nparamAll - nSeries*(nSeries+1)/2));

    return(IC);
}

#' Functions that extracts type of error from the model
#'
#' This function allows extracting error type from any model.
#'
#' \code{errorType} extracts the type of error from the model
#' (either additive or multiplicative).
#'
#' @template author
#' @template keywords
#'
#' @param object Model estimated using one of the functions of smooth package.
#' @param ... Currently nothing is accepted via ellipsis.
#' @return Either \code{"A"} for additive error or \code{"M"} for multiplicative.
#' All the other functions return strings of character.
#' @examples
#'
#' xreg <- cbind(rnorm(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rnorm(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#' ourModel <- alm(y~x1+x2,as.data.frame(xreg))
#'
#' errorType(ourModel)
#'
#' @export errorType
errorType <- function(object, ...) UseMethod("errorType")

#' @export
errorType.default <- function(object, ...){
    return("A");
}

#' @export
errorType.ets <- function(object, ...){
    if(substr(object$method,5,5)=="M"){
        return("M");
    }
    else{
        return("A");
    }
}

#' @importFrom stats logLik
#' @export
logLik.alm <- function(object, ...){
    if(is.occurrence(object$occurrence)){
        return(structure(object$logLik,nobs=nobs(object),df=nparam(object)+nparam(object$occurrence),class="logLik"));
    }
    else{
        return(structure(object$logLik,nobs=nobs(object),df=nparam(object),class="logLik"));
    }
}


#' Point likelihood values
#'
#' This function returns a vector of logarithms of likelihoods for each observation
#'
#' Instead of taking the expected log-likelihood for the whole series, this function
#' calculates the individual value for each separate observation. Note that these
#' values are biased, so you would possibly need to take number of degrees of freedom
#' into account in order to have an unbiased estimator.
#'
#' This value is based on the general likelihood (not its concentrated version), so
#' the sum of these values may slightly differ from the output of logLik.
#'
#' @aliases pointLik
#' @param object Time series model.
#' @param ...  Some stuff.
#' @return This function returns a vector.
#' @template author
#' @seealso \link[stats]{AIC}, \link[stats]{BIC}
#' @keywords htest
#' @examples
#'
#' xreg <- cbind(rnorm(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rnorm(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#' ourModel <- alm(y~x1+x2,as.data.frame(xreg))
#'
#' pointLik(ourModel)
#'
#' # Bias correction
#' pointLik(ourModel) - nparam(ourModel)
#'
#' # Bias correction in AIC style
#' 2*(nparam(ourModel)/nobs(ourModel) - pointLik(ourModel))
#'
#' # BIC calculation based on pointLik
#' log(nobs(ourModel))*nparam(ourModel) - 2*sum(pointLik(ourModel))
#'
#' @export pointLik
pointLik <- function(object, ...) UseMethod("pointLik")

#' @export
pointLik.default <- function(object, ...){
    likValues <- dnorm(residuals(object), mean=0, sd=sigma(object), log=TRUE);

    return(likValues);
}

#' @export
pointLik.alm <- function(object, ...){
    distribution <- object$distribution;
    y <- actuals(object);
    ot <- y!=0;
    if(is.occurrence(object$occurrence)){
        otU <- y!=0;
        y <- y[otU];
        mu <- object$mu[otU];
    }
    else{
        otU <- rep(TRUE, nobs(object));
        mu <- object$mu;
    }
    scale <- object$scale;

    likValues <- vector("numeric",nobs(object));
    likValues[otU] <- switch(distribution,
                            "dnorm" = dnorm(y, mean=mu, sd=scale, log=TRUE),
                            "dlnorm" = dlnorm(y, meanlog=mu, sdlog=scale, log=TRUE),
                            "dgnorm" = dgnorm(y, mu=mu, alpha=scale, beta=object$other$beta, log=TRUE),
                            "dlgnorm" = dgnorm(log(y), mu=mu, alpha=scale, beta=object$other$beta, log=TRUE),
                            "dfnorm" = dfnorm(y, mu=mu, sigma=scale, log=TRUE),
                            "dbcnorm" = dbcnorm(y, mu=mu, sigma=scale, lambda=object$other$lambdaBC, log=TRUE),
                            "dlogitnorm" = dlogitnorm(y, mu=mu, sigma=scale, log=TRUE),
                            "dinvgauss" = dinvgauss(y, mean=mu, dispersion=scale/mu, log=TRUE),
                            "dlaplace" = dlaplace(y, mu=mu, scale=scale, log=TRUE),
                            "dllaplace" = dlaplace(log(y), mu=mu, scale=scale, log=TRUE),
                            "dalaplace" = dalaplace(y, mu=mu, scale=scale, alpha=object$other$alpha, log=TRUE),
                            "dlogis" = dlogis(y, location=mu, scale=scale, log=TRUE),
                            "dt" = dt(y-mu, df=scale, log=TRUE),
                            "ds" = ds(y, mu=mu, scale=scale, log=TRUE),
                            "dls" = ds(log(y), mu=mu, scale=scale, log=TRUE),
                            "dpois" = dpois(y, lambda=mu, log=TRUE),
                            "dnbinom" = dnbinom(y, mu=mu, size=object$other$size, log=TRUE),
                            "dchisq" = dchisq(y, df=object$other$nu, ncp=mu, log=TRUE),
                            "dbeta" = dbeta(y, shape1=mu, shape2=scale, log=TRUE),
                            "plogis" = c(plogis(mu[ot], location=0, scale=1, log.p=TRUE),
                                         plogis(mu[!ot], location=0, scale=1, lower.tail=FALSE, log.p=TRUE)),
                            "pnorm" = c(pnorm(mu[ot], mean=0, sd=1, log.p=TRUE),
                                        pnorm(mu[!ot], mean=0, sd=1, lower.tail=FALSE, log.p=TRUE))
    );
    if(any(distribution==c("dllaplace","dls","dlgnorm"))){
        likValues[otU] <- likValues[otU] - log(y);
    }

    # If this is a mixture model, take the respective probabilities into account (differential entropy)
    if(is.occurrence(object$occurrence)){
        likValues[!otU] <- -switch(distribution,
                                   "dnorm" =,
                                   "dfnorm" =,
                                   "dbcnorm" =,
                                   "dlogitnorm" =,
                                   "dlnorm" = log(sqrt(2*pi)*scale)+0.5,
                                   "dgnorm" =,
                                   "dlgnorm" = 1/object$other$beta -
                                       log(object$other$beta / (2*scale*gamma(1/object$other$beta))),
                                   "dinvgauss" = 0.5*(log(pi/2)+1+log(scale)),
                                   "dlaplace" =,
                                   "dllaplace" =,
                                   "dalaplace" = (1 + log(2*scale)),
                                   "dlogis" = 2,
                                   "dt" = ((scale+1)/2 *
                                               (digamma((scale+1)/2)-digamma(scale/2)) +
                                               log(sqrt(scale) * beta(scale/2,0.5))),
                                   "ds" = ,
                                   "dls" = (2 + 2*log(2*scale)),
                                   "dchisq" = (log(2)*gamma(scale/2)-
                                                   (1-scale/2)*digamma(scale/2)+
                                                   scale/2),
                                   "dbeta" = log(beta(mu,scale))-
                                       (mu-1)*
                                       (digamma(mu)-
                                            digamma(mu+scale))-
                                       (scale-1)*
                                       (digamma(scale)-
                                            digamma(mu+scale)),
                                   # This is a normal approximation of the real entropy
                                   "dpois" = 0.5*log(2*pi*scale)+0.5,
                                   "dnbinom" = log(sqrt(2*pi)*scale)+0.5,
                                   0
        );

        likValues <- likValues + pointLik(object$occurrence);
    }

    return(likValues);
}

#' @export
pointLik.ets <- function(object, ...){

    likValues <- pointLik.default(object);
    if(errorType(object)=="M"){
        likValues[] <- likValues - log(abs(fitted(object)));
    }
    # This correction is needed so that logLik(object) ~ sum(pointLik(object))
    likValues[] <- likValues + 0.5 * (log(2 * pi) + 1 - log(nobs(object)));

    return(likValues);
}

#' Point AIC
#'
#' This function returns a vector of AIC values for the in-sample observations
#'
#' This is based on \link[greybox]{pointLik} function. The formula for this is:
#' pAIC_t = 2 * k - 2 * T * l_t ,
#' where k is the number of parameters, T is the number of observations and l_t is
#' the point likelihood. This way we preserve the property that AIC = mean(pAIC).
#'
#' @aliases pAIC
#' @param object Time series model.
#' @param ...  Some stuff.
#' @return The function returns the vector of point AIC values.
#' @template author
#' @seealso \link[greybox]{pointLik}
#' @keywords htest
#' @examples
#'
#' xreg <- cbind(rnorm(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rnorm(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#' ourModel <- alm(y~x1+x2,as.data.frame(xreg))
#'
#' pAICValues <- pAIC(ourModel)
#'
#' mean(pAICValues)
#' AIC(ourModel)
#'
#' @rdname pointIC
#' @export pAIC
pAIC <- function(object, ...) UseMethod("pAIC")

#' @export
pAIC.default <- function(object, ...){
    obs <- nobs(object);
    k <- nparam(object);
    return(2 * k - 2 * obs * pointLik(object));
}

#' @rdname pointIC
#' @export pAICc
pAICc <- function(object, ...) UseMethod("pAICc")

#' @export
pAICc.default <- function(object, ...){
    obs <- nobs(object);
    k <- nparam(object);
    return(2 * k - 2 * obs * pointLik(object) + 2 * k * (k + 1) / (obs - k - 1));
}

#' @rdname pointIC
#' @export pBIC
pBIC <- function(object, ...) UseMethod("pBIC")

#' @export
pBIC.default <- function(object, ...){
    obs <- nobs(object);
    k <- nparam(object);
    return(log(obs) * k - 2 * obs * pointLik(object));
}

#' @rdname pointIC
#' @export pBIC
pBICc <- function(object, ...) UseMethod("pBICc")

#' @export
pBICc.default <- function(object, ...){
    obs <- nobs(object);
    k <- nparam(object);
    return((k * log(obs) * obs) / (obs - k - 1)  - 2 * obs * pointLik(object));
}


#### Coefficients and extraction functions ####

#' Function extracts the actual values from the function
#'
#' This is a simple method that returns the values of the response variable of the model
#'
#' @template author
#'
#' @param object Model estimated using one of the functions of smooth package.
#' @param all If \code{FALSE}, then in the case of the occurrence model, only demand
#' sizes will be returned.
#' @param ... Other parameters to pass to the method. Currently nothing is supported here.
#' @return The vector of the response variable.
#' @examples
#'
#' xreg <- cbind(rnorm(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rnorm(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#'
#' ourModel <- stepwise(xreg)
#'
#' actuals(ourModel)
#'
#' @rdname actuals
#' @export
actuals <- function(object, all=TRUE, ...) UseMethod("actuals")

#' @rdname actuals
#' @export
actuals.default <- function(object, all=TRUE, ...){
    return(object$y);
}

#' @rdname actuals
#' @export
actuals.alm <- function(object, all=TRUE, ...){
    if(all){
        return(object$data[,1])
    }
    else{
        return(object$data[object$data[,1]!=0,1]);
    }
}

#' Coefficients of the model and their statistics
#'
#' These are the basic methods for the alm and greybox models that extract coefficients,
#' their covariance matrix, confidence intervals or generating the summary of the model.
#' If the non-likelihood related loss was used in the process, then it is recommended to
#' use bootstrap (which is slow, but more reliable).
#'
#' The \code{coef()} method returns the vector of parameters of the model. If
#' \code{bootstrap=TRUE}, then the coefficients are calculated as the mean values of the
#' bootstrapped ones.
#'
#' The \code{vcov()} method returns the covariance matrix of parameters. If
#' \code{bootstrap=TRUE}, then the bootstrap is done using \link[greybox]{coefbootstrap}
#' function
#'
#' The \code{confint()} constructs the confidence intervals for parameters. Once again,
#' this can be done using \code{bootstrap=TRUE}.
#'
#' Finally, the \code{summary()} returns the table with parameters, their standard errors,
#' confidence intervals and general information about the model.
#'
#' @param object The model estimated using alm or other greybox function.
#' @param bootstrap The logical, which determines, whether to use bootstrap in the
#' process or not.
#' @param level The confidence level for the construction of the interval.
#' @param parm The parameters that need to be extracted.
#' @param ... Parameters passed to \link[greybox]{coefbootstrap} function.
#'
#' @return Depending on the used method, different values are returned.
#'
#' @template author
#' @template keywords
#'
#' @seealso \code{\link[greybox]{alm}, \link[greybox]{coefbootstrap}}
#'
#' @examples
#' # An example with ALM
#' ourModel <- alm(mpg~., mtcars, distribution="dlnorm")
#' coef(ourModel)
#' vcov(ourModel)
#' confint(ourModel)
#' summary(ourModel)
#'
#' @rdname coef.alm
#' @aliases coef.greybox coef.alm
#' @importFrom stats coef
#' @export
coef.greybox <- function(object, bootstrap=FALSE, ...){
    if(bootstrap){
        return(colMeans(coefbootstrap(object, ...)$coefficients));
    }
    else{
        return(object$coefficients);
    }
}

#' @export
coef.greyboxD <- function(object, ...){
    coefReturned <- list(coefficients=object$coefficients,
                         dynamic=object$coefficientsDynamic,importance=object$importance);
    return(structure(coefReturned,class="coef.greyboxD"));
}

#' @aliases confint.alm
#' @rdname coef.alm
#' @importFrom stats confint qt quantile
#' @export
confint.alm <- function(object, parm, level=0.95, bootstrap=FALSE, ...){

    confintNames <- c(paste0((1-level)/2*100,"%"),
                      paste0((1+level)/2*100,"%"));
    # Extract parameters
    parameters <- coef(object);
    if(!bootstrap){
        parametersSE <- sqrt(diag(vcov(object)));
        # Define quantiles using Student distribution
        paramQuantiles <- qt((1+level)/2,df=object$df.residual);

        # We can use normal distribution, because of the asymptotics of MLE
        confintValues <- cbind(parameters-paramQuantiles*parametersSE,
                               parameters+paramQuantiles*parametersSE);
        colnames(confintValues) <- confintNames;
        rownames(confintValues) <- names(parameters);

        # Return S.E. as well, so not to repeat the thing twice...
        confintValues <- cbind(parametersSE, confintValues);
        # Give the name to the first column
        colnames(confintValues)[1] <- "S.E.";
    }
    else{
        coefValues <- coefbootstrap(object, ...);
        confintValues <- cbind(sqrt(diag(coefValues$vcov)),
                               apply(coefValues$coefficients,2,quantile,probs=(1-level)/2),
                               apply(coefValues$coefficients,2,quantile,probs=(1+level)/2));
        colnames(confintValues) <- c("S.E.",confintNames);
    }

    # If parm was not provided, return everything.
    if(!exists("parm",inherits=FALSE)){
        parm <- names(parameters);
    }

    return(confintValues[parm,,drop=FALSE]);
}

#' @export
confint.greyboxC <- function(object, parm, level=0.95, ...){

    # Extract parameters
    parameters <- coef(object);
    # Extract SE
    parametersSE <- sqrt(abs(diag(vcov(object))));
    # Define quantiles using Student distribution
    paramQuantiles <- qt((1+level)/2,df=object$df.residual);
    # Do the stuff
    confintValues <- cbind(parameters-paramQuantiles*parametersSE,
                           parameters+paramQuantiles*parametersSE);
    confintNames <- c(paste0((1-level)/2*100,"%"),
                                 paste0((1+level)/2*100,"%"));
    colnames(confintValues) <- confintNames;
    # If parm was not provided, return everything.
    if(!exists("parm",inherits=FALSE)){
        parm <- names(parameters);
    }
    confintValues <- confintValues[parm,];
    if(!is.matrix(confintValues)){
        confintValues <- matrix(confintValues,1,2);
        colnames(confintValues) <- confintNames;
        rownames(confintValues) <- names(parameters);
    }
    return(confintValues);
}

#' @export
confint.greyboxD <- function(object, parm, level=0.95, ...){

    # Extract parameters
    parameters <- coef(object)$coefficients;
    # Extract SE
    parametersSE <- sqrt(abs(diag(vcov(object))));
    # Define quantiles using Student distribution
    paramQuantiles <- qt((1+level)/2,df=object$df.residual);
    # Do the stuff
    confintValues <- array(NA,c(length(parameters),2),
                           dimnames=list(dimnames(parameters)[[2]],
                                         c(paste0((1-level)/2*100,"%"),paste0((1+level)/2*100,"%"))));
    confintValues[,1] <- parameters-paramQuantiles*parametersSE;
    confintValues[,2] <- parameters+paramQuantiles*parametersSE;

    # If parm was not provided, return everything.
    if(!exists("parm",inherits=FALSE)){
        parm <- colnames(parameters);
    }
    return(confintValues[parm,]);
}

# This is needed for lmCombine and other functions, using fast regressions
#' @export
confint.lmGreybox <- function(object, parm, level=0.95, ...){
    # Extract parameters
    parameters <- coef(object);
    parametersSE <- sqrt(diag(vcov(object)));
    # Define quantiles using Student distribution
    paramQuantiles <- qt((1+level)/2,df=object$df.residual);

    # We can use normal distribution, because of the asymptotics of MLE
    confintValues <- cbind(parameters-qt((1+level)/2,df=object$df.residual)*parametersSE,
                           parameters+qt((1+level)/2,df=object$df.residual)*parametersSE);
    confintNames <- c(paste0((1-level)/2*100,"%"),
                                 paste0((1+level)/2*100,"%"));
    colnames(confintValues) <- confintNames;
    rownames(confintValues) <- names(parameters);

    if(!is.matrix(confintValues)){
        confintValues <- matrix(confintValues,1,2);
        colnames(confintValues) <- confintNames;
        rownames(confintValues) <- names(parameters);
    }

    # Return S.E. as well, so not to repeat the thing twice...
    confintValues <- cbind(parametersSE, confintValues);
    colnames(confintValues)[1] <- "S.E.";
    return(confintValues);
}

#' @importFrom stats nobs fitted
#' @export
nobs.alm <- function(object, ...){
    return(length(actuals(object, ...)));
}

#' @export
nobs.greybox <- function(object, ...){
    return(length(fitted(object)));
}

#' @export
nobs.varest <- function(object, ...){
    return(object$obs);
}

#' Number of parameters in the model
#'
#' This function returns the number of estimated parameters in the model
#'
#' This is a very basic and a simple function which does what it says:
#' extracts number of parameters in the estimated model.
#'
#' @aliases nparam
#' @param object Time series model.
#' @param ... Some other parameters passed to the method.
#' @return This function returns a numeric value.
#' @template author
#' @seealso \link[stats]{nobs}, \link[stats]{logLik}
#' @keywords htest
#' @examples
#'
#' ### Simple example
#' xreg <- cbind(rnorm(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rnorm(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#' ourModel <- lm(y~.,data=as.data.frame(xreg))
#'
#' nparam(ourModel)
#'
#' @rdname nparam
#' @importFrom stats coef
#' @export nparam
nparam <- function(object, ...) UseMethod("nparam")

#' @export
nparam.default <- function(object, ...){
    # The length of the vector of parameters + variance
    return(length(coef(object))+1);
}

#' @export
nparam.alm <- function(object, ...){
    # The number of parameters in the model + in the occurrence part
    # if(!is.null(object$occurrence)){
    #     return(object$df+object$occurrence$df);
    # }
    # else{
        return(object$df);
    # }
}

#' @export
nparam.logLik <- function(object, ...){
    # The length of the vector of parameters + variance
    return(attributes(object)$df);
}

#' @export
nparam.greyboxC <- function(object, ...){
    # The length of the vector of parameters + variance
    return(sum(object$importance)+1);
}

#' @export
nparam.varest <- function(object, ...){
    ### This is the nparam per series
    # Parameters in all the matrices + the elements of the covariance matrix
    return(nrow(coef(object)[[1]])*object$K + 0.5*object$K*(object$K+1));
}

#' @importFrom stats sigma
#' @export
sigma.greybox <- function(object, all=FALSE, ...){
    return(sqrt(sum(residuals(object)^2)/(nobs(object, all=all)-nparam(object))));
}

#' @export
sigma.alm <- function(object, ...){
    if(any(object$distribution==c("plogis","pnorm"))){
        return(object$scale);
    }
    else{
        return(sigma.greybox(object, ...));
    }
}

#' @export
sigma.ets <- function(object, ...){
    return(sqrt(object$sigma2));
}

#' @export
sigma.varest <- function(object, ...){
    # OLS estimate of Sigma, without the covariances
    return(t(residuals(object)) %*% residuals(object) / (nobs(object)-nparam(object)+object$K));
}

#' @aliases vcov.alm
#' @rdname coef.alm
#' @importFrom stats vcov
#' @export
vcov.alm <- function(object, bootstrap=FALSE, ...){
    nVariables <- length(coef(object));
    variablesNames <- names(coef(object));
    interceptIsNeeded <- any(variablesNames=="(Intercept)");

    # Try the basic method, if not a bootstrap
    if(!bootstrap){
        # If the likelihood is not available, then this is a non-conventional loss
        if(is.na(logLik(object))){
            warning(paste0("You used the non-likelihood compatible loss, so the covariance matrix might be incorrect. ",
                           "It is recommended to use bootstrap=TRUE option in this case."),
                    call.=FALSE);
        }

        if(any(object$distribution==c("dnorm","dlnorm","dbcnorm","dlogitnorm"))){
            matrixXreg <- object$data[,-1,drop=FALSE];
            if(interceptIsNeeded){
                matrixXreg <- cbind(1,matrixXreg);
                colnames(matrixXreg)[1] <- "(Intercept)";
            }
            # colnames(matrixXreg) <- names(coef(object));
            matrixXreg <- crossprod(matrixXreg);
            vcovMatrixTry <- try(chol2inv(chol(matrixXreg)), silent=TRUE);
            if(any(class(vcovMatrixTry)=="try-error")){
                warning(paste0("Choleski decomposition of covariance matrix failed, so we had to revert to the simple inversion.\n",
                               "The estimate of the covariance matrix of parameters might be inaccurate.\n"),
                        call.=FALSE);
                vcovMatrix <- try(solve(matrixXreg, diag(nVariables), tol=1e-20), silent=TRUE);

                # If the conventional approach failed, do bootstrap
                if(any(class(vcovMatrix)=="try-error")){
                    warning(paste0("Sorry, but the hessian is singular, so we could not invert it.\n",
                                   "Switching to bootstrap of covariance matrix of parameters.\n"),
                            call.=FALSE, immediate.=TRUE);
                    vcov <- coefbootstrap(object, ...)$vcov;
                }
                else{
                    # vcov <- object$scale^2 * vcovMatrix;
                    vcov <- sigma(object, all=FALSE)^2 * vcovMatrix;
                }
            }
            else{
                vcovMatrix <- vcovMatrixTry;
                # vcov <- object$scale^2 * vcovMatrix;
                vcov <- sigma(object, all=FALSE)^2 * vcovMatrix;
            }
            rownames(vcov) <- colnames(vcov) <- variablesNames;
        }
        else if(object$distribution=="dpois"){
            matrixXreg <- object$data[,-1,drop=FALSE];
            obsInsample <- nobs(object);
            if(interceptIsNeeded){
                matrixXreg <- cbind(1,matrixXreg);
                colnames(matrixXreg)[1] <- "(Intercept)";
            }
            # colnames(matrixXreg) <- names(coef(object));
            FIMatrix <- matrixXreg[1,] %*% t(matrixXreg[1,]) * object$mu[1];
            for(j in 2:obsInsample){
                FIMatrix[] <- FIMatrix + matrixXreg[j,] %*% t(matrixXreg[j,]) * object$mu[j];
            }

            # See if Choleski works... It sometimes fails, when we don't get to the max of likelihood.
            vcovMatrixTry <- try(chol2inv(chol(FIMatrix)), silent=TRUE);
            if(any(class(vcovMatrixTry)=="try-error")){
                warning(paste0("Choleski decomposition of hessian failed, so we had to revert to the simple inversion.\n",
                               "The estimate of the covariance matrix of parameters might be inaccurate.\n"),
                        call.=FALSE);
                vcov <- try(solve(FIMatrix, diag(nVariables), tol=1e-20), silent=TRUE);

                # If the conventional approach failed, do bootstrap
                if(any(class(FIMatrix)=="try-error")){
                    warning(paste0("Sorry, but the hessian is singular, so we could not invert it.\n",
                                   "Switching to bootstrap of covariance matrix of parameters.\n"),
                            call.=FALSE, immediate.=TRUE);
                    vcov <- coefbootstrap(object, ...)$vcov;
                }
            }
            else{
                vcov <- vcovMatrixTry;
            }

            rownames(vcov) <- colnames(vcov) <- variablesNames;
        }
        else{
            # Form the call for alm
            newCall <- object$call;
            if(interceptIsNeeded){
                newCall$formula <- as.formula(paste0("`",all.vars(newCall$formula)[1],"`~."));
            }
            else{
                newCall$formula <- as.formula(paste0("`",all.vars(newCall$formula)[1],"`~.-1"));
            }
            newCall$data <- object$data;
            newCall$subset <- object$subset;
            newCall$distribution <- object$distribution;
            if(object$loss=="custom"){
                newCall$loss <- object$lossFunction;
            }
            else{
                newCall$loss <- object$loss;
            }
            newCall$ar <- object$call$ar;
            newCall$i <- object$call$i;
            newCall$parameters <- coef(object);
            newCall$fast <- TRUE;
            if(any(object$distribution==c("dchisq","dt"))){
                newCall$nu <- object$other$nu;
            }
            else if(object$distribution=="dnbinom"){
                newCall$size <- object$other$size;
            }
            else if(object$distribution=="dalaplace"){
                newCall$alpha <- object$other$alpha;
            }
            else if(object$distribution=="dfnorm"){
                newCall$sigma <- object$other$sigma;
            }
            else if(object$distribution=="dbcnorm"){
                newCall$lambdaBC <- object$other$lambdaBC;
            }
            else if(any(object$distribution==c("dgnorm","dlgnorm"))){
                newCall$beta <- object$other$beta;
            }
            newCall$FI <- TRUE;
            # newCall$occurrence <- NULL;
            newCall$occurrence <- object$occurrence;
            # Recall alm to get hessian
            FIMatrix <- eval(newCall)$FI;

            # See if Choleski works... It sometimes fails, when we don't get to the max of likelihood.
            vcovMatrixTry <- try(chol2inv(chol(FIMatrix)), silent=TRUE);
            if(any(class(vcovMatrixTry)=="try-error")){
                warning(paste0("Choleski decomposition of hessian failed, so we had to revert to the simple inversion.\n",
                               "The estimate of the covariance matrix of parameters might be inaccurate.\n"),
                        call.=FALSE);
                FIMatrix <- try(solve(FIMatrix, diag(nVariables), tol=1e-20), silent=TRUE);
                if(any(class(FIMatrix)=="try-error")){
                    vcov <- diag(1e+100,nVariables);
                }
                else{
                    vcov <- FIMatrix;
                }
            }
            else{
                vcov <- vcovMatrixTry;
            }

            # If the conventional approach failed, do bootstrap
            if(any(class(FIMatrix)=="try-error")){
                warning(paste0("Sorry, but the hessian is singular, so we could not invert it.\n",
                               "Switching to bootstrap of covariance matrix of parameters.\n"),
                        call.=FALSE, immediate.=TRUE);
                vcov <- coefbootstrap(object, ...)$vcov;
            }
        }
    }
    else{
        vcov <- coefbootstrap(object, ...)$vcov;
    }

    if(nVariables>1){
        if(object$distribution=="dbeta"){
            dimnames(vcov) <- list(c(paste0("shape1_",variablesNames),paste0("shape2_",variablesNames)),
                                   c(paste0("shape1_",variablesNames),paste0("shape2_",variablesNames)));
        }
        else{
            dimnames(vcov) <- list(variablesNames,variablesNames);
        }
    }
    else{
        names(vcov) <- variablesNames;
    }

    # Sometimes the diagonal elements in the covariance matrix are negative because likelihood is not fully maximised...
    if(any(diag(vcov)<0)){
        diag(vcov) <- abs(diag(vcov));
    }
    return(vcov);
}

#' @export
vcov.greyboxC <- function(object, ...){
    # xreg <- as.matrix(object$data);
    # xreg[,1] <- 1;
    # colnames(xreg)[1] <- "(Intercept)";
    #
    # vcovValue <- sigma(object)^2 * solve(t(xreg) %*% xreg) * object$importance %*% t(object$importance);
    # warning("The covariance matrix for combined models is approximate. Don't rely too much on that.",call.=FALSE);
    return(object$vcov);
}

#' @export
vcov.greyboxD <- function(object, ...){
    return(object$vcov);
    # xreg <- as.matrix(object$data);
    # xreg[,1] <- 1;
    # colnames(xreg)[1] <- "(Intercept)";
    # importance <- apply(object$importance,2,mean);
    #
    # vcovValue <- sigma(object)^2 * solve(t(xreg) %*% xreg) * importance %*% t(importance);
    # warning("The covariance matrix for combined models is approximate. Don't rely too much on that.",call.=FALSE);
    # return(vcovValue);
}

# This is needed for lmCombine and other functions, using fast regressions
#' @export
vcov.lmGreybox <- function(object, ...){
    vcov <- sigma(object)^2 * solve(crossprod(object$xreg));
    return(vcov);
}

#### Plot functions ####

#' Plots of the fit and residuals
#'
#' The function produces fitted values and plots for the residuals of the greybox functions
#'
#' The list of produced plots includes:
#' \enumerate{
#' \item Actuals vs Fitted values. Allows analysing, whether there are any issues in the fit.
#' Does the variability of actuals increase with the increase of fitted values? Is the relation
#' well captured? They grey line on the plot corresponds to the perfect fit of the model.
#' \item Standardised residuals vs Fitted. Plots the points and the confidence bounds
#' (red lines) for the specified confidence \code{level}. Useful for the analysis of outliers;
#' \item Studentised residuals vs Fitted. This is similar to the previous plot, but with the
#' residuals divided by the scales with the leave-one-out approach. Should be more sensitive
#' to outliers;
#' \item Absolute residuals vs Fitted. Useful for the analysis of heteroscedasticity;
#' \item Squared residuals vs Fitted - similar to (3), but with squared values;
#' \item Q-Q plot with the specified distribution. Can be used in order to see if the
#' residuals follow the assumed distribution. The type of distribution depends on the one used
#' in the estimation (see \code{distribution} parameter in \link[greybox]{alm});
#' \item Fitted over time. Plots actuals (black line), fitted values (purple line) and
#' prediction interval (red lines) of width \code{level}, but only in the case, when there
#' are some values lying outside of it. Can be used in order to make sure that the model
#' did not miss any important events over time;
#' \item Standardised residuals vs Time. Useful if you want to see, if there is autocorrelation or
#' if there is heteroscedasticity in time. This also shows, when the outliers happen;
#' \item Studentised residuals vs Time. Similar to previous, but with studentised residuals;
#' \item ACF of the residuals. Are the residuals autocorrelated? See \link[stats]{acf} for
#' details;
#' \item PACF of the residuals. No, really, are they autocorrelated? See \link[stats]{pacf}
#' for details;
#' \item Cook's distance over time. Shows influential observations. If a value is above 0.5, then
#' this means that the observation influences the parameters of the model. This does not work well
#' for non-normal distributions.
#' }
#' Which of the plots to produce, is specified via the \code{which} parameter. The plots 2, 3, 7,
#' 8 and 9 also use the parameters \code{level}, which specifies the confidence level for
#' the intervals.
#'
#' @param x Time series model for which forecasts are required.
#' @param which Which of the plots to produce. The possible options (see details for explanations):
#' \enumerate{
#' \item Actuals vs Fitted values;
#' \item Standardised residuals vs Fitted;
#' \item Studentised residuals vs Fitted;
#' \item Absolute residuals vs Fitted;
#' \item Squared residuals vs Fitted;
#' \item Q-Q plot with the specified distribution;
#' \item Fitted over time;
#' \item Standardised residuals vs Time;
#' \item Studentised residuals vs Time;
#' \item ACF of the residuals;
#' \item PACF of the residuals;
#' \item Cook's distance over time with 0.5, 0.75 and 0.95 quantile lines from Fisher's distribution.
#' }
#' @param level Confidence level. Defines width of confidence interval. Used in plots (2), (3), (7),
#' (8), (9), (10) and (11).
#' @param legend If \code{TRUE}, then the legend is produced on plots (2), (3) and (7).
#' @param ask Logical; if \code{TRUE}, the user is asked to press Enter before each plot.
#' @param lowess Logical; if \code{TRUE}, LOWESS lines are drawn on scatterplots, see \link[stats]{lowess}.
#' @param ... The parameters passed to the plot functions. Recommended to use with separate plots.
#' @return The function produces the number of plots, specified in the parameter \code{which}.
#'
#' @template author
#' @seealso \link[stats]{plot.lm}, \link[stats]{rstandard}, \link[stats]{rstudent}
#' @keywords ts univar
#' @examples
#'
#' xreg <- cbind(rlaplace(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rlaplace(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#'
#' ourModel <- alm(y~x1+x2, xreg, distribution="dnorm")
#'
#' par(mfcol=c(3,4))
#' plot(ourModel, c(1:12))
#'
#' @importFrom stats ppoints qqline qqnorm qqplot acf pacf lowess qf
#' @importFrom grDevices dev.interactive devAskNewPage grey
#' @aliases plot.alm
#' @export
plot.greybox <- function(x, which=c(1,2,4,6), level=0.95, legend=FALSE,
                         ask=prod(par("mfcol")) < length(which) && dev.interactive(),
                         lowess=TRUE, ...){

    # Define, whether to wait for the hit of "Enter"
    if(ask){
        oask <- devAskNewPage(TRUE);
        on.exit(devAskNewPage(oask));
    }

    # 1. Fitted vs Actuals values
    plot1 <- function(x, ...){
        ellipsis <- list(...);

        # Get the actuals and the fitted values
        ellipsis$y <- as.vector(actuals(x));
        if(is.occurrence(x)){
            if(any(x$distribution==c("plogis","pnorm"))){
                ellipsis$y <- (ellipsis$y!=0)*1;
            }
        }
        ellipsis$x <- as.vector(fitted(x));

        # If this is a mixture model, remove zeroes
        if(is.occurrence(x$occurrence)){
            ellipsis$x <- ellipsis$x[ellipsis$y!=0];
            ellipsis$y <- ellipsis$y[ellipsis$y!=0];
        }

        # Remove NAs
        if(any(is.na(ellipsis$x))){
            ellipsis$y <- ellipsis$y[!is.na(ellipsis$x)];
            ellipsis$x <- ellipsis$x[!is.na(ellipsis$x)];
        }
        if(any(is.na(ellipsis$y))){
            ellipsis$x <- ellipsis$x[!is.na(ellipsis$y)];
            ellipsis$y <- ellipsis$y[!is.na(ellipsis$y)];
        }

        # Title
        if(!any(names(ellipsis)=="main")){
            ellipsis$main <- "Actuals vs Fitted";
        }
        # If type and ylab are not provided, set them...
        if(!any(names(ellipsis)=="type")){
            ellipsis$type <- "p";
        }
        if(!any(names(ellipsis)=="ylab")){
            ellipsis$ylab <- paste0("Actual ",all.vars(x$call$formula)[1]);
        }
        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Fitted";
        }
        # xlim and ylim
        # if(!any(names(ellipsis)=="xlim")){
        #     ellipsis$xlim <- range(c(ellipsis$x,ellipsis$y),na.rm=TRUE);
        # }
        # if(!any(names(ellipsis)=="ylim")){
        #     ellipsis$ylim <- range(c(ellipsis$x,ellipsis$y),na.rm=TRUE);
        # }

        # Start plotting
        do.call(plot,ellipsis);
        abline(a=0,b=1,col="grey",lwd=2,lty=2)
        if(lowess){
            lines(lowess(ellipsis$x, ellipsis$y), col="red");
        }
    }

    # 2 and 3: Standardised  / studentised residuals vs Fitted
    plot2 <- function(x, type="rstandard", ...){
        ellipsis <- list(...);

        ellipsis$x <- as.vector(fitted(x));
        if(type=="rstandard"){
            ellipsis$y <- as.vector(rstandard(x));
            yName <- "Standardised";
        }
        else{
            ellipsis$y <- as.vector(rstudent(x));
            yName <- "Studentised";
        }

        if(!any(names(ellipsis)=="main")){
            ellipsis$main <- paste0(yName," Residuals vs Fitted");
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Fitted";
        }
        if(!any(names(ellipsis)=="ylab")){
            ellipsis$ylab <- paste0(yName," Residuals");
        }

        if(legend){
            if(ellipsis$x[length(ellipsis$x)]>mean(ellipsis$x)){
                legendPosition <- "bottomright";
            }
            else{
                legendPosition <- "topright";
            }
        }

        # Get the IDs of outliers and statistic
        outliers <- outlierdummy(x, level=level, type=type);
        outliersID <- outliers$id;
        statistic <- outliers$statistic;
        # Analyse stuff in logarithms if the distribution is dinvgauss
        if(x$distribution=="dinvgauss"){
            ellipsis$y[] <- log(ellipsis$y);
            statistic[] <- log(statistic);
        }
        # Substitute zeroes with NAs if there was an occurrence
        if(is.occurrence(x$occurrence)){
            ellipsis$x[actuals(x$occurrence)==0] <- NA;
        }

        if(!any(names(ellipsis)=="ylim")){
            ellipsis$ylim <- range(c(ellipsis$y,statistic), na.rm=TRUE)*1.2;
            if(legend){
                if(legendPosition=="bottomright"){
                    ellipsis$ylim[1] <- ellipsis$ylim[1] - 0.2*diff(ellipsis$ylim);
                }
                else{
                    ellipsis$ylim[2] <- ellipsis$ylim[2] + 0.2*diff(ellipsis$ylim);
                }
            }
        }

        xRange <- range(ellipsis$x, na.rm=TRUE);
        xRange[1] <- xRange[1] - sd(ellipsis$x, na.rm=TRUE);
        xRange[2] <- xRange[2] + sd(ellipsis$x, na.rm=TRUE);

        do.call(plot,ellipsis);
        abline(h=0, col="grey", lty=2);
        polygon(c(xRange,rev(xRange)),c(statistic[1],statistic[1],statistic[2],statistic[2]),
                col="lightgrey", border=NA, density=10);
        abline(h=statistic, col="red", lty=2);
        if(length(outliersID)>0){
            points(ellipsis$x[outliersID], ellipsis$y[outliersID], pch=16);
            text(ellipsis$x[outliersID], ellipsis$y[outliersID], labels=outliersID, pos=(ellipsis$x[outliersID]>0)*2+1);
        }
        if(lowess){
            # Remove NAs
            if(any(is.na(ellipsis$x))){
                ellipsis$y <- ellipsis$y[!is.na(ellipsis$x)];
                ellipsis$x <- ellipsis$x[!is.na(ellipsis$x)];
            }
            lines(lowess(ellipsis$x, ellipsis$y), col="red");
        }

        if(legend){
            if(lowess){
                legend(legendPosition,
                       legend=c(paste0(round(level,3)*100,"% bounds"),"outside the bounds","LOWESS line"),
                       col=c("red", "black","red"), lwd=c(1,NA,1), lty=c(2,1,1), pch=c(NA,16,NA));
            }
            else{
                legend(legendPosition,
                       legend=c(paste0(round(level,3)*100,"% bounds"),"outside the bounds"),
                       col=c("red", "black"), lwd=c(1,NA), lty=c(2,1), pch=c(NA,16));
            }
        }
    }

    # 4 and 5. Fitted vs |Residuals| or Fitted vs Residuals^2
    plot3 <- function(x, type="abs", ...){
        ellipsis <- list(...);

        ellipsis$x <- as.vector(fitted(x));
        ellipsis$y <- as.vector(residuals(x));
        if(x$distribution=="dinvgauss"){
            ellipsis$y[] <- log(ellipsis$y);
        }
        if(type=="abs"){
            ellipsis$y[] <- abs(ellipsis$y);
        }
        else{
            ellipsis$y[] <- ellipsis$y^2;
        }

        if(is.occurrence(x$occurrence)){
            ellipsis$x <- ellipsis$x[ellipsis$y!=0];
            ellipsis$y <- ellipsis$y[ellipsis$y!=0];
        }
        if(!any(names(ellipsis)=="main")){
            if(type=="abs"){
                ellipsis$main <- "|Residuals| vs Fitted";
            }
            else{
                ellipsis$main <- "Residuals^2 vs Fitted";
            }
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Fitted";
        }
        if(!any(names(ellipsis)=="ylab")){
            if(type=="abs"){
                ellipsis$ylab <- "|Residuals|";
            }
            else{
                ellipsis$ylab <- "Residuals^2";
            }
        }

        do.call(plot,ellipsis);
        abline(h=0, col="grey", lty=2);
        if(lowess){
            lines(lowess(ellipsis$x, ellipsis$y), col="red");
        }
    }

    # 6. Q-Q with the specified distribution
    plot4 <- function(x, ...){
        ellipsis <- list(...);

        ellipsis$y <- as.vector(residuals(x));
        if(is.occurrence(x$occurrence)){
            ellipsis$y <- ellipsis$y[actuals(x$occurrence)!=0];
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Theoretical Quantile";
        }
        if(!any(names(ellipsis)=="ylab")){
            ellipsis$ylab <- "Actual Quantile";
        }

        if(any(x$distribution==c("dnorm","dlnorm","dbcnorm","dlogitnorm","dfnorm","plogis","pnorm"))){
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ plot of normal distribution";
            }

            do.call(qqnorm, ellipsis);
            qqline(ellipsis$y);
        }
        else if(any(x$distribution==c("dgnorm","dlgnorm"))){
            # Standardise residuals
            ellipsis$y[] <- ellipsis$y / sd(ellipsis$y);
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ-plot of Generalised Normal distribution";
            }
            ellipsis$x <- qgnorm(ppoints(500), mu=0, alpha=x$scale, beta=x$other$beta);

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) qgnorm(p, mu=0, alpha=x$scale, beta=x$other$beta));
        }
        else if(any(x$distribution==c("dlaplace","dllaplace"))){
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ-plot of Laplace distribution";
            }
            ellipsis$x <- qlaplace(ppoints(500), mu=0, scale=x$scale);

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) qlaplace(p, mu=0, scale=x$scale));
        }
        else if(x$distribution=="dalaplace"){
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- paste0("QQ-plot of Asymmetric Laplace distribution with alpha=",round(x$other$alpha,3));
            }
            ellipsis$x <- qalaplace(ppoints(500), mu=0, scale=x$scale, alpha=x$other$alpha);

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) qalaplace(p, mu=0, scale=x$scale, alpha=x$other$alpha));
        }
        else if(x$distribution=="dlogis"){
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ-plot of Logistic distribution";
            }
            ellipsis$x <- qlogis(ppoints(500), location=0, scale=x$scale);

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) qlogis(p, location=0, scale=x$scale));
        }
        else if(any(x$distribution==c("ds","dls"))){
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ-plot of S distribution";
            }
            ellipsis$x <- qs(ppoints(500), mu=0, scale=x$scale);

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) qs(p, mu=0, scale=x$scale));
        }
        else if(x$distribution=="dt"){
            # Standardise residuals
            ellipsis$y[] <- ellipsis$y / sd(ellipsis$y);
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ-plot of Student's distribution";
            }
            ellipsis$x <- qt(ppoints(500), df=x$scale);

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) qt(p, df=x$scale));
        }
        else if(x$distribution=="dinvgauss"){
            # Transform residuals for something meaningful
            # This is not 100% accurate, because the dispersion should change as well as mean...
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ-plot of Inverse Gaussian distribution";
            }
            ellipsis$x <- qinvgauss(ppoints(500), mean=1, dispersion=x$scale);

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) qinvgauss(p, mean=1, dispersion=x$scale));
        }
        else if(x$distribution=="dchisq"){
            message("Sorry, but we don't produce QQ plots for the Chi-Squared distribution");
        }
        else if(x$distribution=="dbeta"){
            message("Sorry, but we don't produce QQ plots for the Beta distribution");
        }
        else if(x$distribution=="dpois"){
            message("Sorry, but we don't produce QQ plots for the Poisson distribution");
        }
        else if(x$distribution=="dnbinom"){
            message("Sorry, but we don't produce QQ plots for the Negative Binomial distribution");
        }
    }

    # 7. Linear graph,
    plot5 <- function(x, ...){
        ellipsis <- list(...);
        if(!any(names(ellipsis)=="main")){
            ellipsis$main <- "Fit over time";
        }

        # If type and ylab are not provided, set them...
        if(!any(names(ellipsis)=="type")){
            ellipsis$type <- "l";
        }
        if(!any(names(ellipsis)=="ylab")){
            ellipsis$ylab <- all.vars(x$call$formula)[1];
        }
        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Time";
        }

        # Get the actuals and the fitted values
        ellipsis$x <- actuals(x);
        if(is.alm(x)){
            if(any(x$distribution==c("plogis","pnorm"))){
                ellipsis$x <- (ellipsis$x!=0)*1;
            }
        }
        yFitted <- fitted(x);

        if(legend){
            if(yFitted[length(yFitted)]>mean(yFitted)){
                legendPosition <- "bottomright";
            }
            else{
                legendPosition <- "topright";
            }
            if(!any(names(ellipsis)=="ylim")){
                ellipsis$ylim <- range(c(actuals(x),yFitted),na.rm=TRUE);
                if(legendPosition=="bottomright"){
                    ellipsis$ylim[1] <- ellipsis$ylim[1] - 0.2*diff(ellipsis$ylim);
                }
                else{
                    ellipsis$ylim[2] <- ellipsis$ylim[2] + 0.2*diff(ellipsis$ylim);
                }
            }
        }

        # Start plotting
        do.call(plot,ellipsis);
        lines(yFitted, col="purple");
        if(legend){
            legend(legendPosition,legend=c("Actuals","Fitted"),
                   col=c("black","purple"), lwd=rep(1,2), lty=c(1,1));
        }
    }

    # 8 and 9. Standardised / Studentised residuals vs time
    plot6 <- function(x, type="rstandard", ...){

        ellipsis <- list(...);
        if(type=="rstandard"){
            ellipsis$x <- rstandard(x);
            yName <- "Standardised";
        }
        else{
            ellipsis$x <- rstudent(x);
            yName <- "Studentised";
        }

        if(!any(names(ellipsis)=="main")){
            ellipsis$main <- paste0(yName," Residuals vs Time");
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Time";
        }
        if(!any(names(ellipsis)=="ylab")){
            ellipsis$ylab <- paste0(yName," Residuals");
        }

        # If type and ylab are not provided, set them...
        if(!any(names(ellipsis)=="type")){
            ellipsis$type <- "l";
        }

        # Get the IDs of outliers and statistic
        outliers <- outlierdummy(x, level=level, type=type);
        outliersID <- outliers$id;
        statistic <- outliers$statistic;
        # Analyse stuff in logarithms if the distribution is dinvgauss
        if(x$distribution=="dinvgauss"){
            ellipsis$x[] <- log(ellipsis$x);
            statistic[] <- log(statistic);
        }

        if(!any(names(ellipsis)=="ylim")){
            ellipsis$ylim <- range(c(ellipsis$x,statistic),na.rm=TRUE)*1.2;
        }

        if(legend){
            legendPosition <- "topright";
            ellipsis$ylim[2] <- ellipsis$ylim[2] + 0.2*diff(ellipsis$ylim);
            ellipsis$ylim[1] <- ellipsis$ylim[1] - 0.2*diff(ellipsis$ylim);
        }

        # Start plotting
        do.call(plot,ellipsis);
        if(is.occurrence(x$occurrence)){
            points(ellipsis$x);
        }
        if(length(outliersID)>0){
            points(outliersID, ellipsis$x[outliersID], pch=16);
            text(outliersID, ellipsis$x[outliersID], labels=outliersID, pos=(ellipsis$x[outliersID]>0)*2+1);
        }
        if(lowess){
            # Substitute NAs with the mean
            if(any(is.na(ellipsis$x))){
                ellipsis$x[is.na(ellipsis$x)] <- mean(ellipsis$x, na.rm=TRUE);
            }
            lines(lowess(c(1:length(ellipsis$x)),ellipsis$x), col="red");
        }
        abline(h=0, col="grey", lty=2);
        abline(h=statistic[1], col="red", lty=2);
        abline(h=statistic[2], col="red", lty=2);
        polygon(c(1:nobs(x), c(nobs(x):1)),
                c(rep(statistic[1],nobs(x)), rep(statistic[2],nobs(x))),
                col="lightgrey", border=NA, density=10);
        if(legend){
            legend(legendPosition,legend=c("Residuals",paste0(level*100,"% prediction interval")),
                   col=c("black","red"), lwd=rep(1,3), lty=c(1,1,2));
        }
    }

    # 10 and 11. ACF and PACF
    plot7 <- function(x, type="acf", ...){
        ellipsis <- list(...);

        if(!any(names(ellipsis)=="main")){
            if(type=="acf"){
                ellipsis$main <- "Autocorrelation Function of Residuals";
            }
            else{
                ellipsis$main <- "Partial Autocorrelation Function of Residuals";
            }
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Lags";
        }
        if(!any(names(ellipsis)=="ylab")){
            if(type=="acf"){
                ellipsis$ylab <- "ACF";
            }
            else{
                ellipsis$ylab <- "PACF";
            }
        }

        if(!any(names(ellipsis)=="ylim")){
            ellipsis$ylim <- c(-1,1);
        }

        if(type=="acf"){
            theValues <- acf(as.vector(residuals(x)), plot=FALSE)
        }
        else{
            theValues <- pacf(as.vector(residuals(x)), plot=FALSE);
        }
        ellipsis$x <- theValues$acf[-1];
        statistic <- qnorm(c((1-level)/2, (1+level)/2),0,sqrt(1/nobs(x)));

        ellipsis$type <- "h"

        do.call(plot,ellipsis);
        abline(h=0, col="black", lty=1);
        abline(h=statistic, col="red", lty=2);
        if(any(ellipsis$x>statistic[2] | ellipsis$x<statistic[1])){
            outliers <- which(ellipsis$x >statistic[2] | ellipsis$x <statistic[1]);
            points(outliers, ellipsis$x[outliers], pch=16);
            text(outliers, ellipsis$x[outliers], labels=outliers, pos=(ellipsis$x[outliers]>0)*2+1);
        }
    }

    # 12. Cook's distance over time
    plot8 <- function(x, ...){
        ellipsis <- list(...);
        if(!any(names(ellipsis)=="main")){
            ellipsis$main <- "Cook's distance over Time";
        }

        # If type and ylab are not provided, set them...
        if(!any(names(ellipsis)=="type")){
            ellipsis$type <- "h";
        }
        if(!any(names(ellipsis)=="ylab")){
            ellipsis$ylab <- "Cook's distance";
        }
        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Time";
        }

        # Get the cook's distance. Take abs() just in case... Not a very reasonable thing to do...
        ellipsis$x <- abs(cooks.distance(x));
        thresholdsF <- qf(c(0.5,0.75,0.95), nparam(x), nobs(x)-nparam(x))
        thresholdsColours <- c("red","red","red")
        thresholdsLty <- c(3,2,5)
        thresholdsLwd <- c(1,1,2)
        outliers <- which(ellipsis$x>=thresholdsF[2]);

        # Start plotting
        do.call(plot,ellipsis);
        for(i in 1:length(thresholdsF)){
            abline(h=thresholdsF[i], col=thresholdsColours[i], lty=thresholdsLty[i], lwd=thresholdsLwd[i]);
        }
        if(length(outliers)>0){
            text(outliers, ellipsis$x[outliers], labels=outliers, pos=2);
        }
        if(legend){
            legend("topright",
                   legend=paste0("F(",c(0.5,0.75,0.95),",",nparam(x),",",nobs(x)-nparam(x),")"),
                   col=thresholdsColours, lwd=thresholdsLwd, lty=thresholdsLty);
        }
    }

    if(any(which==1)){
        plot1(x, ...);
    }

    if(any(which==2)){
        plot2(x, ...);
    }

    if(any(which==3)){
        plot2(x, type="rstudent", ...);
    }

    if(any(which==4)){
        plot3(x, ...);
    }

    if(any(which==5)){
        plot3(x, type="squared", ...);
    }

    if(any(which==6)){
        plot4(x, ...);
    }

    if(any(which==7)){
        plot5(x, ...);
    }

    if(any(which==8)){
        plot6(x, ...);
    }

    if(any(which==9)){
        plot6(x, type="rstudent", ...);
    }

    if(any(which==10)){
        plot7(x, type="acf", ...);
    }

    if(any(which==11)){
        plot7(x, type="pacf", ...);
    }

    if(any(which==12)){
        plot8(x, ...);
    }

}

#' @export
plot.predict.greybox <- function(x, ...){
    yActuals <- actuals(x$model);
    yStart <- start(yActuals);
    yFrequency <- frequency(yActuals);
    yForecastStart <- time(yActuals)[length(yActuals)]+deltat(yActuals);

    if(!is.null(x$newdata)){
        yName <- all.vars(x$model$call$formula)[1];
        if(any(colnames(x$newdata)==yName)){
            yHoldout <- x$newdata[,yName];
            if(!any(is.na(yHoldout))){
                if(x$newdataProvided){
                    yActuals <- ts(c(yActuals,unlist(yHoldout)), start=yStart, frequency=yFrequency);
                }
                else{
                    yActuals <- ts(unlist(yHoldout), start=yForecastStart, frequency=yFrequency);
                }
                # If this is occurrence model, then transform actual to the occurrence
                if(any(x$distribution==c("pnorm","plogis"))){
                    yActuals <- (yActuals!=0)*1;
                }
            }
        }
    }

    # Change values of fitted and forecast, depending on whethere there was a newdata or not
    if(x$newdataProvided){
        yFitted <- ts(fitted(x$model), start=yStart, frequency=yFrequency);
        yForecast <- ts(x$mean, start=yForecastStart, frequency=yFrequency);
        vline <- TRUE;
    }
    else{
        yForecast <- ts(NA, start=yForecastStart, frequency=yFrequency);
        yFitted <- ts(x$mean, start=yStart, frequency=yFrequency);
        vline <- FALSE;
    }

    ellipsis <- list(...);
    ellipsis$actuals <- yActuals;
    ellipsis$forecast <- yForecast;
    ellipsis$fitted <- yFitted;
    ellipsis$vline <- vline;

    if(!is.null(x$lower) || !is.null(x$upper)){
        if(x$newdataProvided){
            yLower <- ts(x$lower, start=yForecastStart, frequency=yFrequency);
            yUpper <- ts(x$upper, start=yForecastStart, frequency=yFrequency);
        }
        else{
            yLower <- ts(x$lower, start=yStart, frequency=yFrequency);
            yUpper <- ts(x$upper, start=yStart, frequency=yFrequency);
        }

        if(is.matrix(x$level)){
            level <- x$level[1];
        }
        else{
            level <- x$level;
        }
        ellipsis$level <- level;
        ellipsis$lower <- yLower;
        ellipsis$upper <- yUpper;

        if((any(is.infinite(yLower)) & any(is.infinite(yUpper))) | (any(is.na(yLower)) & any(is.na(yUpper)))){
            ellipsis$lower[is.infinite(yLower) | is.na(yLower)] <- 0;
            ellipsis$upper[is.infinite(yUpper) | is.na(yUpper)] <- 0;
        }
        else if(any(is.infinite(yLower)) | any(is.na(yLower))){
            ellipsis$lower[is.infinite(yLower) | is.na(yLower)] <- 0;
        }
        else if(any(is.infinite(yUpper)) | any(is.na(yUpper))){
            ellipsis$upper <- NA;
        }
    }

    if(is.null(ellipsis$legend)){
        ellipsis$legend <- FALSE;
        ellipsis$parReset <- FALSE;
    }

    if(is.null(ellipsis$main)){
        if(x$newdataProvided){
            ellipsis$main <- paste0("Forecast for the variable ",colnames(x$model$data)[1]);
        }
        else{
            ellipsis$main <- paste0("Fitted values for the variable ",colnames(x$model$data)[1]);
        }
    }

    do.call(graphmaker,ellipsis);
}

#' @export
plot.coef.greyboxD <- function(x, ...){
    ellipsis <- list(...);
    # If type and ylab are not provided, set them...
    if(!any(names(ellipsis)=="type")){
        ellipsis$type <- "l";
    }
    if(!any(names(ellipsis)=="ylab")){
        ellipsis$ylab <- "Importance";
    }
    if(!any(names(ellipsis)=="ylim")){
        ellipsis$ylim <- c(0,1);
    }

    ourData <- x$importance;
    # We are not interested in intercept, so skip it in plot

    parDefault <- par(no.readonly=TRUE);

    pages <- ceiling((ncol(ourData)-1) / 8);
    perPage <- ceiling((ncol(ourData)-1) / pages);
    if(pages>1){
        parCols <- ceiling(perPage/4);
        perPage <- ceiling(perPage/parCols);
    }
    else{
        parCols <- 1;
    }

    parDims <- c(perPage,parCols);
    par(mfcol=parDims);

    if(pages>1){
        message(paste0("Too many variables. Ploting several per page, on ",pages," pages."));
    }

    for(i in 2:ncol(ourData)){
        ellipsis$x <- ourData[,i];
        ellipsis$main <- colnames(ourData)[i];
        do.call(plot,ellipsis);
    }

    par(parDefault);
}

#' @importFrom grDevices rgb
#' @export
plot.rollingOrigin <- function(x, ...){
    y <- x$actuals;
    yDeltat <- deltat(y);

    # How many tables we have
    dimsOfHoldout <- dim(x$holdout);
    dimsOfThings <- lapply(x,dim);
    thingsToPlot <- 0;
    # 1 - actuals, 2 - holdout
    for(i in 3:length(dimsOfThings)){
        thingsToPlot <- thingsToPlot + all(dimsOfThings[[i]]==dimsOfHoldout)*1;
    }

    # Define basic parameters
    co <- !any(is.na(x$holdout[,ncol(x$holdout)]));
    h <- nrow(x$holdout);
    roh <- ncol(x$holdout);

    # Define the start of the RO
    roStart <- length(y)-h;
    roStart <- start(y)[1]+yDeltat*(roStart-roh+(h-1)*(!co));

    # Start plotting
    plot(y, ylab="Actuals", ylim=range(min(unlist(lapply(x,min,na.rm=T)),na.rm=T),
                                       max(unlist(lapply(x,max,na.rm=T)),na.rm=T),
                                       na.rm=TRUE),
         type="l", ...);
    abline(v=roStart, col="red", lwd=2);
    for(j in 1:thingsToPlot){
        colCurrent <- rgb((j-1)/thingsToPlot,0,(thingsToPlot-j+1)/thingsToPlot,1);
        for(i in 1:roh){
            points(roStart+i*yDeltat,x[[2+j]][1,i],col=colCurrent,pch=16);
            lines(c(roStart + (0:(h-1)+i)*yDeltat),c(x[[2+j]][,i]),col=colCurrent);
        }
    }
}

#### Print ####
#' @export
print.greybox <- function(x, ...){
    cat("Call:\n");
    print(x$call);
    cat("\nCoefficients:\n");
    print(coef(x));
}

#' @export
print.coef.greyboxD <- function(x, ...){
    print(x$coefficients);
}

#' @export
print.association <- function(x, ...){
    ellipsis <- list(...);
    if(!any(names(ellipsis)=="digits")){
        digits <- 4;
    }
    else{
        digits <- ellipsis$digits;
    }

    cat("Associations: ")
    cat("\nvalues:\n"); print(round(x$value,digits));
    cat("\np-values:\n"); print(round(x$p.value,digits));
    cat("\ntypes:\n"); print(x$type);
    cat("\n");
}

#' @export
print.pcor <- function(x, ...){
    ellipsis <- list(...);
    if(!any(names(ellipsis)=="digits")){
        digits <- 4;
    }
    else{
        digits <- ellipsis$digits;
    }

    cat(paste0("Partial correlations using ",x$method," method: "))
    cat("\nvalues:\n"); print(round(x$value,digits));
    cat("\np-values:\n"); print(round(x$p.value,digits));
    cat("\n");
}

#' @export
print.cramer <- function(x, ...){
    ellipsis <- list(...);
    if(!any(names(ellipsis)=="digits")){
        digits <- 4;
    }
    else{
        digits <- ellipsis$digits;
    }

    cat("Cramer's V: "); cat(round(x$value,digits));
    cat("\nChi^2 statistics = "); cat(round(x$statistic,digits));
    cat(", df: "); cat(x$df);
    cat(", p-value: "); cat(round(x$p.value,digits));
    cat("\n");
}

#' @export
print.mcor <- function(x, ...){
    ellipsis <- list(...);
    if(!any(names(ellipsis)=="digits")){
        digits <- 4;
    }
    else{
        digits <- ellipsis$digits;
    }

    cat("Multiple correlations value: "); cat(round(x$value,digits));
    cat("\nF-statistics = "); cat(round(x$statistic,digits));
    cat(", df: "); cat(x$df);
    cat(", df resid: "); cat(x$df.residual);
    cat(", p-value: "); cat(round(x$p.value,digits));
    cat("\n");
}

#' @export
print.summary.alm <- function(x, ...){
    ellipsis <- list(...);
    if(!any(names(ellipsis)=="digits")){
        digits <- 4;
    }
    else{
        digits <- ellipsis$digits;
    }

    distrib <- switch(x$distribution,
                      "dnorm" = "Normal",
                      "dgnorm" = paste0("Generalised Normal Distribution with shape=",round(x$other$beta,digits)),
                      "dlgnorm" = paste0("Log Generalised Normal Distribution with shape=",round(x$other$beta,digits)),
                      "dlogis" = "Logistic",
                      "dlaplace" = "Laplace",
                      "dllaplace" = "Log Laplace",
                      "dalaplace" = paste0("Asymmetric Laplace with alpha=",round(x$other$alpha,digits)),
                      "dt" = paste0("Student t with nu=",round(x$other$nu, digits)),
                      "ds" = "S",
                      "dls" = "Log S",
                      "dfnorm" = "Folded Normal",
                      "dlnorm" = "Log Normal",
                      "dbcnorm" = paste0("Box-Cox Normal with lambda=",round(x$other$lambdaBC,digits)),
                      "dlogitnorm" = "Logit Normal",
                      "dinvgauss" = "Inverse Gaussian",
                      "dchisq" = paste0("Chi-Squared with nu=",round(x$other$nu,digits)),
                      "dpois" = "Poisson",
                      "dnbinom" = paste0("Negative Binomial with size=",round(x$other$size,digits)),
                      "dbeta" = "Beta",
                      "plogis" = "Cumulative logistic",
                      "pnorm" = "Cumulative normal"
    );
    if(is.occurrence(x$occurrence)){
        distribOccurrence <- switch(x$occurrence$distribution,
                                    "plogis" = "Cumulative logistic",
                                    "pnorm" = "Cumulative normal"
        );
        distrib <- paste0("Mixture of ", distrib," and ", distribOccurrence);
    }

    cat(paste0("Response variable: ", paste0(x$responseName,collapse="")));
    cat(paste0("\nDistribution used in the estimation: ", distrib));
    cat(paste0("\nLoss function used in estimation: ",x$loss));
    if(any(x$loss==c("LASSO","RIDGE"))){
        cat(paste0(" with lambda=",round(x$other$lambda,digits)));
    }
    if(!is.null(x$arima)){
        cat(paste0("\n",x$arima," components were included in the model"));
    }
    if(x$bootstrap){
        cat("\nBootstrap was used for the estimation of uncertainty of parameters");
    }
    cat("\nCoefficients:\n");
    print(round(x$coefficients,digits));

    cat("\nError standard deviation: "); cat(round(sqrt(x$s2),digits));
    cat("\nSample size: "); cat(x$dfTable[1]);
    cat("\nNumber of estimated parameters: "); cat(x$dfTable[2]);
    cat("\nNumber of degrees of freedom: "); cat(x$dfTable[3]);
    if(!is.null(x$ICs)){
        cat("\nInformation criteria:\n");
        print(round(x$ICs,digits));
    }
    cat("\n");
}

#' @export
print.summary.greybox <- function(x, ...){
    ellipsis <- list(...);
    if(!any(names(ellipsis)=="digits")){
        digits <- 4;
    }
    else{
        digits <- ellipsis$digits;
    }

    distrib <- switch(x$distribution,
                      "dnorm" = "Normal",
                      "dgnorm" = "Generalised Normal Distribution",
                      "dlgnorm" = "Log Generalised Normal Distribution",
                      "dlogis" = "Logistic",
                      "dlaplace" = "Laplace",
                      "dllaplace" = "Log Laplace",
                      "dalaplace" = "Asymmetric Laplace",
                      "dt" = "Student t",
                      "ds" = "S",
                      "ds" = "Log S",
                      "dfnorm" = "Folded Normal",
                      "dlnorm" = "Log Normal",
                      "dbcnorm" = "Box-Cox Normal",
                      "dlogitnorm" = "Logit Normal",
                      "dinvgauss" = "Inverse Gaussian",
                      "dchisq" = "Chi-Squared",
                      "dpois" = "Poisson",
                      "dnbinom" = "Negative Binomial",
                      "dbeta" = "Beta",
                      "plogis" = "Cumulative logistic",
                      "pnorm" = "Cumulative normal"
    );

    cat(paste0("Response variable: ", paste0(x$responseName,collapse=""),"\n"));
    cat(paste0("Distribution used in the estimation: ", distrib));
    if(!is.null(x$arima)){
        cat(paste0("\n",x$arima," components were included in the model"));
    }
    cat("\nCoefficients:\n");
    print(round(x$coefficients,digits));
    cat("\nError standard deviation: "); cat(round(x$sigma,digits));
    cat("\nSample size: "); cat(x$dfTable[1]);
    cat("\nNumber of estimated parameters: "); cat(x$dfTable[2]);
    cat("\nNumber of degrees of freedom: "); cat(x$dfTable[3]);
    cat("Information criteria:\n");
    print(round(x$ICs,digits));
}

#' @export
print.summary.greyboxC <- function(x, ...){
    ellipsis <- list(...);
    if(!any(names(ellipsis)=="digits")){
        digits <- 4;
    }
    else{
        digits <- ellipsis$digits;
    }

    distrib <- switch(x$distribution,
                      "dnorm" = "Normal",
                      "dgnorm" = "Generalised Normal Distribution",
                      "dlgnorm" = "Log Generalised Normal Distribution",
                      "dlogis" = "Logistic",
                      "dlaplace" = "Laplace",
                      "dllaplace" = "Log Laplace",
                      "dalaplace" = "Asymmetric Laplace",
                      "dt" = "Student t",
                      "ds" = "S",
                      "dls" = "Log S",
                      "dfnorm" = "Folded Normal",
                      "dlnorm" = "Log Normal",
                      "dbcnorm" = "Box-Cox Normal",
                      "dlogitnorm" = "Logit Normal",
                      "dinvgauss" = "Inverse Gaussian",
                      "dchisq" = "Chi-Squared",
                      "dpois" = "Poisson",
                      "dnbinom" = "Negative Binomial",
                      "dbeta" = "Beta",
                      "plogis" = "Cumulative logistic",
                      "pnorm" = "Cumulative normal"
    );

    # The name of the model used
    if(is.null(x$dynamic)){
        cat(paste0("The ",x$ICType," combined model\n"));
    }
    else{
        cat(paste0("The p",x$ICType," combined model\n"));
    }
    cat(paste0("Response variable: ", paste0(x$responseName,collapse=""),"\n"));
    cat(paste0("Distribution used in the estimation: ", distrib));
    cat("\nCoefficients:\n");
    print(round(x$coefficients,digits));
    cat("\nError standard deviation: "); cat(round(x$sigma,digits));
    cat("\nSample size: "); cat(round(x$dfTable[1],digits));
    cat("\nNumber of estimated parameters: "); cat(round(x$dfTable[2],digits));
    cat("\nNumber of degrees of freedom: "); cat(round(x$dfTable[3],digits));
    cat("\nApproximate combined information criteria:\n");
    print(round(x$ICs,digits));
}

#' @export
print.predict.greybox <- function(x, ...){
    ourMatrix <- as.matrix(x$mean);
    colnames(ourMatrix) <- "Mean";
    if(!is.null(x$lower)){
        ourMatrix <- cbind(ourMatrix, x$lower, x$upper);
        if(is.matrix(x$level)){
            level <- colMeans(x$level)[-1];
        }
        else{
            level <- x$level;
        }
    }
    print(ourMatrix);
}

#' @export
print.rollingOrigin <- function(x, ...){
    co <- !any(is.na(x$holdout[,ncol(x$holdout)]));
    h <- nrow(x$holdout);
    roh <- ncol(x$holdout);

    if(co){
        cat(paste0("Rolling Origin with constant holdout was done.\n"));
    }
    else{
        cat(paste0("Rolling Origin with decreasing holdout was done.\n"));
    }
    cat(paste0("Forecast horizon is ",h,"\n"));
    cat(paste0("Number of origins is ",roh,"\n"));
}

#### Regression diagnostics ####

#' @importFrom stats hatvalues hat
#' @export
hatvalues.greybox <- function(model, ...){
    # Prepare the hat values
    if(any(names(coef(model))=="(Intercept)")){
        xreg <- model$data;
        xreg[,1] <- 1;
    }
    else{
        xreg <- model$data[,-1,drop=FALSE];
    }
    # Hatvalues for different distributions
    if(any(model$distribution==c("dt","dnorm","dlnorm","dbcnorm","dlogitnorm","dnbinom","dpois"))){
        hatValue <- hat(xreg);
    }
    else{
        hatValue <- diag(xreg %*% vcov(model) %*% t(xreg))/sigma(model)^2;
    }
    names(hatValue) <- names(actuals(model));
    return(hatValue);
}

#' @export
residuals.greybox <- function(object, ...){
    errors <- object$residuals;
    names(errors) <- names(actuals(object));
    ellipsis <- list(...);
    # Remove zeroes if they are not needed
    if(!is.null(ellipsis$all) && (!ellipsis$all)){
        if(is.occurrence(object$occurrence)){
            errors <- errors[actuals(object)!=0];
        }
    }
    return(errors)
}

#' @importFrom stats rstandard
#' @export
rstandard.greybox <- function(model, ...){
    obs <- nobs(model);
    df <- obs - nparam(model);
    errors <- residuals(model, ...);
    # If this is an occurrence model, then only modify the non-zero obs
    if(is.occurrence(model$occurrence)){
        residsToGo <- (actuals(model$occurrence)!=0);
    }
    else{
        residsToGo <- rep(TRUE,obs);
    }
    # The proper residuals with leverage are currently done only for normal-based distributions
    if(any(model$distribution==c("dt","dnorm","dlnorm","dbcnorm","dlogitnorm","dnbinom","dpois"))){
        errors[] <- errors / (sigma(model)*sqrt(1-hatvalues(model)));
    }
    else if(any(model$distribution==c("ds","dls"))){
        errors[residsToGo] <- (errors[residsToGo] - mean(errors[residsToGo])) / (model$scale * obs / df)^2;
    }
    else if(any(model$distribution==c("dgnorm","dlgnorm"))){
        errors[residsToGo] <- ((errors[residsToGo] - mean(errors[residsToGo])) /
                         (model$scale^model$other$beta * obs / df)^{1/model$other$beta});
    }
    else if(model$distribution=="dinvgauss"){
        errors[residsToGo] <- errors[residsToGo] / mean(errors[residsToGo]);
    }
    else{
        errors[residsToGo] <- (errors[residsToGo] - mean(errors[residsToGo])) / (model$scale * obs / df);
    }

    # Fill in values with NAs if there is occurrence model
    if(is.occurrence(model$occurrence)){
        errors[!residsToGo] <- NA;
    }

    return(errors);
}

#' @importFrom stats rstudent
#' @export
rstudent.greybox <- function(model, ...){
    obs <- nobs(model);
    df <- obs - nparam(model) - 1;
    rstudentised <- errors <- residuals(model, ...);
    # If this is an occurrence model, then only modify the non-zero obs
    if(is.occurrence(model$occurrence)){
        residsToGo <- (actuals(model$occurrence)!=0);
    }
    else{
        residsToGo <- rep(TRUE,obs);
    }

    # The proper residuals with leverage are currently done only for normal-based distributions
    if(any(model$distribution==c("dt","dnorm","dlnorm","dlogitnorm","dbcnorm"))){
        # Prepare the hat values
        if(any(names(coef(model))=="(Intercept)")){
            xreg <- model$data;
            xreg[,1] <- 1;
        }
        else{
            xreg <- model$data[,-1,drop=FALSE];
        }
        hatValues <- hat(xreg);
        errors[] <- errors - mean(errors);
        for(i in which(residsToGo)){
            rstudentised[i] <- errors[i] / sqrt(sum(errors[-i]^2) / df * (1-hatValues[i]));
        }
    }
    else if(any(model$distribution==c("ds","dls"))){
        # This is an approximation from the vcov matrix
        # hatValues <- diag(xreg %*% vcov(model) %*% t(xreg))/sigma(model)^2;
        errors[] <- errors - mean(errors);
        for(i in which(residsToGo)){
            rstudentised[i] <- errors[i] /  (sum(sqrt(abs(errors[-i]))) / (2*df))^2;
        }
    }
    else if(any(model$distribution==c("dlaplace","dllaplace"))){
        errors[] <- errors - mean(errors);
        for(i in which(residsToGo)){
            rstudentised[i] <- errors[i] / (sum(abs(errors[-i])) / df);
        }
    }
    else if(model$distribution=="dalaplace"){
        for(i in which(residsToGo)){
            rstudentised[i] <- errors[i] / (sum(errors[-i] * (model$other$alpha - (errors[-i]<=0)*1)) / df);
        }
    }
    else if(any(model$distribution==c("dgnorm","dlgnorm"))){
        for(i in which(residsToGo)){
            rstudentised[i] <- errors[i] /  (sum(abs(errors[-i])^model$other$beta) * (model$other$beta/df))^{1/model$other$beta};
        }
    }
    else if(model$distribution=="dlogis"){
        errors[] <- errors - mean(errors);
        for(i in which(residsToGo)){
            rstudentised[i] <- errors[i] / (sqrt(sum(errors[-i]^2) / df) * sqrt(3) / pi);
        }
    }
    else if(model$distribution=="dinvgauss"){
        for(i in which(residsToGo)){
            rstudentised[i] <- errors[i] / mean(errors[-i]);
        }
    }
    else{
        for(i in which(residsToGo)){
            rstudentised[i] <- errors[i] / sqrt(sum(errors[-i]^2) / df);
        }
    }

    # Fill in values with NAs if there is occurrence model
    if(is.occurrence(model$occurrence)){
        rstudentised[!residsToGo] <- NA;
    }

    return(rstudentised);
}

#' @importFrom stats cooks.distance
#' @export
cooks.distance.greybox <- function(model, ...){
    # Number of parameters
    nParam <- nparam(model);
    # Hat values
    hatValues <- hatvalues(model);
    # Standardised residuals
    errors <- rstandard(model);

    return(errors^2 / nParam * hatValues/(1-hatValues));
}

#' Outlier detection and matrix creation
#'
#' Function detects outliers and creates a matrix with dummy variables. Only point
#' outliers are considered (no level shifts).
#'
#' The detection is done based on the type of distribution used and confidence level
#' specified by user.
#'
#' @template author
#'
#' @param object Model estimated using one of the functions of smooth package.
#' @param level Confidence level to use. Everything that is outside the constructed
#' bounds based on that is flagged as outliers.
#' @param type Type of residuals to use: either standardised or studentised.
#' @param ... Other parameters. Not used yet.
#' @return The class "outlierdummy", which contains the list:
#' \itemize{
#' \item outliers - the matrix with the dummy variables, flagging outliers;
#' \item statistic - the value of the statistic for the normalised variable;
#' \item id - the ids of the outliers (which observations have them);
#' \item level - the confidence level used in the process;
#' \item type - the type of the residuals used.
#' }
#'
#' @seealso \link[stats]{influence.measures}
#' @examples
#'
#' # Generate the data with S distribution
#' xreg <- cbind(rnorm(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rs(100,0,3),xreg)
#' colnames(xreg) <- c("y","x1","x2")
#'
#' # Fit the normal distribution model
#' ourModel <- alm(y~x1+x2, xreg, distribution="dnorm")
#'
#' # Detect outliers
#' xregOutlierDummy <- outlierdummy(ourModel)
#'
#' @export outlierdummy
outlierdummy <-  function(object, level=0.999, type=c("rstandard","rstudent"),
                          ...) UseMethod("outlierdummy")

#' @export
outlierdummy.default <- function(object, level=0.999, type=c("rstandard","rstudent"), ...){
    # Function returns the matrix of dummies with outliers
    type <- match.arg(type);
    errors <- switch(type,"rstandard"=rstandard(object),"rstudent"=rstudent(object));
    statistic <- qnorm(c((1-level)/2, (1+level)/2), 0, 1);
    outliersID <- which(errors>statistic[2] | errors <statistic[1]);
    outliersNumber <- length(outliersID);
    if(outliersNumber>0){
        outliers <- matrix(0, nobs(object), outliersNumber,
                           dimnames=list(rownames(object$data),
                                         paste0("outlier",c(1:outliersNumber))));
        outliers[cbind(outliersID,c(1:outliersNumber))] <- 1;
    }
    else{
        outliers <- NULL;
    }

    return(structure(list(outliers=outliers, statistic=statistic, id=outliersID,
                          level=level, type=type),
                     class="outlierdummy"));
}

#' @export
outlierdummy.alm <- function(object, level=0.999, type=c("rstandard","rstudent"), ...){
    # Function returns the matrix of dummies with outliers
    type <- match.arg(type);
    errors <- switch(type,"rstandard"=rstandard(object),"rstudent"=rstudent(object));
    statistic <- switch(object$distribution,
                      "dlaplace"=,
                      "dllaplace"=qlaplace(c((1-level)/2, (1+level)/2), 0, 1),
                      "dalaplace"=qalaplace(c((1-level)/2, (1+level)/2), 0, 1, object$other$alpha),
                      "dlogis"=qlogis(c((1-level)/2, (1+level)/2), 0, 1),
                      "dt"=qt(c((1-level)/2, (1+level)/2), nobs(object)-nparam(object)),
                      "dgnorm"=,
                      "dlgnorm"=qgnorm(c((1-level)/2, (1+level)/2), 0, 1, object$other$beta),
                      "ds"=,
                      "dls"=qs(c((1-level)/2, (1+level)/2), 0, 1),
                      # In the next one, the scale is debiased, taking n-k into account
                      "dinvgauss"=qinvgauss(c((1-level)/2, (1+level)/2), mean=1,
                                            dispersion=object$scale * nobs(object) /
                                                (nobs(object)-nparam(object))),
                      qnorm(c((1-level)/2, (1+level)/2), 0, 1));
    outliersID <- which(errors>statistic[2] | errors<statistic[1]);
    outliersNumber <- length(outliersID);
    if(outliersNumber>0){
        outliers <- matrix(0, nobs(object), outliersNumber,
                           dimnames=list(rownames(object$data),
                                         paste0("outlier",c(1:outliersNumber))));
        outliers[cbind(outliersID,c(1:outliersNumber))] <- 1;
    }
    else{
        outliers <- NULL;
    }

    return(structure(list(outliers=outliers, statistic=statistic, id=outliersID,
                          level=level, type=type),
                     class="outlierdummy"));
}

#' @export
print.outlierdummy <- function(x, ...){
    cat(paste0("Number of identified outliers: ", length(x$id),
               "\nConfidence level: ",x$level,
               "\nType of residuals: ",x$type, "\n"));
}

#### Summary ####
#' @importFrom stats summary.lm
#' @export
summary.greybox <- function(object, level=0.95, ...){
    ourReturn <- summary.lm(object, ...);
    errors <- residuals(object);
    obs <- nobs(object, all=TRUE);

    # Collect parameters and their standard errors
    parametersTable <- ourReturn$coefficients[,1:2];
    parametersTable <- cbind(parametersTable,confint(object, level=level));
    rownames(parametersTable) <- names(coef(object));
    colnames(parametersTable) <- c("Estimate","Std. Error",
                                   paste0("Lower ",(1-level)/2*100,"%"),
                                   paste0("Upper ",(1+level)/2*100,"%"));
    ourReturn$coefficients <- parametersTable;

    ICs <- c(AIC(object),AICc(object),BIC(object),BICc(object));
    names(ICs) <- c("AIC","AICc","BIC","BICc");
    ourReturn$ICs <- ICs;
    ourReturn$distribution <- object$distribution;
    ourReturn$responseName <- formula(object)[[2]];

    # Table with degrees of freedom
    dfTable <- c(obs,nparam(object),obs-nparam(object));
    names(dfTable) <- c("n","k","df");

    ourReturn$r.squared <- 1 - sum(errors^2) / sum((actuals(object)-mean(actuals(object)))^2);
    ourReturn$adj.r.squared <- 1 - (1 - ourReturn$r.squared) * (obs - 1) / (dfTable[3]);

    ourReturn$dfTable <- dfTable;
    ourReturn$arima <- object$other$arima;

    ourReturn$bootstrap <- FALSE;

    ourReturn <- structure(ourReturn,class="summary.greybox");
    return(ourReturn);
}

#' @aliases summary.alm
#' @rdname coef.alm
#' @export
summary.alm <- function(object, level=0.95, bootstrap=FALSE, ...){
    errors <- residuals(object);
    obs <- nobs(object, all=TRUE);

    # Collect parameters and their standard errors
    parametersConfint <- confint(object, level=level, bootstrap=bootstrap, ...);
    parametersTable <- cbind(coef(object),parametersConfint);
    rownames(parametersTable) <- names(coef(object));
    colnames(parametersTable) <- c("Estimate","Std. Error",
                                   paste0("Lower ",(1-level)/2*100,"%"),
                                   paste0("Upper ",(1+level)/2*100,"%"));
    ourReturn <- list(coefficients=parametersTable);

    # If there is a likelihood, then produce ICs
    if(!is.na(logLik(object))){
        ICs <- c(AIC(object),AICc(object),BIC(object),BICc(object));
        names(ICs) <- c("AIC","AICc","BIC","BICc");
        ourReturn$ICs <- ICs;
    }
    ourReturn$distribution <- object$distribution;
    ourReturn$loss <- object$loss;
    ourReturn$occurrence <- object$occurrence;
    ourReturn$other <- object$other;
    ourReturn$responseName <- formula(object)[[2]];

    # Table with degrees of freedom
    dfTable <- c(obs,nparam(object),obs-nparam(object));
    names(dfTable) <- c("n","k","df");

    ourReturn$r.squared <- 1 - sum(errors^2) / sum((actuals(object)-mean(actuals(object)))^2);
    ourReturn$adj.r.squared <- 1 - (1 - ourReturn$r.squared) * (obs - 1) / (dfTable[3]);

    ourReturn$dfTable <- dfTable;
    ourReturn$arima <- object$other$arima;
    ourReturn$s2 <- sigma(object)^2;

    ourReturn$bootstrap <- bootstrap;

    ourReturn <- structure(ourReturn,class=c("summary.alm","summary.greybox"));
    return(ourReturn);
}

#' @export
summary.greyboxC <- function(object, level=0.95, ...){

    # Extract the values from the object
    errors <- residuals(object);
    obs <- nobs(object);
    parametersTable <- cbind(coef(object),sqrt(abs(diag(vcov(object)))),object$importance);

    # Calculate the quantiles for parameters and add them to the table
    parametersTable <- cbind(parametersTable,confint(object, level=level));
    rownames(parametersTable) <- names(coef(object));
    colnames(parametersTable) <- c("Estimate","Std. Error","Importance",
                                   paste0("Lower ",(1-level)/2*100,"%"),
                                   paste0("Upper ",(1+level)/2*100,"%"));

    # Extract degrees of freedom
    df <- c(object$df, object$df.residual, object$rank);
    # Calculate s.e. of residuals
    residSE <- sqrt(sum(errors^2)/df[2]);

    ICs <- c(AIC(object),AICc(object),BIC(object),BICc(object));
    names(ICs) <- c("AIC","AICc","BIC","BICc");

    # Table with degrees of freedom
    dfTable <- c(obs, nparam(object), object$df.residual);
    names(dfTable) <- c("n","k","df");

    R2 <- 1 - sum(errors^2) / sum((actuals(object)-mean(actuals(object)))^2);
    R2Adj <- 1 - (1 - R2) * (obs - 1) / (dfTable[3]);

    ourReturn <- structure(list(coefficients=parametersTable, sigma=residSE,
                                ICs=ICs, ICType=object$ICType, df=df, r.squared=R2, adj.r.squared=R2Adj,
                                distribution=object$distribution, responseName=formula(object)[[2]],
                                dfTable=dfTable, bootstrap=FALSE),
                           class=c("summary.greyboxC","summary.greybox"));
    return(ourReturn);
}

#' @export
summary.greyboxD <- function(object, level=0.95, ...){

    # Extract the values from the object
    errors <- residuals(object);
    obs <- nobs(object);
    parametersTable <- cbind(coef.greybox(object),sqrt(abs(diag(vcov(object)))),apply(object$importance,2,mean));

    parametersConfint <- confint(object, level=level);
    # Calculate the quantiles for parameters and add them to the table
    parametersTable <- cbind(parametersTable,parametersConfint);

    rownames(parametersTable) <- names(coef.greybox(object));
    colnames(parametersTable) <- c("Estimate","Std. Error","Importance",
                                   paste0("Lower ",(1-level)/2*100,"%"),
                                   paste0("Upper ",(1+level)/2*100,"%"));

    # Extract degrees of freedom
    df <- c(object$df, object$df.residual, object$rank);
    # Calculate s.e. of residuals
    residSE <- sqrt(sum(errors^2)/df[2]);

    ICs <- c(AIC(object),AICc(object),BIC(object),BICc(object));
    names(ICs) <- c("AIC","AICc","BIC","BICc");

    R2 <- 1 - sum(errors^2) / sum((actuals(object)-mean(actuals(object)))^2)
    R2Adj <- 1 - (1 - R2) * (obs - 1) / (obs - df[1]);

    # Table with degrees of freedom
    dfTable <- c(nobs(object), nparam(object), object$df.residual);
    names(dfTable) <- c("n","k","df");

    ourReturn <- structure(list(coefficients=parametersTable, sigma=residSE,
                                dynamic=coef(object)$dynamic,
                                ICs=ICs, ICType=object$ICType, df=df, r.squared=R2, adj.r.squared=R2Adj,
                                distribution=object$distribution, responseName=formula(object)[[2]],
                                nobs=nobs(object), nparam=nparam(object), dfTable=dfTable, bootstrap=FALSE),
                           class=c("summary.greyboxC","summary.greybox"));
    return(ourReturn);
}

#' @export
summary.lmGreybox <- function(object, level=0.95, ...){
    parametersTable <- cbind(coef(object),sqrt(diag(vcov(object))));
    parametersTable <- cbind(parametersTable, parametersTable[,1]+qnorm((1-level)/2,0,parametersTable[,2]),
                             parametersTable[,1]+qnorm((1+level)/2,0,parametersTable[,2]));
    colnames(parametersTable) <- c("Estimate","Std. Error",
                                   paste0("Lower ",(1-level)/2*100,"%"),
                                   paste0("Upper ",(1+level)/2*100,"%"));
    return(parametersTable)
}


#' @export
as.data.frame.summary.greybox <- function(x, ...){
    return(as.data.frame(x$coefficients, ...));
}

#' @importFrom texreg extract createTexreg
extract.greybox <- function(model, ...){
    summaryALM <- summary(model, ...);
    tr <- extract(summaryALM);
    return(tr)
}

extract.summary.greybox <- function(model, ...){
    gof <- c(model$dfTable, model$ICs)
    gof.names <- c("Num.\\ obs.", "Num.\\ param.", "Num.\\ df", names(model$ICs))

    tr <- createTexreg(
        coef.names=rownames(model$coefficients),
        coef=model$coef[, 1],
        se=model$coef[, 2],
        ci.low=model$coef[, 3],
        ci.up=model$coef[, 4],
        gof.names=gof.names,
        gof=gof
    )
    return(tr)
}

#' @importFrom methods setMethod
setMethod("extract", signature=className("alm","greybox"), definition=extract.greybox)
setMethod("extract", signature=className("greybox","greybox"), definition=extract.greybox)
setMethod("extract", signature=className("greyboxC","greybox"), definition=extract.greybox)
setMethod("extract", signature=className("greyboxD","greybox"), definition=extract.greybox)
setMethod("extract", signature=className("summary.alm","greybox"), definition=extract.summary.greybox)
setMethod("extract", signature=className("summary.greybox","greybox"), definition=extract.summary.greybox)
setMethod("extract", signature=className("summary.greyboxC","greybox"), definition=extract.summary.greybox)

#### Predictions and forecasts ####

#' @rdname predict.greybox
#' @importFrom stats predict qchisq qlnorm qlogis qpois qnbinom qbeta
#' @importFrom statmod qinvgauss
#' @export
predict.alm <- function(object, newdata=NULL, interval=c("none", "confidence", "prediction"),
                            level=0.95, side=c("both","upper","lower"), ...){
    if(is.null(newdata)){
        newdataProvided <- FALSE;
    }
    else{
        newdataProvided <- TRUE;
    }
    interval <- match.arg(interval);
    side <- match.arg(side);
    if(!is.null(newdata)){
        h <- nrow(newdata);
    }
    else{
        h <- nobs(object);
    }
    levelOriginal <- level;
    nLevels <- length(level);

    ariOrderNone <- is.null(object$other$polynomial);
    if(ariOrderNone){
        greyboxForecast <- predict.greybox(object, newdata, interval, level, side=side, ...);
    }
    else{
        greyboxForecast <- predict.almari(object, newdata, interval, level, side=side, ...);
    }
    greyboxForecast$location <- greyboxForecast$mean;
    greyboxForecast$scale <- sqrt(greyboxForecast$variance);
    greyboxForecast$distribution <- object$distribution;

    # If there is an occurrence part of the model, use it
    if(is.occurrence(object$occurrence)){
        occurrence <- predict(object$occurrence, newdata, interval=interval, level=level, side=side, ...);
        # Reset horizon, just in case
        h <- length(occurrence$mean);
        # Create a matrix of levels for each horizon and level
        level <- matrix(level, h, nLevels, byrow=TRUE);
        # The probability of having zero should be subtracted from that thing...
        if(interval=="prediction"){
            level[] <- (level - (1 - occurrence$mean)) / occurrence$mean;
        }
        level[level<0] <- 0;
        greyboxForecast$occurrence <- occurrence;
    }
    else{
        # Create a matrix of levels for each horizon and level
        level <- matrix(level, h, nLevels, byrow=TRUE);
    }

    # levelLow and levelUp are matrices here...
    levelLow <- levelUp <- matrix(level, h, nLevels, byrow=TRUE);
    if(side=="upper"){
        levelLow[] <- 0;
        levelUp[] <- level;
    }
    else if(side=="lower"){
        levelLow[] <- 1-level;
        levelUp[] <- 1;
    }
    else{
        levelLow[] <- (1-level)/2;
        levelUp[] <- (1+level)/2;
    }

    levelLow[levelLow<0] <- 0;
    levelUp[levelUp<0] <- 0;

    if(object$distribution=="dnorm"){
        if(is.occurrence(object$occurrence) & interval!="none"){
            greyboxForecast$lower[] <- qnorm(levelLow,greyboxForecast$mean,greyboxForecast$scale);
            greyboxForecast$upper[] <- qnorm(levelUp,greyboxForecast$mean,greyboxForecast$scale);
        }
    }
    else if(object$distribution=="dlaplace"){
        # Use the connection between the variance and MAE in Laplace distribution
        scaleValues <- sqrt(greyboxForecast$variances/2);
        if(interval=="prediction"){
            for(i in 1:nLevels){
                greyboxForecast$lower[,i] <- qlaplace(levelLow[,i],greyboxForecast$mean,scaleValues);
                greyboxForecast$upper[,i] <- qlaplace(levelUp[,i],greyboxForecast$mean,scaleValues);
            }
        }
        greyboxForecast$scale <- scaleValues;
    }
    else if(object$distribution=="dllaplace"){
        # Use the connection between the variance and MAE in Laplace distribution
        scaleValues <- sqrt(greyboxForecast$variances/2);
        if(interval=="prediction"){
            for(i in 1:nLevels){
                greyboxForecast$lower[,i] <- exp(qlaplace(levelLow[,i],greyboxForecast$mean,scaleValues));
                greyboxForecast$upper[,i] <- exp(qlaplace(levelUp[,i],greyboxForecast$mean,scaleValues));
            }
        }
        else if(interval=="confidence"){
            greyboxForecast$lower[] <- exp(greyboxForecast$lower);
            greyboxForecast$upper[] <- exp(greyboxForecast$upper);
        }
        greyboxForecast$mean[] <- exp(greyboxForecast$mean);
        greyboxForecast$scale <- scaleValues;
    }
    else if(object$distribution=="dalaplace"){
        # Use the connection between the variance and scale in ALaplace distribution
        alpha <- object$other$alpha;
        scaleValues <- sqrt(greyboxForecast$variances * alpha^2 * (1-alpha)^2 / (alpha^2 + (1-alpha)^2));
        if(interval=="prediction"){
            for(i in 1:nLevels){
                greyboxForecast$lower[,i] <- qalaplace(levelLow[,i],greyboxForecast$mean,scaleValues,alpha);
                greyboxForecast$upper[,i] <- qalaplace(levelUp[,i],greyboxForecast$mean,scaleValues,alpha);
            }
        }
        greyboxForecast$scale <- scaleValues;
    }
    else if(object$distribution=="ds"){
        # Use the connection between the variance and scale in S distribution
        scaleValues <- (greyboxForecast$variances/120)^0.25;
        if(interval=="prediction"){
            for(i in 1:nLevels){
                greyboxForecast$lower[,i] <- qs(levelLow[,i],greyboxForecast$mean,scaleValues);
                greyboxForecast$upper[,i] <- qs(levelUp[,i],greyboxForecast$mean,scaleValues);
            }
        }
        greyboxForecast$scale <- scaleValues;
    }
    else if(object$distribution=="dls"){
        # Use the connection between the variance and scale in S distribution
        scaleValues <- (greyboxForecast$variances/120)^0.25;
        if(interval=="prediction"){
            for(i in 1:nLevels){
                greyboxForecast$lower[,i] <- exp(qs(levelLow[,i],greyboxForecast$mean,scaleValues));
                greyboxForecast$upper[,i] <- exp(qs(levelUp[,i],greyboxForecast$mean,scaleValues));
            }
        }
        else if(interval=="confidence"){
            greyboxForecast$lower[] <- exp(greyboxForecast$lower);
            greyboxForecast$upper[] <- exp(greyboxForecast$upper);
        }
        greyboxForecast$mean <- exp(greyboxForecast$mean);
        greyboxForecast$scale <- scaleValues;
    }
    else if(object$distribution=="dgnorm"){
        # Use the connection between the variance and scale in Generalised Normal distribution
        scaleValues <- sqrt(greyboxForecast$variances*(gamma(1/object$other$beta)/gamma(3/object$other$beta)));
        if(interval=="prediction"){
            greyboxForecast$lower[] <- qgnorm(levelLow,greyboxForecast$mean,scaleValues,object$other$beta);
            greyboxForecast$upper[] <- qgnorm(levelUp,greyboxForecast$mean,scaleValues,object$other$beta);
        }
        greyboxForecast$scale <- scaleValues;
    }
    else if(object$distribution=="dlgnorm"){
        # Use the connection between the variance and scale in Generalised Normal distribution
        scaleValues <- sqrt(greyboxForecast$variances*(gamma(1/object$other$beta)/gamma(3/object$other$beta)));
        if(interval=="prediction"){
            greyboxForecast$lower[] <- exp(qgnorm(levelLow,greyboxForecast$mean,scaleValues,object$other$beta));
            greyboxForecast$upper[] <- exp(qgnorm(levelUp,greyboxForecast$mean,scaleValues,object$other$beta));
        }
        else if(interval=="confidence"){
            greyboxForecast$lower[] <- exp(greyboxForecast$lower);
            greyboxForecast$upper[] <- exp(greyboxForecast$upper);
        }
        greyboxForecast$mean <- exp(greyboxForecast$mean);
        greyboxForecast$scale <- scaleValues;
    }
    else if(object$distribution=="dt"){
        # Use df estimated by the model and then construct conventional intervals. df=2 is the minimum in this model.
        df <- nobs(object) - nparam(object);
        if(interval=="prediction"){
            greyboxForecast$lower[] <- greyboxForecast$mean + sqrt(greyboxForecast$variances) * qt(levelLow,df);
            greyboxForecast$upper[] <- greyboxForecast$mean + sqrt(greyboxForecast$variances) * qt(levelUp,df);
        }
    }
    else if(object$distribution=="dfnorm"){
        if(interval=="prediction"){
            for(i in 1:nLevels){
                greyboxForecast$lower[,i] <- qfnorm(levelLow[,i],greyboxForecast$mean,sqrt(greyboxForecast$variance));
                greyboxForecast$upper[,i] <- qfnorm(levelUp[,i],greyboxForecast$mean,sqrt(greyboxForecast$variance));
            }
        }
        # Correct the mean value
        greyboxForecast$mean <- (sqrt(2/pi)*sqrt(greyboxForecast$variance)*exp(-greyboxForecast$mean^2 /
                                                                                   (2*greyboxForecast$variance)) +
                                     greyboxForecast$mean*(1-2*pnorm(-greyboxForecast$mean/sqrt(greyboxForecast$variance))));
    }
    else if(object$distribution=="dchisq"){
        greyboxForecast$mean <- greyboxForecast$mean^2;
        if(interval=="prediction"){
            greyboxForecast$lower[] <- qchisq(levelLow,df=object$other$nu,ncp=greyboxForecast$mean);
            greyboxForecast$upper[] <- qchisq(levelUp,df=object$other$nu,ncp=greyboxForecast$mean);
        }
        else if(interval=="confidence"){
            greyboxForecast$lower[] <- (greyboxForecast$lower)^2;
            greyboxForecast$upper[] <- (greyboxForecast$upper)^2;
        }
        greyboxForecast$mean <- greyboxForecast$mean + object$scale;
        greyboxForecast$scale <- object$scale;
    }
    else if(object$distribution=="dlnorm"){
        if(interval=="prediction"){
            sdlog <- sqrt(greyboxForecast$variance - sigma(object)^2 + object$scale^2);
        }
        else{
            sdlog <- sqrt(greyboxForecast$variance);
        }
        if(interval!="none"){
            greyboxForecast$lower[] <- qlnorm(levelLow,greyboxForecast$mean,sdlog);
            greyboxForecast$upper[] <- qlnorm(levelUp,greyboxForecast$mean,sdlog);
        }
        greyboxForecast$mean <- exp(greyboxForecast$mean);
        greyboxForecast$scale <- sdlog;
    }
    else if(object$distribution=="dbcnorm"){
        sigma <- sqrt(greyboxForecast$variance);
        # If negative values were produced, zero them out
        if(any(greyboxForecast$mean<0)){
            greyboxForecast$mean[greyboxForecast$mean<0] <- 0;
        }
        if(interval=="prediction"){
            greyboxForecast$lower[] <- qbcnorm(levelLow,greyboxForecast$mean,sigma,object$other$lambdaBC);
            greyboxForecast$upper[] <- qbcnorm(levelUp,greyboxForecast$mean,sigma,object$other$lambdaBC);
        }
        else if(interval=="confidence"){
            greyboxForecast$lower[] <- (greyboxForecast$lower*object$other$lambdaBC+1)^{1/object$other$lambdaBC};
            greyboxForecast$upper[] <- (greyboxForecast$upper*object$other$lambdaBC+1)^{1/object$other$lambdaBC};
        }
        if(object$other$lambdaBC==0){
            greyboxForecast$mean[] <- exp(greyboxForecast$mean)
        }
        else{
            greyboxForecast$mean[] <- (greyboxForecast$mean*object$other$lambdaBC+1)^{1/object$other$lambdaBC};
        }
        greyboxForecast$scale <- sigma;
    }
    else if(object$distribution=="dlogitnorm"){
        if(interval=="prediction"){
            sigma <- sqrt(greyboxForecast$variance - sigma(object)^2 + object$scale^2);
        }
        else{
            sigma <- sqrt(greyboxForecast$variance);
        }
        if(interval=="prediction"){
            greyboxForecast$lower[] <- qlogitnorm(levelLow,greyboxForecast$mean,sigma);
            greyboxForecast$upper[] <- qlogitnorm(levelUp,greyboxForecast$mean,sigma);
        }
        else if(interval=="confidence"){
            greyboxForecast$lower[] <- exp(greyboxForecast$lower)/(1+exp(greyboxForecast$lower));
            greyboxForecast$upper[] <- exp(greyboxForecast$upper)/(1+exp(greyboxForecast$upper));
        }
        greyboxForecast$mean[] <- exp(greyboxForecast$mean)/(1+exp(greyboxForecast$mean));
        greyboxForecast$scale <- sigma;
    }
    else if(object$distribution=="dinvgauss"){
        greyboxForecast$mean <- exp(greyboxForecast$mean);
        if(interval=="prediction"){
            greyboxForecast$scale <- object$scale;
            greyboxForecast$lower[] <- greyboxForecast$mean*qinvgauss(levelLow,mean=1,
                                                                      dispersion=greyboxForecast$scale);
            greyboxForecast$upper[] <- greyboxForecast$mean*qinvgauss(levelUp,mean=1,
                                                                      dispersion=greyboxForecast$scale);
        }
        else if(interval=="confidence"){
            greyboxForecast$lower[] <- exp(greyboxForecast$lower);
            greyboxForecast$upper[] <- exp(greyboxForecast$upper);
        }
    }
    else if(object$distribution=="dlogis"){
        # Use the connection between the variance and scale in logistic distribution
        scale <- sqrt(greyboxForecast$variances * 3 / pi^2);
        if(interval=="prediction"){
            greyboxForecast$lower[] <- qlogis(levelLow,greyboxForecast$mean,scale);
            greyboxForecast$upper[] <- qlogis(levelUp,greyboxForecast$mean,scale);
        }
        greyboxForecast$scale <- scale;
    }
    else if(object$distribution=="dpois"){
        greyboxForecast$mean <- exp(greyboxForecast$mean);
        if(interval=="prediction"){
            greyboxForecast$lower[] <- qpois(levelLow,greyboxForecast$mean);
            greyboxForecast$upper[] <- qpois(levelUp,greyboxForecast$mean);
        }
        else if(interval=="confidence"){
            greyboxForecast$lower[] <- exp(greyboxForecast$lower);
            greyboxForecast$upper[] <- exp(greyboxForecast$upper);
        }
        greyboxForecast$scale <- greyboxForecast$mean;
    }
    else if(object$distribution=="dnbinom"){
        greyboxForecast$mean <- exp(greyboxForecast$mean);
        if(is.null(object$scale)){
            # This is a very approximate thing in order for something to work...
            greyboxForecast$scale <- abs(greyboxForecast$mean^2 / (greyboxForecast$variances - greyboxForecast$mean));
        }
        else{
            greyboxForecast$scale <- object$scale;
        }
        if(interval=="prediction"){
            greyboxForecast$lower[] <- qnbinom(levelLow,mu=greyboxForecast$mean,size=greyboxForecast$scale);
            greyboxForecast$upper[] <- qnbinom(levelUp,mu=greyboxForecast$mean,size=greyboxForecast$scale);
        }
        else if(interval=="confidence"){
            greyboxForecast$lower[] <- exp(greyboxForecast$lower);
            greyboxForecast$upper[] <- exp(greyboxForecast$upper);
        }
    }
    else if(object$distribution=="dbeta"){
        greyboxForecast$shape1 <- greyboxForecast$mean;
        greyboxForecast$shape2 <- greyboxForecast$variances;
        greyboxForecast$mean <- greyboxForecast$shape1 / (greyboxForecast$shape1 + greyboxForecast$shape2);
        greyboxForecast$variances <- (greyboxForecast$shape1 * greyboxForecast$shape2 /
                                          ((greyboxForecast$shape1+greyboxForecast$shape2)^2 *
                                               (greyboxForecast$shape1 + greyboxForecast$shape2 + 1)));
        if(interval=="prediction"){
            greyboxForecast$lower <- qbeta(levelLow,greyboxForecast$shape1,greyboxForecast$shape2);
            greyboxForecast$upper <- qbeta(levelUp,greyboxForecast$shape1,greyboxForecast$shape2);
        }
        else if(interval=="confidence"){
            greyboxForecast$lower <- (greyboxForecast$mean + qt(levelLow,df=object$df.residual)*
                                          sqrt(greyboxForecast$variances/nobs(object)));
            greyboxForecast$upper <- (greyboxForecast$mean + qt(levelUp,df=object$df.residual)*
                                          sqrt(greyboxForecast$variances/nobs(object)));
        }
    }
    else if(object$distribution=="plogis"){
        # The intervals are based on the assumption that a~N(0, sigma^2), and p=exp(a) / (1 + exp(a))
        greyboxForecast$scale <- object$scale;

        greyboxForecast$mean <- plogis(greyboxForecast$location, location=0, scale=1);

        if(interval!="none"){
            greyboxForecast$lower[] <- plogis(qnorm(levelLow, greyboxForecast$location, sqrt(greyboxForecast$variances)),
                                            location=0, scale=1);
            greyboxForecast$upper[] <- plogis(qnorm(levelUp, greyboxForecast$location, sqrt(greyboxForecast$variances)),
                                            location=0, scale=1);
        }
    }
    else if(object$distribution=="pnorm"){
        # The intervals are based on the assumption that a~N(0, sigma^2), and pnorm link
        greyboxForecast$scale <- object$scale;

        greyboxForecast$mean <- pnorm(greyboxForecast$location, mean=0, sd=1);

        if(interval!="none"){
            greyboxForecast$lower[] <- pnorm(qnorm(levelLow, greyboxForecast$location, sqrt(greyboxForecast$variances)),
                                            mean=0, sd=1);
            greyboxForecast$upper[] <- pnorm(qnorm(levelUp, greyboxForecast$location, sqrt(greyboxForecast$variances)),
                                            mean=0, sd=1);
        }
    }

    # If there is an occurrence part of the model, use it
    if(is.occurrence(object$occurrence)){
        greyboxForecast$mean <- greyboxForecast$mean * occurrence$mean;
        #### This is weird and probably wrong. But I don't know yet what the confidence intervals mean in case of occurrence model.
        if(interval=="confidence"){
            greyboxForecast$lower[] <- greyboxForecast$lower * occurrence$mean;
            greyboxForecast$upper[] <- greyboxForecast$upper * occurrence$mean;
        }
    }

    greyboxForecast$level <- cbind(levelLow, levelUp);
    colnames(greyboxForecast$level) <- c(paste0("Lower",c(1:nLevels)),paste0("Upper",c(1:nLevels)));
    greyboxForecast$level <- rbind(switch(side,
                                          "both"=c((1-levelOriginal)/2,(1+levelOriginal)/2),
                                          "lower"=c(levelOriginal,rep(0,nLevels)),
                                          "upper"=c(rep(0,nLevels),levelOriginal)),
                                   greyboxForecast$level);
    rownames(greyboxForecast$level) <- c("Original",paste0("h",c(1:h)));
    greyboxForecast$newdataProvided <- newdataProvided;
    return(structure(greyboxForecast,class="predict.greybox"));
}

#' Forecasting using greybox functions
#'
#' The functions allow producing forecasts based on the provided model and newdata.
#'
#' \code{predict} produces predictions for the provided model and \code{newdata}. If
#' \code{newdata} is not provided, then the data from the model is extracted and the
#' fitted values are reproduced. This might be useful when confidence / prediction
#' intervals are needed for the in-sample values.
#'
#' \code{forecast} function produces forecasts for \code{h} steps ahead. There are four
#' scenarios in this function:
#' \enumerate{
#' \item If the \code{newdata} is  not provided, then it will produce forecasts of the
#' explanatory variables to the horizon \code{h} (using \code{es} from smooth package
#' or using Naive if \code{smooth} is not installed) and use them as \code{newdata}.
#' \item If \code{h} and \code{newdata} are provided, then the number of rows to use
#' will be regulated by \code{h}.
#' \item If \code{h} is \code{NULL}, then it is set equal to the number of rows in
#' \code{newdata}.
#' \item If both \code{h} and \code{newdata} are not provided, then it will use the
#' data from the model itself, reproducing the fitted values.
#' }
#' After forming the \code{newdata} the \code{forecast} function calls for
#' \code{predict}, so you can provide parameters \code{interval}, \code{level} and
#' \code{side} in the call for \code{forecast}.
#'
#' @aliases forecast forecast.greybox
#' @param object Time series model for which forecasts are required.
#' @param newdata The new data needed in order to produce forecasts.
#' @param interval Type of intervals to construct: either "confidence" or
#' "prediction". Can be abbreviated
#' @param level Confidence level. Defines width of prediction interval.
#' @param side What type of interval to produce: \code{"both"} - produces both
#' lower and upper bounds of the interval, \code{"upper"} - upper only, \code{"lower"}
#' - respectively lower only. In the \code{"both"} case the probability is split into
#' two parts: ((1-level)/2, (1+level)/2). When \code{"upper"} is specified, then
#' the intervals for (0, level) are constructed Finally, with \code{"lower"} the interval
#' for (1-level, 1) is returned.
#' @param h The forecast horizon.
#' @param ...  Other arguments passed to \code{vcov} function (see \link[greybox]{coef.alm}
#' for details).
#' @return \code{predict.greybox()} returns object of class "predict.greybox",
#' which contains:
#' \itemize{
#' \item \code{model} - the estimated model.
#' \item \code{mean} - the expected values.
#' \item \code{fitted} - fitted values of the model.
#' \item \code{lower} - lower bound of prediction / confidence intervals.
#' \item \code{upper} - upper bound of prediction / confidence intervals.
#' \item \code{level} - confidence level.
#' \item \code{newdata} - the data provided in the call to the function.
#' \item \code{variances} - conditional variance for the holdout sample.
#' In case of \code{interval="prediction"} includes variance of the error.
#' }
#'
#' \code{predict.alm()} is based on \code{predict.greybox()} and returns
#' object of class "predict.alm", which in addition contains:
#' \itemize{
#' \item \code{location} - the location parameter of the distribution.
#' \item \code{scale} - the scale parameter of the distribution.
#' \item \code{distribution} - name of the fitted distribution.
#' }
#'
#' \code{forecast()} functions return the same "predict.alm" and
#' "predict.greybox" classes, with the same set of output variables.
#'
#' @template author
#' @seealso \link[stats]{predict.lm}, \link[forecast]{forecast}
#' @keywords ts univar
#' @examples
#'
#' xreg <- cbind(rlaplace(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rlaplace(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#' inSample <- xreg[1:80,]
#' outSample <- xreg[-c(1:80),]
#'
#' ourModel <- alm(y~x1+x2, inSample, distribution="dlaplace")
#'
#' predict(ourModel,outSample)
#' predict(ourModel,outSample,interval="c")
#'
#' plot(predict(ourModel,outSample,interval="p"))
#' plot(forecast(ourModel,h=10,interval="p"))
#'
#' @rdname predict.greybox
#' @export
predict.greybox <- function(object, newdata=NULL, interval=c("none", "confidence", "prediction"),
                            level=0.95, side=c("both","upper","lower"), ...){
    interval <- match.arg(interval);
    side <- match.arg(side);

    parameters <- coef.greybox(object);
    parametersNames <- names(parameters);
    ourVcov <- vcov(object, ...);

    nLevels <- length(level);
    levelLow <- levelUp <- vector("numeric",nLevels);
    if(side=="upper"){
        levelLow[] <- 0;
        levelUp[] <- level;
    }
    else if(side=="lower"){
        levelLow[] <- 1-level;
        levelUp[] <- 1;
    }
    else{
        levelLow[] <- (1-level) / 2;
        levelUp[] <- (1+level) / 2;
    }
    paramQuantiles <- qt(c(levelLow, levelUp),df=object$df.residual);

    if(is.null(newdata)){
        matrixOfxreg <- object$data;
        newdataProvided <- FALSE;
        # The first column is the response variable. Either substitute it by ones or remove it.
        if(any(parametersNames=="(Intercept)")){
            matrixOfxreg[,1] <- 1;
        }
        else{
            matrixOfxreg <- matrixOfxreg[,-1,drop=FALSE];
        }
    }
    else{
        newdataProvided <- TRUE;

        if(!is.data.frame(newdata)){
            if(is.vector(newdata)){
                newdataNames <- names(newdata);
                newdata <- matrix(newdata, nrow=1, dimnames=list(NULL, newdataNames));
            }
            newdata <- as.data.frame(newdata);
        }
        else{
            dataOrders <- unlist(lapply(newdata,is.ordered));
            # If there is an ordered factor, remove the bloody ordering!
            if(any(dataOrders)){
                newdata[dataOrders] <- lapply(newdata[dataOrders],function(x) factor(x, levels=levels(x), ordered=FALSE));
            }
        }

        # The gsub is needed in order to remove accidental special characters
        colnames(newdata) <- make.names(colnames(newdata), unique=TRUE);

        # Extract the formula and get rid of the response variable
        testFormula <- formula(object);
        testFormula[[2]] <- NULL;
        # Expand the data frame
        newdataExpanded <- model.frame(testFormula, newdata);
        interceptIsNeeded <- attr(terms(newdataExpanded),"intercept")!=0;
        matrixOfxreg <- model.matrix(newdataExpanded,data=newdataExpanded);
        matrixOfxreg <- matrixOfxreg[,parametersNames,drop=FALSE];
    }

    h <- nrow(matrixOfxreg);

    if(object$distribution=="dbeta"){
        parametersNames <- substr(parametersNames[1:(length(parametersNames)/2)],8,nchar(parametersNames));
    }

    if(any(is.greyboxC(object),is.greyboxD(object))){
        if(ncol(matrixOfxreg)==2){
            colnames(matrixOfxreg) <- parametersNames;
        }
        else{
            colnames(matrixOfxreg)[1] <- parametersNames[1];
        }
        matrixOfxreg <- matrixOfxreg[,parametersNames,drop=FALSE];
    }

    if(!is.matrix(matrixOfxreg)){
        matrixOfxreg <- as.matrix(matrixOfxreg);
        h <- nrow(matrixOfxreg);
    }

    if(h==1){
        matrixOfxreg <- matrix(matrixOfxreg, nrow=1);
    }

    if(object$distribution=="dbeta"){
        # We predict values for shape1 and shape2 and write them down in mean and variance.
        ourForecast <- as.vector(exp(matrixOfxreg %*% parameters[1:(length(parameters)/2)]));
        vectorOfVariances <- as.vector(exp(matrixOfxreg %*% parameters[-c(1:(length(parameters)/2))]));
        # ourForecast <- ourForecast / (ourForecast + as.vector(exp(matrixOfxreg %*% parameters[-c(1:(length(parameters)/2))])));

        yLower <- NULL;
        yUpper <- NULL;
    }
    else{
        ourForecast <- as.vector(matrixOfxreg %*% parameters);
        # abs is needed for some cases, when the likelihood was not fully optimised
        vectorOfVariances <- abs(diag(matrixOfxreg %*% ourVcov %*% t(matrixOfxreg)));

        if(interval!="none"){
            yUpper <- yLower <- matrix(NA, h, nLevels);
            if(interval=="confidence"){
                for(i in 1:nLevels){
                    yLower[,i] <- ourForecast + paramQuantiles[i] * sqrt(vectorOfVariances);
                    yUpper[,i] <- ourForecast + paramQuantiles[i+nLevels] * sqrt(vectorOfVariances);
                }
            }
            else if(interval=="prediction"){
                vectorOfVariances[] <- vectorOfVariances + sigma(object)^2;
                for(i in 1:nLevels){
                    yLower[,i] <- ourForecast + paramQuantiles[i] * sqrt(vectorOfVariances);
                    yUpper[,i] <- ourForecast + paramQuantiles[i+nLevels] * sqrt(vectorOfVariances);
                }
            }

            colnames(yLower) <- switch(side,
                                       "both"=,
                                       "lower"=paste0("Lower bound (",levelLow*100,"%)"),
                                       "upper"=rep("Lower 0%",nLevels));

            colnames(yUpper) <- switch(side,
                                       "both"=,
                                       "upper"=paste0("Upper bound (",levelUp*100,"%)"),
                                       "lower"=rep("Upper 100%",nLevels));
        }
        else{
            yLower <- NULL;
            yUpper <- NULL;
        }
    }

    ourModel <- list(model=object, mean=ourForecast, lower=yLower, upper=yUpper, level=c(levelLow, levelUp), newdata=newdata,
                     variances=vectorOfVariances, newdataProvided=newdataProvided);
    return(structure(ourModel,class="predict.greybox"));
}

# The internal function for the predictions from the model with ARI
predict.almari <- function(object, newdata=NULL, interval=c("none", "confidence", "prediction"),
                            level=0.95, side=c("both","upper","lower"), ...){
    interval <- match.arg(interval);
    side <- match.arg(side);

    y <- actuals(object, all=FALSE);

    # Write down the AR order
    if(is.null(object$call$ar)){
        arOrder <- 0;
    }
    else{
        arOrder <- object$call$ar;
    }

    ariOrder <- length(object$other$polynomial);
    ariParameters <- object$other$polynomial;
    ariNames <- names(ariParameters);

    parameters <- coef.greybox(object);
    # Split the parameters into normal and polynomial (for ARI)
    if(arOrder>0){
        parameters <- parameters[-c(length(parameters)+(1-arOrder):0)];
    }
    nonariParametersNumber <- length(parameters);
    parametersNames <- names(parameters);
    ourVcov <- vcov(object);

    nLevels <- length(level);
    levelLow <- levelUp <- vector("numeric",nLevels);
    if(side=="upper"){
        levelLow[] <- 0;
        levelUp[] <- level;
    }
    else if(side=="lower"){
        levelLow[] <- 1-level;
        levelUp[] <- 1;
    }
    else{
        levelLow[] <- (1-level) / 2;
        levelUp[] <- (1+level) / 2;
    }
    paramQuantiles <- qt(c(levelLow, levelUp),df=object$df.residual);

    if(is.null(newdata)){
        matrixOfxreg <- object$data[,-1,drop=FALSE];
        newdataProvided <- FALSE;
        interceptIsNeeded <- any(names(coef(object))=="(Intercept)");
        if(interceptIsNeeded){
            matrixOfxreg <- cbind(1,matrixOfxreg);
        }
    }
    else{
        newdataProvided <- TRUE;

        if(!is.data.frame(newdata)){
            if(is.vector(newdata)){
                newdataNames <- names(newdata);
                newdata <- matrix(newdata, nrow=1, dimnames=list(NULL, newdataNames));
            }
            newdata <- as.data.frame(newdata);
        }
        else{
            dataOrders <- unlist(lapply(newdata,is.ordered));
            # If there is an ordered factor, remove the bloody ordering!
            if(any(dataOrders)){
                newdata[dataOrders] <- lapply(newdata[dataOrders],function(x) factor(x, levels=levels(x), ordered=FALSE));
            }
        }

        colnames(newdata) <- make.names(colnames(newdata), unique=TRUE);
        # Extract the formula and get rid of the response variable
        testFormula <- formula(object)
        testFormula[[2]] <- NULL;
        # Expand the data frame
        newdataExpanded <- model.frame(testFormula, newdata);
        interceptIsNeeded <- attr(terms(newdataExpanded),"intercept")!=0;
        matrixOfxreg <- model.matrix(newdataExpanded,data=newdataExpanded);

        matrixOfxreg <- matrixOfxreg[,parametersNames,drop=FALSE];
    }

    h <- nrow(matrixOfxreg);

    if(object$distribution=="dbeta"){
        parametersNames <- substr(parametersNames[1:(length(parametersNames)/2)],8,nchar(parametersNames));
    }

    if(any(is.greyboxC(object),is.greyboxD(object))){
        matrixOfxreg <- as.matrix(cbind(rep(1,nrow(newdata)),newdata[,-1]));
        if(ncol(matrixOfxreg)==2){
            colnames(matrixOfxreg) <- parametersNames;
        }
        else{
            colnames(matrixOfxreg)[1] <- parametersNames[1];
        }
        matrixOfxreg <- matrixOfxreg[,parametersNames,drop=FALSE];
    }

    if(!is.matrix(matrixOfxreg)){
        matrixOfxreg <- matrix(matrixOfxreg,ncol=1);
        h <- nrow(matrixOfxreg);
    }

    if(h==1){
        matrixOfxreg <- matrix(matrixOfxreg, nrow=1);
    }

    # Add ARI polynomials to the parameters
    parameters <- c(parameters,ariParameters);

    # If the newdata is provided, do the recursive thingy
    if(newdataProvided){
    # Fill in the tails with the available data
        if(any(object$distribution==c("plogis","pnorm"))){
            matrixOfxregFull <- cbind(matrixOfxreg, matrix(NA,h,ariOrder,dimnames=list(NULL,ariNames)));
            matrixOfxregFull <- rbind(matrix(NA,ariOrder,ncol(matrixOfxregFull)),matrixOfxregFull);
            if(interceptIsNeeded){
                matrixOfxregFull[1:ariOrder,-1] <- tail(object$data[,-1,drop=FALSE],ariOrder);
                matrixOfxregFull[1:ariOrder,1] <- 1;
            }
            else{
                matrixOfxregFull[1:ariOrder,] <- tail(object$data[,-1,drop=FALSE],ariOrder);
            }
            h <- h+ariOrder;
        }
        else{
            matrixOfxregFull <- cbind(matrixOfxreg, matrix(NA,h,ariOrder,dimnames=list(NULL,ariNames)));
            for(i in 1:ariOrder){
                matrixOfxregFull[1:min(h,i),nonariParametersNumber+i] <- tail(y,i)[1:min(h,i)];
            }
        }

        # Transform the lagged response variables
        if(any(object$distribution==c("dlnorm","dllaplace","dls","dpois","dnbinom"))){
            if(any(y==0) & !is.occurrence(object$occurrence)){
                # Use Box-Cox if there are zeroes
                matrixOfxregFull[,nonariParametersNumber+c(1:ariOrder)] <- (matrixOfxregFull[,nonariParametersNumber+c(1:ariOrder)]^0.01-1)/0.01;
                colnames(matrixOfxregFull)[nonariParametersNumber+c(1:ariOrder)] <- paste0(ariNames,"Box-Cox");
            }
            else{
                matrixOfxregFull[,nonariParametersNumber+c(1:ariOrder)] <- log(matrixOfxregFull[,nonariParametersNumber+c(1:ariOrder)]);
                colnames(matrixOfxregFull)[nonariParametersNumber+c(1:ariOrder)] <- paste0(ariNames,"Log");
            }
        }
        else if(object$distribution=="dbcnorm"){
            # Use Box-Cox if there are zeroes
            matrixOfxregFull[,nonariParametersNumber+c(1:ariOrder)] <- (matrixOfxregFull[,nonariParametersNumber+c(1:ariOrder)]^
                                                                            object$other$lambdaBC-1)/object$other$lambdaBC;
            colnames(matrixOfxregFull)[nonariParametersNumber+c(1:ariOrder)] <- paste0(ariNames,"Box-Cox");
        }
        else if(object$distribution=="dlogitnorm"){
            matrixOfxregFull[,nonariParametersNumber+c(1:ariOrder)] <-
                log(matrixOfxregFull[,nonariParametersNumber+c(1:ariOrder)]/
                        (1-matrixOfxregFull[,nonariParametersNumber+c(1:ariOrder)]));
            colnames(matrixOfxregFull)[nonariParametersNumber+c(1:ariOrder)] <- paste0(ariNames,"Logit");
        }
        else if(object$distribution=="dchisq"){
            matrixOfxregFull[,nonariParametersNumber+c(1:ariOrder)] <- sqrt(matrixOfxregFull[,nonariParametersNumber+c(1:ariOrder)]);
            colnames(matrixOfxregFull)[nonariParametersNumber+c(1:ariOrder)] <- paste0(ariNames,"Sqrt");
        }

        # if(object$distribution=="dbeta"){
        #     # We predict values for shape1 and shape2 and write them down in mean and variance.
        #     ourForecast <- as.vector(exp(matrixOfxregFull %*% parameters[1:(length(parameters)/2)]));
        #     vectorOfVariances <- as.vector(exp(matrixOfxregFull %*% parameters[-c(1:(length(parameters)/2))]));
        #     # ourForecast <- ourForecast / (ourForecast + as.vector(exp(matrixOfxregFull %*% parameters[-c(1:(length(parameters)/2))])));
        #
        #     lower <- NULL;
        #     upper <- NULL;
        # }
        # else{

        # Produce forecasts iteratively
        ourForecast <- vector("numeric", h);
        for(i in 1:h){
            ourForecast[i] <- matrixOfxregFull[i,] %*% parameters;
            for(j in 1:ariOrder){
                if(i+j-1==h){
                    break;
                }
                matrixOfxregFull[i+j,nonariParametersNumber+j] <- ourForecast[i];
            }
        }

        if(any(object$distribution==c("plogis","pnorm"))){
            matrixOfxreg <- matrixOfxregFull[-c(1:ariOrder),1:(nonariParametersNumber+arOrder),drop=FALSE];
            ourForecast <- ourForecast[-c(1:ariOrder)];
        }
        else{
            matrixOfxreg <- matrixOfxregFull[,1:(nonariParametersNumber+arOrder),drop=FALSE];
        }
    }
    else{
        ourForecast <- object$mu;
    }

    # abs is needed for some cases, when the likelihoond was not fully optimised
    vectorOfVariances <- abs(diag(matrixOfxreg %*% ourVcov %*% t(matrixOfxreg)));

    if(interval!="none"){
        yUpper <- yLower <- matrix(NA, h, nLevels);
        if(interval=="confidence"){
            for(i in 1:nLevels){
                yLower[,i] <- ourForecast + paramQuantiles[i] * sqrt(vectorOfVariances);
                yUpper[,i] <- ourForecast + paramQuantiles[i+nLevels] * sqrt(vectorOfVariances);
            }
        }
        else if(interval=="prediction"){
            vectorOfVariances[] <- vectorOfVariances + sigma(object)^2;
            for(i in 1:nLevels){
                yLower[,i] <- ourForecast + paramQuantiles[i] * sqrt(vectorOfVariances);
                yUpper[,i] <- ourForecast + paramQuantiles[i+nLevels] * sqrt(vectorOfVariances);
            }
        }

        colnames(yLower) <- switch(side,
                                   "both"=,
                                   "lower"=paste0("Lower bound (",levelLow*100,"%)"),
                                   "upper"=rep("Lower 0%",nLevels));

        colnames(yUpper) <- switch(side,
                                   "both"=,
                                   "upper"=paste0("Upper bound (",levelUp*100,"%)"),
                                   "lower"=rep("Upper 100%",nLevels));
    }
    else{
        yLower <- NULL;
        yUpper <- NULL;
    }
    # }

    ourModel <- list(model=object, mean=ourForecast, lower=yLower, upper=yUpper, level=c(levelLow, levelUp), newdata=newdata,
                     variances=vectorOfVariances, newdataProvided=newdataProvided);
    return(structure(ourModel,class="predict.greybox"));
}

#' @importFrom forecast forecast
#' @export forecast
#' @rdname predict.greybox
#' @export
forecast.greybox <- function(object, newdata=NULL, h=NULL, ...){
    if(!is.null(newdata) & is.null(h)){
        h <- nrow(newdata);
    }

    if(!is.null(newdata) & !is.null(h)){
        if(nrow(newdata)>h){
            newdata <- head(newdata, h);
        }
        # If not enough values in the newdata, use naive
        else if(nrow(newdata)<h){
            warning("Not enough observations in the newdata. Using Naive in order to fill in the values.", call.=FALSE);
            newdata <- rbind(newdata,newdata[rep(nrow(newdata),h-nrow(newdata)),]);
        }
    }
    else if(is.null(newdata) & !is.null(h)){
        warning("No newdata provided, the values will be forecasted", call.=FALSE, immediate.=TRUE);
        if(ncol(object$data)>1){
            # If smooth is not installed, use Naive
            if(!requireNamespace("smooth", quietly = TRUE)){
                newdata <- matrix(object$data[nobs(object),], h, ncol(object$data), byrow=TRUE,
                                  dimnames=list(NULL, colnames(object$data)));
            }
            # Otherwise use es()
            else{
                newdata <- matrix(NA, h, ncol(object$data)-1, dimnames=list(NULL, colnames(object$data)[-1]));

                for(i in 1:ncol(newdata)){
                    newdata[,i] <- smooth::es(object$data[,i+1], occurrence="i", h=h)$forecast;
                }
            }
        }
        else{
            newdata <- matrix(NA, h, 1, dimnames=list(NULL, colnames(object$data)[1]));
        }
    }
    return(predict(object, newdata, ...));
}

#' @rdname predict.greybox
#' @export
forecast.alm <- function(object, newdata=NULL, h=NULL, ...){
    if(!is.null(newdata) & is.null(h)){
        h <- nrow(newdata);
    }

    if(!is.null(newdata) & !is.null(h)){
        if(nrow(newdata)>h){
            newdata <- head(newdata, h);
        }
        # If not enough values in the newdata, use naive
        else if(nrow(newdata)<h){
            warning("Not enough observations in the newdata. Using Naive in order to fill in the values.", call.=FALSE);
            newdata <- rbind(newdata,newdata[rep(nrow(newdata),h-nrow(newdata)),]);
        }
    }
    else if(is.null(newdata) & !is.null(h)){
        warning("No newdata provided, the values will be forecasted", call.=FALSE, immediate.=TRUE);
        if(ncol(object$data)>1){
            # If smooth is not installed, use Naive
            if(!requireNamespace("smooth", quietly = TRUE)){
                newdata <- matrix(object$data[nobs(object),], h, ncol(object$data), byrow=TRUE,
                                  dimnames=list(NULL, colnames(object$data)));
            }
            # Otherwise use es()
            else{
                if(!is.null(object$other$polynomial)){
                    ariLength <- length(object$other$polynomial);
                    newdata <- matrix(NA, h, ncol(object$data)-ariLength-1,
                                      dimnames=list(NULL, colnames(object$data)[-c(1, (ncol(object$data)-ariLength+1):ncol(object$data))]));
                }
                else{
                    newdata <- matrix(NA, h, ncol(object$data)-1, dimnames=list(NULL, colnames(object$data)[-1]));
                }

                for(i in 1:ncol(newdata)){
                    newdata[,i] <- smooth::es(object$data[,i+1], occurrence="i", h=h)$forecast;
                }
            }
        }
        else{
            newdata <- matrix(NA, h, 1, dimnames=list(NULL, colnames(object$data)[1]));
        }
    }
    return(predict(object, newdata, ...));
}
