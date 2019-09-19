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
#' @return     Either \code{"A"} for additive error or \code{"M"} for multiplicative.
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
    if(is.alm(object$occurrence)){
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
    if(is.alm(object$occurrence)){
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
                            "dfnorm" = dfnorm(y, mu=mu, sigma=scale, log=TRUE),
                            "dlnorm" = dlnorm(y, meanlog=mu, sdlog=scale, log=TRUE),
                            "dbcnorm" = dbcnorm(y, mu=mu, sigma=scale, lambda=object$other$lambda, log=TRUE),
                            "dlaplace" = dlaplace(y, mu=mu, scale=scale, log=TRUE),
                            "dalaplace" = dalaplace(y, mu=mu, scale=scale, alpha=object$other$alpha, log=TRUE),
                            "dlogis" = dlogis(y, location=mu, scale=scale, log=TRUE),
                            "dt" = dt(y-mu, df=scale, log=TRUE),
                            "ds" = ds(y, mu=mu, scale=scale, log=TRUE),
                            "dpois" = dpois(y, lambda=mu, log=TRUE),
                            "dnbinom" = dnbinom(y, mu=mu, size=object$other$size, log=TRUE),
                            "dchisq" = dchisq(y, df=object$other$df, ncp=mu, log=TRUE),
                            "dbeta" = dbeta(y, shape1=mu, shape2=scale, log=TRUE),
                            "plogis" = c(plogis(mu[ot], location=0, scale=1, log.p=TRUE),
                                         plogis(mu[!ot], location=0, scale=1, lower.tail=FALSE, log.p=TRUE)),
                            "pnorm" = c(pnorm(mu[ot], mean=0, sd=1, log.p=TRUE),
                                        pnorm(mu[!ot], mean=0, sd=1, lower.tail=FALSE, log.p=TRUE))
    );

    # If this is a mixture model, take the respective probabilities into account (differential entropy)
    if(is.alm(object$occurrence)){
        likValues[!otU] <- -switch(distribution,
                                   "dnorm" =,
                                   "dfnorm" =,
                                   "dbcnorm" =,
                                   "dlnorm" = log(sqrt(2*pi)*scale)+0.5,
                                   "dlaplace" =,
                                   "dalaplace" = (1 + log(2*scale)),
                                   "dlogis" = 2,
                                   "dt" = ((scale+1)/2 *
                                               (digamma((scale+1)/2)-digamma(scale/2)) +
                                               log(sqrt(scale) * beta(scale/2,0.5))),
                                   "ds" = (2 + 2*log(2*scale)),
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
#' @param ... A parameter all can also be provided here. If it is \code{FALSE}, then
#' in the case of occurrence model, only demand sizes will be returned. Works only with
#' 'alm' class.
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
actuals <- function(object, ...) UseMethod("actuals")

#' @rdname actuals
#' @export
actuals.default <- function(object, ...){
    return(object$y);
}

#' @rdname actuals
#' @export
actuals.alm <- function(object, ...){
    ellipsis <- list(...);
    returnValues <- rep(TRUE,nobs(object,all=TRUE));
    if(!is.null(ellipsis$all) && !ellipsis$all){
        returnValues[] <- object$data[,1]!=0;
    }

    return(object$data[returnValues,1]);
}


#' @importFrom stats coef
#' @export
coef.greybox <- function(object, ...){
    return(object$coefficients);
}

#' @export
coef.greyboxD <- function(object, ...){
    coefReturned <- list(coefficients=object$coefficients,se=object$se,
                         dynamic=object$coefficientsDynamic,importance=object$importance);
    return(structure(coefReturned,class="coef.greyboxD"));
}

#' @importFrom stats confint qt
#' @export
confint.alm <- function(object, parm, level=0.95, ...){
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

    # Return S.E. as well, so not to repeat the thing twice...
    confintValues <- cbind(parametersSE, confintValues);
    colnames(confintValues)[1] <- "S.E.";
    return(confintValues);
}

#' @export
confint.greyboxC <- function(object, parm, level=0.95, ...){

    # Extract parameters
    parameters <- coef(object);
    # Extract SE
    parametersSE <- object$se;
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
    parameters <- coef(object)$dynamic;
    # Extract SE
    parametersSE <- object$se;
    # Define quantiles using Student distribution
    paramQuantiles <- qt((1+level)/2,df=object$df.residual);
    # Do the stuff
    confintValues <- array(NA,c(dim(parameters),2),
                           dimnames=list(NULL, dimnames(parameters)[[2]],
                                         c(paste0((1-level)/2*100,"%"),paste0((1+level)/2*100,"%"))));
    confintValues[,,1] <- parameters-paramQuantiles*parametersSE;
    confintValues[,,2] <- parameters+paramQuantiles*parametersSE;

    # If parm was not provided, return everything.
    if(!exists("parm",inherits=FALSE)){
        parm <- colnames(parameters);
    }
    return(confintValues[,parm,]);
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

#' @rdname predict.greybox
#' @importFrom stats predict qchisq qlnorm qlogis qpois qnbinom qbeta
#' @export
predict.alm <- function(object, newdata=NULL, interval=c("none", "confidence", "prediction"),
                            level=0.95, side=c("both","upper","lower"), ...){
    if(is.null(newdata)){
        newdataProvided <- FALSE;
    }
    else{
        newdataProvided <- TRUE;
    }
    interval <- substr(interval[1],1,1);
    side <- substr(side[1],1,1);
    h <- nrow(newdata);
    levelOriginal <- level;

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
    if(is.alm(object$occurrence)){
        occurrence <- predict(object$occurrence, newdata, interval=interval, level=level, side=side, ...);
        # The probability of having zero should be subtracted from that thing...
        if(interval=="p"){
            level <- (level - (1 - occurrence$mean)) / occurrence$mean;
        }
        level[level<0] <- 0;
        greyboxForecast$occurrence <- occurrence;
    }

    if(side=="u"){
        levelLow <- rep(0,length(level));
        levelUp <- level;
    }
    else if(side=="l"){
        levelLow <- 1-level;
        levelUp <- rep(1,length(level));
    }
    else{
        levelLow <- (1 - level) / 2;
        levelUp <- (1 + level) / 2;
    }

    levelLow[levelLow<0] <- 0;
    levelUp[levelUp<0] <- 0;

    if(object$distribution=="dnorm"){
        if(is.alm(object$occurrence) & interval!="n"){
            greyboxForecast$lower[] <- qnorm(levelLow,greyboxForecast$mean,greyboxForecast$scale);
            greyboxForecast$upper[] <- qnorm(levelUp,greyboxForecast$mean,greyboxForecast$scale);
        }
    }
    else if(object$distribution=="dlaplace"){
        # Use the connection between the variance and MAE in Laplace distribution
        scaleValues <- sqrt(greyboxForecast$variances/2);
        if(interval!="n"){
            greyboxForecast$lower[] <- qlaplace(levelLow,greyboxForecast$mean,scaleValues);
            greyboxForecast$upper[] <- qlaplace(levelUp,greyboxForecast$mean,scaleValues);
        }
        greyboxForecast$scale <- scaleValues;
    }
    else if(object$distribution=="dalaplace"){
        # Use the connection between the variance and MAE in Laplace distribution
        alpha <- object$other$alpha;
        scaleValues <- sqrt(greyboxForecast$variances * alpha^2 * (1-alpha)^2 / (alpha^2 + (1-alpha)^2));
        if(interval!="n"){
            # warning("We don't have the proper prediction intervals for ALD yet. The uncertainty is underestimated!", call.=FALSE);
            greyboxForecast$lower[] <- qalaplace(levelLow,greyboxForecast$mean,scaleValues,alpha);
            greyboxForecast$upper[] <- qalaplace(levelUp,greyboxForecast$mean,scaleValues,alpha);
        }
        greyboxForecast$scale <- scaleValues;
    }
    else if(object$distribution=="dt"){
        # Use df estimated by the model and then construct conventional intervals. df=2 is the minimum in this model.
        df <- object$scale^{-2};
        if(interval!="n"){
            greyboxForecast$lower[] <- greyboxForecast$mean + sqrt(greyboxForecast$variances) * qt(levelLow,df);
            greyboxForecast$upper[] <- greyboxForecast$mean + sqrt(greyboxForecast$variances) * qt(levelUp,df);
        }
    }
    else if(object$distribution=="ds"){
        # Use the connection between the variance and scale in S distribution
        scaleValues <- (greyboxForecast$variances/120)^0.25;
        if(interval!="n"){
            greyboxForecast$lower[] <- qs(levelLow,greyboxForecast$mean,scaleValues);
            greyboxForecast$upper[] <- qs(levelUp,greyboxForecast$mean,scaleValues);
        }
        greyboxForecast$scale <- scaleValues;
    }
    else if(object$distribution=="dfnorm"){
        if(interval!="n"){
            greyboxForecast$lower[] <- qfnorm(levelLow,greyboxForecast$mean,sqrt(greyboxForecast$variance));
            greyboxForecast$upper[] <- qfnorm(levelUp,greyboxForecast$mean,sqrt(greyboxForecast$variance));
        }
        # Correct the mean value
        greyboxForecast$mean <- (sqrt(2/pi)*sqrt(greyboxForecast$variance)*exp(-greyboxForecast$mean^2 /
                                                                                   (2*greyboxForecast$variance)) +
                                     greyboxForecast$mean*(1-2*pnorm(-greyboxForecast$mean/sqrt(greyboxForecast$variance))));
    }
    else if(object$distribution=="dchisq"){
        greyboxForecast$mean <- greyboxForecast$mean^2;
        if(interval=="p"){
            greyboxForecast$lower[] <- qchisq(levelLow,df=object$other$df,ncp=greyboxForecast$mean);
            greyboxForecast$upper[] <- qchisq(levelUp,df=object$other$df,ncp=greyboxForecast$mean);
        }
        else if(interval=="c"){
            greyboxForecast$lower[] <- (greyboxForecast$lower)^2;
            greyboxForecast$upper[] <- (greyboxForecast$upper)^2;
        }
        greyboxForecast$mean <- greyboxForecast$mean + object$scale;
        greyboxForecast$scale <- object$scale;
    }
    else if(object$distribution=="dlnorm"){
        if(interval=="p"){
            sdlog <- sqrt(greyboxForecast$variance - sigma(object)^2 + object$scale^2);
        }
        else{
            sdlog <- sqrt(greyboxForecast$variance);
        }
        if(interval!="n"){
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
        if(interval!="n"){
            greyboxForecast$lower[] <- qbcnorm(levelLow,greyboxForecast$mean,sigma,object$other$lambda);
            greyboxForecast$upper[] <- qbcnorm(levelUp,greyboxForecast$mean,sigma,object$other$lambda);
        }
        if(object$other$lambda==0){
            greyboxForecast$mean[] <- exp(greyboxForecast$mean)
        }
        else{
            greyboxForecast$mean[] <- (greyboxForecast$mean*object$other$lambda+1)^{1/object$other$lambda};
        }
        greyboxForecast$scale <- sigma;
    }
    else if(object$distribution=="dlogis"){
        # Use the connection between the variance and scale in logistic distribution
        scale <- sqrt(greyboxForecast$variances * 3 / pi^2);
        if(interval!="n"){
            greyboxForecast$lower[] <- qlogis(levelLow,greyboxForecast$mean,scale);
            greyboxForecast$upper[] <- qlogis(levelUp,greyboxForecast$mean,scale);
        }
        greyboxForecast$scale <- scale;
    }
    else if(object$distribution=="dpois"){
        greyboxForecast$mean <- exp(greyboxForecast$mean);
        if(interval=="p"){
            greyboxForecast$lower[] <- qpois(levelLow,greyboxForecast$mean);
            greyboxForecast$upper[] <- qpois(levelUp,greyboxForecast$mean);
        }
        else if(interval=="c"){
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
        if(interval=="p"){
            greyboxForecast$lower[] <- qnbinom(levelLow,mu=greyboxForecast$mean,size=greyboxForecast$scale);
            greyboxForecast$upper[] <- qnbinom(levelUp,mu=greyboxForecast$mean,size=greyboxForecast$scale);
        }
        else if(interval=="c"){
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
        if(interval=="p"){
            greyboxForecast$lower <- qbeta(levelLow,greyboxForecast$shape1,greyboxForecast$shape2);
            greyboxForecast$upper <- qbeta(levelUp,greyboxForecast$shape1,greyboxForecast$shape2);
        }
        else if(interval=="c"){
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

        if(interval!="n"){
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

        if(interval!="n"){
            greyboxForecast$lower[] <- pnorm(qnorm(levelLow, greyboxForecast$location, sqrt(greyboxForecast$variances)),
                                            mean=0, sd=1);
            greyboxForecast$upper[] <- pnorm(qnorm(levelUp, greyboxForecast$location, sqrt(greyboxForecast$variances)),
                                            mean=0, sd=1);
        }
    }

    # If there is an occurrence part of the model, use it
    if(is.alm(object$occurrence)){
        greyboxForecast$mean <- greyboxForecast$mean * occurrence$mean;
        #### This is weird and probably wrong. But I don't know yet what the confidence intervals mean in case of occurrence model.
        if(interval=="c"){
            greyboxForecast$lower[] <- greyboxForecast$lower * occurrence$mean;
            greyboxForecast$upper[] <- greyboxForecast$upper * occurrence$mean;
        }
    }

    greyboxForecast$level <- cbind(levelOriginal,levelLow, levelUp);
    colnames(greyboxForecast$level) <- c("Original","Lower","Upper");
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
#' @param newdata Forecast horizon
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
#' @param ...  Other arguments.
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
    interval <- substr(interval[1],1,1);

    side <- substr(side[1],1,1);

    parameters <- coef.greybox(object);
    parametersNames <- names(parameters);
    ourVcov <- vcov(object);

    if(side=="u"){
        levelLow <- 0;
        levelUp <- level;
    }
    else if(side=="l"){
        levelLow <- 1-level;
        levelUp <- 1;
    }
    else{
        levelLow <- (1 - level) / 2;
        levelUp <- (1 + level) / 2;
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
            matrixOfxreg <- matrixOfxreg[,-1];
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

        # Extract the formula and get rid of the response variable
        testFormula <- formula(object)
        testFormula[[2]] <- NULL;
        # Expand the data frame
        newdataExpanded <- model.frame(testFormula, newdata);
        interceptIsNeeded <- attr(terms(newdataExpanded),"intercept")!=0;
        matrixOfxreg <- model.matrix(newdataExpanded,data=newdataExpanded);
        matrixOfxreg <- matrixOfxreg[,parametersNames,drop=FALSE];
    }

    nRows <- nrow(matrixOfxreg);

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
        nRows <- nrow(matrixOfxreg);
    }

    if(nRows==1){
        matrixOfxreg <- matrix(matrixOfxreg, nrow=1);
    }

    if(object$distribution=="dbeta"){
        # We predict values for shape1 and shape2 and write them down in mean and variance.
        ourForecast <- as.vector(exp(matrixOfxreg %*% parameters[1:(length(parameters)/2)]));
        vectorOfVariances <- as.vector(exp(matrixOfxreg %*% parameters[-c(1:(length(parameters)/2))]));
        # ourForecast <- ourForecast / (ourForecast + as.vector(exp(matrixOfxreg %*% parameters[-c(1:(length(parameters)/2))])));

        lower <- NULL;
        upper <- NULL;
    }
    else{
        ourForecast <- as.vector(matrixOfxreg %*% parameters);
        # abs is needed for some cases, when the likelihood was not fully optimised
        vectorOfVariances <- abs(diag(matrixOfxreg %*% ourVcov %*% t(matrixOfxreg)));

        if(interval=="c"){
            lower <- ourForecast + paramQuantiles[1] * sqrt(vectorOfVariances);
            upper <- ourForecast + paramQuantiles[2] * sqrt(vectorOfVariances);
        }
        else if(interval=="p"){
            vectorOfVariances <- vectorOfVariances + sigma(object)^2;
            lower <- ourForecast + paramQuantiles[1] * sqrt(vectorOfVariances);
            upper <- ourForecast + paramQuantiles[2] * sqrt(vectorOfVariances);
        }
        else{
            lower <- NULL;
            upper <- NULL;
        }
    }

    ourModel <- list(model=object, mean=ourForecast, lower=lower, upper=upper, level=c(levelLow, levelUp), newdata=newdata,
                     variances=vectorOfVariances, newdataProvided=newdataProvided);
    return(structure(ourModel,class="predict.greybox"));
}

# The internal function for the predictions from the model with ARI
predict.almari <- function(object, newdata=NULL, interval=c("none", "confidence", "prediction"),
                            level=0.95, side=c("both","upper","lower"), ...){
    interval <- substr(interval[1],1,1);

    side <- substr(side[1],1,1);

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

    if(side=="u"){
        levelLow <- 0;
        levelUp <- level;
    }
    else if(side=="l"){
        levelLow <- 1-level;
        levelUp <- 1;
    }
    else{
        levelLow <- (1 - level) / 2;
        levelUp <- (1 + level) / 2;
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

        # Extract the formula and get rid of the response variable
        testFormula <- formula(object)
        testFormula[[2]] <- NULL;
        # Expand the data frame
        newdataExpanded <- model.frame(testFormula, newdata);
        interceptIsNeeded <- attr(terms(newdataExpanded),"intercept")!=0;
        matrixOfxreg <- model.matrix(newdataExpanded,data=newdataExpanded);

        matrixOfxreg <- matrixOfxreg[,parametersNames,drop=FALSE];
    }

    nRows <- nrow(matrixOfxreg);

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
        nRows <- nrow(matrixOfxreg);
    }

    if(nRows==1){
        matrixOfxreg <- matrix(matrixOfxreg, nrow=1);
    }

    # Add ARI polynomials to the parameters
    parameters <- c(parameters,ariParameters);

    # If the newdata is provided, do the recursive thingy
    if(newdataProvided){
    # Fill in the tails with the available data
        if(any(object$distribution==c("plogis","pnorm"))){
            matrixOfxregFull <- cbind(matrixOfxreg, matrix(NA,nRows,ariOrder,dimnames=list(NULL,ariNames)));
            matrixOfxregFull <- rbind(matrix(NA,ariOrder,ncol(matrixOfxregFull)),matrixOfxregFull);
            if(interceptIsNeeded){
                matrixOfxregFull[1:ariOrder,-1] <- tail(object$data[,-1,drop=FALSE],ariOrder);
                matrixOfxregFull[1:ariOrder,1] <- 1;
            }
            else{
                matrixOfxregFull[1:ariOrder,] <- tail(object$data[,-1,drop=FALSE],ariOrder);
            }
            nRows <- nRows+ariOrder;
        }
        else{
            matrixOfxregFull <- cbind(matrixOfxreg, matrix(NA,nRows,ariOrder,dimnames=list(NULL,ariNames)));
            for(i in 1:ariOrder){
                matrixOfxregFull[1:i,nonariParametersNumber+i] <- tail(y,i);
            }
        }

        # Transform the lagged response variables
        if(any(object$distribution==c("dlnorm","dpois","dnbinom"))){
            if(any(y==0) & !is.alm(object$occurrence)){
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
                                                                            object$other$lambda-1)/object$other$lambda;
            colnames(matrixOfxregFull)[nonariParametersNumber+c(1:ariOrder)] <- paste0(ariNames,"Box-Cox");
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
        ourForecast <- vector("numeric", nRows);
        for(i in 1:nRows){
            ourForecast[i] <- matrixOfxregFull[i,] %*% parameters;
            for(j in 1:ariOrder){
                if(i+j-1==nRows){
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

    if(interval=="c"){
        lower <- ourForecast + paramQuantiles[1] * sqrt(vectorOfVariances);
        upper <- ourForecast + paramQuantiles[2] * sqrt(vectorOfVariances);
    }
    else if(interval=="p"){
        vectorOfVariances <- vectorOfVariances + sigma(object)^2;
        lower <- ourForecast + paramQuantiles[1] * sqrt(vectorOfVariances);
        upper <- ourForecast + paramQuantiles[2] * sqrt(vectorOfVariances);
    }
    else{
        lower <- NULL;
        upper <- NULL;
    }
    # }

    ourModel <- list(model=object, mean=ourForecast, lower=lower, upper=upper, level=c(levelLow, levelUp), newdata=newdata,
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

#' @importFrom stats nobs fitted
#' @export
nobs.alm <- function(object, ...){
    ellipsis <- list(...);
    # if all==FALSE is provided, return non-zeroes only
    # otherwise return all
    if(!is.null(ellipsis$all) && !ellipsis$all){
        returnValue <- sum(object$data[,1]!=0);
    }
    else{
        returnValue <- nobs.greybox(object);
    }
    return(returnValue);
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

#### Plot functions ####
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

#' @export
plot.greybox <- function(x, ...){
    ellipsis <- list(...);
    # If type and ylab are not provided, set them...
    if(!any(names(ellipsis)=="type")){
        ellipsis$type <- "l";
    }
    if(!any(names(ellipsis)=="ylab")){
        ellipsis$ylab <- all.vars(x$call$formula)[1];
    }

    ellipsis$x <- actuals(x);
    if(is.alm(x)){
        if(any(x$distribution==c("plogis","pnorm"))){
            ellipsis$x <- (ellipsis$x!=0)*1;
        }
    }
    yFitted <- fitted(x);
    if(is.null(ellipsis$ylim)){
        ellipsis$ylim <- range(c(actuals(x),yFitted));
    }

    do.call(plot,ellipsis);
    lines(yFitted, col="red");
    if(yFitted[length(yFitted)]>mean(yFitted)){
        legelndPosition <- "bottomright";
    }
    else{
        legelndPosition <- "topright";
    }
    legend(legelndPosition,legend=c("Actuals","Fitted"),col=c("black","red"),lwd=rep(1,2));
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

    graphmakerCall <- list(...);
    graphmakerCall$actuals <- yActuals;
    graphmakerCall$forecast <- yForecast;
    graphmakerCall$fitted <- yFitted;
    graphmakerCall$vline <- vline;

    if(!is.null(x$lower)){
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
        graphmakerCall$level <- level;
        graphmakerCall$lower <- yLower;
        graphmakerCall$upper <- yUpper;

        if((any(is.infinite(yLower)) & any(is.infinite(yUpper))) | (any(is.na(yLower)) & any(is.na(yUpper)))){
            graphmakerCall$lower[is.infinite(yLower) | is.na(yLower)] <- 0;
            graphmakerCall$upper[is.infinite(yUpper) | is.na(yUpper)] <- 0;
        }
        else if(any(is.infinite(yLower)) | any(is.na(yLower))){
            graphmakerCall$lower[is.infinite(yLower) | is.na(yLower)] <- 0;
        }
        else if(any(is.infinite(yUpper)) | any(is.na(yUpper))){
            graphmakerCall$upper <- NA;
        }
    }

    do.call(graphmaker,graphmakerCall);
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
                                       max(unlist(lapply(x,max,na.rm=T)),na.rm=T)),
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

#### Print and summary ####
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
                      "dlogis" = "Logistic",
                      "dlaplace" = "Laplace",
                      "dalaplace" = paste0("Asymmetric Laplace with alpha=",round(x$other$alpha,2)),
                      "dt" = "Student t",
                      "ds" = "S",
                      "dfnorm" = "Folded Normal",
                      "dlnorm" = "Log Normal",
                      "dbcnorm" = paste0("Box-Cox Normal with lambda=",round(x$other$lambda,2)),
                      "dchisq" = paste0("Chi-Squared with df=",round(x$other$df,2)),
                      "dpois" = "Poisson",
                      "dnbinom" = paste0("Negative Binomial with size=",round(x$other$size,2)),
                      "dbeta" = "Beta",
                      "plogis" = "Cumulative logistic",
                      "pnorm" = "Cumulative normal"
    );
    if(is.alm(x$occurrence)){
        distribOccurrence <- switch(x$occurrence$distribution,
                                    "plogis" = "Cumulative logistic",
                                    "pnorm" = "Cumulative normal"
        );
        distrib <- paste0("Mixture of ", distrib," and ", distribOccurrence);
    }

    cat(paste0("Response variable: ", paste0(x$responseName,collapse=""),"\n"));
    cat(paste0("Distribution used in the estimation: ", distrib));
    if(!is.null(x$arima)){
        cat(paste0("\n",x$arima," components were included in the model"));
    }
    cat("\nCoefficients:\n");
    print(round(x$coefficients,digits));
    cat("\nError standard deviation: "); cat(round(sqrt(x$s2),digits));
    cat("\nSample size: "); cat(x$dfTable[1]);
    cat("\nNumber of estimated parameters: "); cat(x$dfTable[2]);
    cat("\nNumber of degrees of freedom: "); cat(x$dfTable[3]);
    cat("\nInformation criteria:\n");
    print(round(x$ICs,digits));
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
                      "dlogis" = "Logistic",
                      "dlaplace" = "Laplace",
                      "dalaplace" = "Asymmetric Laplace",
                      "dt" = "Student t",
                      "ds" = "S",
                      "dfnorm" = "Folded Normal",
                      "dlnorm" = "Log Normal",
                      "dbcnorm" = "Box-Cox Normal",
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
                      "dlogis" = "Logistic",
                      "dlaplace" = "Laplace",
                      "dalaplace" = "Asymmetric Laplace",
                      "dt" = "Student t",
                      "ds" = "S",
                      "dfnorm" = "Folded Normal",
                      "dlnorm" = "Log Normal",
                      "dbcnorm" = "Box-Cox Normal",
                      "dchisq" = "Chi-Squared",
                      "dpois" = "Poisson",
                      "dnbinom" = "Negative Binomial",
                      "dbeta" = "Beta",
                      "plogis" = "Cumulative logistic",
                      "pnorm" = "Cumulative normal"
    );

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
        colnames(ourMatrix)[2:3] <- c(paste0("Lower ",round(level[1],3)*100,"%"),paste0("Upper ",round(level[2],3)*100,"%"));
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

#' @importFrom stats sigma
#' @export
sigma.greybox <- function(object, ...){
    return(sqrt(sum(residuals(object)^2)/nobs(object, ...)));
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

#' @export
summary.alm <- function(object, level=0.95, ...){

    # Collect parameters and their standard errors
    parametersConfint <- confint(object, level=level);
    parametersTable <- cbind(coef(object),parametersConfint);
    rownames(parametersTable) <- names(coef(object));
    colnames(parametersTable) <- c("Estimate","Std. Error",
                                   paste0("Lower ",(1-level)/2*100,"%"),
                                   paste0("Upper ",(1+level)/2*100,"%"));
    ourReturn <- list(coefficients=parametersTable);

    ICs <- c(AIC(object),AICc(object),BIC(object),BICc(object));
    names(ICs) <- c("AIC","AICc","BIC","BICc");
    ourReturn$ICs <- ICs;
    ourReturn$distribution <- object$distribution;
    ourReturn$occurrence <- object$occurrence;
    ourReturn$other <- object$other;
    ourReturn$responseName <- formula(object)[[2]];

    # Table with degrees of freedom
    dfTable <- c(nobs(object, all=TRUE),nparam(object),nobs(object, all=TRUE)-nparam(object));
    names(dfTable) <- c("n","k","df");
    ourReturn$dfTable <- dfTable;
    ourReturn$arima <- object$other$arima;
    ourReturn$s2 <- sigma(object)^2;

    ourReturn <- structure(ourReturn,class="summary.alm");
    return(ourReturn);
}

#' @importFrom stats summary.lm
#' @export
summary.greybox <- function(object, level=0.95, ...){
    ourReturn <- summary.lm(object, ...);

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
    dfTable <- c(nobs(object, all=TRUE),nparam(object),nobs(object, all=TRUE)-nparam(object));
    names(dfTable) <- c("n","k","df");
    ourReturn$dfTable <- dfTable;
    ourReturn$arima <- object$other$arima;

    ourReturn <- structure(ourReturn,class="summary.greybox");
    return(ourReturn);
}

#' @export
summary.greyboxC <- function(object, level=0.95, ...){

    # Extract the values from the object
    errors <- residuals(object);
    obs <- nobs(object);
    parametersTable <- cbind(coef(object),object$se,object$importance);

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

    R2 <- 1 - sum(errors^2) / sum((actuals(object)-mean(actuals(object)))^2)
    R2Adj <- 1 - (1 - R2) * (obs - 1) / (obs - df[1]);

    # Table with degrees of freedom
    dfTable <- c(nobs(object), nparam(object), object$df.residual);
    names(dfTable) <- c("n","k","df");

    ourReturn <- structure(list(coefficients=parametersTable, sigma=residSE,
                                ICs=ICs, df=df, r.squared=R2, adj.r.squared=R2Adj,
                                distribution=object$distribution, responseName=formula(object)[[2]],
                                dfTable=dfTable),
                           class="summary.greyboxC");
    return(ourReturn);
}

#' @export
summary.greyboxD <- function(object, level=0.95, ...){

    # Extract the values from the object
    errors <- residuals(object);
    obs <- nobs(object);
    parametersTable <- cbind(coef.greybox(object),apply(object$se,2,mean),apply(object$importance,2,mean));

    parametersConfint <- confint(object, level=level);
    # Calculate the quantiles for parameters and add them to the table
    parametersTable <- cbind(parametersTable,apply(parametersConfint,c(2,3),mean));

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
                                confintDynamic=parametersConfint, dynamic=coef(object)$dynamic,
                                ICs=ICs, df=df, r.squared=R2, adj.r.squared=R2Adj,
                                distribution=object$distribution, responseName=formula(object)[[2]],
                                nobs=nobs(object), nparam=nparam(object), dfTable=dfTable),
                           class="summary.greyboxC");
    return(ourReturn);
}

#' @importFrom stats vcov
#' @export
vcov.alm <- function(object, ...){
    # Are i orders provided? If not, use simpler methods for calculation, when possible
    iOrderNone <- is.null(object$call$i) || (object$call$i==0);

    interceptIsNeeded <- any(names(coef(object))=="(Intercept)");

    if(iOrderNone & any(object$distribution==c("dlnorm","dbcnorm","plogis","pnorm"))){
        # This is based on the underlying normal distribution of logit / probit model
        matrixXreg <- object$data[object$subset,-1,drop=FALSE];
        if(interceptIsNeeded){
            matrixXreg <- cbind(1,matrixXreg);
            colnames(matrixXreg)[1] <- "(Intercept)";
        }
        # colnames(matrixXreg) <- names(coef(object));
        nVariables <- ncol(matrixXreg);
        matrixXreg <- crossprod(matrixXreg);
        vcovMatrixTry <- try(chol2inv(chol(matrixXreg)), silent=TRUE);
        if(class(vcovMatrixTry)=="try-error"){
            warning(paste0("Choleski decomposition of covariance matrix failed, so we had to revert to the simple inversion.\n",
                           "The estimate of the covariance matrix of parameters might be inaccurate."),
                    call.=FALSE);
            vcovMatrix <- try(solve(matrixXreg, diag(nVariables), tol=1e-20), silent=TRUE);
            if(class(vcovMatrix)=="try-error"){
                warning(paste0("Sorry, but the covariance matrix is singular, so we could not invert it.\n",
                               "We failed to produce the covariance matrix of parameters."),
                        call.=FALSE);
                vcovMatrix <- diag(1e+100,nVariables);
            }
        }
        else{
            vcovMatrix <- vcovMatrixTry;
        }
        vcov <- object$scale^2 * vcovMatrix;
        rownames(vcov) <- colnames(vcov) <- names(coef(object));
    }
    else if(iOrderNone & (object$distribution=="dnorm")){
        # matrixXreg <- model.matrix(formula(object),data=object$data);
        # rownames(vcov) <- colnames(vcov) <- names(coef(object));
        matrixXreg <- object$data[object$subset,-1,drop=FALSE];
        if(interceptIsNeeded){
            matrixXreg <- cbind(1,matrixXreg);
            colnames(matrixXreg)[1] <- "(Intercept)";
        }
        # colnames(matrixXreg) <- names(coef(object));
        matrixXreg <- crossprod(matrixXreg);
        vcovMatrixTry <- try(chol2inv(chol(matrixXreg)), silent=TRUE);
        if(class(vcovMatrixTry)=="try-error"){
            warning(paste0("Choleski decomposition of covariance matrix failed, so we had to revert to the simple inversion.\n",
                           "The estimate of the covariance matrix of parameters might be inaccurate."),
                    call.=FALSE);
            vcovMatrix <- try(solve(matrixXreg, diag(nVariables), tol=1e-20), silent=TRUE);
            if(class(vcovMatrix)=="try-error"){
                warning(paste0("Sorry, but the covariance matrix is singular, so we could not invert it.\n",
                               "We failed to produce the covariance matrix of parameters."),
                        call.=FALSE);
                vcovMatrix <- diag(1e+100,nVariables);
            }
        }
        else{
            vcovMatrix <- vcovMatrixTry;
        }
        vcov <- sigma(object)^2 * vcovMatrix;
        rownames(vcov) <- colnames(vcov) <- names(coef(object));
    }
    else{
        # Form the call for alm
        newCall <- object$call;
        if(interceptIsNeeded){
            newCall$formula <- as.formula(paste0(all.vars(newCall$formula)[1],"~."));
        }
        else{
            newCall$formula <- as.formula(paste0(all.vars(newCall$formula)[1],"~.-1"));
        }
        newCall$data <- object$data;
        newCall$subset <- object$subset;
        newCall$distribution <- object$distribution;
        newCall$ar <- object$call$ar;
        newCall$i <- object$call$i;
        newCall$parameters <- coef(object);
        newCall$fast <- TRUE;
        if(object$distribution=="dchisq"){
            newCall$df <- object$other$df;
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
            newCall$lambda <- object$other$lambda;
        }
        newCall$vcovProduce <- TRUE;
        # newCall$occurrence <- NULL;
        newCall$occurrence <- object$occurrence;
        # Recall alm to get hessian
        vcov <- eval(newCall)$vcov;

        if(!is.matrix(vcov)){
            vcov <- as.matrix(vcov);
            colnames(vcov) <- rownames(vcov);
        }
    }
    return(vcov);
}

#' @export
vcov.greyboxC <- function(object, ...){
    s2 <- sigma(object)^2;
    xreg <- as.matrix(object$data[,-1]);
    xreg <- cbind(1,xreg);
    colnames(xreg)[1] <- "(Intercept)";
    importance <- object$importance;

    vcovValue <- s2 * solve(t(xreg) %*% xreg) * importance %*% t(importance);
    warning("The covariance matrix for combined models is approximate. Don't rely too much on that.",call.=FALSE);
    return(vcovValue);
}

#' @export
vcov.greyboxD <- function(object, ...){
    s2 <- sigma(object)^2;
    xreg <- as.matrix(object$data[,-1]);
    xreg <- cbind(1,xreg);
    colnames(xreg)[1] <- "(Intercept)";
    importance <- apply(object$importance,2,mean);

    vcovValue <- s2 * solve(t(xreg) %*% xreg) * importance %*% t(importance);
    warning("The covariance matrix for combined models is approximate. Don't rely too much on that.",call.=FALSE);
    return(vcovValue);
}

# This is needed for lmCombine and other functions, using fast regressions
#' @export
vcov.lmGreybox <- function(object, ...){
    vcov <- sigma(object)^2 * solve(crossprod(object$xreg));
    return(vcov);
}
