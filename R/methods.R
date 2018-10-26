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
#' AICc(ourModel,h=10)
#' BICc(ourModel,h=10)
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
    nParamAll <- nParam(object);
    llikelihood <- llikelihood[1:length(llikelihood)];
    obs <- nobs(object);

    IC <- 2*nParamAll - 2*llikelihood + 2 * nParamAll * (nParamAll + 1) / (obs - nParamAll - 1);

    return(IC);
}

#' @export
BICc.default <- function(object, ...){
    llikelihood <- logLik(object);
    nParamAll <- nParam(object);
    llikelihood <- llikelihood[1:length(llikelihood)];
    obs <- nobs(object);

    IC <- - 2*llikelihood + (nParamAll * log(obs) * obs) / (obs - nParamAll - 1);

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
        return(structure(object$logLik,nobs=nobs(object),df=nParam(object)+nParam(object$occurrence),class="logLik"));
    }
    else{
        return(structure(object$logLik,nobs=nobs(object),df=nParam(object),class="logLik"));
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
#' pointLik(ourModel) - nParam(ourModel)
#'
#' # Bias correction in AIC style
#' 2*(nParam(ourModel)/nobs(ourModel) - pointLik(ourModel))
#'
#' # BIC calculation based on pointLik
#' log(nobs(ourModel))*nParam(ourModel) - 2*sum(pointLik(ourModel))
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
    y <- getResponse(object);
    ot <- y!=0;
    if(is.alm(object$occurrence)){
        y <- y[ot];
    }
    mu <- object$mu;
    scale <- object$scale;

    likValues <- switch(distribution,
                        "dnorm" = dnorm(y, mean=mu, sd=scale, log=TRUE),
                        "dfnorm" = dfnorm(y, mu=mu, sigma=scale, log=TRUE),
                        "dlnorm" = dlnorm(y, meanlog=mu, sdlog=scale, log=TRUE),
                        "dlaplace" = dlaplace(y, mu=mu, b=scale, log=TRUE),
                        "dlogis" = dlogis(y, location=mu, scale=scale, log=TRUE),
                        "ds" = ds(y, mu=mu, b=scale, log=TRUE),
                        "dpois" = dpois(y, lambda=mu, log=TRUE),
                        "dnbinom" = dnbinom(y, mu=mu, size=scale, log=TRUE),
                        "dchisq" = dchisq(y, df=scale, ncp=mu, log=TRUE),
                        "plogis" = c(plogis(mu[ot], location=0, scale=1, log.p=TRUE),
                                     plogis(mu[!ot], location=0, scale=1, lower.tail=FALSE, log.p=TRUE)),
                        "pnorm" = c(pnorm(mu[ot], mean=0, sd=1, log.p=TRUE),
                                    pnorm(mu[!ot], mean=0, sd=1, lower.tail=FALSE, log.p=TRUE))
    );

    # Sort values if plogis or pnorm was used
    if(any(distribution==c("plogis","pnorm"))){
        likValuesNew <- likValues;
        likValues[ot] <- likValuesNew[1:sum(ot)];
        likValues[!ot] <- likValuesNew[-c(1:sum(ot))];
    }

    # If this is a mixture model, take the respective probabilities into account
    if(is.alm(object$occurrence)){
        likValuesNew <- pointLik(object$occurrence);
        likValuesNew[ot] <- likValuesNew[ot] + likValues;
        likValues <- likValuesNew;
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
    k <- nParam(object);
    return(2 * k - 2 * obs * pointLik(object));
}

#' @rdname pointIC
#' @export pAICc
pAICc <- function(object, ...) UseMethod("pAICc")

#' @export
pAICc.default <- function(object, ...){
    obs <- nobs(object);
    k <- nParam(object);
    return(2 * k - 2 * obs * pointLik(object) + 2 * k * (k + 1) / (obs - k - 1));
}

#' @rdname pointIC
#' @export pBIC
pBIC <- function(object, ...) UseMethod("pBIC")

#' @export
pBIC.default <- function(object, ...){
    obs <- nobs(object);
    k <- nParam(object);
    return(log(obs) * k - 2 * obs * pointLik(object));
}

#' @rdname pointIC
#' @export pBIC
pBICc <- function(object, ...) UseMethod("pBICc")

#' @export
pBICc.default <- function(object, ...){
    obs <- nobs(object);
    k <- nParam(object);
    return((k * log(obs) * obs) / (obs - k - 1)  - 2 * obs * pointLik(object));
}


#### Coefficients and extraction functions ####
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

#' @rdname predict.greybox
#' @importFrom stats predict qchisq qlnorm qlogis qpois qnbinom
#' @export
predict.alm <- function(object, newdata=NULL, interval=c("none", "confidence", "prediction"),
                            level=0.95, side=c("both","upper","lower"), ...){
    if(is.null(newdata)){
        newdata <- object$data;
        newdataProvided <- FALSE;
    }
    else{
        newdataProvided <- TRUE;
    }
    interval <- substr(interval[1],1,1);
    side <- substr(side[1],1,1);
    h <- nrow(newdata);
    levelOriginal <- level;
    greyboxForecast <- predict.greybox(object, newdata, interval, level, side=side, ...);
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
        bValues <- sqrt(greyboxForecast$variances/2);
        if(interval!="n"){
            greyboxForecast$lower[] <- qlaplace(levelLow,greyboxForecast$mean,bValues);
            greyboxForecast$upper[] <- qlaplace(levelUp,greyboxForecast$mean,bValues);
        }
        greyboxForecast$scale <- bValues;
    }
    else if(object$distribution=="dalaplace"){
        # Use the connection between the variance and MAE in Laplace distribution
        alpha <- object$other$alpha;
        bValues <- sqrt(greyboxForecast$variances * alpha^2 * (1-alpha)^2 / (alpha^2 + (1-alpha)^2));
        if(interval!="n"){
            # warning("We don't have the proper prediction intervals for ALD yet. The uncertainty is underestimated!", call.=FALSE);
            greyboxForecast$lower[] <- qalaplace(levelLow,greyboxForecast$mean,bValues,alpha);
            greyboxForecast$upper[] <- qalaplace(levelUp,greyboxForecast$mean,bValues,alpha);
        }
        greyboxForecast$scale <- bValues;
    }
    else if(object$distribution=="ds"){
        # Use the connection between the variance and b in S distribution
        bValues <- (greyboxForecast$variances/120)^0.25;
        if(interval!="n"){
            greyboxForecast$lower[] <- qs(levelLow,greyboxForecast$mean,bValues);
            greyboxForecast$upper[] <- qs(levelUp,greyboxForecast$mean,bValues);
        }
        greyboxForecast$scale <- bValues;
    }
    else if(object$distribution=="dfnorm"){
        if(interval!="n"){
            greyboxForecast$lower[] <- qfnorm(levelLow,greyboxForecast$mean,sqrt(greyboxForecast$variance));
            greyboxForecast$upper[] <- qfnorm(levelUp,greyboxForecast$mean,sqrt(greyboxForecast$variance));
        }
        # Correct the mean value
        greyboxForecast$mean <- (sqrt(2/pi)*sqrt(greyboxForecast$variance)*exp(-greyboxForecast$mean^2 /
                                                                                   (2*sqrt(greyboxForecast$variance)^2)) +
                                     greyboxForecast$mean*(1-2*pnorm(-greyboxForecast$mean/sqrt(greyboxForecast$variance))));
    }
    else if(object$distribution=="dchisq"){
        greyboxForecast$mean <- greyboxForecast$mean^2;
        if(interval=="p"){
            greyboxForecast$lower[] <- qchisq(levelLow,df=object$scale,ncp=greyboxForecast$mean);
            greyboxForecast$upper[] <- qchisq(levelUp,df=object$scale,ncp=greyboxForecast$mean);
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
            greyboxForecast$lower[] <- greyboxForecast$lower * occurrence$lower;
            greyboxForecast$upper[] <- greyboxForecast$upper * occurrence$upper;
        }
    }

    greyboxForecast$level <- cbind(levelOriginal,levelLow, levelUp);
    colnames(greyboxForecast$level) <- c("Original","Lower","Upper");
    greyboxForecast$newdataProvided <- newdataProvided;
    return(structure(greyboxForecast,class="predict.greybox"));
}

#' Forecasting using greybox functions
#'
#' \code{predict} is a function for predictions from various model fitting
#' functions. The function invokes particular method, corresponding to the
#' class of the first argument.
#'
#' Although this function is called "forecast", it has functionality similar to
#' "predict" function.
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
#' @template author
#' @seealso \code{\link[stats]{predict.lm}}
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
#' plot(predict(ourModel,outSample,interval="p"))
#'
#' @rdname predict.greybox
#' @export
predict.greybox <- function(object, newdata=NULL, interval=c("none", "confidence", "prediction"),
                            level=0.95, side=c("both","upper","lower"), ...){
    interval <- substr(interval[1],1,1);

    side <- substr(side[1],1,1);

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

    if(is.null(newdata)){
        newdata <- object$data;
        newdataProvided <- FALSE;
    }
    else{
        newdataProvided <- TRUE;
    }
    if(!is.data.frame(newdata)){
        if(is.vector(newdata)){
            newdataNames <- names(newdata);
            newdata <- matrix(newdata, nrow=1, dimnames=list(NULL, newdataNames));
        }
        newdata <- as.data.frame(newdata);
    }
    nRows <- nrow(newdata);

    parameters <- coef.greybox(object);
    parametersNames <- names(parameters);
    ourVcov <- vcov(object);

    if(object$distribution=="dalaplace"){
        if(parametersNames[1]=="alpha"){
            parametersNames <- parametersNames[-1];
            parameters <- parameters[-1];
            ourVcov <- ourVcov[-1,-1];
        }
    }

    if(any(parametersNames=="(Intercept)")){
        matrixOfxreg <- as.matrix(cbind(rep(1,nrow(newdata)),newdata[,-1]));
        if(ncol(matrixOfxreg)==2){
            colnames(matrixOfxreg) <- parametersNames;
        }
        else{
            colnames(matrixOfxreg)[1] <- parametersNames[1];
        }
    }

    matrixOfxreg <- matrixOfxreg[,parametersNames];

    if(nRows==1){
        matrixOfxreg <- matrix(matrixOfxreg, nrow=1);
    }

    ourForecast <- as.vector(matrixOfxreg %*% parameters);

    paramQuantiles <- qt(c(levelLow, levelUp),df=object$df.residual);

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

    ourModel <- list(model=object, mean=ourForecast, lower=lower, upper=upper, level=c(levelLow, levelUp), newdata=newdata,
                     variances=vectorOfVariances, newdataProvided=newdataProvided);
    return(structure(ourModel,class="predict.greybox"));
}

#' @importFrom forecast forecast
#' @export forecast
#' @export
forecast.greybox <- function(object, newdata, ...){
    return(predict(object, newdata, ...));
}

#' @export
forecast.alm <- function(object, newdata, ...){
    return(predict(object, newdata, ...));
}

#' @importFrom forecast getResponse
#' @export
getResponse.greybox <- function(object, ...){
    responseVariable <- object$data[,1];
    return(responseVariable);
}

#' @importFrom stats nobs fitted
#' @export
nobs.greybox <- function(object, ...){
    return(length(fitted(object)));
}


#' Number of parameters in the model
#'
#' This function returns the number of estimated parameters in the model
#'
#' This is a very basic and a simple function which does what it says:
#' extracts number of parameters in the estimated model.
#'
#' @aliases nParam
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
#' nParam(ourModel)
#'
#' @importFrom stats coef
#' @export nParam
nParam <- function(object, ...) UseMethod("nParam")

#' @export
nParam.default <- function(object, ...){
    # The length of the vector of parameters + variance
    return(length(coef(object))+1);
}

#' @export
nParam.alm <- function(object, ...){
    # The length of the vector of parameters + variance
    return(object$df);
}

#' @export
nParam.logLik <- function(object, ...){
    # The length of the vector of parameters + variance
    return(attributes(object)$df);
}

#' @export
nParam.greyboxC <- function(object, ...){
    # The length of the vector of parameters + variance
    return(sum(object$importance)+1);
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

    ellipsis$x <- getResponse(x);
    if(is.alm(x)){
        if(any(x$distribution==c("plogis","pnorm"))){
            ellipsis$x <- (ellipsis$x!=0)*1;
        }
    }
    yFitted <- fitted(x);

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
    yActuals <- getResponse(x$model);
    yStart <- start(yActuals);
    yFrequency <- frequency(yActuals);
    yForecastStart <- time(yActuals)[length(yActuals)]+deltat(yActuals);

    if(!is.null(x$newdata)){
        yName <- all.vars(x$model$call$formula)[1];
        if(any(colnames(x$newdata)==yName)){
            yHoldout <- x$newdata[,yName];
            if(!any(is.na(yHoldout))){
                if(x$newdataProvided){
                    yActuals <- ts(c(yActuals,yHoldout), start=yStart, frequency=yFrequency);
                }
                else{
                    yActuals <- ts(yHoldout, start=yForecastStart, frequency=yFrequency);
                }
                # If this is occurrence model, then transform actual to the occurrence
                if(any(x$distribution==c("pnorm","plogis"))){
                    yActuals <- (yActuals!=0)*1;
                }
            }
        }
    }

    if(x$newdataProvided){
        yFitted <- ts(x$model$fitted.values, start=yStart, frequency=yFrequency);
    }
    else{
        yFitted <- NA;
    }
    yForecast <- ts(x$mean, start=yForecastStart, frequency=yFrequency);
    if(!is.null(x$lower)){
        yLower <- ts(x$lower, start=yForecastStart, frequency=yFrequency);
        yUpper <- ts(x$upper, start=yForecastStart, frequency=yFrequency);

        if(is.matrix(x$level)){
            level <- x$level[1];
        }
        else{
            level <- x$level;
        }

        if((any(is.infinite(yLower)) & any(is.infinite(yUpper))) | (any(is.na(yLower)) & any(is.na(yUpper)))){
            yLower[is.infinite(yLower) | is.na(yLower)] <- 0;
            yUpper[is.infinite(yUpper) | is.na(yUpper)] <- 0;
            graphmaker(yActuals, yForecast, yFitted, lower=yLower, upper=yUpper, level=level, ...);
        }
        else if(any(is.infinite(yLower)) | any(is.na(yLower))){
            yLower[is.infinite(yLower) | is.na(yLower)] <- 0;
            graphmaker(yActuals, yForecast, yFitted, lower=yLower, upper=yUpper, level=level, ...);
        }
        else if(any(is.infinite(yUpper)) | any(is.na(yUpper))){
            graphmaker(yActuals, yForecast, yFitted, lower=yLower, upper=NA, level=level, ...);
        }
        else{
            graphmaker(yActuals, yForecast, yFitted, yLower, yUpper, level=level, ...);
        }
    }
    else{
        graphmaker(yActuals, yForecast, yFitted, ...);
    }
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
print.summary.alm <- function(x, ...){
    ellipsis <- list(...);
    if(!any(names(ellipsis)=="digits")){
        digits <- 5;
    }
    else{
        digits <- ellipsis$digits;
    }

    distrib <- switch(x$distribution,
                      "dnorm" = "Normal",
                      "dlogis" = "Logistic",
                      "dlaplace" = "Laplace",
                      "dalaplace" = paste0("Asymmetric Laplace with alpha=",round(x$other$alpha,2)),
                      "ds" = "S",
                      "dfnorm" = "Folded Normal",
                      "dlnorm" = "Log Normal",
                      "dchisq" = "Chi-Squared",
                      "dpois" = "Poisson",
                      "dnbinom" = "Negative Binomial",
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

    cat(paste0("Distribution used in the estimation: ", distrib));
    cat("\nCoefficients:\n");
    print(round(x$coefficients,digits));
    cat("ICs:\n");
    print(round(x$ICs,digits));
}

#' @export
print.summary.greybox <- function(x, ...){
    ellipsis <- list(...);
    if(!any(names(ellipsis)=="digits")){
        digits <- 5;
    }
    else{
        digits <- ellipsis$digits;
    }

    distrib <- switch(x$distribution,
                      "dnorm" = "Normal",
                      "dlogis" = "Logistic",
                      "dlaplace" = "Laplace",
                      "dalaplace" = "Asymmetric Laplace",
                      "ds" = "S",
                      "dfnorm" = "Folded Normal",
                      "dlnorm" = "Log Normal",
                      "dchisq" = "Chi-Squared",
                      "dpois" = "Poisson",
                      "dnbinom" = "Negative Binomial",
                      "plogis" = "Cumulative logistic",
                      "pnorm" = "Cumulative normal"
    );

    cat(paste0("Distribution used in the estimation: ", distrib));
    cat("\nCoefficients:\n");
    print(round(x$coefficients,digits));
    cat("---\n");
    cat(paste0("Residual standard error: ",round(x$sigma,digits)," on ",
               round(x$df[2],digits)," degrees of freedom:\n"));
    cat("ICs:\n");
    print(round(x$ICs,digits));
}

#' @export
print.summary.greyboxC <- function(x, ...){
    ellipsis <- list(...);
    if(!any(names(ellipsis)=="digits")){
        digits <- 5;
    }
    else{
        digits <- ellipsis$digits;
    }

    distrib <- switch(x$distribution,
                      "dnorm" = "Normal",
                      "dlogis" = "Logistic",
                      "dlaplace" = "Laplace",
                      "dalaplace" = "Asymmetric Laplace",
                      "ds" = "S",
                      "dfnorm" = "Folded Normal",
                      "dlnorm" = "Log Normal",
                      "dchisq" = "Chi-Squared",
                      "dpois" = "Poisson",
                      "dnbinom" = "Negative Binomial",
                      "plogis" = "Cumulative logistic",
                      "pnorm" = "Cumulative normal"
    );

    cat(paste0("Distribution used in the estimation: ", distrib));
    cat("\nCoefficients:\n");
    print(round(x$coefficients,digits));
    cat("---\n");
    cat(paste0("Residual standard error: ",round(x$sigma,digits)," on ",
               round(x$df[2],digits)," degrees of freedom:\n"));
    cat("Combined ICs:\n");
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
    return(sqrt(sum(residuals(object)^2)/(nobs(object)-nParam(object))));
}

#' @export
sigma.alm <- function(object, ...){
    if(any(object$distribution==c("plogis","pnorm"))){
        return(object$scale);
    }
    else{
        return(sigma.greybox(object));
    }
}

#' @export
sigma.ets <- function(object, ...){
    return(sqrt(object$sigma2));
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

    R2 <- 1 - sum(errors^2) / sum((getResponse(object)-mean(getResponse(object)))^2)
    R2Adj <- 1 - (1 - R2) * (obs - 1) / (obs - df[1]);

    ourReturn <- structure(list(coefficients=parametersTable, sigma=residSE,
                                ICs=ICs, df=df, r.squared=R2, adj.r.squared=R2Adj,
                                distribution=object$distribution),
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

    R2 <- 1 - sum(errors^2) / sum((getResponse(object)-mean(getResponse(object)))^2)
    R2Adj <- 1 - (1 - R2) * (obs - 1) / (obs - df[1]);

    ourReturn <- structure(list(coefficients=parametersTable, sigma=residSE,
                                confintDynamic=parametersConfint, dynamic=coef(object)$dynamic,
                                ICs=ICs, df=df, r.squared=R2, adj.r.squared=R2Adj,
                                distribution=object$distribution),
                           class="summary.greyboxC");
    return(ourReturn);
}

#' @importFrom stats vcov
#' @export
vcov.alm <- function(object, ...){
    if(any(object$distribution==c("dlnorm","plogis","pnorm"))){
        # This is based on the underlying normal distribution of logit / probit model
        matrixXreg <- as.matrix(object$data[object$subset,-1]);
        if(any(names(coef(object))=="(Intercept)")){
            matrixXreg <- cbind(1,matrixXreg);
        }
        colnames(matrixXreg) <- names(coef(object));
        vcov <- object$scale^2 * solve(crossprod(matrixXreg));
    }
    else if(object$distribution=="dnorm"){
        matrixXreg <- as.matrix(object$data[object$subset,-1]);
        if(any(names(coef(object))=="(Intercept)")){
            matrixXreg <- cbind(1,matrixXreg);
        }
        colnames(matrixXreg) <- names(coef(object));
        vcov <- sigma(object)^2 * solve(crossprod(matrixXreg));
    }
    else{
        # Form the call for alm
        newCall <- object$call;
        newCall$data <- object$data;
        newCall$subset <- object$subset;
        newCall$distribution <- object$distribution;
        newCall$B <- coef(object);
        if(any(object$distribution==c("dchisq","dnbinom"))){
            newCall$B <- c(object$scale, newCall$B);
        }
        newCall$vcovProduce <- TRUE;
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

#' @export
vcov.lmGreybox <- function(object, ...){
    vcov <- sigma(object)^2 * solve(crossprod(object$xreg));
    return(vcov);
}
