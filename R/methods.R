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
#' ourModel <- lm(y~x1+x2,as.data.frame(xreg))
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
    return(structure(object$logLik,nobs=nobs(object),df=nParam(object),class="logLik"));
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
#' ourModel <- lm(y~x1+x2,as.data.frame(xreg))
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
    obs <- nobs(object);
    errors <- residuals(object);
    likValues <- dnorm(errors, 0, sigma(object), TRUE);

    return(likValues);
}

#' @export
pointLik.ets <- function(object, ...){
    likValues <- pointLik.default(object);
    if(errorType(object)=="M"){
        likValues <- likValues - log(abs(fitted(object)));
    }

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
#' ourModel <- lm(y~x1+x2,as.data.frame(xreg))
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
                         dynamic=object$dynamic,importance=object$importance);
    return(structure(coefReturned,class="coef.greyboxD"));
}

#' @importFrom stats confint
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
        confintValues <- matrix(confintValues,1,2,);
        colnames(confintValues) <- confintNames;
        rownames(confintValues) <- names(parameters);
    }
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
        confintValues <- matrix(confintValues,1,2,);
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

#' @importFrom forecast forecast
#' @export forecast
forecast::forecast

#' @importFrom stats predict.lm
#' @export
forecast.greybox <- function(object, newdata, ...){
    if(!is.data.frame(newdata)){
        newdata <- as.data.frame(newdata);
    }
    ellipsis <- list(...);
    if(!any(names(ellipsis)=="interval")){
        ellipsis$interval <- "p";
    }
    ellipsis$object <- object;
    ellipsis$newdata <- newdata;

    parameters <- coef.greybox(object);
    parametersNames <- names(parameters);

    matrixOfxreg <- as.matrix(cbind(rep(1,nrow(newdata)),newdata[,-1]));
    colnames(matrixOfxreg)[1] <- parametersNames[1];
    ourForecast <- as.vector(matrixOfxreg[,parametersNames] %*% coef.greybox(object));

    if(any(names(ellipsis)=="level")){
        level <- ellipsis$level;
    }
    else{
        level <- 0.95;
    }

    if(is.matrix(ourForecast)){
        lower <- ourForecast[,"lwr"];
        upper <- ourForecast[,"upr"];
        ourForecast <- ourForecast[,"fit"];
    }
    else{
        lower <- NULL;
        upper <- NULL;
    }
    ourModel <- list(model=object, mean=ourForecast, lower=lower, upper=upper, level=level, newdata=newdata);
    return(structure(ourModel,class="forecast.greybox"));
}

#' @importFrom forecast getResponse

#' @export
getResponse.greybox <- function(object, ...){
    responseVariable <- object$model[,1];
    names(responseVariable) <- c(1:length(responseVariable));
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
        ellipsis$ylab <- colnames(x$model)[1];
    }

    ellipsis$x <- getResponse(x);
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
plot.forecast.greybox <- function(x, ...){
    yActuals <- getResponse(x$model);
    yStart <- start(yActuals);
    yFrequency <- frequency(yActuals);
    yForecastStart <- time(yActuals)[length(yActuals)]+deltat(yActuals);

    if(!is.null(x$newdata)){
        yName <- colnames(x$model$model)[1];
        if(any(colnames(x$newdata)==yName)){
            yHoldout <- x$newdata[,yName];
            if(!any(is.na(yHoldout))){
                yActuals <- ts(c(yActuals,yHoldout), start=yStart, frequency=yFrequency);
            }
        }
    }

    yFitted <- ts(x$model$fitted.values, start=yStart, frequency=yFrequency);
    yForecast <- ts(x$mean, start=yForecastStart, frequency=yFrequency);
    if(!is.null(x$lower)){
        yLower <- ts(x$lower, start=yForecastStart, frequency=yFrequency);
        yUpper <- ts(x$upper, start=yForecastStart, frequency=yFrequency);
        if(!requireNamespace("smooth", quietly = TRUE)){
            plot(yActuals, type="l", xlim=range(start(yActuals),end(yForecast)), ...);
            lines(yFitted, col="purple", lwd=2);
            lines(yForecast, col="blue", lwd=2);
            lines(yLower, col="gray", lwd=2);
            lines(yUpper, col="gray", lwd=2);
        }
        else{
            smooth::graphmaker(yActuals, yForecast, yFitted, yLower, yUpper, level=x$level);
        }
    }
    else{
        if(!requireNamespace("smooth", quietly = TRUE)){
            plot(yActuals, type="l", xlim=range(start(yActuals),end(yForecast)), ...);
            lines(yForecast, col="blue", lwd=2);
            lines(yFitted, col="purple", lwd=2);
        }
        else{
            smooth::graphmaker(yActuals, yForecast, yFitted);
        }
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

    if(x$distribution=="laplace"){
        distrib <- "Laplace";
    }
    else if(x$distribution=="s"){
        distrib <- "S";
    }
    else if(x$distribution=="fnorm"){
        distrib <- "Folded Normal";
    }
    else if(x$distribution=="chisq"){
        distrib <- "Chi-Squared";
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
    cat("Coefficients:\n");
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
    cat("Coefficients:\n");
    print(round(x$coefficients,digits));
    cat("---\n");
    cat(paste0("Residual standard error: ",round(x$sigma,digits)," on ",
               round(x$df[2],digits)," degrees of freedom:\n"));
    cat("Combined ICs:\n");
    print(round(x$ICs,digits));
}

#' @export
print.forecast.greybox <- function(x, ...){
    ourMatrix <- as.matrix(x$mean);
    colnames(ourMatrix) <- "Mean";
    if(!is.null(x$lower)){
        ourMatrix <- cbind(ourMatrix, x$lower, x$upper);
        level <- x$level;
        colnames(ourMatrix)[2:3] <- c(paste0("Lower ",round((1-level)/2,3)*100,"%"),paste0("Upper ",round((1+level)/2,3)*100,"%"));
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
summary.alm <- function(object, level=0.95, ...){

    # Collect parameters and their standard errors
    parametersTable <- cbind(coef(object),sqrt(diag(vcov(object))));
    parametersTable <- cbind(parametersTable,confint(object, level=level));
    rownames(parametersTable) <- names(coef(object));
    colnames(parametersTable) <- c("Estimate","Std. Error",
                                   paste0("Lower ",(1-level)/2*100,"%"),
                                   paste0("Upper ",(1+level)/2*100,"%"));
    ourReturn <- list(coefficients=parametersTable);

    ICs <- c(AIC(object),AICc(object),BIC(object),BICc(object));
    names(ICs) <- c("AIC","AICc","BIC","BICc");
    ourReturn$ICs <- ICs;
    ourReturn$distribution <- object$distribution;

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

    R2 <- 1 - sum(errors^2) / sum((object$model[,1]-mean(object$model[,1]))^2)
    R2Adj <- 1 - (1 - R2) * (obs - 1) / (obs - df[1]);

    ourReturn <- structure(list(coefficients=parametersTable, sigma=residSE,
                                ICs=ICs, df=df, r.squared=R2, adj.r.squared=R2Adj),
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

    R2 <- 1 - sum(errors^2) / sum((object$model[,1]-mean(object$model[,1]))^2)
    R2Adj <- 1 - (1 - R2) * (obs - 1) / (obs - df[1]);

    ourReturn <- structure(list(coefficients=parametersTable, sigma=residSE,
                                confintDynamic=parametersConfint, dynamic=coef(object)$dynamic,
                                ICs=ICs, df=df, r.squared=R2, adj.r.squared=R2Adj),
                           class="summary.greyboxC");
    return(ourReturn);
}

#' @importFrom stats vcov
#' @export
vcov.alm <- function(object, ...){
    vcov <- object$vcov;
    if(!is.matrix(vcov)){
        vcov <- as.matrix(object$vcov);
        colnames(vcov) <- rownames(vcov);
    }
    return(vcov);
}

#' @export
vcov.greyboxC <- function(object, ...){
    s2 <- sigma(object)^2;
    xreg <- as.matrix(object$model[,-1]);
    xreg <- cbind(1,xreg);
    colnames(xreg)[1] <- "Intercept";
    importance <- object$importance;

    vcovValue <- s2 * solve(t(xreg) %*% xreg) * importance %*% t(importance);
    warning("The covariance matrix for combined models is approximate. Don't rely too much on that.",call.=FALSE);
    return(vcovValue);
}

#' @export
vcov.greyboxD <- function(object, ...){
    s2 <- sigma(object)^2;
    xreg <- as.matrix(object$model[,-1]);
    xreg <- cbind(1,xreg);
    colnames(xreg)[1] <- "Intercept";
    importance <- apply(object$importance,2,mean);

    vcovValue <- s2 * solve(t(xreg) %*% xreg) * importance %*% t(importance);
    warning("The covariance matrix for combined models is approximate. Don't rely too much on that.",call.=FALSE);
    return(vcovValue);
}
