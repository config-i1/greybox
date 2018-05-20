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

#' @importFrom stats confint
#' @export
confint.greyboxC <- function(object, parm, level=0.95, ...){

    # Extract parameters
    parameters <- coef(object);
    # Extract SE
    parametersSE <- object$coefficientsSE;
    # Define quantiles using Student distribution
    paramQuantiles <- qt((1+level)/2,df=object$df.residual);
    # Do the stuff
    confintValues <- cbind(parameters-paramQuantiles*parametersSE,
                           parameters+paramQuantiles*parametersSE);
    colnames(confintValues) <- c(paste0((1-level)/2*100,"%"),
                                 paste0((1+level)/2*100,"%"));
    # If parm was not provided, return everything.
    if(!exists("parm",inherits=FALSE)){
        parm <- names(parameters);
    }
    return(confintValues[parm,]);
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

    if(nobs(object) <= nParam(object)){
        matrixOfxreg <- as.matrix(cbind(rep(1,nrow(newdata)),newdata[,-1]));
        ourForecast <- as.vector(matrixOfxreg %*% coef(object));
    }
    else{
        ourForecast <- do.call(predict.lm, ellipsis);
    }

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
forecast::getResponse

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
    roStart <- start(y)[1]+yDeltat*(roStart-roh*co);

    # Start plotting
    plot(y, ylab="Actuals", ylim=range(min(unlist(lapply(x,min,na.rm=T)),na.rm=T),
                                       max(unlist(lapply(x,max,na.rm=T)),na.rm=T)),
         ...);
    abline(v=roStart, col="red", lwd=2);
    for(j in 1:thingsToPlot){
        colCurrent <- rgb((j-1)/thingsToPlot,0,(thingsToPlot-j+1)/thingsToPlot,1);
        for(i in 1:roh){
            points(roStart+i*yDeltat,x[[2+j]][1,i],col=colCurrent,pch=16);
            lines(c(roStart + (0:(h-1)+i)*yDeltat),c(x[[2+j]][,i]),col=colCurrent);
        }
    }
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
    obs <- length(errors);
    parametersTable <- cbind(coef(object),object$coefficientsSE,object$importance);

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

#' @importFrom stats vcov
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
