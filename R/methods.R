##### IC functions #####

#' Corrected Akaike's Information Criterion
#'
#' This function extracts AICc from "smooth" objects.
#'
#' AICc was proposed by Nariaki Sugiura in 1978 and is used on small samples.
#'
#' @aliases AICc
#' @param object Time series model.
#' @param ...  Some stuff.
#' @return This function returns numeric value.
#' @author Ivan Svetunkov, \email{ivan@@svetunkov.ru}
#' @seealso \link[stats]{AIC}, \link[stats]{BIC}
#' @references Kenneth P. Burnham, David R. Anderson (1998). Model Selection
#' and Multimodel Inference. Springer Science & Business Media.
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
#'
#' @export AICc
AICc <- function(object, ...) UseMethod("AICc")


#' @export
AICc.default <- function(object, ...){
    obs <- nobs(object);

    llikelihood <- logLik(object);
    nParam <- attributes(llikelihood)$df;
    llikelihood <- llikelihood[1:length(llikelihood)];

    IC <- 2*nParam - 2*llikelihood + 2 * nParam * (nParam + 1) / (obs - nParam - 1);

    return(IC);
}


#' @importFrom forecast getResponse
#' @export
forecast::getResponse

#' @importFrom forecast forecast
#' @export forecast
forecast::forecast

#' @importFrom stats predict.lm
#' @export
forecast.lm.combined <- function(object, newdata, ...){
    if(!is.data.frame(newdata)){
        newdata <- as.data.frame(newdata);
    }
    ellipsis <- list(...);
    if(!any(names(ellipsis)=="interval")){
        ellipsis$interval <- "p";
    }
    ellipsis$object <- object;
    ellipsis$newdata <- newdata;
    ourForecast <- do.call(predict.lm, ellipsis);
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

#' @export
getResponse.lm.combined <- function(object, ...){
    responseVariable <- object$model[,1];
    names(responseVariable) <- c(1:length(responseVariable));
    return(responseVariable);
}

#' @importFrom stats nobs fitted
#' @export
nobs.lm.combined <- function(object, ...){
    return(length(fitted(object)));
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
            plot(yActuals, type="l", xlim=range(min(yActuals),max(yActuals,yForecast)));
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
            plot(yActuals, type="l", xlim=range(min(yActuals),max(yActuals,yForecast)));
            lines(yForecast, col="blue", lwd=2);
        }
        else{
            smooth::graphmaker(yActuals, yForecast, yFitted);
        }
    }
}

#' @export
print.summary.lm.combined <- function(x, ...){
    cat("Coefficients:\n");
    print(x$parametersTable);
    cat("---\n");
    cat(paste0("Residual standard error: ",x$sigma," on ",x$df[2]," degrees of freedom:\n"));
    cat("Combined ICs:\n");
    print(x$ICs);
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
summary.lm.combined <- function(object, level=0.95, digits=5, ...){

    # Extract the values from the object
    errors <- residuals(object);
    obs <- length(errors);
    parametersTable <- cbind(coef(object),object$coefficientsSE,object$importance);

    # Calculate the quantiles for parameters and add them to the table
    paramQuantiles <- qt((1+level)/2,df=object$df.residual);
    parametersTable <- cbind(parametersTable,parametersTable[,1]-paramQuantiles*parametersTable[,2],
                             parametersTable[,1]+paramQuantiles*parametersTable[,2])
    rownames(parametersTable) <- names(object$coefficients);
    colnames(parametersTable) <- c("Estimate","Std. Error","Importance",
                                   paste0("Lower ",(1-level)/2*100,"%"), paste0("Upper ",(1+level)/2*100,"%"));
    parametersTable <- round(parametersTable,digits);

    # Extract degrees of freedom
    df <- c(object$df, object$df.residual, object$rank);
    # Calculate s.e. of residuals
    residSE <- round(sqrt(sum(errors^2)/df[2]),digits);

    ICs <- round(c(AIC(object),AICc(object),BIC(object)),digits);
    names(ICs) <- c("AIC","AICc","BIC");

    R2 <- 1 - sum(errors^2) / sum((object$model[,1]-mean(object$model[,1]))^2)
    R2Adj <- 1 - (1 - R2) * (obs - 1) / (obs - df[1]);

    ourReturn <- structure(list(parametersTable=parametersTable, sigma=residSE,
                                ICs=ICs, df=df, r.squared=R2, adj.r.squared=R2Adj),
                           class="summary.lm.combined");
    return(ourReturn);
}

plot.lm.combined <- function(x, ...){
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
