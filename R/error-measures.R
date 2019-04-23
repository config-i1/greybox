#' Error measures
#'
#' Functions allow to calculate different types of errors for point and
#' interval predictions:
#' \enumerate{
#' \item MAE - Mean Absolute Error,
#' \item MSE - Mean Squared Error,
#' \item MRE - Mean Root Error (Kourentzes, 2014),
#' \item MIS - Mean Interval Score (Gneiting & Raftery, 2007),
#' \item MPE - Mean Percentage Error,
#' \item MAPE - Mean Absolute Percentage Error (See Svetunkov, 2017 for
#' the critique),
#' \item MASE - Mean Absolute Scaled Error (Hyndman & Koehler, 2006)),
#' \item RelMAE - Relative Mean Absolute Error (Davydenko & Fildes, 2013),
#' \item RelRMSE - Relative Root Mean Squared Error,
#' \item RelAME - Relative Absolute Mean Error,
#' \item RelMIS - Relative Mean Interval Score,
#' \item sMSE - Scaled Mean Squared Error (Petropoulos & Kourentzes, 2015),
#' \item sPIS- Scaled Periods-In-Stock (Wallstrom & Segerstedt, 2010),
#' \item sCE - Scaled Cumulative Error,
#' \item sMIS - Scaled Mean Interval Score.
#' }
#'
#' In case of \code{sMSE}, \code{scale} needs to be a squared value. Typical
#' one -- squared mean value of in-sample actuals.
#'
#' If all the measures are needed, then \link[greybox]{measures} function
#' can help.
#'
#' There are several other measures, see details of \link[greybox]{pinball}
#' and \link[greybox]{hm}.
#'
#' @template author
#'
#' @aliases Errors
#' @param actual The vector or matrix of actual values.
#' @param forecast The vector or matrix of forecasts values.
#' @param lower The lower bound of the prediction interval.
#' @param upper The upper bound of the prediction interval.
#' @param scale The value that should be used in the denominator of MASE. Can
#' be anything but advised values are: mean absolute deviation of in-sample one
#' step ahead Naive error or mean absolute value of the in-sample actuals.
#' @param benchmark The vector or matrix of the forecasts of the benchmark
#' model.
#' @param benchmarkLower The lower bound of the prediction interval of the
#' benchmark model.
#' @param benchmarkUpper The upper bound of the prediction interval of the
#' benchmark model.
#' @param level The confidence level of the constructed interval.
#' @return All the functions return the scalar value.
#' @references \itemize{
#' \item Kourentzes N. (2014). The Bias Coefficient: a new metric for forecast bias
#' \url{https://kourentzes.com/forecasting/2014/12/17/the-bias-coefficient-a-new-metric-for-forecast-bias/}
#' \item Svetunkov, I. (2017). Naughty APEs and the quest for the holy grail.
#' \url{https://forecasting.svetunkov.ru/en/2017/07/29/naughty-apes-and-the-quest-for-the-holy-grail/}
#' \item Fildes R. (1992). The evaluation of
#' extrapolative forecasting methods. International Journal of Forecasting, 8,
#' pp.81-98.
#' \item Hyndman R.J., Koehler A.B. (2006). Another look at measures of
#' forecast accuracy. International Journal of Forecasting, 22, pp.679-688.
#' \item Petropoulos F., Kourentzes N. (2015). Forecast combinations for
#' intermittent demand. Journal of the Operational Research Society, 66,
#' pp.914-924.
#' \item Wallstrom P., Segerstedt A. (2010). Evaluation of forecasting error
#' measurements and techniques for intermittent demand. International Journal
#' of Production Economics, 128, pp.625-636.
#' \item Davydenko, A., Fildes, R. (2013). Measuring Forecasting Accuracy:
#' The Case Of Judgmental Adjustments To Sku-Level Demand Forecasts.
#' International Journal of Forecasting, 29(3), 510-522.
#' \url{https://doi.org/10.1016/j.ijforecast.2012.09.002}
#' \item Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
#' prediction, and estimation. Journal of the American Statistical Association,
#' 102(477), 359–378. \url{https://doi.org/10.1198/016214506000001437}
#' }
#'
#' @seealso \link[greybox]{pinball}, \link[greybox]{hm}, \link[greybox]{measures}
#'
#' @examples
#'
#'
#' y <- rnorm(100,10,2)
#' testForecast <- rep(mean(y[1:90]),10)
#'
#' MAE(y[91:100],testForecast)
#' MSE(y[91:100],testForecast)
#'
#' MPE(y[91:100],testForecast)
#' MAPE(y[91:100],testForecast)
#'
#' # Measures from Petropoulos & Kourentzes (2015)
#' MASE(y[91:100],testForecast,mean(abs(y[1:90])))
#' sMSE(y[91:100],testForecast,mean(abs(y[1:90]))^2)
#' sPIS(y[91:100],testForecast,mean(abs(y[1:90])))
#' sCE(y[91:100],testForecast,mean(abs(y[1:90])))
#'
#' # Original MASE from Hyndman & Koehler (2006)
#' MASE(y[91:100],testForecast,mean(abs(diff(y[1:90]))))
#'
#' testForecast2 <- rep(y[91],10)
#' # Relative measures, from and inspired by Davydenko & Fildes (2013)
#' RelMAE(y[91:100],testForecast2,testForecast)
#' RelRMSE(y[91:100],testForecast2,testForecast)
#' RelAME(y[91:100],testForecast2,testForecast)
#'
#' #### Measures for the prediction intervals
#' # An example with mtcars data
#' ourModel <- alm(mpg~., mtcars[1:30,], distribution="dnorm")
#' ourBenchmark <- alm(mpg~1, mtcars[1:30,], distribution="dnorm")
#'
#' # Produce predictions with the interval
#' ourForecast <- predict(ourModel, mtcars[-c(1:30),], interval="p")
#' ourBenchmarkForecast <- predict(ourBenchmark, mtcars[-c(1:30),], interval="p")
#'
#' MIS(mtcars$mpg[-c(1:30)],ourForecast$lower,ourForecast$upper,0.95)
#' sMIS(mtcars$mpg[-c(1:30)],ourForecast$lower,ourForecast$upper,mean(mtcars$mpg[1:30]),0.95)
#' RelMIS(mtcars$mpg[-c(1:30)],ourForecast$lower,ourForecast$upper,
#'        ourBenchmarkForecast$lower,ourBenchmarkForecast$upper,0.95)
#'
#' ### Also, see pinball function for other measures for the intervals
#'
#' @rdname error-measures


#' @rdname error-measures
#' @export MAE
#' @aliases MAE
MAE <- function(actual,forecast){
# This function calculates Mean Absolute Error
# actual - actual values,
# forecast - forecasted values.
    if(length(actual) != length(forecast)){
        message("The length of the provided data differs.");
        message(paste0("Length of actual: ",length(actual)));
        message(paste0("Length of forecast: ",length(forecast)));
        stop("Cannot proceed.",call.=FALSE);
    }
    else{
        return(mean(abs(actual-forecast),na.rm=TRUE));
    }
}

#' @rdname error-measures
#' @export MSE
#' @aliases MSE
MSE <- function(actual,forecast){
# This function calculates Mean squared Error
# actual - actual values,
# forecast - forecasted values.
    if(length(actual) != length(forecast)){
        message("The length of the provided data differs.");
        message(paste0("Length of actual: ",length(actual)));
        message(paste0("Length of forecast: ",length(forecast)));
        stop("Cannot proceed.",call.=FALSE);
    }
    else{
        return(mean((actual-forecast)^2,na.rm=TRUE));
    }
}

#' @rdname error-measures
#' @export MRE
#' @aliases MRE
MRE <- function(actual,forecast){
# This function calculates Mean squared Error
# actual - actual values,
# forecast - forecasted values.
    if(length(actual) != length(forecast)){
        message("The length of the provided data differs.");
        message(paste0("Length of actual: ",length(actual)));
        message(paste0("Length of forecast: ",length(forecast)));
        stop("Cannot proceed.",call.=FALSE);
    }
    else{
        return(mean(sqrt(as.complex(actual-forecast)),na.rm=TRUE));
    }
}

#' @rdname error-measures
#' @export MIS
#' @aliases MIS
MIS <- function(actual,lower,upper,level=0.95){
# This function calculates Mean Interval Score from Gneiting & Raftery, 2007
# actual - actual values,
# lower - the lower bound of the interval,
# upper - the upper bound of the interval,
    if(level>1){
        level[] <- level / 100;
    }
    alpha <- 1-level;
    lengthsVector <- c(length(actual),length(upper),length(lower))
    if(any(lengthsVector>min(lengthsVector))){
        message("The length of the provided data differs.");
        message(paste0("Length of actual: ",length(actual)));
        message(paste0("Length of lower: ",length(lower)));
        message(paste0("Length of upper: ",length(upper)));
        stop("Cannot proceed.",call.=FALSE);
    }
    else{
        h <- length(actual);
        MISValue <- sum(upper-lower) + 2/alpha*(sum((lower-actual)*(actual<lower)) + sum((actual-upper)*(actual>upper)));
        MISValue[] <- MISValue / h;
        return(MISValue);
    }
}

#' @rdname error-measures
#' @export MPE
#' @aliases MPE
MPE <- function(actual,forecast){
# This function calculates Mean / Median Percentage Error
# actual - actual values,
# forecast - forecasted or fitted values.
    if(length(actual) != length(forecast)){
        message("The length of the provided data differs.");
        message(paste0("Length of actual: ",length(actual)));
        message(paste0("Length of forecast: ",length(forecast)));
        stop("Cannot proceed.",call.=FALSE);
    }
    else{
        return(mean((actual-forecast)/actual,na.rm=TRUE));
    }
}

#' @rdname error-measures
#' @export MAPE
#' @aliases MAPE
MAPE <- function(actual,forecast){
# This function calculates Mean Absolute Percentage Error
# actual - actual values,
# forecast - forecasted values.
    if(length(actual) != length(forecast)){
        message("The length of the provided data differs.");
        message(paste0("Length of actual: ",length(actual)));
        message(paste0("Length of forecast: ",length(forecast)));
        stop("Cannot proceed.",call.=FALSE);
    }
    else{
        return(mean(abs((actual-forecast)/actual),na.rm=TRUE));
    }
}

#' @rdname error-measures
#' @export MASE
#' @aliases MASE
MASE <- function(actual,forecast,scale){
# This function calculates Mean Absolute Scaled Error as in Hyndman & Koehler, 2006
# actual - actual values,
# forecast - forecasted values.
# scale - the measure to scale errors with. Usually - MAE of in-sample.
    if(length(actual) != length(forecast)){
        message("The length of the provided data differs.");
        message(paste0("Length of actual: ",length(actual)));
        message(paste0("Length of forecast: ",length(forecast)));
        stop("Cannot proceed.",call.=FALSE);
    }
    else{
        return(mean(abs(actual-forecast),na.rm=TRUE)/scale);
    }
}

#' @rdname error-measures
#' @export RelMAE
#' @aliases RelMAE
RelMAE <-function(actual,forecast,benchmark){
# This function calculates Average Relative MAE
# actual - actual values,
# forecast - forecasted or fitted values.
# benchmark - forecasted or fitted values of etalon method.
    if((length(actual) != length(forecast)) | (length(actual) != length(benchmark)) | (length(benchmark) != length(forecast))){
        message("The length of the provided data differs.");
        message(paste0("Length of actual: ",length(actual)));
        message(paste0("Length of forecast: ",length(forecast)));
        message(paste0("Length of benchmark: ",length(benchmark)));
        stop("Cannot proceed.",call.=FALSE);
    }
    else{
        if(all(forecast==benchmark)){
            return(1);
        }
        else{
            return(mean(abs(actual-forecast),na.rm=TRUE)/
                             mean(abs(actual-benchmark),na.rm=TRUE));
        }
    }
}

#' @rdname error-measures
#' @export RelRMSE
#' @aliases RelRMSE
RelRMSE <-function(actual,forecast,benchmark){
    # This function calculates Relative MSE
    # actual - actual values,
    # forecast - forecasted or fitted values.
    # benchmark - forecasted or fitted values of etalon method.
    if((length(actual) != length(forecast)) | (length(actual) != length(benchmark)) | (length(benchmark) != length(forecast))){
        message("The length of the provided data differs.");
        message(paste0("Length of actual: ",length(actual)));
        message(paste0("Length of forecast: ",length(forecast)));
        message(paste0("Length of benchmark: ",length(benchmark)));
        stop("Cannot proceed.",call.=FALSE);
    }
    else{
        if(all(forecast==benchmark)){
            return(1);
        }
        else{
            return(sqrt(mean((actual-forecast)^2,na.rm=TRUE)/
                             mean((actual-benchmark)^2,na.rm=TRUE)));
        }
    }
}

#' @rdname error-measures
#' @export RelAME
#' @aliases RelAME
RelAME <-function(actual,forecast,benchmark){
    # This function calculates Relative Absolute ME
    # actual - actual values,
    # forecast - forecasted or fitted values.
    # benchmark - forecasted or fitted values of etalon method.
    if((length(actual) != length(forecast)) | (length(actual) != length(benchmark)) | (length(benchmark) != length(forecast))){
        message("The length of the provided data differs.");
        message(paste0("Length of actual: ",length(actual)));
        message(paste0("Length of forecast: ",length(forecast)));
        message(paste0("Length of benchmark: ",length(benchmark)));
        stop("Cannot proceed.",call.=FALSE);
    }
    else{
        if(all(forecast==benchmark)){
            return(1);
        }
        else{
            return(abs(mean((actual-forecast),na.rm=TRUE))/
                             abs(mean((actual-benchmark),na.rm=TRUE)));
        }
    }
}

#' @rdname error-measures
#' @export RelMIS
#' @aliases RelMIS
RelMIS <-function(actual,lower,upper,benchmarkLower,benchmarkUpper,level=0.95){
# This function calculates scaled MIS
# actual - actual values,
# forecast - forecasted values.
# scale - the measure to scale errors with.
# lower - the lower bound of the interval,
# upper - the upper bound of the interval,
# benchmarkLower - the lower bound of the interval of the benchmark method.
# benchmarkUpper - the upper bound of the interval of the benchmark method.
    lengthsVector <- c(length(actual),length(upper),length(lower),length(benchmarkLower),length(benchmarkUpper));
    if(any(lengthsVector>min(lengthsVector))){
        message("The length of the provided data differs.");
        stop("Cannot proceed.",call.=FALSE);
    }
    else{
        return(MIS(actual=actual,lower=lower,upper=upper,level=level) /
                   MIS(actual=actual,lower=benchmarkLower,upper=benchmarkUpper,level=level));
    }
}

#' @rdname error-measures
#' @export sMSE
#' @aliases sMSE
sMSE <- function(actual,forecast,scale){
# This function calculates scaled Mean Squared Error.
# Attention! Scale factor should be provided as squares of something!
# actual - actual values,
# forecast - forecasted values.
# scale - the measure to scale errors with. Usually - MAE of in-sample.
    if(length(actual) != length(forecast)){
        message("The length of the provided data differs.");
        message(paste0("Length of actual: ",length(actual)));
        message(paste0("Length of forecast: ",length(forecast)));
        stop("Cannot proceed.",call.=FALSE);
    }
    else{
        return(mean((actual-forecast)^2,na.rm=TRUE)/scale);
    }
}

#' @rdname error-measures
#' @export sPIS
#' @aliases sPIS
sPIS <- function(actual,forecast,scale){
# This function calculates scaled Periods-In-Stock.
# actual - actual values,
# forecast - forecasted values.
# scale - the measure to scale errors with.
    if(length(actual) != length(forecast)){
        message("The length of the provided data differs.");
        message(paste0("Length of actual: ",length(actual)));
        message(paste0("Length of forecast: ",length(forecast)));
        stop("Cannot proceed.",call.=FALSE);
    }
    else{
        return(sum(cumsum(forecast-actual))/scale);
    }
}

#' @rdname error-measures
#' @export sCE
#' @aliases sCE
sCE <- function(actual,forecast,scale){
# This function calculates scaled Cumulative Error.
# actual - actual values,
# forecast - forecasted values.
# scale - the measure to scale errors with.
    if(length(actual) != length(forecast)){
        message("The length of the provided data differs.");
        message(paste0("Length of actual: ",length(actual)));
        message(paste0("Length of forecast: ",length(forecast)));
        stop("Cannot proceed.",call.=FALSE);
    }
    else{
        return(sum(forecast-actual)/scale);
    }
}

#' @rdname error-measures
#' @export sMIS
#' @aliases sMIS
sMIS <- function(actual,lower,upper,scale,level=0.95){
# This function calculates scaled MIS
# actual - actual values,
# forecast - forecasted values.
# scale - the measure to scale errors with.
# lower - the lower bound of the interval,
# upper - the upper bound of the interval,
# scale - the measure to scale errors with.
    lengthsVector <- c(length(actual),length(upper),length(lower))
    if(any(lengthsVector>min(lengthsVector))){
        message("The length of the provided data differs.");
        message(paste0("Length of actual: ",length(actual)));
        message(paste0("Length of lower: ",length(lower)));
        message(paste0("Length of upper: ",length(upper)));
        stop("Cannot proceed.",call.=FALSE);
    }
    else{
        return(MIS(actual=actual,lower=lower,upper=upper,level=level)/scale);
    }
}

#' Error measures for the provided forecasts
#'
#' Function calculates several error measures using the provided
#' forecasts and the data for the holdout sample.
#'
#' @template author
#'
#' @aliases measures
#' @param holdout The vector of the holdout values.
#' @param forecast The vector of forecasts produced by a model.
#' @param actual The vector of actual in-sample values.
#' @param digits Number of digits of the output. If \code{NULL}
#' then no rounding is done.
#' @return The functions returns the named vector of errors:
#' \itemize{
#' \item MAE,
#' \item MSE
#' \item MPE,
#' \item MAPE,
#' \item MASE,
#' \item sMAE,
#' \item sMSE,
#' \item sCE,
#' \item RelMAE,
#' \item RelRMSE,
#' \item RelAME,
#' \item cbias,
#' \item sPIS.
#' }
#' For the details on these errors, see \link[greybox]{Errors}.
#' @references \itemize{
#' \item Svetunkov, I. (2017). Naughty APEs and the quest for the holy grail.
#' \url{https://forecasting.svetunkov.ru/en/2017/07/29/naughty-apes-and-the-quest-for-the-holy-grail/}
#' \item Fildes R. (1992). The evaluation of
#' extrapolative forecasting methods. International Journal of Forecasting, 8,
#' pp.81-98.
#' \item Hyndman R.J., Koehler A.B. (2006). Another look at measures of
#' forecast accuracy. International Journal of Forecasting, 22, pp.679-688.
#' \item Petropoulos F., Kourentzes N. (2015). Forecast combinations for
#' intermittent demand. Journal of the Operational Research Society, 66,
#' pp.914-924.
#' \item Wallstrom P., Segerstedt A. (2010). Evaluation of forecasting error
#' measurements and techniques for intermittent demand. International Journal
#' of Production Economics, 128, pp.625-636.
#' \item Davydenko, A., Fildes, R. (2013). Measuring Forecasting Accuracy:
#' The Case Of Judgmental Adjustments To Sku-Level Demand Forecasts.
#' International Journal of Forecasting, 29(3), 510-522.
#' \url{https://doi.org/10.1016/j.ijforecast.2012.09.002}
#' }
#' @examples
#'
#'
#' y <- rnorm(100,10,2)
#' ourForecast <- rep(mean(y[1:90]),10)
#'
#' measures(y[91:100],ourForecast,y[1:90],digits=5)
#'
#' @export measures
measures <- function(holdout, forecast, actual, digits=NULL){
    holdout <- as.vector(holdout);
    forecast <- as.vector(forecast);
    actual <- as.vector(actual);
    benchmark <- rep(actual[length(actual)],length(holdout));
    errormeasures <- c(MAE(holdout,forecast),
                       MSE(holdout,forecast),
                       MPE(holdout,forecast),
                       MAPE(holdout,forecast),
                       MASE(holdout,forecast,mean(abs(diff(actual)))),
                       MASE(holdout,forecast,mean(abs(actual))),
                       sMSE(holdout,forecast,mean(abs(actual[actual!=0]))^2),
                       sCE(holdout,forecast,mean(abs(actual[actual!=0]))),
                       RelMAE(holdout,forecast,benchmark),
                       RelRMSE(holdout,forecast,benchmark),
                       RelAME(holdout,forecast,benchmark),
                       cbias(holdout-forecast,0),
                       sPIS(holdout,forecast,mean(abs(actual[actual!=0]))));
    if(!is.null(digits)){
        errormeasures[] <- round(errormeasures, digits);
    }
    names(errormeasures) <- c("MAE","MSE",
                              "MPE","MAPE",
                              "MASE","sMAE","sMSE","sCE",
                              "RelMAE","RelRMSE","RelAME","cbias","sPIS");
    return(errormeasures);
}


#' Half moment of a distribution and its derivatives.
#'
#' \code{hm} function estimates half moment from some predefined constant
#' \code{C}. \code{ham} estimates half absolute moment. Finally, \code{cbias}
#' function returns bias based on \code{hm}.
#'
#' \code{NA} values of \code{x} are excluded on the first step of calculation.
#'
#' @template author
#'
#' @aliases hm
#' @param x A variable based on which HM is estimated.
#' @param C Centering parameter.
#' @param ...  Other parameters passed to mean function.
#' @return A complex variable is returned for \code{hm} function and real values
#' are returned for \code{cbias} and \code{ham}.
#' @examples
#'
#' x <- rnorm(100,0,1)
#' hm(x)
#' ham(x)
#' cbias(x)
#'
#' @export hm
#' @rdname hm
hm <- function(x,C=mean(x),...){
    # This function calculates half moment
    return(mean(sqrt(as.complex(x[!is.na(x)]-C)),...));
}

#' @rdname hm
#' @export ham
#' @aliases ham
ham <- function(x,C=mean(x),...){
    # This function calculates half moment
    return(mean(sqrt(abs(x[!is.na(x)]-C)),...));
}

#' @rdname hm
#' @export cbias
#' @aliases cbias
cbias <- function(x,C=mean(x),...){
    # This function calculates half moment
    return(1 - Arg(hm(x,C,...))/(pi/4));
}

#' Pinball function
#'
#' The function returns the value from the pinball function for the specified level and
#' the type of loss
#'
#' @template author
#'
#' @param holdout The vector or matrix of the holdout values.
#' @param forecast The forecast of prediction interval (should be the same length as the
#' holdout).
#' @param level The level of the prediction interval associated with the forecast.
#' @param loss The type of loss to use. The number which corresponds to L1, L2 etc.
#' @return The function returns the scalar value.
#' @examples
#' # An example with mtcars data
#' ourModel <- alm(mpg~., mtcars[1:30,], distribution="dnorm")
#'
#' # Produce predictions with the interval
#' ourForecast <- predict(ourModel, mtcars[-c(1:30),], interval="p")
#'
#' # Pinball with the L1 (quantile value)
#' pinball(mtcars$mpg[-c(1:30)],ourForecast$upper,level=0.975,loss=1)
#' pinball(mtcars$mpg[-c(1:30)],ourForecast$lower,level=0.025,loss=1)
#'
#' # Pinball with the L2 (expectile value)
#' pinball(mtcars$mpg[-c(1:30)],ourForecast$upper,level=0.975,loss=2)
#' pinball(mtcars$mpg[-c(1:30)],ourForecast$lower,level=0.025,loss=2)
#'
#' @export pinball
pinball <- function(holdout, forecast, level, loss=1){
    # This function calculates pinball cost function for the bound of prediction interval
    if(length(holdout) != length(forecast)){
        message("The length of the provided data differs.");
        message(paste0("Length of holdout: ",length(holdout)));
        message(paste0("Length of forecast: ",length(forecast)));
        stop("Cannot proceed.",call.=FALSE);
    }
    else{
        result <- ((1-level)*sum(abs((holdout-forecast))^loss * (holdout<=forecast)) +
                            level*sum(abs((holdout-forecast))^loss * (holdout>forecast)));
        return(result);
    }
}
