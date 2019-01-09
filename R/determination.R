#' Determination coefficients
#'
#' Function produces determination coefficient for the provided data
#'
#' The function calculates determination coefficients (aka R^2)
#' between all the provided variables. The higher the coefficient is,
#' the higher the potential multicollinearity effect in the model with
#' the variables will be. Coefficients of determination are connected
#' directly to Variance Inflation Factor (VIF): VIF = 1 / (1 -
#' determination). Arguably it is easier to interpret, because it is
#' restricted with (0, 1) bounds. The multicollinearity can be
#' considered as serious, when determination > 0.9 (which corresponds
#' to VIF > 10).
#'
#' @template author
#' @keywords models
#'
#' @param xreg Data frame or a matrix, containing the exogenous variables.
#' @param bruteForce If \code{TRUE}, then all the variables will be used
#' for the regression construction (sink regression). If the number of
#' observations is smaller than the number of series, the function will
#' use \link[greybox]{stepwise} function and select only meaningful
#' variables.
#' @param ... Other values passed to cor function.
#'
#' @return Function returns the vector of determination coefficients.
#'
#' @seealso \link[stats]{cor}, \link[greybox]{mcor}, \link[greybox]{stepwise}
#'
#' @examples
#'
#' ### Simple example
#' xreg <- cbind(rnorm(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rnorm(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("x1","x2","x3","Noise")
#' determination(xreg)
#'
#' @rdname determination
#' @aliases determ
#' @export determination
determination <- function(xreg, bruteForce=TRUE, ...){

    nVariables <- ncol(xreg);
    nSeries <- nrow(xreg);
    # Form the vector to return
    vectorCorrelationsMultiple <- rep(NA,nVariables);
    names(vectorCorrelationsMultiple) <- colnames(xreg);
    if(nSeries<=nVariables & bruteForce){
        # vectorCorrelationsMultiple[] <- 1;
        warning(paste0("The number of variables is larger than the number of observations. ",
                       "Sink regression cannot be constructed. Using stepwise."),
                call.=FALSE);
        bruteForce <- FALSE;
    }

    if(!bruteForce){
        determinationCalculator <- function(residuals, actuals){
            return(1 - sum(residuals^2) / sum((actuals-mean(actuals))^2));
        }
    }

    # Calculate the multiple determinations
    if(bruteForce & nVariables>1){
        for(i in 1:nVariables){
            vectorCorrelationsMultiple[i] <- suppressWarnings(mcor(xreg[,-i],xreg[,i])$value);
        }
        vectorCorrelationsMultiple <- vectorCorrelationsMultiple^2;
    }
    else if(!bruteForce & nVariables>1){
        testXreg <- xreg;
        testModel <- suppressWarnings(stepwise(testXreg));
        vectorCorrelationsMultiple[1] <- determinationCalculator(residuals(testModel),
                                                                 getResponse(testModel));
        for(i in 2:nVariables){
            testXreg[] <- xreg;
            testXreg[,1] <- xreg[,i];
            testXreg[,i] <- xreg[,1];
            testModel <- suppressWarnings(stepwise(testXreg));
            vectorCorrelationsMultiple[i] <- determinationCalculator(residuals(testModel),
                                                                     getResponse(testModel));
        }
    }
    else{
        vectorCorrelationsMultiple <- 0;
    }

    return(vectorCorrelationsMultiple);
}

#' @rdname determination
#' @export determ
determ <- determination;
