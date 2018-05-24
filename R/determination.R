#' Determination coefficients
#'
#' Function produces determintation coefficient for the provided data
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
#' @param ... Other values passed to cor function.
#'
#' @return Function returns the vector of determintation coefficients.
#'
#' @seealso \code{\link[stats]{cor}}
#'
#' @examples
#'
#' ### Simple example
#' xreg <- cbind(rnorm(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rnorm(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("x1","x2","x3","Noise")
#' determination(xreg)
#'
#' @export determination
determination <- function(xreg, ...){

    # Produce correlation matrix
    matrixCorrelations <- cor(xreg, ...);
    # Calculate its determinant
    detCorrelations <- det(matrixCorrelations);
    nVariables <- nrow(matrixCorrelations);
    # Form the vector to return
    vectorCorrelationsMultiple <- rep(NA,nVariables);
    names(vectorCorrelationsMultiple) <- colnames(matrixCorrelations);
    if(nrow(xreg)<=ncol(xreg)){
        vectorCorrelationsMultiple[] <- 1;
        warning(paste0("The number of variables is larger than the number of observations. ",
                       "All the coefficients of determination are equal to one."), call.=FALSE);
    }
    # Calculate the multiple determinations
    for(i in 1:nVariables){
        vectorCorrelationsMultiple[i] <- 1 - detCorrelations / det(matrixCorrelations[-i,-i]);
    }

    return(vectorCorrelationsMultiple);
}