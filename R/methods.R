#' @importFrom stats nobs fitted
#' @export
nobs.lm.combined <- function(object, ...){
    return(length(fitted(object)));
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
    df <- c(object$rank, object$df.residual, object$rank);
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
