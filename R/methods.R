#' @export
print.lm.combined <- function(x, digits=5, ...){
    cat("Coefficients:\n");
    print(round(coef(x),digits));
}

#' @export
summary.lm.combined <- function(object, level=0.95, digits=5, ...){
    parametersTable <- cbind(coef(object),object$coefficientsSE,object$importance);
    paramQuantiles <- qt((1+level)/2,df=object$df.residual);
    parametersTable <- cbind(parametersTable,parametersTable[,1]-paramQuantiles*parametersTable[,2],
                             parametersTable[,1]+paramQuantiles*parametersTable[,2])
    rownames(parametersTable) <- names(object$coefficients);
    colnames(parametersTable) <- c("Estimate","Std. Error","Rel. Importance",
                                   paste0("Lower ",(1-level)/2*100,"%"), paste0("Upper ",(1+level)/2*100,"%"));
    residSE <- round(sqrt(sum(residuals(object)^2)/object$df.residual),digits);

    cat("Coefficients:\n");
    print(round(parametersTable,digits));
    cat("---\n");
    cat(paste0("Residual standard error: ",residSE," on ",object$df.residual," degrees of freedom:\n"));
    cat(paste0("Combined ",names(object$IC),": ",round(object$IC, digits)));
}
