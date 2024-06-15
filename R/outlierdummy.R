#' Outlier detection and matrix creation
#'
#' Function detects outliers and creates a matrix with dummy variables. Only point
#' outliers are considered (no level shifts).
#'
#' The detection is done based on the type of distribution used and confidence level
#' specified by user.
#'
#' @template author
#'
#' @param object Model estimated using one of the functions of smooth package.
#' @param level Confidence level to use. Everything that is outside the constructed
#' bounds based on that is flagged as outliers.
#' @param type Type of residuals to use: either standardised or studentised. Ignored
#' if count distributions used.
#' @param ... Other parameters. Not used yet.
#' @return The class "outlierdummy", which contains the list:
#' \itemize{
#' \item outliers - the matrix with the dummy variables, flagging outliers;
#' \item statistic - the value of the statistic for the normalised variable;
#' \item id - the ids of the outliers (which observations have them);
#' \item level - the confidence level used in the process;
#' \item type - the type of the residuals used;
#' \item errors - the errors used in the detection. In case of count distributions,
#' probabilities are returned.
#' }
#'
#' @seealso \link[stats]{influence.measures}
#' @examples
#'
#' # Generate the data with S distribution
#' xreg <- cbind(rnorm(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rs(100,0,3),xreg)
#' colnames(xreg) <- c("y","x1","x2")
#'
#' # Fit the normal distribution model
#' ourModel <- alm(y~x1+x2, xreg, distribution="dnorm")
#'
#' # Detect outliers
#' xregOutlierDummy <- outlierdummy(ourModel)
#'
#' @rdname outlierdummy
#' @export outlierdummy
outlierdummy <-  function(object, ...) UseMethod("outlierdummy")

#' @rdname outlierdummy
#' @export
outlierdummy.default <- function(object, level=0.999, type=c("rstandard","rstudent"), ...){
    # Function returns the matrix of dummies with outliers
    type <- match.arg(type);
    errors <- switch(type,"rstandard"=rstandard(object),"rstudent"=rstudent(object));
    statistic <- qnorm(c((1-level)/2, (1+level)/2), 0, 1);
    outliersID <- which(errors>statistic[2] | errors <statistic[1]);
    outliersNumber <- length(outliersID);
    if(outliersNumber>0){
        outliers <- matrix(0, nobs(object), outliersNumber,
                           dimnames=list(rownames(object$data),
                                         paste0("outlier",c(1:outliersNumber))));
        outliers[cbind(outliersID,c(1:outliersNumber))] <- 1;
    }
    else{
        outliers <- NULL;
    }

    return(structure(list(outliers=outliers, statistic=statistic, id=outliersID,
                          level=level, type=type),
                     class="outlierdummy"));
}

#' @rdname outlierdummy
#' @export
outlierdummy.alm <- function(object, level=0.999, type=c("rstandard","rstudent"), ...){
    # Function returns the matrix of dummies with outliers
    type <- match.arg(type);

    errors <- switch(type,"rstandard"=rstandard(object),"rstudent"=rstudent(object));
    statistic <- switch(object$distribution,
                        "dlaplace"=,
                        "dllaplace"=qlaplace(c((1-level)/2, (1+level)/2), 0, 1),
                        "dalaplace"=qalaplace(c((1-level)/2, (1+level)/2), 0, 1, object$other$alpha),
                        "dlogis"=qlogis(c((1-level)/2, (1+level)/2), 0, 1),
                        "dt"=qt(c((1-level)/2, (1+level)/2), nobs(object)-nparam(object)),
                        "dgnorm"=,
                        "dlgnorm"=qgnorm(c((1-level)/2, (1+level)/2), 0, 1, object$other$shape),
                        "ds"=,
                        "dls"=qs(c((1-level)/2, (1+level)/2), 0, 1),
                        # In the next one, the scale is debiased, taking n-k into account
                        "dinvgauss"=qinvgauss(c((1-level)/2, (1+level)/2), mean=1,
                                              dispersion=extractScale(object) * nobs(object) /
                                                  (nobs(object)-nparam(object))),
                        "dgamma"=qgamma(c((1-level)/2, (1+level)/2), shape=1/extractScale(object), scale=extractScale(object)),
                        "dexp"=qexp(c((1-level)/2, (1+level)/2), rate=1),
                        "dfnorm"=qfnorm(c((1-level)/2, (1+level)/2), 0, 1),
                        "drectnorm"=qrectnorm(c((1-level)/2, (1+level)/2), 0, 1),
                        # For count distributions, we do transform into std.Normal
                        "dpois"=,
                        "dnbinom"=,
                        "dgeom"=,
                        qnorm(c((1-level)/2, (1+level)/2), 0, 1));
    outliersID <- which(errors>statistic[2] | errors<statistic[1]);
    outliersNumber <- length(outliersID);
    if(outliersNumber>0){
        outliers <- matrix(0, nobs(object), outliersNumber,
                           dimnames=list(rownames(object$data),
                                         paste0("outlier",c(1:outliersNumber))));
        outliers[cbind(outliersID,c(1:outliersNumber))] <- 1;
    }
    else{
        outliers <- NULL;
    }

    return(structure(list(outliers=outliers, statistic=statistic, id=outliersID,
                          level=level, type=type, errors=errors),
                     class="outlierdummy"));
}

#' @export
print.outlierdummy <- function(x, ...){
    cat(paste0("Number of identified outliers: ", length(x$id),
               "\nConfidence level: ",x$level,
               "\nType of residuals: ",x$type, "\n"));
}
