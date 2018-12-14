#' Intraclass correlation
#'
#' Function calculates Intraclass correlation for two variables
#'
#' This is based on the linear regression model with the set of dummies providede in x.
#' The returned value is just a coefficient of multiple correlation from such a model,
#' the F-statistics of the model (thus testing the null hypothesis that all the
#' parameters are equal to zero), the associated p-value and the degrees of freedom.
#'
#' @template author
#' @template keywords
#'
#' @param x Either factor or a matrix with dummies
#' @param y The numerical variable.
#'
#' @return The following list of values is returned:
#' \itemize{
#' \item{value}{The value of the coefficient;}
#' \item{statistic}{The value of F-statistics associated with the parameter;}
#' \item{p.value}{The p-value of F-statistics associated with the parameter;}
#' \item{df.residual}{The number of degrees of freedom for the residuals;}
#' \item{df}{The number of degrees of freedom for the data.}
#' }
#'
#' @seealso \code{\link[base]{table}, \link[greybox]{tableplot}, \link[greybox]{spread},
#' \link[greybox]{cramer}}
#'
#' @examples
#'
#' icc(mtcars$am, mtcars$mpg)
#'
#' @importFrom stats cor.test pf
#' @export icc
icc <- function(x,y){

    # Remove the bloody ordering
    if(is.factor(x)){
        if(is.ordered(x)){
            x <- factor(x,ordered=FALSE);
        }
        x <- model.matrix(~x-1);
    }
    else{
        if(!is.matrix(x)){
            x <- factor(x,levels=unique(x));
            x <- model.matrix(~x-1);
        }
    }

    if(!is.numeric(y)){
        warning("The y variable should be numeric in order for this to work! The reported value might be meaningless",call.=FALSE);
        if(is.factor(y)){
            if(is.ordered(y)){
                y <- factor(y,ordered=FALSE);
            }
            y <- model.matrix(~y-1);
        }
    }

    lmFit <- .lm.fit(x,y);

    value <- sqrt(1 - sum(residuals(lmFit)^2) / sum((y-mean(y))^2));
    dfFull <- length(residuals(lmFit))-1;
    dfReg <- lmFit$rank;
    statistic <- (value^2/dfFull)/((1-value^2)/dfReg);

    return(structure(list(value=value,statistic=statistic,df=dfFull,
                          df.residual=dfReg,p.value=pf(statistic,dfFull,dfReg)),class="icc"));
}
