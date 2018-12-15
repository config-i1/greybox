#' Calculate Cramer's V for categorical variables
#'
#' Function calculates Cramer's V for two categorical variables based on the table
#' function
#'
#' The function calculates Cramer's V and also returns the associated statistics from
#' Chi-Squared test with the null hypothesis of independence of the two variables.
#'
#' @template author
#' @keywords htest
#'
#' @param x First categorical variable.
#' @param y Second categorical variable.
#'
#' @return The following list of values is returned:
#' \itemize{
#' \item{value}{The value of Cramer's V;}
#' \item{statistic}{The value of Chi squared statistic associated with the Cramer's V;}
#' \item{p.value}{The p-value of Chi squared test associated with the Cramer's V;}
#' \item{df}{The number of degrees of freedom from the test.}
#' }
#'
#' @seealso \code{\link[base]{table}, \link[greybox]{tableplot}, \link[greybox]{spread},
#' \link[greybox]{mcor}, \link[greybox]{association}}
#'
#' @examples
#'
#' cramer(mtcars$am, mtcars$gear)
#'
#' @importFrom stats chisq.test
#' @export cramer
cramer <- function(x, y){

    # Check, whether x and y ar categorical or at least numerical with only 10 levels
    if(is.numeric(x)){
        if(length(unique(x))>10){
            warning("It seems that x is numeric, not categorical. Other measures of association might be more informative.",
                    call.=FALSE);
        }
    }
    if(is.numeric(y)){
        if(length(unique(y))>10){
            warning("It seems that y is numeric, not categorical. Other measures of association might be more informative.",
                    call.=FALSE);
        }
    }

    dataTable <- table(x,y);
    chiTest <- suppressWarnings(chisq.test(dataTable));
    cramerValue <- sqrt(chiTest$statistic/(min(dim(dataTable)-1)*sum(dataTable)));
    # names(cramerValue) <- "Cramer's V";
    return(structure(list(value=cramerValue,statistic=chiTest$statistic,
                          p.value=chiTest$p.value,df=chiTest$parameter),class="cramer"));
}