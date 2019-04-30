#' Calculate Cramer's V for categorical variables
#'
#' Function calculates Cramer's V for two categorical variables based on the table
#' function
#'
#' The function calculates Cramer's V and also returns the associated statistics from
#' Chi-Squared test with the null hypothesis of independence of the two variables.
#'
#' See details in the vignette "Marketing analytics with greybox":
#' \code{vignette("maUsingGreybox","greybox")}
#'
#' @template author
#' @keywords htest
#'
#' @param x First categorical variable.
#' @param y Second categorical variable.
#' @param use What observations to use. See \link[stats]{cor} function for details.
#' The only option that is not available here is \code{"pairwise.complete.obs"}.
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
cramer <- function(x, y, use=c("na.or.complete","complete.obs","everything","all.obs")){

    use <- substr(use[1],1,1);

    # Function returns values or NAs or error
    returner <- function(errorType=c(0,1,2)){
        if(errorType==0){
            return(structure(list(value=cramerValue,statistic=chiTest$statistic,
                                  p.value=chiTest$p.value,df=chiTest$parameter),class="cramer"));
        }
        else if(errorType==1){
            return(structure(list(value=NA,statistic=NA,
                                  p.value=NA,df=NA),class="cramer"));
        }
        else{
            stop("Missing observations in cramer", call.=FALSE);
        }
    }

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

    # Check the presence of NAs
    obsNAx <- is.na(x);
    obsNAy <- is.na(y);
    if(any(obsNAx) | any(obsNAy)){
        if(use=="e"){
            return(returner(1));
        }
        else if(use=="a"){
            returner(2);
        }
        else if(any(use==c("n","c"))){
            x <- x[!obsNAx & !obsNAy];
            y <- y[!obsNAx & !obsNAy];
            if(length(x)<2){
                if(use=="c"){
                    return(returner(1));
                }
                else{
                    returner(2);
                }
            }
        }
    }

    dataTable <- table(x,y);
    chiTest <- suppressWarnings(chisq.test(dataTable));
    cramerValue <- sqrt(chiTest$statistic/(min(dim(dataTable)-1)*sum(dataTable)));

    return(returner(0));
}
