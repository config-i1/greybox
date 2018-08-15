#' Greybox classes checkers
#'
#' Functions to check if an object is of the specified class
#'
#' The list of functions includes:
#' \itemize{
#' \item \code{is.greybox()} tests if the object was produced by a greybox function
#' (e.g. \link[greybox]{alm} / \link[greybox]{stepwise} / \link[greybox]{lmCombine}
#' / \link[greybox]{lmDynamic});
#' \item \code{is.alm()} tests if the object was produced by \code{alm()} function;
#' \item \code{is.greyboxC()} tests if the object was produced by \code{lmCombine()}
#' function;
#' \item \code{is.greyboxD()} tests if the object was produced by \code{lmDynamic()}
#' function;
#' \item \code{is.rmc()} tests if the object was produced by \code{rmc()} function;
#' \item \code{is.rollingOrigin()} tests if the object was produced by \code{ro()}
#' function.
#' }
#'
#' @param x The object to check.
#' @return  \code{TRUE} if this is the specified class and \code{FALSE} otherwise.
#'
#' @template author
#' @keywords ts univar
#' @examples
#'
#' xreg <- cbind(rlaplace(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rlaplace(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#'
#' ourModel <- alm(y~x1+x2, xreg, distribution="dnorm")
#'
#' is.alm(ourModel)
#' is.greybox(ourModel)
#' is.greyboxC(ourModel)
#' is.greyboxD(ourModel)
#'
#' @rdname isFunctions
#' @export
is.greybox <- function(x){
    if(any(class(x)=="greybox")){
        return(TRUE);
    }
    else{
        return(FALSE);
    }
}

#' @rdname isFunctions
#' @export
is.alm <- function(x){
    if(any(class(x)=="alm")){
        return(TRUE);
    }
    else{
        return(FALSE);
    }
}

#' @rdname isFunctions
#' @export
is.greyboxC <- function(x){
    if(any(class(x)=="greyboxC")){
        return(TRUE);
    }
    else{
        return(FALSE);
    }
}

#' @rdname isFunctions
#' @export
is.greyboxD <- function(x){
    if(any(class(x)=="greyboxD")){
        return(TRUE);
    }
    else{
        return(FALSE);
    }
}

#' @rdname isFunctions
#' @export
is.rollingOrigin <- function(x){
    if(any(class(x)=="rollingOrigin")){
        return(TRUE);
    }
    else{
        return(FALSE);
    }
}

#' @rdname isFunctions
#' @export
is.rmc <- function(x){
    if(any(class(x)=="rmc")){
        return(TRUE);
    }
    else{
        return(FALSE);
    }
}
