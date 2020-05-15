#' Exogenous variables cross-products
#'
#' Function generates the cross-products of the provided exogenous variables.
#'
#' This function might be useful if you have several variables and want to
#' introduce their cross-products. This might be useful when introducing the
#' interactions between dummy and continuous variables.
#'
#' @param xreg matrix or data.frame, containing variables that need
#' to be expanded. This matrix needs to contain at least two columns.
#' @param silent If \code{silent=FALSE}, then the progress is printed out.
#' Otherwise the function won't print anything in the console.
#'
#' @return \code{ts} matrix with the transformed and the original variables
#' is returned.
#'
#' @template author
#' @template keywords
#'
#' @seealso \code{\link[smooth]{es}, \link[greybox]{stepwise},
#' \link[greybox]{xregExpander}, \link[greybox]{xregTransformer}}
#'
#' @examples
#' # Create matrix of two variables and expand it
#' x <- cbind(rnorm(100,100,1),rnorm(100,50,3))
#' xregMultiplier(x)
#'
#' @export
xregMultiplier <- function(xreg, silent=TRUE){

    # Check and prepare the data
    if(is.data.frame(xreg)){
        xreg <- as.matrix(xreg);
    }

    if(!is.matrix(xreg) & !is.data.frame(xreg)){
        stop("Sorry, but we need to have either a matrix or a data.frame in xreg.", call.=FALSE);
    }

    if(!silent){
        cat("Preparing matrices...    ");
    }

    xregStart <- start(xreg);
    xregFrequency <- frequency(xreg);
    xregNames <- colnames(xreg);
    if(is.null(xregNames)){
        xregNames <- paste0("x",1:ncol(xreg));
    }
    obs <- nrow(xreg);
    nExovars <- ncol(xreg);
    nCombinations <- factorial(nExovars)/(factorial(2)*factorial(nExovars-2));

    xregNew <- matrix(0,obs,nCombinations+nExovars);
    # This is needed fro weird cases, when R fails to create the matrix of the necessary size
    if(ncol(xregNew)!=nCombinations+nExovars){
        xregNew <- cbind(xregNew,0);
    }
    xregNew <- ts(xregNew,start=xregStart,frequency=xregFrequency);
    xregNew[,1:nExovars] <- xreg;
    colnames(xregNew)[1:nExovars] <- xregNames;

    k <- 1;
    for(i in 1:nExovars){
        for(j in i:nExovars){
            if(i==j){
                next;
            }
            if(!silent){
                if(k==1){
                    cat("\b");
                }
                cat(paste0(rep("\b",nchar(round((k-1)/nCombinations,2)*100)+1),collapse=""));
                cat(paste0(round(k/nCombinations,2)*100,"%"));
            }
            xregNew[,nExovars+k] <- xreg[,i] * xreg[,j];
            colnames(xregNew)[nExovars+k] <- paste(xregNames[i],xregNames[j],sep="_");
            k <- k + 1;
        }
    }
    if(!silent){
        cat(paste0(rep("\b",4),collapse=""));
        cat(" Done! \n");
    }

    return(xregNew);
}
