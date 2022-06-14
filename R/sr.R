#' Create a dataset from the provided data and models
#'
#' Function creates a data.frame from the objects provided to it. The objects
#' include vector, matrix, data.frame and any model that supports the standard
#' methods (such as fitted, vcov, predict, forecast). In the latter case,
#' the function will extract fitted values and use them instead of the actual one.
#'
#' The function is needed as a first step in the estimation of a regression model
#' with stochastic regressors.
#'
#' @template author
#' @template keywords
#'
#' @param ... Objects that need to be merged in one dataset.
#'
#' @return Function returns \code{data} - the data.frame of the variables,
#' \code{objects} - the list of all provided objects and \code{dataType} - vector
#' with types of objects provided to folder.
#'
#' @examples
#'
#' ### Simple example
#' xreg <- cbind(rnorm(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rnorm(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#' modelForY <- stepwise(xreg)
#'
#' foldedData <- folder(y=xreg[,4], xreg[,-c(1,4)], z=modelForY)
#' vcov(foldedData)
#'
#' @export
folder <- function(...){
    # Function creates a dataset from the provided objects and models

    ellipsis <- list(...);
    listLength <- length(ellipsis);

    # Identify types of objects
    listDataFrame <- sapply(ellipsis, is.data.frame);
    listMatrix <- sapply(ellipsis, is.matrix);
    listVector <- sapply(ellipsis, is.vector);
    listModel <- !listDataFrame & !listMatrix & !listVector;

    # Create a vector of data types
    dataType <- rep("model",listLength);
    dataType[listDataFrame] <- "data.frame";
    dataType[listMatrix] <- "matrix";
    dataType[listVector] <- "vector";

    # Get the maximum number of observations
    obs <- max(unlist(c(sapply(ellipsis[listDataFrame], nrow),
                        sapply(ellipsis[listMatrix], nrow),
                        sapply(ellipsis[listVector], length),
                        sapply(ellipsis[listModel], nobs))))

    nVariables <- sum(unlist(c(sapply(ellipsis[listDataFrame], ncol),
                               sapply(ellipsis[listMatrix], ncol),
                               sum(listVector),
                               sapply(ellipsis[listModel], nvariate))));

    variablesNames <- vector("character",nVariables);
    variablesNumber <- vector("numeric",nVariables);
    dataCreated <- as.data.frame(matrix(NA,obs,nVariables));

    # Form the design matrix
    m <- 0;
    for(i in 1:listLength){
        if(any(dataType[i]==c("data.frame","matrix"))){
            nVariablesLoop <- ncol(ellipsis[[i]]);
            if(!is.null(colnames(ellipsis[[i]]))){
                variablesNames[m+(1:nVariablesLoop)] <- colnames(ellipsis[[i]]);
            }
            else{
                variablesNames[m+(1:nVariablesLoop)] <- paste0("x",m+c(1:nVariablesLoop));
            }
            dataCreated[1:nrow(ellipsis[[i]]),m+(1:nVariablesLoop)] <- ellipsis[[i]];
        }
        else if(dataType[i]=="vector"){
            nVariablesLoop <- 1;
            if(is.null(names(ellipsis)[i])){
                variablesNames[m+1] <- paste0("x",m+1);
            }
            else{
                variablesNames[m+1] <- names(ellipsis)[i];
            }
            dataCreated[1:length(ellipsis[[i]]),m+1] <- ellipsis[[i]];
        }
        else{
            nVariablesLoop <- nvariate(ellipsis[[i]]);
            if(nVariablesLoop>1){
                variablesNames[m+(1:nVariablesLoop)] <- colnames(actuals(ellipsis[[i]]));
                variablesNames[m+(1:nVariablesLoop)] <- paste0(variablesNames[m+(1:nVariablesLoop)],"Fit");
            }
            else{
                if(!is.null(names(ellipsis)[i])){
                    variablesNames[m+1] <- names(ellipsis)[i];
                }
                else{
                    variablesNames[m+1] <- as.character(formula(ellipsis[[i]])[[2]]);
                    variablesNames[m+(1:nVariablesLoop)] <- paste0(variablesNames[m+(1:nVariablesLoop)],"Fit");
                }
            }
            dataCreated[1:nobs(ellipsis[[i]]),m+(1:nVariablesLoop)] <- fitted(ellipsis[[i]]);
        }
        m <- m + nVariablesLoop;
    }
    colnames(dataCreated) <- variablesNames;

    return(structure(list(data=dataCreated, objects=ellipsis, dataType=dataType),
                     class="folder"));
}

#' Stochastic Regressors Model
#'
#' This function is a wrapper for \link[greybox]{alm}, which extracts the design matrix from
#' the \link[greybox]{folder} object and calls for \code{alm()} to estimate the model on it.
#'
#' The function accepts all the parameters of \link[greybox]{alm}, so have a look at the
#' documentation of that function.
#'
#' @template author
#' @template keywords
#'
#' @param formula an object of class "formula" (or one that can be coerced to
#' that class): a symbolic description of the model to be fitted. Can also include
#' \code{trend}, which would add the global trend.
#' @param folder the \link[greybox]{folder} object.
#' @param subset an optional vector specifying a subset of observations to be
#' used in the fitting process.
#' @param na.action	a function which indicates what should happen when the
#' data contain NAs. The default is set by the na.action setting of
#' \link[base]{options}, and is \link[stats]{na.fail} if that is unset. The
#' factory-fresh default is \link[stats]{na.omit}. Another possible value
#' is NULL, no action. Value \link[stats]{na.exclude} can be useful.
#' @param ... All the other parameters passed to \link[greybox]{alm}.
#'
#' @return Function returns exactly the same set of variables as
#' \link[greybox]{alm} but with addition of \code{objects} list, containing
#' all the objects used in the original \link[greybox]{folder} call.
#'
#' @seealso \code{\link[greybox]{stepwise}, \link[greybox]{lmCombine},
#' \link[greybox]{alm}}.
#'
#' @examples
#' xreg <- cbind(rnorm(100,10+0.5*c(1:100),3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rnorm(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#'
#' # Estimate the model for the variable x1
#' modelForX1 <- alm(x1~trend, xreg)
#'
#' foldedData <- folder(xreg, z=modelForX1)
#'
#' srModel <- srm(y~x2+z, foldedData)
#' summary(srModel)
#'
#' @export
srm <- function(formula, folder, subset=NULL, na.action=NULL, ...){
    # Function is a wrapper for alm(), estimating the model on the provided folder
    clOriginal <- cl <- match.call();

    if(is.folder(folder)){
        cl[[1]] <- as.name("alm");
        cl$folder <- NULL;
        cl$data <- substitute(folder$data);
        ourReturn <- eval(cl);
        ourReturn$call <- clOriginal;
        ourReturn$objects <- folder$objects;
        return(structure(ourReturn,class=c("srm","alm","greybox")));
    }
    # If this is not folder object, then return basic alm estimate
    else{
        cl[[1]] <- as.name("alm");
        cl$folder <- NULL;
        cl$data <- substitute(folder);
        return(eval(cl));
    }
}

#' @export
print.folder <- function(x, ...){
    cat("Folder object\n");
    print(x$data);
}

#' @importFrom utils head tail
#' @export
head.folder <- function(x, ...){
    return(head(x$data, ...));
}

#' @export
tail.folder <- function(x, ...){
    return(tail(x$data, ...));
}

#' @export
vcov.folder <- function(object, ...){
    # Function returns covariance matrix fo folder object
    ourData <- object$data;
    dataType <- object$dataType;
    obsInsample <- nrow(ourData[apply(!is.na(ourData),1,all),]);

    m <- 0;
    for(i in length(dataType)){
        if(any(dataType[i]==c("data.frame","matrix"))){
            nVariablesLoop <- ncol(object$objects[[i]]);
        }
        else if(dataType[i]=="vector"){
            nVariablesLoop <- 1;
        }
        else{
            nVariablesLoop <- nvariate(object$objects[[i]]);
            ourData[1:nobs(object$objects[[i]]),m+(1:nVariablesLoop)] <- residuals(object$objects[[i]]);
        }
        m[] <- m + nVariablesLoop;
    }

    ourData <- model.matrix(~.-1,data=ourData)
    ourData[] <- ourData - matrix(apply(ourData,2,mean), nrow(ourData), ncol(ourData), byrow=TRUE);

    return(crossprod(ourData)/obsInsample);
}


# #' @export
# predict.srm <- forecast(object, newdata=NULL, h=NULL, ...){
#     # Function produces point forecasts and prediction intervals for SRM
#
# }


