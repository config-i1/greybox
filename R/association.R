#' Measures of association
#'
#' Function returns the matrix of measures of association for different types of
#' variables.
#'
#' The function looks at the types of the variables and calculates different
#' measures depending on the result:
#' \itemize{
#' \item If both variables are numeric, then Pearson's correlation is calculated;
#' \item If both variables are categorical, then Cramer's V is calculated;
#' \item Finally, if one of the variables is categorical, and the other is numeric,
#' then multiple correlation is returned.
#' }
#' After that the measures are wrapped up in a matrix.
#'
#' Function also calculates the p-values associated with the respective measures
#' (see the return).
#'
#' See details in the vignette "Marketing analytics with greybox":
#' \code{vignette("maUsingGreybox","greybox")}
#'
#' \code{assoc()} is just a short name for the \code{association{}}.
#'
#' @template author
#' @keywords htest
#'
#' @param x Either data.frame or a matrix
#' @param y The numerical variable.
#' @param use What observations to use. See \link[stats]{cor} function for details.
#' The only option that is not available here is \code{"pairwise.complete.obs"}.
#'
#' @return The following list of values is returned:
#' \itemize{
#' \item{value - Matrix of the coefficients of association;}
#' \item{p.value - The p-values for the parameters;}
#' \item{type - The matrix of the types of measures of association.}
#' }
#'
#' @seealso \code{\link[base]{table}, \link[greybox]{tableplot}, \link[greybox]{spread},
#' \link[greybox]{cramer}, \link[greybox]{mcor}}
#'
#' @examples
#'
#' association(mtcars)
#'
#' @aliases assoc
#' @rdname association
#' @export association
association <- function(x, y=NULL, use=c("na.or.complete","complete.obs","everything","all.obs")){
    # Function returns the measures of association between the variables based on their type

    use <- substr(use[1],1,1);

    if(is.matrix(x) | is.data.frame(x)){
        nVariablesX <- ncol(x);
        namesX <- colnames(x);
    }
    else if(is.factor(x)){
        nVariablesX <- 1;
        namesX <- deparse(substitute(x));
        x <- as.data.frame(x);
    }
    else{
        nVariablesX <- 1;
        namesX <- deparse(substitute(x));
        x <- as.matrix(x);
    }

    if(!is.null(y)){
        if(is.matrix(y) | is.data.frame(y)){
            nVariablesY <- ncol(y);
            namesY <- colnames(y);
        }
        else if(is.factor(y)){
            nVariablesY <- 1;
            namesY <- deparse(substitute(y));
            y <- as.data.frame(y);
        }
        else{
            nVariablesY <- 1;
            namesY <- deparse(substitute(y));
            y <- as.matrix(y);
        }

        data <- cbind(x,y);
        nVariables <- nVariablesX+nVariablesY;
        namesData <- c(namesX, namesY);
    }
    else{
        if(nVariablesX>1){
            data <- x;
            nVariables <- nVariablesX;
            namesData <- namesX;
        }
        else{
            return(structure(list(value=1, p.value=1, type="none"),class="association"));
        }
    }

    matrixAssociation <- matrix(1,nVariables,nVariables, dimnames=list(namesData,namesData));
    matrixPValues <- matrix(0,nVariables,nVariables, dimnames=list(namesData,namesData));
    matrixTypes <- matrix("none",nVariables,nVariables, dimnames=list(namesData,namesData));

    numericDataX <- vector(mode="logical", length=nVariablesX);
    for(i in 1:nVariablesX){
        numericDataX[i] <- is.numeric(x[,i]);
        if(numericDataX[i]){
            if(length(unique(x[,i]))<=10){
                numericDataX[i] <- FALSE;
            }
        }
    }

    if(!is.null(y)){
        numericDataY <- vector(mode="logical", length=nVariablesY);
        for(i in 1:nVariablesY){
            numericDataY[i] <- is.numeric(y[,i]);
            if(numericDataY[i]){
                if(length(unique(y[,i]))<=10){
                    numericDataY[i] <- FALSE;
                }
            }
        }
        numericData <- c(numericDataX, numericDataY);
    }
    else{
        numericData <- numericDataX;
    }

    if(any(is.na(data))){
        if((use=="c" & nrow(data[!apply(is.na(data),1,any),])<2) | use=="a"){
            variablesNA <- apply(is.na(data),2,any);
            stop(paste0("Missing observations in the variables: ",paste0(namesData[variablesNA],collapse=", ")), call.=FALSE);
        }
    }

    for(i in 1:nVariables){
        for(j in 1:nVariables){
            if(i>=j){
                next;
            }

            if(numericData[i] & numericData[j]){
                matrixAssociation[i,j] <- cor(data[,i],data[,j],use=use);
                matrixPValues[i,j] <- cor.test(data[,i],data[,j])$p.value;
                matrixTypes[i,j] <- "cor";
            }
            else if(!numericData[i] & !numericData[j]){
                cramerOutput <- cramer(data[,i],data[,j],use=use);
                matrixAssociation[i,j] <- cramerOutput$value;
                matrixPValues[i,j] <- cramerOutput$p.value;
                matrixTypes[i,j] <- "cramer";
            }
            else if(!numericData[i] & numericData[j]){
                mcorOutput <- mcor(data[,i],data[,j],use=use);
                matrixAssociation[i,j] <- mcorOutput$value;
                matrixPValues[i,j] <- mcorOutput$p.value;
                matrixTypes[i,j] <- "mcor";
            }
            else{
                mcorOutput <- mcor(data[,j],data[,i],use=use);
                matrixAssociation[i,j] <- mcorOutput$value;
                matrixPValues[i,j] <- mcorOutput$p.value;
                matrixTypes[i,j] <- "mcor";
            }
        }
    }

    matrixAssociation[lower.tri(matrixAssociation)] <- t(matrixAssociation)[lower.tri(matrixAssociation)];
    matrixPValues[lower.tri(matrixPValues)] <- t(matrixPValues)[lower.tri(matrixPValues)];
    matrixTypes[lower.tri(matrixTypes)] <- t(matrixTypes)[lower.tri(matrixTypes)];

    return(structure(list(value=matrixAssociation, p.value=matrixPValues, type=matrixTypes),
                     class="association"));
}

#' @export assoc
#' @rdname association
assoc <- association;
