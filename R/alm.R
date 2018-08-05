#' Advanced Linear Model
#'
#' Function estimates model based on the selected distribution
#'
#' This is a function, similar to \link[stats]{lm}, but for the cases of several
#' non-normal distributions. These include:
#' \enumerate{
#' \item Laplace distribution, \link[greybox]{dlaplace},
#' \item S-distribution, \link[greybox]{ds},
#' \item Folded-normal distribution, \link[greybox]{dfnorm},
#' \item Chi-Squared Distribution, \link[stats]{dchisq}.
#' }
#'
#' Probably some other distributions will be added to this function at some point...
#'
#' The estimation is done using likelihood of respective distributions.
#'
#' @template author
#' @template keywords
#'
#' @param formula an object of class "formula" (or one that can be coerced to
#' that class): a symbolic description of the model to be fitted.
#' @param data a data frame or a matrix, containing the variables in the model.
#' @param subset an optional vector specifying a subset of observations to be
#' used in the fitting process.
#' @param na.action	a function which indicates what should happen when the
#' data contain NAs. The default is set by the na.action setting of
#' \link[base]{options}, and is \link[stats]{na.fail} if that is unset. The
#' factory-fresh default is \link[stats]{na.omit}. Another possible value
#' is NULL, no action. Value \link[stats]{na.exclude} can be useful.
#' @param distribution What density function to use in the process.
#'
#' @return Function returns \code{model} - the final model of the class
#' "alm".
#'
#' @seealso \code{\link[greybox]{stepwise}, \link[greybox]{lmCombine}}
#'
#' @examples
#'
#' xreg <- cbind(rlaplace(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rlaplace(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#' inSample <- xreg[1:80,]
#' outSample <- xreg[-c(1:80),]
#' # Combine all the possible models
#' ourModel <- alm(y~x1+x2, inSample, distribution="laplace")
#' summary(ourModel)
#' plot(forecast(ourModel,outSample))
#'
#' @importFrom numDeriv hessian
#' @importFrom nloptr nloptr
#' @importFrom stats model.frame sd terms dchisq
#' @export alm
alm <- function(formula, data, subset=NULL,  na.action,
                distribution=c("laplace","s","fnorm","chisq")){

    cl <- match.call();

    distribution <- distribution[1];
    if(!is.data.frame(data)){
        data <- as.data.frame(data);
    }
    dataWork <- model.frame(formula, data, na.action=na.action);
    ourTerms <- terms(dataWork);
    obsInsample <- nrow(dataWork);
    variablesNames <- colnames(dataWork)[-1];

    matrixXreg <- as.matrix(dataWork[,-1]);
    nVariables <- length(variablesNames);
    colnames(matrixXreg) <- variablesNames;

    #### Checks of the exogenous variables ####
    # Remove the data for which sd=0
    noVariability <- apply(matrixXreg,2,sd)==0;
    if(any(noVariability)){
        if(all(noVariability)){
            warning("None of exogenous variables has variability. Fitting the straight line.",
                    call.=FALSE);
            matrixXreg <- matrix(1,obsInsample,1);
            nVariables <- 1;
            variablesNames <- "(Intercept)";
        }
        else{
            warning("Some exogenous variables did not have any variability. We dropped them out.",
                    call.=FALSE);
            matrixXreg <- matrixXreg[,!noVariability];
            nVariables <- ncol(matrixXreg);
            variablesNames <- variablesNames[!noVariability];
        }
    }

    if(nVariables>1){
        # Check perfectly correlated cases
        corThreshold <- 0.999;
        corMatrix <- cor(matrixXreg);
        corHigh <- upper.tri(corMatrix) & abs(corMatrix)>=corThreshold;
        if(any(corHigh)){
            removexreg <- unique(which(corHigh,arr.ind=TRUE)[,1]);
            if(ncol(matrixXreg)-length(removexreg)>1){
                matrixXreg <- matrixXreg[,-removexreg];
            }
            else{
                matrixXreg <- matrix(matrixXreg[,-removexreg],ncol=ncol(matrixXreg)-length(removexreg),
                                     dimnames=list(rownames(matrixXreg),c(colnames(matrixXreg)[-removexreg])));
            }
            nVariables <- ncol(matrixXreg);
            variablesNames <- colnames(matrixXreg);
            warning("Some exogenous variables were perfectly correlated. We've dropped them out.",
                    call.=FALSE);
        }
    }

    if(nVariables>1){
        # Check dummy variables trap
        detHigh <- determination(matrixXreg)>=corThreshold;
        if(any(detHigh)){
            while(any(detHigh)){
                removexreg <- which(detHigh>=corThreshold)[1];
                if(ncol(matrixXreg)-length(removexreg)>1){
                    matrixXreg <- matrixXreg[,-removexreg];
                }
                else{
                    matrixXreg <- matrix(matrixXreg[,-removexreg],ncol=ncol(matrixXreg)-length(removexreg),
                                         dimnames=list(rownames(matrixXreg),c(colnames(matrixXreg)[-removexreg])));
                }
                nVariables <- ncol(matrixXreg);
                variablesNames <- colnames(matrixXreg);

                detHigh <- determination(matrixXreg)>=corThreshold;
            }
            warning("Some combinations of exogenous variables were perfectly correlated. We've dropped them out.",
                    call.=FALSE);
        }
    }

    if(attr(ourTerms,"intercept")!=0){
        matrixXreg <- cbind(1,matrixXreg);
        variablesNames <- c("(Intercept)",variablesNames);
        nVariables <- length(variablesNames);
        colnames(matrixXreg) <- variablesNames;
    }

    y <- as.matrix(dataWork[,1]);

    fitter <- function(A, distribution, y, matrixXreg){
        # if(distribution=="fnorm"){
        #     scale <- A[length(A)];
        #     A <- A[-length(A)];
        # }

        yFitted <- matrixXreg %*% A;
        errors <- y - yFitted;

        if(distribution=="laplace"){
            scale <- mean(abs(y-yFitted));
        }
        else if(distribution=="s"){
            scale <- mean(sqrt(abs(y-yFitted))) / 2;
        }
        else if(distribution=="fnorm"){
            scale <- mean((y-yFitted)^2);
        }
        else if(distribution=="chisq"){
            scale <- 2*yFitted;
        }

        return(list(yFitted=yFitted,scale=scale,errors=errors));
    }

    CF <- function(A, distribution, y, matrixXreg){
        fitterReturn <- fitter(A, distribution, y, matrixXreg);

        if(distribution=="laplace"){
            CFReturn <- dlaplace(y, mu=fitterReturn$yFitted, b=fitterReturn$scale, log=TRUE);
        }
        else if(distribution=="s"){
            CFReturn <- ds(y, mu=fitterReturn$yFitted, b=fitterReturn$scale, log=TRUE);
        }
        else if(distribution=="fnorm"){
            CFReturn <- dfnorm(y, mu=fitterReturn$yFitted, sigma=fitterReturn$scale, log=TRUE);
        }
        else if(distribution=="chisq"){
            if(any(fitterReturn$yFitted<=0)){
                CFReturn <- -1E+300;
            }
            else{
                CFReturn <- dchisq(y, fitterReturn$yFitted, log=TRUE);
            }
        }

        CFReturn <- -sum(CFReturn[is.finite(CFReturn)]);

        if(is.nan(CFReturn) | is.na(CFReturn) | is.infinite(CFReturn)){
            CFReturn <- 1E+300;
        }

        return(CFReturn);
    }

    A <- as.vector(solve(t(matrixXreg) %*% matrixXreg, tol=1e-10) %*% t(matrixXreg) %*% y);

#     if(distribution=="fnorm"){
#         A <- c(A,sd(y));
#     }

    res <- nloptr(A, CF,
                  opts=list("algorithm"="NLOPT_LN_SBPLX", xtol_rel=1e-8, maxeval=500),
                  distribution=distribution, y=y, matrixXreg=matrixXreg);

    # res <- nloptr(A, CF, opts=list("algorithm"="NLOPT_LN_BOBYQA", xtol_rel=1e-8),
    #               distribution=distribution, y=y, matrixXreg=matrixXreg);
    # res <- nloptr(res$solution, CF, opts=list("algorithm"="NLOPT_LN_NELDERMEAD", xtol_rel=1e-6),
    #               distribution=distribution, y=y, matrixXreg=matrixXreg);
    A <- res$solution;
    CFValue <- res$objective;

    fitterReturn <- fitter(A, distribution, y, matrixXreg);
    yFitted <- fitterReturn$yFitted;
    errors <- fitterReturn$errors;
    scale <- fitterReturn$scale;

    # Parameters of the model + scale
    df <- nVariables + 1;

    vcovMatrix <- hessian(CF, A, distribution=distribution, y=y, matrixXreg=matrixXreg);

    if(any(is.nan(vcovMatrix))){
        warning(paste0("Something went wrong and we failed to produce the covariance matrix of the parameters.\n",
                       "Obviously, it's not our fault. Probably Russians have hacked your computer...\n",
                       "Try a different distribution maybe?"), call.=FALSE);
        vcovMatrix <- diag(nVariables+(distribution=="fnorm"));
    }
    else{
        if(any(vcovMatrix==0)){
            warning(paste0("Something went wrong and we failed to produce the covariance matrix of the parameters.\n",
                           "Obviously, it's not our fault. Probably Russians have hacked your computer...\n",
                           "Try a different distribution maybe?"), call.=FALSE);
            vcovMatrix <- diag(nVariables+(distribution=="fnorm"));
        }
        else{
            vcovMatrix <- solve(vcovMatrix);
        }

        # Sometimes the diagonal elements in the covariance matrix are negative because likelihood is not fully maximised...
        if(any(diag(vcovMatrix)<0)){
            diag(vcovMatrix) <- abs(diag(vcovMatrix));
        }
    }



    if(distribution=="fnorm"){
        # A <- A[-(nVariables+1)];
        # Correction so that the the expectation of the folded is returned
        mu <- yFitted;
        sigma <- scale;
        yFitted <- sqrt(2/pi)*scale*exp(-yFitted^2/(2*scale^2))+yFitted*(1-2*pnorm(-yFitted/scale));
        errors <- y - yFitted;
        vcovMatrix <- vcovMatrix[1:nVariables,1:nVariables];
    }
    else{
        mu <- yFitted;
        if(distribution=="chisq"){
            scale <- yFitted * 2;
        }
    }
    names(A) <- variablesNames;
    if(nVariables>1){
        dimnames(vcovMatrix) <- list(variablesNames,variablesNames);
    }
    else{
        names(vcovMatrix) <- variablesNames;
    }

    finalModel <- list(coefficients=A, vcov=vcovMatrix, residuals=as.vector(errors), fitted.values=yFitted, actuals=y, mu=mu,
                       df.residual=obsInsample-df, df=df, call=cl, rank=df, model=dataWork, scale=scale,
                       qr=qr(dataWork), terms=ourTerms, logLik=-CFValue, distribution=distribution);
    return(structure(finalModel,class=c("alm","greybox")));
}
