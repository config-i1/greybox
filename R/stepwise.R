#' Stepwise selection of regressors
#'
#' Function selects variables that give linear regression with the lowest
#' information criteria. The selection is done stepwise (forward) based on
#' partial correlations. This should be a simpler and faster implementation
#' than step() function from `stats' package.
#'
#' The algorithm uses alm() to fit different models and cor() to select the next
#' regressor in the sequence.
#'
#' @template AICRef
#' @template author
#' @template keywords
#'
#' @param data Data frame containing dependant variable in the first column and
#' the others in the rest.
#' @param ic Information criterion to use.
#' @param silent If \code{silent=FALSE}, then nothing is silent, everything is
#' printed out. \code{silent=TRUE} means that nothing is produced.
#' @param df Number of degrees of freedom to add (should be used if stepwise is
#' used on residuals).
#' @param method Method of correlations calculation. The default is Kendall's
#' Tau, which should be applicable to a wide range of data in different scales.
#' @param distribution Distribution to pass to \code{alm()}.
#'
#' @return Function returns \code{model} - the final model of the class "alm".
#' See \link[greybox]{alm} for details of the output.
#'
#' @seealso \code{\link[stats]{step}, \link[greybox]{xregExpander},
#' \link[greybox]{lmCombine}}
#'
#' @examples
#'
#' ### Simple example
#' xreg <- cbind(rnorm(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rnorm(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#' stepwise(xreg)
#'
#' ### Fat regression example
#' xreg <- matrix(rnorm(20000,10,3),100,200)
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rnorm(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y",paste0("x",c(1:200)),"Noise")
#' ourModel <- stepwise(xreg,ic="AICc")
#' plot(ourModel$ICs,type="l",ylim=range(min(ourModel$ICs),max(ourModel$ICs)+5))
#' points(ourModel$ICs)
#' text(c(1:length(ourModel$ICs))+0.1,ourModel$ICs+5,names(ourModel$ICs))
#'
#' @importFrom stats .lm.fit
#' @export stepwise
stepwise <- function(data, ic=c("AICc","AIC","BIC","BICc"), silent=TRUE, df=NULL,
                     method=c("pearson","kendall","spearman"),
                     distribution=c("dnorm","dfnorm","dlnorm","dlaplace","ds","dchisq","dlogis",
                                    "plogis","pnorm")){
##### Function that selects variables based on IC and using partial correlations
    if(is.null(df)){
        df <- 0;
    }

    distribution <- distribution[1];
    if(distribution=="dnorm"){
        lmCall <- function(formula, data){
            model <- .lm.fit(as.matrix(cbind(1,data[,all.vars(formula)[-1]])),
                             as.matrix(data[,all.vars(formula)[1]]));
            return(structure(model,class="lm"));
        }
        listToCall <- vector("list");
    }
    else{
        lmCall <- alm;
        listToCall <- list(distribution=distribution);
    }

    nCols <- ncol(data)+1;
    if(any(is.na(data))){
        nonNARows <- apply(!is.na(data),1,all);
    }
    else{
        nonNARows <- rep(TRUE,nrow(data));
    }
    nRows <- sum(nonNARows);

    # Names of the variables
    ourDataNames <- colnames(data);

    # Create data frame to work with
    listToCall$data <- as.data.frame(data);
    listToCall$data$resid <- 0;

    # Create substitute and remove the original data
    dataSubstitute <- substitute(data);
    rm(data)

    responseName <- ourDataNames[1];
    ourDataNames <- ourDataNames[-1];
#
#     if(any(sapply(data, is.factor))){
#         if(any(sapply(data, is.ordered))){
#             association <- Kendall
#         }
#         else{
#             association <- Cramer's V
#         }
#     }
#     else{
#         association <- cor;
#     }
#

    # Select IC
    ic <- ic[1];
    if(ic=="AIC"){
        IC <- AIC;
    }
    else if(ic=="AICc"){
        IC <- AICc;
    }
    else if(ic=="BIC"){
        IC <- BIC;
    }
    else if(ic=="BICc"){
        IC <- BICc;
    }

    method <- method[1];

    bestICNotFound <- TRUE;
    allICs <- list(NA);
    # Run the simplest model y = const
    testFormula <- paste0(responseName,"~ 1");
    listToCall$formula <- as.formula(testFormula);
    testModel <- do.call(lmCall,listToCall);
    # Write down the logLik and take df into account
    logLikValue <- logLik(testModel);
    attributes(logLikValue)$df <- nParam(logLikValue) + df;
    # Write down the IC. This one needs to be calculated from the logLik
    # in order to take the additional df into account.
    currentIC <- bestIC <- IC(logLikValue);
    names(currentIC) <- "Intercept";
    allICs[[1]] <- currentIC;
    # Add residuals to the ourData
    listToCall$data$resid <- residuals(testModel);

    bestFormula <- testFormula;
    if(!silent){
        cat(testFormula); cat(", "); cat(currentIC); cat("\n\n");
    }

    m <- 2;
    # Start the loop
    while(bestICNotFound){
        ourCorrelation <- cor(listToCall$data$resid,listToCall$data,use="complete.obs",method=method)[-c(1,nCols)];
        newElement <- ourDataNames[which(abs(ourCorrelation)==max(abs(ourCorrelation)))[1]];
        # If the newElement is the same as before, stop
        if(any(newElement==all.vars(as.formula(bestFormula)))){
            bestICNotFound <- FALSE;
            break;
        }
        # Include the new element in the original model
        testFormula <- paste0(testFormula,"+",newElement);
        listToCall$formula <- as.formula(testFormula);
        testModel <- do.call(lmCall,listToCall);
        # Modify logLik
        logLikValue <- logLik(testModel);
        attributes(logLikValue)$df <- nParam(logLikValue) + df;
        if(attributes(logLikValue)$df >= (nRows+1)){
            if(!silent){
                warning("Number of degrees of freedom is greater than number of observations. Cannot proceed.");
            }
            bestICNotFound <- FALSE;
            break;
        }

        # Calculate the IC
        currentIC <- IC(logLikValue);
        if(!silent){
            cat(testFormula); cat(", "); cat(currentIC); cat("\n");
            cat(round(ourCorrelation,3)); cat("\n\n");
        }
        # If IC is greater than the previous, then the previous model is the best
        if(currentIC >= bestIC){
            bestICNotFound <- FALSE;
        }
        else{
            bestIC <- currentIC;
            bestFormula <- testFormula;
            listToCall$data$resid <- residuals(testModel);
        }
        names(currentIC) <- newElement;
        allICs[[m]] <- currentIC;
        m <- m+1;
    }

    # Remove "1+" from the best formula
    bestFormula <- sub(" 1+", "", bestFormula,fixed=T);
    bestFormula <- as.formula(bestFormula);

    # If this is a big data just wrap up the stuff using lmCall
    if(distribution=="dnorm"){
        varsNames <- all.vars(bestFormula)[-1];

        listToCall$formula <- bestFormula;

        bestModel <- do.call(lmCall,listToCall);
        bestModel$data <- listToCall$data[,all.vars(bestFormula)];
        rm(listToCall);

        bestModel$distribution <- distribution;
        bestModel$logLik <- logLik(bestModel);
        bestModel$fitted.values <- bestModel$data[,1] - c(bestModel$residuals);
        bestModel$df <- length(varsNames) + 1;
        bestModel$df.residual <- nRows - bestModel$df;
        names(bestModel$coefficients) <- c("(Intercept)",varsNames);
        # Remove redundant bits
        bestModel$effects <- NULL;
        bestModel$qr <- NULL;
        bestModel$qraux <- NULL;
        bestModel$pivot <- NULL;
        bestModel$tol <- NULL;
        # Form the pseudocall to alm
        bestModel$call <- quote(alm(formula=bestFormula, data=data, distribution="dnorm"));
        bestModel$call$formula <- bestFormula;
        class(bestModel) <- c("alm","greybox");
    }
    else{
        bestModel <- do.call("alm", list(formula=bestFormula,
                                         data=dataSubstitute,
                                         distribution=distribution));
        class(bestModel) <- c("alm","greybox");
    }

    bestModel$ICs <- unlist(allICs);

    return(bestModel);
}
