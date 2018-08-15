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
    ourData <- data;
    ourData <- ourData[apply(!is.na(ourData),1,all),]
    obs <- nrow(ourData)
    if(is.null(df)){
        df <- 0;
    }
    if(!is.data.frame(ourData)){
        ourData <- as.data.frame(ourData);
    }
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

    distribution <- distribution[1];
    if(distribution=="dnorm"){
        lmCall <- function(formula, data){
            model <- .lm.fit(as.matrix(cbind(1,data[,as.character(all.vars(formula[[3]]))])),
                             as.matrix(data[,as.character(formula[[2]])]));
            return(structure(model,class="lm"));
        }
        listToCall <- vector("list");
    }
    else{
        lmCall <- alm;
        listToCall <- list(distribution=distribution);
    }

    method <- method[1];

    bestICNotFound <- TRUE;
    allICs <- list(NA);
    # Run the simplest model y = const
    testFormula <- paste0(colnames(ourData)[1],"~ 1");
    listToCall$formula <- as.formula(testFormula);
    listToCall$data <- ourData;
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
    ourData <- cbind(ourData,residuals(testModel));
    colnames(ourData)[ncol(ourData)] <- "resid";
    nCols <- ncol(ourData);

    bestFormula <- testFormula;
    if(!silent){
        cat(testFormula); cat(", "); cat(currentIC); cat("\n\n");
    }

    m <- 2;
    # Start the loop
    while(bestICNotFound){
        ourCorrelation <- cor(ourData[,nCols],ourData,use="complete.obs",method=method)[-c(1,nCols)];
        newElement <- names(ourData)[which(abs(ourCorrelation)==max(abs(ourCorrelation)))[1] + 1];
        # If the newElement is the same as before, stop
        if(any(newElement==all.vars(as.formula(bestFormula)))){
            bestICNotFound <- FALSE;
            break;
        }
        # Include the new element in the original model
        testFormula <- paste0(testFormula,"+",newElement);
        listToCall$formula <- as.formula(testFormula);
        listToCall$data <- ourData;
        testModel <- do.call(lmCall,listToCall);
        # Modify logLik
        logLikValue <- logLik(testModel);
        attributes(logLikValue)$df <- nParam(logLikValue) + df;
        if(attributes(logLikValue)$df >= (obs+1)){
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
            ourData[,ncol(ourData)] <- residuals(testModel);
        }
        names(currentIC) <- newElement;
        allICs[[m]] <- currentIC;
        m <- m+1;
    }

    # Create an object of the same name as the original data
    # If it was a call on its own, make it one string
    assign(paste0(deparse(substitute(data)),collapse=""),as.data.frame(data));
    # Remove "1+" from the best formula
    bestFormula <- sub(" 1+", "", bestFormula,fixed=T);

    # listToCall$formula <- as.formula(bestFormula);
    # listToCall$data <- substitute(data);
    # bestModel <- do.call(lmCall,listToCall);

    if(distribution=="dnorm"){
        bestModel <- do.call("lm", list(formula=as.formula(bestFormula),
                                         data=substitute(data)));
        bestModel$distribution <- distribution;
        bestModel$logLik <- logLik(bestModel);
        bestModel$actuals <- ourData[,1];
    }
    else{
        bestModel <- do.call("alm", list(formula=as.formula(bestFormula),
                                         data=substitute(data),
                                         distribution=distribution));
    }

    bestModel$ICs <- unlist(allICs);

    return(structure(bestModel,class=c("alm","greybox")));
}
