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
#' Some details and examples of application are also given in the vignette
#' "Greybox": \code{vignette("greybox","greybox")}
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
#' @param occurrence what distribution to use for occurrence part. See
#' \link[greybox]{alm} for details.
#' @param ... This is temporary and is needed in order to capture "silent"
#' parameter if it is provided.
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
#' ### Mixture distribution of Log Normal and Cumulative Logit
#' xreg[,1] <- xreg[,1] * round(exp(xreg[,1]-70) / (1 + exp(xreg[,1]-70)),0)
#' colnames(xreg) <- c("y","x1","x2","Noise")
#' ourModel <- stepwise(xreg, distribution="dlnorm",
#'                      occurrence=stepwise(xreg, distribution="plogis"))
#' summary(ourModel)
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
                                    "plogis","pnorm"),
                     occurrence=c("none","plogis","pnorm"), ...){
##### Function that selects variables based on IC and using partial correlations
    if(is.null(df)){
        df <- 0;
    }

    # Check, whether the response is numeric
    if(is.data.frame(data)){
        if(!is.numeric(data[[1]])){
            warning(paste0("The response variable (first column of the data) is not numeric! ",
                           "We will make it numeric, but we cannot promise anything."),
                    call.=FALSE);
            data[[1]] <- as.numeric(data[[1]]);
        }
    }

    distribution <- distribution[1];
    if(distribution=="dnorm"){
        useALM <- FALSE;
    }
    else{
        useALM <- TRUE;
        if(any(distribution==c("plogis","pnorm"))){
            data[,1] <- (data[,1]!=0)*1;
        }
    }

    # Check the data for NAs
    if(any(is.na(data))){
        rowsSelected <- apply(!is.na(data),1,all);
    }
    else{
        rowsSelected <- rep(TRUE,nrow(data));
    }

    # Check occurrence. If it is not "none" then use alm().
    if(is.alm(occurrence)){
        useALM <- TRUE;
        rowsSelected <- rowsSelected & (data[,1]!=0);
    }
    else{
        occurrence <- occurrence[1];
        if(all(occurrence!=c("none","plogis","pnorm"))){
            warning(paste0("Sorry, but we don't know what to do with the occurrence '",occurrence,
                        "'. Switching to 'none'."), call.=FALSE);
            occurrence <- "none";
        }

        if(any(occurrence==c("plogis","pnorm"))){
            useALM <- TRUE;
            rowsSelected <- rowsSelected & (data[,1]!=0);
        }
    }

    # Define what function to use in the estimation
    if(useALM){
        lmCall <- alm;
        listToCall <- list(distribution=distribution, fast=TRUE);
    }
    else{
        if(!is.data.frame(data)){
            lmCall <- function(formula, data){
                model <- .lm.fit(as.matrix(cbind(1,data[,all.vars(formula)[-1]])),
                                 as.matrix(data[,all.vars(formula)[1]]));
                colnames(model$qr) <- c("(Intercept)",all.vars(formula)[-1]);
                return(structure(model,class="lm"));
            }
        }
        else{
            lmCall <- function(formula, data){
                model <- .lm.fit(model.matrix(formula, data=data),
                                 as.matrix(data[,all.vars(formula)[1]]));
                return(structure(model,class="lm"));
            }
        }
        listToCall <- vector("list");
    }

    # Names of the variables
    variablesNames <- colnames(data);

    # The number of explanatory variables and the number of observations
    nVariables <- ncol(data)-1;
    obsInsample <- sum(rowsSelected);

    ## Check the variability in the data. If it is none, remove variables
    noVariability <- vector("logical",nVariables+1);
    # First column is the response variable, so we assume that it has variability
    if(is.data.frame(data)){
        noVariability[1] <- FALSE;
        for(i in 1:nVariables){
            noVariability[i+1] <- length(unique(data[[i]]))<=1;
        }
        # noVariability[] <- c(FALSE,sapply(apply(data[,-1],2,unique),length)<=1);
    }
    else{
        noVariability[1] <- FALSE;
        for(i in 1:nVariables){
            noVariability[i+1] <- length(unique(data[,i]))<=1;
        }
        # noVariability[] <- c(FALSE,sapply(as.data.frame(apply(data[,-1],2,unique)),length)<=1);
    }
    if(any(noVariability)){
        if(all(noVariability)){
            stop("None of exogenous variables has variability. There's nothing to select!",
                    call.=FALSE);
        }
        else{
            warning("Some exogenous variables did not have any variability. We dropped them out.",
                    call.=FALSE);
            nVariables <- sum(!noVariability)-1;
            variablesNames <- variablesNames[!noVariability];
        }
    }

    # Create data frame to work with
    listToCall$data <- as.data.frame(data[rowsSelected,variablesNames]);
    errors <- matrix(0,obsInsample,1);

    # Create substitute and remove the original data
    dataSubstitute <- substitute(data);
    # Remove the data and clean after yourself
    rm(data);
    gc(verbose=FALSE);

    # Record the names of the response and the explanatory variables
    responseName <- variablesNames[1];
    variablesNames <- variablesNames[-1];

    # Define, which of the variables are factors, excluding the response variable
    numericData <- sapply(listToCall$data, is.numeric)[-1]
    # If the value is binary, treat it as a factor # & apply(listToCall$data!=0 & listToCall$data!=1,2,any)[-1];

    #### The function-analogue of mcor, but without checks ####
    mcorFast <- function(x){
        x <- model.matrix(~x);
        lmFit <- .lm.fit(x,errors);
        # abs() is needed for technical purposes - for some reason sometimes this stuff becomes
        # very small negative (e.g. -1e-16).
        return(sqrt(abs(1 - sum(residuals(lmFit)^2) / sum((errors-mean(errors))^2))));
    }

    assocValues <- vector("numeric",nVariables);
    names(assocValues) <- variablesNames;
    #### The function that works similar to association(), but faster ####
    assocFast <- function(){
        # Measures of association with numeric data
        assocValues[which(numericData)] <- suppressWarnings(cor(errors,listToCall$data[,which(numericData)+1],
                                                                use="complete.obs",method=method));

        # Measures of association with categorical data
        for(i in which(!numericData)+1){
            assocValues[i-1] <- suppressWarnings(mcorFast(listToCall$data[[i]]));
        }
        return(assocValues);
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

    method <- method[1];

    bestICNotFound <- TRUE;
    allICs <- list(NA);
    # Run the simplest model y = const
    testFormula <- paste0("`",responseName,"`~ 1");
    listToCall$formula <- as.formula(testFormula);
    testModel <- do.call(lmCall,listToCall);
    # Write down the logLik and take df into account
    logLikValue <- logLik(testModel);
    attributes(logLikValue)$df <- nparam(logLikValue) + df;
    # Write down the IC. This one needs to be calculated from the logLik
    # in order to take the additional df into account.
    currentIC <- bestIC <- IC(logLikValue);
    names(currentIC) <- "Intercept";
    allICs[[1]] <- currentIC;
    # Add residuals to the ourData
    errors[] <- residuals(testModel);

    bestFormula <- testFormula;
    if(!silent){
        cat("Formula: "); cat(testFormula);
        cat(", IC: "); cat(currentIC); cat("\n\n");
    }

    m <- 2;
    # Start the loop
    while(bestICNotFound){
        ourCorrelation <- assocFast();

        newElement <- variablesNames[which(abs(ourCorrelation)==max(abs(ourCorrelation),na.rm=TRUE))[1]];
        # If the newElement is the same as before, stop
        if(any(newElement==all.vars(as.formula(bestFormula)))){
            bestICNotFound <- FALSE;
            break;
        }
        # Include the new element in the original model
        testFormula <- paste0(testFormula,"+`",newElement,"`");
        listToCall$formula <- as.formula(testFormula);
        testModel <- do.call(lmCall,listToCall);
        # Modify logLik
        logLikValue <- logLik(testModel);
        attributes(logLikValue)$df <- nparam(logLikValue) + df;
        if(attributes(logLikValue)$df >= (obsInsample+1)){
            if(!silent){
                warning("Number of degrees of freedom is greater than number of observations. Cannot proceed.");
            }
            bestICNotFound <- FALSE;
            break;
        }

        # Calculate the IC
        currentIC <- IC(logLikValue);
        if(!silent){
            cat(paste0("Step ",m-1,". "));
            cat("Formula: "); cat(testFormula);
            cat(", IC: "); cat(currentIC);
            cat("\nCorrelations: \n"); print(round(ourCorrelation,3)); cat("\n");
        }
        # If IC is greater than the previous, then the previous model is the best
        if(currentIC >= bestIC){
            bestICNotFound <- FALSE;
        }
        else{
            bestIC <- currentIC;
            bestFormula <- testFormula;
            errors[] <- residuals(testModel);
        }
        names(currentIC) <- newElement;
        allICs[[m]] <- currentIC;
        m <- m+1;
    }

    # Remove "1+" from the best formula
    bestFormula <- sub(" 1+", "", bestFormula,fixed=TRUE);
    bestFormula <- as.formula(bestFormula);

    # If this is a big data just wrap up the stuff using lmCall
    if(distribution=="dnorm"){
        varsNames <- all.vars(bestFormula)[-1];

        listToCall$formula <- bestFormula;

        bestModel <- do.call(lmCall,listToCall);
        # Expand the data from the final model
        bestModel$data <- cbind(listToCall$data[[1]],model.matrix(bestFormula,listToCall$data)[,-1]);
        colnames(bestModel$data) <- c(responseName,colnames(bestModel$qr)[-1]);
        rm(listToCall);

        bestModel$distribution <- distribution;
        bestModel$logLik <- logLik(bestModel);
        bestModel$mu <- bestModel$fitted <- bestModel$data[,1] - c(bestModel$residuals);
        # This is number of variables + constant + variance
        bestModel$df <- length(varsNames) + 1 + 1;
        bestModel$df.residual <- obsInsample - bestModel$df;
        names(bestModel$coefficients) <- colnames(bestModel$qr);
        # Remove redundant bits
        bestModel$effects <- NULL;
        bestModel$qr <- NULL;
        bestModel$qraux <- NULL;
        bestModel$pivot <- NULL;
        bestModel$tol <- NULL;
        # Form the pseudocall to alm
        bestModel$call <- quote(alm(formula=bestFormula, data=data, distribution="dnorm"));
        bestModel$call$formula <- bestFormula;
        bestModel$subset <- rep(TRUE, obsInsample);
        bestModel$scale <- sqrt(sum(bestModel$residuals^2) / obsInsample);
        class(bestModel) <- c("alm","greybox");
    }
    else{
        bestModel <- do.call("alm", list(formula=bestFormula,
                                         data=dataSubstitute,
                                         distribution=distribution,
                                         occurrence=occurrence,
                                         fast=TRUE),
                             envir = parent.frame());
        bestModel$call$occurrence <- substitute(occurrence);
        class(bestModel) <- c("alm","greybox");
    }

    bestModel$ICs <- unlist(allICs);

    return(bestModel);
}
