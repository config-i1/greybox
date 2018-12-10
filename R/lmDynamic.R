#' Combine regressions based on point information criteria
#'
#' Function combines parameters of linear regressions of the first variable
#' on all the other provided data using pAIC weights
#'
#' The algorithm uses alm() to fit different models and then combines the models
#' based on the selected point IC. This is a dynamic counterpart of
#' \link[greybox]{lmCombine} function.
#'
#' @template AICRef
#' @template author
#' @template keywords
#'
#' @param data Data frame containing dependent variable in the first column and
#' the others in the rest.
#' @param ic Information criterion to use.
#' @param bruteForce If \code{TRUE}, then all the possible models are generated
#' and combined. Otherwise the best model is found and then models around that
#' one are produced and then combined.
#' @param silent If \code{FALSE}, then nothing is silent, everything is printed
#' out. \code{TRUE} means that nothing is produced.
#' @param distribution Distribution to pass to \code{alm()}.
#'
#' @return Function returns \code{model} - the final model of the class
#' "greyboxD", which includes time varying parameters and dynamic importance
#' of each variable. The list of variables:
#' \itemize{
#' \item coefficients - mean (over time) parameters of the model,
#' \item se - mean standard errors of the parameters of the model,
#' \item actuals - actual values of the response variable,
#' \item fitted.values - the fitted values,
#' \item residuals - residual of the model,
#' \item distribution - distribution used in the estimation,
#' \item logLik - mean (over time) log-likelihood of the model,
#' \item IC - dynamic values of the information criterion (pIC),
#' \item df.residual - mean number of degrees of freedom of the residuals of
#' the model,
#' \item df - mean number of degrees of freedom of the model,
#' \item importance - dynamic importance of the parameters,
#' \item coefficientsDynamic - table with parameters of the model, varying over
#' the time,
#' \item df.residualDynamic - dynamic df.residual,
#' \item dfDynamic - dynamic df,
#' \item call - call used in the function,
#' \item rank - rank of the combined model,
#' \item data - the data used in the model,
#' \item mu - the location value of the distribution.
#' }
#'
#' @seealso \code{\link[greybox]{stepwise}, \link[greybox]{lmCombine}}
#'
#' @examples
#'
#' ### Simple example
#' xreg <- cbind(rnorm(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rnorm(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#' inSample <- xreg[1:80,]
#' outSample <- xreg[-c(1:80),]
#' # Combine all the possible models
#' ourModel <- lmDynamic(inSample,bruteForce=TRUE)
#' predict(ourModel,outSample)
#' plot(predict(ourModel,outSample))
#'
#' @export lmDynamic
lmDynamic <- function(data, ic=c("AICc","AIC","BIC","BICc"), bruteForce=FALSE, silent=TRUE,
                      distribution=c("dnorm","dfnorm","dlnorm","dlaplace","ds","dchisq","dlogis",
                                    "plogis","pnorm")){
    # Function combines linear regression models and produces the combined lm object.
    cl <- match.call();
    cl$formula <- as.formula(paste0(colnames(data)[1]," ~ ."));

    ourData <- data;
    if(!is.data.frame(ourData)){
        ourData <- as.data.frame(ourData);
    }

    if((ncol(ourData)>nrow(ourData)) & bruteForce){
        warning("You have too many variables. We have to be smart here. Switching 'bruteForce=FALSE'.");
        bruteForce <- FALSE;
    }

    ic <- ic[1];
    if(ic=="AIC"){
        IC <- pAIC;
    }
    else if(ic=="AICc"){
        IC <- pAICc;
    }
    else if(ic=="BIC"){
        IC <- pBIC;
    }
    else if(ic=="BICc"){
        IC <- pBICc;
    }

    distribution <- distribution[1];
    if(distribution=="dnorm"){
        lmCall <- lm;
        listToCall <- vector("list");
    }
    else{
        lmCall <- alm;
        listToCall <- list(distribution=distribution);
    }

    # Observations in sample, assuming that the missing values are for the holdout
    obsInsample <- sum(!is.na(ourData[,1]));
    # Number of variables
    nVariables <- ncol(ourData)-1;
    # Names of variables
    variablesNames <- colnames(ourData)[-1];
    exoNames <- c("(Intercept)",variablesNames);
    responseName <- colnames(ourData)[1];
    y <- as.matrix(ourData[,1]);
    listToCall$data <- ourData;

    # If this is a simple one, go through all the models
    if(bruteForce){
        # Number of combinations in the loop
        nCombinations <- 2^nVariables;
        # Matrix of all the combinations
        variablesBinary <- rep(1,nVariables);
        variablesCombinations <- matrix(NA,nCombinations,nVariables);

        #Produce matrix with binaries for inclusion of variables in the loop
        variablesCombinations[,1] <- rep(c(0:1),times=prod(variablesBinary[-1]+1));
        for(i in 2:nVariables){
            variablesCombinations[,i] <- rep(c(0:variablesBinary[i]),each=prod(variablesBinary[1:(i-1)]+1));
        }

        # Vector of ICs
        pICs <- matrix(NA,obsInsample,nCombinations);
        # Matrix of parameters
        parameters <- matrix(0,nCombinations,nVariables+1);
        # Matrix of s.e. of parameters
        parametersSE <- matrix(0,nCombinations,nVariables+1);

        # Starting estimating the models with just a constant
        listToCall$formula <- as.formula(paste0(responseName,"~1"));
        # listToCall$data <- ourData;
        ourModel <- do.call(lmCall,listToCall);
        pICs[,1] <- IC(ourModel);
        parameters[1,1] <- coef(ourModel)[1];
        parametersSE[1,1] <- diag(vcov(ourModel));
    }
    else{
        if(!silent){
            cat("Selecting the best model...\n");
        }
        bestModel <- stepwise(ourData, ic=ic, distribution=distribution);

        # If the number of variables is small, do bruteForce
        if(nParam(bestModel)<16){
            newData <-  ourData[,c(colnames(ourData)[1],names(bestModel$ICs)[-1])];
            bestModel <- lmDynamic(newData, ic=ic, bruteForce=TRUE, silent=silent, distribution=distribution);
            bestModel$call <- cl;
            return(bestModel);
        }
        # If we have too many variables, use "stress" analysis
        else{
            # Extract names of the used variables
            bestExoNames <- names(coef(bestModel))[-1];
            nVariablesInModel <- length(bestExoNames);
            # Define number of combinations:
            # 1. Best model
            nCombinations <- 1;
            # 2. BM + {each variable not included};
            nCombinations <- nCombinations + nVariables - nVariablesInModel;
            # 3. BM - {each variable included};
            nCombinations <- nCombinations + nVariablesInModel;

            ## 4. BM + {each pair of variables not included};
            ## 5. BM - {each pair of variables included}.

            # Form combinations of variables
            variablesCombinations <- matrix(NA,nCombinations,nVariables);
            colnames(variablesCombinations) <- variablesNames;
            # Fill in the first row with the variables from the best model
            for(j in 1:nVariables){
                variablesCombinations[,j] <- any(colnames(variablesCombinations)[j]==bestExoNames)*1;
            }

            # Fill in the first part with sequential inclusion
            for(i in 1:(nVariables - nVariablesInModel)){
                variablesCombinations[i+1,which(variablesCombinations[i+1,]==0)[i]] <- 1;
            }

            # Fill in the second part with sequential exclusion
            for(i in 1:nVariablesInModel){
                index <- i+(nVariables-nVariablesInModel)+1;
                variablesCombinations[index,which(variablesCombinations[index,]==1)[i]] <- 0;
            }

            # Vector of ICs
            pICs <- matrix(NA,obsInsample,nCombinations);
            # Matrix of parameters
            parameters <- matrix(0,nCombinations,nVariables+1);
            # Matrix of s.e. of parameters
            parametersSE <- matrix(0,nCombinations,nVariables+1);

            # Starting estimating the models with writing down the best one
            pICs[,1] <- pAIC(bestModel);
            bufferCoef <- coef(bestModel)[exoNames];
            parameters[1,c(1,variablesCombinations[1,])==1] <- bufferCoef[!is.na(bufferCoef)];
            bufferCoef <- diag(vcov(bestModel))[exoNames];
            parametersSE[1,c(1,variablesCombinations[1,])==1] <- bufferCoef[!is.na(bufferCoef)];
        }
    }

    if(!silent){
        cat(paste0("Estimation progress: ", round(1/nCombinations,2)*100,"%"));
    }
    # Go for the loop of lm models
    for(i in 2:nCombinations){
        if(!silent){
            cat(paste0(rep("\b",nchar(round((i-1)/nCombinations,2)*100)+1),collapse=""));
            cat(paste0(round(i/nCombinations,2)*100,"%"));
        }
        lmFormula <- paste0(responseName,"~",paste0(variablesNames[variablesCombinations[i,]==1],collapse="+"));
        # ourModel <- alm(as.formula(lmFormula),data=ourData,distribution=distribution);
        listToCall$formula <- as.formula(lmFormula);
        # listToCall$data <- ourData;
        ourModel <- do.call(lmCall,listToCall);
        pICs[,i] <- pAIC(ourModel);
        parameters[i,c(1,variablesCombinations[i,])==1] <- coef(ourModel);
        parametersSE[i,c(1,variablesCombinations[i,])==1] <- diag(vcov(ourModel));
    }

    # Calculate IC weights
    pICWeights <- pICs - apply(pICs,1,min);
    pICWeights <- exp(-0.5*pICWeights) / apply(exp(-0.5*pICWeights),1,sum)

    # Calculate weighted parameters
    parametersWeighted <- pICWeights %*% parameters;
    colnames(parametersWeighted) <- exoNames;

    parametersMean <- colMeans(parametersWeighted);
    names(parametersMean) <- exoNames;

    # From the matrix of exogenous variables without the response variable
    ourDataExo <- cbind(rep(1,nrow(ourData)),ourData[,-1]);
    colnames(ourDataExo) <- exoNames;

    # Calculate the mean based on the mean values of parameters
    mu <- as.matrix(ourDataExo) %*% parametersMean;
    # mu <- rowSums(parametersWeighted * as.matrix(ourDataExo));

    yFitted <- switch(distribution,
                      "dfnorm" =,
                      "dnorm" =,
                      "dlaplace" =,
                      "dlogis" =,
                      "ds" = mu,
                      "dchisq" = mu^2,
                      "dpois" =,
                      "dnbinom" =,
                      "dlnorm" = exp(mu),
                      "pnorm" = pnorm(mu, mean=0, sd=1),
                      "plogis" = plogis(mu, location=0, scale=1)
    );

    errors <- switch(distribution,
                     "dfnorm" =,
                     "dlaplace" =,
                     "dlogis" =,
                     "ds" =,
                     "dnorm" =,
                     "dpois" =,
                     "dnbinom" = y - yFitted,
                     "dchisq" = sqrt(y) - sqrt(mu),
                     "dlnorm"= log(y) - mu,
                     "pnorm" = qnorm((y - pnorm(mu, 0, 1) + 1) / 2, 0, 1),
                     "plogis" = log((1 + y * (1 + exp(mu))) / (1 + exp(mu) * (2 - y) - y)) # Here we use the proxy from Svetunkov et al. (2018)
    );

    # Relative importance of variables
    importance <- cbind(1,pICWeights %*% variablesCombinations);
    colnames(importance) <- exoNames;

    # Some of the variables have partial inclusion, 1 stands for constant
    # This is the dynamic degrees of freedom
    df <- obsInsample - apply(importance,1,sum) - 1;

    # Dynamic weighted mean pAIC
    ICValue <- apply(pICWeights * pICs,1,sum);

    # Models SE
    parametersSECombined <- pICWeights %*% sqrt(parametersSE +(parameters - matrix(parametersMean,nrow(parameters),ncol(parameters),byrow=T))^2);
    colnames(parametersSECombined) <- exoNames;

    #Calcualte logLik
    logLikCombined <- sum(dnorm(errors,0,sd=sqrt(sum(errors^2)/df),log=TRUE));

    finalModel <- list(coefficients=parametersMean, se=parametersSECombined, fitted.values=as.vector(yFitted),
                       residuals=as.vector(errors), distribution=distribution, logLik=logLikCombined, IC=ICValue,
                       df.residual=mean(df), df=sum(apply(importance,2,mean))+1, importance=importance,
                       coefficientsDynamic=parametersWeighted, df.residualDynamic=df, dfDynamic=apply(importance,1,sum)+1,
                       call=cl, rank=nVariables+1, data=as.matrix(ourData), mu=mu);

    return(structure(finalModel,class=c("greyboxD","alm","greybox")));
}
