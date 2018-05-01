#' Combine regressions based on information criteria
#'
#' Function combines parameters of linear regressions of the first variable
#' on all the other provided data.
#'
#' The algorithm uses lm() to fit different models and then combines the models
#' based on the selected IC.
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
#'
#' @return Function returns \code{model} - the final model of the class
#' "lm.combined".
#'
#' @seealso \code{\link[stats]{step}, \link[greybox]{xregExpander},
#' \link[greybox]{stepwise}}
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
#' ourModel <- combiner(inSample,bruteForce=TRUE)
#' forecast(ourModel,outSample)
#' plot(forecast(ourModel,outSample))
#'
#' ### Fat regression example
#' xreg <- matrix(rnorm(5000,10,3),50,100)
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rnorm(50,0,3),xreg,rnorm(50,300,10))
#' colnames(xreg) <- c("y",paste0("x",c(1:100)),"Noise")
#' inSample <- xreg[1:40,]
#' outSample <- xreg[-c(1:40),]
#' # Combine only the models close to the optimal
#' ourModel <- combiner(inSample,ic="AICc",bruteForce=FALSE)
#' summary(ourModel)
#' plot(forecast(ourModel,outSample))
#'
#' @importFrom stats dnorm
#'
#' @export combiner
combiner <- function(data, ic=c("AICc","AIC","BIC"), bruteForce=FALSE, silent=TRUE){
    # Function combines linear regression models and produces the combined lm object.
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
        IC <- AIC;
    }
    else if(ic=="AICc"){
        IC <- AICc;
    }
    else if(ic=="BIC"){
        IC <- BIC;
    }

    # Observations in sample, assuming that the missing values are for the holdout
    obsInsample <- sum(!is.na(ourData[,1]));
    # Number of variables
    nVariables <- ncol(ourData)-1;
    # Names of variables
    variablesNames <- colnames(ourData)[-1];
    exoNames <- c("(Intercept)",variablesNames);
    responseName <- colnames(ourData)[1];

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
        ICs <- rep(NA,nCombinations);
        # Matrix of parameters
        parameters <- matrix(0,nCombinations,nVariables+1);
        # Matrix of s.e. of parameters
        parametersSE <- matrix(0,nCombinations,nVariables+1);

        # Starting estimating the models with just a constant
        ourModel <- lm(as.formula(paste0(responseName,"~1")),data=ourData);
        ICs[1] <- IC(ourModel);
        parameters[1,1] <- coef(ourModel)[1];
        parametersSE[1,1] <- diag(vcov(ourModel));
    }
    else{
        if(nVariables>=obsInsample){
            if(!silent){
                cat("Selecting the best model...\n");
            }
            bestModel <- stepwise(ourData, ic=ic, method="kendall");
        }
        else{
            bestModel <- stepwise(ourData, ic=ic);
        }

        # If the number of variables is small, do bruteForce
        if(ncol(bestModel$model)<16){
            newData <-  ourData[,c(colnames(ourData)[1],names(bestModel$ICs)[-1])];
            return(combiner(newData, ic=ic, bruteForce=TRUE, silent=silent));
        }
        # If we have too many variables, use "stress" analysis
        else{
            # Extract names of the used variables
            bestExoNames <- colnames(bestModel$model)[-1];
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
            ICs <- rep(NA,nCombinations);
            # Matrix of parameters
            parameters <- matrix(0,nCombinations,nVariables+1);
            # Matrix of s.e. of parameters
            parametersSE <- matrix(0,nCombinations,nVariables+1);

            # Starting estimating the models with writing down the best one
            ICs[1] <- IC(bestModel);
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
        ourModel <- lm(as.formula(lmFormula),data=ourData);
        ICs[i] <- IC(ourModel);
        parameters[i,c(1,variablesCombinations[i,])==1] <- coef(ourModel);
        parametersSE[i,c(1,variablesCombinations[i,])==1] <- diag(vcov(ourModel));
    }

    # Calculate IC weights
    ICWeights <- ICs - min(ICs);
    ICWeights <- exp(-0.5*ICWeights) / sum(exp(-0.5*ICWeights));

    # Calculate weighted parameters
    parametersWeighted <- parameters * matrix(ICWeights,nrow(parameters),ncol(parameters));

    parametersCombined <- apply(parametersWeighted,2,sum);
    names(parametersCombined) <- exoNames;

    # From the matrix of exogenous variables without the response variable
    ourDataExo <- cbind(rep(1,nrow(ourData)),ourData[,-1]);
    colnames(ourDataExo) <- exoNames;

    yFitted <- as.matrix(ourDataExo) %*% parametersCombined;
    errors <- ourData[,1] - yFitted;

    # Relative importance of variables
    importance <- c(1,round(ICWeights %*% variablesCombinations,3));
    names(importance) <- exoNames;

    # Some of the variables have partial inclusion, 1 stands for constant
    df <- obsInsample - sum(importance) - 1;

    ICValue <- c(ICWeights %*% ICs);
    names(ICValue) <- ic;

    # Models SE
    parametersSECombined <- c(ICWeights %*% sqrt(parametersSE +(parameters - matrix(apply(parametersWeighted,2,sum),nrow(parameters),ncol(parameters),byrow=T))^2))
    names(parametersSECombined) <- exoNames;

    # Create an object of the same name as the original data
    # If it was a call on its own, make it one string
    assign(paste0(deparse(substitute(data)),collapse=""),as.data.frame(data));
    testModel <- do.call("lm", list(formula=as.formula(paste0(responseName,"~.")), data=substitute(data)));

    #Calcualte logLik
    logLikCombined <- sum(dnorm(errors,0,sd=sqrt(sum(errors^2)/df),log=TRUE));

    ourTerms <- testModel$terms;

    finalModel <- list(coefficients=parametersCombined, residuals=as.vector(errors), fitted.values=as.vector(yFitted),
                       df.residual=df, coefficientsSE=parametersSECombined, importance=importance,
                       IC=ICValue, call=testModel$call, logLik=logLikCombined, rank=nVariables+1,
                       model=ourData, terms=ourTerms, qr=qr(ourData), df=sum(importance)+1);

    return(structure(finalModel,class=c("greyboxC","greybox","lm")));
}
