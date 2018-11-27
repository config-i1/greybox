#' Combine regressions based on information criteria
#'
#' Function combines parameters of linear regressions of the first variable
#' on all the other provided data.
#'
#' The algorithm uses alm() to fit different models and then combines the models
#' based on the selected IC. The parameters are combined so that if they are not
#' present in some of models, it is assumed that they are equal to zero. Thus,
#' there is a shrinkage effect in the combination.
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
#' "greyboxC". The list of variables:
#' \itemize{
#' \item coefficients - combined parameters of the model,
#' \item se - combined standard errors of the parameters of the model,
#' \item actuals - actual values of the response variable,
#' \item fitted.values - the fitted values,
#' \item residuals - residual of the model,
#' \item distribution - distribution used in the estimation,
#' \item logLik - combined log-likelihood of the model,
#' \item IC - the values of the combined information criterion,
#' \item df.residual - number of degrees of freedom of the residuals of
#' the combined model,
#' \item df - number of degrees of freedom of the combined model,
#' \item importance - importance of the parameters,
#' \item call - call used in the function,
#' \item rank - rank of the combined model,
#' \item data - the data used in the model,
#' \item mu - the location value of the distribution.
#' \item combination - the table, indicating which variables were used in every
#' model construction and what were the weights for each model.
#' }
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
#' ourModel <- lmCombine(inSample,bruteForce=TRUE)
#' predict(ourModel,outSample)
#' plot(predict(ourModel,outSample))
#'
#' ### Fat regression example
#' xreg <- matrix(rnorm(5000,10,3),50,100)
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rnorm(50,0,3),xreg,rnorm(50,300,10))
#' colnames(xreg) <- c("y",paste0("x",c(1:100)),"Noise")
#' inSample <- xreg[1:40,]
#' outSample <- xreg[-c(1:40),]
#' # Combine only the models close to the optimal
#' ourModel <- lmCombine(inSample,ic="BICc",bruteForce=FALSE)
#' summary(ourModel)
#' plot(predict(ourModel,outSample))
#'
#' @importFrom stats dnorm
#'
#' @aliases combine combiner
#' @export lmCombine
lmCombine <- function(data, ic=c("AICc","AIC","BIC","BICc"), bruteForce=FALSE, silent=TRUE,
                      distribution=c("dnorm","dfnorm","dlnorm","dlaplace","ds","dchisq","dlogis",
                                    "plogis","pnorm")){
    # Function combines linear regression models and produces the combined lm object.
    cl <- match.call();
    cl$formula <- as.formula(paste0(colnames(data)[1]," ~ 1"));

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
    else if(ic=="BICc"){
        IC <- BICc;
    }

    distribution <- distribution[1];
    if(distribution=="dnorm"){
        lmCall <- function(formula, data){
            model <- list(xreg=as.matrix(cbind(1,data[,as.character(all.vars(formula[[3]]))])))
            model <- c(model,.lm.fit(model$xreg, as.matrix(data[,as.character(formula[[2]])])));
            return(structure(model,class=c("lmGreybox","lm")));
        }
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
    listToCall$data <- ourData;

    # If this is a simple one, go through all the models
    if(bruteForce){
        # Number of combinations in the loop
        nCombinations <- 2^nVariables;
        # Matrix of all the combinations
        variablesBinary <- rep(1,nVariables);
        variablesCombinations <- matrix(NA,nCombinations,nVariables);
        colnames(variablesCombinations) <- variablesNames;

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
        # ourModel <- alm(as.formula(paste0(responseName,"~1")),data=ourData,distribution=distribution);
        listToCall$formula <- as.formula(paste0(responseName,"~1"));
        # listToCall$data <- ourData;
        ourModel <- do.call(lmCall,listToCall);
        ICs[1] <- IC(ourModel);
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
            bestModel <- lmCombine(newData, ic=ic, bruteForce=TRUE, silent=silent, distribution=distribution);
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
        # ourModel <- alm(as.formula(lmFormula),data=ourData,distribution=distribution);
        listToCall$formula <- as.formula(lmFormula);
        # listToCall$data <- ourData;
        ourModel <- do.call(lmCall,listToCall);
        # print(ourModel$data)
        ICs[i] <- IC(ourModel);
        parameters[i,c(1,variablesCombinations[i,])==1] <- coef(ourModel);
        parametersSE[i,c(1,variablesCombinations[i,])==1] <- diag(vcov(ourModel));
    }

    # Calculate IC weights
    # is.finite is needed in the case of overfitting the data
    modelsGood <- is.finite(ICs);
    ICWeights <- ICs - min(ICs[modelsGood]);
    ICWeights[modelsGood] <- exp(-0.5*ICWeights[modelsGood]) / sum(exp(-0.5*ICWeights[modelsGood]));
    # If we awfully overfitted the data with some of models, discard them.
    if(any(!modelsGood)){
        ICWeights[!modelsGood] <- 0;
    }

    # Calculate weighted parameters
    parametersWeighted <- parameters * matrix(ICWeights,nrow(parameters),ncol(parameters));
    parametersCombined <- colSums(parametersWeighted);

    # If shrinkage is not needed, then calculate weights for each parameter and modify the combined parameters
    # if(!shrink){
    #     modifiedWeights <- colSums(cbind(1,variablesCombinations) * matrix(ICWeights,nrow(parameters),nVariables+1));
    #     weightsZero <- (modifiedWeights==0);
    #     parametersCombined <- parametersCombined / modifiedWeights;
    #     # If some of summary weights were zero, then make parameters zero as well
    #     parametersCombined[weightsZero] <- 0;
    # }
    names(parametersCombined) <- exoNames;

    # From the matrix of exogenous variables without the response variable
    ourDataExo <- cbind(rep(1,nrow(ourData)),ourData[,-1]);
    colnames(ourDataExo) <- exoNames;

    mu <- as.matrix(ourDataExo) %*% parametersCombined;

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
                     "dnbinom" = ourData[,1] - yFitted,
                     "dchisq" = sqrt(ourData[,1]) - sqrt(mu),
                     "dlnorm"= log(ourData[,1]) - mu,
                     "pnorm" = qnorm((ourData[,1] - pnorm(mu, 0, 1) + 1) / 2, 0, 1),
                     "plogis" = log((1 + ourData[,1] * (1 + exp(mu))) / (1 + exp(mu) * (2 - ourData[,1]) - ourData[,1])) # Here we use the proxy from Svetunkov et al. (2018)
    );

    # Relative importance of variables
    importance <- c(1,round(ICWeights %*% variablesCombinations,3));
    names(importance) <- exoNames;

    # Some of the variables have partial inclusion, 1 stands for constant
    df <- obsInsample - sum(importance) - 1;

    ICValue <- c(ICWeights %*% ICs);
    names(ICValue) <- ic;

    variablesCombinations <- cbind(variablesCombinations,ICWeights,ICs);
    colnames(variablesCombinations)[nVariables+1] <- "IC weights";

    # Models SE
    parametersSECombined <- c(ICWeights %*% sqrt(parametersSE +(parameters - matrix(apply(parametersWeighted,2,sum),nrow(parameters),ncol(parameters),byrow=T))^2))
    names(parametersSECombined) <- exoNames;

    if(any(is.nan(parametersSECombined)) | any(is.infinite(parametersSECombined))){
        warning("The standard errors of the parameters cannot be produced properly. It seems that we have overfitted the data.", call.=FALSE);
    }

    #Calcualte logLik
    logLikCombined <- sum(dnorm(errors,0,sd=sqrt(sum(errors^2)/df),log=TRUE));

    finalModel <- list(coefficients=parametersCombined, se=parametersSECombined, fitted.values=as.vector(yFitted),
                       residuals=as.vector(errors), distribution=distribution, logLik=logLikCombined, IC=ICValue,
                       df.residual=df, df=sum(importance)+1, importance=importance,
                       call=cl, rank=nVariables+1, data=ourData, mu=mu, combination=variablesCombinations);

    return(structure(finalModel,class=c("greyboxC","alm","greybox")));
}
