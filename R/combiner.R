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
#' @param data Data frame containing dependant variable in the first column and
#' the others in the rest.
#' @param ic Information criterion to use.
#' @param silent If \code{silent=FALSE}, then nothing is silent, everything is
#' printed out. \code{silent=TRUE} means that nothing is produced.
#'
#' @return Function returns \code{model} - the final model of the class
#' "lm.combined".
#'
#' @seealso \code{\link[stats]{step}, \link[greybox]{xregExpander},
#' \link[greybox]{stepwise}}
#'
#' @examples
#'
#' xreg <- cbind(rnorm(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rnorm(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#'
#' combiner(xreg)
#'
#' @export combiner
combiner <- function(data, ic=c("AICc","AIC","BIC"), silent=TRUE){
    # Function combines linear regression models and produces the combined lm object.
    ourData <- data;
    if(!is.data.frame(ourData)){
        ourData <- as.data.frame(ourData);
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
    # Number of combinations in the loop
    nCombinations <- 2^nVariables;
    # Names of variables
    variablesNames <- colnames(ourData)[-1];
    exoNames <- c("(Intercept)",variablesNames);
    responseName <- colnames(ourData)[1];
    # Matrix of all the combinations
    variablesBinary <- rep(1,nVariables);
    variablesCombinations <- matrix(NA,nCombinations,nVariables);

    variablesCombinations[,1] <- rep(c(0:1),times=prod(variablesBinary[-1]+1));
    for(i in 2:nVariables){
        variablesCombinations[,i] <- rep(c(0:variablesBinary[i]),each=prod(variablesBinary[1:(i-1)]+1));
    }

    # Vector of AICs
    AICs <- rep(NA,nCombinations);
    # Matrix of parameters
    parameters <- matrix(0,nCombinations,nVariables+1);
    # Matrix of s.e. of parameters
    parametersSE <- matrix(0,nCombinations,nVariables+1);

    if(!silent){
        cat(paste0("Estimation progress: ", round(1/nCombinations,2)*100,"%"));
    }
    # Starting estimating the models with just a constant
    ourModel <- lm(as.formula(paste0(responseName,"~1")),data=ourData);
    AICs[1] <- IC(ourModel);
    parameters[1,1] <- coef(ourModel)[1];
    parametersSE[1,1] <- diag(vcov(ourModel));
    # Go for the loop of lm models
    for(i in 2:nCombinations){
        if(!silent){
            cat(paste0(rep("\b",nchar(round((i-1)/nCombinations,2)*100)+1),collapse=""));
            cat(paste0(round(i/nCombinations,2)*100,"%"));
        }
        lmFormula <- paste0(responseName,"~",paste0(variablesNames[variablesCombinations[i,]==1],collapse="+"));
        ourModel <- lm(as.formula(lmFormula),data=ourData);
        AICs[i] <- IC(ourModel);
        parameters[i,c(1,variablesCombinations[i,])==1] <- coef(ourModel);
        parametersSE[i,c(1,variablesCombinations[i,])==1] <- diag(vcov(ourModel));
    }

    # Calculate AIC weights
    AICWeights <- AICs - min(AICs);
    AICWeights <- exp(-0.5*AICWeights) / sum(exp(-0.5*AICWeights));

    # Calculate weighted parameters
    parametersWeighted <- parameters * matrix(AICWeights,nrow(parameters),ncol(parameters));

    parametersCombined <- apply(parametersWeighted,2,sum);
    names(parametersCombined) <- exoNames;

    # From the matrix of exogenous variables without the response variable
    ourDataExo <- cbind(rep(1,nrow(ourData)),ourData[,-1]);
    colnames(ourDataExo) <- exoNames;

    yFitted <- as.matrix(ourDataExo) %*% parametersCombined;
    errors <- ourData[,1] - yFitted;

    # Combined model
    # plot(ourData$Sales,type="l")
    # lines(as.matrix(ourDataExo) %*% apply(parametersWeighted,2,sum),col="red")
    # abline(v=156.5,col="blue")

    # Relative importance of variables
    importance <- c(1,round(AICWeights %*% variablesCombinations,3));
    names(importance) <- exoNames;

    # Some of the variables have partial inclusion, 1 stands for constant
    df <- obsInsample - sum(importance) - 1;

    ICValue <- c(AICWeights %*% AICs);
    names(ICValue) <- ic;

    # Models SE
    parametersSECombined <- c(AICWeights %*% sqrt(parametersSE +(parameters - matrix(apply(parametersWeighted,2,sum),nrow(parameters),ncol(parameters),byrow=T))^2))
    names(parametersSECombined) <- exoNames;

    finalModel <- list(coefficients=parametersCombined, residuals=errors, fitted.values=yFitted,
                       df.residual=obsInsample, coefficientsSE=parametersSECombined, importance=importance,
                       IC=ICValue);
                       # logLik=);

    return(structure(finalModel,class=c("lm.combined","lm")));
}
