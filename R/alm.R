#' Advanced Linear Model
#'
#' Function estimates model based on the selected distribution
#'
#' This is a function, similar to \link[stats]{lm}, but for the cases of several
#' non-normal distributions. These include:
#' \enumerate{
#' \item Normal distribution, \link[stats]{dnorm},
#' \item Folded normal distribution, \link[greybox]{dfnorm},
#' \item Log normal distribution, \link[stats]{dlnorm},
#' \item Laplace distribution, \link[greybox]{dlaplace},
#' \item S-distribution, \link[greybox]{ds},
#' \item Chi-Squared Distribution, \link[stats]{dchisq}.
#' }
#'
#' This function is slower than \code{lm}, because it relies on likelihood estimation
#' of parameters, hessian calculation and matrix multiplication. So think twice when
#' using \code{distribution="norm"} here.
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
#' @param A Vector of parameters of the linear model. When \code{NULL}, it
#' is estimated.
#' @param vcovProduce Whether to produce variance-covariance matrix of
#' coefficients or not. This is done via hessian calculation, so might be
#' computationally costly.
#'
#' @return Function returns \code{model} - the final model of the class
#' "alm", which contains:
#' \itemize{
#' \item coefficients - estimated parameters of the model,
#' \item vcov - covariance matrix of parameters of the model (based on Fisher
#' Information). Returned only when \code{vcovProduce=TRUE}.
#' \item actuals - actual values of the response variable,
#' \item fitted.values - fitted values,
#' \item residuals - residuals of the model,
#' \item mu - the estimated location parameter of the distribution,
#' \item scale - the estimated scale parameter of the distribution,
#' \item distribution - distribution used in the estimation,
#' \item logLik - log-likelihood of the model,
#' \item df.residual - number of degrees of freedom of the residuals of the model,
#' \item df - number of degrees of freedom of the model,
#' \item call - how the model was called,
#' \item rank - rank of the model,
#' \item model - data on which the model was fitted,
#' \item qr - QR decomposition of the data,
#' \item terms - terms of the model.
#' }
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
#'
#' ourModel <- alm(y~x1+x2, inSample, distribution="laplace")
#' summary(ourModel)
#' plot(predict(ourModel,outSample))
#'
#' @importFrom numDeriv hessian
#' @importFrom nloptr nloptr
#' @importFrom stats model.frame sd terms dchisq dlnorm dnorm
#' @export alm
alm <- function(formula, data, subset=NULL,  na.action,
                distribution=c("norm","fnorm","lnorm","laplace","s","chisq"),
                A=NULL, vcovProduce=FALSE){

    cl <- match.call();

    distribution <- distribution[1];
    if(all(distribution!=c("norm","fnorm","lnorm","laplace","s","chisq"))){
        stop(paste0("Sorry, but the distribution '",distribution,"' is not yet supported"), call.=FALSE);
    }
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

    ifelseFast <- function(condition, yes, no){
        if(condition){
            return(yes);
        }
        else{
            return(no);
        }
    }

    fitter <- function(A, distribution, y, matrixXreg){
        mu <- matrixXreg %*% A;

        scale <- switch(distribution,
                        "norm"=,
                        "fnorm" = sqrt(mean((y-mu)^2)),
                        "lnorm"= sqrt(mean((log(y)-mu)^2)),
                        "laplace" = mean(abs(y-mu)),
                        "s" = mean(sqrt(abs(y-mu))) / 2,
                        "chisq" = 2*mu
        );

        return(list(mu=mu,scale=scale));
    }

    CF <- function(A, distribution, y, matrixXreg){
        fitterReturn <- fitter(A, distribution, y, matrixXreg);

        CFReturn <- switch(distribution,
                           "norm" = dnorm(y, mean=fitterReturn$mu, sd=fitterReturn$scale, log=TRUE),
                           "fnorm" = dfnorm(y, mu=fitterReturn$mu, sigma=fitterReturn$scale, log=TRUE),
                           "lnorm" = dlnorm(y, meanlog=fitterReturn$mu, sdlog=fitterReturn$scale, log=TRUE),
                           "laplace" = dlaplace(y, mu=fitterReturn$mu, b=fitterReturn$scale, log=TRUE),
                           "s" = ds(y, mu=fitterReturn$mu, b=fitterReturn$scale, log=TRUE),
                           "chisq" = ifelseFast(any(fitterReturn$mu<=0),-1E+300,dchisq(y, df=fitterReturn$mu, log=TRUE))
        );

        CFReturn <- -sum(CFReturn[is.finite(CFReturn)]);

        if(is.nan(CFReturn) | is.na(CFReturn) | is.infinite(CFReturn)){
            CFReturn <- 1E+300;
        }

        return(CFReturn);
    }

    if(is.null(A)){
        if(distribution=="lnorm"){
            A <- as.vector(chol2inv(chol(t(matrixXreg) %*% matrixXreg)) %*% t(matrixXreg) %*% log(y));
        }
        else{
            A <- as.vector(chol2inv(chol(t(matrixXreg) %*% matrixXreg)) %*% t(matrixXreg) %*% y);
        }

        # Although this is not needed in case of distribution="norm", we do that in a way, for the code consistency purposes
        res <- nloptr(A, CF,
                      opts=list("algorithm"="NLOPT_LN_SBPLX", xtol_rel=1e-8, maxeval=500),
                      distribution=distribution, y=y, matrixXreg=matrixXreg);

        # res <- nloptr(A, CF, opts=list("algorithm"="NLOPT_LN_BOBYQA", xtol_rel=1e-8, print_level=1),
        #               distribution=distribution, y=y, matrixXreg=matrixXreg);
        # res <- nloptr(res$solution, CF, opts=list("algorithm"="NLOPT_LN_NELDERMEAD", xtol_rel=1e-6),
        #               distribution=distribution, y=y, matrixXreg=matrixXreg);
        A <- res$solution;
        CFValue <- res$objective;
    }
    else{
        CFValue <- CF(A, distribution, y, matrixXreg);
    }

    if(vcovProduce){
        vcovMatrix <- hessian(CF, A, #method.args=list(d=1e-10,r=6),
                              distribution=distribution, y=y, matrixXreg=matrixXreg);

        if(any(is.nan(vcovMatrix))){
            warning(paste0("Something went wrong and we failed to produce the covariance matrix of the parameters.\n",
                           "Obviously, it's not our fault. Probably Russians have hacked your computer...\n",
                           "Try a different distribution maybe?"), call.=FALSE);
            vcovMatrix <- 1e-10*diag(nVariables);
        }
        else{
            if(any(vcovMatrix==0)){
                warning(paste0("Something went wrong and we failed to produce the covariance matrix of the parameters.\n",
                               "Obviously, it's not our fault. Probably Russians have hacked your computer...\n",
                               "Try a different distribution maybe?"), call.=FALSE);
                vcovMatrix <- 1e-10*diag(nVariables);
            }
            else{
                # vcovMatrix <- solve(vcovMatrix, diag(nVariables));
                vcovMatrix <- chol2inv(chol(vcovMatrix));
            }

            # Sometimes the diagonal elements in the covariance matrix are negative because likelihood is not fully maximised...
            if(any(diag(vcovMatrix)<0)){
                diag(vcovMatrix) <- abs(diag(vcovMatrix));
            }
        }

        if(nVariables>1){
            dimnames(vcovMatrix) <- list(variablesNames,variablesNames);
        }
        else{
            names(vcovMatrix) <- variablesNames;
        }
    }
    else{
        vcovMatrix <- NULL;
    }

    fitterReturn <- fitter(A, distribution, y, matrixXreg);
    mu <- fitterReturn$mu;
    scale <- fitterReturn$scale;

    # Parameters of the model + scale
    df <- nVariables + 1;

    if(distribution=="fnorm"){
        # Correction so that the the expectation of the folded is returned
        # Calculate the conditional expectation based on the parameters of the distribution
        yFitted <- sqrt(2/pi)*scale*exp(-mu^2/(2*scale^2))+mu*(1-2*pnorm(-mu/scale));
    }
    else if(distribution=="lnorm"){
        yFitted <- exp(mu);
    }
    else{
        yFitted <- mu;
        if(distribution=="chisq"){
            scale <- mu * 2;
        }
    }
    errors <- y - yFitted;

    names(A) <- variablesNames;

    finalModel <- list(coefficients=A, vcov=vcovMatrix, actuals=y, fitted.values=yFitted, residuals=as.vector(errors),
                       mu=mu, scale=scale, distribution=distribution, logLik=-CFValue,
                       df.residual=obsInsample-df, df=df, call=cl, rank=df, model=dataWork,
                       qr=qr(dataWork), terms=ourTerms);
    return(structure(finalModel,class=c("alm","greybox")));
}
