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
#' \item Chi-Squared Distribution, \link[stats]{dchisq},
#' \item Logistic Distribution, \link[stats]{dlogis}.
#' }
#'
#' This function is slower than \code{lm}, because it relies on likelihood estimation
#' of parameters, hessian calculation and matrix multiplication. So think twice when
#' using \code{distribution="dnorm"} here.
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
#' @param distribution what density function to use in the process. The full
#' name of the distribution should be provided here. Values with "d" in the
#' beginning of the name refer to the density function, while "p" stands for
#' "probability" (cumulative distribution function). The names align with the
#' names of distribution functions in R. For example, see \link[stats]{dnorm}.
#' @param occurrence what distribution to use for occurrence variable. Can be
#' \code{"none"}, then nothing happens; \code{"plogis"} - then the logistic
#' regression using \code{alm()} is estimated for the occurrence part;
#' \code{"pnorm"} - then probit is constructed via \code{alm()} for the
#' occurrence part. In both of the latter cases, the formula used is the same
#' as the formula for the sizes. Finally, an "alm" model can be provided and
#' its estimates will be used in the model construction.
#'
#' If this is not \code{"none"}, then the model is estimated
#' in two steps: 1. Occurrence part of the model; 2. Sizes part of the model
#' (excluding zeroes from the data).
#' @param B vector of parameters of the linear model. When \code{NULL}, it
#' is estimated.
#' @param vcovProduce whether to produce variance-covariance matrix of
#' coefficients or not. This is done via hessian calculation, so might be
#' computationally costly.
#' @param ... additional parameters to pass to distribution functions
#' (e.g. \code{alpha} value for Asymmetric Laplace distribution).
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
#' \item data - data used for the model construction,
#' \item occurrence - the occurrence model used in the estimation,
#' \item other - the list of all the other parameters either passed to the
#' function or estimated in the process, but not included in the standard output
#' (e.g. \code{alpha} for Asymmetric Laplace).
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
#' ourModel <- alm(y~x1+x2, inSample, distribution="dlaplace")
#' summary(ourModel)
#' plot(predict(ourModel,outSample))
#'
#' # An example with binary response variable
#' xreg[,1] <- round(exp(xreg[,1]-70) / (1 + exp(xreg[,1]-70)),0)
#' colnames(xreg) <- c("y","x1","x2","Noise")
#' inSample <- xreg[1:80,]
#' outSample <- xreg[-c(1:80),]
#'
#' # Logistic distribution (logit regression)
#' ourModel <- alm(y~x1+x2, inSample, distribution="plogis")
#' summary(ourModel)
#' plot(predict(ourModel,outSample,interval="c"))
#'
#' # Normal distribution (probit regression)
#' ourModel <- alm(y~x1+x2, inSample, distribution="pnorm")
#' summary(ourModel)
#' plot(predict(ourModel,outSample,interval="p"))
#'
#' @importFrom numDeriv hessian
#' @importFrom nloptr nloptr
#' @importFrom stats model.frame sd terms
#' @importFrom stats dchisq dlnorm dnorm dlogis dpois dnbinom
#' @importFrom stats plogis
#' @export alm
alm <- function(formula, data, subset, na.action,
                distribution=c("dnorm","dlogis","dlaplace","dalaplace","ds",
                               "dfnorm","dlnorm","dchisq",
                               "dpois","dnbinom",
                               "plogis","pnorm"),
                occurrence=c("none","plogis","pnorm"),
                B=NULL, vcovProduce=FALSE, ...){
# Useful stuff for dnbinom: https://scialert.net/fulltext/?doi=ajms.2010.1.15

    cl <- match.call();

    distribution <- distribution[1];
    if(all(distribution!=c("dnorm","dlogis","dlaplace","dalaplace","ds","dfnorm","dlnorm","dchisq",
                           "dpois","dnbinom","plogis","pnorm"))){
        if(any(distribution==c("norm","fnorm","lnorm","laplace","s","chisq","logis"))){
            warning(paste0("You are using the old value of the distribution parameter.\n",
                           "Use distribution='d",distribution,"' instead."),
                    call.=FALSE);
            distribution <- paste0("d",distribution);
        }
        else{
            stop(paste0("Sorry, but the distribution '",distribution,"' is not yet supported"), call.=FALSE);
        }
    }

    ellipsis <- list(...);

    # If this is ALD, then see if alpha was provided. Otherwise estimate it.
    if(distribution=="dalaplace"){
        if(is.null(ellipsis$alpha)){
            ellipsis$alpha <- alpha <- 0.5
            alphaEstimate <- TRUE;
        }
        else{
            alpha <- ellipsis$alpha;
            alphaEstimate <- FALSE;
        }
    }

    if(is.alm(occurrence)){
        occurrenceModel <- TRUE;
        occurrenceProvided <- TRUE;
    }
    else{
        occurrence <- occurrence[1];
        occurrenceProvided <- FALSE;
        if(all(occurrence!=c("none","plogis","pnorm"))){
            warning(paste0("Sorry, but we don't know what to do with the occurrence '",occurrence,
                        "'. Switching to 'none'."), call.=FALSE);
            occurrence <- "none";
        }

        if(any(occurrence==c("plogis","pnorm"))){
            occurrenceModel <- TRUE;
        }
        else{
            occurrenceModel <- FALSE;
            occurrence <- NULL;
        }
    }

    #### Form the necessary matrices ####
    # Call similar to lm in order to form appropriate data.frame
    mf <- match.call(expand.dots = FALSE);
    m <- match(c("formula", "data", "subset", "na.action"), names(mf), 0L);
    mf <- mf[c(1L, m)];
    mf$drop.unused.levels <- TRUE;
    mf[[1L]] <- quote(stats::model.frame);

    if(!is.data.frame(data)){
        data <- as.data.frame(data);
        mf$data <- data;
    }

    responseName <- all.vars(formula)[1];
    # If this is a model with occurrence, use only non-zero observations
    if(occurrenceModel){
        occurrenceNonZero <- data[,responseName]!=0;
        mf$subset <- occurrenceNonZero;
    }

    dataWork <- eval(mf, parent.frame());
    y <- dataWork[,1];

    interceptIsNeeded <- attr(terms(dataWork),"intercept")!=0;
    obsInsample <- nrow(dataWork);
    variablesNames <- colnames(dataWork)[-1];

    # Record the subset used in the model
    if(is.null(mf$subset)){
        subset <- rep(TRUE, obsInsample);
    }
    else{
        subset <- mf$subset;
    }

    matrixXreg <- as.matrix(dataWork[,-1]);
    nVariables <- length(variablesNames);
    colnames(matrixXreg) <- variablesNames;

    mu <- vector("numeric", obsInsample);
    yFitted <- vector("numeric", obsInsample);
    errors <- vector("numeric", obsInsample);
    ot <- vector("logical", obsInsample);

    if(any(y<0) & any(distribution==c("dfnorm","dlnorm","dchisq","dpois","dnbinom"))){
        stop(paste0("Negative values are not allowed in the response variable for the distribution '",distribution,"'"),
             call.=FALSE);
    }

    if(any(distribution==c("dpois","dnbinom"))){
        if(any(y!=trunc(y))){
            stop(paste0("Count data is needed for the distribution '",distribution,"', but you have fractional numbers. ",
                        "Maybe you should try some other distribution?"),
                 call.=FALSE);
        }
    }

    if(any(distribution==c("plogis","pnorm"))){
        CDF <- TRUE;
    }
    else{
        CDF <- FALSE;
    }

    if(CDF & any(y!=0 & y!=1)){
        warning(paste0("You have defined CDF '",distribution,"' as a distribution.\n",
                       "This means that the response variable needs to be binary with values of 0 and 1.\n",
                       "Don't worry, we will encode it for you. But, please, be careful next time!"),
                call.=FALSE);
        y <- (y!=0)*1;
    }

    if(CDF){
        ot[] <- y!=0;
    }
    else{
        ot[] <- rep(TRUE,obsInsample);
    }

    #### Checks of the exogenous variables ####
    # Remove the data for which sd=0
    noVariability <- vector("logical",nVariables);
    noVariability[] <- apply(matrixXreg==matrix(matrixXreg[1,],obsInsample,nVariables,byrow=TRUE),2,all);
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

    corThreshold <- 0.999;
    if(nVariables>1){
        # Check perfectly correlated cases
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

    #### Finish forming the matrix of exogenous variables ####
    if(interceptIsNeeded){
        matrixXreg <- cbind(1,matrixXreg);
        variablesNames <- c("(Intercept)",variablesNames);
        nVariables <- length(variablesNames);
        colnames(matrixXreg) <- variablesNames;
    }

    #### Functions used in the estimation ####
    ifelseFast <- function(condition, yes, no){
        if(condition){
            return(yes);
        }
        else{
            return(no);
        }
    }

    meanFast <- function(x){
        return(sum(x) / length(x));
    }

    fitter <- function(B, distribution, y, matrixXreg){
        if(distribution=="dalaplace"){
            if(alphaEstimate){
                alpha <- B[1];
                B <- B[-1];
            }
        }

        mu[] <- switch(distribution,
                       "dpois" = exp(matrixXreg %*% B),
                       "dchisq" = ifelseFast(any(matrixXreg %*% B[-1] <0),1E+100,(matrixXreg %*% B[-1])^2),
                       "dnbinom" = exp(matrixXreg %*% B[-1]),
                       "dnorm" =,
                       "dfnorm" =,
                       "dlnorm" =,
                       "dlaplace" =,
                       "dalaplace" =,
                       "dlogis" =,
                       "ds" =,
                       "pnorm" =,
                       "plogis" = matrixXreg %*% B
        );

        scale <- switch(distribution,
                        "dnorm" =,
                        "dfnorm" = sqrt(meanFast((y-mu)^2)),
                        "dlnorm" = sqrt(meanFast((log(y)-mu)^2)),
                        "dlaplace" = meanFast(abs(y-mu)),
                        "dalaplace" = meanFast((y-mu) * (alpha - (y<=mu)*1)),
                        "dlogis" = sqrt(meanFast((y-mu)^2) * 3 / pi^2),
                        "ds" = meanFast(sqrt(abs(y-mu))) / 2,
                        "dchisq" =,
                        "dnbinom" = abs(B[1]),
                        "dpois" = mu,
                        "pnorm" = sqrt(meanFast(qnorm((y - pnorm(mu, 0, 1) + 1) / 2, 0, 1)^2)),
                        "plogis" = sqrt(meanFast(log((1 + y * (1 + exp(mu))) / (1 + exp(mu) * (2 - y) - y))^2)) # Here we use the proxy from Svetunkov et al. (2018)
        );

        return(list(mu=mu,scale=scale));
    }

    CF <- function(B, distribution, y, matrixXreg){
        fitterReturn <- fitter(B, distribution, y, matrixXreg);

        CFReturn <- switch(distribution,
                           "dnorm" = dnorm(y, mean=fitterReturn$mu, sd=fitterReturn$scale, log=TRUE),
                           "dfnorm" = dfnorm(y, mu=fitterReturn$mu, sigma=fitterReturn$scale, log=TRUE),
                           "dlnorm" = dlnorm(y, meanlog=fitterReturn$mu, sdlog=fitterReturn$scale, log=TRUE),
                           "dlaplace" = dlaplace(y, mu=fitterReturn$mu, b=fitterReturn$scale, log=TRUE),
                           "dalaplace" = dalaplace(y, mu=fitterReturn$mu, b=fitterReturn$scale, alpha=alpha, log=TRUE),
                           "dlogis" = dlogis(y, location=fitterReturn$mu, scale=fitterReturn$scale, log=TRUE),
                           "ds" = ds(y, mu=fitterReturn$mu, b=fitterReturn$scale, log=TRUE),
                           "dchisq" = dchisq(y, df=fitterReturn$scale, ncp=fitterReturn$mu, log=TRUE),
                           "dpois" = dpois(y, lambda=fitterReturn$mu, log=TRUE),
                           "dnbinom" = dnbinom(y, mu=fitterReturn$mu, size=fitterReturn$scale, log=TRUE),
                           "pnorm" = c(pnorm(fitterReturn$mu[ot], mean=0, sd=1, log.p=TRUE),
                                       pnorm(fitterReturn$mu[!ot], mean=0, sd=1, lower.tail=FALSE, log.p=TRUE)),
                           "plogis" = c(plogis(fitterReturn$mu[ot], location=0, scale=1, log.p=TRUE),
                                        plogis(fitterReturn$mu[!ot], location=0, scale=1, lower.tail=FALSE, log.p=TRUE))
        );

        CFReturn <- -sum(CFReturn);

        if(is.nan(CFReturn) | is.na(CFReturn) | is.infinite(CFReturn)){
            CFReturn <- 1E+300;
        }

        return(CFReturn);
    }

    #### Estimate parameters of the model ####
    if(is.null(B)){
        if(any(distribution==c("dlnorm","dpois","dnbinom"))){
            if(any(y[ot]==0)){
                # Use Box-Cox if there are zeroes
                B <- .lm.fit(matrixXreg,(y^0.01-1)/0.01)$coefficients;
            }
            else{
                B <- .lm.fit(matrixXreg,log(y))$coefficients;
            }
        }
        else if(any(distribution==c("plogis","pnorm"))){
            # Box-Cox transform in order to get meaningful initials
            B <- .lm.fit(matrixXreg,(y^0.01-1)/0.01)$coefficients;
        }
        else{
            B <- .lm.fit(matrixXreg,y)$coefficients;
            BLower <- -Inf;
            BUpper <- Inf;
        }

        if(distribution=="dnbinom"){
            B <- c(var(y), B);
            BLower <- c(0,rep(-Inf,length(B)-1));
            BUpper <- rep(Inf,length(B));
        }
        else if(distribution=="dchisq"){
            B <- c(1, B);
            BLower <- c(0,rep(-Inf,length(B)-1));
            BUpper <- rep(Inf,length(B));
        }
        else if(distribution=="dalaplace"){
            if(alphaEstimate){
                B <- c(alpha, B);
                BLower <- c(0,rep(-Inf,length(B)-1));
                BUpper <- c(1,rep(Inf,length(B)-1));
            }
            else{
                BLower <- rep(-Inf,length(B));
                BUpper <- rep(Inf,length(B));
            }
        }
        else{
            BLower <- rep(-Inf,length(B));
            BUpper <- rep(Inf,length(B));
        }

        if(any(distribution==c("dpois","dnbinom","plogos","pnorm"))){
            maxeval <- 500;
        }
        else{
            maxeval <- 100;
        }

        # Although this is not needed in case of distribution="dnorm", we do that in a way, for the code consistency purposes
        res <- nloptr(B, CF,
                      opts=list("algorithm"="NLOPT_LN_SBPLX", xtol_rel=1e-6, maxeval=maxeval, print_level=0),
                      lb=BLower, ub=BUpper,
                      distribution=distribution, y=y, matrixXreg=matrixXreg);
        B[] <- res$solution;

        CFValue <- res$objective;
    }
    else{
        CFValue <- CF(B, distribution, y, matrixXreg);
    }

    #### Form the fitted values, location and scale ####
    fitterReturn <- fitter(B, distribution, y, matrixXreg);
    mu[] <- fitterReturn$mu;
    scale <- fitterReturn$scale;

    if(distribution=="dnbinom"){
        names(B) <- c("size",variablesNames);
    }
    else if(distribution==c("dchisq")){
        scale <- abs(B[1]);
        names(B) <- c("df",variablesNames);
    }
    else if(distribution=="dalaplace"){
        if(alphaEstimate){
            ellipsis$alpha <- alpha <- B[1];
            variablesNames <- c("alpha",variablesNames);
            names(B) <- variablesNames;
        }
        else{
            names(B) <- variablesNames;
        }
    }
    else{
        names(B) <- variablesNames;
    }

    # Parameters of the model + scale
    df <- nVariables + 1;

    if(distribution=="dalaplace"){
        if(alphaEstimate){
            df <- df + 1;
            nVariables <- nVariables + 1;
        }
    }

    ### Fitted values in the scale of the original variable
    yFitted[] <- switch(distribution,
                       "dfnorm" = sqrt(2/pi)*scale*exp(-mu^2/(2*scale^2))+mu*(1-2*pnorm(-mu/scale)),
                       "dnorm" =,
                       "dlaplace" =,
                       "dalaplace" =,
                       "dlogis" =,
                       "ds" =,
                       "dpois" =,
                       "dnbinom" = mu,
                       "dchisq" = mu + scale,
                       "dlnorm" = exp(mu),
                       "pnorm" = pnorm(mu, mean=0, sd=1),
                       "plogis" = plogis(mu, location=0, scale=1)
    );

    ### Error term in the transformed scale
    errors[] <- switch(distribution,
                       "dfnorm" =,
                       "dlaplace" =,
                       "dalaplace" =,
                       "dlogis" =,
                       "ds" =,
                       "dnorm" =,
                       "dpois" =,
                       "dnbinom" = y - mu,
                       "dchisq" = sqrt(y) - sqrt(mu),
                       "dlnorm"= log(y) - mu,
                       "pnorm" = qnorm((y - pnorm(mu, 0, 1) + 1) / 2, 0, 1),
                       "plogis" = log((1 + y * (1 + exp(mu))) / (1 + exp(mu) * (2 - y) - y)) # Here we use the proxy from Svetunkov et al. (2018)
    );

    #### Produce covariance matrix using hessian ####
    if(vcovProduce){
        if(CDF){
            method.args <- list(d=1e-6, r=6);
        }
        else{
            if(any(distribution==c("dnbinom","dlaplace","dalaplace"))){
                method.args <- list(d=1e-6, r=6);
            }
            else{
                method.args <- list(d=1e-4, r=4);
            }
        }

        if(distribution=="dpois"){
            # Produce analytical hessian for Poisson distribution
            vcovMatrix <- matrixXreg[1,] %*% t(matrixXreg[1,]) * mu[1];
            for(i in 2:obsInsample){
                vcovMatrix <- vcovMatrix + matrixXreg[i,] %*% t(matrixXreg[i,]) * mu[i];
            }
        }
        else{
            vcovMatrix <- hessian(CF, B, method.args=method.args,
                                  distribution=distribution, y=y, matrixXreg=matrixXreg);
        }

        if(any(distribution==c("dchisq","dnbinom"))){
            vcovMatrix <- vcovMatrix[-1,-1];
        }

        if(any(is.nan(vcovMatrix))){
            warning(paste0("Something went wrong and we failed to produce the covariance matrix of the parameters.\n",
                           "Obviously, it's not our fault. Probably Russians have hacked your computer...\n",
                           "Try a different distribution maybe?"), call.=FALSE);
            vcovMatrix <- diag(1e+100,nVariables);
        }
        else{
            # See if Choleski works... It sometimes fails, when we don't get to the max of likelihood.
            vcovMatrixTry <- try(chol2inv(chol(vcovMatrix)), silent=TRUE);
            if(class(vcovMatrixTry)=="try-error"){
                warning(paste0("Choleski decomposition of hessian failed, so we had to revert to the simple inversion.\n",
                               "The estimate of the covariance matrix of parameters might be inacurate."),
                        call.=FALSE);
                vcovMatrix <- try(solve(vcovMatrix, diag(nVariables), tol=1e-20), silent=TRUE);
                if(class(vcovMatrix)=="try-error"){
                    warning(paste0("Sorry, but the hessian is singular, so we could not invert it.\n",
                                   "The estimate of the covariance matrix of parameters might be inacurate."),
                            call.=FALSE);
                    vcovMatrix <- diag(1e+100,nVariables);
                }
            }
            else{
                vcovMatrix <- vcovMatrixTry;
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

    if(occurrenceModel){
        mf$subset <- NULL;

        # New data and new response variable
        dataNew <- as.matrix(data);
        y <- as.matrix(dataNew[,all.vars(formula)[1]]);
        ot <- y!=0;
        dataNew[,all.vars(formula)[1]] <- (ot)*1;

        if(!occurrenceProvided){
            occurrence <- alm(formula, dataNew, distribution=occurrence);
        }

        # Corrected fitted (with zeroes, when y=0)
        yFittedNew <- yFitted;
        yFitted <- vector("numeric",length(y));
        yFitted[] <- 0;
        yFitted[ot] <- yFittedNew;

        # Corrected errors (with zeroes, when y=0)
        errorsNew <- errors;
        errors <- vector("numeric",length(y));
        errors[] <- 0;
        errors[ot] <- errorsNew;

        # Correction of the likelihood
        CFValue <- CFValue - occurrence$logLik;

        dataWork <- eval(mf, parent.frame());
    }

    if(any(distribution==c("dchisq","dnbinom"))){
        B <- B[-1];
    }
    else if(distribution=="dalaplace"){
        if(alphaEstimate){
            variablesNames <- variablesNames[-1];
            nVariables <- nVariables - 1;
        }
    }

    finalModel <- list(coefficients=B, vcov=vcovMatrix, fitted.values=yFitted, residuals=as.vector(errors),
                       mu=mu, scale=scale, distribution=distribution, logLik=-CFValue,
                       df.residual=obsInsample-df, df=df, call=cl, rank=df,
                       data=matrix(as.matrix(dataWork[,c(responseName,variablesNames[-1])]), ncol=nVariables,
                                   dimnames=list(NULL, c(responseName,variablesNames[-1]))),
                       occurrence=occurrence, subset=subset, other=ellipsis);
    return(structure(finalModel,class=c("alm","greybox")));
}
