#' Advanced Linear Model
#'
#' Function estimates model based on the selected distribution
#'
#' This is a function, similar to \link[stats]{lm}, but for the cases of several
#' non-normal distributions. These include:
#' \enumerate{
#' \item Normal distribution, \link[stats]{dnorm},
#' \item Logistic Distribution, \link[stats]{dlogis},
#' \item Laplace distribution, \link[greybox]{dlaplace},
#' \item Asymmetric Laplace distribution, \link[greybox]{dalaplace},
#' \item T-distribution, \link[stats]{dt},
#' \item S-distribution, \link[greybox]{ds},
#' \item Folded normal distribution, \link[greybox]{dfnorm},
#' \item Log normal distribution, \link[stats]{dlnorm},
#' \item Chi-Squared Distribution, \link[stats]{dchisq},
#' \item Beta distribution, \link[stats]{dbeta},
#' \item Poisson Distribution, \link[stats]{dpois},
#' \item Negative Binomial Distribution, \link[stats]{dnbinom},
#' \item Cumulative Logistic Distribution, \link[stats]{plogis},
#' \item Cumulative Normal distribution, \link[stats]{pnorm}.
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
#' ALM function currently does not work with factors and does not accept
#' transformations of variables in the formula. So you need to do transformations
#' separately before using the function.
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
#' @param checks if \code{FALSE}, then the function won't check whether
#' the data has variability and whether the regressors are correlated. Might
#' cause trouble, especially in cases of multicollinearity.
#' @param ... additional parameters to pass to distribution functions. This
#' includes: \code{alpha} value for Asymmetric Laplace distribution,
#' \code{size} for the Negative Binomial or \code{df} for the Chi-Squared.
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
#' @seealso \code{\link[greybox]{stepwise}, \link[greybox]{lmCombine},
#' \link[greybox]{xregTransformer}}
#'
#' @examples
#'
#' ### An example with mtcars data and factors
#' mtcars2 <- within(mtcars, {
#'    vs <- factor(vs, labels = c("V", "S"))
#'    am <- factor(am, labels = c("automatic", "manual"))
#'    cyl  <- ordered(cyl)
#'    gear <- ordered(gear)
#'    carb <- ordered(carb)
#' })
#' # The standard model with Log Normal distribution
#' ourModel <- alm(mpg~., mtcars2[1:30,], distribution="dlnorm")
#' summary(ourModel)
#' plot(ourModel)
#'
#' # Produce predictions with the one sided interval (upper bound)
#' predict(ourModel, mtcars2[-c(1:30),], interval="p", side="u")
#'
#'
#' ### Artificial data for the other examples
#' xreg <- cbind(rlaplace(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rlaplace(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#' inSample <- xreg[1:80,]
#' outSample <- xreg[-c(1:80),]
#'
#' # An example with Laplace distribution
#' ourModel <- alm(y~x1+x2, inSample, distribution="dlaplace")
#' summary(ourModel)
#' plot(predict(ourModel,outSample))
#'
#' # And another one with Asymmetric Laplace distribution (quantile regression)
#' # with optimised alpha
#' ourModel <- alm(y~x1+x2, inSample, distribution="dalaplace")
#' summary(ourModel)
#' plot(predict(ourModel,outSample))
#'
#'
#' ### Examples with the count data
#' xreg[,1] <- round(exp(xreg[,1]-70),0)
#' inSample <- xreg[1:80,]
#' outSample <- xreg[-c(1:80),]
#'
#' # Negative Binomial distribution
#' ourModel <- alm(y~x1+x2, inSample, distribution="dnbinom")
#' summary(ourModel)
#' predict(ourModel,outSample,interval="p",side="u")
#'
#' # Poisson distribution
#' ourModel <- alm(y~x1+x2, inSample, distribution="dpois")
#' summary(ourModel)
#' predict(ourModel,outSample,interval="p",side="u")
#'
#'
#' ### Examples with binary response variable
#' xreg[,1] <- round(xreg[,1] / (1 + xreg[,1]),0)
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
#' @importFrom stats model.frame sd terms model.matrix
#' @importFrom stats dchisq dlnorm dnorm dlogis dpois dnbinom dt dbeta
#' @importFrom stats plogis
#' @export alm
alm <- function(formula, data, subset, na.action,
                distribution=c("dnorm","dlogis","dlaplace","dalaplace","ds","dt",
                               "dfnorm","dlnorm","dchisq",
                               "dpois","dnbinom",
                               "dbeta",
                               "plogis","pnorm"),
                occurrence=c("none","plogis","pnorm"),
                B=NULL, vcovProduce=FALSE, checks=TRUE, ...){
# Useful stuff for dnbinom: https://scialert.net/fulltext/?doi=ajms.2010.1.15

    cl <- match.call();

    distribution <- distribution[1];
    if(all(distribution!=c("dnorm","dlogis","dlaplace","dalaplace","ds","dt","dfnorm","dlnorm","dchisq",
                           "dpois","dnbinom","dbeta","plogis","pnorm"))){
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
            ellipsis$alpha <- 0.5
            aParameterProvided <- FALSE;
        }
        else{
            alpha <- ellipsis$alpha;
            aParameterProvided <- TRUE;
        }
    }
    else if(distribution=="dchisq"){
        if(is.null(ellipsis$df)){
            aParameterProvided <- FALSE;
        }
        else{
            df <- ellipsis$df;
            aParameterProvided <- TRUE;
        }
    }
    else if(distribution=="dnbinom"){
        if(is.null(ellipsis$size)){
            aParameterProvided <- FALSE;
        }
        else{
            size <- ellipsis$size;
            aParameterProvided <- TRUE;
        }
    }
    else if(distribution=="dfnorm"){
        if(is.null(ellipsis$sigma)){
            aParameterProvided <- FALSE;
        }
        else{
            sigma <- ellipsis$sigma;
            aParameterProvided <- TRUE;
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
    }
    else{
        dataOrders <- unlist(lapply(data,is.ordered));
        # If there is an ordered factor, remove the bloody ordering!
        if(any(dataOrders)){
            data[dataOrders] <- lapply(data[dataOrders],function(x) factor(x, levels=levels(x), ordered=FALSE));
        }
    }
    mf$data <- data;

    responseName <- all.vars(formula)[1];
    # If this is a model with occurrence, use only non-zero observations
    if(occurrenceModel){
        occurrenceNonZero <- data[,responseName]!=0;
        mf$subset <- occurrenceNonZero;
    }

    dataWork <- eval(mf, parent.frame());
    y <- dataWork[,1];

    interceptIsNeeded <- attr(terms(dataWork),"intercept")!=0;
    # Create a model from the provided stuff. This way we can work with factors
    dataWork <- model.matrix(dataWork,data=dataWork);
    obsInsample <- nrow(dataWork);

    if(interceptIsNeeded){
        variablesNames <- colnames(dataWork)[-1];
        matrixXreg <- as.matrix(dataWork[,-1]);
        # Include response to the data
        dataWork <- cbind(y,dataWork[,-1]);
    }
    else{
        variablesNames <- colnames(dataWork);
        matrixXreg <- dataWork;
        # Include response to the data
        dataWork <- cbind(y,dataWork);
        warning(paste0("You have asked not to include intercept in the model. We will try to fit the model, ",
                      "but this is a very naughty thing to do, and we cannot guarantee that it will work..."), call.=FALSE);
    }
    colnames(dataWork) <- c(responseName, variablesNames);

    nVariables <- length(variablesNames);
    colnames(matrixXreg) <- variablesNames;

    # Record the subset used in the model
    if(is.null(mf$subset)){
        subset <- rep(TRUE, obsInsample);
    }
    else{
        subset <- mf$subset;
    }

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

    if(distribution=="dbeta"){
        if(any((y>1) | (y<0))){
            stop("The response variable should lie between 0 and 1 in the Beta distribution", call.=FALSE);
        }
        else if(any(y==c(0,1))){
            warning(paste0("The response variable contains boundary values (either zero or one). ",
                           "Beta distribution is not estimable in this case. ",
                           "So we used a minor correction for it, in order to overcome this limitation."), call.=FALSE);
            y <- y*(1-2*1e-10);
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

    if(checks){
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

        # Do these checks only when intercept is needed. Otherwise in case of dummies this might cause chaos
        if(nVariables>1 & interceptIsNeeded){
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
            if(!aParameterProvided){
                other <- B[1];
                B <- B[-1];
            }
            else{
                other <- alpha;
            }
        }
        else if(distribution=="dnbinom"){
            if(!aParameterProvided){
                other <- B[1];
                B <- B[-1];
            }
            else{
                other <- size;
            }
        }
        else if(distribution=="dchisq"){
            if(!aParameterProvided){
                other <- B[1];
                B <- B[-1];
            }
            else{
                other <- df;
            }
        }
        else if(distribution=="dfnorm"){
            if(!aParameterProvided){
                other <- B[1];
                B <- B[-1];
            }
            else{
                other <- sigma;
            }
        }
        else{
            other <- NULL;
        }

        mu[] <- switch(distribution,
                       "dpois" =,
                       "dnbinom" = exp(matrixXreg %*% B),
                       "dchisq" = ifelseFast(any(matrixXreg %*% B <0),1E+100,(matrixXreg %*% B)^2),
                       "dbeta" = exp(matrixXreg %*% B[1:(length(B)/2)]),
                       "dnorm" =,
                       "dfnorm" =,
                       "dlnorm" =,
                       "dlaplace" =,
                       "dalaplace" =,
                       "dlogis" =,
                       "dt" =,
                       "ds" =,
                       "pnorm" =,
                       "plogis" = matrixXreg %*% B
        );

        scale <- switch(distribution,
                        "dbeta" = exp(matrixXreg %*% B[-c(1:(length(B)/2))]),
                        "dnorm" = sqrt(meanFast((y-mu)^2)),
                        "dfnorm" = abs(other),
                        "dlnorm" = sqrt(meanFast((log(y)-mu)^2)),
                        "dlaplace" = meanFast(abs(y-mu)),
                        "dalaplace" = meanFast((y-mu) * (other - (y<=mu)*1)),
                        "dlogis" = sqrt(meanFast((y-mu)^2) * 3 / pi^2),
                        "ds" = meanFast(sqrt(abs(y-mu))) / 2,
                        "dt" = max(2,2/(1-(meanFast((y-mu)^2))^{-1})),
                        "dchisq" =,
                        "dnbinom" = abs(other),
                        "dpois" = mu,
                        "pnorm" = sqrt(meanFast(qnorm((y - pnorm(mu, 0, 1) + 1) / 2, 0, 1)^2)),
                        "plogis" = sqrt(meanFast(log((1 + y * (1 + exp(mu))) / (1 + exp(mu) * (2 - y) - y))^2)) # Here we use the proxy from Svetunkov et al. (2018)
        );

        return(list(mu=mu,scale=scale,other=other));
    }

    CF <- function(B, distribution, y, matrixXreg){
        fitterReturn <- fitter(B, distribution, y, matrixXreg);

        CFReturn <- switch(distribution,
                           "dnorm" = dnorm(y, mean=fitterReturn$mu, sd=fitterReturn$scale, log=TRUE),
                           "dfnorm" = dfnorm(y, mu=fitterReturn$mu, sigma=fitterReturn$scale, log=TRUE),
                           "dlnorm" = dlnorm(y, meanlog=fitterReturn$mu, sdlog=fitterReturn$scale, log=TRUE),
                           "dlaplace" = dlaplace(y, mu=fitterReturn$mu, scale=fitterReturn$scale, log=TRUE),
                           "dalaplace" = dalaplace(y, mu=fitterReturn$mu, scale=fitterReturn$scale, alpha=fitterReturn$other, log=TRUE),
                           "dlogis" = dlogis(y, location=fitterReturn$mu, scale=fitterReturn$scale, log=TRUE),
                           "dt" = dt(y-fitterReturn$mu, df=fitterReturn$scale, log=TRUE),
                           "ds" = ds(y, mu=fitterReturn$mu, scale=fitterReturn$scale, log=TRUE),
                           "dchisq" = dchisq(y, df=fitterReturn$scale, ncp=fitterReturn$mu, log=TRUE),
                           "dpois" = dpois(y, lambda=fitterReturn$mu, log=TRUE),
                           "dnbinom" = dnbinom(y, mu=fitterReturn$mu, size=fitterReturn$scale, log=TRUE),
                           "dbeta" = dbeta(y, shape1=fitterReturn$mu, shape2=fitterReturn$scale, log=TRUE),
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
            if(any(y==0)){
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
        else if(distribution=="dbeta"){
            # In Beta we set B to be twice longer, using first half of parameters for shape1, and the second for shape2
            # Transform y, just in case, to make sure that it does not hit boundary values
            B <- .lm.fit(matrixXreg,log(y/(1-y)))$coefficients;
            B <- c(B, -B);
        }
        else if(distribution=="dchisq"){
            B <- .lm.fit(matrixXreg,sqrt(y))$coefficients;
            if(aParameterProvided){
                BLower <- rep(-Inf,length(B));
                BUpper <- rep(Inf,length(B));
            }
            else{
                B <- c(1, B);
                BLower <- c(0,rep(-Inf,length(B)-1));
                BUpper <- rep(Inf,length(B));
            }
        }
        else{
            B <- .lm.fit(matrixXreg,y)$coefficients;
            BLower <- -Inf;
            BUpper <- Inf;
        }

        if(distribution=="dnbinom"){
            if(!aParameterProvided){
                B <- c(var(y), B);
                BLower <- c(0,rep(-Inf,length(B)-1));
                BUpper <- rep(Inf,length(B));
            }
            else{
                BLower <- rep(-Inf,length(B));
                BUpper <- rep(Inf,length(B));
            }
        }
        else if(distribution=="dalaplace"){
            if(!aParameterProvided){
                B <- c(0.5, B);
                BLower <- c(0,rep(-Inf,length(B)-1));
                BUpper <- c(1,rep(Inf,length(B)-1));
            }
            else{
                BLower <- rep(-Inf,length(B));
                BUpper <- rep(Inf,length(B));
            }
        }
        else if(distribution=="dfnorm"){
            B <- c(1,B);
            BLower <- c(0,rep(-Inf,length(B)-1));
            BUpper <- rep(Inf,length(B));
        }
        else{
            BLower <- rep(-Inf,length(B));
            BUpper <- rep(Inf,length(B));
        }

        if(any(distribution==c("dchisq","dpois","dnbinom","plogis","pnorm"))){
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
        if(!aParameterProvided){
            ellipsis$size <- B[1];
            B <- B[-1];
        }
        names(B) <- c(variablesNames);
    }
    else if(distribution==c("dchisq")){
        if(!aParameterProvided){
            ellipsis$df <- df <- abs(B[1]);
            B <- B[-1];
        }
        names(B) <- c(variablesNames);
    }
    else if(distribution==c("dfnorm")){
        if(!aParameterProvided){
            ellipsis$sigma <- sigma <- abs(B[1]);
            B <- B[-1];
        }
        names(B) <- c(variablesNames);
    }
    else if(distribution=="dalaplace"){
        if(!aParameterProvided){
            ellipsis$alpha <- alpha <- B[1];
            B <- B[-1];
        }
        names(B) <- variablesNames;
    }
    else if(distribution=="dbeta"){
        if(!vcovProduce){
            names(B) <- c(paste0("shape1_",variablesNames),paste0("shape2_",variablesNames));
        }
    }
    else{
        names(B) <- variablesNames;
    }

    # Parameters of the model + scale
    nParam <- nVariables + 1;

    if(distribution=="dalaplace"){
        if(!aParameterProvided){
            nParam <- nParam + 1;
        }
    }
    else if(any(distribution==c("dnbinom","dchisq","dfnorm"))){
        if(aParameterProvided){
            nParam <- nParam - 1;
        }
    }
    else if(distribution=="dbeta"){
        nParam <- nVariables*2;
        if(!vcovProduce){
            nVariables <- nVariables*2;
        }
    }

    ### Fitted values in the scale of the original variable
    yFitted[] <- switch(distribution,
                       "dfnorm" = sqrt(2/pi)*scale*exp(-mu^2/(2*scale^2))+mu*(1-2*pnorm(-mu/scale)),
                       "dnorm" =,
                       "dlaplace" =,
                       "dalaplace" =,
                       "dlogis" =,
                       "dt" =,
                       "ds" =,
                       "dpois" =,
                       "dnbinom" = mu,
                       "dchisq" = mu + df,
                       "dlnorm" = exp(mu),
                       "dbeta" = mu / (mu + scale),
                       "pnorm" = pnorm(mu, mean=0, sd=1),
                       "plogis" = plogis(mu, location=0, scale=1)
    );

    ### Error term in the transformed scale
    errors[] <- switch(distribution,
                       "dbeta" = y - yFitted,
                       "dfnorm" =,
                       "dlaplace" =,
                       "dalaplace" =,
                       "dlogis" =,
                       "dt" =,
                       "ds" =,
                       "dnorm" =,
                       "dpois" =,
                       "dnbinom" = y - mu,
                       "dchisq" = sqrt(y) - sqrt(mu),
                       "dlnorm"= log(y) - mu,
                       "pnorm" = qnorm((y - pnorm(mu, 0, 1) + 1) / 2, 0, 1),
                       "plogis" = log((1 + y * (1 + exp(mu))) / (1 + exp(mu) * (2 - y) - y)) # Here we use the proxy from Svetunkov et al. (2018)
    );

    # If we had huge numbers for cumulative models, fix errors and scale
    if(any(distribution==c("plogis","pnorm")) & any(is.nan(errors))){
        errorsNaN <- is.nan(errors);

        # Demand occurrs and we predict that
        errors[y==1 & yFitted>0 & errorsNaN] <- 0;
        # Demand occurrs, but we did not predict that
        errors[y==1 & yFitted<0 & errorsNaN] <- 1E+100;
        # Demand does not occurr, and we predict that
        errors[y==0 & yFitted<0 & errorsNaN] <- 0;
        # Demand does not occurr, but we did not predict that
        errors[y==0 & yFitted>0 & errorsNaN] <- -1E+100;

        # Recalculate scale
        scale <- sqrt(meanFast(errors^2));
    }

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

        # if(any(distribution==c("dchisq","dnbinom"))){
        #     vcovMatrix <- vcovMatrix[-1,-1];
        # }

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
                                   "We failed to produce the covariance matrix of parameters."),
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
            if(distribution=="dbeta"){
                dimnames(vcovMatrix) <- list(c(paste0("shape1_",variablesNames),paste0("shape2_",variablesNames)),
                                             c(paste0("shape1_",variablesNames),paste0("shape2_",variablesNames)));
            }
            else{
                dimnames(vcovMatrix) <- list(variablesNames,variablesNames);
            }
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
        dataNew <- data;
        y <- as.matrix(dataNew[,all.vars(formula)[1]]);
        ot <- y!=0;
        dataNew[,all.vars(formula)[1]] <- (ot)*1;

        if(!occurrenceProvided){
            occurrence <- do.call("alm", list(formula=formula, data=dataNew, distribution=occurrence));
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

        # Form the final dataWork in order to return it in the data.
        dataWork <- eval(mf, parent.frame());
        dataWork <- model.matrix(dataWork,data=dataWork);
        if(interceptIsNeeded){
            dataWork <- cbind(y,dataWork[,-1]);
            variablesUsed <- variablesNames[variablesNames!="(Intercept)"];
        }
        else{
            dataWork <- cbind(y,dataWork);
            variablesUsed <- variablesNames;
        }
        colnames(dataWork)[1] <- responseName;
        dataWork <- dataWork[,c(responseName, variablesUsed)]
        if(!is.matrix(dataWork)){
            dataWork <- matrix(dataWork,ncol=1,dimnames=list(names(dataWork),responseName));
        }
    }

    if(distribution=="dbeta"){
        variablesNames <- variablesNames[1:(nVariables/2)];
        nVariables <- nVariables/2;

        # Write down shape parameters
        ellipsis$shape1 <- mu;
        ellipsis$shape2 <- scale;
        # mu and scale are set top be equal to mu and variance.
        scale <- mu * scale / ((mu+scale)^2 * (mu + scale + 1));
        mu <- yFitted;
    }

    finalModel <- list(coefficients=B, vcov=vcovMatrix, fitted.values=yFitted, residuals=as.vector(errors),
                       mu=mu, scale=scale, distribution=distribution, logLik=-CFValue,
                       df.residual=obsInsample-nParam, df=nParam, call=cl, rank=nParam,
                       data=dataWork,
                       occurrence=occurrence, subset=subset, other=ellipsis);

    return(structure(finalModel,class=c("alm","greybox")));
}
