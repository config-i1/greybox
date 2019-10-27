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
#' See more details and examples in the vignette "ALM":
#' \code{vignette("alm","greybox")}
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
#' @param ar the order of AR to include in the model. Only non-seasonal
#' orders are accepted.
#' @param i the order of I to include in the model. Only non-seasonal
#' orders are accepted.
#' @param parameters vector of parameters of the linear model. When \code{NULL}, it
#' is estimated.
#' @param vcovProduce whether to produce variance-covariance matrix of
#' coefficients or not. This is done via hessian calculation, so might be
#' computationally costly.
#' @param fast if \code{FALSE}, then the function won't check whether
#' the data has variability and whether the regressors are correlated. Might
#' cause trouble, especially in cases of multicollinearity.
#' @param ... additional parameters to pass to distribution functions. This
#' includes: \code{alpha} value for Asymmetric Laplace distribution,
#' \code{size} for the Negative Binomial or \code{df} for the Chi-Squared and
#' Student's t. You can also pass two parameters to the optimiser: 1.
#' \code{maxeval} - maximum number of evaluations to carry out (default is
#' 100); 2. \code{xtol_rel} - the precision of the optimiser (the default is
#' 1E-6); 3. \code{algorithm} - the algorithm to use in optimisation
#' (\code{"NLOPT_LN_SBPLX"} by default). 4. \code{print_level} - the level of
#' output for the optimiser (0 by default). You can read more about these
#' parameters in the documentation of \link[nloptr]{nloptr} function.
#'
#' @return Function returns \code{model} - the final model of the class
#' "alm", which contains:
#' \itemize{
#' \item coefficients - estimated parameters of the model,
#' \item vcov - covariance matrix of parameters of the model (based on Fisher
#' Information). Returned only when \code{vcovProduce=TRUE},
#' \item fitted - fitted values,
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
#' # An example with AR(1) order
#' ourModel <- alm(y~x1+x2, inSample, distribution="dnorm", ar=1)
#' summary(ourModel)
#' plot(predict(ourModel,outSample))
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
#' @importFrom forecast Arima
#' @export alm
alm <- function(formula, data, subset, na.action,
                distribution=c("dnorm","dlogis","dlaplace","dalaplace","ds","dt",
                               "dfnorm","dlnorm","dchisq","dbcnorm",
                               "dpois","dnbinom",
                               "dbeta",
                               "plogis","pnorm"),
                occurrence=c("none","plogis","pnorm"),
                ar=0, i=0,
                parameters=NULL, vcovProduce=FALSE, fast=TRUE, ...){
# Useful stuff for dnbinom: https://scialert.net/fulltext/?doi=ajms.2010.1.15

    cl <- match.call();

    #### This is temporary and needs to be removed at some point! ####
    B <- depricator(parameters, list(...));
    fast <- depricator(fast, list(...));

    distribution <- distribution[1];
    if(all(distribution!=c("dnorm","dlogis","dlaplace","dalaplace","ds","dt","dfnorm","dlnorm","dchisq","dbcnorm",
                           "dpois","dnbinom","dbeta","plogis","pnorm"))){
        if(any(distribution==c("norm","fnorm","lnorm","laplace","s","chisq","logis"))){
            warning(paste0("You are using the old value of the distribution parameter.\n",
                           "Use distribution='d",distribution,"' instead."),
                    call.=FALSE);
            distribution <- paste0("d",distribution);
        }
        else{
            stop(paste0("Sorry, but the distribution '",distribution,
                        "' is not yet supported"), call.=FALSE);
        }
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

    # Function for the Box-Cox transform
    bcTransform <- function(y, lambda){
        if(lambda==0){
            return(log(y));
        }
        else{
            return((y^lambda-1)/lambda);
        }
    }

    # Function for the inverse Box-Cox transform
    bcTransformInv <- function(y, lambda){
        if(lambda==0){
            return(exp(y));
        }
        else{
            return((y*lambda+1)^{1/lambda});
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
        else if(distribution=="dbcnorm"){
            if(!aParameterProvided){
                other <- B[1];
                B <- B[-1];
            }
            else{
                other <- lambda;
            }
        }
        else if(distribution=="dt"){
            if(!aParameterProvided){
                other <- B[1];
                B <- B[-1];
            }
            else{
                other <- df;
            }
        }
        else{
            other <- NULL;
        }

        # If there is ARI, then calculate polynomials
        if(all(c(arOrder,iOrder)>0)){
            poly1[-1] <- -B[(nVariablesExo+1):nVariables];
            # This condition is needed for cases of only ARI models
            if(nVariables>arOrder){
                B <- c(B[1:nVariablesExo], -polyprod(poly2,poly1)[-1]);
            }
            else{
                B <- -polyprod(poly2,poly1)[-1];
            }
        }
        else if(iOrder>0){
            B <- c(B, -poly2[-1]);
        }
        else if(arOrder>0){
            poly1[-1] <- -B[(nVariablesExo+1):nVariables];
        }

        mu[] <- switch(distribution,
                       "dpois" =,
                       "dnbinom" = exp(matrixXreg %*% B),
                       "dchisq" = ifelseFast(any(matrixXreg %*% B <0),1E+100,(matrixXreg %*% B)^2),
                       "dbeta" = exp(matrixXreg %*% B[1:(length(B)/2)]),
                       "dbcnorm"=,
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
                        "dnorm" = sqrt(sum((y[otU]-mu[otU])^2)/obsInsample),
                        "dfnorm" = abs(other),
                        "dlnorm" = sqrt(sum((log(y[otU])-mu[otU])^2)/obsInsample),
                        "dbcnorm" = sqrt(sum((bcTransform(y[otU],other)-mu[otU])^2)/obsInsample),
                        "dlaplace" = sum(abs(y[otU]-mu[otU]))/obsInsample,
                        "dalaplace" = sum((y[otU]-mu[otU]) * (other - (y[otU]<=mu[otU])*1))/obsInsample,
                        "dlogis" = sqrt(sum((y[otU]-mu[otU])^2)/obsInsample * 3 / pi^2),
                        "ds" = sum(sqrt(abs(y[otU]-mu[otU]))) / (obsInsample*2),
                        "dt" = ,
                        "dchisq" =,
                        "dnbinom" = abs(other),
                        "dpois" = mu[otU],
                        "pnorm" = sqrt(meanFast(qnorm((y - pnorm(mu, 0, 1) + 1) / 2, 0, 1)^2)),
                        # Here we use the proxy from Svetunkov & Boylan (2019)
                        "plogis" = sqrt(meanFast(log((1 + y * (1 + exp(mu))) /
                                                         (1 + exp(mu) * (2 - y) - y))^2))
        );

        return(list(mu=mu,scale=scale,other=other,poly1=poly1));
    }

    fitterRecursive <- function(B, distribution, y, matrixXreg){
        fitterReturn <- fitter(B, distribution, y, matrixXreg);
        for(i in 1:ariOrder){
            matrixXreg[ariZeroes[,i],nVariablesExo+i] <- switch(distribution,
                                                                "dnbinom" =,
                                                                "dpois" =,
                                                                "dbeta" = log(fitterReturn$mu),
                                                                fitterReturn$mu)[!otU][1:ariZeroesLengths[i]];
            if(distribution=="dbcnorm"){
                matrixXreg[,nVariablesExo+i][!ariZeroes[,i]] <- bcTransform(ariElementsOriginal[!ariZeroes[,i],i],
                                                                            fitterReturn$other);
            }
        }
        # matrixXreg <- fitterRecursion(matrixXreg, B, y, ariZeroes, nVariablesExo, distribution);
        fitterReturn <- fitter(B, distribution, y, matrixXreg);
        fitterReturn$matrixXreg <- matrixXreg;
        return(fitterReturn);
    }

    CF <- function(B, distribution, y, matrixXreg, recursiveModel){
        if(recursiveModel){
            fitterReturn <- fitterRecursive(B, distribution, y, matrixXreg);
        }
        else{
            fitterReturn <- fitter(B, distribution, y, matrixXreg);
        }

        # The original log-likelilhood
        CFReturn <- -sum(switch(distribution,
                                "dnorm" = dnorm(y[otU], mean=fitterReturn$mu[otU], sd=fitterReturn$scale, log=TRUE),
                                "dfnorm" = dfnorm(y[otU], mu=fitterReturn$mu[otU], sigma=fitterReturn$scale, log=TRUE),
                                "dlnorm" = dlnorm(y[otU], meanlog=fitterReturn$mu[otU], sdlog=fitterReturn$scale, log=TRUE),
                                "dbcnorm" = dbcnorm(y[otU], mu=fitterReturn$mu[otU], sigma=fitterReturn$scale,
                                                    lambda=fitterReturn$other, log=TRUE),
                                "dlaplace" = dlaplace(y[otU], mu=fitterReturn$mu[otU], scale=fitterReturn$scale, log=TRUE),
                                "dalaplace" = dalaplace(y[otU], mu=fitterReturn$mu[otU], scale=fitterReturn$scale,
                                                        alpha=fitterReturn$other, log=TRUE),
                                "dlogis" = dlogis(y[otU], location=fitterReturn$mu[otU], scale=fitterReturn$scale, log=TRUE),
                                "dt" = dt(y[otU]-fitterReturn$mu[otU], df=fitterReturn$scale, log=TRUE),
                                "ds" = ds(y[otU], mu=fitterReturn$mu[otU], scale=fitterReturn$scale, log=TRUE),
                                "dchisq" = dchisq(y[otU], df=fitterReturn$scale, ncp=fitterReturn$mu[otU], log=TRUE),
                                "dpois" = dpois(y[otU], lambda=fitterReturn$mu[otU], log=TRUE),
                                "dnbinom" = dnbinom(y[otU], mu=fitterReturn$mu[otU], size=fitterReturn$scale, log=TRUE),
                                "dbeta" = dbeta(y[otU], shape1=fitterReturn$mu[otU], shape2=fitterReturn$scale[otU], log=TRUE),
                                "pnorm" = c(pnorm(fitterReturn$mu[ot], mean=0, sd=1, log.p=TRUE),
                                            pnorm(fitterReturn$mu[!ot], mean=0, sd=1, lower.tail=FALSE, log.p=TRUE)),
                                "plogis" = c(plogis(fitterReturn$mu[ot], location=0, scale=1, log.p=TRUE),
                                             plogis(fitterReturn$mu[!ot], location=0, scale=1, lower.tail=FALSE, log.p=TRUE))
        ));

        # The differential entropy for the models with the missing data
        if(occurrenceModel){
            CFReturn[] <- CFReturn + switch(distribution,
                                            "dnorm" =,
                                            "dfnorm" =,
                                            "dbcnorm" =,
                                            "dlnorm" = obsZero*(log(sqrt(2*pi)*fitterReturn$scale)+0.5),
                                            "dlaplace" =,
                                            "dalaplace" = obsZero*(1 + log(2*fitterReturn$scale)),
                                            "dlogis" = obsZero*2,
                                            "dt" = obsZero*((fitterReturn$scale+1)/2 *
                                                                (digamma((fitterReturn$scale+1)/2)-digamma(fitterReturn$scale/2)) +
                                                                log(sqrt(fitterReturn$scale) * beta(fitterReturn$scale/2,0.5))),
                                            "ds" = obsZero*(2 + 2*log(2*fitterReturn$scale)),
                                            "dchisq" = obsZero*(log(2)*gamma(fitterReturn$scale/2)-
                                                                    (1-fitterReturn$scale/2)*digamma(fitterReturn$scale/2)+
                                                                    fitterReturn$scale/2),
                                            "dbeta" = sum(log(beta(fitterReturn$mu[otU],fitterReturn$scale[otU]))-
                                                              (fitterReturn$mu[otU]-1)*
                                                              (digamma(fitterReturn$mu[otU])-
                                                                   digamma(fitterReturn$mu[otU]+fitterReturn$scale[otU]))-
                                                              (fitterReturn$scale[otU]-1)*
                                                              (digamma(fitterReturn$scale[otU])-
                                                                   digamma(fitterReturn$mu[otU]+fitterReturn$scale[otU]))),
                                            # This is a normal approximation of the real entropy
                                            # "dpois" = sum(0.5*log(2*pi*fitterReturn$scale)+0.5),
                                            # "dnbinom" = obsZero*(log(sqrt(2*pi)*fitterReturn$scale)+0.5),
                                            0
            );
        }

        if(is.nan(CFReturn) | is.na(CFReturn) | is.infinite(CFReturn)){
            CFReturn[] <- 1E+300;
        }

        # Check the roots of polynomials
        if(arOrder>0 && any(abs(polyroot(fitterReturn$poly1))<1)){
            CFReturn[] <- CFReturn / min(abs(polyroot(fitterReturn$poly1)));
        }

        return(CFReturn);
    }

    #### Define the rest of parameters ####
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
    else if(distribution=="dbcnorm"){
        if(is.null(ellipsis$lambda)){
            aParameterProvided <- FALSE;
        }
        else{
            lambda <- ellipsis$lambda;
            aParameterProvided <- TRUE;
        }
    }
    else if(distribution=="dt"){
        if(is.null(ellipsis$df)){
            aParameterProvided <- FALSE;
        }
        else{
            df <- ellipsis$df;
            aParameterProvided <- TRUE;
        }
    }

    # See if the occurrence model is provided, and whether we need to treat the data as intermittent
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

    arOrder <- ar;
    iOrder <- i;
    # Check AR, I and form ARI order
    if(length(arOrder)>1){
        warning("ar must be a scalar, not a vector. Using the first value.", call.=FALSE);
        arOrder <- arOrder[1];
    }
    if(length(iOrder)>1){
        warning("i must be a scalar, not a vector. Using the first value.", call.=FALSE);
        iOrder <- iOrder[1];
    }
    ariOrder <- arOrder + iOrder;
    ariModel <- ifelseFast(ariOrder>0, TRUE, FALSE);
    # Create polynomials for the i and ar orders
    if(iOrder>0){
        poly2 <- c(1,-1);
        if(iOrder>1){
            for(j in 1:(iOrder-1)){
                poly2 <- polyprod(poly2,c(1,-1));
            }
        }
    }
    if(arOrder>0){
        poly1 <- rep(1,arOrder+1);
    }
    else{
        poly1 <- c(1,1);
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

    # If there are NaN values, remove the respective observations
    if(any(sapply(mf$data,is.nan))){
        warning("There are NaN values in the data. This might cause problems. Removing these observations.", call.=FALSE);
        NonNaNValues <- !apply(sapply(mf$data,is.nan),1,any);
        # If subset was not provided, change it
        if(is.null(mf$subset)){
            mf$subset <- NonNaNValues
        }
        else{
            mf$subset <- NonNaNValues & mf$subset;
        }
        dataContainsNaNs <- TRUE;
    }
    else{
        dataContainsNaNs <- FALSE;
    }

    responseName <- all.vars(formula)[1];

    # Make recursive fitted for missing values in case of occurrence model.
    recursiveModel <- occurrenceModel && ariModel;
    # In case of plogis and pnorm, all the ARI values need to be refitted
    if(any(distribution==c("plogis","pnorm")) && ariModel){
        recursiveModel <- TRUE;
    }

    dataWork <- eval(mf, parent.frame());
    y <- dataWork[,1];

    interceptIsNeeded <- attr(terms(dataWork),"intercept")!=0;
    # Create a model from the provided stuff. This way we can work with factors
    dataWork <- model.matrix(dataWork,data=dataWork);
    obsInsample <- nrow(dataWork);

    if(interceptIsNeeded){
        variablesNames <- colnames(dataWork)[-1];
        matrixXreg <- as.matrix(dataWork[,-1,drop=FALSE]);
        # Include response to the data
        # dataWork <- cbind(y,dataWork[,-1,drop=FALSE]);
    }
    else{
        variablesNames <- colnames(dataWork);
        matrixXreg <- dataWork;
        # Include response to the data
        # dataWork <- cbind(y,dataWork);
        warning(paste0("You have asked not to include intercept in the model. We will try to fit the model, ",
                      "but this is a very naughty thing to do, and we cannot guarantee that it will work..."), call.=FALSE);
    }
    # colnames(dataWork) <- c(responseName, variablesNames);
    rm(dataWork);

    nVariables <- length(variablesNames);
    colnames(matrixXreg) <- variablesNames;

    # Record the subset used in the model
    if(is.null(mf$subset)){
        subset <- rep(TRUE, obsInsample);
    }
    else{
        if(dataContainsNaNs){
            subset <- mf$subset[NonNaNValues];
        }
        else{
            subset <- mf$subset;
        }
    }

    mu <- vector("numeric", obsInsample);
    yFitted <- vector("numeric", obsInsample);
    errors <- vector("numeric", obsInsample);
    ot <- vector("logical", obsInsample);

    if(any(y<0) & any(distribution==c("dfnorm","dlnorm","dbcnorm","dchisq","dpois","dnbinom"))){
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
        y[] <- (y!=0)*1;
    }

    # Vector with logical for the non-zero observations
    if(CDF || occurrenceModel){
        ot[] <- y!=0;
    }
    else{
        ot[] <- rep(TRUE,obsInsample);
    }

    # Set the number of zeroes and non-zeroes, depending on the type of the model
    if(occurrenceModel){
        # otU is the vector of {0,1} for the occurrence model, not for the CDF!
        otU <- ot;
        obsNonZero <- sum(ot);
        obsZero <- sum(!ot);
    }
    else{
        otU <- rep(TRUE,obsInsample);
        obsNonZero <- obsInsample;
        obsZero <- 0;
    }

    if(!fast){
        #### Checks of the exogenous variables ####
        # Remove the data for which sd=0
        noVariability <- vector("logical",nVariables);
        noVariability[] <- apply((matrixXreg==matrix(matrixXreg[1,],obsInsample,nVariables,byrow=TRUE))[otU,,drop=FALSE],2,all);
        if(any(noVariability)){
            if(all(noVariability)){
                warning("None of exogenous variables has variability. Fitting the straight line.",
                        call.=FALSE);
                matrixXreg <- matrix(1,obsInsample,1);
                nVariables <- 1;
                variablesNames <- "(Intercept)";
            }
            else{
                if(!occurrenceModel && !CDF){
                    warning("Some exogenous variables did not have any variability. We dropped them out.",
                            call.=FALSE);
                }
                matrixXreg <- matrixXreg[,!noVariability,drop=FALSE];
                nVariables <- ncol(matrixXreg);
                variablesNames <- variablesNames[!noVariability];
            }
        }

        corThreshold <- 0.999;
        if(nVariables>1){
            # Check perfectly correlated cases
            corMatrix <- cor(matrixXreg[otU,,drop=FALSE]);
            corHigh <- upper.tri(corMatrix) & abs(corMatrix)>=corThreshold;
            if(any(corHigh)){
                removexreg <- unique(which(corHigh,arr.ind=TRUE)[,1]);
                matrixXreg <- matrixXreg[,-removexreg,drop=FALSE];
                nVariables <- ncol(matrixXreg);
                variablesNames <- colnames(matrixXreg);
                if(!occurrenceModel && !CDF){
                    warning("Some exogenous variables were perfectly correlated. We've dropped them out.",
                            call.=FALSE);
                }
            }
        }

        # Do these checks only when intercept is needed. Otherwise in case of dummies this might cause chaos
        if(nVariables>1 & interceptIsNeeded){
            # Check dummy variables trap
            detHigh <- determination(matrixXreg[otU,,drop=FALSE])>=corThreshold;
            if(any(detHigh)){
                while(any(detHigh)){
                    removexreg <- which(detHigh>=corThreshold)[1];
                    matrixXreg <- matrixXreg[,-removexreg,drop=FALSE];
                    nVariables <- ncol(matrixXreg);
                    variablesNames <- colnames(matrixXreg);

                    detHigh <- determination(matrixXreg)>=corThreshold;
                }
                if(!occurrenceModel){
                    warning("Some combinations of exogenous variables were perfectly correlated. We've dropped them out.",
                            call.=FALSE);
                }
            }
        }
    }

    #### Finish forming the matrix of exogenous variables ####
    # Remove the redudant dummies, if there are any
    varsToLeave <- apply(matrixXreg[otU,,drop=FALSE],2,var)!=0;
    matrixXreg <- matrixXreg[,varsToLeave,drop=FALSE];
    variablesNames <- variablesNames[varsToLeave];
    nVariables <- length(variablesNames);

    if(interceptIsNeeded){
        matrixXreg <- cbind(1,matrixXreg);
        variablesNames <- c("(Intercept)",variablesNames);
        colnames(matrixXreg) <- variablesNames;

        if(is.null(parameters)){
            # Check, if redundant dummies are left. Remove the first if this is the case
            determValues <- determination(matrixXreg[otU, -1, drop=FALSE]);
            if(any(determValues==1)){
                matrixXreg <- matrixXreg[,-(which(determValues==1)[1]+1),drop=FALSE];
                variablesNames <- colnames(matrixXreg);
            }
            nVariables <- length(variablesNames);
        }
    }
    # The number of exogenous variables (no ARI elements)
    nVariablesExo <- nVariables;

    #### Estimate parameters of the model ####
    if(is.null(parameters)){
        #### Add AR and I elements in the regression ####
        # This is only done, if the regression is estimated. In the other cases it will already have the expanded values
        if(ariModel){
            # In case of plogis and pnorm, the AR elements need to be generated from a model, i.e. oes from smooth.
            if(any(distribution==c("plogis","pnorm"))){
                if(!requireNamespace("smooth", quietly = TRUE)){
                    yNew <- abs(fitted(Arima(y, order=c(0,1,1))));
                    yNew[is.na(yNew)] <- min(yNew);
                    yNew[yNew==0] <- 1E-10;
                    yNew[] <- log(yNew / (1-yNew));
                    yNew[is.infinite(yNew) & yNew>0] <- max(yNew[is.finite(yNew)]);
                    yNew[is.infinite(yNew) & yNew<0] <- min(yNew[is.finite(yNew)]);
                }
                else{
                    yNew <- smooth::oes(y, occurrence="i", model="MNN", h=1)$fittedModel
                }
                ariElements <- xregExpander(yNew, lags=-c(1:ariOrder), gaps="auto")[,-1,drop=FALSE];
                ariZeroes <- matrix(TRUE,nrow=obsInsample,ncol=ariOrder);
                for(i in 1:ariOrder){
                    ariZeroes[(1:i),i] <- FALSE;
                }
                ariZeroesLengths <- apply(ariZeroes, 2, sum);
            }
            else{
                ariElements <- xregExpander(y, lags=-c(1:ariOrder), gaps="auto")[,-1,drop=FALSE];
            }

            # Get rid of "ts" class
            class(ariElements) <- "matrix";
            ariNames <- paste0(responseName,"Lag",c(1:ariOrder));
            ariTransformedNames <- ariNames;
            colnames(ariElements) <- ariNames;

            # Non-zero sequencies for the recursion mechanism of ar
            if(occurrenceModel){
                ariZeroes <- ariElements == 0;
                ariZeroesLengths <- apply(ariZeroes, 2, sum);
            }

            if(ar>0){
                arNames <- paste0(responseName,"Lag",c(1:ar));
                variablesNames <- c(variablesNames,arNames);
            }
            else{
                arNames <- vector("character",0);
            }
            nVariables <- nVariables + arOrder;
            # Write down the values for the matrixXreg in the necessary transformations
            if(any(distribution==c("dlnorm","dpois","dnbinom"))){
                if(any(y[otU]==0)){
                    # Use Box-Cox if there are zeroes
                    ariElements[] <- bcTransform(ariElements,0.01);
                    ariTransformedNames <- paste0(ariNames,"Box-Cox");
                    colnames(ariElements) <- ariTransformedNames;
                }
                else{
                    ariElements[ariElements<0] <- 0;
                    ariElements[] <- suppressWarnings(log(ariElements));
                    ariElements[is.infinite(ariElements)] <- 0;
                    ariTransformedNames <- paste0(ariNames,"Log");
                    colnames(ariElements) <- ariTransformedNames;
                }
            }
            else if(distribution=="dbcnorm"){
                ariElementsOriginal <- ariElements;
                ariElements[] <- bcTransform(ariElements,0.1);
                ariTransformedNames <- paste0(ariNames,"Box-Cox");
                colnames(ariElements) <- ariTransformedNames;
            }
            else if(distribution=="dchisq"){
                ariElements[] <- sqrt(ariElements);
                ariTransformedNames <- paste0(ariNames,"Sqrt");
                colnames(ariElements) <- ariTransformedNames;
            }

            # Fill in zeroes with the mean values
            ariElements[ariElements==0] <- mean(ariElements[ariElements[,1]!=0,1]);

            matrixXreg <- cbind(matrixXreg, ariElements);
            # dataWork <- cbind(dataWork, ariElements);
        }

        #### I(0) initialisation ####
        if(iOrder==0){
            if(any(distribution==c("dlnorm","dpois","dnbinom"))){
                if(any(y[otU]==0)){
                    # Use Box-Cox if there are zeroes
                    B <- .lm.fit(matrixXreg[otU,,drop=FALSE],bcTransform(y[otU],0.01))$coefficients;
                }
                else{
                    B <- .lm.fit(matrixXreg[otU,,drop=FALSE],log(y[otU]))$coefficients;
                }
            }
            else if(any(distribution==c("plogis","pnorm"))){
                # Box-Cox transform in order to get meaningful initials
                B <- .lm.fit(matrixXreg,bcTransform(y[otU],0.01))$coefficients;
            }
            else if(distribution=="dbcnorm"){
                if(!aParameterProvided){
                    B <- c(0.1,.lm.fit(matrixXreg[otU,,drop=FALSE],bcTransform(y[otU],0.1))$coefficients);
                }
                else{
                    B <- c(.lm.fit(matrixXreg[otU,,drop=FALSE],bcTransform(y[otU],lambda))$coefficients);
                }
            }
            else if(distribution=="dbeta"){
                # In Beta we set B to be twice longer, using first half of parameters for shape1, and the second for shape2
                # Transform y, just in case, to make sure that it does not hit boundary values
                B <- .lm.fit(matrixXreg[otU,,drop=FALSE],log(y[otU]/(1-y[otU])))$coefficients;
                B <- c(B, -B);
            }
            else if(distribution=="dchisq"){
                B <- .lm.fit(matrixXreg[otU,,drop=FALSE],sqrt(y[otU]))$coefficients;
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
            else if(distribution=="dt"){
                B <- .lm.fit(matrixXreg[otU,,drop=FALSE],y[otU])$coefficients;
                if(aParameterProvided){
                    BLower <- rep(-Inf,length(B));
                    BUpper <- rep(Inf,length(B));
                }
                else{
                    B <- c(2, B);
                    BLower <- c(0,rep(-Inf,length(B)-1));
                    BUpper <- rep(Inf,length(B));
                }
            }
            else{
                B <- .lm.fit(matrixXreg[otU,,drop=FALSE],y[otU])$coefficients;
                BLower <- -Inf;
                BUpper <- Inf;
            }
        }
        #### I(d) initialisation ####
        # If this is an I(d) model, do the primary estimation in differences
        else{
            # Matrix without the first D rows and without the last D columns
            # matrixXregForDiffs <- matrixXreg[otU,-(nVariables+1:iOrder),drop=FALSE][-c(1:iOrder),,drop=FALSE];

            # Use only AR elements of the matrix, take differences for the initialisation purposes
            matrixXregForDiffs <- matrixXreg[otU,-(nVariables+1:iOrder),drop=FALSE];
            if(arOrder>0){
                matrixXregForDiffs[-c(1:iOrder),nVariablesExo+c(1:arOrder)] <- diff(matrixXregForDiffs[,nVariablesExo+c(1:arOrder)],
                                                                                    differences=iOrder);
                matrixXregForDiffs <- matrixXregForDiffs[-c(1:iOrder),,drop=FALSE];
                matrixXregForDiffs[c(1:iOrder),nVariablesExo+c(1:arOrder)] <- colMeans(matrixXregForDiffs[,nVariablesExo+c(1:arOrder), drop=FALSE]);
            }
            else{
                matrixXregForDiffs <- matrixXregForDiffs[-c(1:iOrder),,drop=FALSE];
            }

            if(any(distribution==c("dlnorm","dpois","dnbinom"))){
                B <- .lm.fit(matrixXregForDiffs,diff(log(y[otU]),differences=iOrder))$coefficients;
            }
            else if(any(distribution==c("plogis","pnorm"))){
                # Box-Cox transform in order to get meaningful initials
                B <- .lm.fit(matrixXregForDiffs,diff(bcTransform(y[otU],0.01),differences=iOrder))$coefficients;
            }
            else if(distribution=="dbcnorm"){
                if(!aParameterProvided){
                    B <- c(0.1,.lm.fit(matrixXregForDiffs,diff(bcTransform(y[otU],0.1),differences=iOrder))$coefficients);
                }
                else{
                    B <- c(.lm.fit(matrixXregForDiffs,diff(bcTransform(y[otU],0.1),differences=iOrder))$coefficients);
                }
            }
            else if(distribution=="dbeta"){
                # In Beta we set B to be twice longer, using first half of parameters for shape1, and the second for shape2
                # Transform y, just in case, to make sure that it does not hit boundary values
                B <- .lm.fit(matrixXregForDiffs,diff(log(y/(1-y)),differences=iOrder)[c(1:nrow(matrixXregForDiffs))])$coefficients;
                B <- c(B, -B);
            }
            else if(distribution=="dchisq"){
                B <- .lm.fit(matrixXregForDiffs,diff(sqrt(y[otU]),differences=iOrder))$coefficients;
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
            else if(distribution=="dt"){
                B <- .lm.fit(matrixXregForDiffs,diff(y[otU],differences=iOrder))$coefficients;
                if(aParameterProvided){
                    BLower <- rep(-Inf,length(B));
                    BUpper <- rep(Inf,length(B));
                }
                else{
                    B <- c(2, B);
                    BLower <- c(0,rep(-Inf,length(B)-1));
                    BUpper <- rep(Inf,length(B));
                }
            }
            else{
                B <- .lm.fit(matrixXregForDiffs,diff(y[otU],differences=iOrder))$coefficients;
                BLower <- -Inf;
                BUpper <- Inf;
            }
        }

        if(distribution=="dnbinom"){
            if(!aParameterProvided){
                B <- c(var(y[otU]), B);
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
            B <- c(sd(y),B);
            BLower <- c(0,rep(-Inf,length(B)-1));
            BUpper <- rep(Inf,length(B));
        }
        else if(distribution=="dbcnorm"){
            if(aParameterProvided){
                BLower <- rep(-Inf,length(B));
                BUpper <- rep(Inf,length(B));
            }
            else{
                BLower <- c(0,rep(-Inf,length(B)-1));
                BUpper <- c(1,rep(Inf,length(B)-1));
            }
        }
        else{
            BLower <- rep(-Inf,length(B));
            BUpper <- rep(Inf,length(B));
        }

        # Parameters for the nloptr from the ellipsis
        if(is.null(ellipsis$maxeval)){
            if(any(distribution==c("dchisq","dpois","dnbinom","dbcnorm","plogis","pnorm")) || recursiveModel){
                maxeval <- 500;
            }
            else{
                maxeval <- 100;
            }
        }
        else{
            maxeval <- ellipsis$maxeval;
        }
        if(is.null(ellipsis$xtol_rel)){
            xtol_rel <- 1E-6;
        }
        else{
            xtol_rel <- ellipsis$xtol_rel;
        }
        if(is.null(ellipsis$algorithm)){
            # if(recursiveModel){
                algorithm <- "NLOPT_LN_BOBYQA";
            # }
            # else{
                algorithm <- "NLOPT_LN_SBPLX";
            # }
        }
        else{
            algorithm <- ellipsis$algorithm;
        }
        if(is.null(ellipsis$print_level)){
            print_level <- 0;
        }
        else{
            print_level <- ellipsis$print_level;
        }

        # Change otU to FALSE everywhere, so that the lags are refitted for the occurrence models
        if(any(distribution==c("plogis","pnorm")) && ariModel){
            otU <- rep(FALSE,obsInsample);
        }

        # Although this is not needed in case of distribution="dnorm", we do that in a way, for the code consistency purposes
        res <- nloptr(B, CF,
                      opts=list(algorithm=algorithm, xtol_rel=xtol_rel, maxeval=maxeval, print_level=print_level),
                      lb=BLower, ub=BUpper,
                      distribution=distribution, y=y, matrixXreg=matrixXreg,
                      recursiveModel=recursiveModel);
        if(recursiveModel){
            res2 <- nloptr(res$solution, CF,
                           opts=list(algorithm="NLOPT_LN_SBPLX", xtol_rel=xtol_rel, maxeval=maxeval, print_level=print_level),
                           lb=BLower, ub=BUpper,
                           distribution=distribution, y=y, matrixXreg=matrixXreg,
                           recursiveModel=recursiveModel);
            if(res2$objective<res$objective){
                res[] <- res2;
            }
        }
        B[] <- res$solution;

        CFValue <- res$objective;

        # If there were ARI, write down the polynomial
        if(ariModel){
            # Some models save the first parameter for scale
            nVariablesForReal <- length(B);
            if(all(c(arOrder,iOrder)>0)){
                poly1[-1] <- -B[nVariablesForReal-c(1:arOrder)+1];
                ellipsis$polynomial <- -polyprod(poly2,poly1)[-1];
                ellipsis$arima <- c(arOrder,iOrder,0);
            }
            else if(iOrder>0){
                ellipsis$polynomial <- -poly2[-1];
                ellipsis$arima <- c(0,iOrder,0);
            }
            else{
                ellipsis$polynomial <- B[nVariablesForReal-c(1:arOrder)+1];
                ellipsis$arima <- c(arOrder,0,0);
            }
            names(ellipsis$polynomial) <- ariNames;
            ellipsis$arima <- paste0("ARIMA(",paste0(ellipsis$arima,collapse=","),")");
        }
    }
    else{
        # The data are provided, so no need to do recursive fitting
        recursiveModel <- FALSE;
        # If this was ARI, then don't count the AR parameters
        if(ariModel){
            nVariablesExo <- nVariablesExo - ariOrder;
        }
        nVariables <- length(B);
        variablesNames <- names(B);
        CFValue <- CF(B, distribution, y, matrixXreg, recursiveModel);
    }

    #### Form the fitted values, location and scale ####
    if(recursiveModel){
        fitterReturn <- fitterRecursive(B, distribution, y, matrixXreg);
        matrixXreg[] <- fitterReturn$matrixXreg;
    }
    else{
        fitterReturn <- fitter(B, distribution, y, matrixXreg);
    }
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
    else if(distribution==c("dt")){
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
    else if(distribution=="dbcnorm"){
        if(!aParameterProvided){
            ellipsis$lambda <- lambda <- B[1];
            B <- B[-1];
        }
        names(B) <- variablesNames;
    }
    else if(distribution=="dt"){
        ellipsis$df <- scale;
        names(B) <- variablesNames;
    }
    else{
        names(B) <- variablesNames;
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
                       "dbcnorm" = bcTransformInv(mu,lambda),
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
                       "dnbinom" =,
                       "dpois" = y - mu,
                       "dchisq" = sqrt(y) - sqrt(mu),
                       "dlnorm"= log(y) - mu,
                       "dbcnorm"= bcTransform(y,lambda) - mu,
                       "pnorm" = qnorm((y - pnorm(mu, 0, 1) + 1) / 2, 0, 1),
                       "plogis" = log((1 + y * (1 + exp(mu))) / (1 + exp(mu) * (2 - y) - y))
                       # Here we use the proxy from Svetunkov & Boylan (2019)
    );

    # If this is the occurrence model, then set unobserved errors to zero
    if(occurrenceModel){
        errors[!otU] <- 0;
    }

    # If negative values are produced in the mu of dbcnorm, correct the fit
    if(distribution=="dbcnorm" && any(mu<0)){
        yFitted[mu<0] <- 0;
    }

    # Parameters of the model + scale
    nParam <- nVariables + 1;

    if(distribution=="dalaplace"){
        if(!aParameterProvided){
            nParam <- nParam + 1;
        }
    }
    else if(any(distribution==c("dnbinom","dchisq","dt","dfnorm","dbcnorm"))){
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

    # If we had huge numbers for cumulative models, fix errors and scale
    if(any(distribution==c("plogis","pnorm")) && (any(is.nan(errors)) || any(is.infinite(errors)))){
        errorsNaN <- is.nan(errors) | is.infinite(errors);

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
        # Only vcov is needed, no point in redoing the occurrenceModel
        occurrenceModel <- FALSE;
        method.args <- list(eps=1e-4, d=0.1, r=4)
        # if(CDF){
        #     method.args <- list(d=1e-6, r=6);
        # }
        # else{
        #     if(any(distribution==c("dnbinom","dlaplace","dalaplace","dbcnorm"))){
        #         method.args <- list(d=1e-6, r=6);
        #     }
        #     else{
        #         method.args <- list(d=1e-4, r=4);
        #     }
        # }

        if(distribution=="dpois"){
            # Produce analytical hessian for Poisson distribution
            vcovMatrix <- matrixXreg[1,] %*% t(matrixXreg[1,]) * mu[1];
            for(j in 2:obsInsample){
                vcovMatrix[] <- vcovMatrix + matrixXreg[j,] %*% t(matrixXreg[j,]) * mu[j];
            }
            if(iOrder>0){
                vcovMatrix <- vcovMatrix[1:nVariablesExo,1:nVariablesExo];
            }
        }
        else{
            vcovMatrix <- hessian(CF, B, method.args=method.args,
                                  distribution=distribution, y=y, matrixXreg=matrixXreg,
                                  recursiveModel=recursiveModel);
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
                               "The estimate of the covariance matrix of parameters might be inaccurate."),
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

    #### Deal with the occurrence part of the model ####
    if(occurrenceModel){
        mf$subset <- NULL;

        # If there are NaN values, remove the respective observations
        if(any(sapply(mf$data,is.nan))){
            mf$subset <- !apply(sapply(mf$data,is.nan),1,any);
        }
        else{
            mf$subset <- rep(TRUE,nrow(mf$data));
        }

        # New data and new response variable
        dataNew <- mf$data[mf$subset,,drop=FALSE];
        # if(ncol(dataNew)>1){
        #     y <- as.matrix(dataNew[,all.vars(formula)[1],drop=FALSE]);
        # }
        # else{
        #     y <- dataNew[,1,drop=FALSE]
        # }
        # If there are NaN values, substitute them by zeroes
        # if(any(is.nan(y))){
        #     y[is.nan(y)] <- 0;
        # }
        # ot <- y!=0;
        dataNew[,all.vars(formula)[1]] <- (ot)*1;

        if(!occurrenceProvided){
            occurrence <- do.call("alm", list(formula=formula, data=dataNew, distribution=occurrence, ar=arOrder, i=iOrder));
        }

        # Corrected fitted (with zeroes, when y=0)
        # yFittedNew <- yFitted;
        # yFitted <- vector("numeric",length(y));
        # yFitted[] <- 0;
        yFitted[] <- yFitted * fitted(occurrence);

        # Corrected errors (with zeroes, when y=0)
        # errorsNew <- errors;
        # errors <- vector("numeric",length(y));
        # errors[] <- 0;
        # errors[ot] <- errorsNew;

        # Correction of the likelihood
        CFValue <- CFValue - occurrence$logLik;

        # Form the final dataWork in order to return it in the data.
        # dataWork <- eval(mf, parent.frame());
        # dataWork <- model.matrix(dataWork,data=dataWork);

        # Add AR elements if needed
        # if(ariModel){
        #     ariMatrix <- matrix(0, nrow(dataWork), ariOrder, dimnames=list(NULL, ariTransformedNames));
        #     ariMatrix[] <- ariElements;
        #     dataWork <- cbind(dataWork,ariMatrix);
        # }

        if(interceptIsNeeded){
            # This shit is needed, because R has habit of converting everything into vectors...
            dataWork <- cbind(y,matrixXreg[,-1,drop=FALSE]);
            variablesUsed <- variablesNames[variablesNames!="(Intercept)"];
        }
        else{
            dataWork <- cbind(y,matrixXreg);
            variablesUsed <- variablesNames;
        }

        # Change the names of variables used, if ARI was constructed.
        if(ariModel){
            variablesUsed <- variablesUsed[!(variablesUsed %in% arNames)];
            variablesUsed <- c(variablesUsed,ariTransformedNames);
        }
        colnames(dataWork)[1] <- responseName;
        dataWork <- dataWork[,c(responseName, variablesUsed), drop=FALSE]
    }
    else{
        if(interceptIsNeeded){
            # This shit is needed, because R has habit of converting everything into vectors...
            dataWork <- cbind(y,matrixXreg[,-1,drop=FALSE]);
            variablesUsed <- variablesNames[variablesNames!="(Intercept)"];
        }
        else{
            dataWork <- cbind(y,matrixXreg);
            variablesUsed <- variablesNames;
        }
        colnames(dataWork) <- c(responseName, variablesUsed);
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

    finalModel <- list(coefficients=B, vcov=vcovMatrix, fitted=yFitted, residuals=as.vector(errors),
                       mu=mu, scale=scale, distribution=distribution, logLik=-CFValue,
                       df.residual=obsInsample-nParam, df=nParam, call=cl, rank=nParam,
                       data=dataWork,
                       occurrence=occurrence, subset=subset, other=ellipsis);

    return(structure(finalModel,class=c("alm","greybox")));
}
