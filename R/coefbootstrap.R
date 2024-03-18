#' Bootstrap for parameters of models
#'
#' The function does the bootstrap for parameters of models and returns covariance matrix
#' together with the original bootstrapped data.
#'
#' The function applies the same model as in the provided object on a smaller sample in
#' order to get the estimates of parameters and capture the uncertainty about them. This is
#' a simple implementation of the case resampling, which assumes that the observations are
#' independent.
#'
#' @param object The model estimated using either lm, or alm, or glm.
#' @param nsim Number of iterations (simulations) to run.
#' @param size A non-negative integer giving the number of items to choose (the sample size),
#' passed to \link[base]{sample} function in R. If not provided and model contains ARIMA
#' components, this value will be selected at random on each iteration.
#' @param replace Should sampling be with replacement? Also, passed to \link[base]{sample}
#' function in R.
#' @param prob A vector of probability weights for obtaining the elements of the vector
#' being sampled. This is passed to the \link[base]{sample} as well.
#' @param parallel Either a logical, specifying whether to do the calculations in parallel,
#' or the number, specifying the number of cores to use for the parallel calculation.
#'
#' @return Class "bootstrap" is returned, which contains:
#' \itemize{
#' \item vcov - the covariance matrix of parameters;
#' \item coefficients - the matrix with the bootstrapped coefficients.
#' \item nsim - number of runs done;
#' \item size - the sample size used in the bootstrap;
#' \item replace - whether the sampling was done with replacement;
#' \item prob - a vector of probability weights used in the process;
#' \item parallel - whether the calculations were done in parallel;
#' \item model - the name of the model used (the name of the function);
#' \item timeElapsed - the time that was spend on the calculations.
#' }
#'
#' @template author
#' @template keywords
#'
#' @seealso \code{\link[greybox]{alm}}
#'
#' @examples
#' # An example with ALM
#' ourModel <- alm(mpg~., mtcars, distribution="dlnorm", loss="HAM")
#' # A fast example with 10 iterations. Use at least 1000 to get better results
#' coefbootstrap(ourModel, nsim=10)
#'
#' @rdname coefbootstrap
#' @export
coefbootstrap <- function(object, nsim=1000, size=floor(0.75*nobs(object)),
                          replace=FALSE, prob=NULL, parallel=FALSE) UseMethod("coefbootstrap")

#' @export
coefbootstrap.default <- function(object, nsim=1000, size=floor(0.75*nobs(object)),
                                  replace=FALSE, prob=NULL, parallel=FALSE){

    startTime <- Sys.time();

    # Form the call for the function
    if(!is.null(object$call)){
        newCall <- object$call;
    }
    else{
        stop("In order for the function to work, the object should have the variable 'call' in it. Cannot proceed.",
             call.=FALSE);
    }

    if(is.numeric(parallel)){
        nCores <- parallel;
        parallel <- TRUE;
    }
    else if(is.logical(parallel) && parallel){
        # Detect number of cores for parallel calculations
        nCores <- min(parallel::detectCores() - 1, nsim);
    }

    # If they asked for parallel, make checks and try to do that
    if(parallel){
        if(!requireNamespace("foreach", quietly = TRUE)){
            stop("In order to run the function in parallel, 'foreach' package must be installed.", call. = FALSE);
        }
        if(!requireNamespace("parallel", quietly = TRUE)){
            stop("In order to run the function in parallel, 'parallel' package must be installed.", call. = FALSE);
        }

        # Check the system and choose the package to use
        if(Sys.info()['sysname']=="Windows"){
            if(requireNamespace("doParallel", quietly = TRUE)){
                cluster <- parallel::makeCluster(nCores);
                doParallel::registerDoParallel(cluster);
            }
            else{
                stop("Sorry, but in order to run the function in parallel, you need 'doParallel' package.",
                     call. = FALSE);
            }
        }
        else{
            if(requireNamespace("doMC", quietly = TRUE)){
                doMC::registerDoMC(nCores);
                cluster <- NULL;
            }
            else if(requireNamespace("doParallel", quietly = TRUE)){
                cluster <- parallel::makeCluster(nCores);
                doParallel::registerDoParallel(cluster);
            }
            else{
                stop("Sorry, but in order to run the function in parallel, you need either 'doMC' (prefered) or 'doParallel' packages.",
                     call. = FALSE);
            }
        }
    }

    # Coefficients of the model
    coefficientsOriginal <- coef(object);
    nVariables <- length(coefficientsOriginal);
    variablesNames <- names(coefficientsOriginal);
    interceptIsNeeded <- any(variablesNames=="(Intercept)");
    obsInsample <- nobs(object);

    # The matrix with coefficients
    coefBootstrap <- matrix(0, nsim, nVariables, dimnames=list(NULL, variablesNames));
    # Indices for the observations to use and the vector of subsets
    indices <- c(1:obsInsample);

    if(!parallel){
        for(i in 1:nsim){
            newCall$subset <- sample(indices,size=size,replace=replace,prob=prob);
            testModel <- suppressWarnings(eval(newCall));
            coefBootstrap[i,names(coef(testModel))] <- coef(testModel);
        }
    }
    else{
        # We don't do rbind for security reasons - in order to deal with skipped variables
        coefBootstrapParallel <- foreach::`%dopar%`(foreach::foreach(i=1:nsim),{
            newCall$subset <- sample(indices,size=size,replace=replace,prob=prob);
            testModel <- eval(newCall);
            return(coef(testModel));
        })
        # Prepare the matrix with parameters
        for(i in 1:nsim){
            coefBootstrap[i,names(coefBootstrapParallel[[i]])] <- coefBootstrapParallel[[i]];
        }
    }
    # Get rid of NAs. They mean "zero"
    coefBootstrap[is.na(coefBootstrap)] <- 0;

    # Centre the coefficients for the calculation of the vcov
    coefvcov <- coefBootstrap - matrix(coefficientsOriginal, nsim, nVariables, byrow=TRUE);

    return(structure(list(vcov=(t(coefvcov) %*% coefvcov)/nsim,
                          coefficients=coefBootstrap,
                          nsim=nsim, size=size, replace=replace, prob=prob, parallel=parallel,
                          model=object$call[[1]], timeElapsed=Sys.time()-startTime),
                     class="bootstrap"));
}

#' @rdname coefbootstrap
#' @export
coefbootstrap.lm <- function(object, nsim=1000, size=floor(0.75*nobs(object)),
                              replace=FALSE, prob=NULL, parallel=FALSE){
    return(coefbootstrap.default(object, nsim, size, replace, prob, parallel));
}

#' @rdname coefbootstrap
#' @export
coefbootstrap.alm <- function(object, nsim=1000, size=floor(0.75*nobs(object)),
                              replace=FALSE, prob=NULL, parallel=FALSE){

    startTime <- Sys.time();

    cl <- match.call();

    if(is.numeric(parallel)){
        nCores <- parallel;
        parallel <- TRUE;
    }
    else if(is.logical(parallel) && parallel){
        # Detect number of cores for parallel calculations
        nCores <- min(parallel::detectCores() - 1, nsim);
    }

    # If they asked for parallel, make checks and try to do that
    if(parallel){
        if(!requireNamespace("foreach", quietly = TRUE)){
            stop("In order to run the function in parallel, 'foreach' package must be installed.", call. = FALSE);
        }
        if(!requireNamespace("parallel", quietly = TRUE)){
            stop("In order to run the function in parallel, 'parallel' package must be installed.", call. = FALSE);
        }

        # Check the system and choose the package to use
        if(Sys.info()['sysname']=="Windows"){
            if(requireNamespace("doParallel", quietly = TRUE)){
                cluster <- parallel::makeCluster(nCores);
                doParallel::registerDoParallel(cluster);
            }
            else{
                stop("Sorry, but in order to run the function in parallel, you need 'doParallel' package.",
                     call. = FALSE);
            }
        }
        else{
            if(requireNamespace("doMC", quietly = TRUE)){
                doMC::registerDoMC(nCores);
                cluster <- NULL;
            }
            else if(requireNamespace("doParallel", quietly = TRUE)){
                cluster <- parallel::makeCluster(nCores);
                doParallel::registerDoParallel(cluster);
            }
            else{
                stop("Sorry, but in order to run the function in parallel, you need either 'doMC' (prefered) or 'doParallel' packages.",
                     call. = FALSE);
            }
        }
    }

    # Coefficients of the model
    coefficientsOriginal <- coef(object);
    nVariables <- length(coefficientsOriginal);
    variablesNames <- names(coefficientsOriginal);
    interceptIsNeeded <- any(variablesNames=="(Intercept)");
    variablesNamesMade <- make.names(variablesNames);
    if(interceptIsNeeded){
        variablesNamesMade[1] <- variablesNames[1];
    }
    obsInsample <- nobs(object);

    # The matrix with coefficients
    coefBootstrap <- matrix(0, nsim, nVariables, dimnames=list(NULL, variablesNames));
    # Indices for the observations to use and the vector of subsets
    indices <- c(1:obsInsample);

    # Form the call for alm
    newCall <- object$call;
    # Tuning for srm, to call alm() instead
    # if(is.srm(object)){
    #     newCall[[1]] <- as.name("alm");
    #     newCall$folder <- NULL;
    # }
    # This is based on the expanded data, so that we don't need to redo everything
    if(interceptIsNeeded){
        newCall$formula <- as.formula(paste0("`",colnames(object$data)[1],"`~."));
    }
    else{
        newCall$formula <- as.formula(paste0("`",colnames(object$data)[1],"`~.-1"));
    }
    newCall$data <- substitute(object$data);
    newCall$distribution <- object$distribution;
    if(object$loss=="custom"){
        newCall$loss <- object$lossFunction;
    }
    else{
        newCall$loss <- object$loss;
    }

    # Deal with ari components. Remove the columns with lagged variables
    arimaModel <- !is.null(object$other$polynomial);
    if(arimaModel){
        newCall$data <- newCall$data[,!(colnames(newCall$data) %in% names(object$other$polynomial))];
    }

    # If this is ARIMA, and the size wasn't specified, make it changeable
    if(arimaModel && is.null(cl$size)){
        changeSize <- TRUE;
    }
    else{
        changeSize <- FALSE;
    }

    newCall$fast <- TRUE;
    if(any(object$distribution==c("dchisq","dt"))){
        newCall$nu <- object$other$nu;
    }
    else if(object$distribution=="dnbinom"){
        newCall$size <- object$other$size;
    }
    else if(object$distribution=="dalaplace"){
        newCall$alpha <- object$other$alpha;
    }
    else if(object$distribution=="dfnorm"){
        newCall$sigma <- object$other$sigma;
    }
    else if(object$distribution=="dbcnorm"){
        newCall$lambdaBC <- object$other$lambdaBC;
    }
    else if(any(object$distribution==c("dgnorm","dlgnorm"))){
        newCall$shape <- object$other$shape;
    }
    newCall$occurrence <- object$occurrence;
    # Only pre-initialise the parameters if non-normal stuff is used
    if(all(object$distribution!=c("dnorm","dlnorm","dlogitnorm"))){
        newCall$B <- coef(object);
    }
    # If there is scale model, remove it
    if(is.scale(object$scale)){
        newCall$scale <- NULL;
    }

    # Function creates a random sample. Needed for dynamic models
    sampler <- function(indices,size,replace,prob,arimaModel=FALSE,changeSize=FALSE){
        if(arimaModel){
            if(changeSize){
                size[] <- floor(runif(1,nVariables,obsInsample))+1;
            }
            # This way we return the continuos sample, but with random starting point
            return(floor(runif(1,0,obsInsample-size))+c(1:size));
        }
        else{
            return(sample(indices,size=size,replace=replace,prob=prob));
        }
    }

    if(!parallel){
        for(i in 1:nsim){
            newCall$subset <- sampler(indices,size,replace,prob,arimaModel,changeSize);
            testModel <- suppressWarnings(eval(newCall));
            coefBootstrap[i,variablesNamesMade %in% names(coef(testModel))] <- coef(testModel);
        }
    }
    else{
        # We don't do rbind for security reasons - in order to deal with skipped variables
        coefBootstrapParallel <- foreach::`%dopar%`(foreach::foreach(i=1:nsim),{
            newCall$subset <- sampler(indices,size,replace,prob,arimaModel,changeSize);
            testModel <- eval(newCall);
            return(coef(testModel));
        })
        # Prepare the matrix with parameters
        for(i in 1:nsim){
            coefBootstrap[i,variablesNamesMade %in% names(coefBootstrapParallel[[i]])] <- coefBootstrapParallel[[i]];
        }
    }

    # Get rid of NAs. They mean "zero"
    coefBootstrap[is.na(coefBootstrap)] <- 0;

    # Rename the variables to the originals
    colnames(coefBootstrap) <- names(coefficientsOriginal);

    # Centre the coefficients for the calculation of the vcov
    coefvcov <- coefBootstrap - matrix(colMeans(coefBootstrap), nsim, nVariables, byrow=TRUE);

    return(structure(list(vcov=(t(coefvcov) %*% coefvcov)/(nsim-1),
                          coefficients=coefBootstrap,
                          nsim=nsim, size=size, replace=replace, prob=prob, parallel=parallel,
                          model=object$call[[1]], timeElapsed=Sys.time()-startTime),
                     class="bootstrap"));
}


#' @export
print.bootstrap <- function(x, ...){
    cat(paste0("Bootstrap for the ", x$model, " model with nsim=",x$nsim," and size=",x$size,"\n"))
    cat(paste0("Time elapsed: ",round(as.numeric(x$timeElapsed,units="secs"),2)," seconds\n"));
}


#' Time series bootstrap
#'
#' The function implements a bootstrap inspired by the Maximum Entropy Bootstrap
#'
#' The function implements the following algorithm:
#'
#' 1. Sort the original data, recording the original order of elements;
#' 2. Smooth the sorted data using LOWESS to get intermediate points;
#' 3. Take first differences of the sorted series and sort them;
#' 4. Create contingency table based on the differences and take the cumulative sum
#' of it. This way we end up with an empirical CDF of differences;
#' 5. Generate random numbers from the uniform distribution between 0 and 1;
#' 6. Get the differences that correspond to the random number (randomly extract
#' empirical quantiles);
#' 7. Add the random differences to the smooth series from (2) to get new series;
#' 8. Reorder (7) based on the initial order of series.
#'
#' If the multiplicative bootstrap is used then logarithms of the sorted series and LOWESS
#' are used and at the very end, the exponent of the resulting data is taken. This way the
#' discrepancies in the data have similar scale no matter what the level of the original
#' series is. In case of the additive bootstrap, the trended series will be noisier when
#' the level of series is low.
#'
#' @param y The original time series
#' @param nsim Number of iterations (simulations) to run.
#' @param scale Parameter that defines how to scale the variability around the data.
#' @param type Type of bootstrap to use. \code{"additive"} means that the randomness is
#' added, while \code{"multiplicative"} implies the multiplication. By default the function
#' will try using the latter, unless the data has non-positive values.
#' @param f The smoother span parameter for LOWESS (see \link[stats]{lowess}). Small value
#' should be used to capture the trend in the sorted data correctly.
#'
#' @return The function returns the matrix with the new series in columns and observations
#' in rows.
#'
#' @template author
#' @template keywords
#'
#' @seealso \code{\link[stats]{lowess}}
#'
#' @references Vinod HD, López-de-Lacalle J (2009). "Maximum Entropy Bootstrap for Time Series:
#' The meboot R Package." Journal of Statistical Software, 29(5), 1–19. \doi{doi:10.18637/jss.v029.i05}.
#'
#' @examples
#' plot(AirPassengers, type="l")
#' timeboot(AirPassengers) |> lines(col="blue")
#'
#' @rdname timeboot
#' @export
timeboot <- function(y, nsim=100, scale=NULL,
                     type=c("auto","multiplicative","additive"),
                     f=0.002){
    type <- match.arg(type);

    if(type=="auto"){
        if(any(y<=0)){
            type[] <- "additive";
        }
        else{
            type[] <- "multiplicative";
        }
    }

    # Heurstic: strong trend -> scale ~ 0; no trend -> scale ~ 2
    if(is.null(scale)){
        scale <- (1-abs(mean(diff(y)))/mean(abs(diff(y))))*2;
    }

    # Sample size and ordered values
    obsInsample <- length(y);
    yOrder <- order(y);
    ySorted <- sort(y);
    # Intermediate points are done via a sensitive lowess
    # This is because the sorted values have "trend"
    yIntermediate <- lowess(ySorted, f=f)$y

    if(type=="multiplicative"){
        ySorted[] <- log(ySorted);
        yIntermediate[] <- log(yIntermediate);
    }
    yDiffs <- sort(diff(ySorted));

    # Intervals for the uniform densities
    # yMean <- mean(abs(diff(y)), trim=trim);
    # yIntermediate <- c(ySorted[1] - yMean,
    #                    (ySorted[-1] + ySorted[-obsInsample])/2,
    #                    ySorted[obsInsample] + yMean);
    # yIntermediate <- (ySorted[-1] + ySorted[-obsInsample])/2;

    yDiffsTable <- cumsum(table(yDiffs)/(obsInsample-1));
    yDiffsUnique <- unique(yDiffs);

    yRandom <- runif(obsInsample, 0, 1);
    yDiffsNew <- yDiffsUnique[findInterval(yRandom,yDiffsTable)+1];

    yNew <- y;
    # Sample introduces +-1 to not just add but also subtract
    yNew[yOrder] <- yIntermediate + scale*sample(c(-1,1), size=obsInsample, replace=TRUE)*yDiffsNew;
    # yNew[yOrder] <- c(yIntermediate[1]-scale*yDiffsNew[1],
    #                   yIntermediate + scale*sample(c(-1,1), size=obsInsample-1, replace=TRUE)*yDiffsNew[-1]);

    if(type=="multiplicative"){
        yNew[] <- exp(yNew);
    }

    return(yNew);
}
