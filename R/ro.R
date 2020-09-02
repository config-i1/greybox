#' Rolling Origin
#'
#' The function does rolling origin for any forecasting function
#'
#' This function produces rolling origin forecasts using the \code{data} and a
#' \code{call} passed as parameters. The function can do all of that either in
#' serial or in parallel, but it needs \code{foreach} and either \code{doMC}
#' (Linux only) or \code{doParallel} packages installed in order to do the latter.
#'
#' This is a dangerous function, so be careful with the call that you pass to
#' it, and make sure that it is well formulated before the execution. Also, do not
#' forget to provide the value that needs to be returned or you might end up with
#' very messy results.
#'
#' For more details and more examples of usage, please see vignette for the function.
#' In order to do that, just run the command: \code{vignette("ro","greybox")}
#'
#' @keywords ts
#'
#' @param data Data vector or ts object with the response variable passed to the
#' function.
#' @param h The forecasting horizon.
#' @param origins The number of rolling origins.
#' @param call The call that is passed to the function. The call must be in quotes.
#' Example: \code{"forecast(ets(data),h)"}. Here \code{data} shows where the data is
#' and \code{h} defines where the horizon should be passed in the \code{call}. Some
#' hidden parameters can also be specified in the call. For example, parameters
#' \code{counti}, \code{counto} and \code{countf} are used in the inner loop
#' and can be used for the regulation of exogenous variables sizes. See examples
#' for the details.
#' @param value The variable or set of variables returned by the \code{call}.
#' For example, \code{mean} for functions of \code{forecast} package. This can
#' also be a vector of variables. See examples for the details. If the parameter
#' is \code{NULL}, then all the values from the call are returned (could be really
#' messy!). Note that if your function returns a list with matrices, then ro will
#' return an array. If your function returns a list, then you will have a list of
#' lists in the end. So it makes sense to understand what you want to get before
#' running the function.
#' @param ci The parameter defines if the in-sample window size should be constant.
#' If \code{TRUE}, then with each origin one observation is added at the end of
#' series and another one is removed from the beginning.
#' @param co The parameter defines whether the holdout sample window size should
#' be constant. If \code{TRUE}, the rolling origin will stop when less than
#' \code{h} observations are left in the holdout.
#' @param silent If \code{TRUE}, nothing is printed out in the console.
#' @param parallel If \code{TRUE}, then the model fitting is done in parallel.
#' WARNING! Packages \code{foreach} and either \code{doMC} (Linux and Mac only)
#' or \code{doParallel} are needed in order to run the function in parallel.
#' @param ... This is temporary and is needed in order to capture "silent"
#' parameter if it is provided.
#'
#' @return Function returns the following variables:
#' \itemize{
#' \item{\code{actuals} - the data provided to the function.}
#' \item{\code{holdout} - the matrix of actual values corresponding to the
#' produced forecasts from each origin.}
#' \item{\code{value} - the matrices / array / lists with the produced data
#' from each origin. Name of each object corresponds to the names in the
#' parameter \code{value}.}
#' }
#'
#' @author Yves Sagaert
#' @template author
#'
#' @references \itemize{
#' \item Tashman, (2000) Out-of-sample tests of forecasting accuracy:
#' an analysis and review International Journal of Forecasting, 16,
#' pp. 437-450. \url{https://doi.org/10.1016/S0169-2070(00)00065-0}.
#' }
#'
#' @examples
#'
#' y <- rnorm(100,0,1)
#' ourCall <- "predict(arima(x=data,order=c(0,1,1)),n.ahead=h)"
#' # NOTE that the "data" needs to be used in the call, not "y".
#' # This way we tell the function, where "y" should be used in the call of the function.
#'
#' # The default call and values
#' ourValue <- "pred"
#' ourRO <- ro(y, h=5, origins=5, ourCall, ourValue)
#'
#' # We can now plot the results of this evaluation:
#' plot(ourRO)
#'
#' # You can also use dolar sign
#' ourValue <- "$pred"
#' # And you can have constant in-sample size
#' ro(y, h=5, origins=5, ourCall, ourValue, ci=TRUE)
#'
#' # You can ask for several values
#' ourValue <- c("pred","se")
#' # And you can have constant holdout size
#' ro(y, h=5, origins=20, ourCall, ourValue, ci=TRUE, co=TRUE)
#'
#' #### The following code will give exactly the same result as above,
#' #### but computed in parallel using all but 1 core of CPU:
#' \dontrun{ro(y, h=5, origins=20, ourCall, ourValue, ci=TRUE, co=TRUE, parallel=TRUE)}
#'
#' #### If you want to use functions from forecast package, please note that you need to
#' #### set the values that need to be returned explicitly. There are two options for this.
#' # Example 1:
#' \dontrun{ourCall <- "forecast(ets(data), h=h, level=95)"
#' ourValue <- c("mean", "lower", "upper")
#' ro(y,h=5,origins=5,ourCall,ourValue)}
#'
#' # Example 2:
#' \dontrun{ourCall <- "forecast(ets(data), h=h, level=c(80,95))"
#' ourValue <- c("mean", "lower[,1]", "upper[,1]", "lower[,2]", "upper[,2]")
#' ro(y,h=5,origins=5,ourCall,ourValue)}
#'
#' #### A more complicated example using the for loop and
#' #### several time series
#' x <- matrix(rnorm(120*3,0,1), 120, 3)
#'
#' ## Form an array for the forecasts we will produce
#' ## We will have 4 origins with 6-steps ahead forecasts
#' ourForecasts <- array(NA,c(6,4,3))
#'
#' ## Define models that need to be used for each series
#' ourModels <- list(c(0,1,1), c(0,0,1), c(0,1,0))
#'
#' ## This call uses specific models for each time series
#' ourCall <- "predict(arima(data, order=ourModels[[i]]), n.ahead=h)"
#' ourValue <- "pred"
#'
#' ## Start the loop. The important thing here is to use the same variable 'i' as in ourCall.
#' for(i in 1:3){
#'     ourData <- x[,i]
#'     ourForecasts[,,i] <- ro(data=ourData,h=6,origins=4,call=ourCall,
#'                             value=ourValue,co=TRUE,silent=TRUE)$pred
#' }
#'
#' ## ourForecasts array now contains rolling origin forecasts from specific
#' ## models.
#'
#' ##### An example with exogenous variables
#' x <- rnorm(100,0,1)
#' xreg <- matrix(rnorm(200,0,1),100,2,dimnames=list(NULL,c("x1","x2")))
#'
#' ## 'counti' is used to define in-sample size of xreg,
#' ## 'counto' - the size of the holdout sample of xreg
#'
#' ourCall <- "predict(arima(x=data, order=c(0,1,1), xreg=xreg[counti,,drop=FALSE]),
#'             n.ahead=h, newxreg=xreg[counto,,drop=FALSE])"
#' ourValue <- "pred"
#' ro(x,h=5,origins=5,ourCall,ourValue)
#'
#' ##### Poisson regression with alm
#' x <- rpois(100,2)
#' xreg <- cbind(x,matrix(rnorm(200,0,1),100,2,dimnames=list(NULL,c("x1","x2"))))
#' ourCall <- "predict(alm(x~., data=xreg[counti,,drop=FALSE], distribution='dpois'),
#'                     newdata=xreg[counto,,drop=FALSE])"
#' ourValue <- "mean"
#' testRO <- ro(xreg[,1],h=5,origins=5,ourCall,ourValue,co=TRUE)
#' plot(testRO)
#'
#' ## 'countf' is used to take xreg of the size corresponding to the whole
#' ## sample on each iteration
#' ## This is useful when working with functions from smooth package.
#' ## The following call will return the forecasts from es() function of smooth.
#' \dontrun{ourCall <- "es(data=data, h=h, xreg=xreg[countf,,drop=FALSE])"
#' \dontrun{ourValue <- "forecast"}
#' \dontrun{ro(x,h=5,origins=5,ourCall,ourValue)}}
#'
#' @export ro
ro <- function(data,h=10,origins=10,call,value=NULL,
               ci=FALSE,co=TRUE,silent=TRUE,parallel=FALSE, ...){
    # Function makes Rolling Origin for the data using the call
    #    Copyright (C) 2016  Yves Sagaert & Ivan Svetunkov

    # Names of variables ivan41 and yves14 are given in order not to mess with the possible inner loops of "for(i in 1:n)" type.
    valueLength <- length(value);
    if(!is.null(value)){
        for(ivan41 in 1:valueLength){
            if(substring(value[ivan41],1,1)!="$"){
                value[ivan41] <- paste0("$",value[ivan41]);
            }
        }
    }
    else{
        warning(paste0("You have not specified the 'value' to produce.",
                       "We will try to return everything, but we cannot promise anything."), call.=FALSE);
        value <- "";
        valueLength <- 1;
    }

    # Check if the data is vector or ts
    if(!is.numeric(data) && !is.ts(data) && !is.data.frame(data)){
        stop("The provided data is not a vector or data.frame object! Can't work with it!", call.=FALSE);
    }

    # If they asked for parallel, make checks and try to do that
    if(parallel){
        if(!requireNamespace("foreach", quietly = TRUE)){
            stop("In order to run the function in parallel, 'foreach' package must be installed.", call. = FALSE);
        }
        if(!requireNamespace("parallel", quietly = TRUE)){
            stop("In order to run the function in parallel, 'parallel' package must be installed.", call. = FALSE);
        }
        # Detect number of cores for parallel calculations
        nCores <- min(parallel::detectCores() - 1, origins);

        # Check the system and choose the package to use
        if(Sys.info()['sysname']=="Windows"){
            if(requireNamespace("doParallel", quietly = TRUE)){
                cat(paste0("Setting up ", nCores, " clusters using 'doParallel'..."));
                cat("\n");
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
                cat(paste0("Setting up ", nCores, " clusters using 'doParallel'..."));
                cat("\n");
                cluster <- parallel::makeCluster(nCores);
                doParallel::registerDoParallel(cluster);
            }
            else{
                stop("Sorry, but in order to run the function in parallel, you need either 'doMC' (prefered) or 'doParallel' packages.",
                     call. = FALSE);
            }
        }
    }

    # Write down the start and frequency
    dataStart <- start(data);
    dataFrequency <- frequency(data);

    # Write down the entire dataset y and the original horizon
    y <- data;
    hh <- h;
    obs <- length(y);
    inSample <- obs - origins;

    holdout <- matrix(NA,nrow=h,ncol=origins);
    if(co){
        colnames(holdout) <- paste0("origin",inSample+c(1:origins)-h);
    }
    else{
        colnames(holdout) <- paste0("origin",inSample+c(1:origins));
    }
    rownames(holdout) <- paste0("h",c(1:h));

    forecasts <- list(NA);
    if(!silent & !parallel){
        cat(paste0("Origins done: "));
        cat(paste(rep(" ",origins),collapse=""))
    }
    else if(!silent & parallel){
        cat(paste0("Working..."));
    }

    ##### Start the main loop #####
    if(!parallel){
        if(!co){
            for(ivan41 in 1:origins){
                # Adjust forecasting horizon to not exeed the sample size
                h[] <- min(hh,obs - (inSample+ivan41-1));
                # Make the in-sample
                if(!ci){
                    counti <- 1:(inSample+ivan41-1);
                }
                else{
                    counti <- ivan41:(inSample+ivan41-1);
                }
                counto <- (inSample+ivan41):(inSample+ivan41+h-1);
                countf <- c(counti,counto);

                data <- ts(y[counti],start=dataStart,frequency=dataFrequency);
                # Evaluate the call string and save to the object
                callEvaluated <- eval(parse(text=call));
                # Save the forecast and the corresponding holdout in matrices
                for(yves14 in 1:valueLength){
                    forecasts[[(ivan41-1)*valueLength+yves14]] <- eval(parse(text=paste0("callEvaluated",value[yves14])));
                }
                holdout[1:h,ivan41] <- y[counto];
                if(!silent){
                    cat(paste(rep("\b",nchar(ivan41)),collapse=""));
                    cat(ivan41);
                }
            }
        }
        else{
            for(ivan41 in 1:origins){
                # Make the in-sample
                if(!ci){
                    counti <- 1:(inSample-h+ivan41);
                }
                else{
                    counti <- ivan41:(inSample-h+ivan41);
                }
                counto <- (inSample+ivan41-h+1):(inSample+ivan41);
                countf <- c(counti,counto);

                data <- ts(y[counti],start=dataStart,frequency=dataFrequency);
                # Evaluate the call string and save to object callEvaluated.
                callEvaluated <- eval(parse(text=call));
                # Save the forecast and the corresponding holdout in matrices
                for(yves14 in 1:valueLength){
                    forecasts[[(ivan41-1)*valueLength+yves14]] <- eval(parse(text=paste0("callEvaluated",value[yves14])));
                }
                holdout[,ivan41] <- y[counto];
                if(!silent){
                    cat(paste(rep("\b",nchar(ivan41)),collapse=""));
                    cat(ivan41);
                }
            }
        }
    }
    else{
        ##### Use foreach for the loop #####
        # But first make the list of the needed packages to pass to doParallel
        callenvir <- globalenv();
        callpackages <- search();
        callpackages <- callpackages[c(-1,-length(callpackages))];
        callpackages <- callpackages[substring(callpackages,1,7)=="package"];
        callpackages <- substring(callpackages,9,nchar(callpackages));
        callpackages <- callpackages[callpackages!="timeDate"];
        callpackages <- callpackages[callpackages!="zoo"];
        callpackages <- callpackages[callpackages!="stats"];
        callpackages <- callpackages[callpackages!="graphics"];
        callpackages <- callpackages[callpackages!="grDevices"];
        callpackages <- callpackages[callpackages!="utils"];
        callpackages <- callpackages[callpackages!="datasets"];
        callpackages <- callpackages[callpackages!="methods"];

        if(!co){
            forecasts <- foreach::`%dopar%`(foreach::foreach(ivan41=1:origins, .packages=callpackages, .export=ls(envir=callenvir)),{
                # Adjust forecasting horizon to not exeed the sample size
                h <- min(hh,obs - (inSample+ivan41-1));
                # Make the in-sample
                if(ci==FALSE){
                    counti <- 1:(inSample+ivan41-1);
                }
                else{
                    counti <- ivan41:(inSample+ivan41-1);
                }
                counto <- (inSample+ivan41):(inSample+ivan41+h-1);
                countf <- c(counti,counto);

                data <- ts(y[counti],start=dataStart,frequency=dataFrequency);
                # Evaluate the call string and save to object callEvaluated.
                callEvaluated <- eval(parse(text=call));
                # Save the forecast and the corresponding holdout in matrices
                for(yves14 in 1:valueLength){
                    forecasts[[yves14]] <- eval(parse(text=paste0("callEvaluated",value[yves14])));
                }

                return(forecasts);
            })
        }
        else{
            forecasts <- foreach::`%dopar%`(foreach::foreach(ivan41=1:origins, .packages=callpackages, .export=ls(envir=callenvir)),{
                # Make the in-sample
                if(ci==FALSE){
                    counti <- 1:(inSample-h+ivan41);
                }
                else{
                    counti <- ivan41:(inSample-h+ivan41);
                }
                counto <- (inSample+ivan41-h+1):(inSample+ivan41);
                countf <- c(counti,counto);

                data <- ts(y[counti],start=dataStart,frequency=dataFrequency);
                # Evaluate the call string and save to object callEvaluated.
                callEvaluated <- eval(parse(text=call));
                # Save the forecast and the corresponding holdout in matrices
                for(yves14 in 1:valueLength){
                    forecasts[[yves14]] <- eval(parse(text=paste0("callEvaluated",value[yves14])));
                }

                return(forecasts);
            })
        }

        # Make the needed list if there were several values
        forecasts <- unlist(forecasts,recursive=FALSE);

        # Check if the clusters have been made
        if(!is.null(cluster)){
            parallel::stopCluster(cluster);
        }

        # Form matrix of holdout in a different loop...
        if(co==FALSE){
            for(ivan41 in 1:origins){
                holdout[1:h,ivan41] <- y[(inSample+ivan41):(inSample+ivan41+h-1)];
            }
        }
        else{
            for(ivan41 in 1:origins){
                holdout[,ivan41] <- y[(inSample+ivan41-h+1):(inSample+ivan41)];
            }
        }
    }

    if(!silent){
        cat("\n");
    }

    listReturned <- list(holdout);

    # Transform the list of forecasts into something general
    if(all(value=="")){
        if(is.matrix(forecasts[[1]])){
            value <- colnames(forecasts[[1]]);
            forecasts <- unlist(lapply(lapply(forecasts,as.data.frame),as.list),
                                recursive=FALSE);
        }
        else if(is.list(forecasts[[1]])){
            forecasts <- as.list(unlist(forecasts,recursive=FALSE));
            value <- names(forecasts[[1]]);
        }
        valueLength <- length(value);
    }
    else{
        value <- substring(value,2,nchar(value));
    }

    # Reconstruct the list, so that each object is a separate element
    for(ivan41 in 1:valueLength){
        if(is.array(forecasts[[ivan41]])){
            # If it is matrix, form arrays
            # The length is the max between the first and the last elements of this type
            stuffMaxLength <- max(nrow(forecasts[[ivan41]]),
                                  nrow(forecasts[[origins*(valueLength-1)+ivan41]]));
            if(length(dim(forecasts[[ivan41]]))==2){
                if(ncol(forecasts[[ivan41]])>1){
                    # If it contains several columns, make an array
                    # We assume that this thing has the same number of columns for all the origins
                    stuffNcol <- ncol(forecasts[[ivan41]]);
                    stuff <- array(NA,c(stuffMaxLength,origins,stuffNcol));
                    if(stuffMaxLength==nrow(holdout)){
                        dimnames(stuff) <- list(rownames(holdout), colnames(holdout),
                                                colnames(forecasts[[ivan41]]));
                    }
                    else{
                        dimnames(stuff) <- list(NULL, colnames(holdout), colnames(forecasts[[ivan41]]));
                    }
                    # Do the loop and fill in the new array
                    for(yves14 in 1:origins){
                        stuffLength <- nrow(forecasts[[(yves14-1)*valueLength+ivan41]]);
                        stuff[1:stuffLength,yves14,] <- forecasts[[(yves14-1)*valueLength+ivan41]];
                    }
                }
                else{
                    # If it is a one-column, then stack it to the matrix
                    stuff <- matrix(NA, stuffMaxLength, origins);
                    if(stuffMaxLength==nrow(holdout)){
                        dimnames(stuff) <- dimnames(holdout);
                    }
                    else{
                        dimnames(stuff) <- list(NULL, colnames(holdout));
                    }
                    # Do the loop and fill in the new array
                    for(yves14 in 1:origins){
                        stuffLength <- nrow(forecasts[[(yves14-1)*valueLength+ivan41]]);
                        stuff[1:stuffLength,yves14] <- forecasts[[(yves14-1)*valueLength+ivan41]];
                    }
                }
            }
            else{
                # If this is an array (3d and more), return a list
                stuff <- list(NA);
                for(yves14 in 1:origins){
                    stuff[[yves14]] <- forecasts[[(yves14-1)*valueLength+ivan41]];
                }
                names(stuff) <- colnames(holdout);
            }
        }
        else if(is.list(forecasts[[ivan41]])){
            # If it is a list, just make a list of lists
            stuff <- list(NA);
            for(yves14 in 1:origins){
                stuff[[yves14]] <- forecasts[[(yves14-1)*valueLength+ivan41]];
            }
            names(stuff) <- colnames(holdout);
        }
        else{
            # If it is something else (vector?), then stack everything together
            # The length is the max between the first and the last elements of this type
            stuffMaxLength <- max(length(forecasts[[ivan41]]),
                                  length(forecasts[[(origins-1)*valueLength+ivan41]]));
            # If it is a one-column, then stack it to the matrix
            stuff <- matrix(NA, stuffMaxLength, origins);
            if(stuffMaxLength==nrow(holdout)){
                dimnames(stuff) <- dimnames(holdout);
            }
            else{
                dimnames(stuff) <- list(NULL, colnames(holdout));
            }

            # Do the loop and fill in the new array
            for(yves14 in 1:origins){
                stuffLength <- length(forecasts[[(yves14-1)*valueLength+ivan41]]);
                # cat(stuffLength); cat("\n")
                stuff[1:stuffLength,yves14] <- forecasts[[(yves14-1)*valueLength+ivan41]];
            }
        }
        listReturned[[ivan41+1]] <- stuff;
    }

    listReturned <- c(list(actuals=y),listReturned);
    names(listReturned)[-1] <- c("holdout",value);
    return(structure(listReturned,class="rollingOrigin"));
}
