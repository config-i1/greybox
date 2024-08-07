#' Time series bootstrap
#'
#' The function implements a bootstrap inspired by the Maximum Entropy Bootstrap
#'
#' The function implements the following algorithm:
#'
#' 1. Sort the data in the ascending order, recording the original order of elements;
#' 2. Take first differences of the sorted series and sort them;
#' 3. Create contingency table based on the differences and take the cumulative sum
#' of it. This way we end up with an empirical CDF of differences;
#' 4. Generate random numbers from the uniform distribution between 0 and 1;
#' 5. Get the differences that correspond to the random numbers (randomly extract
#' empirical quantiles). This way we take the empirical density into account when
#' selecting the differences;
#' 6. Add the random differences to the sorted series from (1) to get a new time series;
#' 7. Sort the new time series in the ascending order;
#' 8. Reorder (7) based on the initial order of series.
#'
#' If the multiplicative bootstrap is used then logarithms of the sorted series
#' are used and at the very end, the exponent of the resulting data is taken. This way the
#' discrepancies in the data have similar scale no matter what the level of the original
#' series is. In case of the additive bootstrap, the trended series will be noisier when
#' the level of series is low.
#'
#' @param y The original time series
#' @param nsim Number of iterations (simulations) to run.
#' @param intermittent Whether to treat the demand as intermittent or not.
#' @param type Type of bootstrap to use. \code{"additive"} means that the randomness is
#' added, while \code{"multiplicative"} implies the multiplication.
#' @param kind A kind of the bootstrap to do: nonparametric or parametric. The latter
#' relies on the normal distribution, while the former uses the empirical distribution of
#' differences of the data.
#' @param lag The lag to use in the calculation of differences. Should be 1 for non-seasonal
#' data.
#' @param sd Standard deviation to use in the normal distribution. Estimated as mean absolute
#' differences of the data if omitted.
#' @param scale Whether or not to do scaling of time series to the bootstrapped ones to have
#' similar variance to the original data.
#'
#' @return The function returns:
#' \itemize{
#' \item \code{call} - the call that was used originally;
#' \item \code{data} - the original data used in the function;
#' \item \code{boot} - the matrix with the new series in columns and observations in rows.
#' \item \code{type} - type of the bootstrap used.
#' \item \code{sd} - the value of sd used in case of parameteric bootstrap.
#' \item \code{scale} - whether the scaling was needed.}
#'
#' @template author
#' @template keywords
#'
#' @references Vinod HD, López-de-Lacalle J (2009). "Maximum Entropy Bootstrap for Time Series:
#' The meboot R Package." Journal of Statistical Software, 29(5), 1–19. \doi{doi:10.18637/jss.v029.i05}.
#'
#' @examples
#' timeboot(AirPassengers) |> plot()
#'
#' @rdname timeboot
#' @importFrom stats supsmu
#' @export
timeboot <- function(y, nsim=100, intermittent=TRUE,
                     type=c("multiplicative","additive"),
                     kind=c("nonparametric","parametric"),
                     lag=frequency(y), sd=NULL,
                     scale=TRUE){
    cl <- match.call();

    type <- match.arg(type);
    kind <- match.arg(kind);

    # Sample size and new data
    obsInsample <- length(y);
    otU <- rep(TRUE, obsInsample);
    yIsInteger <- FALSE;

    # If we have intermittent demand, do timeboot for demand sizes and intervals
    if(any(y==0)){
        otU[] <- y!=0;
        if(intermittent){
            obsNonZero <- sum(otU);
            # Get rid of class
            ySizes <- as.vector(y[otU]);
            yIsInteger <- all(ySizes==trunc(ySizes));
            # -1 is needed to get the thing closer to the geometric distribution (start from zero)
            yIntervals <- diff(c(0,which(otU)))-1;

            # Bootstrap the demand sizes
            ySizesBoot <- timeboot(ySizes, nsim=nsim, intermittent=FALSE,
                                   type=type, kind=kind, lag=1, sd=sd, scale=scale);

            # Make sure that we don't have negative values if there are no in the original data
            if(all(ySizes>0)){
                ySizesBoot$boot[] <- abs(ySizesBoot$boot);
            }

            # Round up sizes if they are integer in the original data
            if(yIsInteger){
                ySizesBoot$boot[] <- ceiling(ySizesBoot$boot);
            }

            # Bootstrap the interval sizes
            yIntervalsBoot <- timeboot(yIntervals, nsim=nsim, intermittent=FALSE,
                                       type=type, kind=kind, lag=1, sd=sd, scale=scale);

            # Round up intervals and add 1 to get back to the original data
            yIntervalsBoot$boot[] <- ceiling(abs(yIntervalsBoot$boot))+1;
            yIndices <- apply(yIntervalsBoot$boot, 2, cumsum);

            # Form a bigger matrix
            yNew <- matrix(0, max(c(yIndices, obsInsample)), nsim);
            # Insert demand sizes
            yNew[cbind(as.vector(yIndices), rep(1:nsim, each=obsNonZero))] <- ySizesBoot$boot;

            return(structure(list(call=cl, data=y,
                                  # Trim the data to return the same sample size
                                  boot=ts(yNew[1:obsInsample,], frequency=frequency(y), start=start(y)),
                                  type=type,
                                  sizes=ySizesBoot, intervals=yIntervalsBoot),
                             class="timeboot"));
        }
        else{
            yIsInteger <- all(y==trunc(y));
        }
    }
    # Record, where zeroes were originally
    otUIDsZeroes <- which(!otU);

    # if(any(y<=0) && type=="multiplicative"){
    #     type[] <- "additive";
    #     warning("Data has non-positive values. Switching bootstrap type to 'additive'.",
    #             call.=FALSE);
    # }

    # The matrix for the new data
    yNew <- ts(matrix(NA, obsInsample, nsim),
               frequency=frequency(y), start=start(y));

    yTransformed <- y;

    # Ordered values
    yOrder <- order(yTransformed);
    ySorted <- sort(yTransformed);

    # Logarithmic transformations of the original data
    if(type=="multiplicative"){
        yTransformed[yTransformed==0] <- NA;
        yTransformed[] <- log(yTransformed);
        ySorted[ySorted==0] <- NA;
        ySorted[] <- log(ySorted);
    }
    yIntermediate <- ySorted;
    idsNonNAs <- !is.na(ySorted);

    # Reset sample size to remove NAs
    obsInsample[] <- sum(idsNonNAs);

    # Smooth the sorted series. This reduces impact of outliers and is just cool to do
    yIntermediate[idsNonNAs] <- supsmu(x=1:obsInsample, ySorted[idsNonNAs])$y;

    if(kind=="nonparametric"){
        if(obsInsample>1){
            #### Differences are needed only for the non-parametric approach ####
            # Prepare differences
            # yDiffs <- sort(diff(ySorted));
            yDiffs <- sort(diff(yTransformed[!is.na(yTransformed)]));
            yDiffsLength <- length(yDiffs);
            # Remove potential outliers
            # yDiffs <- yDiffs[1:round(yDiffsLength*(1-0.02),0)];
            # Remove NaNs if they exist
            yDiffs <- yDiffs[!is.nan(yDiffs)];
            # Leave only finite values
            yDiffs <- yDiffs[is.finite(yDiffs)];
            # Create a contingency table
            yDiffsLength[] <- length(yDiffs);
            # yDiffsCumulative <- cumsum(table(yDiffs)/(yDiffsLength));
            # Form vector larger than yDifsLength to "take care" of tails
            yDiffsCumulative <- seq(0,1,length.out=yDiffsLength+2)[-c(1,yDiffsLength+2)];
            # smooth the original differences
            ySplined <- supsmu(yDiffsCumulative, yDiffs);

            #### Uniform selection of differences ####
            # yRandom <- sample(1:yDiffsLength, size=obsInsample*nsim, replace=TRUE);
            # yDiffsNew <- matrix(sample(c(-1,1), size=obsInsample*nsim, replace=TRUE) *
            #                         yDiffs[yRandom],
            #                     obsInsample, nsim);

            # Random probabilities to select differences
            # yRandom <- runif(obsInsample*nsim, 0, 1);

            #### Select differences based on histogram ####
            # yDiffsNew <- matrix(sample(c(-1,1), size=obsInsample*nsim, replace=TRUE) *
            #                         yDiffs[findInterval(yRandom,yDiffsCumulative)+1],
            #                     obsInsample, nsim);

            #### Differences based on interpolated cumulative values ####
            # Generate the new ones
            # yDiffsNew <- matrix(ySplined$y[findInterval(yRandom,ySplined$x)+1],
            #                     obsInsample, nsim);

            # Generate the new ones
            # approx uses linear approximation to interpolate values
            yDiffsNew <- matrix(sample(c(-1,1), size=obsInsample*nsim, replace=TRUE) *
                                    approx(ySplined$x, ySplined$y, xout=runif(obsInsample*nsim, 0, 1),
                                           rule=2)$y,
                                obsInsample, nsim);
        }
        else{
            yDiffsNew <- matrix(0, obsInsample, nsim);
        }
    }
    else{
        # Calculate scale if it is not provided
        if(is.null(sd)){
            if(lag>obsInsample){
                lag <- 1;
            }
            if(obsInsample>1){
                # This gives robust estimate of scale
                sd <- mean(abs(diff(yTransformed[!is.na(yTransformed)],lag=lag)), na.rm=TRUE);
                # This is one sensitive to outliers
                # scale <- sd(diff(yTransformed,lag=lag));
            }
            else{
                sd <- 1;
            }
        }

        #### Normal distribution randomness ####
        yDiffsNew <- matrix(rnorm(obsInsample*nsim, 0, sd), obsInsample, nsim);
    }

    # Sort values to make sure that we have similar structure in the end
    # yNew[yOrder,] <- rbind(matrix(NA, sum(!idsNonNAs), nsim),
    #                        apply(yIntermediate[idsNonNAs] + yDiffsNew, 2, sort));
    yNew[] <- rbind(matrix(NA, sum(!idsNonNAs), nsim),
                    yIntermediate[idsNonNAs] + yDiffsNew);

    # sd of y with one observation is not defined
    if(obsInsample>1 && scale){
        # Scale things to get the same sd of mean as in the sample
        # if(scale){
            sdData <- sd(y)/sqrt(length(y));
            if(type=="multiplicative"){
                sdBoot <- sd(apply(exp(yNew), 2, mean, na.rm=TRUE));
            }
            else{
                sdBoot <- sd(apply(yNew, 2, mean, na.rm=TRUE));
            }
            # Scale data
            yNew[] <- yIntermediate + (sdData/sdBoot) * (yNew - yIntermediate);
        # }
        #
        # Make sure that the SD of the data is constant
        # if(type=="multiplicative"){
        #     yNewSD <- sqrt(apply((exp(yNew) - exp(yIntermediate))^2, 1, mean, na.rm=TRUE));
        # }
        # else{
        #     yNewSD <- sqrt(apply((yNew - yIntermediate)^2, 1, mean, na.rm=TRUE));
        # }
        # yNew[] <- yIntermediate + (sd(y)/yNewSD) * (yNew - yIntermediate);
    }
    # Sort things
    yNew[yOrder,] <- apply(yNew, 2, sort, na.last=FALSE);
    # Centre the points around the original data
    yNew[] <- yTransformed + yNew - apply(yNew, 1, mean, na.rm=TRUE);

    if(obsInsample>1){
        # Make sure that the SD of the data is constant
        yNewSD <- sqrt(apply((yNew - yTransformed)^2, 1, mean, na.rm=TRUE));
        # yNew[] <- yTransformed + (mean(yNewSD, na.rm=TRUE)/(yNewSD)) * (yNew - yTransformed)
        yNew[] <- yTransformed + (mean(yNewSD, na.rm=TRUE)/(yNewSD)) * (yNew - yTransformed)
    }

    if(type=="multiplicative"){
        yNew[] <- exp(yNew);
    }

    # Recreate zeroes where they were in the original data
    yNew[otUIDsZeroes,] <- 0;

    # Round up values if the original data was integer
    if(yIsInteger){
        yNew[] <- ceiling(yNew);
    }

    return(structure(list(call=cl, data=y, boot=yNew, type=type, sd=sd, scale=scale), class="timeboot"));
}

#' @export
print.timeboot <- function(x, ...){
    cat("Bootstrapped", ncol(x$boot), "series,", x$type, "type.\n");
}

#' @export
plot.timeboot <- function(x, sorted=FALSE, ...){
    nsim <- ncol(x$boot);
    ellipsis <- list(...);
    if(sorted){
        ellipsis$x <- seq(0,1,length.out=length(x$data));
        ellipsis$y <- sort(x$data);
        yOrder <- order(x$data);
    }
    else{
        ellipsis$x <- x$data;
    }

    if(is.null(ellipsis$ylim)){
        ellipsis$ylim <- range(x$boot);
    }

    if(is.null(ellipsis$xlab)){
        if(sorted){
            ellipsis$xlab <- "Probability";
        }
        else{
            ellipsis$xlab <- "Time";
        }
    }

    if(is.null(ellipsis$ylab)){
        ellipsis$ylab <- as.character(x$call[[2]]);
    }

    do.call("plot", ellipsis)
    if(sorted){
        for(i in 1:nsim){
            lines(ellipsis$x,x$boot[yOrder,i], col="lightgrey");
        }
        lines(ellipsis$x, ellipsis$y, lwd=2)
    }
    else{
        for(i in 1:nsim){
            lines(x$boot[,i], col="lightgrey");
        }
        lines(ellipsis$x, lwd=2)
    }
}
