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
#' @param scale Parameter that defines how to scale the variability around the data. By
#' default this is based on the ratio of absolute mean differences and mean absolute differences
#' in sample. If the two are the same, the data has a strong trend and no scaling is required.
#' If the two are different, the data does not have a trend and the larger scaling is needed.
#' @param type Type of bootstrap to use. \code{"additive"} means that the randomness is
#' added, while \code{"multiplicative"} implies the multiplication. By default the function
#' will try using the latter, unless the data has non-positive values.
#' @param lag The lag to use in the calculation of differences. Should be 1 for non-seasonal
#' data.
#'
#' @return The function returns:
#' \itemize{
#' \item \code{call} - the call that was used originally;
#' \item \code{data} - the original data used in the function;
#' \item \code{boot} - the matrix with the new series in columns and observations in rows.
#' \item \code{type} - type of the bootstrap used.}
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
timeboot <- function(y, nsim=100, scale=NULL, lag=frequency(y),
                     type=c("auto","multiplicative","additive")){
    type <- match.arg(type);
    cl <- match.call();

    if(type=="auto"){
        if(any(y<=0)){
            type[] <- "additive";
        }
        else{
            type[] <- "multiplicative";
        }
    }

    # This also needs to take sample size into account!!!
    # Heuristic: strong trend -> scale ~ 0; no trend -> scale ~ 10
    if(is.null(scale)){
        if(type=="multiplicative"){
            # This gives robust estimate of scale
            scale <- mean(abs(diff(log(y),lag=lag)));
            # This is one sensitive to outliers
            # scale <- sd(diff(log(y)));
        }
        else{
            # This gives robust estimate of scale
            scale <- mean(abs(diff(y,lag=lag)));
            # This is one sensitive to outliers
            # scale <- sd(diff(y));
        }
        # scale <- sqrt(mean(diff(y)^2));
        # scale <- (1-mean(diff(y), trim=trim)^2 / mean(diff(y)^2, trim=trim))*5;
        # scale <- (1-abs(mean(diff(y), trim=trim))/mean(abs(diff(y)),trim=trim))*10;
    }

    # Sample size and ordered values
    obsInsample <- length(y);
    yOrder <- order(y);
    ySorted <- sort(y);
    # Intermediate points are done via a sensitive lowess
    # This is because the sorted values have "trend"
    # yIntermediate <- lowess(ySorted, f=0.02)$y;
    yIntermediate <- ySorted;

    if(type=="multiplicative"){
        ySorted[] <- log(ySorted);
        yIntermediate[] <- log(yIntermediate);
    }

    # Smooth the sorted series. This reduces impact of outliers and is just cool to do
    yIntermediate <- supsmu(x=1:length(yIntermediate), yIntermediate)$y;

    yNew <- ts(matrix(NA, obsInsample, nsim),
               frequency=frequency(y), start=start(y));

    #### Differences are needed only for the non-parametric approach ####
    # Prepare differences
    # yDiffs <- sort(diff(ySorted));
    # yDiffsLength <- length(yDiffs);
    # # Remove potential outliers
    # yDiffs <- yDiffs[1:round(yDiffsLength*(1-trim),0)];
    # # Remove NaNs if they exist
    # yDiffs <- yDiffs[!is.nan(yDiffs)];
    # # Leave only finite values
    # yDiffs <- yDiffs[is.finite(yDiffs)];
    # # Create a contingency table
    # yDiffsLength[] <- length(yDiffs);
    # yDiffsCumulative <- cumsum(table(yDiffs)/(yDiffsLength));
    # yDiffsUnique <- unique(yDiffs);

    #### Uniform selection of differences ####
    # yRandom <- sample(1:yDiffsLength, size=obsInsample*nsim, replace=TRUE);
    # yDiffsNew <- matrix(sample(c(-1,1), size=obsInsample*nsim, replace=TRUE) *
    #                         yDiffsUnique[yRandom],
    #                     obsInsample, nsim);

    # Random probabilities to select differences
    # yRandom <- runif(obsInsample*nsim, 0, 1);

    #### Select differences based on histogram ####
    # yDiffsNew <- matrix(sample(c(-1,1), size=obsInsample*nsim, replace=TRUE) *
    #                         yDiffsUnique[findInterval(yRandom,yDiffsCumulative)+1],
    #                     obsInsample, nsim);

    #### Differences based on interpolated cumulative values ####
    # ySplined <- spline(yDiffsCumulative, yDiffsUnique, n=1000);
    # yDiffsNew <- matrix(sample(c(-1,1), size=obsInsample*nsim, replace=TRUE) *
    #                         ySplined$y[findInterval(yRandom,ySplined$x)+1],
    #                     obsInsample, nsim);

    # Sort the final values
    # yNew[yOrder,] <- apply(yIntermediate + scale*yDiffsNew, 2, sort);

    #### Normal distribution randomness ####
    yDiffsNew <- matrix(rnorm(obsInsample*nsim, 0, scale), obsInsample, nsim);

    # Sort values to make sure that we have similar structure in the end
    yNew[yOrder,] <- apply(yIntermediate + yDiffsNew, 2, sort);

    # Centre the points around the original data
    yNew[] <- yNew - apply(yNew, 1, mean) + switch(type,
                                                   "multiplicative"=log(y),
                                                   y);

    if(type=="multiplicative"){
        yNew[] <- exp(yNew);
    }

    return(structure(list(call=cl, data=y, boot=yNew, type=type), class="timeboot"));
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
        ellipsis$x <- sort(x$data);
        yOrder <- order(x$data);
    }
    else{
        ellipsis$x <- x$data;
    }

    if(is.null(ellipsis$ylim)){
        ellipsis$ylim <- range(c(ellipsis$x,x$boot));
    }

    if(is.null(ellipsis$ylab)){
        ellipsis$ylab <- as.character(x$call[[2]]);
    }

    do.call("plot", ellipsis)
    if(sorted){
        for(i in 1:nsim){
            lines(x$boot[yOrder,i], col="lightgrey");
        }
    }
    else{
        for(i in 1:nsim){
            lines(x$boot[,i], col="lightgrey");
        }
    }
    lines(ellipsis$x, lwd=2)
}
