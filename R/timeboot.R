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
#' @param type Type of bootstrap to use. \code{"additive"} means that the randomness is
#' added, while \code{"multiplicative"} implies the multiplication. By default the function
#' will try using the latter, unless the data has non-positive values.
#' @param kind A kind of the bootstrap to do: nonparametric or semiparametric. The latter
#' relies on the normal distribution, while the former uses the empirical distribution of
#' differences of the data.
#' @param lag The lag to use in the calculation of differences. Should be 1 for non-seasonal
#' data.
#' @param scale Scale (sd) to use in the normal distribution. Estimated as mean absolute
#' differences of the data if omitted.
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
timeboot <- function(y, nsim=100,
                     type=c("auto","multiplicative","additive"),
                     kind=c("nonparametric","semiparametric"),
                     lag=frequency(y), scale=NULL){
    cl <- match.call();

    type <- match.arg(type);
    kind <- match.arg(kind);

    if(type=="auto"){
        if(any(y<=0)){
            type[] <- "additive";
        }
        else{
            type[] <- "multiplicative";
        }
    }

    # Sample size and ordered values
    obsInsample <- length(y);
    yOrder <- order(y);
    ySorted <- sort(y);
    # Intermediate points are done via a sensitive lowess
    # This is because the sorted values have "trend"
    # yIntermediate <- lowess(ySorted, f=0.02)$y;
    yIntermediate <- ySorted;

    yTransformed <- y;
    if(type=="multiplicative"){
        yTransformed[] <- log(y);
        ySorted[] <- log(ySorted);
        yIntermediate[] <- log(yIntermediate);
    }

    # Smooth the sorted series. This reduces impact of outliers and is just cool to do
    yIntermediate <- supsmu(x=1:length(yIntermediate), yIntermediate)$y;

    yNew <- ts(matrix(NA, obsInsample, nsim),
               frequency=frequency(y), start=start(y));

    if(kind=="nonparametric"){
        #### Differences are needed only for the non-parametric approach ####
        # Prepare differences
        # yDiffs <- sort(diff(ySorted));
        yDiffs <- sort(diff(yTransformed));
        yDiffsLength <- length(yDiffs);
        # Remove potential outliers
        # yDiffs <- yDiffs[1:round(yDiffsLength*(1-trim),0)];
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
        yRandom <- runif(obsInsample*nsim, 0, 1);

        #### Select differences based on histogram ####
        # yDiffsNew <- matrix(sample(c(-1,1), size=obsInsample*nsim, replace=TRUE) *
        #                         yDiffs[findInterval(yRandom,yDiffsCumulative)+1],
        #                     obsInsample, nsim);

        #### Differences based on interpolated cumulative values ####
        # Generate the new ones
        # yDiffsNew <- matrix(ySplined$y[findInterval(yRandom,ySplined$x)+1],
        #                     obsInsample, nsim);

        # Generate the new ones
        # approx uses linear approximation to get values
        yDiffsNew <- matrix(approx(ySplined$x, ySplined$y, xout=yRandom, rule=2)$y,
                            obsInsample, nsim);

        # Sort the final values
        yNew[yOrder,] <- apply(yIntermediate + yDiffsNew, 2, sort);
    }
    else{
        # Calculate scale if it is not provided
        if(is.null(scale)){
            # This gives robust estimate of scale
            scale <- mean(abs(diff(yTransformed,lag=lag)));
            # This is one sensitive to outliers
            # scale <- sd(diff(yTransformed,lag=lag));
        }

        #### Normal distribution randomness ####
        yDiffsNew <- matrix(rnorm(obsInsample*nsim, 0, scale), obsInsample, nsim);

        # Sort values to make sure that we have similar structure in the end
        yNew[yOrder,] <- apply(yIntermediate + yDiffsNew, 2, sort);
    }

    # Centre the points around the original data
    yNew[] <- yNew - apply(yNew, 1, mean) + yTransformed;

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
        ellipsis$xlab <- "Probability";
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
